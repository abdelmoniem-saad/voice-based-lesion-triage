"""Final held-out evaluation of the frozen 25-update experiment.

Run once:
    python evaluate_test.py

Requires compare_losses.py and existing experiment outputs.
Outputs:
    results/final_test/
"""

import hashlib
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from torch.nn import functional as F

from compare_losses import make_model, score


BASE = Path(__file__).resolve().parent
OUT = BASE / "results" / "final_test"
METHODS = ["smote", "weighted_bce", "focal"]
SEEDS = [42, 43, 44]
STEPS = 25
BATCH_SIZE = 64
BOOTSTRAPS = 2000


def load_vectors(rows):
    vectors = []
    for sid in rows.session_id:
        path = BASE / "embeddings" / f"{int(sid)}-a_n.npy"
        vector = np.load(path, allow_pickle=False)
        if vector.shape != (512,) or not np.isfinite(vector).all():
            raise ValueError(f"Invalid embedding: {path}")
        vectors.append(vector)
    return np.stack(vectors).astype(np.float32)


def fit_model(X, y, hidden, method, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    counts = np.bincount(y, minlength=2)
    if counts.min() < 2:
        raise ValueError("Insufficient minority development samples.")

    weights = torch.tensor(
        len(y) / (2.0 * counts), dtype=torch.float32
    )

    if method == "smote":
        X, y = SMOTE(
            random_state=seed,
            k_neighbors=min(5, int(counts.min()) - 1),
        ).fit_resample(X, y)

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    model = make_model(hidden)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=0.0001,
    )

    model.train()
    for _ in range(STEPS):
        index = rng.integers(0, len(y), size=BATCH_SIZE)
        target = y[index]

        optimizer.zero_grad(set_to_none=True)
        logits = model(X[index]).squeeze(1)
        bce = F.binary_cross_entropy_with_logits(
            logits, target, reduction="none"
        )

        if method == "smote":
            loss = bce.mean()
        else:
            terms = weights[target.long()] * bce
            if method == "focal":
                terms = terms * (1 - torch.exp(-bce)).pow(2.0)
            loss = terms.mean()

        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite training loss.")
        loss.backward()
        optimizer.step()

    return model.eval()


def cluster_intervals(test, predictions):
    """Percentile bootstrap of mean per-seed test metrics.

    The same sampled speakers are used for every method and seed.
    Models remain fixed: this does not capture training-data uncertainty.
    """
    rng = np.random.default_rng(2026)
    speakers = test.speaker_id.unique()
    groups = {
        speaker: np.flatnonzero(
            test.speaker_id.to_numpy() == speaker
        )
        for speaker in speakers
    }

    metrics = [
        "auroc", "average_precision", "balanced_accuracy",
        "sensitivity", "specificity",
    ]
    distributions = {
        method: {metric: [] for metric in metrics}
        for method in METHODS
    }
    accepted = 0

    for _ in range(BOOTSTRAPS):
        sampled = rng.choice(
            speakers, size=len(speakers), replace=True
        )
        indices = np.concatenate([groups[s] for s in sampled])
        labels = test.label.to_numpy(dtype=int)[indices]

        # AUROC and sensitivity/specificity require both classes.
        if len(np.unique(labels)) != 2:
            continue

        accepted += 1
        for method in METHODS:
            seed_scores = [
                score(labels, predictions[(method, seed)][indices])
                for seed in SEEDS
            ]
            for metric in metrics:
                distributions[method][metric].append(
                    np.mean([s[metric] for s in seed_scores])
                )

    if accepted < 1000:
        warnings.warn(
            f"Only {accepted} valid speaker-bootstrap replicates."
        )

    rows = []
    for method in METHODS:
        for metric in metrics:
            values = distributions[method][metric]
            if not values:
                raise RuntimeError("No valid bootstrap replicates.")
            lower, upper = np.percentile(values, [2.5, 97.5])
            rows.append({
                "method": method,
                "metric": metric,
                "ci_lower": float(lower),
                "ci_upper": float(upper),
                "valid_bootstraps": accepted,
            })
    return pd.DataFrame(rows)


def main():
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError(
            "Final results already exist. Preserve them; "
            "do not silently rerun or overwrite the test evaluation."
        )

    comparison = BASE / "results" / "loss_comparison_25steps"
    config = json.loads(
        (comparison / "run_config.json").read_text()
    )

    # Verify the frozen protocol.
    expected = {
        "hidden_layer_sizes": [50, 50],
        "steps_per_fit": 25,
        "seeds": SEEDS,
        "folds": list(range(5)),
        "batch_size": 64,
        "learning_rate": 0.001,
        "adam_weight_decay": 0.0001,
        "focal_gamma": 2.0,
        "threshold": 0.5,
        "smoke_test": False,
        "test_evaluated": False,
        "speaker_weights_used": False,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Unexpected frozen setting: {key}")

    split_path = BASE / "splits.csv"
    split_hash = hashlib.sha256(
        split_path.read_bytes()
    ).hexdigest()
    if split_hash != config["split_sha256"]:
        raise ValueError("splits.csv changed.")

    embedding_config = json.loads(
        (BASE / "embeddings" / "extraction_config.json").read_text()
    )
    if embedding_config != config["embedding_config"]:
        raise ValueError("Embedding configuration changed.")

    df = pd.read_csv(split_path)
    required = [
        "session_id", "speaker_id", "label", "split", "dev_fold"
    ]
    if df[required].isna().any().any():
        raise ValueError("Missing split metadata.")
    if df.session_id.duplicated().any():
        raise ValueError("Duplicate sessions.")
    if set(df.split) != {"dev", "test"}:
        raise ValueError("Unexpected partitions.")
    if set(df.label) != {0, 1}:
        raise ValueError("Unexpected labels.")
    if df.groupby("speaker_id").split.nunique().max() != 1:
        raise ValueError("Speaker overlap between dev and test.")

    df["available"] = df.session_id.map(
        lambda sid: (
            BASE / "embeddings" / f"{int(sid)}-a_n.npy"
        ).is_file()
    )

    # Require exactly the same exclusions as the baseline.
    original_missing = pd.read_csv(
        BASE / "results" / "baseline_cv" / "missing_embeddings.csv"
    )
    expected_missing = set(original_missing.session_id.astype(int))
    actual_missing = set(
        df.loc[~df.available, "session_id"].astype(int)
    )
    if actual_missing != expected_missing:
        raise ValueError("Embedding coverage changed since baseline.")

    dev = df[
        (df.split == "dev") & df.available
    ].reset_index(drop=True)
    test = df[
        (df.split == "test") & df.available
    ].reset_index(drop=True)

    if set(dev.label) != {0, 1} or set(test.label) != {0, 1}:
        raise ValueError("A partition lacks a class.")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "models").mkdir()

    final_config = {
        **config,
        "stage": "final held-out test evaluation",
        "training_partition": "all available development sessions",
        "test_evaluated": True,
        "bootstrap_replicates": BOOTSTRAPS,
        "bootstrap_seed": 2026,
        "ci_method": (
            "speaker-cluster percentile bootstrap, "
            "mean of per-seed metrics, fixed trained models"
        ),
        "torch_version": str(torch.__version__),
    }
    (OUT / "run_config.json").write_text(
        json.dumps(final_config, indent=2) + "\n"
    )
    df.drop(columns="available").loc[
        ~df.available
    ].to_csv(OUT / "excluded_sessions.csv", index=False)

    print(
        f"Development: {len(dev)} sessions, "
        f"{dev.speaker_id.nunique()} speakers."
    )
    print(
        f"Test: {len(test)} sessions, "
        f"{test.speaker_id.nunique()} speakers, "
        f"{int(test.label.sum())} positive sessions."
    )

    X_dev = load_vectors(dev)
    y_dev = dev.label.to_numpy(dtype=int)

    scaler = StandardScaler()
    X_dev = scaler.fit_transform(X_dev).astype(np.float32)

    with (OUT / "scaler.pkl").open("wb") as f:
        pickle.dump(scaler, f)

    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)

    # Train and save all nine models before loading test vectors.
    for method in METHODS:
        for seed in SEEDS:
            print(f"Training {method}, seed {seed}", flush=True)
            model = fit_model(
                X_dev, y_dev, config["hidden_layer_sizes"],
                method, seed,
            )
            torch.save(
                model.state_dict(),
                OUT / "models" / f"{method}_seed{seed}.pt",
            )

    X_test = scaler.transform(
        load_vectors(test)
    ).astype(np.float32)
    test_tensor = torch.from_numpy(X_test)
    y_test = test.label.to_numpy(dtype=int)

    records = []
    frames = []
    predictions = {}

    for method in METHODS:
        for seed in SEEDS:
            model = make_model(config["hidden_layer_sizes"])
            model.load_state_dict(torch.load(
                OUT / "models" / f"{method}_seed{seed}.pt",
                map_location="cpu",
                weights_only=True,
            ))
            model.eval()
            with torch.inference_mode():
                probability = torch.sigmoid(
                    model(test_tensor).squeeze(1)
                ).numpy()

            if not np.isfinite(probability).all():
                raise RuntimeError("Nonfinite test predictions.")

            predictions[(method, seed)] = probability
            records.append({
                "method": method,
                "seed": seed,
                **score(y_test, probability),
            })

            frame = test[
                ["session_id", "speaker_id", "label"]
            ].copy()
            frame["method"] = method
            frame["seed"] = seed
            frame["probability"] = probability
            frame["prediction"] = (
                probability >= 0.5
            ).astype(int)
            frames.append(frame)

    results = pd.DataFrame(records)
    results.to_csv(OUT / "test_metrics_per_seed.csv", index=False)
    pd.concat(frames, ignore_index=True).to_csv(
        OUT / "test_predictions.csv", index=False
    )

    metric_names = [
        "auroc", "average_precision", "balanced_accuracy",
        "sensitivity", "specificity",
    ]
    summary = results.groupby("method")[
        metric_names
    ].agg(["mean", "std"])
    summary.columns = [
        f"{metric}_{stat}" for metric, stat in summary.columns
    ]
    summary.to_csv(OUT / "test_summary.csv")

    print("Computing speaker-bootstrap intervals...", flush=True)
    intervals = cluster_intervals(test, predictions)
    intervals.to_csv(
        OUT / "test_confidence_intervals.csv", index=False
    )

    print("\nFinal test results:")
    print(summary.to_string())
    print(f"\nSaved to: {OUT}")


if __name__ == "__main__":
    main()
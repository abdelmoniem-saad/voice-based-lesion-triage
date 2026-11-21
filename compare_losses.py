"""Compare imbalance methods on development folds only.

Run:
    python compare_losses.py

Optional smoke test:
    python compare_losses.py --smoke

The test set is never loaded or evaluated.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import functional as F


BASE = Path(__file__).resolve().parent
SEEDS = [42, 43, 44]
METHODS = ["smote", "weighted_bce", "focal"]
STEPS = 1000
BATCH_SIZE = 64
GAMMA = 2.0


def score(y, probability):
    prediction = (probability >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y, prediction, labels=[0, 1]
    ).ravel()

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    return {
        "auroc": float(roc_auc_score(y, probability)),
        "average_precision": float(
            average_precision_score(y, probability)
        ),
        "balanced_accuracy": float(
            (sensitivity + specificity) / 2
        ),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def make_model(hidden):
    layers = []
    width = 512
    for next_width in hidden:
        layers.extend([
            nn.Linear(width, next_width),
            nn.ReLU(),
        ])
        width = next_width
    layers.append(nn.Linear(width, 1))
    return nn.Sequential(*layers)


def train_predict(
    x_train, y_train, x_validation,
    hidden, method, seed, steps,
):
    # Reset initialization for each method.
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    original_counts = np.bincount(
        y_train.astype(int), minlength=2
    )
    if original_counts.min() < 2:
        raise ValueError("Too few minority training samples.")

    # Balanced weights estimated from real training labels only.
    class_weights = (
        len(y_train) / (2.0 * original_counts)
    ).astype(np.float32)

    if method == "smote":
        sampler = SMOTE(
            random_state=seed,
            k_neighbors=min(5, int(original_counts.min()) - 1),
        )
        x_train, y_train = sampler.fit_resample(
            x_train, y_train
        )

    x_train = torch.from_numpy(
        np.asarray(x_train, dtype=np.float32)
    )
    y_train = torch.from_numpy(
        np.asarray(y_train, dtype=np.float32)
    )
    x_validation = torch.from_numpy(
        np.asarray(x_validation, dtype=np.float32)
    )
    weights = torch.from_numpy(class_weights)

    model = make_model(hidden)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=0.0001,
    )

    model.train()
    final_loss = None

    for _ in range(steps):
        # Sampling with replacement gives every method exactly
        # the same number of updates and batch size.
        indices = rng.integers(
            0, len(y_train), size=BATCH_SIZE
        )
        batch_x = x_train[indices]
        batch_y = y_train[indices]

        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_x).squeeze(1)

        bce = F.binary_cross_entropy_with_logits(
            logits, batch_y, reduction="none"
        )

        if method == "smote":
            loss = bce.mean()
        else:
            sample_weights = weights[batch_y.long()]
            terms = sample_weights * bce

            if method == "focal":
                # exp(-BCE) equals the predicted probability
                # assigned to the true class.
                p_true = torch.exp(-bce)
                terms = terms * (1.0 - p_true).pow(GAMMA)

            loss = terms.mean()

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Nonfinite loss: {method}, seed {seed}"
            )

        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach())

    model.eval()
    with torch.inference_mode():
        probability = torch.sigmoid(
            model(x_validation).squeeze(1)
        ).numpy()

    if not np.isfinite(probability).all():
        raise RuntimeError("Nonfinite predictions.")

    parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return probability, final_loss, parameters


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--output",
        help="Fresh output directory, relative to this script.",
    )
    args = parser.parse_args()

    output_name = args.output or (
        "results/loss_comparison_smoke"
        if args.smoke
        else "results/loss_comparison"
    )
    output = BASE / output_name
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(
            f"{output} contains results. "
            "Choose a new directory with --output."
        )

    baseline_dir = BASE / "results" / "baseline_cv"
    baseline_config = json.loads(
        (baseline_dir / "run_config.json").read_text()
    )
    selected = json.loads(
        (baseline_dir / "selected_architecture.json").read_text()
    )
    hidden = selected["hidden_layer_sizes"]

    split_path = BASE / "splits.csv"
    split_hash = hashlib.sha256(
        split_path.read_bytes()
    ).hexdigest()
    if split_hash != baseline_config["split_sha256"]:
        raise ValueError(
            "splits.csv changed since the baseline run."
        )

    embedding_dir = BASE / "embeddings"
    embedding_config = json.loads(
        (embedding_dir / "extraction_config.json").read_text()
    )
    if embedding_config != baseline_config["embedding_config"]:
        raise ValueError("Embedding configuration changed.")

    df = pd.read_csv(split_path)
    if df.session_id.duplicated().any():
        raise ValueError("Duplicate session IDs.")

    if df.groupby("speaker_id").split.nunique().max() != 1:
        raise ValueError("Development/test speaker leakage.")

    # Remove test rows before any vector loading.
    dev = df[df.split == "dev"].copy()

    if dev.groupby("speaker_id").dev_fold.nunique().max() != 1:
        raise ValueError("Speaker leakage across development folds.")

    if set(dev.dev_fold.unique()) != set(range(5)):
        raise ValueError("Expected five development folds.")

    dev["path"] = dev.session_id.map(
        lambda sid: embedding_dir / f"{int(sid)}-a_n.npy"
    )
    dev["available"] = dev.path.map(Path.is_file)

    # Require the same missing development sessions as the baseline.
    baseline_missing = pd.read_csv(
        baseline_dir / "missing_embeddings.csv"
    )
    expected_missing = set(
        baseline_missing.loc[
            baseline_missing.split == "dev", "session_id"
        ].astype(int)
    )
    current_missing = set(
        dev.loc[~dev.available, "session_id"].astype(int)
    )
    if current_missing != expected_missing:
        raise ValueError(
            "Development embedding coverage differs from baseline."
        )

    dev = dev[dev.available].reset_index(drop=True)
    vectors = []
    for row in dev.itertuples():
        vector = np.load(row.path, allow_pickle=False)
        if vector.shape != (512,) or not np.isfinite(vector).all():
            raise ValueError(
                f"Invalid vector: session {row.session_id}"
            )
        vectors.append(vector)

    X = np.stack(vectors).astype(np.float32)
    y = dev.label.to_numpy(dtype=int)
    folds = dev.dev_fold.to_numpy(dtype=int)

    seeds = [42] if args.smoke else SEEDS
    fold_numbers = [0] if args.smoke else list(range(5))
    steps = 20 if args.smoke else STEPS

    # Use CPU for simple, repeatable execution.
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)

    output.mkdir(parents=True, exist_ok=True)
    config = {
        "hidden_layer_sizes": hidden,
        "seeds": seeds,
        "folds": fold_numbers,
        "steps_per_fit": steps,
        "batch_size": BATCH_SIZE,
        "learning_rate": 0.001,
        "adam_weight_decay": 0.0001,
        "focal_gamma": GAMMA,
        "class_weights": "N / (2 * class_count), training labels only",
        "threshold": 0.5,
        "metric_unit": "session",
        "speaker_weights_used": False,
        "split_sha256": split_hash,
        "embedding_config": embedding_config,
        "test_evaluated": False,
        "smoke_test": args.smoke,
        "versions": {
            "torch": str(torch.__version__),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    (output / "run_config.json").write_text(
        json.dumps(config, indent=2) + "\n"
    )

    records = []
    predictions = []

    for fold in fold_numbers:
        training = folds != fold
        validation = folds == fold

        if set(y[training]) != {0, 1}:
            raise ValueError("Training fold lacks a class.")
        if set(y[validation]) != {0, 1}:
            raise ValueError("Validation fold lacks a class.")

        train_speakers = set(dev.loc[training, "speaker_id"])
        val_speakers = set(dev.loc[validation, "speaker_id"])
        if train_speakers & val_speakers:
            raise ValueError("Speaker leakage.")

        scaler = StandardScaler()
        x_train = scaler.fit_transform(X[training])
        x_validation = scaler.transform(X[validation])

        for seed in seeds:
            for method in METHODS:
                print(
                    f"Fold {fold + 1}/5 | seed {seed} | {method}",
                    flush=True,
                )
                probability, final_loss, parameters = train_predict(
                    x_train, y[training], x_validation,
                    hidden, method, seed, steps,
                )
                scores = score(y[validation], probability)
                records.append({
                    "method": method,
                    "seed": seed,
                    "fold": fold,
                    "parameters": parameters,
                    "final_training_loss": final_loss,
                    **scores,
                })

                frame = dev.loc[
                    validation,
                    ["session_id", "speaker_id", "label", "dev_fold"],
                ].copy()
                frame["method"] = method
                frame["seed"] = seed
                frame["probability"] = probability
                predictions.append(frame)

                pd.DataFrame(records).to_csv(
                    output / "fold_metrics.csv", index=False
                )
                pd.concat(
                    predictions, ignore_index=True
                ).to_csv(
                    output / "oof_predictions.csv", index=False
                )

                print(
                    f"  AUROC={scores['auroc']:.3f}; "
                    f"balanced accuracy="
                    f"{scores['balanced_accuracy']:.3f}; "
                    f"sensitivity={scores['sensitivity']:.3f}",
                    flush=True,
                )

    results = pd.DataFrame(records)
    metric_columns = [
        "auroc", "average_precision", "balanced_accuracy",
        "sensitivity", "specificity",
    ]

    # First average seeds within each fold, then summarize folds.
    # This avoids treating 15 correlated fits as independent folds.
    fold_averages = results.groupby(
        ["method", "fold"]
    )[metric_columns].mean()
    fold_averages.to_csv(output / "seed_averaged_fold_metrics.csv")

    summary = fold_averages.groupby("method").agg(["mean", "std"])
    summary.columns = [
        f"{metric}_{stat}" for metric, stat in summary.columns
    ]
    summary.to_csv(output / "method_summary.csv")

    # Separate view of random-seed variability.
    results.groupby(
        ["method", "seed"]
    )[metric_columns].mean().to_csv(
        output / "per_seed_cv_means.csv"
    )

    print("\nMethod summary:")
    print(summary.to_string())
    print(f"\nSaved to: {output}")
    print("Test set was not evaluated.")


if __name__ == "__main__":
    main()
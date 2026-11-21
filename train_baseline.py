"""Development-only SMOTE + MLP architecture search.

Run:
    python train_baseline.py

Inputs:
    splits.csv
    embeddings/<session_id>-a_n.npy
    embeddings/extraction_config.json

Outputs:
    results/baseline_cv/
"""

import argparse
import hashlib
import json
import platform
import warnings
from pathlib import Path

import imblearn
import numpy as np
import pandas as pd
import sklearn
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


BASE = Path(__file__).resolve().parent
SEED = 42

# All one- and two-hidden-layer combinations using widths 50 and 100.
ARCHITECTURES = [
    (50,),
    (100,),
    (50, 50),
    (50, 100),
    (100, 50),
    (100, 100),
]


def metrics(y, probability):
    """Session-level metrics at a fixed probability threshold of 0.5."""
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


def parameter_count(input_dim, hidden_layers):
    """Weights plus biases, including sklearn's single binary output."""
    widths = [input_dim, *hidden_layers, 1]
    return sum(
        (left + 1) * right
        for left, right in zip(widths[:-1], widths[1:])
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="results/baseline_cv",
        help="Use a fresh directory for each run.",
    )
    args = parser.parse_args()

    output = BASE / args.output
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(
            f"{output} already contains results. "
            "Choose a fresh directory using --output."
        )

    split_path = BASE / "splits.csv"
    embedding_dir = BASE / "embeddings"
    metadata_path = embedding_dir / "extraction_config.json"

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = {
        "model": "facebook/wav2vec2-large-xlsr-53",
        "layer": "final convolution before feature projection",
        "pooling": "time mean",
        "dimension": 512,
        "sample_rate": 16000,
        "normalize": True,
        "vowel": "a_n",
    }
    if metadata != expected:
        raise ValueError(
            "Embedding configuration differs from the expected baseline."
        )

    df = pd.read_csv(split_path)
    required = {
        "session_id", "speaker_id", "label", "split", "dev_fold"
    }
    if not required.issubset(df.columns):
        raise ValueError(f"Required columns: {sorted(required)}")

    if df.empty or df.session_id.duplicated().any():
        raise ValueError("Empty split file or duplicate session IDs.")

    if df[list(required)].isna().any().any():
        raise ValueError("Missing values in required split columns.")

    if not set(df.label.unique()).issubset({0, 1}):
        raise ValueError("Labels must be 0 or 1.")

    if set(df.split.unique()) != {"dev", "test"}:
        raise ValueError("Expected dev and test partitions.")

    # Check separation using the complete split, before filtering coverage.
    if df.groupby("speaker_id").split.nunique().max() != 1:
        raise ValueError("A speaker crosses development/test partitions.")

    original_dev = df[df.split == "dev"].copy()
    if original_dev.groupby("speaker_id").dev_fold.nunique().max() != 1:
        raise ValueError("A development speaker crosses folds.")

    if set(original_dev.dev_fold.unique()) != set(range(5)):
        raise ValueError("Expected development folds numbered 0 through 4.")

    # Only check whether test vectors exist; do not load or score them.
    df["embedding_path"] = df.session_id.map(
        lambda sid: str(embedding_dir / f"{int(sid)}-a_n.npy")
    )
    df["embedding_available"] = df.embedding_path.map(
        lambda path: Path(path).is_file()
    )

    coverage = df.groupby(["split", "label"]).agg(
        planned_sessions=("session_id", "size"),
        available_sessions=("embedding_available", "sum"),
    )
    print("\nEmbedding coverage:")
    print(coverage.to_string())

    missing = df[~df.embedding_available].copy()
    print(f"\nMissing embeddings: {len(missing)}")
    if len(missing):
        print(
            missing[
                ["session_id", "speaker_id", "label", "split", "dev_fold"]
            ].to_string(index=False)
        )

    dev = df[
        (df.split == "dev") & df.embedding_available
    ].copy().reset_index(drop=True)

    if dev.empty:
        raise ValueError("No development embeddings found.")

    vectors = []
    for row in dev.itertuples():
        vector = np.load(row.embedding_path, allow_pickle=False)
        if vector.shape != (512,) or not np.isfinite(vector).all():
            raise ValueError(
                f"Invalid embedding for session {row.session_id}"
            )
        vectors.append(vector)

    X = np.stack(vectors).astype(np.float32)
    y = dev.label.to_numpy(dtype=int)
    folds = dev.dev_fold.to_numpy(dtype=int)

    # Fail before starting a long run if any fold is unusable.
    for fold in range(5):
        training = folds != fold
        validation = folds == fold

        if set(y[training]) != {0, 1}:
            raise ValueError(f"Training fold {fold} lacks a class.")
        if set(y[validation]) != {0, 1}:
            raise ValueError(f"Validation fold {fold} lacks a class.")

        minority_count = np.bincount(
            y[training], minlength=2
        ).min()
        if minority_count < 2:
            raise ValueError(
                f"Training fold {fold} has too few minority samples "
                "for SMOTE."
            )

        training_speakers = set(dev.loc[training, "speaker_id"])
        validation_speakers = set(dev.loc[validation, "speaker_id"])
        if training_speakers & validation_speakers:
            raise ValueError(f"Speaker leakage in fold {fold}.")

    print(
        f"\nDevelopment data: {len(dev)} sessions, "
        f"{dev.speaker_id.nunique()} speakers, "
        f"{int(y.sum())} positive sessions."
    )
    print(
        "Metrics are session-level; partitioning is speaker-grouped."
    )

    output.mkdir(parents=True, exist_ok=True)
    missing.drop(columns=["embedding_path"]).to_csv(
        output / "missing_embeddings.csv", index=False
    )
    coverage.to_csv(output / "coverage.csv")

    run_config = {
        "seed": SEED,
        "architectures": [list(a) for a in ARCHITECTURES],
        "embedding_config": metadata,
        "split_sha256": hashlib.sha256(
            split_path.read_bytes()
        ).hexdigest(),
        "selection": "mean fold AUROC; balanced accuracy breaks ties",
        "threshold": 0.5,
        "metric_unit": "session",
        "speaker_weights_used": False,
        "inputs": "audio embeddings only",
        "training": {
            "scaler": "StandardScaler fit on training fold",
            "smote_sampling_strategy": "auto",
            "smote_k_neighbors": "min(5, training minority count - 1)",
            "activation": "relu",
            "solver": "adam",
            "alpha": 0.0001,
            "learning_rate_init": 0.001,
            "batch_size": 64,
            "max_iter": 500,
            "early_stopping": False,
            "tol": 0.0001,
            "n_iter_no_change": 20,
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
            "imblearn": imblearn.__version__,
        },
    }
    (output / "run_config.json").write_text(
        json.dumps(run_config, indent=2) + "\n",
        encoding="utf-8",
    )

    fold_results = []
    prediction_frames = []

    for architecture in ARCHITECTURES:
        name = "x".join(map(str, architecture))

        for fold in range(5):
            training = folds != fold
            validation = folds == fold
            minority_count = int(
                np.bincount(y[training], minlength=2).min()
            )
            neighbors = min(5, minority_count - 1)

            # Pipeline.fit sees only training rows.
            # Pipeline.predict_proba scales validation rows but never
            # applies SMOTE to them.
            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("smote", SMOTE(
                    random_state=SEED,
                    k_neighbors=neighbors,
                )),
                ("mlp", MLPClassifier(
                    hidden_layer_sizes=architecture,
                    activation="relu",
                    solver="adam",
                    alpha=0.0001,
                    learning_rate_init=0.001,
                    batch_size=64,
                    max_iter=500,
                    early_stopping=False,
                    tol=0.0001,
                    n_iter_no_change=20,
                    random_state=SEED,
                )),
            ])

            print(
                f"\nArchitecture {name}, fold {fold + 1}/5",
                flush=True,
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                pipeline.fit(X[training], y[training])

            convergence_warning = any(
                issubclass(item.category, ConvergenceWarning)
                for item in caught
            )
            for item in caught:
                print(
                    f"Warning: {item.category.__name__}: {item.message}"
                )

            positive_column = int(
                np.flatnonzero(pipeline.classes_ == 1)[0]
            )
            probability = pipeline.predict_proba(
                X[validation]
            )[:, positive_column]

            scores = metrics(y[validation], probability)
            result = {
                "architecture": name,
                "fold": fold,
                "train_sessions": int(training.sum()),
                "validation_sessions": int(validation.sum()),
                "train_positive_sessions": int(y[training].sum()),
                "validation_positive_sessions": int(
                    y[validation].sum()
                ),
                "parameters": parameter_count(
                    X.shape[1], architecture
                ),
                "iterations": int(
                    pipeline.named_steps["mlp"].n_iter_
                ),
                "convergence_warning": convergence_warning,
                **scores,
            }
            fold_results.append(result)

            prediction_frame = dev.loc[
                validation,
                ["session_id", "speaker_id", "label", "dev_fold"],
            ].copy()
            prediction_frame["architecture"] = name
            prediction_frame["probability"] = probability
            prediction_frame["prediction"] = (
                probability >= 0.5
            ).astype(int)
            prediction_frames.append(prediction_frame)

            # Save progress after every completed fit.
            pd.DataFrame(fold_results).to_csv(
                output / "fold_metrics.csv", index=False
            )
            pd.concat(
                prediction_frames, ignore_index=True
            ).to_csv(
                output / "oof_predictions.csv", index=False
            )

            print(
                f"AUROC={scores['auroc']:.3f}, "
                f"balanced accuracy="
                f"{scores['balanced_accuracy']:.3f}, "
                f"sensitivity={scores['sensitivity']:.3f}, "
                f"specificity={scores['specificity']:.3f}",
                flush=True,
            )

    results = pd.DataFrame(fold_results)
    metric_names = [
        "auroc",
        "average_precision",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
    ]
    summary = results.groupby("architecture")[
        metric_names
    ].agg(["mean", "std"])
    summary.columns = [
        f"{metric}_{statistic}"
        for metric, statistic in summary.columns
    ]
    summary = summary.reset_index()
    summary["parameters"] = summary.architecture.map(
        results.groupby("architecture").parameters.first()
    )
    summary["convergence_warnings"] = summary.architecture.map(
        results.groupby("architecture").convergence_warning.sum()
    )

    # A deterministic final name sort resolves any exact remaining tie.
    summary = summary.sort_values(
        ["auroc_mean", "balanced_accuracy_mean", "architecture"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    summary.to_csv(output / "architecture_summary.csv", index=False)

    winner = summary.iloc[0]
    selected = {
        "architecture": winner.architecture,
        "hidden_layer_sizes": [
            int(width) for width in winner.architecture.split("x")
        ],
        "mean_validation_auroc": float(winner.auroc_mean),
        "mean_validation_balanced_accuracy": float(
            winner.balanced_accuracy_mean
        ),
        "test_evaluated": False,
        "note": (
            "Development selection result. Not an independent "
            "estimate of performance after model selection."
        ),
    }
    (output / "selected_architecture.json").write_text(
        json.dumps(selected, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nArchitecture comparison:")
    print(summary.to_string(index=False))
    print(f"\nSelected architecture: {winner.architecture}")
    print(f"Results saved to: {output}")
    print("Test set was not evaluated.")


if __name__ == "__main__":
    main()
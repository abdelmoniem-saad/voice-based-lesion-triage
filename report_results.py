"""Validate and visualize saved development results. No training or test use."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


BASE = Path(__file__).resolve().parent
SOURCE = BASE / "results" / "loss_comparison"
OUT = SOURCE / "report"

METHODS = ["smote", "weighted_bce", "focal"]
LABELS = {
    "smote": "SMOTE",
    "weighted_bce": "Weighted BCE",
    "focal": "Focal loss",
}
COLORS = {
    "smote": "#4477AA",
    "weighted_bce": "#228833",
    "focal": "#CC6677",
}
SEEDS = [42, 43, 44]


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=200)
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)


def main():
    summary = pd.read_csv(SOURCE / "method_summary.csv")
    folds = pd.read_csv(
        SOURCE / "seed_averaged_fold_metrics.csv"
    )
    seeds = pd.read_csv(SOURCE / "per_seed_cv_means.csv")
    predictions = pd.read_csv(SOURCE / "oof_predictions.csv")

    summary = summary.set_index("method").loc[METHODS]

    # Validate the saved experiment before making figures.
    if set(predictions.method) != set(METHODS):
        raise ValueError("Unexpected methods in predictions.")
    if set(predictions.seed) != set(SEEDS):
        raise ValueError("Unexpected seeds in predictions.")
    if predictions[
        ["session_id", "speaker_id", "label", "dev_fold",
         "method", "seed", "probability"]
    ].isna().any().any():
        raise ValueError("Missing prediction values.")
    if predictions.duplicated(
        ["session_id", "method", "seed"]
    ).any():
        raise ValueError("Duplicate session/method/seed predictions.")
    if not predictions.probability.between(0, 1).all():
        raise ValueError("Invalid probability scores.")

    for column in ["speaker_id", "label", "dev_fold"]:
        if predictions.groupby("session_id")[column].nunique().max() != 1:
            raise ValueError(
                f"Inconsistent session metadata: {column}"
            )

    if predictions.groupby("speaker_id").dev_fold.nunique().max() != 1:
        raise ValueError("A speaker appears in multiple folds.")

    # Require the same sessions in all nine method/seed combinations.
    reference_ids = None
    for method in METHODS:
        for seed in SEEDS:
            part = predictions[
                (predictions.method == method)
                & (predictions.seed == seed)
            ]
            ids = set(part.session_id)
            if reference_ids is None:
                reference_ids = ids
            elif ids != reference_ids:
                raise ValueError("Prediction coverage differs.")

    expected = pd.read_csv(
        BASE / "results" / "baseline_cv" / "oof_predictions.csv"
    )
    expected_ids = set(expected.session_id)
    if reference_ids != expected_ids:
        raise ValueError(
            "Session coverage differs from the sklearn baseline."
        )

    OUT.mkdir(parents=True, exist_ok=True)

    unique_sessions = predictions.drop_duplicates("session_id")
    fold_counts = unique_sessions.groupby("dev_fold").agg(
        sessions=("session_id", "size"),
        positive_sessions=("label", "sum"),
        speakers=("speaker_id", "nunique"),
    )
    fold_counts.to_csv(OUT / "fold_counts.csv")

    # 1. Main comparison: means and SD across folds after averaging seeds.
    metrics = [
        ("auroc", "AUROC"),
        ("balanced_accuracy", "Balanced accuracy"),
        ("sensitivity", "Sensitivity"),
        ("specificity", "Specificity"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    x = np.arange(len(METHODS))

    for ax, (metric, title) in zip(axes, metrics):
        means = summary[f"{metric}_mean"].to_numpy()
        sd = summary[f"{metric}_std"].to_numpy()

        # Clip displayed error bars to the valid metric range.
        error = np.vstack([
            np.minimum(sd, means),
            np.minimum(sd, 1 - means),
        ])
        ax.bar(
            x, means,
            color=[COLORS[m] for m in METHODS],
            yerr=error,
            capsize=4,
        )
        ax.set_xticks(
            x, [LABELS[m] for m in METHODS],
            rotation=25, ha="right",
        )
        ax.set_ylim(0, 1.08)
        ax.set_title(title)
        for position, value in zip(x, means):
            ax.text(
                position, value + 0.025, f"{value:.3f}",
                ha="center", fontsize=9,
            )
    fig.suptitle(
        "Development comparison: fold means ± SD\n"
        "Seeds averaged within each fold; threshold = 0.5"
    )
    save(fig, "method_comparison")

    # 2. Paired fold results.
    fig, ax = plt.subplots(figsize=(7, 4))
    for method in METHODS:
        part = folds[folds.method == method].sort_values("fold")
        ax.plot(
            part.fold + 1, part.auroc,
            marker="o", label=LABELS[method],
            color=COLORS[method],
        )
    ax.axhline(0.5, color="gray", linestyle="--")
    ax.set(
        xlabel="Development fold",
        ylabel="AUROC",
        ylim=(0.4, 1),
        title="AUROC by fold, averaged across seeds",
    )
    ax.set_xticks(range(1, 6))
    ax.legend()
    save(fig, "auroc_by_fold")

    # 3. Random-seed variation.
    fig, ax = plt.subplots(figsize=(7, 4))
    for method in METHODS:
        part = seeds[seeds.method == method].sort_values("seed")
        ax.plot(
            part.seed.astype(str), part.auroc,
            marker="o", label=LABELS[method],
            color=COLORS[method],
        )
    ax.set(
        xlabel="Random seed",
        ylabel="Mean fold AUROC",
        title="Variation across random seeds",
    )
    ax.legend()
    save(fig, "auroc_by_seed")

    # 4. Score distributions: one seed, declared in the figure.
    # Scores are clipped only for display, never for evaluation.
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    bins = np.linspace(-12, 0, 40)

    for ax, method in zip(axes, METHODS):
        part = predictions[
            (predictions.method == method)
            & (predictions.seed == 42)
        ]
        for label, color, name in [
            (0, "#4477AA", "Negative"),
            (1, "#CC3311", "Positive"),
        ]:
            values = part.loc[
                part.label == label, "probability"
            ].to_numpy()
            ax.hist(
                np.log10(np.clip(values, 1e-12, 1)),
                bins=bins,
                density=True,
                histtype="step",
                linewidth=2,
                color=color,
                label=f"{name} (n={len(values)})",
            )
        ax.axvline(
            np.log10(0.5), color="black",
            linestyle="--", label="Threshold 0.5",
        )
        ax.set_title(LABELS[method])
        ax.set_xlabel("log10(predicted score)")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)

    fig.suptitle(
        "Out-of-fold score distributions, seed 42\n"
        "Display floor = 10⁻¹²; scores are not calibrated probabilities"
    )
    save(fig, "score_distributions_seed42")

    # 5. Aggregate OOF confusion matrices, separately for every seed.
    confusion_records = []
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))

    for row, method in enumerate(METHODS):
        for col, seed in enumerate(SEEDS):
            part = predictions[
                (predictions.method == method)
                & (predictions.seed == seed)
            ]
            matrix = confusion_matrix(
                part.label,
                (part.probability >= 0.5).astype(int),
                labels=[0, 1],
            )
            tn, fp, fn, tp = matrix.ravel()
            confusion_records.append({
                "method": method,
                "seed": seed,
                "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            })
            ax = axes[row, col]
            ax.imshow(matrix, cmap="Blues")
            for i in range(2):
                for j in range(2):
                    ax.text(
                        j, i, str(matrix[i, j]),
                        ha="center", va="center",
                        color="white"
                        if matrix[i, j] > matrix.max() / 2
                        else "black",
                    )
            ax.set_xticks([0, 1], ["Negative", "Positive"])
            ax.set_yticks([0, 1], ["Negative", "Positive"])
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title(f"{LABELS[method]}, seed {seed}")

    fig.suptitle(
        "Development out-of-fold confusion matrices, threshold = 0.5"
    )
    save(fig, "confusion_matrices")
    pd.DataFrame(confusion_records).to_csv(
        OUT / "confusion_counts.csv", index=False
    )

    lines = [
        "DEVELOPMENT RESULTS",
        "Test set not evaluated.",
        "",
        "Primary summary: mean across folds after averaging seeds.",
        "Fold SD is not a confidence interval.",
        "Metrics count sessions; speakers are grouped across folds.",
        "",
        summary.to_string(),
        "",
        "Fold counts:",
        fold_counts.to_string(),
        "",
        "Interpretation:",
        "- All methods have low sensitivity at threshold 0.5.",
        "- Focal versus weighted BCE is a small development difference.",
        "- PyTorch SMOTE differs from the sklearn training procedure.",
        "- Architecture selection used these development folds;",
        "  these results are not an independent final evaluation.",
    ]
    (OUT / "development_report.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    print("Prediction checks passed.")
    print(f"Reports saved to: {OUT}")
    print("No training or test evaluation was performed.")


if __name__ == "__main__":
    main()
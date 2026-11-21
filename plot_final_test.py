
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)


BASE = Path(__file__).resolve().parent
SOURCE = BASE / "results" / "final_test"
OUT = SOURCE / "figures"

METHODS = ["smote", "weighted_bce", "focal"]
SEEDS = [42, 43, 44]
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
STYLES = {42: "-", 43: "--", 44: ":"}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=250, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    predictions = pd.read_csv(SOURCE / "test_predictions.csv")
    metrics = pd.read_csv(SOURCE / "test_metrics_per_seed.csv")
    summary = pd.read_csv(
        SOURCE / "test_summary.csv"
    ).set_index("method")
    intervals = pd.read_csv(
        SOURCE / "test_confidence_intervals.csv"
    ).set_index(["method", "metric"])

    required = [
        "session_id", "speaker_id", "label",
        "method", "seed", "probability", "prediction",
    ]
    if predictions[required].isna().any().any():
        raise ValueError("Missing prediction values.")
    if predictions.duplicated(
        ["session_id", "method", "seed"]
    ).any():
        raise ValueError("Duplicate predictions.")
    if not predictions.probability.between(0, 1).all():
        raise ValueError("Invalid scores.")
    if set(predictions.method) != set(METHODS):
        raise ValueError("Unexpected methods.")
    if set(predictions.seed) != set(SEEDS):
        raise ValueError("Unexpected seeds.")

    expected_prediction = (
        predictions.probability >= 0.5
    ).astype(int)
    if not np.array_equal(
        predictions.prediction.to_numpy(),
        expected_prediction.to_numpy(),
    ):
        raise ValueError("Saved predictions differ from threshold 0.5.")

    for column in ["speaker_id", "label"]:
        if predictions.groupby("session_id")[column].nunique().max() != 1:
            raise ValueError(f"Inconsistent metadata: {column}")

    reference = None
    for method in METHODS:
        for seed in SEEDS:
            part = predictions[
                (predictions.method == method)
                & (predictions.seed == seed)
            ]
            ids = set(part.session_id)
            if reference is None:
                reference = ids
            elif ids != reference:
                raise ValueError("Methods/seeds have different test sessions.")
            if set(part.label) != {0, 1}:
                raise ValueError("A prediction group lacks a class.")

    if metrics.duplicated(["method", "seed"]).any():
        raise ValueError("Duplicate metric rows.")
    if len(metrics) != len(METHODS) * len(SEEDS):
        raise ValueError("Incomplete per-seed metrics.")

    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    unique_test = predictions.drop_duplicates("session_id")
    prevalence = float(unique_test.label.mean())
    n_positive = int(unique_test.label.sum())
    n_negative = len(unique_test) - n_positive

    # 1. Separate panels preserve all seeds without choosing a winner.
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, method in zip(axes, METHODS):
        for seed in SEEDS:
            part = predictions[
                (predictions.method == method)
                & (predictions.seed == seed)
            ]
            row = metrics[
                (metrics.method == method) & (metrics.seed == seed)
            ].iloc[0]
            fpr, tpr, _ = roc_curve(part.label, part.probability)
            ax.plot(
                fpr, tpr,
                color=COLORS[method],
                linestyle=STYLES[seed],
                label=f"Seed {seed}: AUROC {row.auroc:.3f}",
            )
        ax.plot([0, 1], [0, 1], color="gray", linestyle="--")
        ax.set(
            title=LABELS[method],
            xlabel="False-positive rate",
            ylabel="True-positive rate",
            xlim=(0, 1), ylim=(0, 1.03),
        )
        ax.legend(fontsize=9, loc="lower right")
    fig.suptitle(
        f"Held-out test ROC curves: {n_positive} positive, "
        f"{n_negative} negative sessions"
    )
    save(fig, "test_roc_curves")

    # 2. PR curves with the test prevalence as the reference line.
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, method in zip(axes, METHODS):
        for seed in SEEDS:
            part = predictions[
                (predictions.method == method)
                & (predictions.seed == seed)
            ]
            row = metrics[
                (metrics.method == method) & (metrics.seed == seed)
            ].iloc[0]
            precision, recall, _ = precision_recall_curve(
                part.label, part.probability
            )
            ax.step(
                recall, precision, where="post",
                color=COLORS[method],
                linestyle=STYLES[seed],
                label=f"Seed {seed}: AP {row.average_precision:.3f}",
            )
        ax.axhline(
            prevalence, color="gray", linestyle="--",
            label=f"Prevalence {prevalence:.3f}",
        )
        ax.set(
            title=LABELS[method],
            xlabel="Recall",
            ylabel="Precision",
            xlim=(0, 1), ylim=(0, 1.03),
        )
        ax.legend(fontsize=9)
    fig.suptitle("Held-out test precision–recall curves")
    save(fig, "test_precision_recall_curves")

    # 3. Point estimates and speaker-bootstrap confidence intervals.
    metric_names = [
        ("auroc", "AUROC"),
        ("balanced_accuracy", "Balanced accuracy"),
        ("sensitivity", "Sensitivity"),
        ("specificity", "Specificity"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.5))

    for ax, (metric, title) in zip(axes, metric_names):
        for position, method in enumerate(METHODS):
            mean = float(summary.loc[method, f"{metric}_mean"])
            ci = intervals.loc[(method, metric)]
            lower = float(ci.ci_lower)
            upper = float(ci.ci_upper)

            # Draw the interval independently; percentile intervals
            # need not be symmetric around the point estimate.
            ax.vlines(
                position, lower, upper,
                color=COLORS[method], linewidth=2,
            )
            ax.hlines(
                [lower, upper], position - 0.08, position + 0.08,
                color=COLORS[method],
            )
            ax.scatter(
                position, mean, color=COLORS[method], s=45, zorder=3
            )
            ax.annotate(
                f"{mean:.3f}", (position, mean),
                xytext=(7, 5), textcoords="offset points",
                fontsize=9,
            )
        ax.set_xticks(
            range(3), [LABELS[m] for m in METHODS],
            rotation=25, ha="right",
        )
        ax.set(title=title, ylim=(0, 1.05), xlim=(-0.5, 2.6))

    fig.suptitle(
        "Test metrics: mean across seeds and 95% speaker-bootstrap CI\n"
        "Classification threshold = 0.5; intervals condition on trained models"
    )
    save(fig, "test_metric_comparison")

    # 4. Fixed seed 42, chosen before inspecting final performance.
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, method in zip(axes, METHODS):
        part = predictions[
            (predictions.method == method)
            & (predictions.seed == 42)
        ]
        matrix = confusion_matrix(
            part.label, part.prediction, labels=[0, 1]
        )
        ax.imshow(matrix, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(
                    j, i, str(matrix[i, j]),
                    ha="center", va="center", fontsize=15,
                    color="white"
                    if matrix[i, j] > matrix.max() / 2 else "black",
                )
        ax.set_xticks([0, 1], ["Negative", "Positive"])
        ax.set_yticks([0, 1], ["Negative", "Positive"])
        ax.set(
            title=LABELS[method],
            xlabel="Predicted",
            ylabel="Actual",
        )
    fig.suptitle("Held-out test confusion matrices: seed 42, threshold 0.5")
    save(fig, "test_confusion_matrices_seed42")

    # 5. Export formatted and numeric tables.
    formatted_rows = []
    numeric_rows = []

    for method in METHODS:
        formatted = {"Method": LABELS[method]}
        for metric, title in metric_names + [
            ("average_precision", "Average precision")
        ]:
            mean = float(summary.loc[method, f"{metric}_mean"])
            ci = intervals.loc[(method, metric)]
            lower, upper = float(ci.ci_lower), float(ci.ci_upper)
            formatted[title] = (
                f"{mean:.3f} [{lower:.3f}, {upper:.3f}]"
            )
            numeric_rows.append({
                "method": method,
                "metric": metric,
                "mean_across_seeds": mean,
                "ci_lower": lower,
                "ci_upper": upper,
            })
        formatted_rows.append(formatted)

    table = pd.DataFrame(formatted_rows)
    table.to_csv(OUT / "test_results_formatted.csv", index=False)
    pd.DataFrame(numeric_rows).to_csv(
        OUT / "test_results_numeric.csv", index=False
    )

    table_figure, ax = plt.subplots(figsize=(15, 3))
    ax.axis("off")
    display_table = ax.table(
        cellText=table.values,
        colLabels=table.columns,
        cellLoc="center", loc="center",
    )
    display_table.auto_set_font_size(False)
    display_table.set_fontsize(9)
    display_table.scale(1, 2)
    ax.set_title(
        "Held-out test results: mean [95% speaker-bootstrap CI]",
        pad=15,
    )
    save(table_figure, "test_results_table")

    print("Saved prediction checks passed.")
    print(f"Figures and tables: {OUT}")
    print("No training or new test predictions were performed.")


if __name__ == "__main__":
    main()
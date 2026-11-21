"""Repeat the development comparison with learning-curve diagnostics.

Requires compare_losses.py in the same folder.

Run:
    python diagnose_training.py

Outputs:
    results/learning_curves/

No test evaluation.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)
from torch.nn import functional as F

import compare_losses as original


BASE = Path(__file__).resolve().parent
OUT = BASE / "results" / "learning_curves"
HISTORY = []

METHODS = ["smote", "weighted_bce", "focal"]
COLORS = {
    "smote": "#4477AA",
    "weighted_bce": "#228833",
    "focal": "#CC6677",
}
LABELS = {
    "smote": "SMOTE",
    "weighted_bce": "Weighted BCE",
    "focal": "Focal loss",
}


def diagnostic_metrics(model, x, y):
    model.eval()
    with torch.inference_mode():
        logits = model(x).squeeze(1)
        probability = torch.sigmoid(logits).numpy()
        bce = F.binary_cross_entropy_with_logits(
            logits, y
        ).item()

    labels = y.numpy().astype(int)
    prediction = probability >= 0.5

    return {
        "auroc": float(roc_auc_score(labels, probability)),
        "average_precision": float(
            average_precision_score(labels, probability)
        ),
        "sensitivity": float(prediction[labels == 1].mean()),
        "specificity": float((~prediction[labels == 0]).mean()),
        "unweighted_bce": float(bce),
    }


def train_with_curves(
    x_train, y_train, x_validation,
    hidden, method, seed, steps,
):
    # The existing main() calls methods and seeds in a fixed order
    # within each fold. Infer the fold from the completed fit count.
    fit_number = len({(r["fold"], r["seed"], r["method"])
                      for r in HISTORY})
    fits_per_fold = len(original.SEEDS) * len(original.METHODS)
    fold = fit_number // fits_per_fold

    # Recover validation labels in the same row order used by main().
    # The companion script's loading/filtering order is preserved.
    validation_labels = DEV.loc[
        DEV.dev_fold == fold, "label"
    ].to_numpy(dtype=np.float32)

    if len(validation_labels) != len(x_validation):
        raise ValueError("Validation labels do not match predictions.")

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    real_x = torch.tensor(x_train, dtype=torch.float32)
    real_y = torch.tensor(y_train, dtype=torch.float32)
    val_x = torch.tensor(x_validation, dtype=torch.float32)
    val_y = torch.tensor(validation_labels, dtype=torch.float32)

    counts = np.bincount(
        np.asarray(y_train, dtype=int), minlength=2
    )
    weights = torch.tensor(
        len(y_train) / (2.0 * counts),
        dtype=torch.float32,
    )

    if method == "smote":
        sampler = original.SMOTE(
            random_state=seed,
            k_neighbors=min(5, int(counts.min()) - 1),
        )
        x_train, y_train = sampler.fit_resample(
            x_train, y_train
        )

    train_x = torch.tensor(x_train, dtype=torch.float32)
    train_y = torch.tensor(y_train, dtype=torch.float32)

    model = original.make_model(hidden)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=0.0001,
    )

    checkpoints = {0, 25, 50, 100, 200, 400, 600, 800, steps}

    def record(step):
        training = diagnostic_metrics(model, real_x, real_y)
        validation = diagnostic_metrics(model, val_x, val_y)

        row = {
            "fold": fold,
            "seed": seed,
            "method": method,
            "step": step,
        }
        row.update({f"train_{k}": v for k, v in training.items()})
        row.update({f"validation_{k}": v
                    for k, v in validation.items()})
        HISTORY.append(row)
        pd.DataFrame(HISTORY).to_csv(
            OUT / "learning_curves.csv", index=False
        )
        print(
            f"    step {step}: "
            f"train AUROC={training['auroc']:.3f}; "
            f"validation AUROC={validation['auroc']:.3f}",
            flush=True,
        )

    record(0)
    final_loss = None

    for step in range(1, steps + 1):
        model.train()
        indices = rng.integers(
            0, len(train_y), size=original.BATCH_SIZE
        )
        batch_y = train_y[indices]

        optimizer.zero_grad(set_to_none=True)
        logits = model(train_x[indices]).squeeze(1)
        bce = F.binary_cross_entropy_with_logits(
            logits, batch_y, reduction="none"
        )

        if method == "smote":
            loss = bce.mean()
        else:
            terms = weights[batch_y.long()] * bce
            if method == "focal":
                terms = terms * (
                    1 - torch.exp(-bce)
                ).pow(original.GAMMA)
            loss = terms.mean()

        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite training loss.")

        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach())

        if step in checkpoints:
            record(step)

    model.eval()
    with torch.inference_mode():
        probability = torch.sigmoid(
            model(val_x).squeeze(1)
        ).numpy()

    parameters = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    return probability, final_loss, parameters


def create_plots():
    history = pd.DataFrame(HISTORY)

    # Average seeds within folds, then average folds.
    per_fold = history.groupby(
        ["method", "fold", "step"]
    ).mean(numeric_only=True).reset_index()

    mean_curves = per_fold.groupby(
        ["method", "step"]
    ).mean(numeric_only=True).reset_index()

    mean_curves.to_csv(
        OUT / "mean_learning_curves.csv", index=False
    )

    for metric, ylabel in [
        ("auroc", "AUROC"),
        ("average_precision", "Average precision"),
        ("unweighted_bce", "Unweighted BCE"),
    ]:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        for ax, method in zip(axes, METHODS):
            part = mean_curves[
                mean_curves.method == method
            ].sort_values("step")
            ax.plot(
                part.step, part[f"train_{metric}"],
                label="Real training sessions", linestyle="--",
                color=COLORS[method],
            )
            ax.plot(
                part.step, part[f"validation_{metric}"],
                label="Validation sessions",
                color=COLORS[method],
            )
            ax.set(
                title=LABELS[method],
                xlabel="Optimizer updates",
                ylabel=ylabel,
            )
            ax.legend(fontsize=8)

        fig.suptitle(
            "Development learning curves: seeds averaged within folds"
        )
        fig.tight_layout()
        fig.savefig(OUT / f"{metric}_learning_curves.png", dpi=200)
        plt.close(fig)

    # Preserve fold detail; a mean can hide conflicting trajectories.
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, method in zip(axes, METHODS):
        for fold in range(5):
            part = per_fold[
                (per_fold.method == method)
                & (per_fold.fold == fold)
            ].sort_values("step")
            ax.plot(
                part.step, part.validation_auroc,
                label=f"Fold {fold + 1}",
            )
        ax.set(
            title=LABELS[method],
            xlabel="Optimizer updates",
            ylabel="Validation AUROC",
        )
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT / "validation_auroc_by_fold.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    # Match the existing script's development-session order.
    DEV = pd.read_csv(BASE / "splits.csv")
    DEV = DEV[DEV.split == "dev"].copy()
    available = DEV.session_id.map(
        lambda sid: (
            BASE / "embeddings" / f"{int(sid)}-a_n.npy"
        ).is_file()
    )
    DEV = DEV[available].reset_index(drop=True)

    # Replace only the training function; retain the original
    # coverage checks, split checks, metrics, and result outputs.
    original.train_predict = train_with_curves

    # Always use a separate results directory.
    sys.argv = [
        sys.argv[0],
        "--output", "results/learning_curves",
    ]
    original.main()
    create_plots()
    print(f"Learning-curve reports saved to: {OUT}")
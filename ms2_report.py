"""Generate presentation figures from saved development predictions; no fitting."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, confusion_matrix
from ms2_prepare import ROOT


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=ROOT/"comparison")
    root = p.parse_args().input
    config = json.loads((root/"config.json").read_text())
    if config["smoke"]:
        raise ValueError("Use a full comparison, not the smoke test, for reporting.")
    d = pd.read_csv(root/"oof_predictions.csv")
    m = pd.read_csv(root/"fold_metrics.csv")
    ids = ["session_id","speaker_id","label","dev_fold"]
    keys = set(d[ids].itertuples(index=False,name=None))
    for (method,seed), g in d.groupby(["method","seed"]):
        if len(g) != len(keys) or set(g[ids].itertuples(index=False,name=None)) != keys:
            raise ValueError(f"Unpaired predictions: {method}, {seed}")
    out = root/"report"
    out.mkdir(exist_ok=True)
    metrics = ["average_precision","auroc","balanced_accuracy","sensitivity","specificity"]
    folds = m.groupby(["method","fold"])[metrics].mean().reset_index()
    summary = folds.groupby("method")[metrics].agg(["mean","std"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary.to_csv(out/"method_summary.csv")
    base = "mean_focal"
    new = "contrast_focal"
    pairs = folds.set_index(["method","fold"])
    delta = pairs.loc[new].subtract(pairs.loc[base])
    delta.to_csv(out/"contrast_minus_mean_by_fold.csv")
    print("Paired fold differences (contrast minus mean):")
    print(delta.round(4))
    def export(fig, name):
        fig.tight_layout()
        for ext in ["png","pdf"]:
            fig.savefig(out/f"{name}.{ext}", dpi=180, bbox_inches="tight")
        plt.close(fig)
    fig, axes = plt.subplots(1,2,figsize=(12,5), sharey=True)
    for ax,metric,title in zip(axes,metrics,["Average precision","AUROC"]):
        idx = np.arange(len(summary))
        ax.errorbar(summary[metric+"_mean"],idx,xerr=summary[metric+"_std"],fmt="o",capsize=3)
        ax.set_yticks(idx,summary.index)
        ax.set_xlabel(title+" (mean ± SD across folds)")
        ax.grid(axis="x",alpha=.2)
    export(fig,"development_comparison")
    # Published numbers transcribed from the submitted proposal's benchmark
    # table. Keep protocols visibly separate; never subtract across panels.
    fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
    axes[0].bar(["Voice SVM","Voice + demographics\nlogistic regression"],[.628,.747],
                color="#8896A5")
    axes[0].set_title("Published: FEMH training, external SVD")
    names=["m1_sklearn_reference","mean_focal","contrast_focal"]
    axes[1].bar(["M1 reference","Vowel mean","Mean + contrasts"],
                summary.loc[names,"balanced_accuracy_mean"],color="#548069")
    axes[1].set_title("This project: SVD development folds")
    axes[0].set(ylabel="Balanced accuracy",ylim=(0,1))
    fig.suptitle("Different evaluation protocols; descriptive context only")
    export(fig,"published_context")
    fig,ax=plt.subplots(figsize=(7,4))
    for name in ["a_focal",base,new]:
        line=pairs.loc[name]
        ax.plot(line.index,line.average_precision,"o-",label=name)
    ax.set(xlabel="Development fold",ylabel="Average precision")
    ax.legend()
    export(fig,"paired_folds")
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for name in ["m1_sklearn_reference","a_smote",base,new]:
        g=d[(d.method==name)&(d.seed==42)]
        prec,rec,_=precision_recall_curve(g.label,g.score)
        axes[0].step(rec,prec,where="post",label=name)
    axes[0].axhline(g.label.mean(),color="gray",ls=":",label="Positive prevalence")
    axes[0].set(xlabel="Recall",ylabel="Precision",title="Pooled out-of-fold scores, seed 42")
    axes[0].legend(fontsize=8)
    cm=confusion_matrix(g.label,g.score>=.5,labels=[0,1])
    axes[1].imshow(cm,cmap="Blues")
    for i in range(2):
        for j in range(2):
            axes[1].text(j,i,str(cm[i,j]),ha="center",va="center",
                         color="white" if cm[i,j]>cm.max()/2 else "black")
    axes[1].set(xticks=[0,1],yticks=[0,1],xlabel="Predicted",ylabel="Actual",
                title="Contrast model, seed 42, threshold 0.5")
    export(fig,"development_errors")
    print("Results:",out)


if __name__ == "__main__":
    main()

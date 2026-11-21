import argparse
import json
import time
from itertools import product
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from imblearn.over_sampling import SMOTE
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from ms2_prepare import ROOT, CACHE, VOWELS, development, vector, SPLIT_HASH


class Head(nn.Module):
    def __init__(self, rep, demo):
        super().__init__()
        self.rep, self.demo = rep, demo
        width = 512 if rep == "a" else 1536
        if demo == "only":
            width = 0
        # The three audio arms use the same 1536-input head: /a/ is
        # zero-padded. The attention arm starts as ordinary concatenation.
        self.head = nn.Sequential(nn.Linear((1536 if width else 0) + (2 if demo != "none" else 0),50),
                                  nn.ReLU(),nn.Linear(50,50),nn.ReLU(),nn.Linear(50,1))
        if rep == "attention" and demo != "only":
            self.gate = nn.Sequential(nn.Linear(512,8),nn.Tanh(),nn.Linear(8,1))
            nn.init.zeros_(self.gate[-1].weight)
            nn.init.zeros_(self.gate[-1].bias)

    def forward(self,x):
        blocks = []
        if self.demo != "only":
            a = x[:,:1536].reshape(-1,3,512)
            if self.rep == "attention":
                a = a * (3*self.gate(a).softmax(1))
            blocks.append(a.flatten(1))
        if self.demo != "none":
            blocks.append(x[:,-2:])
        return self.head(torch.cat(blocks,1)).flatten()


def fit(x, xv, y, rep, demo, loss, alpha, gamma, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = Head(rep,demo)
    opt = torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0001)
    weights = len(y)/(2*np.bincount(y,minlength=2))
    if loss == "smote":
        x,y = SMOTE(random_state=seed,k_neighbors=min(5,int(np.bincount(y).min()-1))).fit_resample(x,y)
    x,y,xv = [torch.as_tensor(t,dtype=torch.float32) for t in (x,y,xv)]
    for _ in range(25):
        k = rng.integers(len(y),size=64)
        t = y[k]
        b = F.binary_cross_entropy_with_logits(model(x[k]),t,reduction="none")
        if loss == "weighted":
            b = b * torch.where(t.bool(),float(weights[1]),float(weights[0]))
        elif loss == "focal":
            b = b * torch.where(t.bool(),alpha,1-alpha) * (1-(-b).exp()).pow(gamma)
        opt.zero_grad()
        b.mean().backward()
        opt.step()
    with torch.no_grad():
        return model(xv).sigmoid().numpy()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smoke",action="store_true")
    args = p.parse_args()
    d = development()
    coverage = pd.read_csv(ROOT/"coverage.csv")
    ids = coverage.loc[coverage[list(VOWELS)].notna().all(axis=1),"session_id"]
    d = d[d.session_id.isin(ids)].reset_index(drop=True)
    X = np.stack([[vector(CACHE/f"{s}-{v}_n.npy") for v in VOWELS] for s in d.session_id])
    demo = pd.DataFrame({"age":pd.to_numeric(d.speaker_age,errors="raise"),
                         "sex":d.gender.str.lower().map({"m":1,"w":0,"f":0})}).to_numpy(float)
    # Missing values are imputed on training rows only, never using validation.
    if np.isinf(demo).any():
        raise ValueError("Infinite demographic value")
    out = ROOT/("feedback_smoke" if args.smoke else "feedback_dev")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Existing results at {out}; preserve them before rerunning.")
    out.mkdir(exist_ok=True)
    # Counts of unique speakers with each label; a mixed-label speaker may
    # appear in both label rows. Counts here are for eligible development data.
    d.groupby(["dev_fold","label"]).agg(sessions=("session_id","size"),
        speakers=("speaker_id","nunique")).to_csv(out/"fold_counts.csv")
    pd.DataFrame(demo,columns=["age","sex"]).isna().sum().to_csv(out/"demographic_missing.csv")
    folds = sorted(d.dev_fold.unique())
    seeds = [42] if args.smoke else [42,43,44]
    if args.smoke:
        folds = folds[:1]
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    y = d.label.to_numpy(int)
    predictions, metrics = [], []
    def run(stage,rep,dm,loss,alpha=.5,gamma=2.):
        name=f"{stage}-{rep}-{dm}-{loss}-{alpha}-{gamma}"
        parts=[]
        for f in folds:
            va=d.dev_fold.eq(f).to_numpy()
            tr=~va
            def scaled(vals):
                return StandardScaler().fit(vals[tr]).transform(vals).astype("float32")
            vs=scaled(X.reshape(len(X),-1)) if dm!="only" else None
            if dm!="only" and rep=="a":
                vs[:,512:]=0
            if dm!="none":
                observed=demo[tr,1]
                sex_values=observed[np.isfinite(observed)].astype(int)
                if not len(sex_values) or not np.isfinite(demo[tr,0]).any():
                    raise ValueError("Training fold lacks demographic observations")
                fill=[np.nanmedian(demo[tr,0]),np.bincount(sex_values).argmax()]
                dz=scaled(np.where(np.isnan(demo),fill,demo))
            data=(dz if dm=="only" else vs if dm=="none" else np.column_stack([vs,dz]))
            for seed in seeds:
                start=time.perf_counter()
                pr=fit(data[tr],data[va],y[tr],rep,dm,loss,alpha,gamma,seed)
                ap=average_precision_score(y[va],pr)
                parts.append(ap)
                metrics.append(dict(name=name,fold=int(f),seed=seed,ap=ap,
                    auroc=roc_auc_score(y[va],pr),fit_predict_seconds=time.perf_counter()-start))
                predictions.append(d.loc[va,["session_id","speaker_id","label","dev_fold"]].assign(
                    name=name,seed=seed,score=pr))
        print(name,round(np.mean(parts),4),flush=True)
        return float(np.mean(parts)),name
    reps=["a","concat","attention"]
    av=[run("A",r,"none","smote")[0] for r in reps]
    rep=reps[int(np.argmax(av))]
    # SMOTE result from A is the identical B control.
    candidates=[(max(av),"smote",.5,2.),(run("B",rep,"none","weighted")[0],"weighted",.5,2.)]
    for alpha,gamma in product([.5,.9,.99],[1.,2.]):
        candidates.append((run("B",rep,"none","focal",alpha,gamma)[0],"focal",alpha,gamma))
    _,loss,alpha,gamma=max(candidates,key=lambda row:row[0])
    # For demographics, keep the same objective across all three controls.
    # Use best non-SMOTE objective to avoid interpolating binary sex with SMOTE.
    _,dl,da,dg=max(candidates[1:],key=lambda row:row[0])
    for dm in ["none","plus","only"]:
        run("C",rep,dm,dl,da,dg)
    scores=pd.concat(predictions,ignore_index=True)
    scores.to_csv(out/"oof.csv",index=False)
    measured=pd.DataFrame(metrics)
    measured.to_csv(out/"fold_metrics.csv",index=False)
    measured.groupby(["name","fold"])[["ap","auroc"]].mean().groupby("name").agg(
        ["mean","std"]).to_csv(out/"summary.csv")
    # Thresholds correspond to the mean-score ensemble, never to individual seeds.
    operating=[]
    for name,g in scores.groupby("name"):
        ens=g.groupby("session_id").agg(label=("label","first"),score=("score","mean"))
        fp,tp,thr=roc_curve(ens.label,ens.score,drop_intermediate=False)
        ok=np.flatnonzero(np.isfinite(thr))
        high=ok[tp[ok]>=.8]
        for rule,j in [("youden",ok[np.argmax(tp[ok]-fp[ok])]),("sensitivity80",high[np.argmin(fp[high])])]:
            operating.append(dict(name=name,rule=rule,threshold=thr[j],dev_sensitivity=tp[j],
                                  dev_specificity=1-fp[j]))
    pd.DataFrame(operating).to_csv(out/"development_thresholds.csv",index=False)
    (out/"selection.json").write_text(json.dumps(dict(smoke=args.smoke,representation=rep,
        loss=loss,alpha=alpha if loss=="focal" else None,gamma=gamma if loss=="focal" else None,
        demographic_control_loss=dl,demographic_alpha=da,demographic_gamma=dg,
        primary_metric="mean fold average precision",split_hash=SPLIT_HASH,seeds=seeds,
        steps=25,batch=64,focal_grid={"alpha":[.5,.9,.99],"gamma":[1,2]},
        note="Development-selected; smoke selections have no scientific interpretation."),indent=2))
    print(out)


if __name__=="__main__":
    main()

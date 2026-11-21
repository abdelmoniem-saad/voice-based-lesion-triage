"""Evaluate the frozen development choice and matched controls on complete holdout triplets."""
import json
import hashlib
import time
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
from ms2_feedback_dev import fit
from ms2_prepare import ROOT, BASE, CACHE, VOWELS, SPLIT_HASH, development, vector


def main():
    source = ROOT/"feedback_dev"
    hold = ROOT/"feedback_holdout_inputs"
    out = ROOT/"feedback_test"
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Final results already exist: {out}")
    cfg = json.loads((source/"selection.json").read_text())
    if any(cfg.get(k)!=v for k,v in dict(smoke=False,representation="concat",loss="focal",
          alpha=.9,gamma=2.,steps=25,batch=64,seeds=[42,43,44],split_hash=SPLIT_HASH).items()):
        raise ValueError("This evaluation was specified for the supplied full development selection.")
    config_files=[CACHE/"config.json",hold/"features/config.json"]
    if json.loads(config_files[0].read_text())!=json.loads(config_files[1].read_text()):
        raise ValueError("Inconsistent feature settings")
    thresholds = pd.read_csv(source/"development_thresholds.csv").set_index(["name","rule"])
    arms = {"a_smote":("a","smote","A-a-none-smote-0.5-2.0"),
            "concat_smote":("concat","smote","A-concat-none-smote-0.5-2.0"),
            "concat_focal":("concat","focal","C-concat-none-focal-0.9-2.0")}
    cut={name:{rule:float(thresholds.loc[(tag,rule),"threshold"]) for rule in
               ("youden","sensitivity80")} for name,(_,_,tag) in arms.items()}
    cut["m1_local_reference"]={"fixed_0.5":.5}
    if not all(np.isfinite(v) for rules in cut.values() for v in rules.values()):
        raise ValueError("Invalid development threshold")
    def eligible(frame,path):
        c=pd.read_csv(path)
        if set(frame.session_id)!=set(c.session_id) or c.session_id.duplicated().any():
            raise ValueError("Coverage does not match split")
        return frame[frame.session_id.isin(c.loc[c[list(VOWELS)].notna().all(axis=1),"session_id"])].copy()
    dev=eligible(development(),ROOT/"coverage.csv")
    manifest=pd.read_csv(BASE/"splits.csv")
    test=eligible(manifest[manifest.split.eq("test")],hold/"coverage.csv")
    if set(dev.speaker_id)&set(test.speaker_id) or set(test.label)!={0,1}:
        raise ValueError("Invalid evaluation cohort")
    def features(frame,cache):
        return np.stack([np.concatenate([vector(cache/f"{s}-{v}_n.npy") for v in VOWELS])
                         for s in frame.session_id])
    x=features(dev,CACHE)
    scale=StandardScaler().fit(x)
    x=scale.transform(x)
    xv=scale.transform(features(test,hold/"features"))
    y=dev.label.to_numpy(int)
    truth=test.label.to_numpy(int)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    preds=test[["session_id","speaker_id","label"]].copy()
    durations={}
    for name,(rep,loss,_) in arms.items():
        start=time.perf_counter()
        a,b=x.copy(),xv.copy()
        if rep=="a":
            a[:,512:]=0
            b[:,512:]=0
        cols=[]
        for seed in cfg["seeds"]:
            col=f"{name}_seed{seed}"
            preds[col]=fit(a,b,y,rep,"none",loss,.9,2.,seed)
            cols.append(col)
        preds[name]=preds[cols].mean(axis=1)
        durations[name]=time.perf_counter()-start
    start=time.perf_counter()
    a,yy=SMOTE(random_state=42,k_neighbors=min(5,np.bincount(y).min()-1)).fit_resample(x[:,:512],y)
    ref=MLPClassifier(hidden_layer_sizes=(50,50),random_state=42,batch_size=64,
                      max_iter=500,n_iter_no_change=20).fit(a,yy)
    preds["m1_local_reference"]=ref.predict_proba(xv[:,:512])[:,1]
    durations["m1_local_reference"]=time.perf_counter()-start
    prob={k:preds[k].to_numpy() for k in cut}
    if not all(np.isfinite(a).all() for a in prob.values()):
        raise ValueError("Invalid predictions")
    def metrics(idx):
        result={}
        t=truth[idx]
        for arm,p in prob.items():
            s=p[idx]
            result[arm,"ranking","AP"]=average_precision_score(t,s)
            result[arm,"ranking","AUROC"]=roc_auc_score(t,s)
            for rule,thr in cut[arm].items():
                hit=s>=thr
                sens=hit[t==1].mean()
                spec=1-hit[t==0].mean()
                for k,v in [("sensitivity",sens),("specificity",spec),("balanced_accuracy",(sens+spec)/2)]:
                    result[arm,rule,k]=v
        for left,right in [("concat_smote","a_smote"),("concat_focal","concat_smote")]:
            for k in ["AP","AUROC"]:
                result[left+" minus "+right,"paired_difference",k]=result[left,"ranking",k]-result[right,"ranking",k]
        return result
    point=metrics(np.arange(len(test)))
    draws={k:[] for k in point}
    blocks=list(preds.groupby("speaker_id").indices.values())
    rng=np.random.default_rng(2026)
    for _ in range(2000):
        ix=np.concatenate([blocks[j] for j in rng.integers(len(blocks),size=len(blocks))])
        if len(np.unique(truth[ix]))==2:
            for k,v in metrics(ix).items():
                draws[k].append(v)
    report=[]
    for (arm,rule,k),v in point.items():
        vals=draws[arm,rule,k]
        lo,hi=np.quantile(vals,[.025,.975])
        report.append(dict(model=arm,rule=rule,metric=k,value=v,lower95=lo,upper95=hi,valid_draws=len(vals)))
    out.mkdir(exist_ok=True)
    pd.DataFrame(report).to_csv(out/"results.csv",index=False)
    preds.to_csv(out/"predictions.csv",index=False)
    counts=[]
    for name,frame in [("development",dev),("test",test)]:
        for lab,g in frame.groupby("label"):
            counts.append(dict(split=name,label=lab,sessions=len(g),speakers=g.speaker_id.nunique()))
    pd.DataFrame(counts).to_csv(out/"coverage_counts.csv",index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,5))
    for name,s in prob.items():
        p,r,_=precision_recall_curve(truth,s)
        ax.step(r,p,where="post",label=name)
    ax.axhline(truth.mean(),ls=":",color="gray",label="positive prevalence")
    ax.set(xlabel="Recall",ylabel="Precision",title="Previously evaluated SVD holdout: complete triplets")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out/"precision_recall.png",dpi=180)
    plt.close(fig)
    (out/"protocol.json").write_text(json.dumps(dict(selection=cfg,thresholds=cut,
        fit_predict_seconds=durations,bootstrap="2000 paired speaker-cluster draws, fixed models",
        input_sha256={str(p.name):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [source/"selection.json",source/"development_thresholds.csv"]}),indent=2))
    print(out)


if __name__=="__main__":
    main()

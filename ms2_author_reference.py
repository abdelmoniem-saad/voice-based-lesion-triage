"""Pinned Paterson recording-only MLP on the frozen, matched SVD cohort."""
import argparse
import hashlib
import json
import sys
import warnings
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import ParameterGrid
from threadpoolctl import threadpool_limits
from ms2_prepare import BASE, ROOT, SPLIT_HASH, development

SRC = ROOT / "reference_sources"
OUT = ROOT / "author_reference"
COMMIT = "7811f6d8cc8ac6dbe8535e9e2669febed4f51da4"
HASHES = {
    "FeatureStates_SVD.csv":"0633a7ef29b2c7b8fcebbc8f9b49e3902628784200abadb28baea5b841fb04cc",
    "TrainingUtils.py":"dcb602c1713dd07995dccbd8f634e97c78a918b0cf2cb4724607c7c5cd49e922",
    "custom_preprocessing.py":"a5477132a4422106b3192b10d0b98279ac9a1083b5abae389ce6ad124ce108c8",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fit",action="store_true",help="Perform the one prespecified experiment; default audits only.")
    args=p.parse_args()
    for name,h in HASHES.items():
        if sha(SRC/name)!=h:
            raise ValueError(f"Unexpected upstream file: {name}")
    sys.path.insert(0,str(SRC))
    import TrainingUtils as authors
    dev=development()  # checks split hash and speaker separation
    manifest=pd.read_csv(BASE/"splits.csv")
    dev_cov=pd.read_csv(ROOT/"coverage.csv")
    dev_ids=dev_cov.loc[dev_cov[["a","i","u"]].notna().all(axis=1),"session_id"]
    saved_path=ROOT/"feedback_test/predictions.csv"
    saved=pd.read_csv(saved_path)
    d=dev[dev.session_id.isin(dev_ids)].copy()
    t=manifest[manifest.split.eq("test") & manifest.session_id.isin(saved.session_id)].copy()
    if len(d)!=1066 or len(t)!=252 or len(saved)!=len(t) or saved.session_id.duplicated().any():
        raise ValueError("Cohort differs from completed comparison; stop before fitting.")
    raw=pd.read_csv(SRC/"FeatureStates_SVD.csv")
    sid=raw.file.str.extract(r"^(\d+)-a_n\.wav$",expand=False)
    if sid.isna().any() or sid.astype(int).duplicated().any():
        raise ValueError("Ambiguous release IDs")
    cols=list(map(str,range(512)))
    # Explicit allowlist: upstream SVD includes diagnostic columns.
    raw=raw.assign(session_id=sid.astype(int)).set_index("session_id")
    ids=list(d.session_id)+list(t.session_id)
    X=raw.loc[ids,cols].apply(pd.to_numeric,errors="raise")
    if np.isinf(X.to_numpy()).any() or X.isna().all(axis=1).any():
        raise ValueError("Invalid released features")
    saved=saved.set_index("session_id").loc[t.session_id]
    if not np.array_equal(saved[["speaker_id","label"]].to_numpy(),t[["speaker_id","label"]].to_numpy()):
        raise ValueError("Saved holdout label or speaker mismatch")
    train=X.iloc[:len(d)]
    y=d.label.to_numpy(int)
    fold=d.dev_fold.to_numpy(int)
    folds=[(np.flatnonzero(fold!=i),np.flatnonzero(fold==i)) for i in range(5)]
    pipe=authors.create_classification_pipeline(authors.create_preprocessor("FeatureStates",cols,[]),
                                               {},model_type="MLP")
    grid=list(ParameterGrid(authors.MLP_param_grid))
    protocol=dict(source_commit=COMMIT,source_hashes=HASHES,split_hash=SPLIT_HASH,
                  prediction_hash=sha(saved_path),dev_ids_hash=hashlib.sha256(d.session_id.to_numpy("int64").tobytes()).hexdigest(),
                  grid_size=len(grid),folds=5,criterion="balanced_accuracy",seed=42,
                  versions={pkg:version(pkg) for pkg in ["scikit-learn","imbalanced-learn","numpy","pandas"]},
                  runner_hash=sha(Path(__file__)), stop="Complete original grid once. No holdout-driven changes.")
    if len(grid)!=90:
        raise ValueError("Unexpected upstream grid")
    OUT.mkdir(exist_ok=True)
    lock=OUT/"protocol.json"
    if lock.exists() and json.loads(lock.read_text())!=protocol:
        raise RuntimeError("Existing audit has different inputs or software; review before continuing.")
    lock.write_text(json.dumps(protocol,indent=2))
    counts=pd.concat([d,t]).groupby(["split","label"]).agg(sessions=("session_id","size"),speakers=("speaker_id","nunique"))
    counts.to_csv(OUT/"coverage.csv")
    print(counts)
    print("Audit passed: 512 released features; 90 candidates x 5 speaker folds; one final refit.")
    if not args.fit:
        return
    if (OUT/"results.csv").exists():
        raise RuntimeError("Completed reference exists. Stopping rule: no second holdout run.")
    # Same candidate order, folds, scoring, arithmetic mean, first-maximum
    # tie break and refit as GridSearchCV; checkpoint each candidate on Windows.
    checkpoint=OUT/"cv_candidates.json"
    rows=json.loads(checkpoint.read_text()) if checkpoint.exists() else []
    for j,row in enumerate(rows):
        if row["params"]!=json.loads(json.dumps(grid[j])) or len(row["scores"])!=5:
            raise ValueError("Invalid checkpoint")
    with threadpool_limits(limits=1):
        for j in range(len(rows),len(grid)):
            params=grid[j]
            scores=[]
            with warnings.catch_warnings(record=True) as notes:
                warnings.simplefilter("always")
                for tr,va in folds:
                    m=clone(pipe).set_params(**params).fit(train.iloc[tr],y[tr])
                    scores.append(float(balanced_accuracy_score(y[va],m.predict(train.iloc[va]))))
            rows.append(dict(params=params,scores=scores,mean=float(np.mean(scores)),
                             warnings=sorted({str(w.message) for w in notes})))
            tmp=checkpoint.with_suffix(".tmp")
            tmp.write_text(json.dumps(rows,indent=2))
            tmp.replace(checkpoint)
            print(f"Candidate {j+1}/90 complete",flush=True)
        best=int(np.argmax([r["mean"] for r in rows]))
        (OUT/"selected.json").write_text(json.dumps(rows[best],indent=2))
        model=clone(pipe).set_params(**grid[best]).fit(train,y)
        prob=model.predict_proba(X.iloc[len(d):])[:,list(model.classes_).index(1)]
        pred=model.predict(X.iloc[len(d):])
    if not np.isfinite(prob).all():
        raise ValueError("Invalid holdout predictions")
    truth=t.label.to_numpy(int)
    saved=saved.assign(author_reference=prob,author_prediction=pred)
    saved.to_csv(OUT/"matched_predictions.csv")
    metrics=[]
    def record(name,s):
        metrics.append(dict(model=name,average_precision=average_precision_score(truth,s),
                            auroc=roc_auc_score(truth,s)))
    for name in ["author_reference","m1_local_reference","a_smote","concat_smote","concat_focal"]:
        record(name,saved[name])
    # Paired speaker bootstrap for the prespecified primary comparison.
    groups=list(saved.groupby("speaker_id").indices.values())
    rng=np.random.default_rng(2026)
    focal=saved.concat_focal.to_numpy()
    delta=lambda idx: average_precision_score(truth[idx],focal[idx])-average_precision_score(truth[idx],prob[idx])
    samples=[]
    for _ in range(2000):
        ix=np.concatenate([groups[i] for i in rng.integers(len(groups),size=len(groups))])
        if np.unique(truth[ix]).size==2:
            samples.append(delta(ix))
    (OUT/"paired.json").write_text(json.dumps(dict(contrast="frozen concat focal ensemble minus author reference",
        metric="average precision",difference=float(delta(np.arange(len(t)))),
        percentile95=np.quantile(samples,[.025,.975]).tolist(),valid_draws=len(samples),
        author_holdout_balanced_accuracy_default=float(balanced_accuracy_score(truth,pred))),indent=2))
    pd.DataFrame(metrics).to_csv(OUT/"results.csv",index=False)
    print(OUT)


if __name__=="__main__":
    main()

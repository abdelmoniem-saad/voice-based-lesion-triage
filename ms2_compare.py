"""Paired speaker-grouped development experiments. Does not evaluate the test set."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
from compare_losses import train_predict, score, make_model
from ms2_prepare import ROOT, CACHE, VOWELS, development, vector, SPLIT_HASH

SEEDS = [42, 43, 44]
STEPS = 25
RANK = 8


def contrast_features(tr, va):
    # Orthonormal signed vowel contrasts. Together with the mean these
    # retain all three views before PCA. Order in each session is a,i,u.
    def contrasts(x):
        a, i, u = x.transpose(1, 0, 2)
        return np.column_stack(((a-i)/np.sqrt(2), (a+i-2*u)/np.sqrt(6)))
    scaler = StandardScaler().fit(contrasts(tr))
    t = scaler.transform(contrasts(tr))
    pca = PCA(n_components=RANK, svd_solver="randomized", random_state=42).fit(t)
    z = pca.transform(t)
    last = StandardScaler().fit(z)
    return last.transform(z), last.transform(pca.transform(scaler.transform(contrasts(va))))


def residual_predict(x, y, xv, z, zv, seed, linear=False):
    # Same mean branch, initialization, batches, focal weights and updates
    # as the mean-only control. New logits start at exactly zero.
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = make_model([] if linear else [50, 50])
    correction = nn.Parameter(torch.zeros(RANK))
    optimizer = torch.optim.Adam([*net.parameters(), correction], lr=.001, weight_decay=.0001)
    x, xv, z, zv, labels = [torch.as_tensor(a, dtype=torch.float32)
                             for a in (x, xv, z, zv, y)]
    weights = torch.tensor(len(y)/(2*np.bincount(y)), dtype=torch.float32)
    for _ in range(STEPS):
        k = rng.integers(0, len(y), size=64)
        target = labels[k]
        logits = net(x[k]).flatten() + z[k] @ correction
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        loss = (weights[target.long()] * (1-torch.exp(-bce)).square() * bce).mean()
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite training loss")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        return torch.sigmoid(net(xv).flatten() + zv @ correction).numpy()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smoke", action="store_true", help="First fold and first seed only")
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    out = args.out or ROOT / ("smoke" if args.smoke else "comparison")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Choose a fresh --out directory: {out}")
    d = development()
    coverage = pd.read_csv(ROOT / "coverage.csv", keep_default_na=False)
    d = d.merge(coverage[["session_id", *VOWELS]], validate="one_to_one", how="left", on="session_id")
    if d[list(VOWELS)].isna().any().any():
        raise ValueError("Incomplete coverage manifest; repeat audit")
    d = d[d[list(VOWELS)].ne("").all(axis=1)].reset_index(drop=True)
    X = np.stack([[vector(CACHE / f"{s}-{v}_n.npy") for v in VOWELS] for s in d.session_id])
    y = d.label.to_numpy(int)
    folds = sorted(d.dev_fold.unique())
    if len(folds) != 5 or any(set(y[d.dev_fold.eq(f)]) != {0, 1} for f in folds):
        raise ValueError("Need five usable development folds")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    out.mkdir(parents=True, exist_ok=True)
    config = dict(split_sha256=SPLIT_HASH, steps=STEPS, rank=RANK, threshold=.5, smoke=args.smoke,
                  seeds=SEEDS[:1] if args.smoke else SEEDS, primary_metric="average_precision",
                  primary_pair=["contrast_focal","mean_focal"], cohort="complete development triplets",
                  feature_config=json.loads((CACHE/"config.json").read_text()),
                  source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                  [Path(__file__), Path(__file__).with_name("compare_losses.py"),
                   Path(__file__).with_name("ms2_prepare.py")]})
    (out/"config.json").write_text(json.dumps(config, indent=2))
    rows, preds = [], []
    for f in (folds[:1] if args.smoke else folds):
        va = d.dev_fold.eq(f).to_numpy()
        tr = ~va
        def scaled(a):
            s = StandardScaler().fit(a[tr])
            return s.transform(a[tr]), s.transform(a[va])
        views = {v:scaled(X[:, j]) for j,v in enumerate(VOWELS)}
        mean = scaled(X.mean(axis=1))
        z, zv = contrast_features(X[tr], X[va])
        yt = y[tr]
        def save(name, seed, prob):
            if not np.isfinite(prob).all():
                raise ValueError(f"Nonfinite predictions: {name}")
            rows.append(dict(method=name, seed=seed, fold=int(f), **score(y[va], prob)))
            frame = d.loc[va, ["session_id","speaker_id","label","dev_fold"]].copy()
            preds.append(frame.assign(method=name, seed=seed, score=prob))
        # M1 sklearn reference: original training recipe, selected architecture,
        # re-fitted on exactly the same triplets as every proposed variant.
        a, av = views["a"]
        xs, ys = SMOTE(random_state=42, k_neighbors=min(5,int(np.bincount(yt).min())-1)).fit_resample(a, yt)
        ref = MLPClassifier(hidden_layer_sizes=(50,50), random_state=42, batch_size=64,
                            max_iter=500, n_iter_no_change=20).fit(xs, ys)
        save("m1_sklearn_reference", 42, ref.predict_proba(av)[:,1])
        for seed in config["seeds"]:
            for name, (x,xv), method in [
                ("a_smote",views["a"],"smote"), ("a_weighted",views["a"],"weighted_bce"),
                *((v+"_focal",views[v],"focal") for v in VOWELS),
                ("mean_focal",mean,"focal")]:
                pred, _, _ = train_predict(x,yt,xv,[50,50],method,seed,STEPS)
                save(name,seed,pred)
            for name, linear in [("contrast_focal",False), ("contrast_linear",True)]:
                save(name,seed,residual_predict(mean[0],yt,mean[1],z,zv,seed,linear))
        print(f"Finished fold {f}", flush=True)
    pd.DataFrame(rows).to_csv(out/"fold_metrics.csv", index=False)
    pd.concat(preds).to_csv(out/"oof_predictions.csv", index=False)
    print(out)


if __name__ == "__main__":
    main()

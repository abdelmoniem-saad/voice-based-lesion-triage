"""Audit development vowel triplets and cache missing frozen XLSR features."""
import argparse
import hashlib
import json
import math
import re
import time
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
from extract_embeddings import BASE, MODEL, SR, read_nsp

ROOT = BASE / "milestone2"
CACHE = ROOT / "features"
VOWELS = "aiu"
SPLIT_HASH = "7404ad74654439cc01d476b4b27c49aef043c10f06d471f8c3756465a8cb0d57"


def development():
    path = BASE / "splits.csv"
    if hashlib.sha256(path.read_bytes()).hexdigest() != SPLIT_HASH:
        raise ValueError("Split manifest differs from the Milestone 1 run.")
    d = pd.read_csv(path)
    if d.session_id.duplicated().any() or d.groupby("speaker_id").split.nunique().max() != 1:
        raise ValueError("Duplicate sessions or speaker overlap.")
    d = d.loc[d.split.eq("dev")].copy()
    d["dev_fold"] = d.dev_fold.astype(int)
    if d.groupby("speaker_id").dev_fold.nunique().max() != 1:
        raise ValueError("Development speaker spans multiple folds.")
    return d


def vector(path):
    v = np.load(path, allow_pickle=False)
    if v.shape != (512,) or not np.isfinite(v).all():
        raise ValueError(f"Invalid embedding: {path}")
    return v.astype(np.float32)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audit-only", action="store_true")
    p.add_argument("--holdout", action="store_true", help="Prepare separate test feature cache, without evaluating models")
    p.add_argument("--limit", type=int, help="Maximum number of NEW vectors to extract")
    args = p.parse_args()
    if args.limit is not None and args.limit < 1:
        p.error("--limit must be positive")
    d = development()
    root_out = ROOT
    if args.holdout:
        manifest = pd.read_csv(BASE/"splits.csv")
        d = manifest.loc[manifest.split.eq("test")].copy()
        root_out = ROOT/"feedback_holdout_inputs"
    cache = root_out/"features"
    ids = set(d.session_id)
    root_out.mkdir(parents=True,exist_ok=True)
    paths = {}
    for folder in ["svd_data positive", "svd_data negative"]:
        root = BASE / folder
        if not root.is_dir():
            raise FileNotFoundError(root)
        for path in root.rglob("*.nsp"):
            m = re.fullmatch(r"(\d+)-([aiu])_n\.nsp", path.name, re.I)
            if not m or int(m[1]) not in ids:
                continue
            key = int(m[1]), m[2].lower()
            if key in paths and paths[key].read_bytes() != path.read_bytes():
                raise ValueError(f"Conflicting audio: {key}")
            paths[key] = path
    for v in VOWELS:
        d[v] = [str(paths.get((s, v), "")) for s in d.session_id]
    d["complete"] = d[list(VOWELS)].ne("").all(axis=1)
    d.to_csv(root_out / "coverage.csv", index=False)
    groups = ["label"] if args.holdout else ["dev_fold","label"]
    counts = d.groupby(groups).agg(listed=("session_id","size"),
                                                   complete=("complete","sum"))
    print(counts)
    counts.to_csv(root_out / "coverage_by_fold.csv")
    eligible = d[d.complete]
    if args.holdout and set(eligible.label) != {0,1}:
        raise ValueError("Both classes are required for evaluation")
    if not args.holdout and (eligible.groupby(["dev_fold", "label"]).size().reindex(
            pd.MultiIndex.from_product([sorted(d.dev_fold.unique()), [0, 1]]), fill_value=0) == 0).any():
        raise ValueError("A complete-triplet fold lacks one class. Review coverage before training.")
    if args.audit_only:
        return
    config = json.loads((BASE / "embeddings/extraction_config.json").read_text())
    expected = dict(model=MODEL, layer="final convolution before feature projection",
                    pooling="time mean", dimension=512, sample_rate=SR, normalize=True, vowel="a_n")
    if config != expected:
        raise ValueError("Unexpected Milestone 1 feature configuration")
    import torch
    from scipy.signal import resample_poly
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
    cache.mkdir(exist_ok=True)
    meta = cache / "config.json"
    settings = dict(expected, vowel=list(VOWELS), split_sha256=SPLIT_HASH)
    if meta.exists() and json.loads(meta.read_text()) != settings:
        raise ValueError("Incompatible cache")
    if not meta.exists() and any(cache.glob("*.npy")):
        raise ValueError("Cache has no provenance")
    meta.write_text(json.dumps(settings, indent=2))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = None
    made = 0
    for s in eligible.session_id:
        for v in VOWELS:
            path = paths[s, v]
            stem = f"{s}-{v}_n.npy"
            out = cache / stem
            if out.exists():
                vector(out)
                continue
            old = BASE / "embeddings" / stem
            started = time.perf_counter()
            reuse = v == "a" and old.exists()
            if reuse:
                emb = vector(old)
            else:
                if encoder is None:
                    encoder = Wav2Vec2Model.from_pretrained(MODEL).feature_extractor.to(device).eval()
                    processor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=SR,
                        padding_value=0., do_normalize=True, return_attention_mask=False)
                audio, rate = read_nsp(path)
                g = math.gcd(rate, SR)
                audio = resample_poly(audio, SR//g, rate//g).astype(np.float32)
                inp = processor(audio, sampling_rate=SR, return_tensors="pt").input_values
                with torch.inference_mode():
                    emb = encoder(inp.to(device)).mean(-1)[0].cpu().numpy()
            if emb.shape != (512,) or not np.isfinite(emb).all():
                raise ValueError(f"Invalid vector for {path}")
            with NamedTemporaryFile(dir=cache, suffix=".tmp", delete=False) as f:
                np.save(f, emb)
                temp = Path(f.name)
            temp.replace(out)
            if args.holdout:
                timings = root_out/"feature_times.csv"
                header = not timings.exists()
                with timings.open("a") as log:
                    if header:
                        log.write("session_id,vowel,reused,device,seconds\n")
                    log.write(f"{s},{v},{int(reuse)},{device},{time.perf_counter()-started:.6f}\n")
            made += 1
            if made % 100 == 0:
                print(f"{made} vectors written", flush=True)
            if made == args.limit:
                return
    print(f"Complete: {len(eligible)} triplets; {made} new cache files.")


if __name__ == "__main__":
    main()

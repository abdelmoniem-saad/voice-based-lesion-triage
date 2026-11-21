"""XLSR-53 mean-pooled convolution features for split-listed normal-pitch /a/.
Install: python -m pip install numpy scipy torch transformers
Run: python extract_embeddings.py --audit-only
Then: python extract_embeddings.py --limit 1
Then: python extract_embeddings.py
"""
import argparse
import csv
import hashlib
import json
import math
import re
import struct
import time
from pathlib import Path
import numpy as np

BASE = Path(__file__).resolve().parent
MODEL = 'facebook/wav2vec2-large-xlsr-53'
SR = 16000

def read_nsp(path):
    raw = path.read_bytes()
    if raw[:8] != b'FORMDS16':
        raise ValueError(f'Unsupported NSP header: {path}')
    offset = 12
    rate = count = samples = None
    while offset + 8 <= len(raw):
        tag = raw[offset:offset+4]
        size = struct.unpack_from('<I', raw, offset+4)[0]
        start, end = offset+8, offset+8+size
        if end > len(raw):
            raise ValueError(f'Truncated NSP chunk: {path}')
        if tag == b'HEDR':
            if size < 32:
                raise ValueError(f'Short NSP header: {path}')
            rate, count = struct.unpack_from('<II', raw, start+20)
        elif tag == b'SDA_':
            samples = raw[start:end]
        elif tag == b'SDB_':
            raise ValueError(f'Stereo NSP unsupported: {path}')
        offset = end
    if rate is None or samples is None or not 8000 <= rate <= 192000:
        raise ValueError(f'Missing/invalid NSP metadata: {path}')
    if len(samples) != count*2 or not count:
        raise ValueError(f'NSP sample count mismatch: {path}')
    audio = np.frombuffer(samples, dtype='<i2').astype(np.float32)/32768.0
    if not np.any(audio):
        raise ValueError(f'Silent recording: {path}')
    return audio, rate

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-only', action='store_true')
    parser.add_argument('--allow-missing', action='store_true', help='Explicitly extract available sessions; preserve splits and report omissions')
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    with (BASE/'splits.csv').open(newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    ids = {str(int(r['session_id'])) for r in rows}
    if len(ids) != len(rows):
        raise ValueError('Duplicate session IDs in splits.csv')
    found = {}
    for folder in ('svd_data positive', 'svd_data negative'):
        root = BASE/folder
        if not root.is_dir():
            raise FileNotFoundError(root)
        for path in root.rglob('*.nsp'):
            match = re.fullmatch(r'(\d+)-a_n\.nsp', path.name, re.I)
            if match and match[1] in ids:
                key = match[1]
                if key in found:
                    if hashlib.sha256(found[key].read_bytes()).digest() != hashlib.sha256(path.read_bytes()).digest():
                        raise ValueError(f'Conflicting audio copies: {key}')
                else:
                    found[key] = path
    missing = sorted(ids-found.keys(), key=int)
    print(f'Split sessions: {len(ids)}; available /a/: {len(found)}; missing: {len(missing)}', flush=True)
    with (BASE/'embedding_coverage.csv').open('w', newline='', encoding='utf-8') as f:
        report = csv.DictWriter(f, fieldnames=['session_id', 'label', 'split', 'available', 'audio_path'])
        report.writeheader()
        for row in rows:
            key = str(int(row['session_id']))
            report.writerow(dict(session_id=key, label=row['label'], split=row['split'], available=int(key in found), audio_path=str(found.get(key, ''))))
    paths = [found[k] for k in sorted(found, key=int)]
    for path in paths:
        read_nsp(path)
    print(f'Audio audit passed: {len(paths)} valid NSP recordings.', flush=True)
    if missing:
        print(f'Missing /a/ sessions: {missing}', flush=True)
        if not args.allow_missing and not args.audit_only:
            raise RuntimeError('See embedding_coverage.csv. Recover missing audio or explicitly pass --allow-missing.')
    if args.audit_only:
        return
    import torch
    from scipy.signal import resample_poly
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
    out_dir = BASE/'embeddings'
    out_dir.mkdir(exist_ok=True)
    config = dict(model=MODEL, layer='final convolution before feature projection', pooling='time mean', dimension=512, sample_rate=SR, normalize=True, vowel='a_n')
    meta = out_dir/'extraction_config.json'
    if meta.exists():
        if json.loads(meta.read_text()) != config:
            raise RuntimeError('Cached feature configuration differs; use a separate output folder.')
    elif any(out_dir.glob('*.npy')):
        raise RuntimeError('Existing vectors lack provenance; move them to a separate folder first.')
    meta.write_text(json.dumps(config, indent=2)+'\n', encoding='utf-8')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Loading {MODEL} on {device}; first run downloads weights.', flush=True)
    model = Wav2Vec2Model.from_pretrained(MODEL).eval()
    encoder = model.feature_extractor.to(device).eval()
    del model
    extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=SR, padding_value=0.0, do_normalize=True, return_attention_mask=False)
    if args.limit:
        paths = paths[:args.limit]
    timing = BASE/'embedding_times.csv'
    header = not timing.exists() or timing.stat().st_size == 0
    done = skipped = 0
    with timing.open('a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if header:
            writer.writerow(['file', 'seconds'])
        for path in paths:
            out = out_dir/(path.stem+'.npy')
            if out.exists():
                cached = np.load(out, allow_pickle=False)
                if cached.shape != (512,) or not np.isfinite(cached).all():
                    raise ValueError(f'Invalid cached vector: {out}')
                skipped += 1
                continue
            start = time.perf_counter()
            audio, rate = read_nsp(path)
            divisor = math.gcd(rate, SR)
            audio = resample_poly(audio, SR//divisor, rate//divisor).astype(np.float32)
            inputs = extractor(audio, sampling_rate=SR, return_tensors='pt')
            with torch.inference_mode():
                vector = encoder(inputs.input_values.to(device)).mean(dim=-1).squeeze(0).cpu().numpy()
            if vector.shape != (512,) or not np.isfinite(vector).all():
                raise ValueError(f'Invalid model output: {path}')
            temp = out.with_suffix('.tmp')
            with temp.open('wb') as target:
                np.save(target, vector)
            temp.replace(out)
            elapsed = time.perf_counter()-start
            writer.writerow([path.name, f'{elapsed:.3f}'])
            f.flush()
            done += 1
            print(f'{done+skipped}/{len(paths)}: {path.name} ({elapsed:.2f}s)', flush=True)
    print(f'Done: {done} extracted, {skipped} cached. Output: {out_dir}')

if __name__ == '__main__':
    main()

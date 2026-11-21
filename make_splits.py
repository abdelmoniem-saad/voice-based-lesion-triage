import sqlite3
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

DB = r"svd_data\sbvoice.db"
POSITIVE = [
    "Stimmlippenkarzinom", "Hypopharynxtumor", "Kehlkopftumor",
    "Epiglottiskarzinom", "Mesopharynxtumor", "Carcinoma in situ",
    "Dysplastische Dysphonie", "Dysplastischer Kehlkopf",
]

con = sqlite3.connect(DB)
sessions = pd.read_sql("select * from recording_sessions", con)
speakers = pd.read_sql("select * from speakers", con)
paths    = pd.read_sql("select * from pathologies", con)
rsp      = pd.read_sql("select * from recording_session_pathologies", con)

sp = rsp.merge(paths, left_on="pathology_id", right_on="id")[["session_id", "name"]]
sessions = sessions.merge(speakers[["id", "gender"]], left_on="speaker_id", right_on="id",
                          suffixes=("", "_spk"))
sessions["label"] = sessions.id.isin(set(sp[sp.name.isin(POSITIVE)].session_id)).astype(int)
sessions["pathological"] = sessions.id.isin(set(sp.session_id))

df = sessions[sessions.pathological & (sessions.speaker_age >= 18)].copy()
df = df.rename(columns={"id": "session_id"})[
    ["session_id", "speaker_id", "speaker_age", "gender", "label"]].reset_index(drop=True)

# ---- one row per speaker; a speaker is positive if ANY session is positive ----
spk = df.groupby("speaker_id").agg(n_sessions=("session_id", "count"),
                                   n_pos_sessions=("label", "sum")).reset_index()
spk["any_pos"] = (spk.n_pos_sessions > 0).astype(int)

print("positive sessions per POSITIVE speaker (largest first):",
      sorted(spk[spk.any_pos == 1].n_pos_sessions.tolist(), reverse=True))

SEED = 42   # fixed in advance; do NOT change it after seeing any model results

# test split: 1 of 5 speaker-level folds (about 20 percent of speakers)
outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
_, test_idx = next(outer.split(spk, spk.any_pos, groups=spk.speaker_id))
test_speakers = set(spk.speaker_id.iloc[test_idx])

# 5 development folds, also split by speaker
dev_spk = spk[~spk.speaker_id.isin(test_speakers)].reset_index(drop=True)
inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
fold_of = {}
for k, (_, val_idx) in enumerate(inner.split(dev_spk, dev_spk.any_pos, groups=dev_spk.speaker_id)):
    for s in dev_spk.speaker_id.iloc[val_idx]:
        fold_of[s] = k

df["split"] = df.speaker_id.map(lambda s: "test" if s in test_speakers else "dev")
df["dev_fold"] = df.speaker_id.map(lambda s: fold_of.get(s, -1))
df["weight"] = 1.0 / df.groupby("speaker_id").session_id.transform("count")  # each speaker counts once

assert df.groupby("speaker_id").split.nunique().max() == 1
assert df[df.split == "dev"].groupby("speaker_id").dev_fold.nunique().max() == 1

df.to_csv("splits.csv", index=False)   # overwrites the previous splits.csv

def summary(d):
    return pd.Series({"sessions": len(d), "pos_sessions": int(d.label.sum()),
                      "speakers": d.speaker_id.nunique(),
                      "pos_speakers": d[d.label == 1].speaker_id.nunique()})

print(df.groupby("split").apply(summary))
print(df[df.split == "dev"].groupby("dev_fold").apply(summary))
import sqlite3
import pandas as pd

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

sessions["has_pathology"] = sessions.id.isin(set(sp.session_id))
sessions["positive"] = sessions.id.isin(set(sp[sp.name.isin(POSITIVE)].session_id))

print("type counts:\n", sessions["type"].value_counts())
print("type 'n' sessions that have a pathology row:",
      int(((sessions["type"] == "n") & sessions.has_pathology).sum()))
print("type 'p' sessions with NO pathology row:",
      int(((sessions["type"] == "p") & ~sessions.has_pathology).sum()))

base = sessions[(sessions["type"] == "p") & sessions.has_pathology
                & (sessions.speaker_age >= 18)].copy()

print("\n=== SUBSET: adult, pathological ===")
print("sessions:", len(base), "| positive:", int(base.positive.sum()),
      "| negative:", int((~base.positive).sum()))

g = base.groupby("speaker_id").agg(n_sessions=("id", "count"), n_pos=("positive", "sum"))
print("distinct speakers:", len(g))
print("distinct POSITIVE speakers:", int((g.n_pos > 0).sum()))
print("distinct NEGATIVE-only speakers:", int((g.n_pos == 0).sum()))
print("speakers with >1 session:", int((g.n_sessions > 1).sum()))
print("speakers with >1 POSITIVE session:", int((g.n_pos > 1).sum()))
print("CONFLICTED speakers (positive AND negative sessions):",
      int(((g.n_pos > 0) & (g.n_pos < g.n_sessions)).sum()))

print("\n=== per positive diagnosis (sessions / speakers) ===")
for name in POSITIVE:
    ids = set(sp[sp.name == name].session_id)
    sub = base[base.id.isin(ids)]
    print(f"{name:28s} sessions={len(sub):3d} speakers={sub.speaker_id.nunique():3d}")

pos = base[base.positive]
print("\n=== positive sessions by gender ===\n", pos.gender.value_counts())
print("positive age: min/mean/max =", pos.speaker_age.min(),
      round(pos.speaker_age.mean(), 1), pos.speaker_age.max())
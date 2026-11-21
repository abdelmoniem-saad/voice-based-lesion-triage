import sqlite3
import time
import pandas as pd
from sbvoicedb import database, Pathology

def open_one(name=None):
    if name is None:
        return database.SbVoiceDb('svd_data', include_healthy=False)
    return database.SbVoiceDb('svd_data', pathology_filter=Pathology.name == name,
                              include_healthy=False)

names = open_one().get_pathology_names()
print(len(names), "diagnoses to fetch")

for i, name in enumerate(names, 1):
    ok = False
    for attempt in range(1, 6):
        try:
            db = open_one(name)
            if all(p.downloaded for p in db.iter_pathologies()):
                print(f"[{i}/{len(names)}] {name}: already downloaded, skipping")
                ok = True
                break
            print(f"[{i}/{len(names)}] {name}: downloading (attempt {attempt})")
            db.download_data()
            open_one(name)   # reopen so the recordings table and the downloaded flag update
            ok = True
            break
        except Exception as e:
            print("   failed:", type(e).__name__, "- retrying in 15 s")
            time.sleep(15)
    if not ok:
        print(f"   GAVE UP on {name}; re-run the script to try again")

# ---- final check against your splits ----
db = open_one()
con = sqlite3.connect(db.dbfile)
recs = pd.read_sql("select session_id, utterance from recordings", con)
have = recs[recs.utterance.isin(["a_n", "i_n", "u_n"])].groupby("session_id").utterance.nunique()
splits = pd.read_csv("splits.csv")
splits["audio_downloaded"] = splits.session_id.isin(recs.session_id)
splits["all3_vowels"] = splits.session_id.map(have).fillna(0).eq(3)
print(splits.groupby("label")[["audio_downloaded", "all3_vowels"]].agg(["sum", "count"]))
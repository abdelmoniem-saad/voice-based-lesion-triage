import sqlite3
import pandas as pd
from sbvoicedb import database, Pathology, Recording

POSITIVE = [
    "Stimmlippenkarzinom", "Hypopharynxtumor", "Kehlkopftumor",
    "Epiglottiskarzinom", "Mesopharynxtumor", "Carcinoma in situ",
    "Dysplastische Dysphonie", "Dysplastischer Kehlkopf",
]

def open_db():
    return database.SbVoiceDb(
        'svd_data',
        pathology_filter=Pathology.name.in_(POSITIVE),
        include_healthy=False,
    )

db = open_db()
print("before:", db.number_of_sessions_downloaded, "/", db.number_of_all_sessions)
db.download_data()

db = open_db()   # reopen so the recordings table gets filled in
print("after:", db.number_of_sessions_downloaded, "/", db.number_of_all_sessions)

con = sqlite3.connect(db.dbfile)
recs = pd.read_sql("select utterance, rate, length from recordings", con)
print("\nrecording codes:\n", recs.utterance.value_counts())
print("\nsample rates:\n", recs.rate.value_counts())
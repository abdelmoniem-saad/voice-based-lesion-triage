from sbvoicedb import database
db = database.SbVoiceDb('svd_data')

for name in ["download_data", "lazy_download", "missing_datasets",
             "recording_filter", "recording_summary", "number_of_sessions_downloaded"]:
    print("=" * 20, name)
    help(getattr(db, name))
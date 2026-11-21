# Compute accounting: SVD experiments

The holdout extraction log identifies CPU execution. This workstation reports an Intel Core i7-1185G7 at 3.00 GHz. The PyTorch comparison script requested two CPU threads; the authors' grid runner limited native thread pools to one. Values below are elapsed seconds, not CPU-core seconds.

| Stage | Evidence | Time | Interpretation |
|---|---|---:|---|
| Earlier /a/ extraction | embedding_times.csv, first logged entry per unique filename | 153.84 s for 1,318 unique files | 1,334 rows; overlapping processes and 16 duplicate entries preclude a clean wall-clock total. Historical context only. |
| Development triplet preparation | First and last writes of 3,198 feature files | about 220 s | File timestamp span, not instrumented runtime; includes cached /a/ copies. Startup before first write excluded; interruptions, if any, could be included. |
| Exploratory contrast comparison | comparison/config.json to oof_predictions.csv | about 11 s | File timestamp span. |
| Main development fits | feedback_dev/fold_metrics.csv, 195 fit/predict timers | 9.98 s | Includes all three stages and seeds; excludes preparation and report generation. Main development output timestamp span was about 14 s. |
| Holdout /i/ and /u/, fresh | feedback_holdout_inputs/feature_times.csv, 504 CPU records | 50.82 s | 0.101 s per fresh vector on average, including model setup in the first applicable timer. /i/: 25.67 s; /u/: 25.15 s. |
| Holdout /a/, cached | same log, 252 cache copies | 7.20 s | Copy/validation, not fresh feature extraction. Total logged holdout preparation: 58.01 s. |
| Four holdout systems | feedback_test/protocol.json, fit_predict_seconds | 1.69 s | Fits and predictions only. Excludes triplet preparation, 2,000-draw bootstrap and figures. |
| Authors' reference | author_reference/protocol.json last write to results.csv last write | about 275 s | Timestamp proxy for 450 CV fits, one refit and final reporting. It is not a separately timed training-only figure. |

**Budget interpretation.** The nonoverlapping recorded stages of Milestone 2 above sum to roughly **9.7 minutes** (220 + 11 + 14 + 58 + 1.69 + 275 seconds). This is a reconstruction of the main stages using mixed timing sources, not a measured end-to-end total. It excludes, among other things, downloads, environment startup outside those windows, the separate project holdout bootstrap, and slide/report production outside those windows. Do not add the earlier /a/ total as if it were fresh Milestone 2 work. The preliminary “few hours” was a planning budget rather than measured runtime.

**Slide wording:** “On CPU, 504 newly extracted holdout vowel vectors took about 51 seconds. The 450-fit reference search plus refit and reporting spanned about 4 minutes 35 seconds by file timestamps. The main recorded Milestone 2 stages accounted for roughly ten minutes; a full end-to-end run was not timed.”

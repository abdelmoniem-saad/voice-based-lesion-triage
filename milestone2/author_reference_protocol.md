# Paterson reference: one bounded in-domain experiment

Question: How does the authors' recording-only FeatureStates MLP, retrained on our SVD labels and speaker folds, compare with our already frozen three-vowel system on exactly the same sessions? This does not reproduce a FEMH-trained published score.

Source: [authors' repository at the inspected revision](https://github.com/mary-paterson/LaryngealCancerClassificationBenchmark/tree/7811f6d8cc8ac6dbe8535e9e2669febed4f51da4). The inspected files and released SVD vectors are in reference_sources. The released model was inspected as serialized bytes; the runner never loads that pickle.

| Component | Earlier local reference | Source-based reference |
|---|---|---|
| Audio | Our NSP resampling and time-mean of the convolution module | Released /a/ FeatureStates, columns 0 through 511. The upstream extraction notebook pads short clips, batches padded inputs, and averages model output extract_features. Same checkpoint and width do not imply identical inputs. |
| Preprocessing | Scale all features | Mean imputation, scaling, tree-based selection fit on training rows |
| Training | SMOTE + fixed ReLU/Adam settings; early stopping off | Authors' imported preprocessor and MLP pipeline, including SMOTE and early_stopping=True; sklearn solver determines whether early stopping applies |
| Search | Six layer layouts; mean development AUROC | Five layer layouts × two activations × three solvers × three learning-rate settings. Mean development balanced accuracy, first maximum on ties. The released recording-only MLP artifact also contains this grid. |
| Protocol | Speaker-grouped SVD | Same project labels, folds, eligible triplets and previously evaluated holdout. Upstream training was on FEMH. Record installed package versions; do not claim bitwise replication of the upstream environment. |

The runner implements a resumable equivalent of the published GridSearchCV loop around the imported estimator. All 450 development fits are attempted, then one refit on 1,066 development sessions, then one evaluation on 252 holdout sessions. No diagnosis, demographics, filename, or recording ID is passed to the classifier. Internal MLP early stopping is the authors' implementation, not a separately speaker-grouped inner validation experiment.

The primary comparison is **holdout session-level average precision**, frozen three-seed project ensemble minus the single-seed authors' reference. Report AUROC alongside it, and the reference's balanced accuracy at its own default predictions. The paired percentile interval resamples speakers with both methods' scores kept together; it conditions on these fitted models. This is a whole-system comparison; use the already completed development ablations to discuss individual changes.

**Stopping rule:** independent of the size or direction of the result, stop modeling once source provenance, split and cohort checks pass, the documented grid completes, and the matched report is saved. If a check or candidate fails, diagnose the implementation; do not silently drop candidates or change thresholds. A wide interval is an uncertainty finding, not a reason to launch another search on the same holdout.

From the Milestone 1 directory:

    python ms2_author_reference.py
    python ms2_author_reference.py --fit

The first command audits only. The second resumes completed candidates after an interruption and refuses to repeat a completed evaluation. Run only one instance. Outputs: milestone2/author_reference/{protocol.json,coverage.csv,cv_candidates.json,selected.json,matched_predictions.csv,paired.json,results.csv}.

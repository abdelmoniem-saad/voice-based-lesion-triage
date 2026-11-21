# Architecture

## Design boundaries

The repository separates experimental code, recorded evidence, presentation outputs and the evidence-review application. Existing experiment entry points remain at the root as frozen research scripts. Moving them would change `__file__`-based paths, cross-imports and, when edited, source hashes recorded in completed runs. The organized command catalog in `scripts/project.py` provides a single navigation layer without changing scientific behavior.

## Modules

| Layer | Files | Responsibility |
|---|---|---|
| Historical data acquisition | `download_all.py`, `download_test.py` | Original SVD download utilities; both use the old `svd_data` directory |
| Historical dataset inspection | `test.py`, `str.py`, `make_splits.py` | Database inspection, API help and original speaker split creation; `test.py` is not an automated test suite |
| Audio features | `extract_embeddings.py`, `ms2_prepare.py` | NSP parsing, resampling, frozen convolutional embeddings, triplet coverage and cached vectors |
| Milestone 1 training | `train_baseline.py`, `compare_losses.py`, `compare_short_training.py`, `diagnose_training.py` | Scikit-learn reference, PyTorch loss comparisons and overfitting diagnostics |
| Milestone 1 evaluation | `report_results.py`, `evaluate_test.py`, `plot_final_test.py` | Development reports, fixed holdout scores and figures |
| Exploratory multi-vowel work | `ms2_compare.py`, `ms2_report.py` | Single-vowel, mean and contrast experiments and paired fold summaries |
| Main multi-vowel experiments | `ms2_feedback_dev.py`, `ms2_feedback_test.py` | Matched fusion/loss/demographic ablations; frozen selection, thresholds and holdout results |
| Source-based reference | `ms2_author_reference.py` | Pinned upstream pipeline, fixed 90-candidate grid, matched comparison and provenance checks |
| Evidence application | `evaluation_assistant/evidence.py` | Allowlisted aggregate sources and deterministic evidence cards |
| Application controller | `evaluation_assistant/app.py`, `index.html` | Local HTTP UI, constrained LLM routing and bounded tool calls |
| Application checks | `evaluation_assistant/check_review.py` | Offline checks and opt-in live routing acceptance cases |

## Model path

1. Use the fixed session manifest; keep every speaker in a single partition and development fold.
2. Read sustained normal-pitch /a/, /i/ and /u/ NSP files and resample to 16 kHz.
3. Use the frozen convolutional feature extractor from `facebook/wav2vec2-large-xlsr-53`; average over time to produce 512 features per vowel. This is not the transformer's final contextual layer.
4. Fit preprocessing only on each training fold. Compare /a/, concatenation and learned attention under matched settings.
5. Select the representation/loss configuration on development evidence; freeze thresholds before test scoring. The selected representation is concatenation, not attention.
6. Evaluate the fixed ensemble and source-based reference on matching holdout sessions; resample speakers for uncertainty intervals.

## Evidence application path

```text
Browser question
    -> local controller
    -> supported-scope check / one constrained LLM routing decision
    -> allowlisted evidence tool (+ cohort context when needed)
    -> deterministic values and source locators
    -> browser evidence cards
```

Only the seven aggregate sources declared in `evaluation_assistant/evidence.py` are exposed by the evidence layer. The LLM has no general filesystem, shell, audio, training or threshold-selection capability. Guided mode bypasses the LLM and is labeled separately.

## Preservation

This cleanup adds repository navigation and publication configuration. It does not merge different experiment stages, move recorded artifacts, rename original scripts or recompute results. `results/` and `milestone2/` intentionally preserve experiment history rather than pretending the final system was selected before all earlier development work.

# Reproduction and environment

## Review without retraining

The repository includes aggregate results and the evidence assistant's seven input tables. From the repository root:

```powershell
python evaluation_assistant/check_review.py
python evaluation_assistant/app.py
```

The assistant uses the Python standard library. Guided mode needs no pip packages or model download. Free-form mode requires the local Ollama model described in its [README](../evaluation_assistant/README.md). For live checks, use a new report name so prior evidence is retained:

```powershell
python scripts/project.py review-live --execute -- --report evaluation_assistant/live_results_new.json
```

## Scientific environment

Python 3.12 was used locally. Direct dependency versions captured during repository preparation are in `requirements/experiments.txt`; `requirements/data-download.txt` additionally includes the historical downloader. These are observed environment versions, not a claim that a fresh install was tested or that every historical run used an identical environment. Check each experiment's `run_config.json` or `protocol.json` when comparing results. In particular, the historical baseline records pandas 3.0.3 while the cleanup environment reports pandas 2.3.3.

For a separate environment, if reproducing experiments:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Do not replace a working experiment environment just to review saved results. The upstream `milestone2/reference_sources/requirements.txt` is an upstream environment record, not this project's pip installation file.

## Restore local research inputs

A Git clone intentionally omits raw recordings, individual-session tables, feature caches and trained model binaries. Full reproduction requires authorized access to the original data and the frozen local inputs:

| Location | Required contents |
|---|---|
| `splits.csv` | Original session/speaker labels, partition and fold assignments |
| `svd_data positive/`, `svd_data negative/` | Original SVD directory trees, including session vowel NSP recordings |
| `embeddings/` | Milestone 1 /a/ cache and extraction metadata, or regenerate features |
| `milestone2/features/` | Development /a/, /i/, /u/ cache, or regenerate features |
| `milestone2/coverage.csv` | Original development coverage manifest, or regenerate coverage from audio |
| `milestone2/feedback_holdout_inputs/` | Holdout coverage, timing and feature cache |
| `milestone2/feedback_test/predictions.csv` | Frozen per-session scores required by the authors' matched runner |
| `milestone2/reference_sources/FeatureStates_SVD.csv` | Pinned upstream released features matching the runner's SHA-256 check |

Other excluded row-level outputs are needed to recreate the reports that consume them. The source file's checks remain authoritative for exact filenames and schemas.

The required split SHA-256 is:

```text
7404ad74654439cc01d476b4b27c49aef043c10f06d471f8c3756465a8cb0d57
```

Do not regenerate splits to make an old result run. `make_splits.py` is a historical utility that overwrites `splits.csv` and uses the former `svd_data/sbvoice.db` path. `download_all.py`, `download_test.py`, `test.py` and `str.py` also refer to the old download location. They are preserved for provenance, not included in the current command catalog.

## Commands and dependencies

```powershell
python scripts/project.py list
python scripts/project.py develop
```

The second command previews the original invocation; it does not train. To run a selected stage yourself, add `--execute`. Extra arguments follow `--`, for example:

```powershell
python scripts/project.py develop --execute -- --smoke
```

Scientific order for a separate reproduction workspace:

1. Restore the fixed split and authorized recordings; audit available audio.
2. Extract /a/ features and run the Milestone 1 baseline if recreating that stage.
3. Prepare development triplets, then run the main development ablations.
4. Freeze the selected settings and development thresholds before preparing/scoring holdout inputs.
5. Run the fixed holdout comparison once; preserve its predictions and protocol.
6. Audit the pinned source-based reference, complete its prescribed grid and save the paired comparison.
7. Read the resulting aggregate evidence through the assistant.

The exploratory comparison and long-training diagnostics are separate historical branches, not extra prerequisites for every run. Existing completed-run guards are deliberate. Use a separate workspace for reproduction; do not erase old result directories to bypass them. Run one instance of a stage at a time.

## What the cleanup verified

Repository preparation uses file inventories, hashes, static Python parsing and command-preview checks. It does not run extraction, fitting, holdout evaluation or live LLM inference. The local `.repo_audit/` records the before/after preservation evidence and upload inventory; it is excluded from Git.

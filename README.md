# Multi-vowel voice classification and evidence review

Research project by **Basmala Kamal and Abdelmoniem Saad** for Advanced Machine Learning.

We study whether frozen voice features from sustained **/a/, /i/ and /u/** recordings help distinguish a specified group of malignant, premalignant and other tumour-associated diagnoses from other pathological voice conditions in the Saarbruecken Voice Database (SVD). The negative class is **not healthy controls**. This is an experimental classification task, not a clinical diagnostic service.

## Start here

- [Architecture and module map](docs/ARCHITECTURE.md)
- [Environment and reproduction guide](docs/REPRODUCIBILITY.md)
- [Results and experiment history](docs/RESULTS.md)
- [Publication and local-data policy](docs/PUBLISHING.md)
- [Evaluation assistant](evaluation_assistant/README.md)
- [Author-reference protocol](milestone2/author_reference_protocol.md)
- [Compute accounting](milestone2/presentation_results/compute_accounting.md)
- [Final Milestone 2 slides](output/Milestone2_Revised.pptx) and [speaking script](output/Milestone2_Recording_Script.md)

## System

```text
Local SVD recordings + frozen speaker-separated split
    -> NSP audio reader / 16 kHz preprocessing
    -> frozen XLSR-53 convolutional encoder
    -> 512 features per vowel
    -> single-vowel / concatenation / learned-attention experiments
    -> small MLP classifier
    -> development selection + fixed holdout evaluation
    -> aggregate tables, uncertainty intervals and figures
    -> read-only evaluation assistant with cited evidence
```

The frozen selected system concatenates three vectors into 1,536 features, uses two hidden layers of 50 units, and trains with focal loss selected on development data. Learned attention and demographic inputs were evaluated as ablations. The assistant uses a local LLM to route questions to bounded evidence tools; it does not classify voice recordings or invent metric values.

## Repository layout

```text
docs/                   Architecture, reproduction, result map and publication guide
scripts/                Unified command catalog and launcher
requirements/           Recorded environment dependencies
*.py                    Preserved experiment entry points and legacy data utilities
evaluation_assistant/   Local web UI, controller, evidence tools and acceptance reports
results/                Milestone 1 metrics and figures
milestone2/             Milestone 2 protocols, metrics, figures and reference sources
output/                 Presentation decks and speaking script
```

Raw audio, embeddings, row-level records, model binaries and slide-build intermediates remain on the original machine but are excluded from Git. See the publication guide for the exact categories and how a collaborator restores research inputs. The original experiment scripts and result locations remain unchanged because recorded protocols contain file hashes and existing commands depend on these paths.

## Quick start: review existing evidence

Python 3.12 was used for the experiments. The evaluation assistant itself uses only the Python standard library.

```powershell
python scripts/project.py list
python evaluation_assistant/check_review.py
python evaluation_assistant/app.py
```

Open `http://127.0.0.1:8765`. Guided evidence lookup needs no LLM. Free-form routing additionally needs the local Ollama setup described in the assistant README.

The catalog prints commands by default. An explicit `--execute` runs a selected stage; there is no automatic full-pipeline run. Existing commands such as `python ms2_feedback_dev.py` still work.

## Main finding

Three-vowel concatenation improved development average precision over /a/ alone under matched SMOTE settings; learned attention did not improve on concatenation in that comparison. The selected focal-loss system did **not demonstrate a clear holdout improvement over the source-based authors' reference**: the paired average-precision interval includes zero. See [the result map](docs/RESULTS.md) for exact source files and interpretation limits.

## Attribution

The reference implementation is adapted through an in-domain runner around the [Paterson authors' repository at the recorded revision](https://github.com/mary-paterson/LaryngealCancerClassificationBenchmark/tree/7811f6d8cc8ac6dbe8535e9e2669febed4f51da4). Files under `milestone2/reference_sources/` retain their upstream provenance; they are not presented as our own implementation. Dataset and pretrained-model terms apply separately. No new blanket license is assigned to third-party code, recordings or derived records by this cleanup.

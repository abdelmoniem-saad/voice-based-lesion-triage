# Publishing this folder

All original files remain locally. `.gitignore` controls what Git includes; it does not delete or move any file.

## Included

- Original Python experiment scripts and the new command catalog.
- Evaluation assistant code, UI and saved acceptance reports.
- Aggregate experiment metrics, protocols, selections, confidence intervals and figures.
- Final and previous presentation decks and speaking scripts in `output/`.
- Reference source files, with attribution to their recorded upstream revision.
- Architecture, environment, reproduction and compute documentation.

## Local only

- Approximately 13 GB of original audio/database downloads.
- Embedding caches and trained model/scaler binaries.
- Session-level split, demographic, coverage and prediction tables, including released feature rows.
- Python caches, temporary files, environments, credentials and slide-build intermediates.
- The local preservation inventory in `.repo_audit/`.

Aggregate coverage tables, such as `milestone2/author_reference/coverage.csv`, remain included because the assistant needs them. The complete `.gitignore` is the definitive exclusion list.

## Upload through Git

Use Git from this folder so the exclusion rules apply. Browser drag-and-drop of the entire directory or a ZIP does **not** apply `.gitignore`.

If this folder is still outside a Git repository:

```powershell
git init
git add .
git status --short
git diff --cached --stat
```

Review the staged list, then commit and connect it to your chosen remote using your usual Git workflow. Repository preparation does not create a remote, commit or push for you. There should be no `svd_data` directories, NumPy arrays, session predictions, model binaries or `.deck_build` files in the staged list.

`.gitattributes` disables automatic line-ending conversion so protocol-verified scripts and CSV/JSON bytes retain their recorded hashes on Windows. Do not run blanket formatting or newline conversion across frozen experiment files.

Existing tracked files are not untracked by `.gitignore`. At cleanup time this folder was not a Git repository; if you later copy it into another repository, inspect that repository's tracked list as well.

## Collaborators

The published code and aggregate evidence support review and the assistant demo without raw data. Full experiment reproduction needs separately restored research inputs; see [the reproduction guide](REPRODUCIBILITY.md). Keep dataset/model licensing and upstream attribution separate from your own code. This cleanup does not grant redistribution rights to external assets.

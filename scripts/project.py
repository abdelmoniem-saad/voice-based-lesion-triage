"""List or explicitly launch preserved project stages; never import model code."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
# name: (group, original entry point, fixed arguments, purpose)
STAGES = {
    "audio-audit": ("features", "extract_embeddings.py", ["--audit-only"], "Audit local /a/ recordings"),
    "extract-a": ("features", "extract_embeddings.py", ["--allow-missing"], "Extract available /a/ features"),
    "triplet-audit": ("features", "ms2_prepare.py", ["--audit-only"], "Audit development triplets"),
    "prepare-dev": ("features", "ms2_prepare.py", [], "Prepare development triplet features"),
    "prepare-holdout": ("features", "ms2_prepare.py", ["--holdout"], "Prepare frozen holdout inputs"),
    "baseline": ("milestone1", "train_baseline.py", [], "Original sklearn architecture search"),
    "losses-long": ("milestone1", "compare_losses.py", [], "Original 1,000-update comparison"),
    "learning-curves": ("milestone1", "diagnose_training.py", [], "Training diagnostics"),
    "losses-short": ("milestone1", "compare_short_training.py", [], "25-update loss comparison"),
    "report-m1": ("milestone1", "report_results.py", [], "Original loss-comparison report"),
    "test-m1": ("milestone1", "evaluate_test.py", [], "Frozen Milestone 1 holdout evaluation"),
    "plot-m1": ("milestone1", "plot_final_test.py", [], "Milestone 1 holdout figures"),
    "explore-fusion": ("milestone2", "ms2_compare.py", [], "Exploratory mean/contrast comparison"),
    "report-fusion": ("milestone2", "ms2_report.py", [], "Exploratory comparison figures"),
    "develop": ("milestone2", "ms2_feedback_dev.py", [], "Fusion, loss and demographic ablations"),
    "test-m2": ("milestone2", "ms2_feedback_test.py", [], "Frozen Milestone 2 holdout evaluation"),
    "reference-audit": ("reference", "ms2_author_reference.py", [], "Audit pinned reference inputs"),
    "reference-fit": ("reference", "ms2_author_reference.py", ["--fit"], "Run/resume the bounded authors' grid"),
    "review-check": ("assistant", "evaluation_assistant/check_review.py", [], "Offline evidence/controller checks"),
    "review-live": ("assistant", "evaluation_assistant/check_review.py", ["--live"], "Opt-in local LLM checks"),
    "review-app": ("assistant", "evaluation_assistant/app.py", [], "Start the local evidence viewer"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", nargs="?", default="list", choices=["list", *STAGES])
    parser.add_argument("--execute", action="store_true", help="Actually run the selected stage")
    argv = sys.argv[1:]
    if "--" in argv:
        boundary = argv.index("--")
        options, forwarded = argv[:boundary], argv[boundary + 1:]
    else:
        options, forwarded = argv, []
    args = parser.parse_args(options)
    if args.stage == "list":
        if args.execute or forwarded:
            parser.error("Select a stage before using --execute or forwarded arguments.")
        for name, (group, path, fixed, purpose) in STAGES.items():
            print(f"{group:12} {name:18} {purpose}")
        print("\nPreview: python scripts/project.py develop")
        print("Run explicitly: python scripts/project.py review-check --execute")
        print("Forward flags after --. Existing experiment output guards still apply.")
        return 0
    _, path, fixed, purpose = STAGES[args.stage]
    script = ROOT / path
    if not script.is_file():
        parser.error(f"Missing entry point: {script}")
    command = [sys.executable, str(script), *fixed, *forwarded]
    print(purpose, flush=True)
    print("Working directory:", ROOT, flush=True)
    print("Command:", subprocess.list2cmdline(command), flush=True)
    if not args.execute:
        print("Preview only. Add --execute before -- to launch.")
        return 0
    # Argument array, shell=False: no shell interpolation; preserve original cwd semantics.
    return subprocess.call(command, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())

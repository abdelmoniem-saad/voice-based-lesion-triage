"""Repeat the existing comparison with a shared shorter training duration."""

import sys
import compare_losses as experiment

experiment.STEPS = 25

# Save separately; preserve all previous experiments.
sys.argv = [
    sys.argv[0],
    "--output",
    "results/loss_comparison_25steps",
]

if __name__ == "__main__":
    experiment.main()
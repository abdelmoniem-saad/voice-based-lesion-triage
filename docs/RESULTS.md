# Results and evidence map

Read development and holdout findings separately. Average precision (AP) is not classification accuracy. A numerical difference alone is not a statistically established improvement.

| Evidence directory | What it represents |
|---|---|
| `results/baseline_cv/` | Original scikit-learn architecture search |
| `results/loss_comparison/` | Initial 1,000-update loss comparison |
| `results/learning_curves/` | Training/validation diagnostics showing overfitting |
| `results/loss_comparison_25steps/` | Short-training Milestone 1 loss comparison |
| `results/final_test/` | Milestone 1 holdout evaluation; distinct from the later matched experiment |
| `milestone2/comparison/` | Exploratory mean/contrast and individual-vowel development experiments |
| `milestone2/feedback_dev/` | Main fusion, loss and demographic development ablations |
| `milestone2/feedback_test/` | Fixed selected configuration and matched ablation holdout evaluation |
| `milestone2/author_reference/` | Pinned source-based reference and primary paired comparison |
| `milestone2/presentation_results/` | Presentation figures, compute accounting and earlier speaking notes |
| `evaluation_assistant/live_results*.json` | Recorded routing acceptance runs, including the earlier failed iteration |

## Main development results

Source: [`feedback_dev/summary.csv`](../milestone2/feedback_dev/summary.csv). Rounded mean fold AP:

| Comparison | AP |
|---|---:|
| /a/ only + SMOTE | 0.157 |
| Concatenation + SMOTE | 0.207 |
| Learned attention + SMOTE | 0.206 |
| Concatenation + selected focal loss | 0.233 |
| Voice plus demographics, selected focal loss | 0.219 |
| Demographics only, selected focal loss | 0.106 |

The focal settings were alpha 0.9 and gamma 2, with 25 optimizer updates. Development selection is not an unbiased final performance estimate. Demographic ablations do not measure performance separately for women or men.

## Matched holdout

Sources: [`author_reference/results.csv`](../milestone2/author_reference/results.csv), [`paired.json`](../milestone2/author_reference/paired.json) and [`feedback_test/results.csv`](../milestone2/feedback_test/results.csv).

The frozen focal model has AP approximately 0.072; the source-based authors' reference has AP approximately 0.084. The paired difference is approximately -0.012, with a 95% interval approximately [-0.141, 0.110]. This does not establish an improvement or equivalence. The comparison retrains the reference on SVD under our protocol; it does not reproduce a published FEMH-to-SVD transfer score.

There are 252 eligible holdout sessions, including seven positive sessions from six positive speakers. At the development-selected sensitivity-target threshold, the frozen focal system detects five of those seven positive sessions. Do not describe the development 80% target as achieved on holdout.

The holdout had already been evaluated in Milestone 1. It is neither new external validation nor evidence of clinical readiness. Metrics are session-level; uncertainty resampling groups sessions by speaker. The stored inverse-session weight column is not used to make every reported metric speaker-weighted.

## Evaluation assistant

`live_results.json` records the original 3/10 run; `live_results_v2.json` records 10/10 acceptance cases after the controller revision. The latter includes eight actual LLM-routed cases and two deterministic scope checks. It is a small engineering acceptance suite, not a claim of 100% general LLM accuracy.

## Presentation artifacts

Use `output/Milestone2_Revised.pptx` for the revised deck and `output/Milestone2_Recording_Script.md` for its script. Earlier decks and notes are intentionally retained. Compute accounting distinguishes measured operation time from timestamp-based estimates; it is not a clean end-to-end runtime benchmark.

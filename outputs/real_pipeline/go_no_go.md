# LightScorer Real Data Go/No-Go

- decision: **GO**
- test_auc: 0.7581
- test_pr_auc: 0.7802
- recall_unconstrained: 1.0000
- recall_target: 0.90
- working_threshold: 0.4500
- working_precision: 0.6249
- working_recall: 0.9205
- working_keep_ratio: 0.7785
- working_reject_ratio: 0.2215
- working_good_reject_ratio: 0.0795
- working_bad_reject_ratio: 0.3807
- working_speedup: 1.2840
- working_hours_saved: 11.0598
- theoretical_max_speedup_on_curve: 10.27x
- theoretical_max_threshold: 0.9500
- manifest_samples: 20000
- manifest_positive_ratio: 0.6043

## Notes
- Main decision is based on constrained working point (Recall target).
- Theoretical max speedup is reference only and may be non-deployable.
- Recommend repeating with multiple random seeds and a full-size run.
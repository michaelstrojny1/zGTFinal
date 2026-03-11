# Full Fair Matrix Report

- Timestamp: 2026-03-06T17:36:47
- Output root: `C:\Users\micha\zGTFinal\outputs\publication_phd_final`
- Wall seconds: 140.4
- Baseline calibration source: `dev_holdout`
- Suspicious findings: 0

## Baseline
- FD001: rmse=18.830, mae=14.743, rul_cov=0.965, tau_v=0.090

## Evidence Deltas (marginal-fixed)
- FD001: delta_rul_cov=-0.112, delta_tau_v=-0.090

## Deep Checks
- deep_check_results findings: 0
- deep_check_results (all artifacts) findings: 21
- deep_check_regimes findings: 0
- deep_check_results (all) expected stress findings: 21
- deep_check_results (all) unexpected findings: 0

## Notes
- FD001 marginal_rul coverage is lower than fixed_tau by -0.112; this ablation is expected to be more conservative because it accumulates evidence over static RUL hypotheses.

## Suspicious
- None

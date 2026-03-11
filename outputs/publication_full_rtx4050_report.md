# Full Fair Matrix Report

- Timestamp: 2026-03-09T00:55:50
- Output root: `C:\Users\micha\zGTFinal\outputs\publication_full_rtx4050`
- Wall seconds: 1293.8
- Baseline calibration source: `dev_holdout`
- Suspicious findings: 0

## Baseline
- FD001: rmse=18.835, mae=14.742, rul_cov=0.976, tau_v=0.079
- FD002: rmse=22.684, mae=17.519, rul_cov=0.985, tau_v=0.069
- FD003: rmse=23.539, mae=16.633, rul_cov=0.992, tau_v=0.035
- FD004: rmse=24.779, mae=18.963, rul_cov=0.999, tau_v=0.006

## Evidence Deltas (marginal-fixed)
- FD001: delta_rul_cov=-0.105, delta_tau_v=-0.079
- FD002: delta_rul_cov=-0.078, delta_tau_v=-0.069
- FD003: delta_rul_cov=-0.097, delta_tau_v=-0.035
- FD004: delta_rul_cov=-0.033, delta_tau_v=-0.006

## Deep Checks
- deep_check_results findings: 0
- deep_check_results (all artifacts) findings: 34
- deep_check_regimes findings: 0
- deep_check_results (all) expected stress findings: 34
- deep_check_results (all) unexpected findings: 0

## Notes
- FD001 marginal_rul coverage is lower than fixed_tau by -0.105; this ablation is expected to be more conservative because it accumulates evidence over static RUL hypotheses.
- FD002 marginal_rul coverage is lower than fixed_tau by -0.078; this ablation is expected to be more conservative because it accumulates evidence over static RUL hypotheses.
- FD003 marginal_rul coverage is lower than fixed_tau by -0.097; this ablation is expected to be more conservative because it accumulates evidence over static RUL hypotheses.
- FD004 marginal_rul coverage is lower than fixed_tau by -0.033; this ablation is expected to be more conservative because it accumulates evidence over static RUL hypotheses.

## Suspicious
- None

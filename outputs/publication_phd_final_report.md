# Full Fair Matrix Report

- Timestamp: 2026-03-07T01:15:44
- Output root: `C:\Users\micha\zGTFinal\outputs\publication_phd_final`
- Wall seconds: 30.5
- Baseline calibration source: `dev_holdout`
- Suspicious findings: 0

## Baseline
- FD001: rmse=18.827, mae=14.738, rul_cov=0.976, tau_v=0.079
- FD002: rmse=22.685, mae=17.520, rul_cov=0.985, tau_v=0.064
- FD003: rmse=23.516, mae=16.608, rul_cov=0.992, tau_v=0.035
- FD004: rmse=24.465, mae=18.915, rul_cov=1.000, tau_v=0.006

## Evidence Deltas (marginal-fixed)
- FD001: delta_rul_cov=-0.105, delta_tau_v=-0.079
- FD002: delta_rul_cov=-0.078, delta_tau_v=-0.064
- FD003: delta_rul_cov=-0.097, delta_tau_v=-0.035
- FD004: delta_rul_cov=-0.035, delta_tau_v=-0.006

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
- FD004 marginal_rul coverage is lower than fixed_tau by -0.035; this ablation is expected to be more conservative because it accumulates evidence over static RUL hypotheses.

## Suspicious
- None

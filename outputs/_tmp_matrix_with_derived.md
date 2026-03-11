# Full Fair Matrix Report

- Timestamp: 2026-03-09T14:28:51
- Output root: `C:\Users\micha\zGTFinal\outputs\publication_full_rtx4050`
- Wall seconds: 34.5
- Baseline calibration source: `dev_holdout`
- Suspicious findings: 0

## Baseline
- FD001: rmse=18.835, mae=14.742, rul_cov=0.976, tau_v=0.079

## Evidence Deltas (marginal-fixed)
- FD001: delta_rul_cov=-0.105, delta_tau_v=-0.079

## Deep Checks
- deep_check_results findings: 0
- deep_check_results (all artifacts) findings: 22
- deep_check_regimes findings: 0
- deep_check_results (all) expected stress findings: 22
- deep_check_results (all) unexpected findings: 0

## Notes
- FD001 marginal_rul coverage is lower than fixed_tau by -0.105; this ablation is expected to be more conservative because it accumulates evidence over static RUL hypotheses.

## Suspicious
- None

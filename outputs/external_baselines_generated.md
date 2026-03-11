# External Baselines (Generated)

- Matrix report: `C:\Users\micha\zGTFinal\outputs\publication_full_rtx4050_report.json`
- Alpha: 0.05

## split_conformal_global_a0p05
- Comparator type: external
- Description: External baseline: classical split-conformal absolute-residual interval with global quantile (alpha=0.0500).
- FD001: rmse=18.835, rul_cov=0.949, tau_v=0.348
- FD002: rmse=22.684, rul_cov=0.961, tau_v=0.243
- FD003: rmse=23.539, rul_cov=0.917, tau_v=0.353
- FD004: rmse=24.779, rul_cov=0.960, tau_v=0.294

## split_conformal_global_a0p01
- Comparator type: external
- Description: External baseline: conservative global split-conformal absolute-residual interval (alpha=0.0100).
- FD001: rmse=18.835, rul_cov=0.977, tau_v=0.191
- FD002: rmse=22.684, rul_cov=0.994, tau_v=0.045
- FD003: rmse=23.539, rul_cov=0.968, tau_v=0.176
- FD004: rmse=24.779, rul_cov=0.994, tau_v=0.053

## split_conformal_conditional_a0p05
- Comparator type: external
- Description: External baseline: conditional (Mondrian-style) split-conformal intervals by RUL bin, with predicted-RUL bin assignment (alpha=0.0500).
- FD001: rmse=18.835, rul_cov=0.852, tau_v=0.685
- FD002: rmse=22.684, rul_cov=0.932, tau_v=0.317
- FD003: rmse=23.539, rul_cov=0.877, tau_v=0.506
- FD004: rmse=24.779, rul_cov=0.961, tau_v=0.332

## split_conformal_conditional_a0p01
- Comparator type: external
- Description: External baseline: conservative conditional (Mondrian-style) split-conformal intervals by RUL bin, with predicted-RUL bin assignment (alpha=0.0100).
- FD001: rmse=18.835, rul_cov=0.935, tau_v=0.461
- FD002: rmse=22.684, rul_cov=0.988, tau_v=0.079
- FD003: rmse=23.539, rul_cov=0.921, tau_v=0.353
- FD004: rmse=24.779, rul_cov=0.992, tau_v=0.102

## Notes
- These baselines are generated from strict_main prediction/caching artifacts and use non-sequential split-conformal intervals.
- Includes global and conditional (RUL-bin Mondrian) split-conformal ladders at two alpha levels.
- run_dir is intentionally empty because these baselines do not emit full TEM per-run artifacts.


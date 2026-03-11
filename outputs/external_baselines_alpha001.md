# External Baselines (Generated)

- Matrix report: `C:\Users\micha\zGTFinal\outputs\publication_full_rtx4050_report.json`
- Alpha: 0.01

## split_conformal_global
- Comparator type: external
- Description: External baseline: classical split-conformal absolute-residual interval with global quantile.
- FD001: rmse=18.835, rul_cov=0.977, tau_v=0.191
- FD002: rmse=22.684, rul_cov=0.994, tau_v=0.045
- FD003: rmse=23.539, rul_cov=0.968, tau_v=0.176
- FD004: rmse=24.779, rul_cov=0.991, tau_v=0.099

## split_conformal_binned_pred
- Comparator type: external
- Description: External baseline: split-conformal absolute-residual interval with predicted-RUL binwise quantiles.
- FD001: rmse=18.835, rul_cov=0.954, tau_v=0.337
- FD002: rmse=22.684, rul_cov=0.987, tau_v=0.104
- FD003: rmse=23.539, rul_cov=0.915, tau_v=0.365
- FD004: rmse=24.779, rul_cov=0.988, tau_v=0.155

## Notes
- These baselines are generated from strict_main prediction/caching artifacts and use non-sequential split-conformal intervals.
- run_dir is intentionally empty because these baselines do not emit full TEM per-run artifacts.


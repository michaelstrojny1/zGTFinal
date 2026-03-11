# External Generalization Summary

- Source matrix report: `C:\Users\micha\zGTFinal\outputs\publication_full_rtx4050_report.json`
- Benchmark type: `synthetic_shift_grid`
- Grid points: 9

## Aggregate
- null_alert_rate mean=0.0000, max=0.0000
- degraded_alert_rate mean=0.0639, max=0.1633
- oracle_superuniform_violation_prob mean=0.0000, max=0.0000

## Worst-Case Grid Points
- sigma=8.0, rho=0.5: null_alert=0.0000, degraded_alert=0.0000, oracle_violation=0.0000
- sigma=8.0, rho=1.5: null_alert=0.0000, degraded_alert=0.0007, oracle_violation=0.0000
- sigma=8.0, rho=1.0: null_alert=0.0000, degraded_alert=0.0207, oracle_violation=0.0000

## Real RUL Datasets
- Source: `C:\Users\micha\zGTFinal\outputs\rul_dataset_summary.json`
- Successful datasets: 3 / 3
- cmapss: status=ok, dev_runs=80, test_runs=100
- femto: status=ok, dev_runs=2, test_runs=4
- xjtu_sy: status=ok, dev_runs=2, test_runs=2

## Real External Model Performance
- Source: `C:\Users\micha\zGTFinal\outputs\external_performance_report.json`
- Successful dataset evaluations: 3 / 3
- femto: status=ok, runs=4, rul_cov=1.0, tau_v=0.0, rmse=14.362340764708255, rmse_last=52.873386800339176, mae_last=50.39453125
- xjtu_sy: status=ok, runs=2, rul_cov=1.0, tau_v=0.0, rmse=30.132575039658928, rmse_last=33.171326685294034, mae_last=33.15625
- cmapss: status=ok, runs=100, rul_cov=1.0, tau_v=0.0, rmse=16.395882779406683, rmse_last=16.62348723677968, mae_last=12.2186328125

## Caveat
- Includes synthetic shift stress and real external model-evaluation metrics.


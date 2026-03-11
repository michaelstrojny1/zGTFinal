# PhD Deep Inspect Report

## Runs
- `C:\Users\micha\zGTFinal\outputs\phd_ultra_best\fd001_fulltopo_best`
  - RMSE=17.500, MAE=13.353, calibration_floor=1.0
  - alert_rate=0.010, rul_cov=0.979, tau_cov=0.979, tau_violation=0.056, tau_diag_engines=89/100
  - topology mean_gamma=29.535, local_minima=2.488, persistent_valleys=0.012, ridge_tv=118.510
  - corr(gamma,rul_cov)=-0.3623137080262126, corr(ridge_tv,alert)=0.23994359724884362, corr(minima,alert)=0.17243570774864286
- `C:\Users\micha\zGTFinal\outputs\phd_ultra_best\fd002_fulltopo_best`
  - RMSE=23.244, MAE=18.354, calibration_floor=1.0
  - alert_rate=0.000, rul_cov=0.964, tau_cov=0.953, tau_violation=0.134, tau_diag_engines=202/259
  - topology mean_gamma=17.064, local_minima=3.941, persistent_valleys=0.015, ridge_tv=118.320
  - corr(gamma,rul_cov)=-0.21086139210516824, corr(ridge_tv,alert)=None, corr(minima,alert)=None
- `C:\Users\micha\zGTFinal\outputs\phd_ultra_best\fd003_fulltopo_best`
  - RMSE=22.282, MAE=16.052, calibration_floor=1.0
  - alert_rate=0.010, rul_cov=0.982, tau_cov=0.981, tau_violation=0.071, tau_diag_engines=85/100
  - topology mean_gamma=15.760, local_minima=2.175, persistent_valleys=0.019, ridge_tv=137.700
  - corr(gamma,rul_cov)=-0.02838372978204038, corr(ridge_tv,alert)=0.35949481605236455, corr(minima,alert)=0.1472169393876783
- `C:\Users\micha\zGTFinal\outputs\phd_ultra_best\fd004_fulltopo_best`
  - RMSE=24.778, MAE=18.908, calibration_floor=1.0
  - alert_rate=0.012, rul_cov=0.968, tau_cov=0.957, tau_violation=0.122, tau_diag_engines=181/248
  - topology mean_gamma=38.690, local_minima=3.445, persistent_valleys=0.020, ridge_tv=180.964
  - corr(gamma,rul_cov)=-0.022265241660252746, corr(ridge_tv,alert)=0.23458981799754405, corr(minima,alert)=-0.06974279305961514

## Notes
- `num_tau_diagnostics_engines < num_engines` indicates runs where true tau is not identifiable from capped/censored labels.
- Topology correlations are descriptive, not causal.

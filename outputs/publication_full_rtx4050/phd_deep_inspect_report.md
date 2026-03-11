# PhD Deep Inspect Report

## Runs
- `C:\Users\micha\zGTFinal\outputs\publication_full_rtx4050\strict_main\fd001`
  - RMSE=18.835, MAE=14.742, calibration_floor=1.0
  - alert_rate=0.010, rul_cov=0.976, tau_cov=0.977, tau_violation=0.079, tau_diag_engines=89/100
  - topology mean_gamma=52.319, local_minima=5.794, persistent_valleys=0.011, ridge_tv=112.630
  - corr(gamma,rul_cov)=-0.42792259456547665, corr(ridge_tv,alert)=0.28104826152362616, corr(minima,alert)=-0.07869799148078381
- `C:\Users\micha\zGTFinal\outputs\publication_full_rtx4050\strict_main\fd002`
  - RMSE=22.684, MAE=17.519, calibration_floor=1.0
  - alert_rate=0.000, rul_cov=0.985, tau_cov=0.981, tau_violation=0.069, tau_diag_engines=202/259
  - topology mean_gamma=71.086, local_minima=3.791, persistent_valleys=0.004, ridge_tv=104.208
  - corr(gamma,rul_cov)=0.02410568894364296, corr(ridge_tv,alert)=None, corr(minima,alert)=None
- `C:\Users\micha\zGTFinal\outputs\publication_full_rtx4050\strict_main\fd003`
  - RMSE=23.539, MAE=16.633, calibration_floor=1.0
  - alert_rate=0.000, rul_cov=0.992, tau_cov=0.991, tau_violation=0.035, tau_diag_engines=85/100
  - topology mean_gamma=40.531, local_minima=3.749, persistent_valleys=0.003, ridge_tv=120.890
  - corr(gamma,rul_cov)=-0.2596637465442933, corr(ridge_tv,alert)=None, corr(minima,alert)=None
- `C:\Users\micha\zGTFinal\outputs\publication_full_rtx4050\strict_main\fd004`
  - RMSE=24.779, MAE=18.963, calibration_floor=1.0
  - alert_rate=0.000, rul_cov=0.999, tau_cov=0.999, tau_violation=0.006, tau_diag_engines=181/248
  - topology mean_gamma=27.398, local_minima=3.771, persistent_valleys=0.007, ridge_tv=168.770
  - corr(gamma,rul_cov)=0.006976176223072635, corr(ridge_tv,alert)=None, corr(minima,alert)=None

## Notes
- `num_tau_diagnostics_engines < num_engines` indicates runs where true tau is not identifiable from capped/censored labels.
- Topology correlations are descriptive, not causal.

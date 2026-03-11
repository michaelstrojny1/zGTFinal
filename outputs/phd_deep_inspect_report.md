# PhD Deep Inspect Report

## Runs
- `C:\Users\micha\zGTFinal\outputs\fd001_phd_final`
  - RMSE=17.918, MAE=13.510, calibration_floor=1.0
  - alert_rate=0.000, rul_cov=0.976, tau_cov=0.976, tau_violation=0.079, tau_diag_engines=89/100
  - topology mean_gamma=76.103, local_minima=6.047, persistent_valleys=0.032, ridge_tv=128.870
  - corr(gamma,rul_cov)=-0.19310898511229327, corr(ridge_tv,alert)=None, corr(minima,alert)=None
- `C:\Users\micha\zGTFinal\outputs\fd002_phd_final`
  - RMSE=22.408, MAE=17.572, calibration_floor=1.0
  - alert_rate=0.008, rul_cov=0.940, tau_cov=0.923, tau_violation=0.218, tau_diag_engines=202/259
  - topology mean_gamma=3600.897, local_minima=3.622, persistent_valleys=0.017, ridge_tv=126.633
  - corr(gamma,rul_cov)=0.048470867387436584, corr(ridge_tv,alert)=0.061721796629029164, corr(minima,alert)=-0.06119464516710924

## Notes
- `num_tau_diagnostics_engines < num_engines` indicates runs where true tau is not identifiable from capped/censored labels.
- Topology correlations are descriptive, not causal.

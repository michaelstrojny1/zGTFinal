# PhD Deep Inspect Report

## Runs
- `C:\Users\micha\zGTFinal\outputs\_default2_fd001_fulltopo`
  - RMSE=83.781, MAE=73.652, calibration_floor=1.0
  - alert_rate=0.000, rul_cov=0.959, tau_cov=0.959, tau_violation=0.112, tau_diag_engines=89/100
  - topology mean_gamma=2.178, local_minima=6.027, persistent_valleys=1.909, ridge_tv=175.710
  - corr(gamma,rul_cov)=-0.0870576058833074, corr(ridge_tv,alert)=None, corr(minima,alert)=None
- `C:\Users\micha\zGTFinal\outputs\_default2_fd002_fulltopo`
  - RMSE=22.593, MAE=17.339, calibration_floor=1.0
  - alert_rate=0.000, rul_cov=0.938, tau_cov=0.921, tau_violation=0.203, tau_diag_engines=202/259
  - topology mean_gamma=29.845, local_minima=3.052, persistent_valleys=0.007, ridge_tv=102.934
  - corr(gamma,rul_cov)=-0.0035800329966216016, corr(ridge_tv,alert)=None, corr(minima,alert)=None

## Notes
- `num_tau_diagnostics_engines < num_engines` indicates runs where true tau is not identifiable from capped/censored labels.
- Topology correlations are descriptive, not causal.

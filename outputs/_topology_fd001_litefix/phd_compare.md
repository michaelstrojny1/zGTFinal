# PhD Deep Inspect Report

## Runs
- `C:\Users\micha\zGTFinal\outputs\_topology_fd001_litefix`
  - RMSE=84.089, MAE=73.957, calibration_floor=1.0
  - alert_rate=0.000, rul_cov=0.974, tau_cov=0.972, tau_violation=0.067, tau_diag_engines=89/100
  - topology mean_gamma=1.705, local_minima=6.872, persistent_valleys=0.965, ridge_tv=146.550
  - corr(gamma,rul_cov)=-0.13199926310629434, corr(ridge_tv,alert)=None, corr(minima,alert)=None
- `C:\Users\micha\zGTFinal\outputs\_topology_fd001_full`
  - RMSE=84.089, MAE=73.957, calibration_floor=1.0
  - alert_rate=0.000, rul_cov=0.974, tau_cov=0.972, tau_violation=0.067, tau_diag_engines=89/100
  - topology mean_gamma=1.705, local_minima=6.872, persistent_valleys=1.848, ridge_tv=146.550
  - corr(gamma,rul_cov)=-0.13199926310629434, corr(ridge_tv,alert)=None, corr(minima,alert)=None

## Notes
- `num_tau_diagnostics_engines < num_engines` indicates runs where true tau is not identifiable from capped/censored labels.
- Topology correlations are descriptive, not causal.

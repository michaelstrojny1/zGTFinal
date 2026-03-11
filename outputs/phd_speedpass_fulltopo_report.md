# PhD Deep Inspect Report

## Runs
- `C:\Users\micha\zGTFinal\outputs\_pub_fd001_valcal_fulltopo`
  - RMSE=56.080, MAE=47.806, calibration_floor=1.0
  - alert_rate=0.000, rul_cov=0.957, tau_cov=0.955, tau_violation=0.101, tau_diag_engines=89/100
  - topology mean_gamma=35.803, local_minima=3.794, persistent_valleys=0.000, ridge_tv=46.270
  - corr(gamma,rul_cov)=0.09148958639993696, corr(ridge_tv,alert)=None, corr(minima,alert)=None
- `C:\Users\micha\zGTFinal\outputs\_pub_fd002_valcal_fulltopo`
  - RMSE=22.585, MAE=17.340, calibration_floor=1.0
  - alert_rate=0.000, rul_cov=0.931, tau_cov=0.912, tau_violation=0.228, tau_diag_engines=202/259
  - topology mean_gamma=43.747, local_minima=3.128, persistent_valleys=0.007, ridge_tv=105.614
  - corr(gamma,rul_cov)=-0.04030770685051947, corr(ridge_tv,alert)=None, corr(minima,alert)=None

## Notes
- `num_tau_diagnostics_engines < num_engines` indicates runs where true tau is not identifiable from capped/censored labels.
- Topology correlations are descriptive, not causal.

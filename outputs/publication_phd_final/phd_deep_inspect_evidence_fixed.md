# PhD Deep Inspect Report

## Runs
- `C:\Users\micha\zGTFinal\outputs\publication_phd_final\evidence_mode\fixed_tau\fd001`
  - RMSE=18.830, MAE=14.743, calibration_floor=1.0
  - alert_rate=0.010, rul_cov=0.965, tau_cov=0.966, tau_violation=0.090, tau_diag_engines=89/100
  - topology mean_gamma=104.623, local_minima=5.558, persistent_valleys=0.015, ridge_tv=112.920
  - corr(gamma,rul_cov)=-0.6044033891097116, corr(ridge_tv,alert)=0.2816079522880843, corr(minima,alert)=-0.07920039877204683
- `C:\Users\micha\zGTFinal\outputs\publication_phd_final\evidence_mode\fixed_tau\fd002`
  - RMSE=22.621, MAE=17.499, calibration_floor=1.0
  - alert_rate=0.000, rul_cov=0.980, tau_cov=0.974, tau_violation=0.074, tau_diag_engines=202/259
  - topology mean_gamma=95.006, local_minima=3.807, persistent_valleys=0.005, ridge_tv=106.046
  - corr(gamma,rul_cov)=0.02523931840536323, corr(ridge_tv,alert)=None, corr(minima,alert)=None
- `C:\Users\micha\zGTFinal\outputs\publication_phd_final\evidence_mode\fixed_tau\fd003`
  - RMSE=23.515, MAE=16.617, calibration_floor=1.0
  - alert_rate=0.000, rul_cov=0.989, tau_cov=0.987, tau_violation=0.035, tau_diag_engines=85/100
  - topology mean_gamma=35.779, local_minima=4.244, persistent_valleys=0.003, ridge_tv=120.620
  - corr(gamma,rul_cov)=-0.12826602334393397, corr(ridge_tv,alert)=None, corr(minima,alert)=None
- `C:\Users\micha\zGTFinal\outputs\publication_phd_final\evidence_mode\fixed_tau\fd004`
  - RMSE=24.653, MAE=19.243, calibration_floor=1.0
  - alert_rate=0.000, rul_cov=0.999, tau_cov=0.999, tau_violation=0.011, tau_diag_engines=181/248
  - topology mean_gamma=17.874, local_minima=3.662, persistent_valleys=0.005, ridge_tv=166.875
  - corr(gamma,rul_cov)=-2.324126884044093e-05, corr(ridge_tv,alert)=None, corr(minima,alert)=None

## Notes
- `num_tau_diagnostics_engines < num_engines` indicates runs where true tau is not identifiable from capped/censored labels.
- Topology correlations are descriptive, not causal.

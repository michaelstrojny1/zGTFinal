# PhD Deep Inspect Report

## Runs
- `C:\Users\micha\zGTFinal\outputs\publication_phd_final\evidence_mode\marginal_rul\fd001`
  - RMSE=18.830, MAE=14.743, calibration_floor=1.0
  - alert_rate=0.040, rul_cov=0.853, tau_cov=0.000, tau_violation=0.000, tau_diag_engines=0/100
  - topology mean_gamma=3.759, local_minima=6.013, persistent_valleys=0.547, ridge_tv=18.230
  - corr(gamma,rul_cov)=-0.22410487991315392, corr(ridge_tv,alert)=-0.2133340965701629, corr(minima,alert)=-0.1032662908095274
- `C:\Users\micha\zGTFinal\outputs\publication_phd_final\evidence_mode\marginal_rul\fd002`
  - RMSE=22.621, MAE=17.499, calibration_floor=1.0
  - alert_rate=0.000, rul_cov=0.891, tau_cov=0.000, tau_violation=0.000, tau_diag_engines=0/259
  - topology mean_gamma=5.365, local_minima=7.351, persistent_valleys=0.727, ridge_tv=13.066
  - corr(gamma,rul_cov)=0.04257922522291274, corr(ridge_tv,alert)=None, corr(minima,alert)=None
- `C:\Users\micha\zGTFinal\outputs\publication_phd_final\evidence_mode\marginal_rul\fd003`
  - RMSE=23.515, MAE=16.617, calibration_floor=1.0
  - alert_rate=0.290, rul_cov=0.879, tau_cov=0.000, tau_violation=0.000, tau_diag_engines=0/100
  - topology mean_gamma=91.481, local_minima=5.061, persistent_valleys=0.926, ridge_tv=4.220
  - corr(gamma,rul_cov)=0.026192531350723122, corr(ridge_tv,alert)=-0.17499281712992157, corr(minima,alert)=0.02208311613370728
- `C:\Users\micha\zGTFinal\outputs\publication_phd_final\evidence_mode\marginal_rul\fd004`
  - RMSE=24.653, MAE=19.243, calibration_floor=1.0
  - alert_rate=0.000, rul_cov=0.955, tau_cov=0.000, tau_violation=0.000, tau_diag_engines=0/248
  - topology mean_gamma=5.984, local_minima=5.964, persistent_valleys=1.053, ridge_tv=15.976
  - corr(gamma,rul_cov)=0.04440760197981412, corr(ridge_tv,alert)=None, corr(minima,alert)=None

## Notes
- `num_tau_diagnostics_engines < num_engines` indicates runs where true tau is not identifiable from capped/censored labels.
- Topology correlations are descriptive, not causal.

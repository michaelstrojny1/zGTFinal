# PhD Deep Inspect Report

## Runs
- `C:\Users\micha\zGTFinal\outputs\strict_pipeline_all_e15_phd\fd001`
  - RMSE=17.513, MAE=13.377, calibration_floor=1.0
  - alert_rate=0.010, rul_cov=0.972, tau_cov=0.972, tau_violation=0.079, tau_diag_engines=89/100
  - topology mean_gamma=29.571, local_minima=2.610, persistent_valleys=0.012, ridge_tv=126.350
  - corr(gamma,rul_cov)=-0.08423246177556656, corr(ridge_tv,alert)=0.25622237622620675, corr(minima,alert)=0.23078969480963843
- `C:\Users\micha\zGTFinal\outputs\strict_pipeline_all_e15_phd\fd002`
  - RMSE=23.239, MAE=18.346, calibration_floor=1.0
  - alert_rate=0.000, rul_cov=0.972, tau_cov=0.964, tau_violation=0.114, tau_diag_engines=202/259
  - topology mean_gamma=29.781, local_minima=4.939, persistent_valleys=0.016, ridge_tv=126.506
  - corr(gamma,rul_cov)=-0.06670981056814672, corr(ridge_tv,alert)=None, corr(minima,alert)=None
- `C:\Users\micha\zGTFinal\outputs\strict_pipeline_all_e15_phd\fd003`
  - RMSE=22.289, MAE=16.055, calibration_floor=1.0
  - alert_rate=0.000, rul_cov=0.991, tau_cov=0.990, tau_violation=0.035, tau_diag_engines=85/100
  - topology mean_gamma=125.188, local_minima=2.873, persistent_valleys=0.018, ridge_tv=128.220
  - corr(gamma,rul_cov)=0.03946599702551109, corr(ridge_tv,alert)=None, corr(minima,alert)=None
- `C:\Users\micha\zGTFinal\outputs\strict_pipeline_all_e15_phd\fd004`
  - RMSE=24.966, MAE=18.857, calibration_floor=1.0
  - alert_rate=0.000, rul_cov=0.998, tau_cov=0.997, tau_violation=0.011, tau_diag_engines=181/248
  - topology mean_gamma=32.410, local_minima=3.449, persistent_valleys=0.007, ridge_tv=166.032
  - corr(gamma,rul_cov)=0.022971331713135, corr(ridge_tv,alert)=None, corr(minima,alert)=None

## Notes
- `num_tau_diagnostics_engines < num_engines` indicates runs where true tau is not identifiable from capped/censored labels.
- Topology correlations are descriptive, not causal.

# PhD Deep Inspect Report

## Runs
- `C:\Users\micha\zGTFinal\outputs\final_local_fd001_fulltopo`
  - RMSE=17.500, MAE=13.353, calibration_floor=1.0
  - alert_rate=0.010, rul_cov=0.976, tau_cov=0.976, tau_violation=0.056, tau_diag_engines=89/100
  - topology mean_gamma=92.411, local_minima=4.193, persistent_valleys=0.012, ridge_tv=117.390
  - corr(gamma,rul_cov)=-0.22120746177169656, corr(ridge_tv,alert)=0.2857927029121668, corr(minima,alert)=-0.08111527769373114
- `C:\Users\micha\zGTFinal\outputs\final_local_fd002_fulltopo`
  - RMSE=23.244, MAE=18.354, calibration_floor=1.0
  - alert_rate=0.000, rul_cov=0.957, tau_cov=0.944, tau_violation=0.163, tau_diag_engines=202/259
  - topology mean_gamma=19.451, local_minima=3.967, persistent_valleys=0.015, ridge_tv=121.869
  - corr(gamma,rul_cov)=-0.10764238951643984, corr(ridge_tv,alert)=None, corr(minima,alert)=None
- `C:\Users\micha\zGTFinal\outputs\final_local_fd003_fulltopo`
  - RMSE=22.282, MAE=16.052, calibration_floor=1.0
  - alert_rate=0.010, rul_cov=0.968, tau_cov=0.965, tau_violation=0.141, tau_diag_engines=85/100
  - topology mean_gamma=14.704, local_minima=2.209, persistent_valleys=0.019, ridge_tv=142.080
  - corr(gamma,rul_cov)=0.00533156223972183, corr(ridge_tv,alert)=0.3539199363706558, corr(minima,alert)=0.13103439033763103
- `C:\Users\micha\zGTFinal\outputs\final_local_fd004_fulltopo`
  - RMSE=24.778, MAE=18.908, calibration_floor=1.0
  - alert_rate=0.016, rul_cov=0.959, tau_cov=0.946, tau_violation=0.166, tau_diag_engines=181/248
  - topology mean_gamma=40.728, local_minima=3.424, persistent_valleys=0.020, ridge_tv=182.972
  - corr(gamma,rul_cov)=0.0003023186366873687, corr(ridge_tv,alert)=0.25058370754475, corr(minima,alert)=-0.07016644697283482

## Notes
- `num_tau_diagnostics_engines < num_engines` indicates runs where true tau is not identifiable from capped/censored labels.
- Topology correlations are descriptive, not causal.

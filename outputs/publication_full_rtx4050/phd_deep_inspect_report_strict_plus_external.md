# PhD Deep Inspect Report

## Runs
- `C:\Users\micha\zGTFinal\outputs\publication_full_rtx4050\strict_main\fd001`
  - RMSE=18.835, MAE=14.742, best_val_rmse=19.977, calibration_floor=1.0
  - alert_rate=0.010, rul_cov=0.976, tau_cov=0.977, tau_violation=0.079, tau_diag_engines=89/100
  - topology mean_gamma=52.319, local_minima=5.794, persistent_valleys=0.011, ridge_tv=112.630
  - corr(gamma,rul_cov)=-0.42792259456547665, corr(ridge_tv,alert)=0.28104826152362616, corr(minima,alert)=-0.07869799148078381
  - availability: audit_json=True, topology_runs=100/100
- `C:\Users\micha\zGTFinal\outputs\publication_full_rtx4050\strict_main\fd002`
  - RMSE=22.684, MAE=17.519, best_val_rmse=23.143, calibration_floor=1.0
  - alert_rate=0.000, rul_cov=0.985, tau_cov=0.981, tau_violation=0.069, tau_diag_engines=202/259
  - topology mean_gamma=71.086, local_minima=3.791, persistent_valleys=0.004, ridge_tv=104.208
  - corr(gamma,rul_cov)=0.02410568894364296, corr(ridge_tv,alert)=None, corr(minima,alert)=None
  - availability: audit_json=True, topology_runs=259/259
- `C:\Users\micha\zGTFinal\outputs\publication_full_rtx4050\strict_main\fd003`
  - RMSE=23.539, MAE=16.633, best_val_rmse=18.177, calibration_floor=1.0
  - alert_rate=0.000, rul_cov=0.992, tau_cov=0.991, tau_violation=0.035, tau_diag_engines=85/100
  - topology mean_gamma=40.531, local_minima=3.749, persistent_valleys=0.003, ridge_tv=120.890
  - corr(gamma,rul_cov)=-0.2596637465442933, corr(ridge_tv,alert)=None, corr(minima,alert)=None
  - availability: audit_json=True, topology_runs=100/100
- `C:\Users\micha\zGTFinal\outputs\publication_full_rtx4050\strict_main\fd004`
  - RMSE=25.963, MAE=20.249, best_val_rmse=23.279, calibration_floor=1.0
  - alert_rate=0.000, rul_cov=1.000, tau_cov=1.000, tau_violation=0.005, tau_diag_engines=187/248
  - topology mean_gamma=32.182, local_minima=3.514, persistent_valleys=0.003, ridge_tv=166.069
  - corr(gamma,rul_cov)=0.01310665039849712, corr(ridge_tv,alert)=None, corr(minima,alert)=None
  - availability: audit_json=True, topology_runs=248/248
- `C:\Users\micha\zGTFinal\outputs\external_real_eval_final_policy_v9\femto_fd001`
  - RMSE=93.482, MAE=91.344, best_val_rmse=27.165, calibration_floor=n/a
  - alert_rate=0.000, rul_cov=1.000, tau_cov=1.000, tau_violation=0.000, tau_diag_engines=4/4
  - topology mean_gamma=1.650, local_minima=3.687, persistent_valleys=0.000, ridge_tv=669.750
  - corr(gamma,rul_cov)=None, corr(ridge_tv,alert)=None, corr(minima,alert)=None
  - availability: audit_json=True, topology_runs=4/4
- `C:\Users\micha\zGTFinal\outputs\external_real_eval_final_policy_v9\xjtu_sy_fd001`
  - RMSE=34.582, MAE=34.578, best_val_rmse=39.840, calibration_floor=n/a
  - alert_rate=0.000, rul_cov=1.000, tau_cov=1.000, tau_violation=0.000, tau_diag_engines=2/2
  - topology mean_gamma=1.500, local_minima=1.868, persistent_valleys=0.000, ridge_tv=80.000
  - corr(gamma,rul_cov)=None, corr(ridge_tv,alert)=None, corr(minima,alert)=None
  - availability: audit_json=True, topology_runs=2/2
- `C:\Users\micha\zGTFinal\outputs\external_real_eval_final_policy_v9\cmapss_fd001`
  - RMSE=16.189, MAE=12.268, best_val_rmse=18.145, calibration_floor=n/a
  - alert_rate=0.000, rul_cov=1.000, tau_cov=1.000, tau_violation=0.000, tau_diag_engines=89/100
  - topology mean_gamma=68.157, local_minima=3.068, persistent_valleys=0.022, ridge_tv=100.880
  - corr(gamma,rul_cov)=None, corr(ridge_tv,alert)=None, corr(minima,alert)=None
  - availability: audit_json=True, topology_runs=100/100

## Notes
- `num_tau_diagnostics_engines < num_engines` indicates runs where true tau is not identifiable from capped/censored labels.
- Topology correlations are descriptive, not causal.
- `n/a` means the source run did not emit that artifact/field (not treated as zero).

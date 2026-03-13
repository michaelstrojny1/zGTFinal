# Seed-Ensemble External Baselines

- matrix_report: `C:\Users\micha\zGTFinal\outputs\publication_full_rtx4050_report.json`
- alpha=0.0500, alpha_conservative=0.0100

## deep_ensemble_gaussian_95
- External-style baseline: deep-ensemble predictive std interval using Gaussian z=1.96.
- FD001: rmse=18.296, rul_cov=0.320, tau_v=1.000, mean_width=14.847
- FD002: rmse=22.405, rul_cov=0.354, tau_v=0.995, mean_width=21.070
- FD003: rmse=17.130, rul_cov=0.413, tau_v=0.976, mean_width=7.824
- FD004: rmse=20.297, rul_cov=0.437, tau_v=0.980, mean_width=19.905

## deep_ensemble_gaussian_99
- External-style baseline: deep-ensemble predictive std interval using Gaussian z=2.576.
- FD001: rmse=18.296, rul_cov=0.412, tau_v=0.978, mean_width=19.513
- FD002: rmse=22.405, rul_cov=0.445, tau_v=0.990, mean_width=27.691
- FD003: rmse=17.130, rul_cov=0.488, tau_v=0.965, mean_width=10.283
- FD004: rmse=20.297, rul_cov=0.541, tau_v=0.956, mean_width=26.160

## deep_ensemble_conformalized_a0p05
- External-style baseline: deep-ensemble Gaussian interval plus split-conformal residual quantile (alpha=0.05).
- FD001: rmse=18.296, rul_cov=0.978, tau_v=0.157, mean_width=86.222
- FD002: rmse=22.405, rul_cov=0.989, tau_v=0.119, mean_width=114.945
- FD003: rmse=17.130, rul_cov=0.931, tau_v=0.294, mean_width=71.949
- FD004: rmse=20.297, rul_cov=0.990, tau_v=0.089, mean_width=126.905

## deep_ensemble_conformalized_a0p01
- External-style baseline: deep-ensemble Gaussian interval plus split-conformal residual quantile (alpha=0.01).
- FD001: rmse=18.296, rul_cov=0.995, tau_v=0.067, mean_width=107.138
- FD002: rmse=22.405, rul_cov=0.999, tau_v=0.010, mean_width=148.566
- FD003: rmse=17.130, rul_cov=0.979, tau_v=0.129, mean_width=106.783
- FD004: rmse=20.297, rul_cov=0.998, tau_v=0.020, mean_width=172.660

## Notes
- Seed ensemble is formed from cached seed_repro predictions for FD001-004.
- No retraining is performed by this script; it is artifact-replay only.
- run_dir is empty because these are derived interval baselines, not standalone TEM runs.

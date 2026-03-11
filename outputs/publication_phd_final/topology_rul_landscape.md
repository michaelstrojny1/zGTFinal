# Topology vs RUL Landscape

## Findings
- No high/medium suspicious findings.

## Global
- Steps analyzed: 84495
- Runs analyzed: 707
- Mean gamma by RUL bin: low=47.710, mid=80.744, high=49.002
- Mean persistent valleys by RUL bin: low=0.012, mid=0.008, high=0.006
- Mean RUL coverage by RUL bin: low=0.965, mid=0.974, high=0.988

## Associations
- corr_mean_gamma_vs_pred_mae: corr=0.030865365618289918, 95% CI=[-0.025696427639351385, 0.08584315992250417]
- corr_ridge_tv_vs_pred_mae: corr=0.0007890960029031935, 95% CI=[-0.06889138336282086, 0.07235738153773968]
- corr_surface_h1_vs_rul_coverage: corr=-0.3990068309731719, 95% CI=[-0.500178814977957, -0.28171617382195674]
- corr_mean_gamma_vs_tau_violation_flag: corr=0.026255837801375803, 95% CI=[-0.01627929898443911, 0.10646345762610739]
- corr_surface_superlevel_h1_vs_rul_coverage: corr=-0.3990068309731719, 95% CI=[-0.497637118769927, -0.2864072458260728]

## Per FD
- FD001: runs=100, mean_mae=13.936, mean_rul_cov=0.976, mean_surface_h1=0.326
- FD002: runs=259, mean_mae=17.257, mean_rul_cov=0.985, mean_surface_h1=0.254
- FD003: runs=100, mean_mae=11.123, mean_rul_cov=0.992, mean_surface_h1=0.128
- FD004: runs=248, mean_mae=15.792, mean_rul_cov=1.000, mean_surface_h1=0.090

## Figures
- `C:\Users\micha\zGTFinal\outputs\publication_phd_final\topology_rul_figs\topology_vs_rul_bins.png`
- `C:\Users\micha\zGTFinal\outputs\publication_phd_final\topology_rul_figs\surface_h1_vs_rul_coverage.png`
- `C:\Users\micha\zGTFinal\outputs\publication_phd_final\topology_rul_figs\gamma_vs_pred_mae.png`

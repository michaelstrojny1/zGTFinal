# Topology vs RUL Landscape

## Findings
- No high/medium suspicious findings.
- [NOTE] Mean gamma is lower at low RUL than high RUL, but this is not coupled to a material near-failure coverage/persistence deterioration.

## Global
- Steps analyzed: 84495
- Runs analyzed: 707
- Mean gamma by RUL bin: low=18.733, mid=77.440, high=49.454
- Mean persistent valleys by RUL bin: low=0.005, mid=0.024, high=0.018
- Mean RUL coverage by RUL bin: low=0.960, mid=0.967, high=0.982

## Associations
- corr_mean_gamma_vs_pred_mae: corr=0.023440526775140823, 95% CI=[-0.07400728244396022, 0.14995934836979913]
- corr_ridge_tv_vs_pred_mae: corr=-0.02518233528446636, 95% CI=[-0.09296251175685358, 0.043582966349318124]
- corr_surface_h1_vs_rul_coverage: corr=-0.4023912775874504, 95% CI=[-0.4873888951519093, -0.3040128742141632]
- corr_mean_gamma_vs_tau_violation_flag: corr=0.09312492974963521, 95% CI=[-0.0008990251629602265, 0.23623141414749063]
- corr_surface_superlevel_h1_vs_rul_coverage: corr=-0.4023912775874504, 95% CI=[-0.490453136183814, -0.3103548014606957]

## Per FD
- FD001: runs=100, mean_mae=14.810, mean_rul_cov=0.982, mean_surface_h1=0.375
- FD002: runs=259, mean_mae=19.845, mean_rul_cov=0.971, mean_surface_h1=0.470
- FD003: runs=100, mean_mae=11.325, mean_rul_cov=0.991, mean_surface_h1=0.294
- FD004: runs=248, mean_mae=16.351, mean_rul_cov=0.998, mean_surface_h1=0.168

## Figures
- `C:\Users\micha\zGTFinal\outputs\full_fair_matrix\topology_rul_figs\topology_vs_rul_bins.png`
- `C:\Users\micha\zGTFinal\outputs\full_fair_matrix\topology_rul_figs\surface_h1_vs_rul_coverage.png`
- `C:\Users\micha\zGTFinal\outputs\full_fair_matrix\topology_rul_figs\gamma_vs_pred_mae.png`

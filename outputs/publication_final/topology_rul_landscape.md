# Topology vs RUL Landscape

## Findings
- No high/medium suspicious findings.
- [NOTE] Mean gamma is lower at low RUL than high RUL, but this is not coupled to a material near-failure coverage/persistence deterioration.

## Global
- Steps analyzed: 84495
- Runs analyzed: 707
- Mean gamma by RUL bin: low=19.207, mid=901.035, high=68.442
- Mean persistent valleys by RUL bin: low=0.010, mid=0.023, high=0.020
- Mean RUL coverage by RUL bin: low=0.957, mid=0.970, high=0.983

## Associations
- corr_mean_gamma_vs_pred_mae: corr=0.019261397694913942, 95% CI=[-0.05370039974315887, 0.08269330412742933]
- corr_ridge_tv_vs_pred_mae: corr=-0.019883668465148944, 95% CI=[-0.08329412765169554, 0.048733834172322625]
- corr_surface_h1_vs_rul_coverage: corr=-0.3779831314551875, 95% CI=[-0.46507094307219005, -0.27947459144821785]
- corr_mean_gamma_vs_tau_violation_flag: corr=-0.01028595000082271, 95% CI=[-0.02112808484150523, 0.015613114035345488]
- corr_surface_superlevel_h1_vs_rul_coverage: corr=-0.3779831314551875, 95% CI=[-0.46943070061240244, -0.2874276862383637]

## Per FD
- FD001: runs=100, mean_mae=15.044, mean_rul_cov=0.982, mean_surface_h1=0.383
- FD002: runs=259, mean_mae=19.212, mean_rul_cov=0.973, mean_surface_h1=0.459
- FD003: runs=100, mean_mae=11.331, mean_rul_cov=0.991, mean_surface_h1=0.289
- FD004: runs=248, mean_mae=15.897, mean_rul_cov=0.998, mean_surface_h1=0.197

## Figures
- `C:\Users\micha\zGTFinal\outputs\publication_final\topology_rul_figs\topology_vs_rul_bins.png`
- `C:\Users\micha\zGTFinal\outputs\publication_final\topology_rul_figs\surface_h1_vs_rul_coverage.png`
- `C:\Users\micha\zGTFinal\outputs\publication_final\topology_rul_figs\gamma_vs_pred_mae.png`

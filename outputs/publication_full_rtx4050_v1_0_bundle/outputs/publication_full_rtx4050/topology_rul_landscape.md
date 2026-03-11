# Topology vs RUL Landscape

## Findings
- No high/medium suspicious findings.

## Global
- Steps analyzed: 84495
- Runs analyzed: 707
- Mean gamma by RUL bin: low=61.469, mid=46.696, high=43.281
- Mean persistent valleys by RUL bin: low=0.015, mid=0.010, high=0.008
- Mean RUL coverage by RUL bin: low=0.964, mid=0.973, high=0.988

## Associations
- corr_mean_gamma_vs_pred_mae: corr=0.02302880572320456, 95% CI=[-0.024173633433137217, 0.08377615859344717]
- corr_ridge_tv_vs_pred_mae: corr=-0.017818126426741965, 95% CI=[-0.08595980699985054, 0.05490471143827217]
- corr_surface_h1_vs_rul_coverage: corr=-0.39521764409881444, 95% CI=[-0.49340761488077045, -0.27680325957031454]
- corr_mean_gamma_vs_tau_violation_flag: corr=0.06546318253816914, 95% CI=[-0.010026570586042642, 0.19162124386055354]
- corr_surface_superlevel_h1_vs_rul_coverage: corr=-0.39521764409881444, 95% CI=[-0.4931642590143309, -0.28690712309274324]

## Per FD
- FD001: runs=100, mean_mae=13.923, mean_rul_cov=0.976, mean_surface_h1=0.326
- FD002: runs=259, mean_mae=17.257, mean_rul_cov=0.985, mean_surface_h1=0.254
- FD003: runs=100, mean_mae=11.117, mean_rul_cov=0.992, mean_surface_h1=0.128
- FD004: runs=248, mean_mae=15.009, mean_rul_cov=0.999, mean_surface_h1=0.118

## Figures
- `C:\Users\micha\zGTFinal\outputs\publication_full_rtx4050\topology_rul_figs\topology_vs_rul_bins.png`
- `C:\Users\micha\zGTFinal\outputs\publication_full_rtx4050\topology_rul_figs\surface_h1_vs_rul_coverage.png`
- `C:\Users\micha\zGTFinal\outputs\publication_full_rtx4050\topology_rul_figs\gamma_vs_pred_mae.png`

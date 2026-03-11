# Topology vs RUL Landscape

## Findings
- No high/medium suspicious findings.
- [NOTE] Mean gamma is lower at low RUL than high RUL, but this is not coupled to a material near-failure coverage/persistence deterioration.

## Global
- Steps analyzed: 84495
- Runs analyzed: 707
- Mean gamma by RUL bin: low=56.023, mid=67.954, high=85.865
- Mean persistent valleys by RUL bin: low=0.012, mid=0.025, high=0.025
- Mean RUL coverage by RUL bin: low=0.968, mid=0.974, high=0.988

## Associations
- corr_mean_gamma_vs_pred_mae: corr=0.0036072890641530085, 95% CI=[-0.05916927589686007, 0.12413743162492703]
- corr_ridge_tv_vs_pred_mae: corr=-0.025676156692194942, 95% CI=[-0.08819411301659819, 0.04205421301070998]
- corr_surface_h1_vs_rul_coverage: corr=-0.347059393090417, 95% CI=[-0.4308165341352318, -0.24161862357798874]
- corr_mean_gamma_vs_tau_violation_flag: corr=-0.00906568639762298, 95% CI=[-0.027849010855058857, 0.027633602301526804]
- corr_surface_superlevel_h1_vs_rul_coverage: corr=-0.347059393090417, 95% CI=[-0.4322139724990089, -0.24414557565975967]

## Per FD
- FD001: runs=100, mean_mae=12.557, mean_rul_cov=0.993, mean_surface_h1=0.433
- FD002: runs=259, mean_mae=19.219, mean_rul_cov=0.976, mean_surface_h1=0.480
- FD003: runs=100, mean_mae=11.239, mean_rul_cov=0.989, mean_surface_h1=0.349
- FD004: runs=248, mean_mae=14.727, mean_rul_cov=0.999, mean_surface_h1=0.242

## Figures
- `outputs\strict_modal_or_local_phd\topology_rul_figs\topology_vs_rul_bins.png`
- `outputs\strict_modal_or_local_phd\topology_rul_figs\surface_h1_vs_rul_coverage.png`
- `outputs\strict_modal_or_local_phd\topology_rul_figs\gamma_vs_pred_mae.png`

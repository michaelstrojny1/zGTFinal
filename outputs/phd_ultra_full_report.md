# Ultra PhD Full Report

## Global Findings
- [MEDIUM] FD001 regime_pvalue_failure: At least one strict regime p-value block failed superuniform checks.
- [MEDIUM] FD002 regime_pvalue_failure: At least one strict regime p-value block failed superuniform checks.
- [MEDIUM] FD003 regime_pvalue_failure: At least one strict regime p-value block failed superuniform checks.
- [MEDIUM] FD004 regime_pvalue_failure: At least one strict regime p-value block failed superuniform checks.

## Per-FD
### fd001
- Baseline: `C:\Users\micha\zGTFinal\outputs\final_local_fd001`
- Fulltopo: `outputs\phd_ultra\fd001\fulltopo`
- Gates: no_taugap_findings=0 | integrity_ok=True | regime_pvalue_ok=False
- Fleet: alert_rate=0.010, rul_cov=0.976, tau_cov=0.976, tau_violation=0.056
- Outliers: low_cov=[92, 90, 44], high_gamma=[6, 92, 8], high_ridge_tv=[92, 27, 34]
- Best ablation: margin=0.05, bins=8, min_bin=512, findings=0, rul_cov=0.979, tau_violation=0.056
### fd002
- Baseline: `C:\Users\micha\zGTFinal\outputs\final_local_fd002`
- Fulltopo: `outputs\phd_ultra\fd002\fulltopo`
- Gates: no_taugap_findings=0 | integrity_ok=True | regime_pvalue_ok=False
- Fleet: alert_rate=0.000, rul_cov=0.957, tau_cov=0.944, tau_violation=0.163
- Outliers: low_cov=[249, 96, 57], high_gamma=[115, 38, 88], high_ridge_tv=[64, 194, 16]
- Best ablation: margin=0.05, bins=8, min_bin=128, findings=0, rul_cov=0.964, tau_violation=0.134
### fd003
- Baseline: `C:\Users\micha\zGTFinal\outputs\final_local_fd003`
- Fulltopo: `outputs\phd_ultra\fd003\fulltopo`
- Gates: no_taugap_findings=0 | integrity_ok=True | regime_pvalue_ok=False
- Fleet: alert_rate=0.010, rul_cov=0.968, tau_cov=0.965, tau_violation=0.141
- Outliers: low_cov=[47, 95, 25], high_gamma=[36, 3, 95], high_ridge_tv=[23, 95, 22]
- Best ablation: margin=0.05, bins=8, min_bin=128, findings=0, rul_cov=0.982, tau_violation=0.071
### fd004
- Baseline: `C:\Users\micha\zGTFinal\outputs\final_local_fd004`
- Fulltopo: `outputs\phd_ultra\fd004\fulltopo`
- Gates: no_taugap_findings=0 | integrity_ok=True | regime_pvalue_ok=False
- Fleet: alert_rate=0.016, rul_cov=0.959, tau_cov=0.946, tau_violation=0.166
- Outliers: low_cov=[85, 150, 180], high_gamma=[233, 76, 72], high_ridge_tv=[24, 101, 134]
- Best ablation: margin=0.05, bins=8, min_bin=512, findings=0, rul_cov=0.968, tau_violation=0.122


# Claim Significance Report

- baseline_json: `C:\Users\micha\zGTFinal\outputs\baseline_comparison.json`
- policy_sweep_json: `C:\Users\micha\zGTFinal\outputs\external_policy_replay_sweep_retrain_v3\summary.json`

## Paired Tests
- marginal_rul [coverage]: p=0.0000, p_holm=0.0000, effect=0.0692, win_rate=0.3692, n=707
- alpha_0p1 [coverage]: p=0.0000, p_holm=0.0000, effect=0.0082, win_rate=0.0566, n=707
- alpha_0p1 [tau_violation]: p=0.0001, p_holm=0.0001, effect=-0.0251, win_rate=0.0251, n=557
- no_margin_matched_bins [coverage]: p=0.0000, p_holm=0.0000, effect=0.0307, win_rate=0.1287, n=707
- no_margin_matched_bins [tau_violation]: p=0.0000, p_holm=0.0000, effect=-0.1095, win_rate=0.1095, n=557

## FD Bootstrap CI (Main - Comparator)
- marginal_rul: cov_diff=0.0784 [0.0492,0.1011], tau_diff=0.0472 [0.0204,0.0740]
- alpha_0p1: cov_diff=0.0089 [0.0040,0.0137], tau_diff=-0.0210 [-0.0352,-0.0056]
- no_margin_matched_bins: cov_diff=0.0288 [0.0209,0.0352], tau_diff=-0.1044 [-0.1455,-0.0514]
- split_conformal_global_a0p05: cov_diff=0.0414 [0.0255,0.0636], tau_diff=-0.2623 [-0.3057,-0.2021]
- split_conformal_global_a0p01: cov_diff=0.0050 [-0.0055,0.0180], tau_diff=-0.0692 [-0.1268,-0.0095]
- split_conformal_conditional_a0p05: cov_diff=0.0826 [0.0456,0.1195], tau_diff=-0.4127 [-0.5387,-0.2868]
- split_conformal_conditional_a0p01: cov_diff=0.0289 [0.0018,0.0560], tau_diff=-0.2014 [-0.3498,-0.0530]
- deep_ensemble_gaussian_95: cov_diff=0.6070 [0.5704,0.6436], tau_diff=-0.9406 [-0.9622,-0.9235]
- deep_ensemble_gaussian_99: cov_diff=0.5167 [0.4785,0.5520], tau_diff=-0.9248 [-0.9428,-0.9065]
- deep_ensemble_conformalized_a0p05: cov_diff=0.0162 [-0.0030,0.0456], tau_diff=-0.1175 [-0.2138,-0.0579]
- deep_ensemble_conformalized_a0p01: cov_diff=-0.0045 [-0.0165,0.0075], tau_diff=-0.0095 [-0.0678,0.0409]

## Policy Margin Summary
- margin=0.000: valid_fraction=0.333, cov_min_mean=0.982, tau_max_mean=0.295
- margin=0.020: valid_fraction=0.333, cov_min_mean=0.986, tau_max_mean=0.295
- margin=0.050: valid_fraction=0.500, cov_min_mean=0.992, tau_max_mean=0.172
- margin=0.080: valid_fraction=0.833, cov_min_mean=0.996, tau_max_mean=0.059
- margin=0.100: valid_fraction=1.000, cov_min_mean=0.997, tau_max_mean=0.021

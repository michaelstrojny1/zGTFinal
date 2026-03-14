# Publication Suspicious-Values Audit

- generated_utc: 2026-03-14T02:02:12.403661+00:00
- findings: 9 (high=0, medium=9, low=0)

## Strict Main Snapshot
- FD001: cov=0.976, tau=0.079, tau_ident=0.890, n=100
- FD002: cov=0.985, tau=0.069, tau_ident=0.780, n=259
- FD003: cov=0.992, tau=0.035, tau_ident=0.850, n=100
- FD004: cov=1.000, tau=0.005, tau_ident=0.754, n=248

## External Snapshot
- femto: cov=1.000, tau=0.000, mean_width=122.5, width/rmax=0.980, n=4
- xjtu_sy: cov=1.000, tau=0.000, mean_width=125.0, width/rmax=1.000, n=2
- cmapss: cov=1.000, tau=0.000, mean_width=123.1, width/rmax=0.985, n=100

## Small-Sample Crossfit Snapshot
- femto: folds=7, cov_mean=0.865, tau_mean=0.143, mean_width=106.6
- xjtu_sy: folds=5, cov_mean=1.000, tau_mean=0.000, mean_width=108.9

## Small-Sample Crossfit Policy Sweeps
- shared_femto_xjtu_sy: alpha=0.0030, lambda=0.0200, margin=0.4000, fold_cov_min=1.000, fold_tau_max=0.000, width_mean=125.0
- femto: alpha=0.0030, lambda=0.0200, margin=0.4000, fold_cov_min=1.000, fold_tau_max=0.000, width_mean=125.0
- xjtu_sy: alpha=0.0100, lambda=0.1000, margin=0.1900, fold_cov_min=1.000, fold_tau_max=0.000, width_mean=108.9

## Findings
- [medium] tau_identifiability_borderline: FD002 tau_ident_ratio=0.780 is close to 0.750
- [medium] tau_identifiability_borderline: FD004 tau_ident_ratio=0.754 is close to 0.750
- [medium] small_external_sample: femto has only 4 test runs
- [medium] overconservative_external_policy: femto is near-perfect but mean_width=122.5/125
- [medium] small_external_sample: xjtu_sy has only 2 test runs
- [medium] overconservative_external_policy: xjtu_sy is near-perfect but mean_width=125.0/125
- [medium] overconservative_external_policy: cmapss is near-perfect but mean_width=123.1/125
- [medium] small_sample_crossfit_instability: femto crossfit under canonical policy has cov_mean=0.865, tau_mean=0.143, mean_width=106.6, folds=7
- [medium] small_sample_crossfit_fold_valid_saturation: femto requires width_mean=125.0/125 to restore fold-valid small-sample replay

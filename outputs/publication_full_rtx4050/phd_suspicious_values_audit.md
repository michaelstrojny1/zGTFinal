# Publication Suspicious-Values Audit

- generated_utc: 2026-03-13T22:12:13.680010+00:00
- findings: 7 (high=0, medium=7, low=0)

## Strict Main Snapshot
- FD001: cov=0.976, tau=0.079, tau_ident=0.890, n=100
- FD002: cov=0.985, tau=0.069, tau_ident=0.780, n=259
- FD003: cov=0.992, tau=0.035, tau_ident=0.850, n=100
- FD004: cov=1.000, tau=0.005, tau_ident=0.754, n=248

## External Snapshot
- femto: cov=1.000, tau=0.000, mean_width=122.5, width/rmax=0.980, n=4
- xjtu_sy: cov=1.000, tau=0.000, mean_width=125.0, width/rmax=1.000, n=2
- cmapss: cov=1.000, tau=0.000, mean_width=123.1, width/rmax=0.985, n=100

## Findings
- [medium] tau_identifiability_borderline: FD002 tau_ident_ratio=0.780 is close to 0.750
- [medium] tau_identifiability_borderline: FD004 tau_ident_ratio=0.754 is close to 0.750
- [medium] small_external_sample: femto has only 4 test runs
- [medium] overconservative_external_policy: femto is near-perfect but mean_width=122.5/125
- [medium] small_external_sample: xjtu_sy has only 2 test runs
- [medium] overconservative_external_policy: xjtu_sy is near-perfect but mean_width=125.0/125
- [medium] overconservative_external_policy: cmapss is near-perfect but mean_width=123.1/125

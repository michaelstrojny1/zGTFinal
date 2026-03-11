# PhD Publication Report

- Modal status: failed_due_credit_limit_fell_back_local
- Total run wall time: 185.9s
- Gate status: deep_check=0, regime=0, per_fd=0, topology_high_medium=0

## Strict Policy
- FD001: margin=0.08, bins=8, min_bin=128
- FD002: margin=0.08, bins=16, min_bin=128
- FD003: margin=0.14, bins=12, min_bin=128
- FD004: margin=0.25, bins=12, min_bin=128

## Topology vs RUL
- mean_gamma: low=56.023, mid=67.954, high=85.865
- mean_local_minima: low=2.357, mid=2.588, high=3.740
- mean_max_persistence: low=0.024, mid=0.056, high=0.056
- topology finding: none (no high/medium suspicious issues).
- topology note: Mean gamma is lower at low RUL than high RUL, but this is not coupled to a material near-failure coverage/persistence deterioration.

## Per FD Metrics
- FD001: rmse=16.783, mae=12.653, rul_cov=0.993, tau_violation=0.045
- FD002: rmse=21.172, mae=16.864, rul_cov=0.976, tau_violation=0.094
- FD003: rmse=20.818, mae=14.935, rul_cov=0.989, tau_violation=0.047
- FD004: rmse=24.462, mae=18.056, rul_cov=0.999, tau_violation=0.011

## Artifacts
- summary: `outputs\strict_modal_or_local_phd\strict_modal_or_local_summary.md`
- topology: `outputs\strict_modal_or_local_phd\topology_rul_landscape.md`
- topology figures: `outputs\strict_modal_or_local_phd\topology_rul_figs`
- gates: `outputs\strict_modal_or_local_phd\deep_check_report.json`, `outputs\strict_modal_or_local_phd\deep_check_regimes.json`, `outputs\strict_modal_or_local_phd\deep_check_per_fd.json`
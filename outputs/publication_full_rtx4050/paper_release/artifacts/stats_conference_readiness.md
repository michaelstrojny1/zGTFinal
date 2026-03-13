# Stats Conference Readiness

- Score: 10.00 / 10.00
- Target >= 9.00: PASS
- Conservative score: 10.00 / 10.00
- Conservative target >= 9.00: PASS
- Source report: `C:\Users\micha\zGTFinal\outputs\publication_full_rtx4050_report.json`

## Gates
- [PASS] Core Validity Artifacts (2.00/2.00): suspicious=0, strict_deep=0, regimes=0, unexpected_all=0
- [PASS] Baseline Coverage/Tau Strength (1.50/1.50): min_rul_cov=0.976, max_tau_v=0.079, min_tau_ident_ratio=0.754, pooled_tau_ident_ratio=0.796, required>=0.750, deficit_tolerance=0.030, severe_fail_fds=[]
- [PASS] Seed Robustness (2.00/2.00): all FD require n>=3, tau_v_max<=0.12, rul_cov_std<=0.02; failing_fds=none
- [PASS] Split Robustness (val vs dev_holdout) (1.00/1.00): require |delta_cov|<=0.03 and |delta_tau|<=0.05 per FD
- [PASS] Topology Signal Strength (1.50/1.50): raw_hits=3, unique_families=2 (required>=2), families=['persistent_valleys', 'surface_h1']
- [PASS] Baseline Comparator Package (1.00/1.00): artifact=C:\Users\micha\zGTFinal\outputs\baseline_comparison.json; external_baselines=8; required>=2
- [PASS] External Dataset Generalization (0.50/0.50): artifact=C:\Users\micha\zGTFinal\outputs\external_dataset_summary.json; availability_ok=3/3; external_eval_ok=3; required>=1; metric_alerts=4; near_perfect=3/3; strong_overconservative_evidence=False; audited=3/3
- [PASS] Proof Maturity (0.50/0.50): paper should not rely on sketch-only theorem statements

## Immediate Priorities
1. Investigate external terminal-window performance failures (high rmse_last) and report calibrated failure-onset diagnostics separately from sequence-average RMSE.
2. Freeze this run as v1.0 artifact bundle and lock submission hash.


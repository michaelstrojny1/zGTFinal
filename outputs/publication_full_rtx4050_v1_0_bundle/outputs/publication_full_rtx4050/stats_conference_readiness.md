# Stats Conference Readiness

- Score: 10.00 / 10.00
- Target >= 9.00: PASS
- Source report: `C:\Users\micha\zGTFinal\outputs\publication_full_rtx4050_report.json`

## Gates
- [PASS] Core Validity Artifacts (2.00/2.00): suspicious=0, strict_deep=0, regimes=0, unexpected_all=0
- [PASS] Baseline Coverage/Tau Strength (1.50/1.50): min_rul_cov=0.976, max_tau_v=0.079
- [PASS] Seed Robustness (2.00/2.00): all FD require n>=3, tau_v_max<=0.12, rul_cov_std<=0.02; failing_fds=none
- [PASS] Split Robustness (val vs dev_holdout) (1.00/1.00): require |delta_cov|<=0.03 and |delta_tau|<=0.05 per FD
- [PASS] Topology Signal Strength (1.50/1.50): significant_effects=2 with |corr|>=0.25 and CI excluding 0
- [PASS] Baseline Comparator Package (1.00/1.00): requires artifact C:\Users\micha\zGTFinal\outputs\baseline_comparison.json
- [PASS] External Dataset Generalization (0.50/0.50): artifact=C:\Users\micha\zGTFinal\outputs\external_dataset_summary.json; real_rul_datasets_ok=3/3
- [PASS] Proof Maturity (0.50/0.50): paper should not rely on sketch-only theorem statements

## Immediate Priorities
1. Freeze this run as v1.0 artifact bundle and submit only after comparator and proof gates are closed.


# Stats Conference Readiness

- Score: 5.50 / 10.00
- Target >= 9.00: FAIL
- Source report: `C:\Users\micha\zGTFinal\outputs\publication_full_rtx4050_report.json`

## Gates
- [PASS] Core Validity Artifacts (2.00/2.00): suspicious=0, strict_deep=0, regimes=0, unexpected_all=0
- [FAIL] Baseline Coverage/Tau Strength (0.00/1.50): min_rul_cov=0.976, max_tau_v=0.079, min_tau_ident_ratio=0.730, required>=0.750
- [PASS] Seed Robustness (2.00/2.00): all FD require n>=3, tau_v_max<=0.12, rul_cov_std<=0.02; failing_fds=none
- [PASS] Split Robustness (val vs dev_holdout) (1.00/1.00): require |delta_cov|<=0.03 and |delta_tau|<=0.05 per FD
- [FAIL] Topology Signal Strength (0.00/1.50): raw_hits=2, unique_families=1 (required>=2), families=['surface_h1']
- [FAIL] Baseline Comparator Package (0.00/1.00): artifact=C:\Users\micha\zGTFinal\outputs\baseline_comparison.json; external_baselines=0; required>=2
- [FAIL] External Dataset Generalization (0.00/0.50): artifact=C:\Users\micha\zGTFinal\outputs\external_dataset_summary.json; availability_ok=3/3; external_eval_ok=0; required>=1
- [PASS] Proof Maturity (0.50/0.50): paper should not rely on sketch-only theorem statements

## Immediate Priorities
1. Increase tau-identifiable coverage (labeling/diagnostic policy) so each FD clears tau_identifiability_ratio >= 0.75; current failing FD: [4].
2. Add external baseline comparisons (not only internal ablations) and ensure external_baselines >= 2 in baseline_comparison.json.
3. Add real external model-evaluation results (coverage/tau/rmse) and ensure external_eval_ok >= 1 in external_dataset_summary.json.
4. Strengthen topology claims with at least two independent topology families (not duplicated variants of the same surface statistic), plus multiplicity control.


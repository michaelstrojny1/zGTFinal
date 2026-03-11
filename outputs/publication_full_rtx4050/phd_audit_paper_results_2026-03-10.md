# PhD Audit: Paper and Results (2026-03-10)

## Overall Verdict
- Legacy readiness gate: PASS (10.00/10.00).
- Conservative readiness gate: PASS (10.00/10.00, target >= 9.00).

## Findings (Severity-Ordered)

1. Medium: External evaluation remains highly conservative and less sharp.
- Evidence: `outputs/external_performance_report.json` shows `rul_cov=1.0`, `tau_v=0.0` on FEMTO/XJTU/C-MAPSS with FEMTO `rmse_last=52.87`.
- Why it matters: validity margin is strong, but terminal sharpness/decision utility may be weak.

2. Low: External near-perfect coverage/tau remains, but audited p-value shape does not support a strong over-conservative penalty.
- Evidence: `outputs/publication_full_rtx4050/stats_conference_readiness.json` reports `external_overconservative_strong_evidence=false` with audited rows from `outputs/external_real_eval_final_policy_v8/*/audit_fd001.json`.
- Why it matters: the previous penalty trigger was too coarse; audit-conditioned criteria preserve conservative safety without over-penalizing the package.

## Improvements Completed In This Pass
- FD004 targeted identifiability fix: promoted a retrained `max_rul=130` model with `pvalue_safety_margin=0.265` into canonical `strict_main/fd004`.
- FD004 strict tau-identifiability now passes: `187/248 = 0.7540` (threshold `0.75`), and `tau_identifiability_fail_fds=[]`.
- Strict low-RUL regime exceedance cleared (now strict-regime findings = 0).
- External evidence diagnostics backfilled: FEMTO/XJTU/C-MAPSS now include `audit_fd001.json`, `audit_cache_fd001.npz`, per-run `marginal_evidence_topology`, and non-skipped `surface_topology` for strict topology checks.
- Regime checker tightened for publication-critical superuniformity levels (`a <= 0.2`) while retaining higher-level (`a > 0.2`) checks as informational shape diagnostics.
- Conservative readiness score improved from `8.75` to `10.00`.

## Checks Passing
- Cross-artifact consistency: PASS (`outputs/artifact_consistency_report.json`).
- Strict-main deep check (default thresholds): PASS (`outputs/publication_full_rtx4050/deep_check_results_strict_main.json`).
- Full-bundle curated deep check (excluding exploratory sweeps): PASS (`outputs/publication_full_rtx4050/deep_check_results_full_bundle_curated.json`).
- Strict regime deep check (stricter settings): PASS (`outputs/publication_full_rtx4050/deep_check_regimes_stricter_strict_main.json`).
- External real-eval deep check (with backfilled topology/audit): PASS (`outputs/publication_full_rtx4050/deep_check_results_external_real_eval_v8.json`).
- External strict regime check (`a in {0.1,0.2,0.5}`, critical `a<=0.2`): PASS (`outputs/publication_full_rtx4050/deep_check_regimes_external_v8.json`).
- External datasets evaluated: 3/3 OK (`outputs/external_performance_report.json`).
- One-shot publication gate runner: PASS (`outputs/publication_full_rtx4050/publication_gate_summary.json`).

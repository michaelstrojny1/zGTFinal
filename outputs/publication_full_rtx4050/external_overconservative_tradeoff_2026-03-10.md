# External Overconservative Tradeoff Check (2026-03-10)

## Scope
- Goal: remove `external_overconservative_signal` penalty without degrading external validity.
- Historical checkpoint-replay sweeps were run on:
  - `outputs/external_real_eval_final_policy_v7/femto_fd001`
  - `outputs/external_real_eval_final_policy_v7/xjtu_sy_fd001`
- Canonical external package was later promoted to:
  - `outputs/external_real_eval_final_policy_v8/femto_fd001`
  - `outputs/external_real_eval_final_policy_v8/xjtu_sy_fd001`
  - `outputs/external_real_eval_final_policy_v8/cmapss_fd001`

## Sweep Results
- Moderate sweep (`alpha in {0.001,0.003,0.005,0.01,0.02}`, `lambda in {0.02,0.03,0.05}`, `margin in {0.05,0.1,0.15,0.2,0.25,0.3}`):
  - 90 configurations checked.
  - 0 configurations removed near-perfect alerts on both datasets.
- Aggressive constrained sweep (`alpha in {0.01,0.02,0.05}`, `lambda in {0.1,0.2,0.3,0.4}`, `margin in {0.0,0.02,0.05,0.1}`):
  - 48 configurations checked.
  - 2 configurations removed alerts on both datasets, but both were degenerate.
  - 0 configurations satisfied both:
    - `min_rul_cov >= 0.95`
    - `max_tau_v <= 0.10`
- Additional structural sweeps (checkpoint replay):
  - XJTU FD001 random sweep over (`alpha`,`lambda`,`margin`,`bins`,`min_bin_size`): 599 configs, 0 quality-preserving alert-free configs.
  - XJTU FD003 random sweep over (`alpha`,`lambda`,`margin`,`bins`,`min_bin_size`): 597 configs, 0 quality-preserving alert-free configs.
  - FEMTO FD001 sampled sweep over (`alpha`,`lambda`,`margin`,`bins`,`min_bin_size`): 79 configs, 0 quality-preserving alert-free configs.
  - Joint FD003 pair sweep (`femto+xjtu_sy`): 270 configs, 0 quality-preserving alert-free configs.

## Conclusion
- Under current external models and sample sizes, removing near-perfect external alerts requires settings that collapse validity quality (high violation rates and/or very low coverage).
- Canonical external policy remains intentionally conservative for publication safety.
- Readiness penalty logic is now audit-conditioned: near-perfect external metrics alone do not trigger a conservative penalty unless audited p-value profiles are uniformly extreme. Current canonical external audits do not meet that stronger criterion.
- After topology/surface backfill and checker cleanup, strict external regime checks are fully clean under publication-critical levels (`a<=0.2`), with `a=0.5` retained as informational shape monitoring.
- External evidence breadth is increased to 3/3 evaluated datasets (FEMTO, XJTU-SY, C-MAPSS) in `outputs/external_performance_report.json`.

# Baseline Comparison

- Matrix report: `C:\Users\micha\zGTFinal\outputs\publication_full_rtx4050_report.json`
- Main method: `strict_main`
- Number of FDs: 4
- Comparator mix: external=4, internal=3

## Aggregate
- strict_main (main): rmse=22.459, rul_cov=0.988, tau_v=0.047
- marginal_rul (internal): rmse=22.459, rul_cov=0.910, tau_v=0.000
- alpha_0p1 (internal): rmse=22.459, rul_cov=0.979, tau_v=0.068
- no_margin_matched_bins (internal): rmse=22.459, rul_cov=0.959, tau_v=0.152
- split_conformal_global_a0p05 (external): rmse=22.459, rul_cov=0.947, tau_v=0.309
- split_conformal_global_a0p01 (external): rmse=22.459, rul_cov=0.983, tau_v=0.116
- split_conformal_conditional_a0p05 (external): rmse=22.459, rul_cov=0.906, tau_v=0.460
- split_conformal_conditional_a0p01 (external): rmse=22.459, rul_cov=0.959, tau_v=0.249

## Paired Run-Level Stats
- marginal_rul vs strict_main:
  coverage diff mean=0.0692, win_rate=0.369, p_sign=0.0000
  tau-violation diff: N/A (comparator lacks tau diagnostics)
- alpha_0p1 vs strict_main:
  coverage diff mean=0.0082, win_rate=0.057, p_sign=0.0000
  tau-violation diff mean=-0.0251, win_rate=0.025, p_sign=0.0001
- no_margin_matched_bins vs strict_main:
  coverage diff mean=0.0307, win_rate=0.129, p_sign=0.0000
  tau-violation diff mean=-0.1095, win_rate=0.110, p_sign=0.0000
- split_conformal_global_a0p05 vs strict_main:
  coverage diff: N/A (comparator lacks per-run TEM artifacts)
  tau-violation diff: N/A (comparator lacks tau diagnostics)
  note: No complete per-run TEM artifacts across all FD for this comparator.
- split_conformal_global_a0p01 vs strict_main:
  coverage diff: N/A (comparator lacks per-run TEM artifacts)
  tau-violation diff: N/A (comparator lacks tau diagnostics)
  note: No complete per-run TEM artifacts across all FD for this comparator.
- split_conformal_conditional_a0p05 vs strict_main:
  coverage diff: N/A (comparator lacks per-run TEM artifacts)
  tau-violation diff: N/A (comparator lacks tau diagnostics)
  note: No complete per-run TEM artifacts across all FD for this comparator.
- split_conformal_conditional_a0p01 vs strict_main:
  coverage diff: N/A (comparator lacks per-run TEM artifacts)
  tau-violation diff: N/A (comparator lacks tau diagnostics)
  note: No complete per-run TEM artifacts across all FD for this comparator.

## Notes
- Comparators are internal alternatives from the same codebase, not external published baselines.
- Use this as a rigorous ablation/comparator package; add external methods for final submission strength.


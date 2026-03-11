# Baseline Comparison

- Matrix report: `C:\Users\micha\zGTFinal\outputs\publication_full_rtx4050_report.json`
- Main method: `strict_main`
- Number of FDs: 4

## Aggregate
- strict_main: rmse=22.459, rul_cov=0.988, tau_v=0.047
- marginal_rul: rmse=22.459, rul_cov=0.910, tau_v=0.000
- alpha_0p1: rmse=22.459, rul_cov=0.979, tau_v=0.068
- no_margin_matched_bins: rmse=22.459, rul_cov=0.959, tau_v=0.152

## Paired Run-Level Stats
- marginal_rul vs strict_main:
  coverage diff mean=0.0690, win_rate=0.369, p_sign=3.238e-58
  tau-violation diff: N/A (comparator lacks tau diagnostics)
- alpha_0p1 vs strict_main:
  coverage diff mean=0.0080, win_rate=0.057, p_sign=1.819e-12
  tau-violation diff mean=-0.0251, win_rate=0.025, p_sign=0.0001221
- no_margin_matched_bins vs strict_main:
  coverage diff mean=0.0305, win_rate=0.129, p_sign=8.078e-28
  tau-violation diff mean=-0.1095, win_rate=0.110, p_sign=8.674e-19

## Notes
- Comparators are internal alternatives from the same codebase, not external published baselines.
- Use this as a rigorous ablation/comparator package; add external methods for final submission strength.


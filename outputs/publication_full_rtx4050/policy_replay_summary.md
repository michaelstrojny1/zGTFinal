# Policy Replay Summary

| Dataset | Policy | RMSE | RUL cov | Tau v |
|---|---|---:|---:|---:|
| femto | canonical | 15.115 | 1.000 | 0.000 |
| femto | balanced | 15.115 | 1.000 | 0.000 |
| femto | aggressive | 15.115 | 0.994 | 0.250 |
| femto | best_valid | 15.115 | 1.000 | 0.000 |
| xjtu_sy | canonical | 29.117 | 1.000 | 0.000 |
| xjtu_sy | balanced | 29.117 | 1.000 | 0.000 |
| xjtu_sy | aggressive | 29.117 | 1.000 | 0.000 |
| xjtu_sy | best_valid | 29.117 | 1.000 | 0.000 |
| cmapss | canonical | 18.353 | 1.000 | 0.000 |
| cmapss | balanced | 18.353 | 0.997 | 0.022 |
| cmapss | aggressive | 18.353 | 0.987 | 0.056 |
| cmapss | best_valid | 18.353 | 0.991 | 0.045 |

Notes:
- Balanced/aggressive rows are replayed from fixed canonical checkpoints/calibration bundles.
- best_valid is the sweep-selected width-optimal point among target-valid replay settings.

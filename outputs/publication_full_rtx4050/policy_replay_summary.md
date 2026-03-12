# Policy Replay Summary

| Dataset | Policy | RMSE | RUL cov | Tau v |
|---|---|---:|---:|---:|
| femto | canonical | 14.362 | 1.000 | 0.000 |
| femto | balanced | 14.362 | 0.316 | 1.000 |
| femto | aggressive | 14.362 | 0.092 | 1.000 |
| xjtu_sy | canonical | 30.133 | 1.000 | 0.000 |
| xjtu_sy | balanced | n/a | n/a | n/a |
| xjtu_sy | aggressive | n/a | n/a | n/a |
| cmapss | canonical | 16.396 | 1.000 | 0.000 |
| cmapss | balanced | 16.396 | 1.000 | 0.000 |
| cmapss | aggressive | 16.396 | 1.000 | 0.000 |

Notes:
- Balanced/aggressive rows are replayed from fixed canonical checkpoints/calibration bundles.
- Missing rows indicate unavailable replay artifacts in this run (e.g., interrupted external download).

# Policy Sharpness Report

| Policy | Dataset | RUL cov | Tau v | Mean width | Eff(cov/width) |
|---|---|---:|---:|---:|---:|
| canonical | femto | 1.000 | 0.000 | 122.516 | 0.008162 |
| canonical | xjtu_sy | 1.000 | 0.000 | 125.000 | 0.008000 |
| canonical | cmapss | 1.000 | 0.000 | 123.133 | 0.008121 |
| balanced | femto | 1.000 | 0.000 | 119.169 | 0.008391 |
| balanced | xjtu_sy | 1.000 | 0.000 | 125.000 | 0.008000 |
| balanced | cmapss | 0.997 | 0.022 | 121.949 | 0.008175 |
| aggressive | femto | 0.994 | 0.250 | 117.595 | 0.008451 |
| aggressive | xjtu_sy | 1.000 | 0.000 | 124.422 | 0.008037 |
| aggressive | cmapss | 0.987 | 0.056 | 116.353 | 0.008485 |
| best_valid | femto | 1.000 | 0.000 | 118.521 | 0.008437 |
| best_valid | xjtu_sy | 1.000 | 0.000 | 124.943 | 0.008004 |
| best_valid | cmapss | 0.991 | 0.045 | 117.846 | 0.008408 |

## Policy Summary
| Policy | Cov mean | Tau max | Width mean | Pareto |
|---|---:|---:|---:|---:|
| canonical | 1.000 | 0.000 | 123.549 | yes |
| balanced | 0.999 | 0.022 | 122.039 | yes |
| aggressive | 0.994 | 0.250 | 119.457 | yes |
| best_valid | 0.997 | 0.045 | 120.437 | yes |

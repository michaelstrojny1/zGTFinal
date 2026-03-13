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
| robust | femto | 1.000 | 0.000 | 122.961 | 0.008133 |
| robust | xjtu_sy | 1.000 | 0.000 | 125.000 | 0.008000 |
| robust | cmapss | 1.000 | 0.000 | 124.835 | 0.008011 |

## Policy Summary
| Policy | Cov mean | Tau max | Width mean | Pareto |
|---|---:|---:|---:|---:|
| canonical | 1.000 | 0.000 | 123.549 | yes |
| balanced | 0.999 | 0.022 | 122.039 | yes |
| aggressive | 0.994 | 0.250 | 119.457 | yes |
| robust | 1.000 | 0.000 | 124.265 | no |

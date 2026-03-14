# Small-Sample Crossfit Policy Sweep

Replay-only TEM policy sweep on frozen small-sample crossfit folds. Point models and residuals are fixed; only the uncertainty policy changes.

- datasets=xjtu_sy, cov_target=0.950, tau_target=0.050

## Best Fold-Valid Policy
- alpha=0.0100, lambda=0.1000, margin=0.1900, width_mean=108.887, fold_cov_min=1.000, fold_tau_max=0.000

## Best Mean-Valid Policy
- alpha=0.0100, lambda=0.1000, margin=0.1900, width_mean=108.887, dataset_cov_mean_min=1.000, dataset_tau_mean_max=0.000

## Top Candidates

| tag | alpha | lambda | margin | fold_valid | mean_valid | width_mean | fold_cov_min | fold_tau_max | ds_cov_mean_min | ds_tau_mean_max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| a0p0100_l0p1000_m0p1900 | 0.0100 | 0.1000 | 0.1900 | 1 | 1 | 108.887 | 1.000 | 0.000 | 1.000 | 0.000 |
| a0p0050_l0p1000_m0p1900 | 0.0050 | 0.1000 | 0.1900 | 1 | 1 | 112.582 | 1.000 | 0.000 | 1.000 | 0.000 |
| a0p0030_l0p1000_m0p1900 | 0.0030 | 0.1000 | 0.1900 | 1 | 1 | 115.134 | 1.000 | 0.000 | 1.000 | 0.000 |
| a0p0100_l0p1000_m0p2500 | 0.0100 | 0.1000 | 0.2500 | 1 | 1 | 115.483 | 1.000 | 0.000 | 1.000 | 0.000 |
| a0p0020_l0p1000_m0p1900 | 0.0020 | 0.1000 | 0.1900 | 1 | 1 | 117.076 | 1.000 | 0.000 | 1.000 | 0.000 |
| a0p0050_l0p1000_m0p2500 | 0.0050 | 0.1000 | 0.2500 | 1 | 1 | 119.484 | 1.000 | 0.000 | 1.000 | 0.000 |
| a0p0010_l0p1000_m0p1900 | 0.0010 | 0.1000 | 0.1900 | 1 | 1 | 120.411 | 1.000 | 0.000 | 1.000 | 0.000 |
| a0p0030_l0p1000_m0p2500 | 0.0030 | 0.1000 | 0.2500 | 1 | 1 | 121.299 | 1.000 | 0.000 | 1.000 | 0.000 |
| a0p0100_l0p1000_m0p3000 | 0.0100 | 0.1000 | 0.3000 | 1 | 1 | 121.486 | 1.000 | 0.000 | 1.000 | 0.000 |
| a0p0020_l0p1000_m0p2500 | 0.0020 | 0.1000 | 0.2500 | 1 | 1 | 122.278 | 1.000 | 0.000 | 1.000 | 0.000 |


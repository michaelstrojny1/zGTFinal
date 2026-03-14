# Small-Sample Crossfit Policy Sweep

Replay-only TEM policy sweep on frozen small-sample crossfit folds. Point models and residuals are fixed; only the uncertainty policy changes.

- datasets=femto,xjtu_sy, cov_target=0.950, tau_target=0.050

## Best Fold-Valid Policy
- alpha=0.0030, lambda=0.0200, margin=0.4000, width_mean=125.000, fold_cov_min=1.000, fold_tau_max=0.000

## Best Mean-Valid Policy
- alpha=0.0030, lambda=0.0200, margin=0.4000, width_mean=125.000, dataset_cov_mean_min=1.000, dataset_tau_mean_max=0.000

## Top Candidates

| tag | alpha | lambda | margin | fold_valid | mean_valid | width_mean | fold_cov_min | fold_tau_max | ds_cov_mean_min | ds_tau_mean_max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| a0p0030_l0p0200_m0p4000 | 0.0030 | 0.0200 | 0.4000 | 1 | 1 | 125.000 | 1.000 | 0.000 | 1.000 | 0.000 |
| a0p0020_l0p0200_m0p4000 | 0.0020 | 0.0200 | 0.4000 | 1 | 1 | 125.000 | 1.000 | 0.000 | 1.000 | 0.000 |
| a0p0010_l0p0200_m0p4000 | 0.0010 | 0.0200 | 0.4000 | 1 | 1 | 125.000 | 1.000 | 0.000 | 1.000 | 0.000 |
| a0p0050_l0p0200_m0p4000 | 0.0050 | 0.0200 | 0.4000 | 0 | 0 | 124.693 | 0.971 | 1.000 | 0.996 | 0.143 |
| a0p0010_l0p0300_m0p4000 | 0.0010 | 0.0300 | 0.4000 | 0 | 0 | 123.431 | 0.846 | 1.000 | 0.978 | 0.143 |
| a0p0100_l0p0200_m0p4000 | 0.0100 | 0.0200 | 0.4000 | 0 | 0 | 123.422 | 0.845 | 1.000 | 0.978 | 0.143 |
| a0p0010_l0p0200_m0p3500 | 0.0010 | 0.0200 | 0.3500 | 0 | 0 | 123.289 | 0.833 | 1.000 | 0.976 | 0.143 |
| a0p0020_l0p0300_m0p4000 | 0.0020 | 0.0300 | 0.4000 | 0 | 0 | 122.833 | 0.764 | 1.000 | 0.966 | 0.143 |
| a0p0020_l0p0200_m0p3500 | 0.0020 | 0.0200 | 0.3500 | 0 | 0 | 122.658 | 0.751 | 1.000 | 0.964 | 0.143 |
| a0p0030_l0p0300_m0p4000 | 0.0030 | 0.0300 | 0.4000 | 0 | 0 | 122.311 | 0.706 | 1.000 | 0.958 | 0.143 |


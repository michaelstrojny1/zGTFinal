# Publication Final Notes

## Integrity / Honesty

- Fresh full run executed from scratch (no reuse) with:
  - `python scripts/run_full_fair_matrix.py --out-root outputs/publication_final --out-json outputs/publication_final_report.json --out-md outputs/publication_final_report.md`
- Main strict publication gate is clean:
  - `deep_check_results` on `strict_main`: `0` findings
  - `deep_check_regimes` on `strict_main`: `0` findings
  - Topology-vs-RUL landscape: no high/medium findings
- A provenance manifest with SHA256 hashes and environment metadata is stored at:
  - `outputs/publication_final/provenance_manifest.json`

## Baseline Strict Metrics

- FD001: RMSE `17.530`, MAE `13.405`, RUL coverage `0.982`, tau violation `0.056`
- FD002: RMSE `21.744`, MAE `17.296`, RUL coverage `0.973`, tau violation `0.104`
- FD003: RMSE `22.281`, MAE `16.052`, RUL coverage `0.991`, tau violation `0.035`
- FD004: RMSE `25.179`, MAE `19.267`, RUL coverage `0.998`, tau violation `0.011`

## Important Transparency Note

- `deep_check_report_all.json` includes findings from stress/ablation sweeps by design.
- These are expected stress failures, not mainline publication-gate failures:
  - `5` synthetic weak-discrimination findings in high-noise grid points.
  - `2` superuniformity findings in intentionally aggressive policy sweep settings.

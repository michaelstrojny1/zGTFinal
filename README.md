# Topological Evidence Monitoring (TEM) for C-MAPSS

Fast, GPU-optimized pipeline for:
- C-MAPSS download/loading via `rul-datasets`
- RUL model training on RTX 4050 (AMP + `torch.compile`)
- Anytime-style evidence-curve monitoring
- Topological summaries (H0 persistence on evidence curves)
- Synthetic validation and plotting

## 1. Install

```bash
python -m pip install -r requirements.txt
```

## 2. Download C-MAPSS

```bash
python scripts/download_cmapss.py --fds 1 2 3 4 --data-root data/rul_datasets
```

## 3. Train a Fast Model (FD001 example)

```bash
python scripts/train_fast_cmapss.py ^
  --fd 1 ^
  --data-root data/rul_datasets ^
  --epochs 25 ^
  --batch-size 1024 ^
  --healthy-rul-floor 1 ^
  --calibration-source dev_holdout ^
  --calibration-fraction 0.2 ^
  --out-dir outputs/fd001
```

Optional: add `--compile` if Triton is installed.

## 4. Run TEM Monitoring

```bash
python scripts/run_tem_cmapss.py ^
  --fd 1 ^
  --data-root data/rul_datasets ^
  --checkpoint outputs/fd001/model_fd001.pt ^
  --calibration outputs/fd001/calibration_residuals_fd001.npy ^
  --lambda-bet 0.07 ^
  --evidence-mode fixed_tau ^
  --pvalue-safety-margin 0.02 ^
  --predict-batch-size 8192 ^
  --topology-level lite ^
  --surface-topology-scope none ^
  --tau-max 1000 ^
  --min-persistence 0.5 ^
  --alert-patience 3 ^
  --out-dir outputs/fd001
```

Evidence mode semantics:
- `fixed_tau` (recommended): failure-time-coupled evidence with anytime tau diagnostics.
- `marginal_rul`: direct cumulative evidence over static RUL hypotheses \(r \in \{1,\dots,R_{\max}\}\); useful for topology-ablation analysis but typically more conservative on temporal RUL coverage.

For figures and full surface persistence, add:

```bash
--save-plots --surface-topology-scope plot_run
```

Or run everything in one command:

```bash
python scripts/run_full_pipeline.py --fd 1 --data-root data/rul_datasets --out-dir outputs/fd001
```

`run_full_pipeline.py` now runs: download -> train -> TEM -> audit (unless `--skip-audit`).
Add `--save-plots` only when figure artifacts are required.
`run_full_pipeline.py` executes stages in-process by default for speed; use `--use-subprocess` only if you need strict stage isolation.
Default pipeline calibration source is `val` for more stable finite-sample p-value behavior; override with `--calibration-source dev_holdout` when strict split disjointness is required.
`run_full_pipeline.py` also forwards `--calibration-bins` and `--calibration-min-bin-size` to both TEM and audit.

## 5. Synthetic Theorem Check

```bash
python scripts/run_synthetic_validation.py --lambda-bet 0.07 --out-dir outputs/synthetic
```

## 6. Audit (No-Leakage / Sanity)

```bash
python scripts/audit_tem.py ^
  --fd 1 ^
  --data-root data/rul_datasets ^
  --checkpoint outputs/fd001/model_fd001.pt ^
  --calibration outputs/fd001/calibration_residuals_fd001.npy ^
  --lambda-bet 0.07 ^
  --tau-max 1000 ^
  --pvalue-safety-margin 0.02 ^
  --cache outputs/fd001/audit_cache_fd001.npz ^
  --tem-metrics outputs/fd001/tem_metrics_fd001.json ^
  --out-dir outputs/fd001
```

`--cache` + `--tem-metrics` is the fast path (no model inference, no TEM recompute).

## 7. Deep Artifact Check

```bash
python scripts/deep_check_results.py --outputs-root outputs --report-path outputs/deep_check_report.json
```

Scans all stored TEM/audit/synthetic JSON artifacts and flags suspicious patterns (coverage failures, superuniformity failures, topology-metric gaps, weak null-vs-degraded separation).
Add `--flag-tau-identifiability-gap` if you want censored/capped tau-identifiability gaps to appear as low-severity findings.

## 8. Marginal-Evidence Topology

`run_tem_cmapss.py` now reports per-run:
- curve-level topology of marginal evidence \(V_t(r)\) (persistence and minima structure),
- ridge/path dynamics of \(r_t^\*=\arg\min_r V_t(r)\),
- 2D evidence-surface persistence summaries (sublevel + superlevel \(H_1\) features via GUDHI).

## 9. PhD Deep Inspect

```bash
python scripts/phd_deep_inspect.py ^
  --run-dirs outputs/fd001_phd_final outputs/fd002_phd_final ^
  --out-json outputs/phd_deep_inspect_report.json ^
  --out-md outputs/phd_deep_inspect_report.md
```

## 10. Ultra PhD Runner

Run strict diagnostics + ablation + consolidated report:

```bash
python scripts/run_phd_ultra.py --fds 1 2 3 4 --reuse ^
  --out-root outputs/phd_ultra ^
  --out-json outputs/phd_ultra_full_report.json ^
  --out-md outputs/phd_ultra_full_report.md
```

## 10b. Strict Per-FD Policy Runner

```bash
python scripts/run_full_pipeline_strict.py ^
  --fd 1 ^
  --out-dir outputs/fd001_strict ^
  --epochs 15 ^
  --seed 42 ^
  --calibration-source dev_holdout ^
  --surface-topology-scope all
```

`run_full_pipeline_strict.py` applies the strict per-FD calibration policy (margin + bins + min-bin-size) in a single pipeline pass.
Current strict defaults: FD001 `(0.08, 8, 128)`, FD002 `(0.11, 16, 128)`, FD003 `(0.18, 12, 128)`, FD004 `(0.25, 12, 128)`.
Strict/fair publication runners now default to `--lambda-bet 0.06` for improved finite-sample robustness.
Strict runner default calibration source is `dev_holdout` (split-disjoint).
It now forwards `--seed`, `--calibration-source`, and `--calibration-fraction` to training.

## 10c. Full Fair Matrix Runner

```bash
python scripts/run_full_fair_matrix.py --reuse ^
  --baseline-calibration-source dev_holdout ^
  --out-root outputs/full_fair_matrix ^
  --out-json outputs/full_fair_matrix_report.json ^
  --out-md outputs/full_fair_matrix_report.md
```

Use `--refresh-evidence-mode` to force rerunning only the evidence-mode branch while keeping other cached artifacts.

Runs a publication-style matrix:
- strict FD001-004 baseline (local),
- seed reproducibility,
- split robustness (`val` vs `dev_holdout`),
- evidence-mode / alpha / policy sweeps from fixed trained checkpoints,
- synthetic stress grid,
- deep checks + topology-vs-RUL landscape report.

`run_full_fair_matrix.py` can now auto-build derived artifacts after matrix completion:
- `outputs/baseline_comparison.json` (supports optional external baseline package),
- `outputs/external_dataset_summary.json` (supports optional real external performance package),
- `<out-root>/stats_conference_readiness.json` (strict conference readiness gate),
- `outputs/artifact_consistency_report.json` (cross-artifact metric consistency check).

Optional flags:
- `--external-baselines-json <path>`
- `--external-performance-report <path>`
- `--rul-dataset-summary <path>`
- `--paper-md <path>`
- `--skip-derived-artifacts`

## 10d. Baseline Comparator Artifact

```bash
python scripts/build_baseline_comparison.py ^
  --matrix-report outputs/publication_phd_final_report.json ^
  --out-json outputs/baseline_comparison.json ^
  --out-md outputs/baseline_comparison.md
```

Builds a paired comparator package from existing matrix runs (strict main vs internal alternatives) with run-level sign-test summaries.
Use `--external-baselines-json` to merge real external methods into this artifact:

```bash
python scripts/build_baseline_comparison.py ^
  --matrix-report outputs/publication_full_rtx4050_report.json ^
  --external-baselines-json outputs/templates/external_baselines_template.json ^
  --out-json outputs/baseline_comparison.json ^
  --out-md outputs/baseline_comparison.md
```

## 10e. External/Shift Summary Artifact

```bash
python scripts/build_external_dataset_summary.py ^
  --matrix-report outputs/publication_phd_final_report.json ^
  --out-json outputs/external_dataset_summary.json ^
  --out-md outputs/external_dataset_summary.md
```

Builds a concise generalization summary from the synthetic shift grid. For top-strength claims, add at least one real external dataset in addition to this stress summary.
Use `--external-performance-report` to attach real external model-evaluation metrics:

```bash
python scripts/build_external_dataset_summary.py ^
  --matrix-report outputs/publication_full_rtx4050_report.json ^
  --rul-dataset-summary outputs/rul_dataset_summary.json ^
  --external-performance-report outputs/templates/external_performance_template.json ^
  --out-json outputs/external_dataset_summary.json ^
  --out-md outputs/external_dataset_summary.md
```

## 10g. External Evidence Templates

```bash
python scripts/init_external_evidence_templates.py --fds 1 2 3 4 --out-dir outputs/templates
```

Creates:
- `outputs/templates/external_baselines_template.json`
- `outputs/templates/external_performance_template.json`

## 10h. Generated External Evidence (No Placeholders)

Generate real external model-evaluation rows on FEMTO/XJTU:

```bash
python scripts/run_real_external_evals.py ^
  --data-root C:\Users\micha\.rul-datasets ^
  --datasets femto,xjtu_sy ^
  --fd 1 ^
  --epochs 10 ^
  --batch-size 128 ^
  --num-workers 0 ^
  --no-compile ^
  --out-root outputs/external_real_eval ^
  --out-json outputs/external_performance_report.json ^
  --out-md outputs/external_performance_report.md
```

Use `--num-workers 0` on Windows if you hit Torch shared-event multiprocessing errors.
For terminal-life focused tuning, you can upweight low-RUL samples (optionally only for selected datasets):

```bash
python scripts/run_real_external_evals.py ^
  --data-root C:\Users\micha\.rul-datasets ^
  --datasets femto,xjtu_sy ^
  --fd 1 ^
  --epochs 10 ^
  --batch-size 128 ^
  --no-compile ^
  --num-workers 0 ^
  --alpha 0.001 ^
  --lambda-bet 0.03 ^
  --pvalue-safety-margin 0.3 ^
  --low-rul-loss-weight 3.0 ^
  --low-rul-threshold 25 ^
  --low-rul-weight-power 1.0 ^
  --low-rul-weight-datasets femto ^
  --out-root outputs/external_real_eval_final_policy_v2 ^
  --out-json outputs/external_performance_report.json ^
  --out-md outputs/external_performance_report.md
```

Generate two external comparator baselines from strict FD001-004 caches:

```bash
python scripts/build_external_baselines_from_strict.py ^
  --matrix-report outputs/publication_full_rtx4050_report.json ^
  --alpha 0.05 ^
  --out-json outputs/external_baselines_generated.json ^
  --out-md outputs/external_baselines_generated.md
```

Replay-only policy frontier sweep (frozen checkpoints/calibration bundles, no retraining):

```bash
python scripts/sweep_external_policy_replay.py ^
  --data-root C:\Users\micha\.rul-datasets ^
  --datasets femto,xjtu_sy,cmapss ^
  --reuse-artifacts-root outputs/external_real_eval_final_policy_v8 ^
  --alpha-grid 0.001,0.0015,0.002,0.003 ^
  --lambda-grid 0.03,0.04 ^
  --margin-grid 0.25,0.30 ^
  --out-root outputs/external_policy_replay_sweep_all_v1 ^
  --out-json outputs/external_policy_replay_sweep_all_v1/summary.json ^
  --out-md outputs/external_policy_replay_sweep_all_v1/summary.md
```

Plot the replay frontier:

```bash
python scripts/plot_policy_replay_frontier.py ^
  --summary-json outputs/external_policy_replay_sweep_all_v1/summary.json ^
  --out-png outputs/external_policy_replay_sweep_all_v1/frontier_cov_tau.png
```

Run selected robust replay point (example from sweep summary):

```bash
python scripts/run_real_external_evals.py ^
  --data-root C:\Users\micha\.rul-datasets ^
  --datasets femto,xjtu_sy,cmapss ^
  --fd 1 ^
  --num-workers 0 ^
  --epochs 0 ^
  --no-compile ^
  --reuse-artifacts-root outputs/external_real_eval_final_policy_v8 ^
  --alpha 0.003 ^
  --lambda-bet 0.04 ^
  --pvalue-safety-margin 0.30 ^
  --out-root outputs/external_real_eval_policy_replay_robust_v1 ^
  --out-json outputs/external_performance_report_policy_replay_robust_v1.json ^
  --out-md outputs/external_performance_report_policy_replay_robust_v1.md
```

Generate seed-ensemble interval baselines and merge with conformal baselines:

```bash
python scripts/build_seed_ensemble_baselines.py ^
  --matrix-report outputs/publication_full_rtx4050_report.json ^
  --out-json outputs/external_seed_ensemble_baselines.json ^
  --out-md outputs/external_seed_ensemble_baselines.md

python scripts/merge_external_baseline_packages.py ^
  --inputs outputs/external_baselines_generated.json,outputs/external_seed_ensemble_baselines.json ^
  --out-json outputs/external_baselines_merged.json ^
  --out-md outputs/external_baselines_merged.md

python scripts/build_baseline_comparison.py ^
  --matrix-report outputs/publication_full_rtx4050_report.json ^
  --external-baselines-json outputs/external_baselines_merged.json ^
  --out-json outputs/baseline_comparison.json ^
  --out-md outputs/baseline_comparison.md
```

Build claim-level significance and sharpness reports:

```bash
python scripts/build_claim_significance_report.py ^
  --baseline-json outputs/baseline_comparison.json ^
  --policy-sweep-json outputs/external_policy_replay_sweep_all_v1/summary.json ^
  --out-json outputs/publication_full_rtx4050/claim_significance_report.json ^
  --out-md outputs/publication_full_rtx4050/claim_significance_report.md

python scripts/build_policy_sharpness_report.py ^
  --canonical outputs/external_performance_report.json ^
  --balanced outputs/external_performance_report_policy_replay_balanced_v2.json ^
  --aggressive outputs/external_performance_report_policy_replay_aggressive_v1.json ^
  --robust outputs/external_performance_report_policy_replay_robust_v1.json ^
  --out-json outputs/publication_full_rtx4050/policy_sharpness_report.json ^
  --out-md outputs/publication_full_rtx4050/policy_sharpness_report.md ^
  --out-fig outputs/publication_full_rtx4050/policy_sharpness_frontier.png
```

Optional true-retrain robustness block (no reuse):

```bash
python scripts/run_real_external_evals.py ^
  --data-root C:\Users\micha\.rul-datasets ^
  --datasets femto,xjtu_sy,cmapss ^
  --fd 1 ^
  --epochs 15 ^
  --batch-size 128 ^
  --num-workers 0 ^
  --seed 123 ^
  --no-compile ^
  --dataset-overrides-json outputs/external_dataset_overrides.json ^
  --alpha 0.003 ^
  --lambda-bet 0.04 ^
  --pvalue-safety-margin 0.30 ^
  --out-root outputs/external_real_eval_retrain_robustness_v2 ^
  --out-json outputs/external_performance_report_retrain_robustness_v2.json ^
  --out-md outputs/external_performance_report_retrain_robustness_v2.md
```

Width-aware replay sweep on the fresh retrained external artifacts:

```bash
python scripts/sweep_external_policy_replay.py ^
  --data-root C:\Users\micha\.rul-datasets ^
  --datasets femto,xjtu_sy,cmapss ^
  --fd 1 ^
  --batch-size 128 ^
  --num-workers 0 ^
  --reuse-artifacts-root outputs/external_real_eval_retrain_robustness_v2 ^
  --alpha-grid 0.001,0.002,0.003,0.005 ^
  --lambda-grid 0.03,0.04,0.05 ^
  --margin-grid 0.20,0.25,0.30 ^
  --cov-target 0.95 ^
  --tau-target 0.05 ^
  --out-root outputs/external_policy_replay_sweep_retrain_v2 ^
  --out-json outputs/external_policy_replay_sweep_retrain_v2/summary.json ^
  --out-md outputs/external_policy_replay_sweep_retrain_v2/summary.md
```

Freeze submission artifact checksums:

```bash
python scripts/build_submission_freeze_manifest.py ^
  --out-json outputs/publication_full_rtx4050/submission_freeze_manifest.json
```

## 10i. Cross-Artifact Consistency Check

```bash
python scripts/check_artifact_consistency.py ^
  --external-performance-report outputs/external_performance_report.json ^
  --external-dataset-summary outputs/external_dataset_summary.json ^
  --baseline-comparison outputs/baseline_comparison.json ^
  --out-json outputs/artifact_consistency_report.json ^
  --out-md outputs/artifact_consistency_report.md ^
  --fail-on-mismatch
```

## 10f. Real RUL Dataset Summary (rul_datasets Readers)

```bash
python scripts/summarize_rul_datasets.py ^
  --data-root C:\Users\micha\.rul-datasets ^
  --datasets cmapss,femto,xjtu_sy ^
  --fd 1 ^
  --rebuild-scalers ^
  --out-json outputs/rul_dataset_summary.json ^
  --out-md outputs/rul_dataset_summary.md
```

This uses `rul_datasets` reader classes directly to summarize real dataset splits. You can append `--include-ncmapss` to attempt N-CMAPSS as well.
Use `--rebuild-scalers` whenever the local `scikit-learn` version changed, so cached reader scalers are re-fit and no pickle-version warnings leak into runs.

## 11. Topology vs RUL Analysis

```bash
python scripts/analyze_topology_rul_landscape.py ^
  --run-dirs outputs/full_fair_matrix/strict_main/fd001 outputs/full_fair_matrix/strict_main/fd002 ^
            outputs/full_fair_matrix/strict_main/fd003 outputs/full_fair_matrix/strict_main/fd004 ^
  --out-json outputs/full_fair_matrix/topology_rul_landscape.json ^
  --out-md outputs/full_fair_matrix/topology_rul_landscape.md ^
  --fig-dir outputs/full_fair_matrix/topology_rul_figs
```

Produces step-level topology-vs-RUL summaries, run-level topology/quality associations, and publication figures.

## 12. Stats Conference Readiness Gate

```bash
python scripts/stats_conference_readiness.py ^
  --report-json outputs/publication_phd_final_report.json ^
  --topology-json outputs/publication_phd_final/topology_rul_landscape.json ^
  --paper-md paper/topological_evidence_curves.md ^
  --out-json outputs/publication_phd_final/stats_conference_readiness.json ^
  --out-md outputs/publication_phd_final/stats_conference_readiness.md
```

Generates a weighted 10-point readiness score and concrete gap list for a 9+ stats-conference submission (validity, robustness, topology signal, baseline comparisons, external generalization, and proof maturity).

Tau-identifiability now supports a pooled+censoring-aware gate:
- `--tau-identifiability-deficit-tolerance` (default `0.03`)
- `--max-tau-identifiability-severe-fails` (default `0`)

## Output Artifacts

- `outputs/fdXXX/model_fdXXX.pt`: trained model checkpoint
- `outputs/fdXXX/train_metrics_fdXXX.json`: training + test metrics
- `outputs/fdXXX/tem_metrics_fdXXX.json`: TEM aggregate metrics
- `outputs/fdXXX/audit_fdXXX.json`: leakage + p-value sanity diagnostics
- `outputs/fdXXX/audit_cache_fdXXX.npz`: flattened TEM predictions/targets cache for fast audit reruns
- `outputs/fdXXX/tem_run_<id>_*.png`: evidence/topology trajectories
- `paper/topological_evidence_curves.md`: manuscript draft

## Leakage Hygiene

- Recommended setup is `--calibration-source dev_holdout` to keep calibration runs disjoint from model fitting data.
- Default pipeline setup is `--calibration-source val` to reduce practical superuniformity drift on FD001/FD002 under current split behavior.
- Strict publication runners (`run_full_pipeline_strict.py`, `run_full_fair_matrix.py` baseline) default to `dev_holdout`.
- If you require strict calibration/training disjointness, use `--calibration-source dev_holdout` and consider increasing `--pvalue-safety-margin`.
- For anytime-valid coverage over full lifecycle, keep `--healthy-rul-floor 1` (default). Higher floors intentionally restrict calibration to healthy-prefix data and can invalidate full-lifecycle coverage.
- `--tau-max` is fixed by configuration (default 1000), not inferred from test-run lengths.
- Tau diagnostics are computed only for engines where true failure horizon is identifiable from observed/capped RUL labels; see `num_tau_diagnostics_engines` in TEM metrics.
- `--pvalue-safety-margin` defaults to `0.02` for conservative hedge against calibration/test residual shift; set `0.0` to disable.

## RTX 4050 Speed Notes

- Keep `--batch-size` in the 512-2048 range (depends on window size and model depth).
- AMP is on by default.
- `torch.compile` is intentionally off by default because it requires a working Triton install.
- Use `--min-persistence` (e.g., `0.5`) to avoid false alerts from tiny topological fluctuations.
- TEM speed defaults are now tuned for throughput: `--predict-batch-size 8192`, `--topology-level lite`, no plots, and `--surface-topology-scope none`.
- Use `--save-plots` and `--surface-topology-scope plot_run` only when you need publication figures.
- `run_full_pipeline.py` automatically wires TEM cache + TEM metrics into `audit_tem.py`; this removes duplicate inference and duplicate TEM in audit.
- If `numba` is installed, persistence kernel acceleration is used automatically. Set env var `TEM_DISABLE_NUMBA=1` to force pure-NumPy mode.

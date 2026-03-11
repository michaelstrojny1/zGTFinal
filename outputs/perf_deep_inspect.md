# Performance Deep Inspect

## TEM Runtime (FD001, same checkpoint/calibration)
- cProfile total before optimizations: 52.60s
- cProfile total after structural optimizations (default): 23.35s
- cProfile total after speed-mode defaults: 22.36s
- Speedup (before -> after default): 2.25x
- Speedup (before -> after fast): 2.35x

## Validity Regression Check
- Core fleet/per-run metrics are identical between baseline final outputs and accelerated TEM outputs for FD001/FD002.
- Report: `outputs/perf_regression_check.json`

## Fixed Bottlenecks
- Batched run inference path in `predict_runs` (concatenate all run windows once, split predictions after).
- Shared `ConditionalResidualCalibrator` reused across engines (removed per-engine rebuild/sort overhead).
- `topology_level=lite` path reuses `gamma/max_p/second_p` from TEM core and avoids redundant per-step persistence recomputation.
- Surface topology now scoped (`none|plot_run|all`) and disabled by default for throughput.
- Full pipeline now runs stages in-process by default, avoiding repeated interpreter/import startup.

## Remaining Dominant Cost
- Import overhead from `rul_datasets` dependency chain (`pytorch_lightning -> torchmetrics -> transformers`) is still large per process.
- Per-step `persistence_summary` in TEM core remains nontrivial; exact 1D H0 persistence is still computed every monitoring step.

## Next Massive Opportunities
1. Replace `rul_datasets` runtime import path with a lightweight local C-MAPSS reader to eliminate heavy package import cascade.
2. Optional numba-accelerated backend for `h0_persistence_1d` to reduce persistence compute cost in long trajectories.
3. Long-lived runner daemon for repeated sweeps (single process + cached data/model) to amortize import and data prep costs.


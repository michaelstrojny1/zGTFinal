# Speed + Integrity Deep Pass (2026-03-04)

## Runtime Benchmarks

- Full pipeline (FD001, epochs=1) with cache-wired audit: **33.88 s**
- Legacy-style pipeline (same config, audit recompute): **47.106 s**
- Net pipeline gain: **13.226 s saved** (**1.39x faster**)

- Audit recompute mode: **7.63 s**
- Audit cache+TEM-metrics mode: **0.208 s**
- Net audit gain: **36.68x faster**

- CMAPSS split cache miss (`window_size=31`): **10.5356 s**
- CMAPSS split cache hit (`window_size=31`): **0.1941 s**
- Net data-load gain: **54.28x faster**

- TEM run (FD001, lite, no plots): **8.401 s**

## Integrity / Suspicious-Result Checks

- `outputs/_pipe_cache_fast_bench`: 3 findings (high+medium p-value superuniformity warnings, low tau-identifiability gap).
- `outputs/_pub_fd001_e5`: 3 findings (same pattern).
- `outputs/_pub_fd001_valcal`: **1 finding only** (low tau-identifiability gap); superuniformity warnings resolved.

## Topology Consistency

- Fixed a lite-mode topology issue where `mean_persistent_valleys` was undercounted (0/1 only).
- Lite mode now uses top-2 persistence lower-bound approximation (`curve_backend=lite_precomputed_top2`).
- Core TEM behavior unchanged (fleet summary + per-run alert/coverage trajectories unchanged).

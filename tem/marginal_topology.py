from __future__ import annotations

import numpy as np

from tem.topology import h0_persistence_1d, persistence_summary


def _count_local_minima_2d(curves: np.ndarray) -> np.ndarray:
    """
    Count strict/weak local minima per row on a 1D chain.
    curves: [T, R]
    """
    arr = np.asarray(curves, dtype=np.float64)
    t_steps, r_max = arr.shape
    if t_steps == 0:
        return np.zeros(0, dtype=np.float64)
    if r_max == 1:
        return np.ones(t_steps, dtype=np.float64)

    count = np.zeros(t_steps, dtype=np.float64)
    count += (arr[:, 0] < arr[:, 1]).astype(np.float64)
    count += (arr[:, -1] < arr[:, -2]).astype(np.float64)
    mid = arr[:, 1:-1]
    left = arr[:, :-2]
    right = arr[:, 2:]
    minima_mask = (mid <= left) & (mid <= right) & ((mid < left) | (mid < right))
    count += np.sum(minima_mask, axis=1, dtype=np.int64)
    return count


def analyze_marginal_evidence_topology(
    log_k_hist: np.ndarray,
    min_persistence: float,
    max_p_hist: np.ndarray | None = None,
    second_p_hist: np.ndarray | None = None,
    gamma_hist: np.ndarray | None = None,
    topology_level: str = "lite",
    compute_surface: bool = True,
) -> dict:
    """
    Analyze topology of marginal evidence over candidate RUL values.
    Input shape is [time, r], where each row is V_t(r).

    `topology_level`:
      - "lite": reuse provided persistence trajectories if available (fast).
      - "full": recompute persistence lifetimes per step (slow).
    """
    arr = np.asarray(log_k_hist, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("log_k_hist must be 2D [time, rul].")
    if topology_level not in {"lite", "full"}:
        raise ValueError("topology_level must be 'lite' or 'full'.")

    t_steps, r_max = arr.shape
    if t_steps == 0 or r_max == 0:
        if compute_surface:
            from tem.surface_topology import surface_persistence_features

            surface = surface_persistence_features(arr, min_persistence=min_persistence)
        else:
            surface = {"backend": "skipped"}
        return {
            "n_steps": int(t_steps),
            "r_max": int(r_max),
            "curve_backend": "empty",
            "curve": {
                "mean_max_h0": 0.0,
                "mean_second_h0": 0.0,
                "mean_gamma": 1.0,
                "max_gamma": 1.0,
                "mean_persistent_valleys": 0.0,
                "mean_local_minima": 0.0,
            },
            "ridge": {
                "r_star_start": -1,
                "r_star_end": -1,
                "total_variation_l1": 0.0,
                "mean_jump": 0.0,
                "p90_jump": 0.0,
            },
            "surface": surface,
        }

    local_min_count_hist = _count_local_minima_2d(arr)
    curve_backend = "full_h0"

    if topology_level == "lite" and max_p_hist is not None and second_p_hist is not None and gamma_hist is not None:
        max_p = np.asarray(max_p_hist, dtype=np.float64).reshape(-1)
        second_p = np.asarray(second_p_hist, dtype=np.float64).reshape(-1)
        gamma = np.asarray(gamma_hist, dtype=np.float64).reshape(-1)
        if max_p.shape[0] != t_steps or second_p.shape[0] != t_steps or gamma.shape[0] != t_steps:
            raise ValueError("Provided persistence trajectories must match time dimension of log_k_hist.")
        # Fast approximation: lower-bound count from the top-2 persistence lifetimes.
        persistent_count_hist = (max_p >= float(min_persistence)).astype(np.float64)
        persistent_count_hist += (second_p >= float(min_persistence)).astype(np.float64)
        curve_backend = "lite_precomputed_top2"
    else:
        max_p = np.empty(t_steps, dtype=np.float64)
        second_p = np.empty(t_steps, dtype=np.float64)
        gamma = np.empty(t_steps, dtype=np.float64)
        persistent_count_hist = np.empty(t_steps, dtype=np.float64)
        for i in range(t_steps):
            curve = arr[i]
            m1, m2, g = persistence_summary(curve)
            lifetimes = h0_persistence_1d(curve)
            max_p[i] = m1
            second_p[i] = m2
            gamma[i] = g
            persistent_count_hist[i] = float(np.sum(lifetimes >= min_persistence))

    r_star = np.argmin(arr, axis=1).astype(np.int64) + 1
    if t_steps > 1:
        jumps = np.abs(np.diff(r_star).astype(np.float64))
        total_variation = float(np.sum(jumps))
        mean_jump = float(np.mean(jumps))
        p90_jump = float(np.quantile(jumps, 0.9))
    else:
        total_variation = 0.0
        mean_jump = 0.0
        p90_jump = 0.0

    if compute_surface:
        from tem.surface_topology import surface_persistence_features

        surface = surface_persistence_features(arr, min_persistence=min_persistence)
    else:
        surface = {"backend": "skipped"}

    return {
        "n_steps": int(t_steps),
        "r_max": int(r_max),
        "curve_backend": curve_backend,
        "curve": {
            "mean_max_h0": float(np.mean(max_p)),
            "mean_second_h0": float(np.mean(second_p)),
            "mean_gamma": float(np.mean(gamma)),
            "max_gamma": float(np.max(gamma)),
            "mean_persistent_valleys": float(np.mean(persistent_count_hist)),
            "mean_local_minima": float(np.mean(local_min_count_hist)),
        },
        "ridge": {
            "r_star_start": int(r_star[0]),
            "r_star_end": int(r_star[-1]),
            "total_variation_l1": total_variation,
            "mean_jump": mean_jump,
            "p90_jump": p90_jump,
        },
        "surface": surface,
    }

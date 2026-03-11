from __future__ import annotations

import numpy as np

try:
    import gudhi as gd
except Exception:  # pragma: no cover
    gd = None


def _safe_mean_abs_diff(arr: np.ndarray, axis: int) -> float:
    if arr.shape[axis] <= 1:
        return 0.0
    return float(np.mean(np.abs(np.diff(arr, axis=axis))))


def _lifetimes_from_persistence(pairs: list[tuple[int, tuple[float, float]]], dim: int) -> np.ndarray:
    out = []
    for d, (birth, death) in pairs:
        if d != dim:
            continue
        if np.isfinite(death):
            out.append(float(death - birth))
    if not out:
        return np.zeros(0, dtype=np.float64)
    arr = np.asarray(out, dtype=np.float64)
    arr.sort()
    return arr


def _gudhi_surface_features(arr: np.ndarray, min_persistence: float) -> dict[str, float | int]:
    cc = gd.CubicalComplex(top_dimensional_cells=arr)  # type: ignore[union-attr]
    pairs = cc.persistence(min_persistence=min_persistence)
    h0 = _lifetimes_from_persistence(pairs, dim=0)
    h1 = _lifetimes_from_persistence(pairs, dim=1)
    return {
        "max_h0_persistence": float(h0[-1]) if h0.size else 0.0,
        "max_h1_persistence": float(h1[-1]) if h1.size else 0.0,
        "num_h1_above_min": int(np.sum(h1 >= min_persistence)) if h1.size else 0,
    }


def surface_persistence_features(
    surface: np.ndarray,
    min_persistence: float = 0.0,
) -> dict:
    """
    Topological features of a 2D evidence surface over (time, RUL).
    Uses cubical persistence if GUDHI is available.
    """
    arr = np.asarray(surface, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("surface must be 2D.")

    roughness = _safe_mean_abs_diff(arr, axis=0) + _safe_mean_abs_diff(arr, axis=1)

    if gd is None:
        # Lightweight fallback if GUDHI is unavailable.
        return {
            "backend": "fallback",
            "max_h0_persistence": 0.0,
            "max_h1_persistence": 0.0,
            "num_h1_above_min": 0,
            "sublevel_max_h1_persistence": 0.0,
            "superlevel_max_h1_persistence": 0.0,
            "sublevel_num_h1_above_min": 0,
            "superlevel_num_h1_above_min": 0,
            "surface_roughness": roughness,
        }

    sub = _gudhi_surface_features(arr, min_persistence=min_persistence)
    sup = _gudhi_surface_features(-arr, min_persistence=min_persistence)
    max_h1 = max(float(sub["max_h1_persistence"]), float(sup["max_h1_persistence"]))
    num_h1 = max(int(sub["num_h1_above_min"]), int(sup["num_h1_above_min"]))

    return {
        "backend": "gudhi",
        "max_h0_persistence": float(sub["max_h0_persistence"]),
        "max_h1_persistence": max_h1,
        "num_h1_above_min": num_h1,
        "sublevel_max_h1_persistence": float(sub["max_h1_persistence"]),
        "superlevel_max_h1_persistence": float(sup["max_h1_persistence"]),
        "sublevel_num_h1_above_min": int(sub["num_h1_above_min"]),
        "superlevel_num_h1_above_min": int(sup["num_h1_above_min"]),
        "surface_roughness": roughness,
    }

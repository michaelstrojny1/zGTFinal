from __future__ import annotations

import os

import numpy as np

try:
    from numba import njit

    _NUMBA_AVAILABLE = True
except Exception:
    njit = None  # type: ignore[assignment]
    _NUMBA_AVAILABLE = False

_NUMBA_ENABLED = _NUMBA_AVAILABLE and os.environ.get("TEM_DISABLE_NUMBA", "0") != "1"


if _NUMBA_ENABLED:
    @njit(cache=True)
    def _find_numba(parent: np.ndarray, x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x


    @njit(cache=True)
    def _h0_persistence_1d_numba(vals: np.ndarray) -> np.ndarray:
        n = vals.shape[0]
        if n <= 1:
            return np.zeros(0, dtype=np.float64)
        order = np.argsort(vals)
        parent = np.empty(n, dtype=np.int64)
        parent.fill(-1)
        birth = np.empty(n, dtype=np.float64)
        active = np.zeros(n, dtype=np.uint8)
        lifetimes = np.empty(max(0, n - 1), dtype=np.float64)
        n_life = 0

        for k in range(n):
            idx = int(order[k])
            active[idx] = 1
            parent[idx] = idx
            birth[idx] = vals[idx]

            for nb in (idx - 1, idx + 1):
                if nb < 0 or nb >= n or active[nb] == 0:
                    continue
                ra = _find_numba(parent, idx)
                rb = _find_numba(parent, nb)
                if ra == rb:
                    continue

                if birth[ra] <= birth[rb]:
                    older = ra
                    younger = rb
                else:
                    older = rb
                    younger = ra
                lifetimes[n_life] = vals[idx] - birth[younger]
                n_life += 1
                parent[younger] = older

        return lifetimes[:n_life]


def h0_persistence_1d(values: np.ndarray) -> np.ndarray:
    """
    Compute 0D persistence lifetimes for a 1D function using a lower-star filtration.
    Domain adjacency is 1D chain connectivity.
    """
    vals = np.asarray(values, dtype=np.float64)
    if _NUMBA_ENABLED:
        return _h0_persistence_1d_numba(vals)

    n = vals.shape[0]
    order = np.argsort(vals, kind="mergesort")

    parent = np.full(n, -1, dtype=np.int32)
    birth = np.zeros(n, dtype=np.float64)
    active = np.zeros(n, dtype=bool)
    lifetimes: list[float] = []

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for idx in order:
        active[idx] = True
        parent[idx] = idx
        birth[idx] = vals[idx]

        for nb in (idx - 1, idx + 1):
            if nb < 0 or nb >= n or not active[nb]:
                continue
            ra = find(idx)
            rb = find(nb)
            if ra == rb:
                continue

            # Younger component (higher birth) dies at current filtration value.
            if birth[ra] <= birth[rb]:
                older, younger = ra, rb
            else:
                older, younger = rb, ra

            lifetimes.append(float(vals[idx] - birth[younger]))
            parent[younger] = older

    if not lifetimes:
        return np.zeros(0, dtype=np.float64)
    return np.asarray(lifetimes, dtype=np.float64)


def persistence_summary(values: np.ndarray) -> tuple[float, float, float]:
    lifetimes = h0_persistence_1d(values)
    if lifetimes.size == 0:
        return 0.0, 0.0, 1.0
    if lifetimes.size == 1:
        max_p = float(lifetimes[0])
        second = 0.0
    else:
        top2 = np.partition(lifetimes, lifetimes.size - 2)[-2:]
        max_p = float(np.max(top2))
        second = float(np.min(top2))
    # Gamma is only informative as a signal-to-noise ratio if there are at least
    # two persistent features. Otherwise, treat it as neutral.
    if lifetimes.size < 2 or second <= 1e-8:
        gamma = 1.0
    else:
        gamma = max_p / second
    return max_p, second, gamma

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _global_p_values(scores: np.ndarray, sorted_residuals: np.ndarray) -> np.ndarray:
    n = sorted_residuals.shape[0]
    idx = np.searchsorted(sorted_residuals, scores, side="left")
    count_ge = n - idx
    return (1.0 + count_ge.astype(np.float64)) / (n + 1.0)


@dataclass
class ConditionalResidualCalibrator:
    """
    Compute conformal p-values from residual calibration data with optional
    conditioning on implied RUL bins.
    """

    global_sorted_residuals: np.ndarray
    bin_edges: np.ndarray
    per_bin_sorted_residuals: list[np.ndarray]
    min_bin_size: int
    r_max: int
    pvalue_safety_margin: float = 0.0
    rul_bin_lookup: np.ndarray | None = None

    @classmethod
    def from_arrays(
        cls,
        residuals: np.ndarray,
        true_rul: np.ndarray | None,
        r_max: int,
        n_bins: int = 8,
        min_bin_size: int = 128,
        pvalue_safety_margin: float = 0.0,
    ) -> "ConditionalResidualCalibrator":
        res = np.asarray(residuals, dtype=np.float64).reshape(-1)
        if res.size == 0:
            raise ValueError("Calibration residuals must not be empty.")
        global_sorted = np.sort(res)
        margin = float(np.clip(pvalue_safety_margin, 0.0, 1.0))

        if true_rul is None:
            # Single global bin fallback.
            return cls(
                global_sorted_residuals=global_sorted,
                bin_edges=np.asarray([1.0, float(r_max) + 1.0], dtype=np.float64),
                per_bin_sorted_residuals=[global_sorted],
                min_bin_size=min_bin_size,
                r_max=r_max,
                pvalue_safety_margin=margin,
                rul_bin_lookup=np.zeros(max(1, int(r_max)), dtype=np.int16),
            )

        rul = np.asarray(true_rul, dtype=np.float64).reshape(-1)
        if rul.shape[0] != res.shape[0]:
            raise ValueError("true_rul and residuals must have equal length.")
        rul = np.clip(rul, 1.0, float(r_max))

        n_bins = int(max(1, n_bins))
        # Quantile binning for roughly balanced conditional calibration bins.
        quantiles = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.quantile(rul, quantiles)
        edges[0] = 1.0
        edges[-1] = float(r_max) + 1.0
        # Ensure strictly increasing edges.
        edges = np.maximum.accumulate(edges)
        for i in range(1, edges.shape[0]):
            if edges[i] <= edges[i - 1]:
                edges[i] = edges[i - 1] + 1e-6

        per_bin: list[np.ndarray] = []
        for b in range(n_bins):
            lo, hi = edges[b], edges[b + 1]
            mask = (rul >= lo) & (rul < hi if b < n_bins - 1 else rul <= hi)
            vals = np.sort(res[mask])
            if vals.size < min_bin_size:
                vals = global_sorted
            per_bin.append(vals)

        rul_points = np.arange(1, int(r_max) + 1, dtype=np.float64)
        rul_lookup = np.searchsorted(edges, rul_points, side="right") - 1
        rul_lookup = np.clip(rul_lookup, 0, len(per_bin) - 1).astype(np.int16, copy=False)

        return cls(
            global_sorted_residuals=global_sorted,
            bin_edges=edges,
            per_bin_sorted_residuals=per_bin,
            min_bin_size=min_bin_size,
            r_max=r_max,
            pvalue_safety_margin=margin,
            rul_bin_lookup=rul_lookup,
        )

    def _apply_margin(self, pvals: np.ndarray) -> np.ndarray:
        if self.pvalue_safety_margin <= 0.0:
            return pvals
        return np.minimum(1.0, pvals + self.pvalue_safety_margin)

    def bin_ids_from_implied_rul(self, implied_rul: np.ndarray) -> np.ndarray:
        rul = np.asarray(implied_rul, dtype=np.float64).reshape(-1)
        if rul.size == 0:
            return np.zeros(0, dtype=np.int64)
        clipped = np.clip(rul, 1.0, float(self.r_max))

        if self.rul_bin_lookup is not None:
            clipped_int = np.rint(clipped).astype(np.int64)
            if np.all(np.abs(clipped - clipped_int.astype(np.float64)) <= 1e-8):
                idx = np.clip(clipped_int, 1, int(self.r_max)) - 1
                return self.rul_bin_lookup[idx].astype(np.int64, copy=False)

        ids = np.searchsorted(self.bin_edges, clipped, side="right") - 1
        ids = np.clip(ids, 0, len(self.per_bin_sorted_residuals) - 1)
        return ids.astype(np.int64, copy=False)

    def p_values_with_bin_ids(self, scores: np.ndarray, bin_ids: np.ndarray) -> np.ndarray:
        sc = np.asarray(scores, dtype=np.float64).reshape(-1)
        if len(self.per_bin_sorted_residuals) == 1:
            return self._apply_margin(_global_p_values(sc, self.global_sorted_residuals))

        ids = np.asarray(bin_ids, dtype=np.int64).reshape(-1)
        if ids.shape[0] != sc.shape[0]:
            raise ValueError("scores and bin_ids must have equal length.")
        ids = np.clip(ids, 0, len(self.per_bin_sorted_residuals) - 1)
        out = np.empty_like(sc, dtype=np.float64)
        if sc.size == 0:
            return out

        # TEM uses monotone implied-RUL grids; segmented slices avoid repeated mask allocations.
        if ids.size <= 1 or bool(np.all(ids[:-1] <= ids[1:])):
            boundaries = np.flatnonzero(np.diff(ids)) + 1
            starts = np.concatenate(([0], boundaries))
            ends = np.concatenate((boundaries, [ids.size]))
            for s, e in zip(starts.tolist(), ends.tolist()):
                b = int(ids[s])
                out[s:e] = _global_p_values(sc[s:e], self.per_bin_sorted_residuals[b])
        else:
            for b in np.unique(ids):
                mask = ids == b
                out[mask] = _global_p_values(sc[mask], self.per_bin_sorted_residuals[int(b)])
        return self._apply_margin(out)

    def p_values(self, scores: np.ndarray, implied_rul: np.ndarray | None = None) -> np.ndarray:
        sc = np.asarray(scores, dtype=np.float64).reshape(-1)
        if implied_rul is None or len(self.per_bin_sorted_residuals) == 1:
            return self._apply_margin(_global_p_values(sc, self.global_sorted_residuals))

        rul = np.asarray(implied_rul, dtype=np.float64).reshape(-1)
        if rul.shape[0] != sc.shape[0]:
            raise ValueError("scores and implied_rul must have equal length.")
        ids = self.bin_ids_from_implied_rul(rul)
        return self.p_values_with_bin_ids(sc, ids)

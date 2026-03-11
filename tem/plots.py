from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


def plot_evidence_snapshots(
    log_k_hist: np.ndarray,
    true_rul: np.ndarray,
    out_path: str | Path,
    quantiles: Iterable[float] = (0.25, 0.5, 0.75, 0.9),
) -> None:
    hist = np.asarray(log_k_hist, dtype=np.float64)
    t_steps, r_max = hist.shape
    q = list(quantiles)
    idxs = [min(t_steps - 1, max(0, int(t_steps * f) - 1)) for f in q]
    r_grid = np.arange(1, r_max + 1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=140)
    for ax, t in zip(axes.ravel(), idxs):
        ax.plot(r_grid, hist[t], lw=2)
        tr = int(np.clip(round(true_rul[t]), 1, r_max))
        ax.axvline(tr, color="tab:red", lw=1.5, linestyle="--", label=f"true RUL={tr}")
        ax.set_title(f"t={t + 1}")
        ax.set_xlabel("Hypothesized RUL")
        ax.set_ylabel("log evidence")
        ax.grid(alpha=0.2)
        ax.legend(loc="best")

    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def plot_tem_trajectories(
    gamma_hist: np.ndarray,
    width_hist: np.ndarray,
    r_star_hist: np.ndarray,
    true_rul: np.ndarray,
    out_path: str | Path,
) -> None:
    t = np.arange(1, gamma_hist.shape[0] + 1)
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), dpi=140, sharex=True)

    axes[0].plot(t, gamma_hist, color="tab:blue", lw=1.8)
    axes[0].set_ylabel("gamma_t")
    axes[0].grid(alpha=0.2)

    axes[1].plot(t, width_hist, color="tab:orange", lw=1.8)
    axes[1].set_ylabel("W_t^alpha")
    axes[1].grid(alpha=0.2)

    axes[2].plot(t, r_star_hist, label="r*_t", color="tab:green", lw=1.8)
    axes[2].plot(t, true_rul, label="true RUL", color="tab:red", lw=1.2, linestyle="--")
    axes[2].set_ylabel("RUL")
    axes[2].set_xlabel("Monitoring step")
    axes[2].grid(alpha=0.2)
    axes[2].legend(loc="best")

    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def plot_evidence_surface(
    log_k_hist: np.ndarray,
    out_path: str | Path,
) -> None:
    arr = np.asarray(log_k_hist, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(10, 5), dpi=140)
    im = ax.imshow(arr.T, aspect="auto", origin="lower", interpolation="nearest")
    ax.set_xlabel("Monitoring step")
    ax.set_ylabel("Hypothesized RUL")
    ax.set_title("Evidence Surface V_t(r)")
    fig.colorbar(im, ax=ax, label="log evidence")
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)

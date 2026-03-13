from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot external policy replay frontier from sweep summary JSON.")
    p.add_argument("--summary-json", type=str, default="outputs/external_policy_replay_sweep_all_v1/summary.json")
    p.add_argument("--out-png", type=str, default="outputs/external_policy_replay_sweep_all_v1/frontier_cov_tau.png")
    return p.parse_args()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    summary = _load(Path(args.summary_json).resolve())
    rows = list(summary.get("rows_sorted", []))
    if not rows:
        raise ValueError("No rows found in summary.")

    cov_target = float(summary.get("settings", {}).get("cov_target", 0.95))
    tau_target = float(summary.get("settings", {}).get("tau_target", 0.05))
    best = dict(summary.get("best_policy", {}))

    cov = np.asarray([float(r.get("cov_min", np.nan)) for r in rows], dtype=np.float64)
    tau = np.asarray([float(r.get("tau_max", np.nan)) for r in rows], dtype=np.float64)
    agg = np.asarray([float(r.get("aggressiveness", np.nan)) for r in rows], dtype=np.float64)
    ok = np.asarray([bool(r.get("validity_ok", False)) for r in rows], dtype=bool)

    fig, ax = plt.subplots(figsize=(7.4, 5.4), dpi=140)

    # Non-valid policies in light red for easy review; valid policies colored by aggressiveness.
    ax.scatter(cov[~ok], tau[~ok], s=45, color="#d95f5f", alpha=0.75, edgecolor="none", label="invalid")
    sc = ax.scatter(
        cov[ok],
        tau[ok],
        c=agg[ok] if np.any(ok) else None,
        s=55,
        cmap="viridis",
        alpha=0.95,
        edgecolor="black",
        linewidth=0.25,
        label="valid",
    )
    if np.any(ok):
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("Aggressiveness score")

    if best:
        ax.scatter(
            [float(best.get("cov_min", np.nan))],
            [float(best.get("tau_max", np.nan))],
            marker="*",
            s=240,
            color="#ffcc00",
            edgecolor="black",
            linewidth=0.8,
            zorder=8,
            label="selected",
        )

    ax.axvline(cov_target, color="#2f4f4f", linestyle="--", linewidth=1.2, label=f"cov target={cov_target:.2f}")
    ax.axhline(tau_target, color="#8b0000", linestyle="--", linewidth=1.2, label=f"tau target={tau_target:.2f}")

    ax.set_xlabel("Minimum dataset RUL coverage")
    ax.set_ylabel("Maximum dataset tau violation")
    ax.set_title("External Policy Replay Frontier (All Datasets)")
    ax.set_xlim(max(0.0, float(np.nanmin(cov)) - 0.02), min(1.02, float(np.nanmax(cov)) + 0.02))
    ax.set_ylim(max(0.0, float(np.nanmin(tau)) - 0.02), min(1.02, float(np.nanmax(tau)) + 0.02))
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", framealpha=0.9)

    out_png = Path(args.out_png).resolve()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {out_png}")


if __name__ == "__main__":
    main()

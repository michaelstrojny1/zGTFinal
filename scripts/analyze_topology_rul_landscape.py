from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tem.calibration import ConditionalResidualCalibrator  # noqa: E402
from tem.evidence import TemConfig, infer_true_tau_from_true_rul, run_tem_single_engine  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publication-style topology vs RUL analysis.")
    parser.add_argument("--run-dirs", nargs="+", required=True, help="Run directories containing tem/audit/cache artifacts.")
    parser.add_argument(
        "--rul-cutpoints",
        type=str,
        default="30,80",
        help="Comma-separated cutpoints for low/mid/high RUL bins; default gives <=30, (30,80), >=80.",
    )
    parser.add_argument("--bootstrap", type=int, default=1000, help="Bootstrap samples for correlation CIs.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-json", type=str, default="outputs/topology_rul_landscape.json")
    parser.add_argument("--out-md", type=str, default="outputs/topology_rul_landscape.md")
    parser.add_argument("--fig-dir", type=str, default="outputs/topology_rul_figs")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _one(path: Path, pattern: str) -> Path:
    hits = sorted(path.glob(pattern))
    if not hits:
        raise FileNotFoundError(f"Missing {pattern} under {path}")
    return hits[0]


def _split_flat(flat: np.ndarray, lengths: np.ndarray) -> list[np.ndarray]:
    L = np.asarray(lengths, dtype=np.int64).reshape(-1)
    total = int(np.sum(L))
    if total != int(flat.shape[0]):
        raise ValueError(f"cache mismatch: sum(run_lengths)={total} != flat_len={int(flat.shape[0])}")
    if L.size == 0:
        return []
    cut = np.cumsum(L[:-1], dtype=np.int64)
    return [np.asarray(x, dtype=np.float64) for x in np.split(flat, cut)]


def _pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    xx = np.asarray(x, dtype=np.float64).reshape(-1)
    yy = np.asarray(y, dtype=np.float64).reshape(-1)
    if xx.size < 3 or yy.size != xx.size:
        return None
    if np.std(xx) < 1e-12 or np.std(yy) < 1e-12:
        return None
    return float(np.corrcoef(xx, yy)[0, 1])


def _bootstrap_corr_ci(x: np.ndarray, y: np.ndarray, n_boot: int, seed: int) -> dict[str, float | None]:
    xx = np.asarray(x, dtype=np.float64).reshape(-1)
    yy = np.asarray(y, dtype=np.float64).reshape(-1)
    base = _pearson(xx, yy)
    if base is None or xx.size < 5 or n_boot <= 0:
        return {"corr": base, "ci_low": None, "ci_high": None}
    rng = np.random.default_rng(seed)
    vals = []
    n = xx.size
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        c = _pearson(xx[idx], yy[idx])
        if c is not None and np.isfinite(c):
            vals.append(float(c))
    if not vals:
        return {"corr": base, "ci_low": None, "ci_high": None}
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "corr": base,
        "ci_low": float(np.quantile(arr, 0.025)),
        "ci_high": float(np.quantile(arr, 0.975)),
    }


def _bootstrap_mean_diff_ci(a: np.ndarray, b: np.ndarray, n_boot: int, seed: int) -> dict[str, float | None]:
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    if aa.size < 10 or bb.size < 10:
        return {"diff": None, "ci_low": None, "ci_high": None}
    base = float(np.mean(aa) - np.mean(bb))
    if n_boot <= 0:
        return {"diff": base, "ci_low": None, "ci_high": None}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(int(n_boot)):
        a_s = aa[rng.integers(0, aa.size, size=aa.size)]
        b_s = bb[rng.integers(0, bb.size, size=bb.size)]
        vals.append(float(np.mean(a_s) - np.mean(b_s)))
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "diff": base,
        "ci_low": float(np.quantile(arr, 0.025)),
        "ci_high": float(np.quantile(arr, 0.975)),
    }


def _pvalue_calibrator(tem: dict[str, Any], audit: dict[str, Any]) -> tuple[np.ndarray, np.ndarray | None, TemConfig, ConditionalResidualCalibrator]:
    cfg_obj = tem.get("config", {})
    cal_path = Path(audit["calibration_file"])
    if cal_path.suffix.lower() == ".npz":
        blob = np.load(cal_path)
        cal_res = np.asarray(blob["residuals"], dtype=np.float64).reshape(-1)
        cal_true = np.asarray(blob["true_rul"], dtype=np.float64).reshape(-1) if "true_rul" in blob else None
    else:
        cal_res = np.asarray(np.load(cal_path), dtype=np.float64).reshape(-1)
        cal_true = None
    use_cond = bool(cfg_obj.get("use_conditional_calibration", True) and cal_true is not None)
    cfg = TemConfig(
        r_max=int(cfg_obj.get("r_max", 125)),
        alpha=float(cfg_obj.get("alpha", 0.05)),
        lambda_bet=float(cfg_obj.get("lambda_bet", 0.07)),
        gamma_crit=float(cfg_obj.get("gamma_crit", 1.5)),
        width_crit=int(cfg_obj.get("width_crit", 25)),
        min_persistence=float(cfg_obj.get("min_persistence", 0.5)),
        alert_patience=int(cfg_obj.get("alert_patience", 3)),
        cap_implied_rul=bool(cfg_obj.get("cap_implied_rul", True)),
        evidence_mode=str(cfg_obj.get("evidence_mode", "fixed_tau")),
        compute_tau_diagnostics=bool(cfg_obj.get("compute_tau_diagnostics", True)),
        use_conditional_calibration=use_cond,
        calibration_bins=int(cfg_obj.get("calibration_bins", 8)),
        calibration_min_bin_size=int(cfg_obj.get("calibration_min_bin_size", 128)),
        pvalue_safety_margin=float(cfg_obj.get("pvalue_safety_margin", 0.0)),
    )
    calibrator = ConditionalResidualCalibrator.from_arrays(
        residuals=cal_res,
        true_rul=cal_true if use_cond else None,
        r_max=cfg.r_max,
        n_bins=cfg.calibration_bins,
        min_bin_size=cfg.calibration_min_bin_size,
        pvalue_safety_margin=cfg.pvalue_safety_margin,
    )
    return cal_res, cal_true, cfg, calibrator


def _rul_bin_masks(true_rul: np.ndarray, c1: float, c2: float) -> dict[str, np.ndarray]:
    t = np.asarray(true_rul, dtype=np.float64).reshape(-1)
    return {
        "low_rul": t <= c1,
        "mid_rul": (t > c1) & (t < c2),
        "high_rul": t >= c2,
    }


def _safe_mean(x: np.ndarray) -> float:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def _line_plot(
    x_labels: list[str],
    series: dict[str, list[float]],
    title: str,
    y_label: str,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=150)
    x = np.arange(len(x_labels), dtype=np.float64)
    for name, vals in series.items():
        ax.plot(x, vals, marker="o", linewidth=2.0, label=name)
    ax.set_xticks(x, x_labels)
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _scatter_plot(
    x: np.ndarray,
    y: np.ndarray,
    labels: list[str],
    title: str,
    x_label: str,
    y_label: str,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=150)
    ax.scatter(x, y, s=16, alpha=0.7)
    for i, txt in enumerate(labels):
        ax.annotate(txt, (x[i], y[i]), fontsize=6, alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _to_md(report: dict[str, Any]) -> str:
    lines = ["# Topology vs RUL Landscape", ""]
    lines.append("## Findings")
    for f in report["findings"]:
        lines.append(f"- [{f['severity'].upper()}] {f['message']}")
    if not report["findings"]:
        lines.append("- No high/medium suspicious findings.")
    if report.get("notes"):
        for n in report["notes"]:
            lines.append(f"- [NOTE] {n}")
    lines.append("")
    lines.append("## Global")
    g = report["global"]
    lines.append(f"- Steps analyzed: {g['num_steps_total']}")
    lines.append(f"- Runs analyzed: {g['num_runs_total']}")
    lines.append(
        f"- Mean gamma by RUL bin: low={g['rul_bin_means']['low_rul']['mean_gamma']:.3f}, "
        f"mid={g['rul_bin_means']['mid_rul']['mean_gamma']:.3f}, "
        f"high={g['rul_bin_means']['high_rul']['mean_gamma']:.3f}"
    )
    lines.append(
        f"- Mean persistent valleys by RUL bin: low={g['rul_bin_means']['low_rul']['mean_persistent_valleys']:.3f}, "
        f"mid={g['rul_bin_means']['mid_rul']['mean_persistent_valleys']:.3f}, "
        f"high={g['rul_bin_means']['high_rul']['mean_persistent_valleys']:.3f}"
    )
    lines.append(
        f"- Mean RUL coverage by RUL bin: low={g['rul_bin_means']['low_rul']['mean_rul_coverage']:.3f}, "
        f"mid={g['rul_bin_means']['mid_rul']['mean_rul_coverage']:.3f}, "
        f"high={g['rul_bin_means']['high_rul']['mean_rul_coverage']:.3f}"
    )
    lines.append("")
    lines.append("## Associations")
    for k, v in report["associations"].items():
        lines.append(f"- {k}: corr={v['corr']}, 95% CI=[{v['ci_low']}, {v['ci_high']}]")
    lines.append("")
    lines.append("## Per FD")
    for r in report["per_fd"]:
        lines.append(
            f"- FD{r['fd']:03d}: runs={r['num_runs']}, "
            f"mean_mae={r['mean_pred_mae']:.3f}, mean_rul_cov={r['mean_rul_coverage']:.3f}, "
            f"mean_surface_h1={r['mean_surface_h1']:.3f}"
        )
    lines.append("")
    lines.append("## Figures")
    for p in report["figures"]:
        lines.append(f"- `{p}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    cut_raw = [float(x.strip()) for x in str(args.rul_cutpoints).split(",") if x.strip()]
    if len(cut_raw) != 2:
        raise ValueError("--rul-cutpoints must contain exactly two values, e.g. 30,80")
    c1, c2 = float(min(cut_raw)), float(max(cut_raw))

    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    run_level: list[dict[str, Any]] = []
    step_true_rul: list[np.ndarray] = []
    step_gamma: list[np.ndarray] = []
    step_max_p: list[np.ndarray] = []
    step_local_min: list[np.ndarray] = []
    step_width: list[np.ndarray] = []
    step_true_in_set: list[np.ndarray] = []
    per_fd_rows: list[dict[str, Any]] = []
    findings: list[dict[str, str]] = []
    notes: list[str] = []

    for run_dir_raw in args.run_dirs:
        run_dir = Path(run_dir_raw).resolve()
        tem_path = _one(run_dir, "tem_metrics_*.json")
        audit_path = _one(run_dir, "audit_*.json")
        cache_path = _one(run_dir, "audit_cache_*.npz")
        tem = _load_json(tem_path)
        audit = _load_json(audit_path)
        cache = np.load(cache_path)

        pred_flat = np.asarray(cache["pred_flat"], dtype=np.float64).reshape(-1)
        true_flat = np.asarray(cache["true_flat"], dtype=np.float64).reshape(-1)
        run_lengths = np.asarray(cache["run_lengths"], dtype=np.int64).reshape(-1)
        pred_runs = _split_flat(pred_flat, run_lengths)
        true_runs = _split_flat(true_flat, run_lengths)
        if len(pred_runs) != len(true_runs):
            raise ValueError(f"{run_dir}: pred/true run count mismatch")

        _, cal_true, cfg, calibrator = _pvalue_calibrator(tem, audit)
        tau_max = int(tem.get("config", {}).get("tau_max", 1000))
        per_run_meta = tem.get("per_run", [])
        if len(per_run_meta) != len(pred_runs):
            raise ValueError(
                f"{run_dir}: per_run length mismatch ({len(per_run_meta)} != {len(pred_runs)})."
            )

        fd = int(tem.get("fd", -1))
        fd_pred_mae = []
        fd_cov = []
        fd_surface_h1 = []

        for i, (pred, true) in enumerate(zip(pred_runs, true_runs)):
            res = run_tem_single_engine(
                pred,
                true,
                calibrator.global_sorted_residuals,
                cfg,
                tau_max=tau_max,
                true_tau=infer_true_tau_from_true_rul(true, r_max=cfg.r_max),
                calibration_true_rul=cal_true,
                calibrator=calibrator,
                store_log_k_hist=False,
            )
            true_arr = np.asarray(true, dtype=np.float64).reshape(-1)
            gamma = np.asarray(res["gamma_hist"], dtype=np.float64).reshape(-1)
            max_p = np.asarray(res["max_p_hist"], dtype=np.float64).reshape(-1)
            width = np.asarray(res["width_hist"], dtype=np.float64).reshape(-1)
            local_min = np.asarray(res["local_minima_hist"], dtype=np.float64).reshape(-1)
            true_in_set = np.asarray(res["true_r_in_set_hist"], dtype=np.float64).reshape(-1)
            cov = float(np.mean(np.asarray(res["true_r_in_set_hist"], dtype=np.float64)))
            mae = float(np.mean(np.abs(np.asarray(pred, dtype=np.float64) - true_arr)))

            topo = per_run_meta[i].get("marginal_evidence_topology", {})
            curve = topo.get("curve", {})
            ridge = topo.get("ridge", {})
            surf = per_run_meta[i].get("surface_topology", {})

            run_level.append(
                {
                    "fd": fd,
                    "run_dir": str(run_dir),
                    "run_index": i,
                    "pred_mae": mae,
                    "rul_coverage": cov,
                    "tau_anytime_violation": float(bool(res["tau_anytime_violation"])),
                    "mean_gamma": float(np.mean(gamma)),
                    "mean_local_minima": float(np.mean(local_min)),
                    "mean_persistent_valleys": float(curve.get("mean_persistent_valleys", float("nan"))),
                    "ridge_total_variation_l1": float(ridge.get("total_variation_l1", float("nan"))),
                    "surface_max_h1": float(surf.get("max_h1_persistence", float("nan"))),
                    "surface_sublevel_max_h1": float(surf.get("sublevel_max_h1_persistence", float("nan"))),
                    "surface_superlevel_max_h1": float(surf.get("superlevel_max_h1_persistence", float("nan"))),
                    "surface_num_h1_above_min": float(surf.get("num_h1_above_min", float("nan"))),
                    "surface_roughness": float(surf.get("surface_roughness", float("nan"))),
                    "mean_true_rul": float(np.mean(true_arr)),
                }
            )

            fd_pred_mae.append(mae)
            fd_cov.append(cov)
            fd_surface_h1.append(float(surf.get("max_h1_persistence", float("nan"))))

            step_true_rul.append(true_arr)
            step_gamma.append(gamma)
            step_max_p.append(max_p)
            step_local_min.append(local_min)
            step_width.append(width)
            step_true_in_set.append(true_in_set)

        per_fd_rows.append(
            {
                "fd": fd,
                "num_runs": int(len(pred_runs)),
                "mean_pred_mae": float(np.mean(fd_pred_mae)) if fd_pred_mae else float("nan"),
                "mean_rul_coverage": float(np.mean(fd_cov)) if fd_cov else float("nan"),
                "mean_surface_h1": float(np.nanmean(np.asarray(fd_surface_h1, dtype=np.float64))) if fd_surface_h1 else float("nan"),
            }
        )

    true_all = np.concatenate(step_true_rul, axis=0) if step_true_rul else np.zeros(0, dtype=np.float64)
    gamma_all = np.concatenate(step_gamma, axis=0) if step_gamma else np.zeros(0, dtype=np.float64)
    max_p_all = np.concatenate(step_max_p, axis=0) if step_max_p else np.zeros(0, dtype=np.float64)
    local_min_all = np.concatenate(step_local_min, axis=0) if step_local_min else np.zeros(0, dtype=np.float64)
    width_all = np.concatenate(step_width, axis=0) if step_width else np.zeros(0, dtype=np.float64)
    true_in_set_all = np.concatenate(step_true_in_set, axis=0) if step_true_in_set else np.zeros(0, dtype=np.float64)

    bin_masks = _rul_bin_masks(true_all, c1=c1, c2=c2)
    bin_means: dict[str, dict[str, float | int]] = {}
    for name, mask in bin_masks.items():
        n = int(np.sum(mask))
        bin_means[name] = {
            "n": n,
            "mean_gamma": _safe_mean(gamma_all[mask]) if n else float("nan"),
            "mean_max_persistence": _safe_mean(max_p_all[mask]) if n else float("nan"),
            "mean_local_minima": _safe_mean(local_min_all[mask]) if n else float("nan"),
            "mean_width": _safe_mean(width_all[mask]) if n else float("nan"),
            "mean_rul_coverage": _safe_mean(true_in_set_all[mask]) if n else float("nan"),
            # In lite mode this is the top-2 proxy used in the codebase.
            "mean_persistent_valleys": _safe_mean((max_p_all[mask] >= 0.5).astype(np.float64)) if n else float("nan"),
        }
        if n < 100:
            findings.append(
                {
                    "severity": "medium",
                    "message": f"{name} has low sample size for stable topology statistics (n={n}).",
                }
            )

    run_arr = run_level
    x_gamma = np.asarray([r["mean_gamma"] for r in run_arr], dtype=np.float64)
    x_ridge = np.asarray([r["ridge_total_variation_l1"] for r in run_arr], dtype=np.float64)
    x_h1 = np.asarray([r["surface_max_h1"] for r in run_arr], dtype=np.float64)
    x_h1_super = np.asarray([r["surface_superlevel_max_h1"] for r in run_arr], dtype=np.float64)
    x_h1_sub = np.asarray([r["surface_sublevel_max_h1"] for r in run_arr], dtype=np.float64)
    x_valleys = np.asarray([r["mean_persistent_valleys"] for r in run_arr], dtype=np.float64)
    y_mae = np.asarray([r["pred_mae"] for r in run_arr], dtype=np.float64)
    y_cov = np.asarray([r["rul_coverage"] for r in run_arr], dtype=np.float64)
    y_tauv = np.asarray([r["tau_anytime_violation"] for r in run_arr], dtype=np.float64)

    assoc = {
        "corr_mean_gamma_vs_pred_mae": _bootstrap_corr_ci(x_gamma, y_mae, args.bootstrap, args.seed + 1),
        "corr_ridge_tv_vs_pred_mae": _bootstrap_corr_ci(x_ridge, y_mae, args.bootstrap, args.seed + 2),
        "corr_surface_h1_vs_rul_coverage": _bootstrap_corr_ci(x_h1, y_cov, args.bootstrap, args.seed + 3),
        "corr_mean_gamma_vs_tau_violation_flag": _bootstrap_corr_ci(x_gamma, y_tauv, args.bootstrap, args.seed + 4),
        "corr_surface_superlevel_h1_vs_rul_coverage": _bootstrap_corr_ci(x_h1_super, y_cov, args.bootstrap, args.seed + 5),
        "corr_mean_persistent_valleys_vs_rul_coverage": _bootstrap_corr_ci(x_valleys, y_cov, args.bootstrap, args.seed + 9),
    }

    low_mask = bin_masks["low_rul"]
    high_mask = bin_masks["high_rul"]
    gamma_diff = _bootstrap_mean_diff_ci(gamma_all[low_mask], gamma_all[high_mask], args.bootstrap, args.seed + 6)
    cov_diff = _bootstrap_mean_diff_ci(true_in_set_all[low_mask], true_in_set_all[high_mask], args.bootstrap, args.seed + 7)
    maxp_diff = _bootstrap_mean_diff_ci(max_p_all[low_mask], max_p_all[high_mask], args.bootstrap, args.seed + 8)

    if (
        gamma_diff["ci_high"] is not None
        and cov_diff["ci_high"] is not None
        and maxp_diff["ci_high"] is not None
        and float(gamma_diff["ci_high"]) < 0.0
        and float(cov_diff["ci_high"]) < -0.02
        and float(maxp_diff["ci_high"]) < -0.01
    ):
        findings.append(
            {
                "severity": "medium",
                "message": (
                    "Near-failure topology weakens with a statistically significant drop in gamma/max-persistence "
                    "and lower near-failure coverage versus high RUL."
                ),
            }
        )
    elif (
        gamma_diff["ci_high"] is not None
        and float(gamma_diff["ci_high"]) < 0.0
    ):
        notes.append(
            "Mean gamma is lower at low RUL than high RUL, but this is not coupled to a material near-failure "
            "coverage/persistence deterioration."
        )

    if np.nanmax(np.maximum(x_h1_sub, x_h1_super)) <= 1e-10 and np.nanmean(
        np.asarray([r["surface_roughness"] for r in run_arr], dtype=np.float64)
    ) > 0.08:
        findings.append(
            {
                "severity": "low",
                "message": "Surface H1 is near-zero for both sublevel and superlevel filtrations on this dataset.",
            }
        )

    fd_labels = [f"FD{int(r['fd']):03d}-r{int(r['run_index']):03d}" for r in run_arr]
    fig1 = fig_dir / "topology_vs_rul_bins.png"
    fig2 = fig_dir / "surface_h1_vs_rul_coverage.png"
    fig3 = fig_dir / "gamma_vs_pred_mae.png"

    _line_plot(
        x_labels=["low_rul", "mid_rul", "high_rul"],
        series={
            "mean_gamma": [
                float(bin_means["low_rul"]["mean_gamma"]),
                float(bin_means["mid_rul"]["mean_gamma"]),
                float(bin_means["high_rul"]["mean_gamma"]),
            ],
            "mean_max_persistence": [
                float(bin_means["low_rul"]["mean_max_persistence"]),
                float(bin_means["mid_rul"]["mean_max_persistence"]),
                float(bin_means["high_rul"]["mean_max_persistence"]),
            ],
            "mean_local_minima": [
                float(bin_means["low_rul"]["mean_local_minima"]),
                float(bin_means["mid_rul"]["mean_local_minima"]),
                float(bin_means["high_rul"]["mean_local_minima"]),
            ],
        },
        title="Marginal-Evidence Topology vs RUL Regime",
        y_label="Metric Value",
        out_path=fig1,
    )
    _scatter_plot(
        x=x_h1,
        y=y_cov,
        labels=fd_labels,
        title="Surface H1 Persistence vs Temporal RUL Coverage",
        x_label="surface max_h1_persistence",
        y_label="temporal RUL coverage",
        out_path=fig2,
    )
    _scatter_plot(
        x=x_gamma,
        y=y_mae,
        labels=fd_labels,
        title="Mean Curve Gamma vs Run MAE",
        x_label="mean gamma",
        y_label="run MAE",
        out_path=fig3,
    )

    report = {
        "settings": {
            "run_dirs": [str(Path(x).resolve()) for x in args.run_dirs],
            "rul_cutpoints": [c1, c2],
            "bootstrap": int(args.bootstrap),
            "seed": int(args.seed),
        },
        "global": {
            "num_steps_total": int(true_all.size),
            "num_runs_total": int(len(run_arr)),
            "rul_bin_means": bin_means,
        },
        "effects": {
            "low_minus_high_gamma": gamma_diff,
            "low_minus_high_rul_coverage": cov_diff,
            "low_minus_high_max_persistence": maxp_diff,
        },
        "associations": assoc,
        "per_fd": per_fd_rows,
        "run_level": run_arr,
        "findings": findings,
        "notes": notes,
        "figures": [str(fig1), str(fig2), str(fig3)],
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(_to_md(report), encoding="utf-8")

    print(f"Saved JSON: {out_json}")
    print(f"Saved Markdown: {out_md}")
    print(f"Saved figures under: {fig_dir}")


if __name__ == "__main__":
    main()

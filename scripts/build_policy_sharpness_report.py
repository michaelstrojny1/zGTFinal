from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build policy sharpness-validity report from replay reports.")
    p.add_argument("--canonical", type=str, default="outputs/external_performance_report.json")
    p.add_argument("--balanced", type=str, default="outputs/external_performance_report_policy_replay_balanced_v2.json")
    p.add_argument("--aggressive", type=str, default="outputs/external_performance_report_policy_replay_aggressive_v1.json")
    p.add_argument("--best-valid", type=str, default="outputs/external_performance_report_policy_replay_best_valid_v1.json")
    p.add_argument(
        "--retrain-policy-sweep-json",
        type=str,
        default="outputs/external_policy_replay_sweep_retrain_v3/summary.json",
        help="Optional replay sweep summary used to resolve a selected best-valid report path.",
    )
    p.add_argument("--robust", type=str, default="outputs/external_performance_report_policy_replay_robust_v1.json")
    p.add_argument("--out-json", type=str, default="outputs/publication_full_rtx4050/policy_sharpness_report.json")
    p.add_argument("--out-md", type=str, default="outputs/publication_full_rtx4050/policy_sharpness_report.md")
    p.add_argument("--out-fig", type=str, default="outputs/publication_full_rtx4050/policy_sharpness_frontier.png")
    return p.parse_args()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sanitize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        fv = float(obj)
        if not np.isfinite(fv):
            return None
        return fv
    return obj


def _fmt(v: Any, nd: int = 3) -> str:
    try:
        return f"{float(v):.{nd}f}"
    except Exception:
        return "n/a"


def _rows(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for r in list(report.get("datasets", [])):
        out[str(r.get("dataset", "")).lower()] = r
    return out


def _resolve_aux_policy(
    *,
    best_valid_path: str,
    retrain_policy_sweep_json: str,
    robust_path: str,
) -> tuple[str | None, Path | None]:
    candidates: list[tuple[str, Path]] = []
    if str(best_valid_path).strip():
        candidates.append(("best_valid", Path(best_valid_path).resolve()))

    sweep_path = Path(retrain_policy_sweep_json).resolve()
    if sweep_path.exists():
        try:
            sweep = _load(sweep_path)
        except Exception:
            sweep = {}
        top_level = str(sweep.get("best_policy_report_json", "")).strip()
        if top_level:
            candidates.append(("best_valid", Path(top_level).resolve()))
        nested = sweep.get("best_policy", {}) if isinstance(sweep.get("best_policy", {}), dict) else {}
        nested_path = str(nested.get("report_json", "")).strip()
        if nested_path:
            candidates.append(("best_valid", Path(nested_path).resolve()))

    if str(robust_path).strip():
        candidates.append(("robust", Path(robust_path).resolve()))

    seen: set[tuple[str, str]] = set()
    for label, path in candidates:
        key = (label, str(path))
        if key in seen:
            continue
        seen.add(key)
        if path.exists():
            return label, path
    return None, None


def _tem_stats(tem_path: Path) -> dict[str, float]:
    d = _load(tem_path)
    widths = [float(r.get("mean_width")) for r in list(d.get("per_run", [])) if r.get("mean_width") is not None]
    if not widths:
        return {"mean_width": float("nan"), "median_width": float("nan"), "p90_width": float("nan")}
    arr = np.asarray(widths, dtype=np.float64)
    return {
        "mean_width": float(np.mean(arr)),
        "median_width": float(np.median(arr)),
        "p90_width": float(np.quantile(arr, 0.90)),
    }


def _policy_summary(name: str, report: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, float]]:
    rmap = _rows(report)
    rows: list[dict[str, Any]] = []
    for ds, row in rmap.items():
        if str(row.get("status", "")).lower() != "ok":
            continue
        m = row.get("metrics", {})
        tem_path = Path(str(row.get("artifacts", {}).get("tem_metrics", "")))
        sharp = _tem_stats(tem_path) if tem_path.exists() else {"mean_width": float("nan"), "median_width": float("nan"), "p90_width": float("nan")}
        rows.append(
            {
                "policy": name,
                "dataset": ds,
                "rmse": float(m.get("rmse", np.nan)),
                "rul_cov": float(m.get("rul_cov", np.nan)),
                "tau_v": float(m.get("tau_v", np.nan)),
                "mean_width": float(sharp["mean_width"]),
                "median_width": float(sharp["median_width"]),
                "p90_width": float(sharp["p90_width"]),
                "eff_cov_per_width": float(m.get("rul_cov", np.nan)) / max(float(sharp["mean_width"]), 1e-9),
                "tem_metrics_path": str(tem_path.resolve()) if tem_path.exists() else "",
            }
        )
    if not rows:
        return rows, {
            "coverage_mean": float("nan"),
            "tau_max": float("nan"),
            "width_mean": float("nan"),
            "eff_cov_per_width_mean": float("nan"),
        }
    cov = np.asarray([r["rul_cov"] for r in rows], dtype=np.float64)
    tau = np.asarray([r["tau_v"] for r in rows], dtype=np.float64)
    width = np.asarray([r["mean_width"] for r in rows], dtype=np.float64)
    eff = np.asarray([r["eff_cov_per_width"] for r in rows], dtype=np.float64)
    return rows, {
        "coverage_mean": float(np.nanmean(cov)),
        "tau_max": float(np.nanmax(tau)),
        "width_mean": float(np.nanmean(width)),
        "eff_cov_per_width_mean": float(np.nanmean(eff)),
    }


def _is_pareto_non_dominated(points: list[dict[str, Any]]) -> list[bool]:
    # Objectives: maximize coverage_mean, minimize tau_max, minimize width_mean.
    vals = np.asarray(
        [[float(p["coverage_mean"]), float(p["tau_max"]), float(p["width_mean"])] for p in points], dtype=np.float64
    )
    keep = np.ones(vals.shape[0], dtype=bool)
    for i in range(vals.shape[0]):
        for j in range(vals.shape[0]):
            if i == j:
                continue
            better_or_equal = (
                (vals[j, 0] >= vals[i, 0])
                and (vals[j, 1] <= vals[i, 1])
                and (vals[j, 2] <= vals[i, 2])
            )
            strictly_better = (
                (vals[j, 0] > vals[i, 0])
                or (vals[j, 1] < vals[i, 1])
                or (vals[j, 2] < vals[i, 2])
            )
            if better_or_equal and strictly_better:
                keep[i] = False
                break
    return keep.tolist()


def _plot(summary_rows: list[dict[str, Any]], out_fig: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.2), dpi=140)
    for r in summary_rows:
        ax.scatter(
            [float(r["width_mean"])],
            [float(r["coverage_mean"])],
            s=120,
            label=f"{r['policy']} (tau={r['tau_max']:.3f})",
            alpha=0.9,
            edgecolor="black",
            linewidth=0.4,
        )
        ax.annotate(str(r["policy"]), (float(r["width_mean"]), float(r["coverage_mean"])), xytext=(6, 6), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Mean Set Width (lower better)")
    ax.set_ylabel("Mean RUL Coverage (higher better)")
    ax.set_title("Policy Sharpness-Validity Frontier")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8, framealpha=0.9)
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_fig, bbox_inches="tight")
    plt.close(fig)


def _write_md(path: Path, out: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# Policy Sharpness Report")
    lines.append("")
    lines.append("| Policy | Dataset | RUL cov | Tau v | Mean width | Eff(cov/width) |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for r in out.get("per_dataset_rows", []):
        lines.append(
            f"| {r['policy']} | {r['dataset']} | {_fmt(r['rul_cov'])} | {_fmt(r['tau_v'])} | {_fmt(r['mean_width'])} | {_fmt(r['eff_cov_per_width'],6)} |"
        )
    lines.append("")
    lines.append("## Policy Summary")
    lines.append("| Policy | Cov mean | Tau max | Width mean | Pareto |")
    lines.append("|---|---:|---:|---:|---:|")
    for r in out.get("policy_summary", []):
        lines.append(
            f"| {r['policy']} | {_fmt(r['coverage_mean'])} | {_fmt(r['tau_max'])} | {_fmt(r['width_mean'])} | {'yes' if r['pareto_non_dominated'] else 'no'} |"
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    reports = {
        "canonical": _load(Path(args.canonical).resolve()),
        "balanced": _load(Path(args.balanced).resolve()),
        "aggressive": _load(Path(args.aggressive).resolve()),
    }
    aux_label, aux_path = _resolve_aux_policy(
        best_valid_path=args.best_valid,
        retrain_policy_sweep_json=args.retrain_policy_sweep_json,
        robust_path=args.robust,
    )
    if aux_label is not None and aux_path is not None:
        reports[aux_label] = _load(aux_path)

    per_dataset_rows: list[dict[str, Any]] = []
    policy_summary: list[dict[str, Any]] = []
    for name, rep in reports.items():
        rows, summary = _policy_summary(name, rep)
        per_dataset_rows.extend(rows)
        policy_summary.append({"policy": name, **summary})

    flags = _is_pareto_non_dominated(policy_summary) if policy_summary else []
    for r, f in zip(policy_summary, flags):
        r["pareto_non_dominated"] = bool(f)

    out = {
        "inputs": {
            "canonical": str(Path(args.canonical).resolve()),
            "balanced": str(Path(args.balanced).resolve()),
            "aggressive": str(Path(args.aggressive).resolve()),
            "retrain_policy_sweep_json": str(Path(args.retrain_policy_sweep_json).resolve()) if Path(args.retrain_policy_sweep_json).exists() else "",
            "aux_policy_label": aux_label or "",
            "aux_policy_report": str(aux_path) if aux_path is not None else "",
        },
        "per_dataset_rows": per_dataset_rows,
        "policy_summary": policy_summary,
    }
    out = _sanitize(out)

    out_json = Path(args.out_json).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2, allow_nan=False), encoding="utf-8")
    out_md = Path(args.out_md).resolve()
    _write_md(out_md, out)
    out_fig = Path(args.out_fig).resolve()
    _plot(policy_summary, out_fig)

    print(f"Saved JSON: {out_json}")
    print(f"Saved Markdown: {out_md}")
    print(f"Saved figure: {out_fig}")


if __name__ == "__main__":
    main()

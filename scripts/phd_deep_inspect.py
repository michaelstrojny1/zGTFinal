from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PhD-style deep inspection across TEM run directories.")
    parser.add_argument(
        "--run-dirs",
        nargs="+",
        required=True,
        help="Run directories that contain train_metrics, tem_metrics, and optionally audit JSON files.",
    )
    parser.add_argument("--out-json", type=str, default="outputs/phd_deep_inspect_report.json")
    parser.add_argument("--out-md", type=str, default="outputs/phd_deep_inspect_report.md")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_one(run_dir: Path, pattern: str) -> Path | None:
    hits = sorted(run_dir.glob(pattern))
    return hits[0] if hits else None


def _corr(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size < 3 or y.size != x.size:
        return None
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        fv = float(value)
    except Exception:
        return None
    if not np.isfinite(fv):
        return None
    return fv


def _mean_or_none(arr: np.ndarray) -> float | None:
    if arr.size == 0:
        return None
    return _safe_float(np.mean(arr))


def _sanitize_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_json(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        return _safe_float(obj)
    return obj


def _fmt(v: float | None, digits: int = 3) -> str:
    if v is None:
        return "n/a"
    return f"{v:.{digits}f}"


def _run_summary(run_dir: Path) -> dict[str, Any]:
    train_path = _find_one(run_dir, "train_metrics_*.json")
    tem_path = _find_one(run_dir, "tem_metrics_*.json")
    audit_path = _find_one(run_dir, "audit_*.json")
    if train_path is None or tem_path is None:
        raise FileNotFoundError(f"{run_dir} missing required train/tem metrics JSON.")

    train = _load_json(train_path)
    tem = _load_json(tem_path)
    audit = _load_json(audit_path) if audit_path is not None else {}

    fleet = tem.get("fleet_summary", {})
    per_run = tem.get("per_run", [])

    gamma = []
    topo_minima = []
    topo_valleys = []
    ridge_tv = []
    rul_cov = []
    alert = []
    for r in per_run:
        topo = r.get("marginal_evidence_topology", {})
        curve = topo.get("curve", {})
        ridge = topo.get("ridge", {})
        if curve:
            gamma.append(float(curve.get("mean_gamma", 0.0)))
            topo_minima.append(float(curve.get("mean_local_minima", 0.0)))
            topo_valleys.append(float(curve.get("mean_persistent_valleys", 0.0)))
        elif "mean_gamma" in r:
            # External-eval compact TEM summaries may export mean_gamma directly.
            g = _safe_float(r.get("mean_gamma"))
            if g is not None:
                gamma.append(float(g))
        if ridge:
            ridge_tv.append(float(ridge.get("total_variation_l1", 0.0)))
        rul_cov.append(float(r.get("temporal_rul_coverage", 0.0)))
        if "first_alert_step" in r:
            alert.append(1.0 if int(r.get("first_alert_step", -1)) >= 0 else 0.0)

    gamma_arr = np.asarray(gamma, dtype=np.float64)
    minima_arr = np.asarray(topo_minima, dtype=np.float64)
    valleys_arr = np.asarray(topo_valleys, dtype=np.float64)
    ridge_tv_arr = np.asarray(ridge_tv, dtype=np.float64)
    rul_cov_arr = np.asarray(rul_cov, dtype=np.float64)
    alert_arr = np.asarray(alert, dtype=np.float64)
    per_run_count = int(len(per_run))
    topo_available_runs = int(
        sum(
            1
            for r in per_run
            if bool(r.get("marginal_evidence_topology")) or ("mean_gamma" in r)
        )
    )

    fleet_alert_rate = _safe_float(fleet.get("alert_rate"))
    if fleet_alert_rate is None and alert_arr.size > 0:
        fleet_alert_rate = _safe_float(np.mean(alert_arr))

    return {
        "run_dir": str(run_dir),
        "files": {
            "train": str(train_path),
            "tem": str(tem_path),
            "audit": str(audit_path) if audit_path else None,
        },
        "model_quality": {
            "test_last_rmse": _safe_float(train.get("test_last_rmse")),
            "test_last_mae": _safe_float(train.get("test_last_mae")),
            "best_val_rmse": _safe_float(train.get("best_val_rmse")),
            "calibration_rul_floor": _safe_float(train.get("calibration_rul_floor")),
        },
        "fleet": {
            "num_engines": int(fleet.get("num_engines", 0)),
            "num_tau_diagnostics_engines": int(fleet.get("num_tau_diagnostics_engines", 0)),
            "alert_rate": fleet_alert_rate,
            "mean_temporal_rul_coverage": _safe_float(fleet.get("mean_temporal_rul_coverage")),
            "mean_temporal_tau_coverage": _safe_float(fleet.get("mean_temporal_tau_coverage")),
            "tau_anytime_violation_rate": _safe_float(fleet.get("tau_anytime_violation_rate")),
        },
        "pvalues": {
            "all_frac_le_0.1": _safe_float(audit.get("pvalue_all", {}).get("frac_le_0.1")),
            "all_frac_le_0.2": _safe_float(audit.get("pvalue_all", {}).get("frac_le_0.2")),
            "healthy_mean_p": _safe_float(audit.get("pvalue_healthy_prefix", {}).get("mean_p")),
        },
        "topology_aggregates": {
            "mean_curve_gamma": _mean_or_none(gamma_arr),
            "mean_curve_local_minima": _mean_or_none(minima_arr),
            "mean_curve_persistent_valleys": _mean_or_none(valleys_arr),
            "mean_ridge_total_variation": _mean_or_none(ridge_tv_arr),
        },
        "topology_associations": {
            "corr_gamma_vs_rul_coverage": _corr(gamma_arr, rul_cov_arr) if gamma_arr.size == rul_cov_arr.size else None,
            "corr_ridge_tv_vs_alert": _corr(ridge_tv_arr, alert_arr) if ridge_tv_arr.size == alert_arr.size else None,
            "corr_minima_vs_alert": _corr(minima_arr, alert_arr) if minima_arr.size == alert_arr.size else None,
        },
        "availability": {
            "has_audit_json": bool(audit_path is not None),
            "per_run_count": per_run_count,
            "topology_available_runs": topo_available_runs,
        },
    }


def _to_markdown(report: dict[str, Any]) -> str:
    lines = ["# PhD Deep Inspect Report", ""]
    lines.append("## Runs")
    for row in report["runs"]:
        lines.append(f"- `{row['run_dir']}`")
        mq = row["model_quality"]
        fleet = row["fleet"]
        topo = row["topology_aggregates"]
        assoc = row["topology_associations"]
        avail = row.get("availability", {})
        lines.append(
            f"  - RMSE={_fmt(mq['test_last_rmse'])}, MAE={_fmt(mq['test_last_mae'])}, "
            f"best_val_rmse={_fmt(mq['best_val_rmse'])}, calibration_floor={_fmt(mq['calibration_rul_floor'], 1)}"
        )
        lines.append(
            f"  - alert_rate={_fmt(fleet['alert_rate'])}, rul_cov={_fmt(fleet['mean_temporal_rul_coverage'])}, "
            f"tau_cov={_fmt(fleet['mean_temporal_tau_coverage'])}, tau_violation={_fmt(fleet['tau_anytime_violation_rate'])}, "
            f"tau_diag_engines={fleet['num_tau_diagnostics_engines']}/{fleet['num_engines']}"
        )
        lines.append(
            f"  - topology mean_gamma={_fmt(topo['mean_curve_gamma'])}, local_minima={_fmt(topo['mean_curve_local_minima'])}, "
            f"persistent_valleys={_fmt(topo['mean_curve_persistent_valleys'])}, ridge_tv={_fmt(topo['mean_ridge_total_variation'])}"
        )
        lines.append(
            f"  - corr(gamma,rul_cov)={assoc['corr_gamma_vs_rul_coverage']}, "
            f"corr(ridge_tv,alert)={assoc['corr_ridge_tv_vs_alert']}, corr(minima,alert)={assoc['corr_minima_vs_alert']}"
        )
        lines.append(
            f"  - availability: audit_json={avail.get('has_audit_json')}, "
            f"topology_runs={avail.get('topology_available_runs')}/{avail.get('per_run_count')}"
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("- `num_tau_diagnostics_engines < num_engines` indicates runs where true tau is not identifiable from capped/censored labels.")
    lines.append("- Topology correlations are descriptive, not causal.")
    lines.append("- `n/a` means the source run did not emit that artifact/field (not treated as zero).")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    run_dirs = [Path(p).resolve() for p in args.run_dirs]
    runs = [_run_summary(p) for p in run_dirs]
    report = _sanitize_json({"runs": runs})

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(_to_markdown(report), encoding="utf-8")

    print(f"Saved JSON: {out_json}")
    print(f"Saved Markdown: {out_md}")


if __name__ == "__main__":
    main()

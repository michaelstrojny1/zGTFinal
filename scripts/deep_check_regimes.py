from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tem.calibration import ConditionalResidualCalibrator  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict regime-level deep checks for TEM artifacts.")
    parser.add_argument("--run-dirs", nargs="+", required=True, help="Run dirs containing tem_metrics, audit, and audit_cache.")
    parser.add_argument(
        "--healthy-rul-floor",
        type=float,
        default=100.0,
        help="RUL floor for healthy-prefix p-value checks.",
    )
    parser.add_argument(
        "--require-surface-topology",
        action="store_true",
        help="Require non-skipped surface topology entries in every per-run summary.",
    )
    parser.add_argument("--out-json", type=str, default="outputs/deep_check_regimes_report.json")
    parser.add_argument("--out-md", type=str, default="outputs/deep_check_regimes_report.md")
    parser.add_argument(
        "--superuniformity-levels",
        type=str,
        default="0.1,0.2",
        help="Comma-separated p-value CDF levels used for superuniformity pass/fail checks.",
    )
    parser.add_argument(
        "--superuniformity-fail-excess",
        type=float,
        default=0.01,
        help=(
            "Minimum absolute exceedance over finite-sample bound required to flag a "
            "superuniformity failure; prevents tiny-noise false positives."
        ),
    )
    parser.add_argument(
        "--superuniformity-critical-max",
        type=float,
        default=0.2,
        help=(
            "Only CDF levels <= this value are treated as pass/fail-critical. "
            "Higher levels are reported as informational shape diagnostics."
        ),
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _one(path: Path, pattern: str) -> Path:
    hits = sorted(path.glob(pattern))
    if not hits:
        raise FileNotFoundError(f"Missing {pattern} under {path}")
    return hits[0]


def _p_block(
    pvals: np.ndarray,
    fail_excess: float,
    levels: list[float],
    critical_max: float,
) -> dict[str, Any]:
    p = np.asarray(pvals, dtype=np.float64).reshape(-1)
    out: dict[str, Any] = {"n": int(p.size)}
    if p.size == 0:
        out["status"] = "empty"
        out["checks"] = []
        return out
    checks = []
    ok = True
    info_failures = 0
    critical_count = 0
    for a in levels:
        frac = float(np.mean(p <= a))
        slack = max(0.01, 3.0 * math.sqrt(a * (1.0 - a) / float(p.size)))
        bound = a + slack
        passed = bool(frac <= (bound + float(fail_excess)))
        is_critical = bool(a <= float(critical_max) + 1e-12)
        if is_critical:
            critical_count += 1
            ok = ok and passed
        elif not passed:
            info_failures += 1
        checks.append(
            {
                "a": a,
                "frac_le_a": frac,
                "bound": bound,
                "effective_bound": float(bound + float(fail_excess)),
                "excess_over_bound": float(max(0.0, frac - bound)),
                "critical": is_critical,
                "pass": passed,
            }
        )
    # If no critical levels are provided, fall back to strict behavior over all levels.
    if critical_count == 0:
        ok = bool(all(bool(c.get("pass", False)) for c in checks))
    out["status"] = "ok" if ok else "fail"
    out["mean_p"] = float(np.mean(p))
    out["checks"] = checks
    out["num_info_shape_failures"] = int(info_failures)
    return out


def _run_lengths_to_position_fraction(lengths: np.ndarray) -> np.ndarray:
    L = np.asarray(lengths, dtype=np.int64).reshape(-1)
    total = int(np.sum(L))
    out = np.zeros(total, dtype=np.float64)
    s = 0
    for n in L.tolist():
        e = s + int(n)
        if n <= 1:
            out[s:e] = 1.0
        else:
            out[s:e] = np.linspace(0.0, 1.0, int(n), endpoint=True)
        s = e
    return out


def _strict_checks_for_run(
    run_dir: Path,
    healthy_rul_floor: float,
    require_surface_topology: bool,
    superuniformity_fail_excess: float,
    superuniformity_levels: list[float],
    superuniformity_critical_max: float,
) -> dict[str, Any]:
    tem_path = _one(run_dir, "tem_metrics_*.json")
    audit_path = _one(run_dir, "audit_*.json")
    cache_path = _one(run_dir, "audit_cache_*.npz")

    tem = _load_json(tem_path)
    audit = _load_json(audit_path)
    cache = np.load(cache_path)

    pred = np.asarray(cache["pred_flat"], dtype=np.float64).reshape(-1)
    true = np.asarray(cache["true_flat"], dtype=np.float64).reshape(-1)
    run_lengths = np.asarray(cache["run_lengths"], dtype=np.int64).reshape(-1)
    if pred.shape[0] != true.shape[0]:
        raise ValueError(f"{run_dir}: pred/true cache lengths differ.")
    if int(np.sum(run_lengths)) != pred.shape[0]:
        raise ValueError(f"{run_dir}: run_lengths sum mismatch with flat cache length.")

    score = np.abs(pred - true)
    cal_path = Path(audit["calibration_file"])
    if cal_path.suffix.lower() == ".npz":
        blob = np.load(cal_path)
        cal_res = np.asarray(blob["residuals"], dtype=np.float64).reshape(-1)
        cal_true = np.asarray(blob["true_rul"], dtype=np.float64).reshape(-1) if "true_rul" in blob else None
    else:
        cal_res = np.asarray(np.load(cal_path), dtype=np.float64).reshape(-1)
        cal_true = None

    cfg = tem.get("config", {})
    use_cond = bool(cfg.get("use_conditional_calibration", True) and cal_true is not None)
    calibrator = ConditionalResidualCalibrator.from_arrays(
        residuals=cal_res,
        true_rul=cal_true if use_cond else None,
        r_max=int(cfg.get("r_max", 125)),
        n_bins=int(cfg.get("calibration_bins", 8)),
        min_bin_size=int(cfg.get("calibration_min_bin_size", 128)),
        pvalue_safety_margin=float(cfg.get("pvalue_safety_margin", 0.0)),
    )
    pvals = calibrator.p_values(score, implied_rul=true if use_cond else None)

    pos_frac = _run_lengths_to_position_fraction(run_lengths)
    masks = {
        "global": np.ones_like(true, dtype=bool),
        "healthy_prefix": true >= float(healthy_rul_floor),
        "low_rul": true <= 30.0,
        "mid_rul": (true > 30.0) & (true < 80.0),
        "high_rul": true >= 80.0,
        "early_time": pos_frac <= (1.0 / 3.0),
        "mid_time": (pos_frac > (1.0 / 3.0)) & (pos_frac <= (2.0 / 3.0)),
        "late_time": pos_frac > (2.0 / 3.0),
    }
    blocks = {
        k: _p_block(
            pvals[m],
            fail_excess=superuniformity_fail_excess,
            levels=superuniformity_levels,
            critical_max=superuniformity_critical_max,
        )
        for k, m in masks.items()
    }
    failed_blocks = [k for k, v in blocks.items() if v["status"] == "fail"]

    per = tem.get("per_run", [])
    rul_cov = np.asarray([float(r.get("temporal_rul_coverage", float("nan"))) for r in per], dtype=np.float64) if per else np.zeros(0)
    has_nan = bool(np.isnan(rul_cov).any())
    cov_range_ok = bool(np.all((rul_cov >= 0.0) & (rul_cov <= 1.0))) if rul_cov.size else True
    surface_ok = True
    if require_surface_topology and per:
        surface_ok = all(
            isinstance(r.get("surface_topology"), dict) and str(r["surface_topology"].get("backend", "")) != "skipped"
            for r in per
        )

    findings: list[dict[str, Any]] = []
    if failed_blocks:
        findings.append(
            {
                "severity": "medium",
                "type": "regime_superuniformity_failure",
                "message": f"Failed blocks: {failed_blocks}",
            }
        )
    if has_nan:
        findings.append({"severity": "high", "type": "nan_integrity", "message": "NaN detected in per-run coverage."})
    if not cov_range_ok:
        findings.append(
            {"severity": "high", "type": "coverage_range_integrity", "message": "Per-run coverage outside [0,1]."}
        )
    if not surface_ok:
        findings.append(
            {"severity": "high", "type": "surface_topology_missing", "message": "Surface topology missing/skipped."}
        )

    return {
        "run_dir": str(run_dir),
        "tem_metrics_path": str(tem_path),
        "audit_path": str(audit_path),
        "cache_path": str(cache_path),
        "use_conditional_calibration": use_cond,
        "config": {
            "pvalue_safety_margin": float(cfg.get("pvalue_safety_margin", 0.0)),
            "calibration_bins": int(cfg.get("calibration_bins", 8)),
            "calibration_min_bin_size": int(cfg.get("calibration_min_bin_size", 128)),
            "superuniformity_critical_max": float(superuniformity_critical_max),
            "topology_level": cfg.get("topology_level"),
            "surface_topology_scope": cfg.get("surface_topology_scope"),
        },
        "integrity": {
            "has_nan": has_nan,
            "coverage_in_range": cov_range_ok,
            "surface_topology_ok": surface_ok,
        },
        "blocks": blocks,
        "failed_blocks": failed_blocks,
        "num_findings": len(findings),
        "findings": findings,
    }


def _to_md(report: dict[str, Any]) -> str:
    lines = ["# Regime Deep Check Report", ""]
    lines.append("## Global")
    lines.append(f"- Runs checked: {report['runs_checked']}")
    lines.append(f"- Total findings: {report['num_findings_total']}")
    lines.append("")
    lines.append("## Per Run")
    for r in report["runs"]:
        lines.append(f"- `{r['run_dir']}`")
        lines.append(
            f"  - findings={r['num_findings']}, failed_blocks={r['failed_blocks']}, "
            f"margin={r['config']['pvalue_safety_margin']}, bins={r['config']['calibration_bins']}, "
            f"min_bin={r['config']['calibration_min_bin_size']}"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    runs = [Path(x).resolve() for x in args.run_dirs]
    levels = [float(x.strip()) for x in str(args.superuniformity_levels).split(",") if x.strip()]
    if not levels:
        raise ValueError("--superuniformity-levels must contain at least one value.")
    if any((x <= 0.0 or x >= 1.0) for x in levels):
        raise ValueError("--superuniformity-levels values must be in (0,1).")
    if not (0.0 < float(args.superuniformity_critical_max) < 1.0):
        raise ValueError("--superuniformity-critical-max must be in (0,1).")
    rows = [
        _strict_checks_for_run(
            r,
            args.healthy_rul_floor,
            args.require_surface_topology,
            superuniformity_fail_excess=float(args.superuniformity_fail_excess),
            superuniformity_levels=levels,
            superuniformity_critical_max=float(args.superuniformity_critical_max),
        )
        for r in runs
    ]
    num_findings = int(sum(int(r["num_findings"]) for r in rows))
    report = {
        "runs_checked": len(rows),
        "num_findings_total": num_findings,
        "runs": rows,
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(_to_md(report), encoding="utf-8")

    print(f"Runs checked: {len(rows)}")
    print(f"Findings: {num_findings}")
    print(f"Saved JSON: {out_json}")
    print(f"Saved Markdown: {out_md}")


if __name__ == "__main__":
    main()

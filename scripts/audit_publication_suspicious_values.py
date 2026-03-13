from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit publication artifacts for suspicious-value patterns.")
    p.add_argument("--strict-main-root", type=str, default="outputs/publication_full_rtx4050/strict_main")
    p.add_argument("--external-report", type=str, default="outputs/external_performance_report.json")
    p.add_argument("--r-max", type=int, default=125)
    p.add_argument("--tau-ident-threshold", type=float, default=0.75)
    p.add_argument("--tau-ident-warn-margin", type=float, default=0.05)
    p.add_argument("--min-external-runs", type=int, default=5)
    p.add_argument("--out-json", type=str, default="outputs/publication_full_rtx4050/phd_suspicious_values_audit.json")
    p.add_argument("--out-md", type=str, default="outputs/publication_full_rtx4050/phd_suspicious_values_audit.md")
    return p.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(v: Any, nd: int = 3) -> str:
    try:
        return f"{float(v):.{nd}f}"
    except Exception:
        return "n/a"


def _severity_counts(findings: list[dict[str, Any]]) -> dict[str, int]:
    out = {"high": 0, "medium": 0, "low": 0}
    for row in findings:
        sev = str(row.get("severity", "")).lower()
        if sev in out:
            out[sev] += 1
    return out


def main() -> None:
    args = parse_args()
    strict_root = Path(args.strict_main_root).resolve()
    external_report_path = Path(args.external_report).resolve()

    findings: list[dict[str, Any]] = []
    strict_rows: list[dict[str, Any]] = []
    for run_dir in sorted([p for p in strict_root.glob("fd*") if p.is_dir()]):
        fd_digits = "".join(ch for ch in run_dir.name if ch.isdigit())
        fd = int(fd_digits) if fd_digits else -1
        tem_candidates = sorted(run_dir.glob("tem_metrics_fd*.json"))
        if not tem_candidates:
            findings.append(
                {
                    "severity": "high",
                    "type": "missing_strict_tem_metrics",
                    "message": f"{run_dir} is missing tem_metrics_fd*.json",
                }
            )
            continue
        tem = _load_json(tem_candidates[0])
        fleet = tem.get("fleet_summary", {}) if isinstance(tem.get("fleet_summary", {}), dict) else {}
        cov = float(fleet.get("mean_temporal_rul_coverage", np.nan))
        tau = float(fleet.get("tau_anytime_violation_rate", np.nan))
        num_engines = int(fleet.get("num_engines", 0))
        num_tau = int(fleet.get("num_tau_diagnostics_engines", 0))
        tau_ident = float(num_tau / num_engines) if num_engines > 0 else float("nan")
        strict_rows.append(
            {
                "fd": fd,
                "run_dir": str(run_dir),
                "rul_cov": cov,
                "tau_v": tau,
                "tau_ident_ratio": tau_ident,
                "num_engines": num_engines,
                "num_tau_diagnostics_engines": num_tau,
            }
        )

        if not (0.0 <= cov <= 1.0):
            findings.append(
                {
                    "severity": "high",
                    "type": "strict_metric_out_of_range",
                    "message": f"FD{fd:03d} rul_cov={cov} is outside [0, 1]",
                }
            )
        if not (0.0 <= tau <= 1.0):
            findings.append(
                {
                    "severity": "high",
                    "type": "strict_metric_out_of_range",
                    "message": f"FD{fd:03d} tau_v={tau} is outside [0, 1]",
                }
            )
        if tau_ident < float(args.tau_ident_threshold):
            findings.append(
                {
                    "severity": "high",
                    "type": "tau_identifiability_below_threshold",
                    "message": (
                        f"FD{fd:03d} tau_ident_ratio={tau_ident:.3f} < "
                        f"{float(args.tau_ident_threshold):.3f}"
                    ),
                }
            )
        elif tau_ident < float(args.tau_ident_threshold + args.tau_ident_warn_margin):
            findings.append(
                {
                    "severity": "medium",
                    "type": "tau_identifiability_borderline",
                    "message": (
                        f"FD{fd:03d} tau_ident_ratio={tau_ident:.3f} is close to "
                        f"{float(args.tau_ident_threshold):.3f}"
                    ),
                }
            )

    external_report = _load_json(external_report_path)
    external_rows: list[dict[str, Any]] = []
    width_warn_threshold = 0.98 * float(args.r_max)
    for row in list(external_report.get("datasets", [])):
        dataset = str(row.get("dataset", "unknown")).lower()
        status = str(row.get("status", "")).lower()
        if status != "ok":
            findings.append(
                {
                    "severity": "high",
                    "type": "external_dataset_error",
                    "message": f"{dataset} status={row.get('status', 'unknown')}",
                }
            )
            continue

        metrics = row.get("metrics", {}) if isinstance(row.get("metrics", {}), dict) else {}
        cov = float(metrics.get("rul_cov", np.nan))
        tau = float(metrics.get("tau_v", np.nan))
        rmse = float(metrics.get("rmse", np.nan))
        num_runs = int(row.get("num_runs", 0))
        mean_width = float("nan")
        tem_metrics_path = Path(str(row.get("artifacts", {}).get("tem_metrics", "")))
        if tem_metrics_path.exists():
            tem = _load_json(tem_metrics_path)
            widths = [
                float(r.get("mean_width"))
                for r in list(tem.get("per_run", []))
                if r.get("mean_width") is not None
            ]
            if widths:
                mean_width = float(np.mean(np.asarray(widths, dtype=np.float64)))

        external_rows.append(
            {
                "dataset": dataset,
                "num_runs": num_runs,
                "rul_cov": cov,
                "tau_v": tau,
                "rmse": rmse,
                "mean_width": mean_width if np.isfinite(mean_width) else None,
                "width_frac_of_rmax": (mean_width / float(args.r_max)) if np.isfinite(mean_width) else None,
                "tem_metrics_path": str(tem_metrics_path) if tem_metrics_path.exists() else "",
            }
        )

        if not (0.0 <= cov <= 1.0):
            findings.append(
                {
                    "severity": "high",
                    "type": "external_metric_out_of_range",
                    "message": f"{dataset} rul_cov={cov} is outside [0, 1]",
                }
            )
        if not (0.0 <= tau <= 1.0):
            findings.append(
                {
                    "severity": "high",
                    "type": "external_metric_out_of_range",
                    "message": f"{dataset} tau_v={tau} is outside [0, 1]",
                }
            )
        if num_runs < int(args.min_external_runs):
            findings.append(
                {
                    "severity": "medium",
                    "type": "small_external_sample",
                    "message": f"{dataset} has only {num_runs} test runs",
                }
            )
        if (cov >= 0.999) and (tau <= 0.001) and np.isfinite(mean_width) and (mean_width >= width_warn_threshold):
            findings.append(
                {
                    "severity": "medium",
                    "type": "overconservative_external_policy",
                    "message": (
                        f"{dataset} is near-perfect but mean_width={mean_width:.1f}/{int(args.r_max)}"
                    ),
                }
            )

    counts = _severity_counts(findings)
    report = {
        "summary": {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "strict_main": strict_rows,
            "external": external_rows,
            "num_findings": len(findings),
            "num_high": counts["high"],
            "num_medium": counts["medium"],
            "num_low": counts["low"],
        },
        "findings": findings,
    }

    out_json = Path(args.out_json).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")

    lines: list[str] = []
    lines.append("# Publication Suspicious-Values Audit")
    lines.append("")
    lines.append(f"- generated_utc: {report['summary']['generated_utc']}")
    lines.append(
        f"- findings: {len(findings)} (high={counts['high']}, medium={counts['medium']}, low={counts['low']})"
    )
    lines.append("")
    lines.append("## Strict Main Snapshot")
    for row in strict_rows:
        lines.append(
            f"- FD{int(row['fd']):03d}: cov={_fmt(row['rul_cov'])}, tau={_fmt(row['tau_v'])}, "
            f"tau_ident={_fmt(row['tau_ident_ratio'])}, n={int(row['num_engines'])}"
        )
    lines.append("")
    lines.append("## External Snapshot")
    for row in external_rows:
        width_txt = _fmt(row["mean_width"], 1) if row["mean_width"] is not None else "n/a"
        width_frac = _fmt(row["width_frac_of_rmax"]) if row["width_frac_of_rmax"] is not None else "n/a"
        lines.append(
            f"- {row['dataset']}: cov={_fmt(row['rul_cov'])}, tau={_fmt(row['tau_v'])}, "
            f"mean_width={width_txt}, width/rmax={width_frac}, n={int(row['num_runs'])}"
        )
    lines.append("")
    lines.append("## Findings")
    if findings:
        for row in findings:
            lines.append(f"- [{row['severity']}] {row['type']}: {row['message']}")
    else:
        lines.append("- none")

    out_md = Path(args.out_md).resolve()
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Saved JSON: {out_json}")
    print(f"Saved Markdown: {out_md}")
    print(f"Findings: {len(findings)} (high={counts['high']}, medium={counts['medium']}, low={counts['low']})")


if __name__ == "__main__":
    main()

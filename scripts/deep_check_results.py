from __future__ import annotations

import argparse
import fnmatch
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deep check TEM result artifacts for suspicious patterns.")
    parser.add_argument("--outputs-root", type=str, default="outputs")
    parser.add_argument("--tau-violation-threshold", type=float, default=0.25)
    parser.add_argument("--min-rul-coverage", type=float, default=0.8)
    parser.add_argument("--max-null-alert-rate", type=float, default=0.2)
    parser.add_argument("--min-alert-separation", type=float, default=0.1)
    parser.add_argument(
        "--flag-tau-identifiability-gap",
        action="store_true",
        help="If set, report engines-without-identifiable-true-tau as low-severity findings.",
    )
    parser.add_argument(
        "--exclude-globs",
        type=str,
        default="",
        help=(
            "Optional comma-separated glob patterns (matched against relative POSIX-style paths under outputs-root) "
            "to exclude exploratory artifacts from checks, e.g. 'policy_sweep/**,synthetic_grid/**'."
        ),
    )
    parser.add_argument("--report-path", type=str, default="outputs/deep_check_report.json")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _is_excluded(path: Path, root: Path, patterns: list[str]) -> bool:
    if not patterns:
        return False
    try:
        rel = path.resolve().relative_to(root.resolve()).as_posix()
    except Exception:
        rel = path.as_posix()
    return any(fnmatch.fnmatch(rel, pat) for pat in patterns)


def _check_tem_metrics(
    path: Path,
    obj: dict[str, Any],
    tau_violation_threshold: float,
    min_rul_coverage: float,
    flag_tau_identifiability_gap: bool,
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    fleet = obj.get("fleet_summary", {})
    tau_v = fleet.get("tau_anytime_violation_rate")
    rul_cov = fleet.get("mean_temporal_rul_coverage")
    n_eng = fleet.get("num_engines")
    n_tau = fleet.get("num_tau_diagnostics_engines")
    if flag_tau_identifiability_gap and n_eng is not None and n_tau is not None and int(n_tau) < int(n_eng):
        findings.append(
            {
                "severity": "low",
                "type": "tau_identifiability_gap",
                "message": f"tau diagnostics available for {int(n_tau)}/{int(n_eng)} engines (capped/censored runs excluded)",
            }
        )
    if tau_v is None:
        findings.append({"severity": "medium", "type": "missing_metric", "message": "tau_anytime_violation_rate missing"})
    elif float(tau_v) > tau_violation_threshold:
        findings.append(
            {
                "severity": "high",
                "type": "coverage_risk",
                "message": f"tau_anytime_violation_rate={float(tau_v):.3f} > {tau_violation_threshold:.3f}",
            }
        )
    if rul_cov is None:
        findings.append({"severity": "medium", "type": "missing_metric", "message": "mean_temporal_rul_coverage missing"})
    elif float(rul_cov) < min_rul_coverage:
        findings.append(
            {
                "severity": "high",
                "type": "coverage_risk",
                "message": f"mean_temporal_rul_coverage={float(rul_cov):.3f} < {min_rul_coverage:.3f}",
            }
        )

    per_run = obj.get("per_run", [])
    if per_run and "marginal_evidence_topology" not in per_run[0]:
        findings.append(
            {
                "severity": "medium",
                "type": "topology_gap",
                "message": "marginal_evidence_topology missing in per_run summaries",
            }
        )

    for f in findings:
        f["artifact"] = str(path)
    return findings


def _check_audit(path: Path, obj: dict[str, Any]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    p_all = obj.get("pvalue_all", {})
    p_healthy = obj.get("pvalue_healthy_prefix", {})

    def _check_block(block: dict[str, Any], label: str, severity: str) -> None:
        n = int(block.get("n", 0) or 0)
        if n <= 0:
            return
        for a in (0.1, 0.2, 0.5):
            key = f"frac_le_{a}"
            frac = block.get(key)
            if frac is None:
                continue
            frac_v = float(frac)
            # Finite-sample slack: 3-sigma binomial envelope with a minimum tolerance.
            slack = max(0.01, 3.0 * float(np.sqrt(a * (1.0 - a) / n)))
            if frac_v > a + slack:
                findings.append(
                    {
                        "severity": severity,
                        "type": "superuniformity_failure",
                        "message": (
                            f"{label} {key}={frac_v:.3f} exceeds {a:.1f} + slack({slack:.3f}) "
                            f"with n={n}"
                        ),
                    }
                )

    _check_block(p_healthy, label="pvalue_healthy_prefix", severity="high")
    _check_block(p_all, label="pvalue_all", severity="medium")
    for f in findings:
        f["artifact"] = str(path)
    return findings


def _check_synthetic(
    path: Path,
    obj: dict[str, Any],
    max_null_alert_rate: float,
    min_alert_separation: float,
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    null_alert = float(obj.get("null_cohort", {}).get("alert_rate", 0.0))
    deg_alert = float(obj.get("degraded_cohort", {}).get("alert_rate", 0.0))
    if null_alert > max_null_alert_rate:
        findings.append(
            {
                "severity": "high",
                "type": "false_alarm_risk",
                "message": f"null alert_rate={null_alert:.3f} exceeds {max_null_alert_rate:.3f}",
            }
        )
    if abs(deg_alert - null_alert) < min_alert_separation:
        findings.append(
            {
                "severity": "high",
                "type": "weak_discrimination",
                "message": f"degraded-null alert separation={abs(deg_alert - null_alert):.3f} below {min_alert_separation:.3f}",
            }
        )
    for f in findings:
        f["artifact"] = str(path)
    return findings


def main() -> None:
    args = parse_args()
    root = Path(args.outputs_root)
    exclude_patterns = [p.strip() for p in str(args.exclude_globs).split(",") if p.strip()]

    findings: list[dict[str, Any]] = []
    scanned = {"tem_metrics": 0, "audit": 0, "synthetic": 0}

    for p in root.rglob("tem_metrics_*.json"):
        if _is_excluded(p, root, exclude_patterns):
            continue
        obj = _load_json(p)
        if obj is None:
            continue
        scanned["tem_metrics"] += 1
        findings.extend(
            _check_tem_metrics(
                p,
                obj,
                tau_violation_threshold=float(args.tau_violation_threshold),
                min_rul_coverage=float(args.min_rul_coverage),
                flag_tau_identifiability_gap=bool(args.flag_tau_identifiability_gap),
            )
        )

    for p in root.rglob("audit_*.json"):
        if _is_excluded(p, root, exclude_patterns):
            continue
        obj = _load_json(p)
        if obj is None:
            continue
        scanned["audit"] += 1
        findings.extend(_check_audit(p, obj))

    for p in root.rglob("synthetic_summary.json"):
        if _is_excluded(p, root, exclude_patterns):
            continue
        obj = _load_json(p)
        if obj is None:
            continue
        scanned["synthetic"] += 1
        findings.extend(
            _check_synthetic(
                p,
                obj,
                max_null_alert_rate=float(args.max_null_alert_rate),
                min_alert_separation=float(args.min_alert_separation),
            )
        )

    severity_rank = {"high": 0, "medium": 1, "low": 2}
    findings.sort(key=lambda x: (severity_rank.get(str(x.get("severity", "low")), 3), str(x.get("artifact", ""))))

    report = {
        "inputs": {
            "outputs_root": str(root),
            "exclude_globs": exclude_patterns,
        },
        "scanned": scanned,
        "num_findings": len(findings),
        "findings": findings,
    }
    out_path = Path(args.report_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Scanned: {scanned}")
    print(f"Findings: {len(findings)}")
    print(f"Saved report: {out_path}")


if __name__ == "__main__":
    main()

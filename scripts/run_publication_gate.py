from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run canonical publication gate checks and emit a single summary.")
    p.add_argument("--python", type=str, default=sys.executable)
    p.add_argument("--matrix-report", type=str, default="outputs/publication_full_rtx4050_report.json")
    p.add_argument("--topology-json", type=str, default="outputs/publication_full_rtx4050/topology_rul_landscape.json")
    p.add_argument("--paper-md", type=str, default="paper/topological_evidence_curves.md")
    p.add_argument("--rul-dataset-summary", type=str, default="outputs/rul_dataset_summary.json")
    p.add_argument("--external-performance-report", type=str, default="outputs/external_performance_report.json")
    p.add_argument("--strict-main-root", type=str, default="outputs/publication_full_rtx4050/strict_main")
    p.add_argument("--external-root", type=str, default="outputs/external_real_eval_final_policy_v9")
    p.add_argument("--pub-dir", type=str, default="outputs/publication_full_rtx4050")
    p.add_argument("--out-json", type=str, default="outputs/publication_full_rtx4050/publication_gate_summary.json")
    p.add_argument("--out-md", type=str, default="outputs/publication_full_rtx4050/publication_gate_summary.md")
    return p.parse_args()


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_md(path: Path, rep: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# Publication Gate Summary")
    lines.append("")
    lines.append(f"- Timestamp UTC: {rep['timestamp_utc']}")
    lines.append(f"- Overall pass: {'YES' if rep['overall_pass'] else 'NO'}")
    lines.append("")
    lines.append("## Checks")
    for row in rep["checks"]:
        lines.append(f"- [{'PASS' if row['pass'] else 'FAIL'}] {row['name']}: {row['detail']}")
    lines.append("")
    lines.append("## Snapshot")
    snap = rep["snapshot"]
    lines.append(f"- readiness_score={snap['readiness_score']:.2f}")
    lines.append(f"- readiness_score_conservative={snap['readiness_score_conservative']:.2f}")
    lines.append(f"- readiness_penalties={snap['readiness_penalties']}")
    lines.append(f"- external_num_ok={snap['external_num_ok']}/{snap['external_num_total']}")
    lines.append(f"- external_datasets={','.join(snap['external_datasets'])}")
    lines.append(f"- suspicious_values_high={snap.get('suspicious_values_high', 0)}")
    lines.append(f"- suspicious_values_medium={snap.get('suspicious_values_medium', 0)}")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    py = str(args.python)
    pub_dir = Path(args.pub_dir)
    strict_root = Path(args.strict_main_root)
    ext_root = Path(args.external_root)

    ext_summary_json = Path("outputs/external_dataset_summary.json")
    ext_summary_md = Path("outputs/external_dataset_summary.md")
    strict_deep_json = pub_dir / "deep_check_results_strict_main.json"
    ext_deep_json = pub_dir / "deep_check_results_external_canonical.json"
    full_curated_json = pub_dir / "deep_check_results_full_bundle_curated.json"
    strict_reg_json = pub_dir / "deep_check_regimes_stricter_strict_main.json"
    ext_reg_json = pub_dir / "deep_check_regimes_external_canonical.json"
    plus_reg_json = pub_dir / "deep_check_regimes_strict_plus_external.json"
    readiness_json = pub_dir / "stats_conference_readiness.json"
    readiness_md = pub_dir / "stats_conference_readiness.md"
    consistency_json = Path("outputs/artifact_consistency_report.json")
    consistency_md = Path("outputs/artifact_consistency_report.md")
    inspect_json = pub_dir / "phd_deep_inspect_report_strict_plus_external.json"
    inspect_md = pub_dir / "phd_deep_inspect_report_strict_plus_external.md"
    suspicious_json = pub_dir / "phd_suspicious_values_audit.json"
    suspicious_md = pub_dir / "phd_suspicious_values_audit.md"

    # Build/refresh summaries.
    _run(
        [
            py,
            "scripts/build_external_dataset_summary.py",
            "--matrix-report",
            str(args.matrix_report),
            "--rul-dataset-summary",
            str(args.rul_dataset_summary),
            "--external-performance-report",
            str(args.external_performance_report),
            "--out-json",
            str(ext_summary_json),
            "--out-md",
            str(ext_summary_md),
        ]
    )

    # Deep checks.
    _run([py, "scripts/deep_check_results.py", "--outputs-root", str(strict_root), "--report-path", str(strict_deep_json)])
    _run([py, "scripts/deep_check_results.py", "--outputs-root", str(ext_root), "--report-path", str(ext_deep_json)])
    _run(
        [
            py,
            "scripts/deep_check_results.py",
            "--outputs-root",
            str(pub_dir),
            "--exclude-globs",
            "policy_sweep/**,synthetic_grid/**",
            "--report-path",
            str(full_curated_json),
        ]
    )

    # Regime checks.
    strict_run_dirs = [str(strict_root / f"fd{i:03d}") for i in (1, 2, 3, 4)]
    ext_run_dirs = [str(ext_root / "femto_fd001"), str(ext_root / "xjtu_sy_fd001"), str(ext_root / "cmapss_fd001")]
    _run(
        [
            py,
            "scripts/deep_check_regimes.py",
            "--run-dirs",
            *strict_run_dirs,
            "--healthy-rul-floor",
            "70",
            "--require-surface-topology",
            "--superuniformity-levels",
            "0.1,0.2,0.5",
            "--superuniformity-fail-excess",
            "0.02",
            "--superuniformity-critical-max",
            "0.2",
            "--out-json",
            str(strict_reg_json),
            "--out-md",
            str(pub_dir / "deep_check_regimes_stricter_strict_main.md"),
        ]
    )
    _run(
        [
            py,
            "scripts/deep_check_regimes.py",
            "--run-dirs",
            *ext_run_dirs,
            "--healthy-rul-floor",
            "70",
            "--require-surface-topology",
            "--superuniformity-levels",
            "0.1,0.2,0.5",
            "--superuniformity-fail-excess",
            "0.02",
            "--superuniformity-critical-max",
            "0.2",
            "--out-json",
            str(ext_reg_json),
            "--out-md",
            str(pub_dir / "deep_check_regimes_external_canonical.md"),
        ]
    )
    _run(
        [
            py,
            "scripts/deep_check_regimes.py",
            "--run-dirs",
            *(strict_run_dirs + ext_run_dirs),
            "--healthy-rul-floor",
            "70",
            "--require-surface-topology",
            "--superuniformity-levels",
            "0.1,0.2,0.5",
            "--superuniformity-fail-excess",
            "0.02",
            "--superuniformity-critical-max",
            "0.2",
            "--out-json",
            str(plus_reg_json),
            "--out-md",
            str(pub_dir / "deep_check_regimes_strict_plus_external.md"),
        ]
    )

    # Readiness / consistency / inspect.
    _run(
        [
            py,
            "scripts/stats_conference_readiness.py",
            "--report-json",
            str(args.matrix_report),
            "--topology-json",
            str(args.topology_json),
            "--paper-md",
            str(args.paper_md),
            "--strict-regimes-json",
            str(strict_reg_json),
            "--out-json",
            str(readiness_json),
            "--out-md",
            str(readiness_md),
        ]
    )
    _run(
        [
            py,
            "scripts/check_artifact_consistency.py",
            "--external-performance-report",
            str(args.external_performance_report),
            "--external-dataset-summary",
            str(ext_summary_json),
            "--baseline-comparison",
            "outputs/baseline_comparison.json",
            "--out-json",
            str(consistency_json),
            "--out-md",
            str(consistency_md),
        ]
    )
    _run(
        [
            py,
            "scripts/phd_deep_inspect.py",
            "--run-dirs",
            *(strict_run_dirs + ext_run_dirs),
            "--out-json",
            str(inspect_json),
            "--out-md",
            str(inspect_md),
        ]
    )
    _run(
        [
            py,
            "scripts/audit_publication_suspicious_values.py",
            "--strict-main-root",
            str(strict_root),
            "--external-report",
            str(args.external_performance_report),
            "--out-json",
            str(suspicious_json),
            "--out-md",
            str(suspicious_md),
        ]
    )

    strict_deep = _load_json(strict_deep_json)
    ext_deep = _load_json(ext_deep_json)
    full_cur = _load_json(full_curated_json)
    strict_reg = _load_json(strict_reg_json)
    ext_reg = _load_json(ext_reg_json)
    plus_reg = _load_json(plus_reg_json)
    readiness = _load_json(readiness_json)
    consistency = _load_json(consistency_json)
    ext_summary = _load_json(ext_summary_json)
    suspicious = _load_json(suspicious_json)

    checks = [
        {"name": "deep_check_strict_main", "pass": int(strict_deep.get("num_findings", 1)) == 0, "detail": f"findings={strict_deep.get('num_findings')}"},
        {"name": "deep_check_external_canonical", "pass": int(ext_deep.get("num_findings", 1)) == 0, "detail": f"findings={ext_deep.get('num_findings')}"},
        {"name": "deep_check_full_curated", "pass": int(full_cur.get("num_findings", 1)) == 0, "detail": f"findings={full_cur.get('num_findings')}"},
        {"name": "regimes_strict_main", "pass": int(strict_reg.get("num_findings_total", 1)) == 0, "detail": f"findings={strict_reg.get('num_findings_total')}"},
        {"name": "regimes_external_canonical", "pass": int(ext_reg.get("num_findings_total", 1)) == 0, "detail": f"findings={ext_reg.get('num_findings_total')}"},
        {"name": "regimes_strict_plus_external", "pass": int(plus_reg.get("num_findings_total", 1)) == 0, "detail": f"findings={plus_reg.get('num_findings_total')}"},
        {
            "name": "readiness_scores",
            "pass": float(readiness.get("score_10", 0.0)) >= 9.0 and float(readiness.get("score_10_conservative", 0.0)) >= 9.0,
            "detail": f"score={readiness.get('score_10')}, conservative={readiness.get('score_10_conservative')}, penalties={len(readiness.get('risk_penalties', []))}",
        },
        {
            "name": "artifact_consistency",
            "pass": bool(consistency.get("passed", False)),
            "detail": f"passed={consistency.get('passed')}, mismatches={consistency.get('num_mismatches')}",
        },
        {
            "name": "suspicious_values_audit",
            "pass": int(suspicious.get("summary", {}).get("num_high", 1)) == 0,
            "detail": (
                f"high={suspicious.get('summary', {}).get('num_high')}, "
                f"medium={suspicious.get('summary', {}).get('num_medium')}, "
                f"findings={suspicious.get('summary', {}).get('num_findings')}"
            ),
        },
    ]

    ext_perf = ext_summary.get("real_external_performance", {})
    ext_rows = list(ext_perf.get("datasets", [])) if isinstance(ext_perf.get("datasets", []), list) else []
    snapshot = {
        "readiness_score": float(readiness.get("score_10", 0.0)),
        "readiness_score_conservative": float(readiness.get("score_10_conservative", 0.0)),
        "readiness_penalties": int(len(readiness.get("risk_penalties", []))),
        "external_num_ok": int(ext_perf.get("num_ok", 0)),
        "external_num_total": int(ext_perf.get("num_total", 0)),
        "external_datasets": [str(r.get("dataset", "unknown")) for r in ext_rows],
        "suspicious_values_high": int(suspicious.get("summary", {}).get("num_high", 0)),
        "suspicious_values_medium": int(suspicious.get("summary", {}).get("num_medium", 0)),
    }

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "matrix_report": str(args.matrix_report),
            "topology_json": str(args.topology_json),
            "paper_md": str(args.paper_md),
            "rul_dataset_summary": str(args.rul_dataset_summary),
            "external_performance_report": str(args.external_performance_report),
            "strict_main_root": str(strict_root),
            "external_root": str(ext_root),
            "pub_dir": str(pub_dir),
            "suspicious_audit_json": str(suspicious_json),
        },
        "checks": checks,
        "overall_pass": bool(all(bool(c["pass"]) for c in checks)),
        "snapshot": snapshot,
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    out_md = Path(args.out_md)
    _write_md(out_md, report)

    print(f"Saved JSON: {out_json}")
    print(f"Saved Markdown: {out_md}")
    print(f"Overall pass: {report['overall_pass']}")


if __name__ == "__main__":
    main()

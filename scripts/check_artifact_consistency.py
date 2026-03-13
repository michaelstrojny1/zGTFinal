from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check cross-artifact consistency for publication outputs.")
    p.add_argument("--external-performance-report", type=str, default="outputs/external_performance_report.json")
    p.add_argument("--external-dataset-summary", type=str, default="outputs/external_dataset_summary.json")
    p.add_argument("--baseline-comparison", type=str, default="outputs/baseline_comparison.json")
    p.add_argument("--paper-provenance", type=str, default="paper/generated/provenance.json")
    p.add_argument("--retrain-policy-sweep-json", type=str, default="outputs/external_policy_replay_sweep_retrain_v3/summary.json")
    p.add_argument("--out-json", type=str, default="outputs/artifact_consistency_report.json")
    p.add_argument("--out-md", type=str, default="outputs/artifact_consistency_report.md")
    p.add_argument("--fail-on-mismatch", action="store_true")
    p.add_argument("--atol", type=float, default=1e-9)
    return p.parse_args()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _eq_num(a: Any, b: Any, atol: float) -> bool:
    try:
        fa = float(a)
        fb = float(b)
    except Exception:
        return a == b
    if not math.isfinite(fa) or not math.isfinite(fb):
        return a == b
    return bool(abs(fa - fb) <= float(atol))


def _add_mismatch(mismatches: list[dict[str, str]], *, severity: str, check: str, message: str) -> None:
    mismatches.append({"severity": severity, "check": check, "message": message})


def _resolve_maybe_relative(raw: str, *, base: Path) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = (base / path).resolve()
    else:
        path = path.resolve()
    return path


def _write_md(path: Path, rep: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# Artifact Consistency Report")
    lines.append("")
    lines.append(f"- Passed: {'YES' if rep['passed'] else 'NO'}")
    lines.append(f"- Checks run: {rep['num_checks']}")
    lines.append(f"- Mismatches: {rep['num_mismatches']}")
    lines.append("")
    lines.append("## Findings")
    if not rep["mismatches"]:
        lines.append("- none")
    else:
        for m in rep["mismatches"]:
            lines.append(f"- [{m['severity']}] {m['check']}: {m['message']}")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    ext_perf_path = Path(args.external_performance_report).resolve()
    ext_sum_path = Path(args.external_dataset_summary).resolve()
    comp_path = Path(args.baseline_comparison).resolve()
    provenance_path = Path(args.paper_provenance).resolve()
    retrain_sweep_path = Path(args.retrain_policy_sweep_json).resolve()
    atol = float(args.atol)

    mismatches: list[dict[str, str]] = []
    checks = 0

    if ext_perf_path.exists() and ext_sum_path.exists():
        ext_perf = _load(ext_perf_path)
        ext_sum = _load(ext_sum_path)
        perf_rows = {str(r.get("dataset", "")).lower(): r for r in ext_perf.get("datasets", [])}
        sum_rows_raw = ext_sum.get("real_external_performance", {}).get("datasets", [])
        sum_rows = {str(r.get("dataset", "")).lower(): r for r in sum_rows_raw}

        for ds_name, p_row in perf_rows.items():
            checks += 1
            if ds_name not in sum_rows:
                _add_mismatch(
                    mismatches,
                    severity="high",
                    check="external_dataset_presence",
                    message=f"{ds_name} present in external_performance_report but missing in external_dataset_summary.",
                )
                continue

            s_row = sum_rows[ds_name]
            p_status = str(p_row.get("status", "unknown")).lower()
            s_status = str(s_row.get("status", "unknown")).lower()
            checks += 1
            if p_status != s_status:
                _add_mismatch(
                    mismatches,
                    severity="high",
                    check="external_status_match",
                    message=f"{ds_name} status mismatch: perf={p_status}, summary={s_status}.",
                )

            if p_status == "ok":
                p_metrics = p_row.get("metrics", {}) if isinstance(p_row.get("metrics", {}), dict) else {}
                s_metrics = s_row.get("metrics", {}) if isinstance(s_row.get("metrics", {}), dict) else {}
                for key in ("rul_cov", "tau_v", "rmse", "rmse_last", "mae_last"):
                    checks += 1
                    pv = p_metrics.get(key)
                    sv = s_metrics.get(key)
                    if pv is None and sv is None:
                        continue
                    if not _eq_num(pv, sv, atol=atol):
                        _add_mismatch(
                            mismatches,
                            severity="high",
                            check="external_metric_match",
                            message=f"{ds_name} metric `{key}` mismatch: perf={pv}, summary={sv}.",
                        )
    else:
        if not ext_perf_path.exists():
            _add_mismatch(
                mismatches,
                severity="medium",
                check="external_report_exists",
                message=f"missing file: {ext_perf_path}",
            )
        if not ext_sum_path.exists():
            _add_mismatch(
                mismatches,
                severity="medium",
                check="external_summary_exists",
                message=f"missing file: {ext_sum_path}",
            )

    if comp_path.exists():
        comp = _load(comp_path)
        methods = list(comp.get("methods", []))
        ext_method_count = sum(1 for m in methods if str(m.get("comparator_type", "")) == "external")
        csum = comp.get("comparator_summary", {}) if isinstance(comp.get("comparator_summary", {}), dict) else {}
        num_external = int(csum.get("num_external", -1))
        checks += 1
        if num_external != ext_method_count:
            _add_mismatch(
                mismatches,
                severity="medium",
                check="baseline_external_count",
                message=(
                    "baseline_comparison comparator_summary.num_external "
                    f"({num_external}) != actual external methods ({ext_method_count})."
                ),
            )
    else:
        _add_mismatch(
            mismatches,
            severity="medium",
            check="baseline_comparison_exists",
            message=f"missing file: {comp_path}",
        )

    if provenance_path.exists():
        provenance = _load(provenance_path)
        required_path_keys = (
            "canonical_report",
            "balanced_report",
            "aggressive_report",
            "retrain_robustness_report",
            "aux_policy_report",
            "baseline_json",
            "readiness_json",
            "gate_summary_json",
            "claim_significance_json",
            "policy_sharpness_json",
            "retrain_policy_sweep_json",
            "policy_frontier_fig",
            "policy_sharpness_fig",
        )
        for key in required_path_keys:
            raw = str(provenance.get(key, "")).strip()
            if not raw:
                continue
            checks += 1
            ref_path = _resolve_maybe_relative(raw, base=provenance_path.parent)
            if not ref_path.exists():
                _add_mismatch(
                    mismatches,
                    severity="high",
                    check="paper_provenance_ref_exists",
                    message=f"provenance key `{key}` points to missing path: {ref_path}",
                )
        aux_label = str(provenance.get("aux_policy_label", "")).strip()
        aux_report = str(provenance.get("aux_policy_report", "")).strip()
        checks += 1
        if aux_label and not aux_report:
            _add_mismatch(
                mismatches,
                severity="high",
                check="paper_provenance_aux_policy",
                message=f"aux_policy_label `{aux_label}` is set but aux_policy_report is empty.",
            )
    else:
        _add_mismatch(
            mismatches,
            severity="medium",
            check="paper_provenance_exists",
            message=f"missing file: {provenance_path}",
        )

    if retrain_sweep_path.exists():
        sweep = _load(retrain_sweep_path)
        rows = list(sweep.get("rows_sorted", []))
        settings = sweep.get("settings", {}) if isinstance(sweep.get("settings", {}), dict) else {}
        checks += 1
        if int(settings.get("num_points", len(rows))) != len(rows):
            _add_mismatch(
                mismatches,
                severity="high",
                check="retrain_policy_sweep_num_points",
                message=f"settings.num_points={settings.get('num_points')} != len(rows_sorted)={len(rows)}.",
            )

        valid_rows = [r for r in rows if bool(r.get("validity_ok", False))]
        checks += 1
        if int(sweep.get("num_valid_points", len(valid_rows))) != len(valid_rows):
            _add_mismatch(
                mismatches,
                severity="high",
                check="retrain_policy_sweep_num_valid_points",
                message=f"num_valid_points={sweep.get('num_valid_points')} != counted valid rows={len(valid_rows)}.",
            )

        if valid_rows:
            range_blob = sweep.get("valid_width_range", {}) if isinstance(sweep.get("valid_width_range", {}), dict) else {}
            min_width = min(float(r.get("width_mean", math.inf)) for r in valid_rows)
            max_width = max(float(r.get("width_mean", -math.inf)) for r in valid_rows)
            checks += 1
            if not _eq_num(range_blob.get("min"), min_width, atol=max(atol, 1e-6)):
                _add_mismatch(
                    mismatches,
                    severity="high",
                    check="retrain_policy_sweep_valid_width_min",
                    message=f"valid_width_range.min={range_blob.get('min')} != actual valid min width={min_width}.",
                )
            checks += 1
            if not _eq_num(range_blob.get("max"), max_width, atol=max(atol, 1e-6)):
                _add_mismatch(
                    mismatches,
                    severity="high",
                    check="retrain_policy_sweep_valid_width_max",
                    message=f"valid_width_range.max={range_blob.get('max')} != actual valid max width={max_width}.",
                )

        best = sweep.get("best_policy", {}) if isinstance(sweep.get("best_policy", {}), dict) else {}
        if best:
            best_tag = str(best.get("tag", "")).strip()
            checks += 1
            if best_tag and not any(str(r.get("tag", "")).strip() == best_tag for r in rows):
                _add_mismatch(
                    mismatches,
                    severity="high",
                    check="retrain_policy_sweep_best_policy_tag",
                    message=f"best_policy tag `{best_tag}` is missing from rows_sorted.",
                )
            if valid_rows and bool(best.get("validity_ok", False)):
                min_width = min(float(r.get("width_mean", math.inf)) for r in valid_rows)
                checks += 1
                if not _eq_num(best.get("width_mean"), min_width, atol=max(atol, 1e-6)):
                    _add_mismatch(
                        mismatches,
                        severity="high",
                        check="retrain_policy_sweep_best_width",
                        message=f"best_policy width_mean={best.get('width_mean')} != optimal valid width={min_width}.",
                    )

        report_refs: set[str] = set()
        top_level_best_report = str(sweep.get("best_policy_report_json", "")).strip()
        if top_level_best_report:
            report_refs.add(top_level_best_report)
        nested_best_report = str(best.get("report_json", "")).strip() if best else ""
        if nested_best_report:
            report_refs.add(nested_best_report)
        for row in rows:
            raw = str(row.get("report_json", "")).strip()
            if raw:
                report_refs.add(raw)
        for raw in sorted(report_refs):
            checks += 1
            ref_path = _resolve_maybe_relative(raw, base=retrain_sweep_path.parent)
            if not ref_path.exists():
                _add_mismatch(
                    mismatches,
                    severity="high",
                    check="retrain_policy_sweep_report_ref",
                    message=f"sweep summary references missing report path: {ref_path}",
                )
    else:
        _add_mismatch(
            mismatches,
            severity="medium",
            check="retrain_policy_sweep_exists",
            message=f"missing file: {retrain_sweep_path}",
        )

    passed = len(mismatches) == 0
    out = {
        "inputs": {
            "external_performance_report": str(ext_perf_path),
            "external_dataset_summary": str(ext_sum_path),
            "baseline_comparison": str(comp_path),
            "paper_provenance": str(provenance_path),
            "retrain_policy_sweep_json": str(retrain_sweep_path),
        },
        "passed": bool(passed),
        "num_checks": int(checks),
        "num_mismatches": int(len(mismatches)),
        "mismatches": mismatches,
    }

    out_json = Path(args.out_json).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2, allow_nan=False), encoding="utf-8")
    out_md = Path(args.out_md).resolve()
    _write_md(out_md, out)

    print(f"Saved JSON: {out_json}")
    print(f"Saved Markdown: {out_md}")
    print(f"Consistency passed: {passed}")
    if args.fail_on_mismatch and not passed:
        sys.exit(2)


if __name__ == "__main__":
    main()

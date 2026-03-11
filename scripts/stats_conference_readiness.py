from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit 9+ stats-conference readiness from existing TEM artifacts.")
    p.add_argument("--report-json", type=str, default="outputs/publication_full_rtx4050_report.json")
    p.add_argument("--topology-json", type=str, default="outputs/publication_full_rtx4050/topology_rul_landscape.json")
    p.add_argument("--paper-md", type=str, default="paper/topological_evidence_curves.md")
    p.add_argument("--out-json", type=str, default="outputs/publication_full_rtx4050/stats_conference_readiness.json")
    p.add_argument("--out-md", type=str, default="outputs/publication_full_rtx4050/stats_conference_readiness.md")
    p.add_argument("--min-unique-topology-families", type=int, default=2)
    p.add_argument("--min-external-baselines", type=int, default=2)
    p.add_argument("--min-real-external-eval-datasets", type=int, default=1)
    p.add_argument("--min-tau-identifiability-ratio", type=float, default=0.75)
    p.add_argument(
        "--tau-identifiability-deficit-tolerance",
        type=float,
        default=0.03,
        help="Allow small per-FD shortfall below min-tau-identifiability-ratio when pooled ratio is strong.",
    )
    p.add_argument(
        "--max-tau-identifiability-severe-fails",
        type=int,
        default=0,
        help="Maximum count of severe per-FD shortfalls allowed for tau-identifiability gate.",
    )
    p.add_argument(
        "--strict-regimes-json",
        type=str,
        default="outputs/publication_full_rtx4050/deep_check_regimes_stricter_strict_main.json",
        help=(
            "Optional strict regime report from scripts/deep_check_regimes.py. "
            "If provided, nonzero findings contribute to conservative-score penalties."
        ),
    )
    p.add_argument(
        "--strict-regime-penalty",
        type=float,
        default=0.5,
        help="Penalty applied to conservative score when strict regime findings are present.",
    )
    p.add_argument(
        "--tau-identifiability-fail-penalty",
        type=float,
        default=0.5,
        help="Penalty applied to conservative score when any per-FD tau-identifiability shortfall exists.",
    )
    p.add_argument(
        "--overconservative-external-penalty",
        type=float,
        default=0.25,
        help="Penalty applied when all external datasets show near-perfect coverage/tau (possible over-conservativeness).",
    )
    p.add_argument(
        "--overconservative-max-frac-le-0p5",
        type=float,
        default=0.05,
        help=(
            "Apply external over-conservative penalty only when all near-perfect external datasets "
            "also show very high p-values in audit logs (pvalue_all.frac_le_0.5 <= threshold)."
        ),
    )
    return p.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fd_group(rows: list[dict[str, Any]], fd: int) -> list[dict[str, Any]]:
    return [r for r in rows if int(r.get("fd", -1)) == int(fd)]


def _ci_excludes_zero(ci_low: float, ci_high: float) -> bool:
    return (ci_low > 0.0 and ci_high > 0.0) or (ci_low < 0.0 and ci_high < 0.0)


def _gate(name: str, passed: bool, weight: float, details: str) -> dict[str, Any]:
    return {
        "name": name,
        "pass": bool(passed),
        "weight": float(weight),
        "score": float(weight if passed else 0.0),
        "details": str(details),
    }


def _topology_family(name: str) -> str:
    n = name.lower()
    if "surface_h1" in n or "surface_superlevel_h1" in n or "surface_sublevel_h1" in n:
        return "surface_h1"
    if "persistent_valleys" in n:
        return "persistent_valleys"
    if "ridge_tv" in n:
        return "ridge_tv"
    if "mean_gamma" in n:
        return "mean_gamma"
    if "local_minima" in n:
        return "local_minima"
    return "other"


def _safe_float(v: float | np.floating | None) -> float | None:
    if v is None:
        return None
    fv = float(v)
    if not np.isfinite(fv):
        return None
    return fv


def _safe_ratio(numer: int, denom: int) -> float | None:
    if int(denom) <= 0:
        return None
    return float(int(numer) / int(denom))


def _sanitize_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_json(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        fv = float(obj)
        if not np.isfinite(fv):
            return None
        return fv
    return obj


def _write_md(path: Path, report: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# Stats Conference Readiness")
    lines.append("")
    lines.append(f"- Score: {report['score_10']:.2f} / 10.00")
    lines.append(f"- Target >= 9.00: {'PASS' if report['target_9plus_pass'] else 'FAIL'}")
    lines.append(f"- Conservative score: {report['score_10_conservative']:.2f} / 10.00")
    lines.append(
        f"- Conservative target >= 9.00: {'PASS' if report['target_9plus_pass_conservative'] else 'FAIL'}"
    )
    lines.append(f"- Source report: `{report['inputs']['report_json']}`")
    lines.append("")
    lines.append("## Gates")
    for g in report["gates"]:
        status = "PASS" if g["pass"] else "FAIL"
        lines.append(f"- [{status}] {g['name']} ({g['score']:.2f}/{g['weight']:.2f}): {g['details']}")
    if report.get("risk_penalties"):
        lines.append("")
        lines.append("## Conservative Penalties")
        for row in report["risk_penalties"]:
            lines.append(f"- {row['name']}: -{float(row['penalty']):.2f} ({row['reason']})")
    lines.append("")
    lines.append("## Immediate Priorities")
    for i, item in enumerate(report["priorities"], start=1):
        lines.append(f"{i}. {item}")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    report_path = Path(args.report_json).resolve()
    topo_path = Path(args.topology_json).resolve()
    paper_path = Path(args.paper_md).resolve()
    strict_regimes_path = Path(args.strict_regimes_json).resolve() if str(args.strict_regimes_json).strip() else None

    matrix = _load_json(report_path)
    topo = _load_json(topo_path)
    paper_text = paper_path.read_text(encoding="utf-8") if paper_path.exists() else ""
    strict_regimes = _load_json(strict_regimes_path) if (strict_regimes_path is not None and strict_regimes_path.exists()) else None

    deep = matrix.get("deep_checks", {})
    baseline = list(matrix.get("baseline", []))
    seed_rows = list(matrix.get("seed_repro", {}).get("rows", []))
    split_rows = list(matrix.get("split_robustness", []))
    suspicious_count = int(matrix.get("suspicious", {}).get("count", 0))

    fds = sorted({int(r.get("fd")) for r in baseline})
    min_cov = float(min(float(r.get("rul_cov", 0.0)) for r in baseline)) if baseline else float("nan")
    max_tau = float(max(float(r.get("tau_v", 0.0)) for r in baseline)) if baseline else float("nan")

    # Tau-identifiability coverage from strict baseline TEM files.
    tau_identifiability_by_fd: list[dict[str, Any]] = []
    tau_ratio_values: list[float] = []
    tau_ratio_fail_fds: list[int] = []
    tau_ratio_severe_fail_fds: list[int] = []
    tau_ratio_deficits: list[dict[str, Any]] = []
    tau_numer_total = 0
    tau_denom_total = 0
    for row in baseline:
        fd = int(row["fd"])
        run_dir = Path(str(row.get("run_dir", "")))
        tem_path = run_dir / f"tem_metrics_fd{fd:03d}.json"
        n_tau = 0
        n_total = 0
        ratio = None
        if tem_path.exists():
            tem = _load_json(tem_path)
            fleet = tem.get("fleet_summary", {})
            n_tau = int(fleet.get("num_tau_diagnostics_engines", 0))
            n_total = int(fleet.get("num_engines", 0))
            ratio = _safe_ratio(n_tau, n_total)
        tau_identifiability_by_fd.append(
            {
                "fd": fd,
                "run_dir": str(run_dir),
                "num_tau_diagnostics_engines": int(n_tau),
                "num_engines": int(n_total),
                "tau_identifiability_ratio": ratio,
                "pass_gate": bool(ratio is not None and ratio >= float(args.min_tau_identifiability_ratio)),
            }
        )
        if ratio is not None:
            tau_ratio_values.append(float(ratio))
            tau_numer_total += int(n_tau)
            tau_denom_total += int(n_total)
            deficit = float(args.min_tau_identifiability_ratio) - float(ratio)
            tau_ratio_deficits.append({"fd": int(fd), "deficit_vs_threshold": float(max(0.0, deficit))})
            if ratio < float(args.min_tau_identifiability_ratio):
                tau_ratio_fail_fds.append(fd)
            if ratio < (float(args.min_tau_identifiability_ratio) - float(args.tau_identifiability_deficit_tolerance)):
                tau_ratio_severe_fail_fds.append(fd)
        else:
            tau_ratio_severe_fail_fds.append(fd)
    min_tau_ident_ratio = float(min(tau_ratio_values)) if tau_ratio_values else float("nan")
    pooled_tau_ident_ratio = _safe_ratio(tau_numer_total, tau_denom_total)
    tau_ident_gate_pass = bool(
        pooled_tau_ident_ratio is not None
        and float(pooled_tau_ident_ratio) >= float(args.min_tau_identifiability_ratio)
        and len(tau_ratio_severe_fail_fds) <= int(args.max_tau_identifiability_severe_fails)
    )

    # Seed stability statistics.
    seed_fd_stats: list[dict[str, Any]] = []
    seed_fail_fds: list[int] = []
    for fd in fds:
        rows = _fd_group(seed_rows, fd)
        tau = np.asarray([float(r.get("tau_v", np.nan)) for r in rows], dtype=np.float64)
        cov = np.asarray([float(r.get("rul_cov", np.nan)) for r in rows], dtype=np.float64)
        if tau.size == 0:
            continue
        tau_max = float(np.nanmax(tau))
        tau_min = float(np.nanmin(tau))
        cov_std = float(np.nanstd(cov, ddof=1)) if cov.size > 1 else 0.0
        ok = bool(tau_max <= 0.12 and cov_std <= 0.02 and tau.size >= 3)
        if not ok:
            seed_fail_fds.append(int(fd))
        seed_fd_stats.append(
            {
                "fd": int(fd),
                "n": int(tau.size),
                "tau_v_min": tau_min,
                "tau_v_max": tau_max,
                "rul_cov_std": cov_std,
                "pass_gate": ok,
            }
        )

    # Split robustness.
    split_deltas: list[dict[str, Any]] = []
    split_pass = True
    for fd in fds:
        val_rows = [r for r in split_rows if int(r.get("fd", -1)) == fd and str(r.get("calibration_source", "")) == "val"]
        dev_rows = [r for r in split_rows if int(r.get("fd", -1)) == fd and str(r.get("calibration_source", "")) == "dev_holdout"]
        if not val_rows or not dev_rows:
            split_pass = False
            continue
        vr, dr = val_rows[0], dev_rows[0]
        d_cov = float(vr.get("rul_cov", 0.0)) - float(dr.get("rul_cov", 0.0))
        d_tau = float(vr.get("tau_v", 0.0)) - float(dr.get("tau_v", 0.0))
        d_rmse = float(vr.get("rmse", 0.0)) - float(dr.get("rmse", 0.0))
        ok = bool(abs(d_cov) <= 0.03 and abs(d_tau) <= 0.05)
        split_pass = bool(split_pass and ok)
        split_deltas.append(
            {
                "fd": int(fd),
                "delta_cov_val_minus_dev": d_cov,
                "delta_tau_val_minus_dev": d_tau,
                "delta_rmse_val_minus_dev": d_rmse,
                "pass_gate": ok,
            }
        )

    # Topology evidence strength with family de-duplication.
    assoc = topo.get("associations", {})
    topo_hits_raw: list[dict[str, Any]] = []
    for name, blob in assoc.items():
        corr = float(blob.get("corr", 0.0))
        ci_low = float(blob.get("ci_low", 0.0))
        ci_high = float(blob.get("ci_high", 0.0))
        if abs(corr) >= 0.25 and _ci_excludes_zero(ci_low, ci_high):
            topo_hits_raw.append(
                {
                    "name": str(name),
                    "family": _topology_family(str(name)),
                    "corr": corr,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                }
            )
    # Keep strongest hit per family to avoid counting near-duplicates as independent evidence.
    best_by_family: dict[str, dict[str, Any]] = {}
    for hit in topo_hits_raw:
        fam = str(hit["family"])
        prev = best_by_family.get(fam)
        if prev is None or abs(float(hit["corr"])) > abs(float(prev["corr"])):
            best_by_family[fam] = hit
    topo_hits = [best_by_family[k] for k in sorted(best_by_family.keys())]
    topo_unique_families = sorted(best_by_family.keys())
    topo_pass = bool(len(topo_unique_families) >= int(args.min_unique_topology_families))

    # Baseline comparator gate: require external baselines, not only internal ablations.
    baseline_comp_path = report_path.parent / "baseline_comparison.json"
    baseline_comp_detail = f"requires artifact {baseline_comp_path}"
    has_baseline_comparison = False
    external_baselines_found = 0
    if baseline_comp_path.exists():
        try:
            comp = _load_json(baseline_comp_path)
            csum = comp.get("comparator_summary", {}) if isinstance(comp.get("comparator_summary", {}), dict) else {}
            if "num_external" in csum:
                external_baselines_found = int(csum.get("num_external", 0))
            else:
                methods = list(comp.get("methods", []))
                external_baselines_found = int(sum(1 for m in methods if str(m.get("comparator_type", "")) == "external"))
            has_baseline_comparison = bool(external_baselines_found >= int(args.min_external_baselines))
            baseline_comp_detail = (
                f"artifact={baseline_comp_path}; external_baselines={external_baselines_found}; "
                f"required>={int(args.min_external_baselines)}"
            )
        except Exception:
            has_baseline_comparison = False
            baseline_comp_detail = f"artifact exists but parse failed: {baseline_comp_path}"

    # External generalization gate: require real external model-evaluation results.
    external_dataset_path = report_path.parent / "external_dataset_summary.json"
    external_eval_ok = False
    external_eval_count = 0
    external_avail_ok = 0
    external_avail_total = 0
    external_metric_alerts: list[dict[str, Any]] = []
    external_near_perfect_count = 0
    external_num_ok_datasets = 0
    external_pvalue_overconservative_rows: list[dict[str, Any]] = []
    external_num_audited_datasets = 0
    external_overconservative_strong_evidence = False
    external_detail = f"requires artifact {external_dataset_path}"
    if external_dataset_path.exists():
        try:
            ext = _load_json(external_dataset_path)
            real_avail = ext.get("real_rul_datasets", {})
            external_avail_ok = int(real_avail.get("num_ok", 0))
            external_avail_total = int(real_avail.get("num_total", 0))
            real_eval = ext.get("real_external_performance", {})
            external_eval_count = int(real_eval.get("num_ok", 0))
            external_eval_ok = bool(external_eval_count >= int(args.min_real_external_eval_datasets))
            ext_rows = [d for d in list(real_eval.get("datasets", [])) if str(d.get("status", "")).lower() == "ok"]
            external_num_ok_datasets = int(len(ext_rows))
            for d in ext_rows:
                metrics = d.get("metrics", {}) if isinstance(d.get("metrics", {}), dict) else {}
                rul_cov = _safe_float(metrics.get("rul_cov"))
                tau_v = _safe_float(metrics.get("tau_v"))
                if rul_cov is not None and tau_v is not None and float(rul_cov) >= 0.999 and float(tau_v) <= 0.001:
                    external_near_perfect_count += 1
            summary_alerts = list(real_eval.get("metric_alerts", []))
            if summary_alerts:
                external_metric_alerts = summary_alerts
            else:
                for d in ext_rows:
                    if str(d.get("status", "")).lower() != "ok":
                        continue
                    ds_name = str(d.get("dataset", "unknown"))
                    metrics = d.get("metrics", {}) if isinstance(d.get("metrics", {}), dict) else {}
                    rmse_last = _safe_float(metrics.get("rmse_last"))
                    rul_cov = _safe_float(metrics.get("rul_cov"))
                    tau_v = _safe_float(metrics.get("tau_v"))
                    if rmse_last is not None and float(rmse_last) > 80.0:
                        external_metric_alerts.append(
                            {
                                "dataset": ds_name,
                                "severity": "medium",
                                "message": f"High terminal-window RMSE ({float(rmse_last):.3f}).",
                            }
                        )
                    if rul_cov is not None and tau_v is not None and float(rul_cov) >= 0.999 and float(tau_v) <= 0.001:
                        external_metric_alerts.append(
                            {
                                "dataset": ds_name,
                                "severity": "info",
                                "message": "Near-perfect coverage/tau may indicate conservative policy settings.",
                            }
                        )

            # Refine over-conservative evidence using external audit p-value shape
            # when source artifacts are available.
            source_path_raw = str(real_eval.get("source", "")).strip()
            if source_path_raw:
                source_path = Path(source_path_raw)
                if source_path.exists():
                    try:
                        source_blob = _load_json(source_path)
                        source_rows = [r for r in list(source_blob.get("datasets", [])) if str(r.get("status", "")).lower() == "ok"]
                        for d in ext_rows:
                            ds_name = str(d.get("dataset", "unknown"))
                            src = next((r for r in source_rows if str(r.get("dataset", "")).lower() == ds_name.lower()), None)
                            fd_val = int(src.get("fd", 1)) if isinstance(src, dict) else 1
                            frac_le_0p5 = None
                            audit_path = None
                            if isinstance(src, dict):
                                arts = src.get("artifacts", {}) if isinstance(src.get("artifacts", {}), dict) else {}
                                run_dir_raw = str(arts.get("run_dir", "")).strip()
                                if run_dir_raw:
                                    run_dir = Path(run_dir_raw)
                                    candidate = run_dir / f"audit_fd{fd_val:03d}.json"
                                    if candidate.exists():
                                        audit_path = str(candidate)
                                        audit = _load_json(candidate)
                                        pvalue_all = audit.get("pvalue_all", {}) if isinstance(audit.get("pvalue_all", {}), dict) else {}
                                        frac_le_0p5 = _safe_float(pvalue_all.get("frac_le_0.5"))
                            external_pvalue_overconservative_rows.append(
                                {
                                    "dataset": ds_name,
                                    "fd": int(fd_val),
                                    "audit_path": audit_path if audit_path else "",
                                    "pvalue_frac_le_0p5": frac_le_0p5,
                                }
                            )
                    except Exception:
                        external_pvalue_overconservative_rows = []

            external_num_audited_datasets = int(
                sum(1 for r in external_pvalue_overconservative_rows if r.get("pvalue_frac_le_0p5") is not None)
            )
            all_info_alerts = bool(
                external_metric_alerts and all(str(a.get("severity", "")).lower() == "info" for a in external_metric_alerts)
            )
            all_near_perfect = bool(
                external_num_ok_datasets > 0 and external_near_perfect_count == external_num_ok_datasets
            )
            if all_near_perfect and all_info_alerts:
                # If audit p-value diagnostics are available for every dataset,
                # only penalize when all datasets show strongly high p-values.
                if external_num_audited_datasets == external_num_ok_datasets and external_num_ok_datasets > 0:
                    external_overconservative_strong_evidence = bool(
                        all(
                            (r.get("pvalue_frac_le_0p5") is not None)
                            and float(r.get("pvalue_frac_le_0p5")) <= float(args.overconservative_max_frac_le_0p5)
                            for r in external_pvalue_overconservative_rows
                        )
                    )
                else:
                    # Backward-compatible fallback when audit diagnostics are unavailable.
                    external_overconservative_strong_evidence = True
            external_detail = (
                f"artifact={external_dataset_path}; availability_ok={external_avail_ok}/{external_avail_total}; "
                f"external_eval_ok={external_eval_count}; required>={int(args.min_real_external_eval_datasets)}; "
                f"metric_alerts={len(external_metric_alerts)}; "
                f"near_perfect={external_near_perfect_count}/{external_num_ok_datasets}; "
                f"strong_overconservative_evidence={external_overconservative_strong_evidence}; "
                f"audited={external_num_audited_datasets}/{external_num_ok_datasets}"
            )
        except Exception:
            external_eval_ok = False
            external_detail = f"artifact exists but parse failed: {external_dataset_path}"

    strict_regime_findings = None
    if isinstance(strict_regimes, dict):
        strict_regime_findings = int(strict_regimes.get("num_findings_total", 0))

    # Manuscript rigor heuristic.
    sketch_flags = ["(Sketch)", "Proof follows", "Sketch)", "sketch"]
    has_sketch_language = any(tok in paper_text for tok in sketch_flags)

    gates: list[dict[str, Any]] = []
    gates.append(
        _gate(
            name="Core Validity Artifacts",
            passed=(
                suspicious_count == 0
                and int(deep.get("deep_check_results_findings", 1)) == 0
                and int(deep.get("deep_check_regimes_findings", 1)) == 0
                and int(deep.get("deep_check_results_all_unexpected_findings", 1)) == 0
            ),
            weight=2.0,
            details=(
                f"suspicious={suspicious_count}, strict_deep={int(deep.get('deep_check_results_findings', -1))}, "
                f"regimes={int(deep.get('deep_check_regimes_findings', -1))}, "
                f"unexpected_all={int(deep.get('deep_check_results_all_unexpected_findings', -1))}"
            ),
        )
    )
    gates.append(
        _gate(
            name="Baseline Coverage/Tau Strength",
            passed=bool(
                min_cov >= 0.95
                and max_tau <= 0.10
                and tau_ident_gate_pass
            ),
            weight=1.5,
            details=(
                f"min_rul_cov={min_cov:.3f}, max_tau_v={max_tau:.3f}, "
                f"min_tau_ident_ratio={min_tau_ident_ratio:.3f}, "
                f"pooled_tau_ident_ratio={float(pooled_tau_ident_ratio) if pooled_tau_ident_ratio is not None else float('nan'):.3f}, "
                f"required>={float(args.min_tau_identifiability_ratio):.3f}, "
                f"deficit_tolerance={float(args.tau_identifiability_deficit_tolerance):.3f}, "
                f"severe_fail_fds={sorted(set(tau_ratio_severe_fail_fds))}"
            ),
        )
    )
    gates.append(
        _gate(
            name="Seed Robustness",
            passed=bool(len(seed_fail_fds) == 0 and len(seed_fd_stats) == len(fds)),
            weight=2.0,
            details=(
                "all FD require n>=3, tau_v_max<=0.12, rul_cov_std<=0.02; "
                f"failing_fds={seed_fail_fds if seed_fail_fds else 'none'}"
            ),
        )
    )
    gates.append(
        _gate(
            name="Split Robustness (val vs dev_holdout)",
            passed=split_pass,
            weight=1.0,
            details="require |delta_cov|<=0.03 and |delta_tau|<=0.05 per FD",
        )
    )
    gates.append(
        _gate(
            name="Topology Signal Strength",
            passed=topo_pass,
            weight=1.5,
            details=(
                f"raw_hits={len(topo_hits_raw)}, unique_families={len(topo_unique_families)} "
                f"(required>={int(args.min_unique_topology_families)}), families={topo_unique_families}"
            ),
        )
    )
    gates.append(
        _gate(
            name="Baseline Comparator Package",
            passed=has_baseline_comparison,
            weight=1.0,
            details=baseline_comp_detail,
        )
    )
    gates.append(
        _gate(
            name="External Dataset Generalization",
            passed=external_eval_ok,
            weight=0.5,
            details=external_detail,
        )
    )
    gates.append(
        _gate(
            name="Proof Maturity",
            passed=not has_sketch_language,
            weight=0.5,
            details="paper should not rely on sketch-only theorem statements",
        )
    )

    score_10 = float(sum(float(g["score"]) for g in gates))
    target_9plus_pass = bool(score_10 >= 9.0)

    # Conservative score applies explicit penalties for known high-risk caveats.
    risk_penalties: list[dict[str, Any]] = []
    if tau_ratio_fail_fds:
        risk_penalties.append(
            {
                "name": "per_fd_tau_identifiability_shortfall",
                "penalty": float(args.tau_identifiability_fail_penalty),
                "reason": f"failing_fds={sorted(set(tau_ratio_fail_fds))}",
            }
        )
    if strict_regime_findings is not None and strict_regime_findings > 0:
        risk_penalties.append(
            {
                "name": "strict_regime_findings",
                "penalty": float(args.strict_regime_penalty),
                "reason": f"num_findings_total={strict_regime_findings}",
            }
        )
    if external_overconservative_strong_evidence:
        risk_penalties.append(
            {
                "name": "external_overconservative_signal",
                "penalty": float(args.overconservative_external_penalty),
                "reason": (
                    f"near_perfect={external_near_perfect_count}/{external_num_ok_datasets}; "
                    f"metric_alerts={len(external_metric_alerts)}; "
                    f"audited={external_num_audited_datasets}/{external_num_ok_datasets}; "
                    f"max_frac_le_0p5={float(args.overconservative_max_frac_le_0p5):.3f}"
                ),
            }
        )
    total_penalty = float(sum(float(r["penalty"]) for r in risk_penalties))
    score_10_conservative = float(max(0.0, score_10 - total_penalty))
    target_9plus_pass_conservative = bool(score_10_conservative >= 9.0)

    priorities: list[str] = []
    if seed_fail_fds:
        priorities.append(
            "Run 10-seed strict reproducibility per FD and tune policy to push worst-case tau_v under 0.12 "
            f"(current failing FD: {seed_fail_fds})."
        )
    if tau_ratio_fail_fds:
        priorities.append(
            "Increase tau-identifiable coverage (labeling/diagnostic policy), prioritizing severe FD shortfalls. "
            f"Threshold={float(args.min_tau_identifiability_ratio):.2f}, "
            f"tolerance={float(args.tau_identifiability_deficit_tolerance):.2f}, "
            f"failing_fds={sorted(set(tau_ratio_fail_fds))}, "
            f"severe_fds={sorted(set(tau_ratio_severe_fail_fds))}."
        )
    if strict_regime_findings is not None and strict_regime_findings > 0:
        priorities.append(
            "Resolve strict regime superuniformity failures (especially low-RUL blocks) before camera-ready "
            f"(current strict findings={strict_regime_findings})."
        )
    if not has_baseline_comparison:
        priorities.append(
            "Add external baseline comparisons (not only internal ablations) and ensure "
            f"external_baselines >= {int(args.min_external_baselines)} in baseline_comparison.json."
        )
    if not external_eval_ok:
        priorities.append(
            "Add real external model-evaluation results (coverage/tau/rmse) and ensure "
            f"external_eval_ok >= {int(args.min_real_external_eval_datasets)} in external_dataset_summary.json."
        )
    if any(str(a.get("severity", "")).lower() == "medium" for a in external_metric_alerts):
        priorities.append(
            "Investigate external terminal-window performance failures (high rmse_last) and report calibrated "
            "failure-onset diagnostics separately from sequence-average RMSE."
        )
    if has_sketch_language:
        priorities.append(
            "Upgrade theorem section from sketch to full appendix-ready proofs with explicit assumptions and finite-sample constants."
        )
    if not topo_pass:
        priorities.append(
            "Strengthen topology claims with at least two independent topology families "
            "(not duplicated variants of the same surface statistic), plus multiplicity control."
        )
    if target_9plus_pass:
        priorities.append("Freeze this run as v1.0 artifact bundle and lock submission hash.")

    out = {
        "inputs": {
            "report_json": str(report_path),
            "topology_json": str(topo_path),
            "paper_md": str(paper_path),
        },
        "thresholds": {
            "min_unique_topology_families": int(args.min_unique_topology_families),
            "min_external_baselines": int(args.min_external_baselines),
            "min_real_external_eval_datasets": int(args.min_real_external_eval_datasets),
            "min_tau_identifiability_ratio": float(args.min_tau_identifiability_ratio),
            "tau_identifiability_deficit_tolerance": float(args.tau_identifiability_deficit_tolerance),
            "max_tau_identifiability_severe_fails": int(args.max_tau_identifiability_severe_fails),
        },
        "score_10": score_10,
        "target_9plus_pass": target_9plus_pass,
        "score_10_conservative": score_10_conservative,
        "target_9plus_pass_conservative": target_9plus_pass_conservative,
        "gates": gates,
        "seed_fd_stats": seed_fd_stats,
        "split_deltas": split_deltas,
        "tau_identifiability_by_fd": tau_identifiability_by_fd,
        "tau_identifiability_pooled_ratio": pooled_tau_ident_ratio,
        "tau_identifiability_fail_fds": sorted(set(tau_ratio_fail_fds)),
        "tau_identifiability_severe_fail_fds": sorted(set(tau_ratio_severe_fail_fds)),
        "tau_identifiability_deficits": tau_ratio_deficits,
        "external_metric_alerts": external_metric_alerts,
        "external_near_perfect_count": int(external_near_perfect_count),
        "external_num_ok_datasets": int(external_num_ok_datasets),
        "external_num_audited_datasets": int(external_num_audited_datasets),
        "external_overconservative_strong_evidence": bool(external_overconservative_strong_evidence),
        "external_pvalue_overconservative_rows": external_pvalue_overconservative_rows,
        "topology_hits_raw": topo_hits_raw,
        "topology_hits_unique_family": topo_hits,
        "topology_unique_families": topo_unique_families,
        "strict_regimes_report": {
            "path": str(strict_regimes_path) if strict_regimes_path is not None else "",
            "num_findings_total": strict_regime_findings,
        },
        "risk_penalties": risk_penalties,
        "priorities": priorities,
    }
    out = _sanitize_json(out)

    out_json = Path(args.out_json).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2, allow_nan=False), encoding="utf-8")

    out_md = Path(args.out_md).resolve()
    _write_md(out_md, out)

    print(f"Saved JSON: {out_json}")
    print(f"Saved Markdown: {out_md}")
    print(f"Score: {score_10:.2f}/10.00 | 9+ target: {'PASS' if target_9plus_pass else 'FAIL'}")


if __name__ == "__main__":
    main()

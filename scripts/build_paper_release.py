from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from markdown import markdown


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build organized paper release package with honesty checks and figures.")
    p.add_argument("--paper-md", type=str, default="paper/topological_evidence_curves.md")
    p.add_argument("--paper-pdf", type=str, default="paper/main.pdf")
    p.add_argument("--paper-generated-dir", type=str, default="paper/generated")
    p.add_argument("--paper-fig-dir", type=str, default="paper/figures")
    p.add_argument("--external-report", type=str, default="outputs/external_performance_report.json")
    p.add_argument("--retrain-robustness-json", type=str, default="outputs/external_performance_report_retrain_robustness_v2.json")
    p.add_argument("--retrain-policy-sweep-json", type=str, default="outputs/external_policy_replay_sweep_retrain_v2/summary.json")
    p.add_argument("--baseline-json", type=str, default="outputs/baseline_comparison.json")
    p.add_argument("--claim-significance-json", type=str, default="outputs/publication_full_rtx4050/claim_significance_report.json")
    p.add_argument("--policy-sharpness-json", type=str, default="outputs/publication_full_rtx4050/policy_sharpness_report.json")
    p.add_argument("--suspicious-audit-json", type=str, default="outputs/publication_full_rtx4050/phd_suspicious_values_audit.json")
    p.add_argument("--readiness-json", type=str, default="outputs/publication_full_rtx4050/stats_conference_readiness.json")
    p.add_argument("--gate-summary-json", type=str, default="outputs/publication_full_rtx4050/publication_gate_summary.json")
    p.add_argument("--topology-fig-dir", type=str, default="outputs/publication_full_rtx4050/topology_rul_figs")
    p.add_argument("--out-dir", type=str, default="outputs/publication_full_rtx4050/paper_release")
    return p.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt3(v: float) -> str:
    return f"{float(v):.3f}"


def _dataset_label_to_key(label: str) -> str:
    n = str(label).strip().lower()
    if "c-mapss" in n or "cmapss" in n:
        return "cmapss"
    if "xjtu" in n:
        return "xjtu_sy"
    if "femto" in n:
        return "femto"
    return n


def _parse_snapshot_rows(md_text: str) -> list[dict[str, str]]:
    pat = re.compile(
        r"^- (?P<label>[^:]+): RMSE=(?P<rmse>[0-9.]+), MAE=(?P<mae>[0-9.]+), "
        r"RMSE_last=(?P<rmse_last>[0-9.]+), MAE_last=(?P<mae_last>[0-9.]+), "
        r"RUL coverage=(?P<rul_cov>[0-9.]+), tau violation=(?P<tau_v>[0-9.]+)$",
        re.MULTILINE,
    )
    out: list[dict[str, str]] = []
    for m in pat.finditer(md_text):
        out.append({k: m.group(k) for k in ("label", "rmse", "mae", "rmse_last", "mae_last", "rul_cov", "tau_v")})
    return out


def _parse_readiness_scores(md_text: str) -> tuple[float | None, float | None]:
    pat = re.compile(
        r"Legacy score is ([0-9.]+)/10\.0 .* conservative score is ([0-9.]+)/10\.0",
        re.IGNORECASE,
    )
    m = pat.search(md_text)
    if not m:
        return None, None
    return float(m.group(1)), float(m.group(2))


def _honesty_check(md_text: str, external: dict[str, Any], readiness: dict[str, Any]) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    rows = _parse_snapshot_rows(md_text)
    ext_rows = {
        str(d.get("dataset", "")).lower(): d
        for d in list(external.get("datasets", []))
        if str(d.get("status", "")).lower() == "ok"
    }

    # Check row count by matched datasets.
    if len(rows) != len(ext_rows):
        findings.append(
            {
                "severity": "high",
                "type": "snapshot_row_count_mismatch",
                "message": f"paper rows={len(rows)} vs external ok datasets={len(ext_rows)}",
            }
        )

    for row in rows:
        ds_key = _dataset_label_to_key(row["label"])
        if ds_key not in ext_rows:
            findings.append(
                {
                    "severity": "high",
                    "type": "snapshot_dataset_missing",
                    "message": f"paper row dataset `{row['label']}` not found in external report",
                }
            )
            continue
        m = ext_rows[ds_key].get("metrics", {}) if isinstance(ext_rows[ds_key].get("metrics", {}), dict) else {}
        expect = {
            "rmse": _fmt3(float(m.get("rmse", np.nan))),
            "mae": _fmt3(float(m.get("mae", np.nan))),
            "rmse_last": _fmt3(float(m.get("rmse_last", np.nan))),
            "mae_last": _fmt3(float(m.get("mae_last", np.nan))),
            "rul_cov": _fmt3(float(m.get("rul_cov", np.nan))),
            "tau_v": _fmt3(float(m.get("tau_v", np.nan))),
        }
        for k in ("rmse", "mae", "rmse_last", "mae_last", "rul_cov", "tau_v"):
            if str(row[k]) != str(expect[k]):
                findings.append(
                    {
                        "severity": "high",
                        "type": "snapshot_metric_mismatch",
                        "message": (
                            f"{row['label']} {k} paper={row[k]} != external={expect[k]}"
                        ),
                    }
                )

    p_legacy, p_cons = _parse_readiness_scores(md_text)
    r_legacy = float(readiness.get("score_10", np.nan))
    r_cons = float(readiness.get("score_10_conservative", np.nan))
    if p_legacy is None or p_cons is None:
        findings.append(
            {
                "severity": "medium",
                "type": "readiness_line_missing",
                "message": "Could not parse readiness summary sentence from paper markdown.",
            }
        )
    else:
        if _fmt3(p_legacy) != _fmt3(r_legacy):
            findings.append(
                {
                    "severity": "high",
                    "type": "readiness_mismatch",
                    "message": f"legacy score paper={p_legacy} != readiness={r_legacy}",
                }
            )
        if _fmt3(p_cons) != _fmt3(r_cons):
            findings.append(
                {
                    "severity": "high",
                    "type": "readiness_mismatch",
                    "message": f"conservative score paper={p_cons} != readiness={r_cons}",
                }
            )

    return {
        "passed": len(findings) == 0,
        "num_findings": len(findings),
        "findings": findings,
        "parsed_snapshot_rows": rows,
        "external_ok_datasets": sorted(ext_rows.keys()),
        "readiness_scores": {"paper_legacy": p_legacy, "paper_conservative": p_cons, "artifact_legacy": r_legacy, "artifact_conservative": r_cons},
    }


def _plot_external_metrics(external: dict[str, Any], out_png: Path) -> None:
    rows = [d for d in list(external.get("datasets", [])) if str(d.get("status", "")).lower() == "ok"]
    labels = [str(d.get("dataset", "unknown")) for d in rows]
    rmse = [float(d.get("metrics", {}).get("rmse", np.nan)) for d in rows]
    rmse_last = [float(d.get("metrics", {}).get("rmse_last", np.nan)) for d in rows]
    x = np.arange(len(labels))
    w = 0.38

    plt.figure(figsize=(8.5, 4.8))
    plt.bar(x - w / 2, rmse, width=w, label="RMSE")
    plt.bar(x + w / 2, rmse_last, width=w, label="RMSE_last")
    plt.xticks(x, labels)
    plt.ylabel("Error")
    plt.title("External Dataset Error Metrics (Canonical)")
    plt.legend()
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()


def _plot_readiness_gates(readiness: dict[str, Any], out_png: Path) -> None:
    gates = list(readiness.get("gates", []))
    names = [str(g.get("name", "")) for g in gates]
    scores = [float(g.get("score", 0.0)) for g in gates]
    weights = [float(g.get("weight", 1.0)) for g in gates]
    frac = [s / w if w > 0 else 0.0 for s, w in zip(scores, weights)]

    y = np.arange(len(names))
    plt.figure(figsize=(10, 5.5))
    plt.barh(y, frac)
    plt.yticks(y, names)
    plt.xlim(0.0, 1.05)
    plt.xlabel("Gate Score Fraction (score/weight)")
    plt.title("Readiness Gate Completion")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _copy_with_sidecars(src: Path, dst_dir: Path, dst_stem: str | None = None) -> list[str]:
    copied: list[str] = []
    if not src.exists():
        return copied
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_name = src.name if not dst_stem else f"{dst_stem}{src.suffix}"
    dst = dst_dir / dst_name
    shutil.copy2(src, dst)
    copied.append(dst.name)
    if src.suffix.lower() == ".json":
        md = src.with_suffix(".md")
        if md.exists():
            md_name = md.name if not dst_stem else f"{dst_stem}{md.suffix}"
            shutil.copy2(md, dst_dir / md_name)
            copied.append(md_name)
    return copied


def _build_html_from_md(md_text: str, out_html: Path, title: str) -> None:
    html_body = markdown(md_text, extensions=["fenced_code", "tables", "toc"])
    html = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{title}</title>"
        "<style>body{font-family:Segoe UI,Arial,sans-serif;max-width:980px;margin:2rem auto;padding:0 1rem;line-height:1.5}"
        "code{background:#f4f4f4;padding:0.15rem 0.3rem;border-radius:4px}"
        "pre code{display:block;padding:0.8rem;overflow:auto} img{max-width:100%;height:auto}"
        "table{border-collapse:collapse} th,td{border:1px solid #ddd;padding:0.35rem 0.55rem}</style>"
        "</head><body>"
        f"{html_body}"
        "</body></html>"
    )
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding="utf-8")


def main() -> None:
    args = parse_args()
    paper_path = Path(args.paper_md).resolve()
    paper_pdf_path = Path(args.paper_pdf).resolve()
    paper_generated_dir = Path(args.paper_generated_dir).resolve()
    paper_fig_dir = Path(args.paper_fig_dir).resolve()
    ext_path = Path(args.external_report).resolve()
    retrain_path = Path(args.retrain_robustness_json).resolve()
    retrain_sweep_path = Path(args.retrain_policy_sweep_json).resolve()
    baseline_path = Path(args.baseline_json).resolve()
    claim_sig_path = Path(args.claim_significance_json).resolve()
    policy_sharpness_path = Path(args.policy_sharpness_json).resolve()
    suspicious_audit_path = Path(args.suspicious_audit_json).resolve()
    readiness_path = Path(args.readiness_json).resolve()
    gate_path = Path(args.gate_summary_json).resolve()
    topo_fig_dir = Path(args.topology_fig_dir).resolve()
    out_dir = Path(args.out_dir).resolve()

    paper_text = paper_path.read_text(encoding="utf-8")
    external = _load_json(ext_path)
    readiness = _load_json(readiness_path)
    gate = _load_json(gate_path) if gate_path.exists() else {}
    suspicious = _load_json(suspicious_audit_path) if suspicious_audit_path.exists() else {}

    # Release layout.
    figures_dir = out_dir / "figures"
    reports_dir = out_dir / "reports"
    artifacts_dir = out_dir / "artifacts"
    paper_dir = out_dir / "paper"
    for d in (figures_dir, reports_dir, artifacts_dir, paper_dir):
        d.mkdir(parents=True, exist_ok=True)
    for stale_name in ("summary.json", "summary.md"):
        stale_path = artifacts_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()

    # Honesty check.
    honesty = _honesty_check(paper_text, external, readiness)
    (reports_dir / "paper_honesty_report.json").write_text(json.dumps(honesty, indent=2, allow_nan=False), encoding="utf-8")
    lines = ["# Paper Honesty Report", "", f"- Passed: {'YES' if honesty['passed'] else 'NO'}", f"- Findings: {honesty['num_findings']}", ""]
    for f in honesty["findings"]:
        lines.append(f"- [{f['severity']}] {f['type']}: {f['message']}")
    (reports_dir / "paper_honesty_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Generate figures.
    _plot_external_metrics(external, figures_dir / "external_metrics_overview.png")
    _plot_readiness_gates(readiness, figures_dir / "readiness_gates.png")
    for name in ("gamma_vs_pred_mae.png", "surface_h1_vs_rul_coverage.png", "topology_vs_rul_bins.png"):
        _copy_if_exists(topo_fig_dir / name, figures_dir / name)
    for name in ("policy_replay_frontier.png", "policy_sharpness_frontier.png"):
        _copy_if_exists(paper_fig_dir / name, figures_dir / name)

    # Copy key artifacts.
    copied_artifacts: list[str] = []
    for src in (
        ext_path,
        retrain_path,
        baseline_path,
        claim_sig_path,
        policy_sharpness_path,
        suspicious_audit_path,
        Path("outputs/external_dataset_summary.json").resolve(),
        readiness_path,
        Path("outputs/artifact_consistency_report.json").resolve(),
        Path("outputs/publication_full_rtx4050/deep_check_results_strict_main.json").resolve(),
        Path("outputs/publication_full_rtx4050/deep_check_results_external_real_eval_v8.json").resolve(),
        Path("outputs/publication_full_rtx4050/deep_check_regimes_stricter_strict_main.json").resolve(),
        Path("outputs/publication_full_rtx4050/deep_check_regimes_external_v8.json").resolve(),
        Path("outputs/publication_full_rtx4050/publication_gate_summary.json").resolve(),
        paper_generated_dir / "provenance.json",
        paper_path,
    ):
        copied_artifacts.extend(_copy_with_sidecars(src, artifacts_dir))
    if retrain_sweep_path.exists():
        copied_artifacts.extend(
            _copy_with_sidecars(
                retrain_sweep_path,
                artifacts_dir,
                dst_stem=f"{retrain_sweep_path.parent.name}_summary",
            )
        )

    if paper_pdf_path.exists():
        _copy_if_exists(paper_pdf_path, paper_dir / paper_pdf_path.name)

    # Build compiled paper markdown with auto annex.
    ext_rows = [d for d in list(external.get("datasets", [])) if str(d.get("status", "")).lower() == "ok"]
    annex = []
    annex.append("")
    annex.append("## 10. Auto-Compiled Figures (Canonical Build)")
    annex.append("")
    annex.append("This section is auto-generated from canonical artifacts and is intended for release packaging.")
    annex.append("")
    annex.append("### External Metrics Table")
    annex.append("")
    annex.append("| Dataset | RMSE | MAE | RMSE_last | MAE_last | RUL_cov | Tau_v |")
    annex.append("|---|---:|---:|---:|---:|---:|---:|")
    for d in ext_rows:
        m = d.get("metrics", {}) if isinstance(d.get("metrics", {}), dict) else {}
        annex.append(
            f"| {d.get('dataset')} | {_fmt3(m.get('rmse', np.nan))} | {_fmt3(m.get('mae', np.nan))} | "
            f"{_fmt3(m.get('rmse_last', np.nan))} | {_fmt3(m.get('mae_last', np.nan))} | "
            f"{_fmt3(m.get('rul_cov', np.nan))} | {_fmt3(m.get('tau_v', np.nan))} |"
        )
    annex.append("")
    annex.append("### Figures")
    annex.append("")
    annex.append("![External Metrics Overview](../figures/external_metrics_overview.png)")
    annex.append("")
    annex.append("![Readiness Gate Completion](../figures/readiness_gates.png)")
    annex.append("")
    if (figures_dir / "topology_vs_rul_bins.png").exists():
        annex.append("![Topology vs RUL Bins](../figures/topology_vs_rul_bins.png)")
        annex.append("")
    if (figures_dir / "surface_h1_vs_rul_coverage.png").exists():
        annex.append("![Surface H1 vs RUL Coverage](../figures/surface_h1_vs_rul_coverage.png)")
        annex.append("")
    if (figures_dir / "gamma_vs_pred_mae.png").exists():
        annex.append("![Gamma vs Prediction MAE](../figures/gamma_vs_pred_mae.png)")
    annex.append("")
    if retrain_path.exists():
        retrain = _load_json(retrain_path)
        retrain_rows = [d for d in list(retrain.get("datasets", [])) if str(d.get("status", "")).lower() == "ok"]
        if retrain_rows:
            annex.append("### True Retrain Robustness")
            annex.append("")
            annex.append("| Dataset | RMSE | RUL_cov | Tau_v | Runs |")
            annex.append("|---|---:|---:|---:|---:|")
            for d in retrain_rows:
                m = d.get("metrics", {}) if isinstance(d.get("metrics", {}), dict) else {}
                annex.append(
                    f"| {d.get('dataset')} | {_fmt3(m.get('rmse', np.nan))} | {_fmt3(m.get('rul_cov', np.nan))} | "
                    f"{_fmt3(m.get('tau_v', np.nan))} | {int(d.get('num_runs', 0))} |"
                )
            annex.append("")
    if retrain_sweep_path.exists():
        retrain_sweep = _load_json(retrain_sweep_path)
        best = retrain_sweep.get("best_policy", {}) if isinstance(retrain_sweep.get("best_policy", {}), dict) else {}
        annex.append("### Retrain Replay Sweep")
        annex.append("")
        annex.append(
            f"- selection_mode={retrain_sweep.get('selection_mode', 'n/a')}, "
            f"valid_points={int(retrain_sweep.get('num_valid_points', 0))}"
        )
        annex.append(
            f"- best_policy: alpha={_fmt3(best.get('alpha', np.nan))}, lambda={_fmt3(best.get('lambda_bet', np.nan))}, "
            f"margin={_fmt3(best.get('pvalue_safety_margin', np.nan))}, "
            f"width_mean={_fmt3(best.get('width_mean', np.nan))}, cov_min={_fmt3(best.get('cov_min', np.nan))}, "
            f"tau_max={_fmt3(best.get('tau_max', np.nan))}"
        )
        annex.append("")
    if suspicious:
        summ = suspicious.get("summary", {}) if isinstance(suspicious.get("summary", {}), dict) else {}
        annex.append("### Suspicious-Values Audit")
        annex.append("")
        annex.append(
            f"- findings={int(summ.get('num_findings', 0))}, high={int(summ.get('num_high', 0))}, "
            f"medium={int(summ.get('num_medium', 0))}"
        )
        for item in list(suspicious.get("findings", []))[:10]:
            annex.append(f"- [{item.get('severity', 'n/a')}] {item.get('type', 'unknown')}: {item.get('message', '')}")
        annex.append("")

    compiled_md = paper_text + "\n" + "\n".join(annex).strip() + "\n"
    md_out = paper_dir / "topological_evidence_curves.compiled.md"
    md_out.write_text(compiled_md, encoding="utf-8")
    _build_html_from_md(compiled_md, paper_dir / "topological_evidence_curves.compiled.html", "Topological Evidence Curves")

    manifest = {
        "paper_source": str(paper_path),
        "compiled_outputs": {
            "markdown": str(md_out),
            "html": str(paper_dir / "topological_evidence_curves.compiled.html"),
            "latex_pdf": str(paper_dir / paper_pdf_path.name) if paper_pdf_path.exists() else "",
        },
        "honesty_passed": bool(honesty["passed"]),
        "gate_overall_pass": bool(gate.get("overall_pass", False)),
        "suspicious_values_findings": {
            "num_findings": int(suspicious.get("summary", {}).get("num_findings", 0)) if suspicious else 0,
            "num_high": int(suspicious.get("summary", {}).get("num_high", 0)) if suspicious else 0,
            "num_medium": int(suspicious.get("summary", {}).get("num_medium", 0)) if suspicious else 0,
        },
        "figures": sorted([p.name for p in figures_dir.glob("*.png")]),
        "artifacts": sorted(set(copied_artifacts)),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")

    print(f"Saved release package under: {out_dir}")
    print(f"Honesty passed: {honesty['passed']}")
    print(f"Gate overall pass: {manifest['gate_overall_pass']}")


if __name__ == "__main__":
    main()

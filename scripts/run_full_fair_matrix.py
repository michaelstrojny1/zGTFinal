from __future__ import annotations

import argparse
import itertools
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STRICT_POLICY: dict[int, dict[str, float | int]] = {
    1: {"pvalue_safety_margin": 0.08, "calibration_bins": 8, "calibration_min_bin_size": 128},
    2: {"pvalue_safety_margin": 0.11, "calibration_bins": 16, "calibration_min_bin_size": 128},
    3: {"pvalue_safety_margin": 0.18, "calibration_bins": 12, "calibration_min_bin_size": 128},
    4: {"pvalue_safety_margin": 0.25, "calibration_bins": 12, "calibration_min_bin_size": 128},
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a full and fair publication matrix.")
    p.add_argument("--fds", nargs="+", type=int, default=[1, 2, 3, 4])
    p.add_argument("--out-root", type=str, default="outputs/full_fair_matrix")
    p.add_argument("--data-root", type=str, default="data/rul_datasets")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--seed-list", type=str, default="41,42,43")
    p.add_argument("--split-calibration-fraction", type=float, default=0.2)
    p.add_argument("--alphas", type=str, default="0.01,0.05,0.1")
    p.add_argument("--predict-batch-size", type=int, default=8192)
    p.add_argument("--tau-max", type=int, default=1000)
    p.add_argument("--lambda-bet", type=float, default=0.06)
    p.add_argument("--min-persistence", type=float, default=0.5)
    p.add_argument("--alert-patience", type=int, default=3)
    p.add_argument("--synthetic-rhos", type=str, default="0.5,1.0,1.5")
    p.add_argument("--synthetic-sigmas", type=str, default="3.0,5.0,8.0")
    p.add_argument("--synthetic-n-engines", type=int, default=1500)
    p.add_argument("--reuse", action="store_true")
    p.add_argument("--refresh-evidence-mode", action="store_true")
    p.add_argument("--baseline-calibration-source", type=str, choices=["dev_holdout", "val"], default="dev_holdout")
    p.add_argument(
        "--external-baselines-json",
        type=str,
        default="",
        help="Optional external baseline package JSON passed through to build_baseline_comparison.py.",
    )
    p.add_argument(
        "--external-performance-report",
        type=str,
        default="",
        help="Optional real external model-performance JSON passed to build_external_dataset_summary.py.",
    )
    p.add_argument("--rul-dataset-summary", type=str, default="outputs/rul_dataset_summary.json")
    p.add_argument("--paper-md", type=str, default="paper/topological_evidence_curves.md")
    p.add_argument("--skip-derived-artifacts", action="store_true")
    p.add_argument("--out-json", type=str, default="outputs/full_fair_matrix_report.json")
    p.add_argument("--out-md", type=str, default="outputs/full_fair_matrix_report.md")
    return p.parse_args()


def _ilist(raw: str) -> list[int]:
    return [int(x.strip()) for x in str(raw).split(",") if x.strip()]


def _flist(raw: str) -> list[float]:
    return [float(x.strip()) for x in str(raw).split(",") if x.strip()]


def _run(cmd: list[str]) -> float:
    print(" ".join(cmd), flush=True)
    t0 = time.perf_counter()
    subprocess.run(cmd, cwd=str(ROOT), check=True)
    return float(time.perf_counter() - t0)


def _one(path: Path, pattern: str) -> Path:
    hits = sorted(path.glob(pattern))
    if not hits:
        raise FileNotFoundError(f"Missing {pattern} under {path}")
    return hits[0]


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _metrics(fd: int, run_dir: Path) -> dict[str, Any]:
    train = _load(_one(run_dir, f"train_metrics_fd{fd:03d}.json"))
    tem = _load(_one(run_dir, f"tem_metrics_fd{fd:03d}.json"))
    audit = _load(_one(run_dir, f"audit_fd{fd:03d}.json"))
    fleet = tem["fleet_summary"]
    return {
        "fd": fd,
        "run_dir": str(run_dir),
        "rmse": float(train["test_last_rmse"]),
        "mae": float(train["test_last_mae"]),
        "rul_cov": float(fleet["mean_temporal_rul_coverage"]),
        "tau_v": float(fleet["tau_anytime_violation_rate"]),
        "alert_rate": float(fleet["alert_rate"]),
        "p_all_frac_02": float(audit["pvalue_all"]["frac_le_0.2"]),
        "p_healthy_frac_02": float(audit["pvalue_healthy_prefix"]["frac_le_0.2"]),
    }


def _model_paths(fd: int, run_dir: Path) -> tuple[Path, Path]:
    ckpt = run_dir / f"model_fd{fd:03d}.pt"
    bundle = run_dir / f"calibration_bundle_fd{fd:03d}.npz"
    cal = bundle if bundle.exists() else run_dir / f"calibration_residuals_fd{fd:03d}.npy"
    if not ckpt.exists() or not cal.exists():
        raise FileNotFoundError(f"Missing model/calibration in {run_dir}")
    return ckpt, cal


def _copy_train(fd: int, src: Path, dst: Path) -> None:
    src_p = src / f"train_metrics_fd{fd:03d}.json"
    dst_p = dst / f"train_metrics_fd{fd:03d}.json"
    if src_p.exists():
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_p, dst_p)


def _run_full_strict(args: argparse.Namespace, fd: int, out_dir: Path, seed: int, source: str) -> float:
    out_dir.mkdir(parents=True, exist_ok=True)
    pol = STRICT_POLICY[int(fd)]
    needed = [
        out_dir / f"train_metrics_fd{fd:03d}.json",
        out_dir / f"tem_metrics_fd{fd:03d}.json",
        out_dir / f"audit_fd{fd:03d}.json",
    ]
    request_path = out_dir / f"full_strict_request_fd{fd:03d}.json"
    expected_request = {
        "fd": int(fd),
        "data_root": str(args.data_root),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "seed": int(seed),
        "calibration_source": str(source),
        "calibration_fraction": float(args.split_calibration_fraction),
        "lambda_bet": float(args.lambda_bet),
        "min_persistence": float(args.min_persistence),
        "alert_patience": int(args.alert_patience),
        "tau_max": int(args.tau_max),
        "predict_batch_size": int(args.predict_batch_size),
        "policy_pvalue_safety_margin": float(pol["pvalue_safety_margin"]),
        "policy_calibration_bins": int(pol["calibration_bins"]),
        "policy_calibration_min_bin_size": int(pol["calibration_min_bin_size"]),
    }
    if args.reuse and all(p.exists() for p in needed) and request_path.exists():
        try:
            prior_request = _load(request_path)
            train = _load(needed[0])
            tem = _load(needed[1])
        except Exception:
            prior_request = None
            train = None
            tem = None
        request_match = prior_request == expected_request
        # Backward-compatible reuse: old manifests may not include explicit policy fields.
        if (
            not request_match
            and isinstance(prior_request, dict)
            and isinstance(tem, dict)
            and all(prior_request.get(k) == expected_request.get(k) for k in prior_request.keys())
        ):
            cfg = tem.get("config", {})
            request_match = (
                abs(float(cfg.get("pvalue_safety_margin", -1.0)) - float(expected_request["policy_pvalue_safety_margin"])) <= 1e-12
                and int(cfg.get("calibration_bins", -1)) == int(expected_request["policy_calibration_bins"])
                and int(cfg.get("calibration_min_bin_size", -1)) == int(expected_request["policy_calibration_min_bin_size"])
            )
        if request_match and isinstance(train, dict):
            train_source = str(train.get("calibration_source", ""))
            train_frac = float(train.get("calibration_fraction", -1.0))
            if train_source == str(source) and abs(train_frac - float(args.split_calibration_fraction)) <= 1e-12:
                return 0.0
        print(
            f"[reuse-mismatch] rerun fd={fd} out={out_dir} "
            f"(seed/source/fraction or request manifest mismatch)",
            flush=True,
        )
    dur = _run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_full_pipeline_strict.py"),
            "--fd",
            str(fd),
            "--data-root",
            str(args.data_root),
            "--out-dir",
            str(out_dir),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--seed",
            str(seed),
            "--calibration-source",
            str(source),
            "--calibration-fraction",
            str(args.split_calibration_fraction),
            "--lambda-bet",
            str(args.lambda_bet),
            "--min-persistence",
            str(args.min_persistence),
            "--alert-patience",
            str(args.alert_patience),
            "--tau-max",
            str(args.tau_max),
            "--predict-batch-size",
            str(args.predict_batch_size),
            "--surface-topology-scope",
            "all",
        ]
    )
    request_path.write_text(json.dumps(expected_request, indent=2), encoding="utf-8")
    return float(dur)


def _run_tem_audit(
    args: argparse.Namespace,
    fd: int,
    out_dir: Path,
    ckpt: Path,
    cal: Path,
    alpha: float,
    evidence_mode: str,
    margin: float,
    bins: int,
    min_bin: int,
    topology_level: str = "lite",
    surface_topology_scope: str = "none",
    reuse: bool | None = None,
) -> float:
    out_dir.mkdir(parents=True, exist_ok=True)
    tem_path = out_dir / f"tem_metrics_fd{fd:03d}.json"
    audit_path = out_dir / f"audit_fd{fd:03d}.json"
    cache_path = out_dir / f"audit_cache_fd{fd:03d}.npz"
    request_path = out_dir / f"tem_audit_request_fd{fd:03d}.json"
    expected_request = {
        "fd": int(fd),
        "data_root": str(args.data_root),
        "checkpoint": str(ckpt.resolve()),
        "checkpoint_size": int(ckpt.stat().st_size),
        "checkpoint_mtime_ns": int(ckpt.stat().st_mtime_ns),
        "calibration": str(cal.resolve()),
        "calibration_size": int(cal.stat().st_size),
        "calibration_mtime_ns": int(cal.stat().st_mtime_ns),
        "alpha": float(alpha),
        "evidence_mode": str(evidence_mode),
        "lambda_bet": float(args.lambda_bet),
        "tau_max": int(args.tau_max),
        "predict_batch_size": int(args.predict_batch_size),
        "min_persistence": float(args.min_persistence),
        "alert_patience": int(args.alert_patience),
        "pvalue_safety_margin": float(margin),
        "calibration_bins": int(bins),
        "calibration_min_bin_size": int(min_bin),
        "topology_level": str(topology_level),
        "surface_topology_scope": str(surface_topology_scope),
    }
    elapsed = 0.0
    reuse_flag = bool(args.reuse) if reuse is None else bool(reuse)
    reuse_request_ok = False
    if reuse_flag and request_path.exists():
        try:
            prior_request = _load(request_path)
            reuse_request_ok = prior_request == expected_request
        except Exception:
            reuse_request_ok = False
    if reuse_flag and not reuse_request_ok:
        print(
            f"[reuse-mismatch] rerun tem/audit fd={fd} out={out_dir} (checkpoint/config mismatch)",
            flush=True,
        )
        reuse_flag = False
    if not (reuse_flag and tem_path.exists() and cache_path.exists()):
        elapsed += _run(
            [
                sys.executable,
                str(ROOT / "scripts" / "run_tem_cmapss.py"),
                "--fd",
                str(fd),
                "--data-root",
                str(args.data_root),
                "--checkpoint",
                str(ckpt),
                "--calibration",
                str(cal),
                "--alpha",
                str(alpha),
                "--lambda-bet",
                str(args.lambda_bet),
                "--evidence-mode",
                str(evidence_mode),
                "--tau-max",
                str(args.tau_max),
                "--min-persistence",
                str(args.min_persistence),
                "--alert-patience",
                str(args.alert_patience),
                "--predict-batch-size",
                str(args.predict_batch_size),
                "--pvalue-safety-margin",
                str(margin),
                "--calibration-bins",
                str(bins),
                "--calibration-min-bin-size",
                str(min_bin),
                "--topology-level",
                str(topology_level),
                "--surface-topology-scope",
                str(surface_topology_scope),
                "--audit-cache",
                str(cache_path),
                "--out-dir",
                str(out_dir),
            ]
        )
    if not (reuse_flag and audit_path.exists()):
        elapsed += _run(
            [
                sys.executable,
                str(ROOT / "scripts" / "audit_tem.py"),
                "--fd",
                str(fd),
                "--data-root",
                str(args.data_root),
                "--checkpoint",
                str(ckpt),
                "--calibration",
                str(cal),
                "--alpha",
                str(alpha),
                "--lambda-bet",
                str(args.lambda_bet),
                "--evidence-mode",
                str(evidence_mode),
                "--tau-max",
                str(args.tau_max),
                "--predict-batch-size",
                str(args.predict_batch_size),
                "--pvalue-safety-margin",
                str(margin),
                "--calibration-bins",
                str(bins),
                "--calibration-min-bin-size",
                str(min_bin),
                "--cache",
                str(cache_path),
                "--tem-metrics",
                str(tem_path),
                "--out-dir",
                str(out_dir),
            ]
        )
    request_path.write_text(json.dumps(expected_request, indent=2), encoding="utf-8")
    return elapsed


def _write_md(path: Path, report: dict[str, Any]) -> None:
    lines = ["# Full Fair Matrix Report", ""]
    lines.append(f"- Timestamp: {report['timestamp_local']}")
    lines.append(f"- Output root: `{report['out_root']}`")
    lines.append(f"- Wall seconds: {report['total_wall_seconds']:.1f}")
    lines.append(f"- Baseline calibration source: `{report['settings'].get('baseline_calibration_source', 'unknown')}`")
    lines.append(f"- Suspicious findings: {report['suspicious']['count']}")
    lines.append("")
    lines.append("## Baseline")
    for r in report["baseline"]:
        lines.append(
            f"- FD{r['fd']:03d}: rmse={r['rmse']:.3f}, mae={r['mae']:.3f}, "
            f"rul_cov={r['rul_cov']:.3f}, tau_v={r['tau_v']:.3f}"
        )
    lines.append("")
    lines.append("## Evidence Deltas (marginal-fixed)")
    for d in report.get("evidence_mode_deltas", []):
        lines.append(
            f"- FD{int(d['fd']):03d}: delta_rul_cov={float(d['delta_rul_cov_marginal_minus_fixed']):+.3f}, "
            f"delta_tau_v={float(d['delta_tau_v_marginal_minus_fixed']):+.3f}"
        )
    lines.append("")
    lines.append("## Deep Checks")
    lines.append(f"- deep_check_results findings: {report['deep_checks']['deep_check_results_findings']}")
    lines.append(f"- deep_check_results (all artifacts) findings: {report['deep_checks']['deep_check_results_all_findings']}")
    lines.append(f"- deep_check_regimes findings: {report['deep_checks']['deep_check_regimes_findings']}")
    lines.append(
        f"- deep_check_results (all) expected stress findings: "
        f"{report['deep_checks'].get('deep_check_results_all_expected_stress_findings', 0)}"
    )
    lines.append(
        f"- deep_check_results (all) unexpected findings: "
        f"{report['deep_checks'].get('deep_check_results_all_unexpected_findings', 0)}"
    )
    lines.append("")
    if report.get("notes"):
        lines.append("## Notes")
        for n in report["notes"]:
            lines.append(f"- {n}")
        lines.append("")
    lines.append("## Suspicious")
    if report["suspicious"]["findings"]:
        for f in report["suspicious"]["findings"]:
            lines.append(f"- [{f['severity'].upper()}] {f['message']}")
    else:
        lines.append("- None")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _merge_deep_check_reports(scan_roots: list[Path], out_report: Path) -> dict[str, Any]:
    out_report.parent.mkdir(parents=True, exist_ok=True)
    roots = [Path(r).resolve() for r in scan_roots if Path(r).resolve().exists()]
    if not roots:
        merged = {"scanned": {"tem_metrics": 0, "audit": 0, "synthetic": 0}, "num_findings": 0, "findings": [], "scan_roots": []}
        out_report.write_text(json.dumps(merged, indent=2), encoding="utf-8")
        return merged

    used_roots = [str(r) for r in roots]
    common_root = Path(os.path.commonpath([str(r) for r in roots])).resolve()
    temp_report = out_report.parent / ".deep_check_all_tmp.json"
    _run(
        [
            sys.executable,
            str(ROOT / "scripts" / "deep_check_results.py"),
            "--outputs-root",
            str(common_root),
            "--report-path",
            str(temp_report),
        ]
    )
    part = _load(temp_report)
    temp_report.unlink(missing_ok=True)

    def _is_under_any_root(path: Path) -> bool:
        p = path.resolve()
        for r in roots:
            try:
                p.relative_to(r)
                return True
            except ValueError:
                continue
        return False

    findings_by_key: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for f in part.get("findings", []):
        artifact = Path(str(f.get("artifact", "")))
        if not _is_under_any_root(artifact):
            continue
        key = (
            str(f.get("artifact", "")),
            str(f.get("type", "")),
            str(f.get("message", "")),
            str(f.get("severity", "")),
        )
        findings_by_key[key] = f

    tem_files: set[Path] = set()
    audit_files: set[Path] = set()
    synth_files: set[Path] = set()
    for r in roots:
        for p in r.rglob("tem_metrics_*.json"):
            tem_files.add(p.resolve())
        for p in r.rglob("audit_*.json"):
            audit_files.add(p.resolve())
        for p in r.rglob("synthetic_summary.json"):
            synth_files.add(p.resolve())
    scanned = {
        "tem_metrics": int(len(tem_files)),
        "audit": int(len(audit_files)),
        "synthetic": int(len(synth_files)),
    }

    merged = {
        "scanned": scanned,
        "num_findings": int(len(findings_by_key)),
        "findings": list(findings_by_key.values()),
        "scan_roots": used_roots,
    }
    out_report.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    return merged


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    seeds = _ilist(args.seed_list)
    alphas = _flist(args.alphas)
    synth_rhos = _flist(args.synthetic_rhos)
    synth_sigmas = _flist(args.synthetic_sigmas)
    t0 = time.perf_counter()
    timings: dict[str, float] = {}

    baseline_root = out_root / "strict_main"
    baseline_rows: list[dict[str, Any]] = []
    for fd in args.fds:
        run_dir = baseline_root / f"fd{fd:03d}"
        dur = _run_full_strict(args, fd, run_dir, args.split_seed, str(args.baseline_calibration_source))
        timings[f"baseline_fd{fd:03d}_sec"] = float(dur)
        row = _metrics(fd, run_dir)
        row["calibration_source"] = str(args.baseline_calibration_source)
        baseline_rows.append(row)

    run_dirs = [str((baseline_root / f"fd{fd:03d}").resolve()) for fd in args.fds]
    deep_check_json = out_root / "deep_check_report_strict_main.json"
    deep_reg_json = out_root / "deep_check_regimes.json"
    topology_json = out_root / "topology_rul_landscape.json"
    topology_md = out_root / "topology_rul_landscape.md"
    timings["deep_check_results_sec"] = _run(
        [
            sys.executable,
            str(ROOT / "scripts" / "deep_check_results.py"),
            "--outputs-root",
            str(baseline_root),
            "--report-path",
            str(deep_check_json),
        ]
    )
    timings["deep_check_regimes_sec"] = _run(
        [
            sys.executable,
            str(ROOT / "scripts" / "deep_check_regimes.py"),
            "--run-dirs",
            *run_dirs,
            "--healthy-rul-floor",
            "100",
            "--require-surface-topology",
            "--out-json",
            str(deep_reg_json),
            "--out-md",
            str(out_root / "deep_check_regimes.md"),
        ]
    )
    timings["topology_landscape_sec"] = _run(
        [
            sys.executable,
            str(ROOT / "scripts" / "analyze_topology_rul_landscape.py"),
            "--run-dirs",
            *run_dirs,
            "--out-json",
            str(topology_json),
            "--out-md",
            str(topology_md),
            "--fig-dir",
            str(out_root / "topology_rul_figs"),
            "--bootstrap",
            "1500",
        ]
    )

    seed_rows: list[dict[str, Any]] = []
    for seed, fd in itertools.product(seeds, args.fds):
        run_dir = out_root / "seed_repro" / f"seed_{seed:04d}" / f"fd{fd:03d}"
        timings[f"seed_fd{fd:03d}_{seed:04d}_sec"] = float(
            _run_full_strict(args, fd, run_dir, seed, str(args.baseline_calibration_source))
        )
        row = _metrics(fd, run_dir)
        row["calibration_source"] = str(args.baseline_calibration_source)
        seed_rows.append(row)

    split_rows: list[dict[str, Any]] = []
    for source, fd in itertools.product(["val", "dev_holdout"], args.fds):
        run_dir = out_root / "split_robustness" / source / f"fd{fd:03d}"
        timings[f"split_{source}_fd{fd:03d}_sec"] = float(_run_full_strict(args, fd, run_dir, args.split_seed, source))
        row = _metrics(fd, run_dir)
        row["calibration_source"] = str(source)
        split_rows.append(row)

    evidence_rows: list[dict[str, Any]] = []
    alpha_rows: list[dict[str, Any]] = []
    policy_rows: list[dict[str, Any]] = []
    for fd in args.fds:
        base_dir = baseline_root / f"fd{fd:03d}"
        ckpt, cal = _model_paths(fd, base_dir)
        pol = STRICT_POLICY[int(fd)]
        for mode in ["fixed_tau", "marginal_rul"]:
            run_dir = out_root / "evidence_mode" / mode / f"fd{fd:03d}"
            timings[f"evidence_{mode}_fd{fd:03d}_sec"] = float(
                _run_tem_audit(
                    args,
                    fd,
                    run_dir,
                    ckpt,
                    cal,
                    0.05,
                    mode,
                    float(pol["pvalue_safety_margin"]),
                    int(pol["calibration_bins"]),
                    int(pol["calibration_min_bin_size"]),
                    topology_level="lite",
                    surface_topology_scope="none",
                    reuse=(bool(args.reuse) and not bool(args.refresh_evidence_mode)),
                )
            )
            _copy_train(fd, base_dir, run_dir)
            evidence_rows.append(_metrics(fd, run_dir))
        for alpha in alphas:
            tag = f"a{str(alpha).replace('.', 'p')}"
            run_dir = out_root / "alpha_sweep" / tag / f"fd{fd:03d}"
            timings[f"alpha_{tag}_fd{fd:03d}_sec"] = float(
                _run_tem_audit(
                    args,
                    fd,
                    run_dir,
                    ckpt,
                    cal,
                    float(alpha),
                    "fixed_tau",
                    float(pol["pvalue_safety_margin"]),
                    int(pol["calibration_bins"]),
                    int(pol["calibration_min_bin_size"]),
                    topology_level="lite",
                    surface_topology_scope="none",
                )
            )
            _copy_train(fd, base_dir, run_dir)
            alpha_rows.append(_metrics(fd, run_dir))
        margins = sorted({0.0, float(pol["pvalue_safety_margin"])})
        bins = sorted({max(4, int(pol["calibration_bins"]) // 2), int(pol["calibration_bins"])})
        mins = sorted({64, int(pol["calibration_min_bin_size"])})
        for m, b, mb in itertools.product(margins, bins, mins):
            tag = f"m{str(m).replace('.', 'p')}_b{b}_mb{mb}"
            run_dir = out_root / "policy_sweep" / tag / f"fd{fd:03d}"
            timings[f"policy_{tag}_fd{fd:03d}_sec"] = float(
                _run_tem_audit(
                    args,
                    fd,
                    run_dir,
                    ckpt,
                    cal,
                    0.05,
                    "fixed_tau",
                    float(m),
                    int(b),
                    int(mb),
                    topology_level="lite",
                    surface_topology_scope="none",
                )
            )
            _copy_train(fd, base_dir, run_dir)
            policy_rows.append(_metrics(fd, run_dir))

    synth_rows: list[dict[str, Any]] = []
    for sigma, rho in itertools.product(synth_sigmas, synth_rhos):
        tag = f"s{str(sigma).replace('.', 'p')}_r{str(rho).replace('.', 'p')}"
        run_dir = out_root / "synthetic_grid" / tag
        summary_path = run_dir / "synthetic_summary.json"
        synth_request_path = run_dir / "synthetic_request.json"
        synth_request = {
            "n_engines": int(args.synthetic_n_engines),
            "sigma": float(sigma),
            "rho": float(rho),
            "lambda_bet": float(args.lambda_bet),
            "seed": int(args.split_seed),
        }
        synth_reuse_ok = False
        if args.reuse and summary_path.exists() and synth_request_path.exists():
            try:
                synth_reuse_ok = _load(synth_request_path) == synth_request
            except Exception:
                synth_reuse_ok = False
        if not synth_reuse_ok:
            run_dir.mkdir(parents=True, exist_ok=True)
            if args.reuse and summary_path.exists():
                print(f"[reuse-mismatch] rerun synthetic {tag} (config mismatch)", flush=True)
            timings[f"synthetic_{tag}_sec"] = float(
                _run(
                    [
                        sys.executable,
                        str(ROOT / "scripts" / "run_synthetic_validation.py"),
                        "--n-engines",
                        str(args.synthetic_n_engines),
                        "--sigma",
                        str(sigma),
                        "--rho",
                        str(rho),
                        "--lambda-bet",
                        str(args.lambda_bet),
                        "--seed",
                        str(args.split_seed),
                        "--out-dir",
                        str(run_dir),
                    ]
                )
            )
            synth_request_path.write_text(json.dumps(synth_request, indent=2), encoding="utf-8")
        s = _load(summary_path)
        synth_rows.append(
            {
                "sigma": float(sigma),
                "rho": float(rho),
                "null_alert_rate": float(s["null_cohort"]["alert_rate"]),
                "degraded_alert_rate": float(s["degraded_cohort"]["alert_rate"]),
                "oracle_superuniform_violation_prob": float(s["oracle_superuniform_violation_prob"]),
            }
        )

    deep_check_all_json = out_root / "deep_check_report_all.json"
    policy_tag_roots = sorted({str(Path(r["run_dir"]).resolve().parent) for r in policy_rows})
    deep_check_all_roots = [
        baseline_root,
        out_root / "seed_repro",
        out_root / "split_robustness",
        out_root / "evidence_mode",
        out_root / "alpha_sweep",
        *[Path(p) for p in policy_tag_roots],
        out_root / "synthetic_grid",
    ]
    dc0 = time.perf_counter()
    deep_check_all = _merge_deep_check_reports(deep_check_all_roots, deep_check_all_json)
    timings["deep_check_all_sec"] = float(time.perf_counter() - dc0)
    expected_stress_findings: list[dict[str, Any]] = []
    unexpected_all_findings: list[dict[str, Any]] = []
    for f in deep_check_all.get("findings", []):
        artifact_norm = str(f.get("artifact", "")).replace("/", "\\").lower()
        ftype = str(f.get("type", ""))
        is_expected_policy_stress = ("\\policy_sweep\\m0p0_" in artifact_norm) and (ftype == "superuniformity_failure")
        is_expected_synth_stress = ("\\synthetic_grid\\" in artifact_norm) and (ftype == "weak_discrimination")
        if is_expected_policy_stress or is_expected_synth_stress:
            expected_stress_findings.append(f)
        else:
            unexpected_all_findings.append(f)
    deep_check_final = _load(deep_check_json)
    deep_reg_final = _load(deep_reg_json)
    suspicious: list[dict[str, Any]] = []
    notes: list[str] = []
    evidence_deltas: list[dict[str, Any]] = []
    if int(deep_check_final.get("num_findings", 0)) > 0:
        suspicious.append({"severity": "high", "message": f"deep_check_results has {int(deep_check_final['num_findings'])} findings."})
    if int(deep_reg_final.get("num_findings_total", 0)) > 0:
        suspicious.append({"severity": "medium", "message": f"deep_check_regimes has {int(deep_reg_final['num_findings_total'])} findings."})
    if unexpected_all_findings:
        suspicious.append(
            {
                "severity": "medium",
                "message": (
                    f"deep_check_results(all) has {len(unexpected_all_findings)} unexpected findings "
                    "(outside intentional stress sweeps)."
                ),
            }
        )
    for row in baseline_rows:
        if float(row["tau_v"]) > 0.15:
            suspicious.append({"severity": "medium", "message": f"FD{int(row['fd']):03d} tau violation high ({row['tau_v']:.3f})."})
        if float(row["rul_cov"]) < 0.95:
            suspicious.append({"severity": "medium", "message": f"FD{int(row['fd']):03d} RUL coverage low ({row['rul_cov']:.3f})."})
    for fd in args.fds:
        fixed = [r for r in evidence_rows if int(r["fd"]) == int(fd) and "fixed_tau" in str(r["run_dir"])]
        marginal = [r for r in evidence_rows if int(r["fd"]) == int(fd) and "marginal_rul" in str(r["run_dir"])]
        if not fixed or not marginal:
            continue
        d_cov = float(marginal[0]["rul_cov"]) - float(fixed[0]["rul_cov"])
        d_tau = float(marginal[0]["tau_v"]) - float(fixed[0]["tau_v"])
        evidence_deltas.append({"fd": int(fd), "delta_rul_cov_marginal_minus_fixed": d_cov, "delta_tau_v_marginal_minus_fixed": d_tau})
        if d_cov < -0.03:
            notes.append(
                (
                    f"FD{int(fd):03d} marginal_rul coverage is lower than fixed_tau by {d_cov:.3f}; "
                    "this ablation is expected to be more conservative because it accumulates evidence over static RUL hypotheses."
                )
            )

    seed_summary: list[dict[str, Any]] = []
    for fd in sorted(set(int(r["fd"]) for r in seed_rows)):
        rows = [r for r in seed_rows if int(r["fd"]) == fd]
        rmse = np.asarray([float(r["rmse"]) for r in rows], dtype=np.float64)
        cov = np.asarray([float(r["rul_cov"]) for r in rows], dtype=np.float64)
        tau = np.asarray([float(r["tau_v"]) for r in rows], dtype=np.float64)
        seed_summary.append(
            {
                "fd": fd,
                "n": int(len(rows)),
                "rmse_mean": float(np.mean(rmse)),
                "rmse_std": float(np.std(rmse, ddof=1)) if rmse.size > 1 else 0.0,
                "rul_cov_mean": float(np.mean(cov)),
                "rul_cov_std": float(np.std(cov, ddof=1)) if cov.size > 1 else 0.0,
                "tau_v_mean": float(np.mean(tau)),
                "tau_v_std": float(np.std(tau, ddof=1)) if tau.size > 1 else 0.0,
            }
        )

    report = {
        "timestamp_local": datetime.now().isoformat(timespec="seconds"),
        "out_root": str(out_root),
        "settings": {
            "fds": args.fds,
            "seed_list": seeds,
            "alphas": alphas,
            "synthetic_rhos": synth_rhos,
            "synthetic_sigmas": synth_sigmas,
            "reuse": bool(args.reuse),
            "refresh_evidence_mode": bool(args.refresh_evidence_mode),
            "baseline_calibration_source": str(args.baseline_calibration_source),
            "external_baselines_json": str(args.external_baselines_json),
            "external_performance_report": str(args.external_performance_report),
            "rul_dataset_summary": str(args.rul_dataset_summary),
            "skip_derived_artifacts": bool(args.skip_derived_artifacts),
        },
        "total_wall_seconds": float(time.perf_counter() - t0),
        "timings_seconds": timings,
        "baseline": baseline_rows,
        "seed_repro": {"rows": seed_rows, "summary": seed_summary},
        "split_robustness": split_rows,
        "evidence_mode": evidence_rows,
        "evidence_mode_deltas": evidence_deltas,
        "alpha_sweep": alpha_rows,
        "policy_sweep": policy_rows,
        "synthetic_grid": synth_rows,
        "deep_checks": {
            "deep_check_results_report": str(deep_check_json),
            "deep_check_results_all_report": str(deep_check_all_json),
            "deep_check_regimes_report": str(deep_reg_json),
            "topology_report": str(topology_json),
            "deep_check_results_findings": int(deep_check_final.get("num_findings", 0)),
            "deep_check_results_all_findings": int(deep_check_all.get("num_findings", 0)),
            "deep_check_results_all_expected_stress_findings": int(len(expected_stress_findings)),
            "deep_check_results_all_unexpected_findings": int(len(unexpected_all_findings)),
            "deep_check_regimes_findings": int(deep_reg_final.get("num_findings_total", 0)),
        },
        "notes": notes,
        "suspicious": {"count": int(len(suspicious)), "findings": suspicious},
    }

    out_json = Path(args.out_json).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    out_md = Path(args.out_md).resolve()
    _write_md(out_md, report)

    derived_artifacts: dict[str, Any] = {}
    if not args.skip_derived_artifacts:
        comp_json = out_json.parent / "baseline_comparison.json"
        comp_md = out_json.parent / "baseline_comparison.md"
        comp_cmd = [
            sys.executable,
            str(ROOT / "scripts" / "build_baseline_comparison.py"),
            "--matrix-report",
            str(out_json),
            "--out-json",
            str(comp_json),
            "--out-md",
            str(comp_md),
        ]
        if str(args.external_baselines_json).strip():
            comp_cmd.extend(["--external-baselines-json", str(args.external_baselines_json).strip()])
        timings["build_baseline_comparison_sec"] = _run(comp_cmd)
        derived_artifacts["baseline_comparison_json"] = str(comp_json)
        derived_artifacts["baseline_comparison_md"] = str(comp_md)

        ext_json = out_json.parent / "external_dataset_summary.json"
        ext_md = out_json.parent / "external_dataset_summary.md"
        ext_cmd = [
            sys.executable,
            str(ROOT / "scripts" / "build_external_dataset_summary.py"),
            "--matrix-report",
            str(out_json),
            "--rul-dataset-summary",
            str(args.rul_dataset_summary),
            "--out-json",
            str(ext_json),
            "--out-md",
            str(ext_md),
        ]
        if str(args.external_performance_report).strip():
            ext_cmd.extend(["--external-performance-report", str(args.external_performance_report).strip()])
        timings["build_external_dataset_summary_sec"] = _run(ext_cmd)
        derived_artifacts["external_dataset_summary_json"] = str(ext_json)
        derived_artifacts["external_dataset_summary_md"] = str(ext_md)

        readiness_json = out_root / "stats_conference_readiness.json"
        readiness_md = out_root / "stats_conference_readiness.md"
        timings["stats_conference_readiness_sec"] = _run(
            [
                sys.executable,
                str(ROOT / "scripts" / "stats_conference_readiness.py"),
                "--report-json",
                str(out_json),
                "--topology-json",
                str(topology_json),
                "--paper-md",
                str(args.paper_md),
                "--out-json",
                str(readiness_json),
                "--out-md",
                str(readiness_md),
            ]
        )
        derived_artifacts["stats_conference_readiness_json"] = str(readiness_json)
        derived_artifacts["stats_conference_readiness_md"] = str(readiness_md)
        if readiness_json.exists():
            try:
                ready = _load(readiness_json)
                report["readiness"] = {
                    "score_10": float(ready.get("score_10", 0.0)),
                    "target_9plus_pass": bool(ready.get("target_9plus_pass", False)),
                }
            except Exception:
                report["readiness"] = {"parse_error": True}

        consistency_json = out_json.parent / "artifact_consistency_report.json"
        consistency_md = out_json.parent / "artifact_consistency_report.md"
        consistency_cmd = [
            sys.executable,
            str(ROOT / "scripts" / "check_artifact_consistency.py"),
            "--external-performance-report",
            str(args.external_performance_report) if str(args.external_performance_report).strip() else str(out_json.parent / "external_performance_report.json"),
            "--external-dataset-summary",
            str(ext_json),
            "--baseline-comparison",
            str(comp_json),
            "--out-json",
            str(consistency_json),
            "--out-md",
            str(consistency_md),
        ]
        timings["artifact_consistency_check_sec"] = _run(consistency_cmd)
        derived_artifacts["artifact_consistency_report_json"] = str(consistency_json)
        derived_artifacts["artifact_consistency_report_md"] = str(consistency_md)

    report["derived_artifacts"] = derived_artifacts
    report["timings_seconds"] = timings
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_md(out_md, report)
    print(f"Saved JSON: {out_json}")
    print(f"Saved Markdown: {out_md}")


if __name__ == "__main__":
    main()

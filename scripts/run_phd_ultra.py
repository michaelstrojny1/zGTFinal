from __future__ import annotations

import argparse
import itertools
import json
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tem.calibration import ConditionalResidualCalibrator  # noqa: E402


@dataclass(frozen=True)
class AblationConfig:
    margin: float
    bins: int
    min_bin_size: int

    @property
    def tag(self) -> str:
        m = str(self.margin).replace(".", "p")
        return f"m{m}_b{self.bins}_mbs{self.min_bin_size}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ultra PhD TEM diagnostics and ablation runner.")
    parser.add_argument("--fds", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--baseline-prefix", type=str, default="final_local_fd")
    parser.add_argument("--out-root", type=str, default="outputs/phd_ultra")
    parser.add_argument("--reuse", action="store_true", help="Reuse existing artifacts when available.")
    parser.add_argument("--run-fulltopo", action="store_true", default=True)
    parser.add_argument("--skip-fulltopo", action="store_true")
    parser.add_argument("--run-ablation", action="store_true", default=True)
    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument("--ablation-margins", type=str, default="0.00,0.01,0.02,0.05")
    parser.add_argument("--ablation-bins", type=str, default="8,12")
    parser.add_argument("--ablation-min-bin-sizes", type=str, default="128,512")
    parser.add_argument("--lambda-bet", type=float, default=0.07)
    parser.add_argument("--evidence-mode", type=str, choices=["fixed_tau", "marginal_rul"], default="fixed_tau")
    parser.add_argument("--tau-max", type=int, default=1000)
    parser.add_argument("--predict-batch-size", type=int, default=8192)
    parser.add_argument("--min-persistence", type=float, default=0.5)
    parser.add_argument("--alert-patience", type=int, default=3)
    parser.add_argument("--healthy-rul-floor", type=float, default=100.0)
    parser.add_argument("--out-json", type=str, default="outputs/phd_ultra_full_report.json")
    parser.add_argument("--out-md", type=str, default="outputs/phd_ultra_full_report.md")
    return parser.parse_args()


def _float_list(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def _int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _run(cmd: list[str], cwd: Path) -> None:
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _ks(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    aa = np.sort(a.astype(np.float64, copy=False))
    bb = np.sort(b.astype(np.float64, copy=False))
    x = np.concatenate([aa, bb], axis=0)
    cdf_a = np.searchsorted(aa, x, side="right") / float(aa.size)
    cdf_b = np.searchsorted(bb, x, side="right") / float(bb.size)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def _pcheck_block(pvals: np.ndarray) -> dict[str, Any]:
    p = np.asarray(pvals, dtype=np.float64).reshape(-1)
    out: dict[str, Any] = {"n": int(p.size)}
    if p.size == 0:
        out["status"] = "empty"
        return out
    out["mean_p"] = float(np.mean(p))
    checks = []
    for a in (0.1, 0.2, 0.5):
        frac = float(np.mean(p <= a))
        slack = max(0.01, 3.0 * math.sqrt(a * (1.0 - a) / float(p.size)))
        bound = a + slack
        checks.append(
            {
                "a": a,
                "frac_le_a": frac,
                "bound": bound,
                "pass": bool(frac <= bound),
            }
        )
    out["checks"] = checks
    out["status"] = "ok" if all(c["pass"] for c in checks) else "fail"
    return out


def _prepare_run_paths(fd: int, baseline_prefix: str, out_root: Path) -> dict[str, Path]:
    base = ROOT / "outputs" / f"{baseline_prefix}{fd:03d}"
    if not base.exists():
        raise FileNotFoundError(f"Baseline dir missing: {base}")
    ckpt = base / f"model_fd{fd:03d}.pt"
    bundle = base / f"calibration_bundle_fd{fd:03d}.npz"
    residual = base / f"calibration_residuals_fd{fd:03d}.npy"
    cal = bundle if bundle.exists() else residual
    if not ckpt.exists() or not cal.exists():
        raise FileNotFoundError(f"Missing checkpoint or calibration for FD{fd:03d} in {base}")
    return {
        "baseline": base,
        "checkpoint": ckpt,
        "calibration": cal,
        "fulltopo": out_root / f"fd{fd:03d}" / "fulltopo",
        "ablation_root": out_root / f"fd{fd:03d}" / "ablation",
    }


def _ensure_fulltopo(
    fd: int,
    paths: dict[str, Path],
    args: argparse.Namespace,
) -> Path:
    out_dir = paths["fulltopo"]
    out_dir.mkdir(parents=True, exist_ok=True)
    tem_path = out_dir / f"tem_metrics_fd{fd:03d}.json"
    audit_path = out_dir / f"audit_fd{fd:03d}.json"
    cache_path = out_dir / f"audit_cache_fd{fd:03d}.npz"
    if args.reuse and tem_path.exists() and audit_path.exists() and cache_path.exists():
        return out_dir

    _run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_tem_cmapss.py"),
            "--fd",
            str(fd),
            "--data-root",
            "data/rul_datasets",
            "--checkpoint",
            str(paths["checkpoint"]),
            "--calibration",
            str(paths["calibration"]),
            "--lambda-bet",
            str(args.lambda_bet),
            "--evidence-mode",
            str(args.evidence_mode),
            "--tau-max",
            str(args.tau_max),
            "--min-persistence",
            str(args.min_persistence),
            "--alert-patience",
            str(args.alert_patience),
            "--predict-batch-size",
            str(args.predict_batch_size),
            "--pvalue-safety-margin",
            "0.02",
            "--topology-level",
            "full",
            "--surface-topology-scope",
            "all",
            "--audit-cache",
            str(cache_path),
            "--out-dir",
            str(out_dir),
        ],
        cwd=ROOT,
    )
    _run(
        [
            sys.executable,
            str(ROOT / "scripts" / "audit_tem.py"),
            "--fd",
            str(fd),
            "--data-root",
            "data/rul_datasets",
            "--checkpoint",
            str(paths["checkpoint"]),
            "--calibration",
            str(paths["calibration"]),
            "--lambda-bet",
            str(args.lambda_bet),
            "--evidence-mode",
            str(args.evidence_mode),
            "--tau-max",
            str(args.tau_max),
            "--pvalue-safety-margin",
            "0.02",
            "--cache",
            str(cache_path),
            "--tem-metrics",
            str(tem_path),
            "--out-dir",
            str(out_dir),
        ],
        cwd=ROOT,
    )
    src_train = paths["baseline"] / f"train_metrics_fd{fd:03d}.json"
    if src_train.exists():
        shutil.copy2(src_train, out_dir / src_train.name)
    return out_dir


def _run_deep_checks(fd: int, run_dir: Path) -> dict[str, Any]:
    no_tau = run_dir / "deep_check_no_taugap.json"
    with_tau = run_dir / "deep_check_with_taugap.json"
    _run(
        [
            sys.executable,
            str(ROOT / "scripts" / "deep_check_results.py"),
            "--outputs-root",
            str(run_dir),
            "--report-path",
            str(no_tau),
        ],
        cwd=ROOT,
    )
    _run(
        [
            sys.executable,
            str(ROOT / "scripts" / "deep_check_results.py"),
            "--flag-tau-identifiability-gap",
            "--outputs-root",
            str(run_dir),
            "--report-path",
            str(with_tau),
        ],
        cwd=ROOT,
    )
    return {
        "no_taugap": _load_json(no_tau),
        "with_taugap": _load_json(with_tau),
    }


def _ablation_configs(args: argparse.Namespace) -> list[AblationConfig]:
    margins = _float_list(args.ablation_margins)
    bins = _int_list(args.ablation_bins)
    min_bins = _int_list(args.ablation_min_bin_sizes)
    out = [
        AblationConfig(margin=float(m), bins=int(b), min_bin_size=int(mb))
        for m, b, mb in itertools.product(margins, bins, min_bins)
    ]
    out.sort(key=lambda x: (x.margin, x.bins, x.min_bin_size))
    return out


def _run_ablation_for_fd(fd: int, paths: dict[str, Path], args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ab_root = paths["ablation_root"]
    ab_root.mkdir(parents=True, exist_ok=True)
    for cfg in _ablation_configs(args):
        out_dir = ab_root / cfg.tag
        out_dir.mkdir(parents=True, exist_ok=True)
        tem_path = out_dir / f"tem_metrics_fd{fd:03d}.json"
        audit_path = out_dir / f"audit_fd{fd:03d}.json"
        cache_path = out_dir / f"audit_cache_fd{fd:03d}.npz"
        deep_path = out_dir / "deep_check_no_taugap.json"

        if not (args.reuse and tem_path.exists() and audit_path.exists() and cache_path.exists()):
            _run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "run_tem_cmapss.py"),
                    "--fd",
                    str(fd),
                    "--data-root",
                    "data/rul_datasets",
                    "--checkpoint",
                    str(paths["checkpoint"]),
                    "--calibration",
                    str(paths["calibration"]),
                    "--lambda-bet",
                    str(args.lambda_bet),
                    "--evidence-mode",
                    str(args.evidence_mode),
                    "--tau-max",
                    str(args.tau_max),
                    "--min-persistence",
                    str(args.min_persistence),
                    "--alert-patience",
                    str(args.alert_patience),
                    "--predict-batch-size",
                    str(args.predict_batch_size),
                    "--pvalue-safety-margin",
                    str(cfg.margin),
                    "--calibration-bins",
                    str(cfg.bins),
                    "--calibration-min-bin-size",
                    str(cfg.min_bin_size),
                    "--topology-level",
                    "lite",
                    "--surface-topology-scope",
                    "none",
                    "--audit-cache",
                    str(cache_path),
                    "--out-dir",
                    str(out_dir),
                ],
                cwd=ROOT,
            )
            _run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "audit_tem.py"),
                    "--fd",
                    str(fd),
                    "--data-root",
                    "data/rul_datasets",
                    "--checkpoint",
                    str(paths["checkpoint"]),
                    "--calibration",
                    str(paths["calibration"]),
                    "--lambda-bet",
                    str(args.lambda_bet),
                    "--evidence-mode",
                    str(args.evidence_mode),
                    "--tau-max",
                    str(args.tau_max),
                    "--pvalue-safety-margin",
                    str(cfg.margin),
                    "--calibration-bins",
                    str(cfg.bins),
                    "--calibration-min-bin-size",
                    str(cfg.min_bin_size),
                    "--cache",
                    str(cache_path),
                    "--tem-metrics",
                    str(tem_path),
                    "--out-dir",
                    str(out_dir),
                ],
                cwd=ROOT,
            )
        if not (args.reuse and deep_path.exists()):
            _run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "deep_check_results.py"),
                    "--outputs-root",
                    str(out_dir),
                    "--report-path",
                    str(deep_path),
                ],
                cwd=ROOT,
            )

        tem = _load_json(tem_path)
        audit = _load_json(audit_path)
        deep = _load_json(deep_path)
        fleet = tem["fleet_summary"]
        rows.append(
            {
                "config": {
                    "pvalue_safety_margin": cfg.margin,
                    "calibration_bins": cfg.bins,
                    "calibration_min_bin_size": cfg.min_bin_size,
                },
                "out_dir": str(out_dir),
                "deep_findings_no_taugap": int(deep["num_findings"]),
                "alert_rate": float(fleet["alert_rate"]),
                "mean_temporal_rul_coverage": float(fleet["mean_temporal_rul_coverage"]),
                "tau_anytime_violation_rate": float(fleet["tau_anytime_violation_rate"]),
                "pvalue_all_frac_le_0.2": float(audit["pvalue_all"]["frac_le_0.2"]),
                "pvalue_healthy_frac_le_0.2": float(audit["pvalue_healthy_prefix"]["frac_le_0.2"]),
                "pvalue_all_mean": float(audit["pvalue_all"]["mean_p"]),
                "pvalue_healthy_mean": float(audit["pvalue_healthy_prefix"]["mean_p"]),
            }
        )

    rows.sort(
        key=lambda r: (
            int(r["deep_findings_no_taugap"]),
            float(r["tau_anytime_violation_rate"]),
            -float(r["mean_temporal_rul_coverage"]),
            abs(float(r["pvalue_all_mean"]) - 0.5),
        )
    )
    return rows


def _strict_diagnostics(fd: int, run_dir: Path, healthy_rul_floor: float) -> dict[str, Any]:
    tem = _load_json(run_dir / f"tem_metrics_fd{fd:03d}.json")
    audit = _load_json(run_dir / f"audit_fd{fd:03d}.json")
    cfg = tem["config"]

    cache = np.load(run_dir / f"audit_cache_fd{fd:03d}.npz")
    pred = np.asarray(cache["pred_flat"], dtype=np.float64).reshape(-1)
    true = np.asarray(cache["true_flat"], dtype=np.float64).reshape(-1)
    run_lengths = np.asarray(cache["run_lengths"], dtype=np.int64).reshape(-1)
    scores = np.abs(pred - true)

    cal_path = Path(audit["calibration_file"])
    if cal_path.suffix.lower() == ".npz":
        blob = np.load(cal_path)
        cal_res = np.asarray(blob["residuals"], dtype=np.float64).reshape(-1)
        cal_true = np.asarray(blob["true_rul"], dtype=np.float64).reshape(-1) if "true_rul" in blob else None
    else:
        cal_res = np.asarray(np.load(cal_path), dtype=np.float64).reshape(-1)
        cal_true = None

    use_cond = bool(cfg.get("use_conditional_calibration", True) and cal_true is not None)
    calibrator = ConditionalResidualCalibrator.from_arrays(
        residuals=cal_res,
        true_rul=cal_true if use_cond else None,
        r_max=int(cfg["r_max"]),
        n_bins=int(cfg.get("calibration_bins", 8)),
        min_bin_size=int(cfg.get("calibration_min_bin_size", 128)),
        pvalue_safety_margin=float(cfg.get("pvalue_safety_margin", 0.0)),
    )
    pvals = calibrator.p_values(scores, implied_rul=true if use_cond else None)

    idx = np.arange(true.shape[0], dtype=np.int64)
    pos_frac = np.zeros_like(true, dtype=np.float64)
    if run_lengths.size > 0:
        start = 0
        for L in run_lengths.tolist():
            end = start + int(L)
            if L <= 1:
                pos_frac[start:end] = 1.0
            else:
                pos_frac[start:end] = np.linspace(0.0, 1.0, int(L), endpoint=True)
            start = end

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
    regime_checks = {name: _pcheck_block(pvals[mask]) for name, mask in masks.items()}

    # Residual-shift diagnostics.
    shift = {
        "global_ks_cal_vs_test": _ks(cal_res, scores),
    }
    if cal_true is not None:
        c_true = np.clip(cal_true.astype(np.float64), 1.0, float(cfg["r_max"]))
        q = np.quantile(c_true, [0.0, 0.25, 0.5, 0.75, 1.0])
        q[0] = 1.0
        q[-1] = float(cfg["r_max"]) + 1.0
        q = np.maximum.accumulate(q)
        for i in range(1, q.shape[0]):
            if q[i] <= q[i - 1]:
                q[i] = q[i - 1] + 1e-6
        bins = []
        for i in range(4):
            lo, hi = float(q[i]), float(q[i + 1])
            m_cal = (c_true >= lo) & (c_true < hi if i < 3 else c_true <= hi)
            m_test = (true >= lo) & (true < hi if i < 3 else true <= hi)
            if int(np.sum(m_cal)) == 0 or int(np.sum(m_test)) == 0:
                continue
            bins.append(
                {
                    "bin": i,
                    "lo": lo,
                    "hi": hi,
                    "n_cal": int(np.sum(m_cal)),
                    "n_test": int(np.sum(m_test)),
                    "mean_res_cal": float(np.mean(cal_res[m_cal])),
                    "mean_res_test": float(np.mean(scores[m_test])),
                    "ks": _ks(cal_res[m_cal], scores[m_test]),
                }
            )
        shift["conditional_bins"] = bins

    per = tem["per_run"]
    cov = np.asarray([float(r["temporal_rul_coverage"]) for r in per], dtype=np.float64)
    topo_gamma = np.asarray([float(r["marginal_evidence_topology"]["curve"]["mean_gamma"]) for r in per], dtype=np.float64)
    ridge_tv = np.asarray(
        [float(r["marginal_evidence_topology"]["ridge"]["total_variation_l1"]) for r in per], dtype=np.float64
    )

    outliers = {
        "lowest_rul_coverage_run_indices": np.argsort(cov)[:3].astype(int).tolist(),
        "highest_topo_gamma_run_indices": np.argsort(-topo_gamma)[:3].astype(int).tolist(),
        "highest_ridge_tv_run_indices": np.argsort(-ridge_tv)[:3].astype(int).tolist(),
    }

    integrity = {
        "has_nan_core": bool(np.isnan(cov).any() or np.isnan(topo_gamma).any() or np.isnan(ridge_tv).any()),
        "coverage_in_range": bool(np.all((cov >= 0.0) & (cov <= 1.0))),
        "surface_topology_present_all_runs": bool(
            all(
                isinstance(r.get("surface_topology"), dict)
                and str(r["surface_topology"].get("backend", "")) != "skipped"
                for r in per
            )
        ),
    }
    return {
        "integrity": integrity,
        "regime_pvalue_checks": regime_checks,
        "residual_shift": shift,
        "outliers": outliers,
        "fleet": tem["fleet_summary"],
        "audit_pvalue_all": audit["pvalue_all"],
        "audit_pvalue_healthy_prefix": audit["pvalue_healthy_prefix"],
    }


def _fd_report_to_md(fd_row: dict[str, Any]) -> list[str]:
    lines = []
    lines.append(f"### {fd_row['fd']}")
    lines.append(f"- Baseline: `{fd_row['baseline_dir']}`")
    lines.append(f"- Fulltopo: `{fd_row['fulltopo_dir']}`")
    lines.append(
        f"- Gates: no_taugap_findings={fd_row['gates']['no_taugap_findings']} | "
        f"integrity_ok={fd_row['gates']['integrity_ok']} | regime_pvalue_ok={fd_row['gates']['regime_pvalue_ok']}"
    )
    fleet = fd_row["strict"]["fleet"]
    lines.append(
        f"- Fleet: alert_rate={fleet['alert_rate']:.3f}, "
        f"rul_cov={fleet['mean_temporal_rul_coverage']:.3f}, "
        f"tau_cov={fleet['mean_temporal_tau_coverage']:.3f}, "
        f"tau_violation={fleet['tau_anytime_violation_rate']:.3f}"
    )
    out = fd_row["strict"]["outliers"]
    lines.append(
        f"- Outliers: low_cov={out['lowest_rul_coverage_run_indices']}, "
        f"high_gamma={out['highest_topo_gamma_run_indices']}, "
        f"high_ridge_tv={out['highest_ridge_tv_run_indices']}"
    )
    if fd_row.get("ablation_best") is not None:
        best = fd_row["ablation_best"]
        cfg = best["config"]
        lines.append(
            f"- Best ablation: margin={cfg['pvalue_safety_margin']}, bins={cfg['calibration_bins']}, "
            f"min_bin={cfg['calibration_min_bin_size']}, findings={best['deep_findings_no_taugap']}, "
            f"rul_cov={best['mean_temporal_rul_coverage']:.3f}, tau_violation={best['tau_anytime_violation_rate']:.3f}"
        )
    return lines


def main() -> None:
    args = parse_args()
    run_fulltopo = bool(args.run_fulltopo and not args.skip_fulltopo)
    run_ablation = bool(args.run_ablation and not args.skip_ablation)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    global_findings: list[dict[str, Any]] = []

    for fd in args.fds:
        paths = _prepare_run_paths(fd, args.baseline_prefix, out_root)
        fulltopo_dir = _ensure_fulltopo(fd, paths, args) if run_fulltopo else paths["baseline"]
        deep = _run_deep_checks(fd, fulltopo_dir)
        strict = _strict_diagnostics(fd, fulltopo_dir, healthy_rul_floor=args.healthy_rul_floor)
        integrity = strict["integrity"]

        regime_ok = all(
            block.get("status") in {"ok", "empty"}
            for block in strict["regime_pvalue_checks"].values()
        )
        gates = {
            "no_taugap_findings": int(deep["no_taugap"]["num_findings"]),
            "with_taugap_findings": int(deep["with_taugap"]["num_findings"]),
            "integrity_ok": bool(
                (not integrity["has_nan_core"])
                and integrity["coverage_in_range"]
                and integrity["surface_topology_present_all_runs"]
            ),
            "regime_pvalue_ok": bool(regime_ok),
        }

        if gates["no_taugap_findings"] > 0:
            global_findings.append(
                {
                    "severity": "high",
                    "fd": fd,
                    "type": "deep_check_failure",
                    "message": f"deep_check_no_taugap has {gates['no_taugap_findings']} findings",
                }
            )
        if not gates["integrity_ok"]:
            global_findings.append(
                {
                    "severity": "high",
                    "fd": fd,
                    "type": "integrity_failure",
                    "message": "Core integrity gate failed (NaN/range/surface presence).",
                }
            )
        if not gates["regime_pvalue_ok"]:
            global_findings.append(
                {
                    "severity": "medium",
                    "fd": fd,
                    "type": "regime_pvalue_failure",
                    "message": "At least one strict regime p-value block failed superuniform checks.",
                }
            )

        row: dict[str, Any] = {
            "fd": f"fd{fd:03d}",
            "baseline_dir": str(paths["baseline"]),
            "fulltopo_dir": str(fulltopo_dir),
            "deep_check": deep,
            "strict": strict,
            "gates": gates,
            "ablation": [],
            "ablation_best": None,
        }
        if run_ablation:
            ab_rows = _run_ablation_for_fd(fd, paths, args)
            row["ablation"] = ab_rows
            row["ablation_best"] = ab_rows[0] if ab_rows else None
        all_rows.append(row)

    report = {
        "timestamp_local": datetime.now().isoformat(timespec="seconds"),
        "settings": {
            "fds": args.fds,
            "baseline_prefix": args.baseline_prefix,
            "run_fulltopo": run_fulltopo,
            "run_ablation": run_ablation,
            "ablation_margins": _float_list(args.ablation_margins),
            "ablation_bins": _int_list(args.ablation_bins),
            "ablation_min_bin_sizes": _int_list(args.ablation_min_bin_sizes),
            "lambda_bet": args.lambda_bet,
            "evidence_mode": args.evidence_mode,
            "tau_max": args.tau_max,
            "predict_batch_size": args.predict_batch_size,
            "min_persistence": args.min_persistence,
            "alert_patience": args.alert_patience,
            "healthy_rul_floor": args.healthy_rul_floor,
        },
        "global_findings": global_findings,
        "fds": all_rows,
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    md_lines = ["# Ultra PhD Full Report", ""]
    md_lines.append("## Global Findings")
    if global_findings:
        for f in global_findings:
            md_lines.append(f"- [{f['severity'].upper()}] FD{int(f['fd']):03d} {f['type']}: {f['message']}")
    else:
        md_lines.append("- No high/medium findings across selected gates.")
    md_lines.append("")
    md_lines.append("## Per-FD")
    for row in all_rows:
        md_lines.extend(_fd_report_to_md(row))
    md_lines.append("")

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"Saved JSON: {out_json}")
    print(f"Saved Markdown: {out_md}")


if __name__ == "__main__":
    main()

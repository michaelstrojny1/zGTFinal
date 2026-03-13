from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run replay-only external policy sweep from frozen checkpoint/calibration "
            "artifacts and select the strongest policy that preserves validity targets."
        )
    )
    p.add_argument("--data-root", type=str, default="data/rul_datasets")
    p.add_argument("--datasets", type=str, default="femto")
    p.add_argument("--fd", type=int, default=1)
    p.add_argument("--max-rul", type=int, default=125)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--reuse-artifacts-root",
        type=str,
        default="outputs/external_real_eval_final_policy_v9",
        help="Root with existing model/calibration artifacts for replay mode.",
    )
    p.add_argument("--alpha-grid", type=str, default="0.001,0.0015,0.002,0.003")
    p.add_argument("--lambda-grid", type=str, default="0.02,0.03,0.04")
    p.add_argument("--margin-grid", type=str, default="0.20,0.25,0.30")
    p.add_argument("--cov-target", type=float, default=0.95)
    p.add_argument("--tau-target", type=float, default=0.05)
    p.add_argument("--out-root", type=str, default="outputs/external_policy_replay_sweep_femto")
    p.add_argument("--out-json", type=str, default="outputs/external_policy_replay_sweep_femto/summary.json")
    p.add_argument("--out-md", type=str, default="outputs/external_policy_replay_sweep_femto/summary.md")
    p.add_argument("--python", type=str, default=sys.executable)
    p.add_argument("--force", action="store_true", help="Re-run points even if report JSON already exists.")
    return p.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(v: Any, nd: int = 3) -> str:
    try:
        return f"{float(v):.{nd}f}"
    except Exception:
        return "n/a"


def _parse_grid(raw: str) -> list[float]:
    vals: list[float] = []
    for tok in str(raw).split(","):
        t = tok.strip()
        if not t:
            continue
        vals.append(float(t))
    if not vals:
        raise ValueError("Grid cannot be empty.")
    return vals


def _tag(alpha: float, lam: float, margin: float) -> str:
    def enc(x: float) -> str:
        return f"{x:.4f}".replace(".", "p")

    return f"a{enc(alpha)}_l{enc(lam)}_m{enc(margin)}"


def _report_ref(path: Path) -> str:
    return str(path) if path.exists() else ""


def _run_one(
    *,
    args: argparse.Namespace,
    alpha: float,
    lam: float,
    margin: float,
    out_root: Path,
    report_json: Path,
    report_md: Path,
) -> dict[str, Any]:
    cmd = [
        args.python,
        str(ROOT / "scripts" / "run_real_external_evals.py"),
        "--data-root",
        str(Path(args.data_root).resolve()),
        "--datasets",
        str(args.datasets),
        "--fd",
        str(int(args.fd)),
        "--max-rul",
        str(int(args.max_rul)),
        "--batch-size",
        str(int(args.batch_size)),
        "--num-workers",
        str(int(args.num_workers)),
        "--seed",
        str(int(args.seed)),
        "--epochs",
        "0",
        "--no-compile",
        "--reuse-artifacts-root",
        str(Path(args.reuse_artifacts_root).resolve()),
        "--alpha",
        str(float(alpha)),
        "--lambda-bet",
        str(float(lam)),
        "--pvalue-safety-margin",
        str(float(margin)),
        "--out-root",
        str(out_root),
        "--out-json",
        str(report_json),
        "--out-md",
        str(report_md),
    ]
    subprocess.run(cmd, cwd=str(ROOT), check=True)
    return _load_json(report_json)


def _width_mean_from_report(report: dict[str, Any]) -> float:
    widths: list[float] = []
    for row in list(report.get("datasets", [])):
        if str(row.get("status", "")).lower() != "ok":
            continue
        tem_path = Path(str(row.get("artifacts", {}).get("tem_metrics", "")))
        if not tem_path.exists():
            continue
        try:
            tem = _load_json(tem_path)
        except Exception:
            continue
        for run in list(tem.get("per_run", [])):
            mean_width = run.get("mean_width")
            if mean_width is None:
                continue
            try:
                widths.append(float(mean_width))
            except Exception:
                continue
    if not widths:
        return float("nan")
    return float(sum(widths) / float(len(widths)))


def _metrics_from_report(report: dict[str, Any]) -> tuple[float, float, float, float, list[str]]:
    ok_rows = [r for r in list(report.get("datasets", [])) if str(r.get("status", "")).lower() == "ok"]
    missing = [
        str(r.get("dataset", "unknown"))
        for r in list(report.get("datasets", []))
        if str(r.get("status", "")).lower() != "ok"
    ]
    if not ok_rows:
        return float("nan"), float("nan"), float("nan"), float("nan"), missing

    cov_vals = [float(r.get("metrics", {}).get("rul_cov", float("nan"))) for r in ok_rows]
    tau_vals = [float(r.get("metrics", {}).get("tau_v", float("nan"))) for r in ok_rows]
    rmse_vals = [float(r.get("metrics", {}).get("rmse", float("nan"))) for r in ok_rows]
    width_mean = _width_mean_from_report(report)
    cov_min = min(cov_vals)
    tau_max = max(tau_vals)
    rmse_mean = sum(rmse_vals) / float(len(rmse_vals))
    return cov_min, tau_max, rmse_mean, width_mean, missing


def main() -> None:
    args = parse_args()

    alpha_grid = _parse_grid(args.alpha_grid)
    lambda_grid = _parse_grid(args.lambda_grid)
    margin_grid = _parse_grid(args.margin_grid)

    out_root = Path(args.out_root).resolve()
    reports_dir = out_root / "reports"
    runs_dir = out_root / "runs"
    reports_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for alpha in alpha_grid:
        for lam in lambda_grid:
            for margin in margin_grid:
                tag = _tag(alpha, lam, margin)
                point_report_json = reports_dir / f"{tag}.json"
                point_report_md = reports_dir / f"{tag}.md"
                point_out_root = runs_dir / tag

                if point_report_json.exists() and not args.force:
                    report = _load_json(point_report_json)
                    ran = False
                else:
                    report = _run_one(
                        args=args,
                        alpha=alpha,
                        lam=lam,
                        margin=margin,
                        out_root=point_out_root,
                        report_json=point_report_json,
                        report_md=point_report_md,
                    )
                    ran = True

                cov_min, tau_max, rmse_mean, width_mean, missing = _metrics_from_report(report)
                cov_shortfall = max(0.0, float(args.cov_target) - cov_min)
                tau_excess = max(0.0, tau_max - float(args.tau_target))
                validity_ok = bool((cov_shortfall <= 0.0) and (tau_excess <= 0.0) and (not missing))
                # Higher alpha/lambda and lower margin correspond to less conservative settings.
                aggressiveness = float(alpha) * 1000.0 + float(lam) * 100.0 - float(margin) * 10.0
                penalty = cov_shortfall + tau_excess

                rows.append(
                    {
                        "tag": tag,
                        "alpha": float(alpha),
                        "lambda_bet": float(lam),
                        "pvalue_safety_margin": float(margin),
                        "cov_min": float(cov_min),
                        "tau_max": float(tau_max),
                        "rmse_mean": float(rmse_mean),
                        "width_mean": float(width_mean),
                        "cov_shortfall": float(cov_shortfall),
                        "tau_excess": float(tau_excess),
                        "validity_ok": validity_ok,
                        "aggressiveness": float(aggressiveness),
                        "selection_penalty": float(penalty),
                        "missing_datasets": missing,
                        "ran": ran,
                        "report_json": _report_ref(point_report_json),
                    }
                )

    valid_rows = [r for r in rows if bool(r["validity_ok"])]
    if valid_rows:
        best = sorted(
            valid_rows,
            key=lambda r: (
                -float(r["width_mean"]),
                float(r["aggressiveness"]),
                -float(r["cov_min"]),
                -float(r["tau_max"]),
            ),
            reverse=True,
        )[0]
        selection_mode = "best_valid_by_width_then_aggressiveness"
    else:
        best = sorted(
            rows,
            key=lambda r: (
                float(r["selection_penalty"]),
                float(r["width_mean"]),
                -float(r["aggressiveness"]),
            ),
        )[0]
        selection_mode = "no_valid_point_best_penalty"

    summary = {
        "settings": {
            "data_root": str(Path(args.data_root).resolve()),
            "datasets": [d.strip() for d in str(args.datasets).split(",") if d.strip()],
            "fd": int(args.fd),
            "reuse_artifacts_root": str(Path(args.reuse_artifacts_root).resolve()),
            "cov_target": float(args.cov_target),
            "tau_target": float(args.tau_target),
            "alpha_grid": [float(x) for x in alpha_grid],
            "lambda_grid": [float(x) for x in lambda_grid],
            "margin_grid": [float(x) for x in margin_grid],
            "num_points": len(rows),
        },
        "selection_mode": selection_mode,
        "best_policy": best,
        "best_policy_report_json": (
            _report_ref(Path(str(best.get("report_json", "")).strip()))
            if isinstance(best, dict) and str(best.get("report_json", "")).strip()
            else ""
        ),
        "num_valid_points": len(valid_rows),
        "valid_width_range": {
            "min": min(float(r["width_mean"]) for r in valid_rows) if valid_rows else None,
            "max": max(float(r["width_mean"]) for r in valid_rows) if valid_rows else None,
        },
        "rows_sorted": sorted(
            rows,
            key=lambda r: (
                int(bool(r["validity_ok"])),
                -float(r["selection_penalty"]),
                -float(r["width_mean"]),
                float(r["aggressiveness"]),
            ),
            reverse=True,
        ),
    }

    out_json = Path(args.out_json).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")

    lines: list[str] = []
    lines.append("# External Policy Replay Sweep Summary")
    lines.append("")
    lines.append(f"- datasets: `{','.join(summary['settings']['datasets'])}`")
    lines.append(f"- points: {summary['settings']['num_points']}")
    lines.append(f"- cov target: {float(args.cov_target):.3f}")
    lines.append(f"- tau target: {float(args.tau_target):.3f}")
    lines.append(f"- valid points: {len(valid_rows)}")
    lines.append(f"- selection mode: `{selection_mode}`")
    lines.append(
        f"- valid width range: {_fmt(summary['valid_width_range']['min'])} to "
        f"{_fmt(summary['valid_width_range']['max'])}"
    )
    lines.append("")
    lines.append("## Selected Policy")
    lines.append(
        f"- alpha={_fmt(best.get('alpha'),4)}, lambda={_fmt(best.get('lambda_bet'),4)}, "
        f"margin={_fmt(best.get('pvalue_safety_margin'),4)}"
    )
    lines.append(
        f"- cov_min={_fmt(best.get('cov_min'))}, tau_max={_fmt(best.get('tau_max'))}, "
        f"width_mean={_fmt(best.get('width_mean'))}, rmse_mean={_fmt(best.get('rmse_mean'))}, "
        f"validity_ok={bool(best.get('validity_ok'))}"
    )
    lines.append("")
    lines.append("## Top Policies")
    lines.append("")
    lines.append("| tag | alpha | lambda | margin | cov_min | tau_max | width_mean | valid |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in summary["rows_sorted"][:15]:
        lines.append(
            f"| {row['tag']} | {_fmt(row['alpha'],4)} | {_fmt(row['lambda_bet'],4)} | "
            f"{_fmt(row['pvalue_safety_margin'],4)} | {_fmt(row['cov_min'])} | {_fmt(row['tau_max'])} | "
            f"{_fmt(row['width_mean'])} | {'yes' if row['validity_ok'] else 'no'} |"
        )
    lines.append("")

    out_md = Path(args.out_md).resolve()
    out_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved summary JSON: {out_json}")
    print(f"Saved summary Markdown: {out_md}")
    print(
        "Selected policy:",
        f"alpha={best['alpha']:.6f}",
        f"lambda={best['lambda_bet']:.6f}",
        f"margin={best['pvalue_safety_margin']:.6f}",
        f"validity_ok={best['validity_ok']}",
    )


if __name__ == "__main__":
    main()

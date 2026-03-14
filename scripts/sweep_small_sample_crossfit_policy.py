from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tem.calibration import ConditionalResidualCalibrator  # noqa: E402
from tem.evidence import TemConfig, infer_true_tau_from_true_rul, run_tem_single_engine  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Replay small-sample external crossfit folds under a grid of TEM policies "
            "without retraining the point models."
        )
    )
    p.add_argument(
        "--crossfit-report",
        type=str,
        default="outputs/external_small_sample_crossfit/report.json",
    )
    p.add_argument("--datasets", type=str, default="")
    p.add_argument("--alpha-grid", type=str, default="0.001,0.002,0.003,0.005,0.01")
    p.add_argument("--lambda-grid", type=str, default="0.02,0.03,0.04,0.05,0.10")
    p.add_argument("--margin-grid", type=str, default="0.19,0.25,0.30,0.35")
    p.add_argument("--cov-target", type=float, default=0.95)
    p.add_argument("--tau-target", type=float, default=0.05)
    p.add_argument(
        "--out-json",
        type=str,
        default="outputs/external_small_sample_crossfit_policy_sweep/summary.json",
    )
    p.add_argument(
        "--out-md",
        type=str,
        default="outputs/external_small_sample_crossfit_policy_sweep/summary.md",
    )
    return p.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _sanitize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        val = float(obj)
        if not np.isfinite(val):
            return None
        return val
    return obj


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_sanitize(obj), indent=2, allow_nan=False), encoding="utf-8")


def _parse_grid(raw: str) -> list[float]:
    vals: list[float] = []
    for tok in str(raw).split(","):
        t = tok.strip()
        if t:
            vals.append(float(t))
    if not vals:
        raise ValueError("Grid cannot be empty.")
    return vals


def _fmt(value: Any, ndigits: int = 3) -> str:
    try:
        return f"{float(value):.{ndigits}f}"
    except Exception:
        return "n/a"


def _tag(alpha: float, lam: float, margin: float) -> str:
    def enc(x: float) -> str:
        return f"{x:.4f}".replace(".", "p")

    return f"a{enc(alpha)}_l{enc(lam)}_m{enc(margin)}"


def _iter_datasets(report: dict[str, Any], dataset_filter: set[str]) -> list[dict[str, Any]]:
    rows = [row for row in list(report.get("datasets", [])) if str(row.get("status", "")).lower() == "ok"]
    if not dataset_filter:
        return rows
    out: list[dict[str, Any]] = []
    for row in rows:
        key = str(row.get("dataset", "")).lower().strip()
        if key in dataset_filter:
            out.append(row)
    return out


def _replay_fold(
    *,
    artifact_path: Path,
    alpha: float,
    lambda_bet: float,
    margin: float,
    max_rul: int,
    calibration_bins: int,
    calibration_min_bin_size: int,
) -> dict[str, float]:
    data = np.load(artifact_path)
    calibration_residuals = np.asarray(data["calibration_residuals"], dtype=np.float32)
    calibration_true_rul = np.asarray(data["calibration_true_rul"], dtype=np.float32)
    pred_test = np.asarray(data["pred_test"], dtype=np.float64).reshape(-1)
    true_test = np.asarray(data["true_test"], dtype=np.float64).reshape(-1)

    calibrator = ConditionalResidualCalibrator.from_arrays(
        residuals=calibration_residuals,
        true_rul=calibration_true_rul,
        r_max=int(max_rul),
        n_bins=int(calibration_bins),
        min_bin_size=int(calibration_min_bin_size),
        pvalue_safety_margin=float(margin),
    )
    cfg = TemConfig(
        r_max=int(max_rul),
        alpha=float(alpha),
        lambda_bet=float(lambda_bet),
        evidence_mode="fixed_tau",
        compute_tau_diagnostics=True,
        use_conditional_calibration=True,
        calibration_bins=int(calibration_bins),
        calibration_min_bin_size=int(calibration_min_bin_size),
        pvalue_safety_margin=float(margin),
        cap_implied_rul=True,
    )
    true_tau = infer_true_tau_from_true_rul(true_test, r_max=int(max_rul))
    tau_max = int(true_test.shape[0] + int(max_rul) + 8)
    tem_res = run_tem_single_engine(
        pred_rul=pred_test,
        true_rul=true_test,
        calibration_residuals=calibration_residuals,
        cfg=cfg,
        tau_max=tau_max,
        true_tau=true_tau,
        calibration_true_rul=calibration_true_rul,
        calibrator=calibrator,
        store_log_k_hist=False,
    )
    width_hist = np.asarray(tem_res.get("width_hist", []), dtype=np.float64)
    return {
        "rul_cov": float(np.mean(np.asarray(tem_res["true_r_in_set_hist"], dtype=np.float64))),
        "tau_v": float(bool(tem_res["tau_anytime_violation"])),
        "mean_width": float(np.mean(width_hist)) if width_hist.size else float("nan"),
    }


def _summarize(values: list[float]) -> dict[str, float | None]:
    arr = np.asarray(values, dtype=np.float64)
    if not arr.size:
        return {"mean": None, "std": None, "min": None, "max": None}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _write_md(path: Path, summary: dict[str, Any]) -> None:
    lines = ["# Small-Sample Crossfit Policy Sweep", ""]
    lines.append(
        "Replay-only TEM policy sweep on frozen small-sample crossfit folds. "
        "Point models and residuals are fixed; only the uncertainty policy changes."
    )
    lines.append("")
    settings = summary.get("settings", {})
    lines.append(
        f"- datasets={','.join(settings.get('datasets', []))}, "
        f"cov_target={_fmt(settings.get('cov_target'))}, tau_target={_fmt(settings.get('tau_target'))}"
    )
    lines.append("")
    best_fold = summary.get("best_fold_valid")
    if isinstance(best_fold, dict):
        overall = best_fold.get("overall", {})
        lines.append("## Best Fold-Valid Policy")
        lines.append(
            f"- alpha={_fmt(best_fold.get('alpha'),4)}, lambda={_fmt(best_fold.get('lambda_bet'),4)}, "
            f"margin={_fmt(best_fold.get('pvalue_safety_margin'),4)}, "
            f"width_mean={_fmt(overall.get('dataset_width_mean_mean'))}, "
            f"fold_cov_min={_fmt(overall.get('fold_cov_min'))}, fold_tau_max={_fmt(overall.get('fold_tau_max'))}"
        )
        lines.append("")
    best_mean = summary.get("best_mean_valid")
    if isinstance(best_mean, dict):
        overall = best_mean.get("overall", {})
        lines.append("## Best Mean-Valid Policy")
        lines.append(
            f"- alpha={_fmt(best_mean.get('alpha'),4)}, lambda={_fmt(best_mean.get('lambda_bet'),4)}, "
            f"margin={_fmt(best_mean.get('pvalue_safety_margin'),4)}, "
            f"width_mean={_fmt(overall.get('dataset_width_mean_mean'))}, "
            f"dataset_cov_mean_min={_fmt(overall.get('dataset_cov_mean_min'))}, "
            f"dataset_tau_mean_max={_fmt(overall.get('dataset_tau_mean_max'))}"
        )
        lines.append("")
    lines.append("## Top Candidates")
    lines.append("")
    lines.append("| tag | alpha | lambda | margin | fold_valid | mean_valid | width_mean | fold_cov_min | fold_tau_max | ds_cov_mean_min | ds_tau_mean_max |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in list(summary.get("top_rows", [])):
        overall = row.get("overall", {})
        lines.append(
            f"| {row.get('tag','n/a')} | {_fmt(row.get('alpha'),4)} | {_fmt(row.get('lambda_bet'),4)} | "
            f"{_fmt(row.get('pvalue_safety_margin'),4)} | {int(bool(row.get('fold_validity_ok')))} | "
            f"{int(bool(row.get('mean_validity_ok')))} | {_fmt(overall.get('dataset_width_mean_mean'))} | "
            f"{_fmt(overall.get('fold_cov_min'))} | {_fmt(overall.get('fold_tau_max'))} | "
            f"{_fmt(overall.get('dataset_cov_mean_min'))} | {_fmt(overall.get('dataset_tau_mean_max'))} |"
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    report = _load_json(Path(args.crossfit_report).resolve())
    settings = report.get("settings", {}) if isinstance(report.get("settings", {}), dict) else {}
    max_rul = int(settings.get("max_rul", 125))
    calibration_bins = int(settings.get("calibration_bins", 8))
    calibration_min_bin_size = int(settings.get("calibration_min_bin_size", 128))
    dataset_filter = {d.strip().lower() for d in str(args.datasets).split(",") if d.strip()}
    dataset_rows = _iter_datasets(report, dataset_filter)
    if not dataset_rows:
        raise ValueError("No matching datasets with status=ok in crossfit report.")

    alpha_grid = _parse_grid(args.alpha_grid)
    lambda_grid = _parse_grid(args.lambda_grid)
    margin_grid = _parse_grid(args.margin_grid)

    rows: list[dict[str, Any]] = []
    for alpha in alpha_grid:
        for lam in lambda_grid:
            for margin in margin_grid:
                dataset_summaries: list[dict[str, Any]] = []
                fold_cov_all: list[float] = []
                fold_tau_all: list[float] = []
                width_mean_all: list[float] = []
                missing_artifacts: list[str] = []
                for ds_row in dataset_rows:
                    ds_key = str(ds_row.get("dataset", "")).lower().strip()
                    cov_vals: list[float] = []
                    tau_vals: list[float] = []
                    width_vals: list[float] = []
                    for fold in list(ds_row.get("folds", [])):
                        artifact_ref = str(fold.get("artifacts_npz", "")).strip()
                        artifact_path = Path(artifact_ref)
                        if not artifact_path.exists():
                            missing_artifacts.append(f"{ds_key}:fold_{int(fold.get('fold_index', -1)):02d}")
                            continue
                        replay = _replay_fold(
                            artifact_path=artifact_path,
                            alpha=float(alpha),
                            lambda_bet=float(lam),
                            margin=float(margin),
                            max_rul=max_rul,
                            calibration_bins=calibration_bins,
                            calibration_min_bin_size=calibration_min_bin_size,
                        )
                        cov_vals.append(float(replay["rul_cov"]))
                        tau_vals.append(float(replay["tau_v"]))
                        width_vals.append(float(replay["mean_width"]))
                        fold_cov_all.append(float(replay["rul_cov"]))
                        fold_tau_all.append(float(replay["tau_v"]))
                        width_mean_all.append(float(replay["mean_width"]))
                    cov_stats = _summarize(cov_vals)
                    tau_stats = _summarize(tau_vals)
                    width_stats = _summarize(width_vals)
                    dataset_summaries.append(
                        {
                            "dataset": ds_key,
                            "num_folds": int(len(cov_vals)),
                            "rul_cov": cov_stats,
                            "tau_v": tau_stats,
                            "mean_width": width_stats,
                        }
                    )

                dataset_cov_mean_min = min(
                    float(ds["rul_cov"]["mean"])
                    for ds in dataset_summaries
                    if ds.get("rul_cov", {}).get("mean") is not None
                )
                dataset_tau_mean_max = max(
                    float(ds["tau_v"]["mean"])
                    for ds in dataset_summaries
                    if ds.get("tau_v", {}).get("mean") is not None
                )
                overall = {
                    "fold_cov_min": float(np.min(np.asarray(fold_cov_all, dtype=np.float64))) if fold_cov_all else None,
                    "fold_tau_max": float(np.max(np.asarray(fold_tau_all, dtype=np.float64))) if fold_tau_all else None,
                    "dataset_cov_mean_min": dataset_cov_mean_min,
                    "dataset_tau_mean_max": dataset_tau_mean_max,
                    "dataset_width_mean_mean": float(np.mean(np.asarray(width_mean_all, dtype=np.float64))) if width_mean_all else None,
                }
                fold_validity_ok = bool(
                    (not missing_artifacts)
                    and overall["fold_cov_min"] is not None
                    and overall["fold_tau_max"] is not None
                    and float(overall["fold_cov_min"]) >= float(args.cov_target)
                    and float(overall["fold_tau_max"]) <= float(args.tau_target)
                )
                mean_validity_ok = bool(
                    (not missing_artifacts)
                    and float(dataset_cov_mean_min) >= float(args.cov_target)
                    and float(dataset_tau_mean_max) <= float(args.tau_target)
                )
                fold_penalty = max(0.0, float(args.cov_target) - float(overall["fold_cov_min"] or 0.0)) + max(
                    0.0, float((overall["fold_tau_max"] or 0.0)) - float(args.tau_target)
                )
                mean_penalty = max(0.0, float(args.cov_target) - float(dataset_cov_mean_min)) + max(
                    0.0, float(dataset_tau_mean_max) - float(args.tau_target)
                )
                aggressiveness = float(alpha) * 1000.0 + float(lam) * 100.0 - float(margin) * 10.0
                rows.append(
                    {
                        "tag": _tag(alpha, lam, margin),
                        "alpha": float(alpha),
                        "lambda_bet": float(lam),
                        "pvalue_safety_margin": float(margin),
                        "datasets": dataset_summaries,
                        "overall": overall,
                        "missing_artifacts": missing_artifacts,
                        "fold_validity_ok": fold_validity_ok,
                        "mean_validity_ok": mean_validity_ok,
                        "fold_penalty": float(fold_penalty),
                        "mean_penalty": float(mean_penalty),
                        "aggressiveness": float(aggressiveness),
                    }
                )

    fold_valid_rows = [row for row in rows if bool(row["fold_validity_ok"])]
    mean_valid_rows = [row for row in rows if bool(row["mean_validity_ok"])]
    best_fold_valid = (
        sorted(
            fold_valid_rows,
            key=lambda row: (
                float(row["overall"]["dataset_width_mean_mean"]),
                -float(row["aggressiveness"]),
                -float(row["overall"]["fold_cov_min"]),
                float(row["overall"]["fold_tau_max"]),
            ),
        )[0]
        if fold_valid_rows
        else None
    )
    best_mean_valid = (
        sorted(
            mean_valid_rows,
            key=lambda row: (
                float(row["overall"]["dataset_width_mean_mean"]),
                -float(row["aggressiveness"]),
                -float(row["overall"]["dataset_cov_mean_min"]),
                float(row["overall"]["dataset_tau_mean_max"]),
            ),
        )[0]
        if mean_valid_rows
        else None
    )
    top_rows = sorted(
        rows,
        key=lambda row: (
            0 if row["fold_validity_ok"] else 1,
            0 if row["mean_validity_ok"] else 1,
            float(row["fold_penalty"]),
            float(row["mean_penalty"]),
            float(row["overall"]["dataset_width_mean_mean"] or np.inf),
            -float(row["aggressiveness"]),
        ),
    )[:10]

    summary = {
        "settings": {
            "crossfit_report": str(Path(args.crossfit_report).resolve()),
            "datasets": [str(row.get("dataset", "")).lower().strip() for row in dataset_rows],
            "max_rul": int(max_rul),
            "calibration_bins": int(calibration_bins),
            "calibration_min_bin_size": int(calibration_min_bin_size),
            "cov_target": float(args.cov_target),
            "tau_target": float(args.tau_target),
            "alpha_grid": alpha_grid,
            "lambda_grid": lambda_grid,
            "margin_grid": margin_grid,
        },
        "best_fold_valid": best_fold_valid,
        "best_mean_valid": best_mean_valid,
        "top_rows": top_rows,
        "rows": rows,
    }
    out_json = Path(args.out_json).resolve()
    _write_json(out_json, summary)
    out_md = Path(args.out_md).resolve()
    _write_md(out_md, summary)
    print(f"Saved JSON: {out_json}")
    print(f"Saved Markdown: {out_md}")


if __name__ == "__main__":
    main()

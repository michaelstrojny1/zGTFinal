from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tem.evidence import infer_true_tau_from_true_rul


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build seed-ensemble external-style baseline package from cached seed predictions.")
    p.add_argument("--matrix-report", type=str, default="outputs/publication_full_rtx4050_report.json")
    p.add_argument("--alpha", type=float, default=0.05, help="Miscoverage for conformalized ensemble interval.")
    p.add_argument("--alpha-conservative", type=float, default=0.01, help="Conservative miscoverage.")
    p.add_argument("--z-alpha", type=float, default=1.959963984540054, help="Gaussian z-score for nominal 95%%.")
    p.add_argument("--z-alpha-conservative", type=float, default=2.5758293035489004, help="Gaussian z-score for nominal 99%%.")
    p.add_argument("--out-json", type=str, default="outputs/external_seed_ensemble_baselines.json")
    p.add_argument("--out-md", type=str, default="outputs/external_seed_ensemble_baselines.md")
    return p.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _save_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_sanitize_json(obj), indent=2, allow_nan=False), encoding="utf-8")


def _conformal_q(abs_residuals: np.ndarray, alpha: float) -> float:
    vals = np.asarray(abs_residuals, dtype=np.float64).reshape(-1)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0
    vals = np.sort(vals)
    n = int(vals.size)
    k = int(math.ceil((n + 1) * (1.0 - float(alpha))))
    idx = int(np.clip(k - 1, 0, n - 1))
    return float(vals[idx])


def _load_cache(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    blob = np.load(path)
    pred = np.asarray(blob["pred_flat"], dtype=np.float64).reshape(-1)
    true = np.asarray(blob["true_flat"], dtype=np.float64).reshape(-1)
    run_lengths = np.asarray(blob["run_lengths"], dtype=np.int64).reshape(-1)
    if pred.shape != true.shape:
        raise ValueError(f"pred/true shape mismatch: {path}")
    if int(np.sum(run_lengths)) != int(pred.shape[0]):
        raise ValueError(f"run_lengths mismatch: {path}")
    return pred, true, run_lengths


def _split_runs(flat: np.ndarray, run_lengths: np.ndarray) -> list[np.ndarray]:
    cuts = np.cumsum(run_lengths[:-1], dtype=np.int64)
    return [np.ascontiguousarray(x) for x in np.split(flat, cuts)]


def _eval_interval(
    pred_runs: list[np.ndarray],
    true_runs: list[np.ndarray],
    q_runs: list[np.ndarray],
    r_max: int,
) -> tuple[float, float, float, int]:
    cov: list[float] = []
    tau_flags: list[bool] = []
    widths: list[float] = []
    n_tau = 0
    for p, y, q in zip(pred_runs, true_runs, q_runs):
        in_set = np.abs(p - y) <= q
        cov.append(float(np.mean(in_set)))
        widths.append(float(np.mean(2.0 * q)))

        tau = infer_true_tau_from_true_rul(y, r_max=int(r_max))
        if tau is None:
            continue
        n_tau += 1
        t_idx = np.arange(1, y.shape[0] + 1, dtype=np.int64)
        implied = np.minimum(np.maximum(tau - (t_idx - 1), 1), int(r_max)).astype(np.float64)
        in_set_tau = np.abs(p - implied) <= q
        tau_flags.append(bool(np.any(~in_set_tau)))

    rul_cov = float(np.mean(np.asarray(cov, dtype=np.float64))) if cov else 0.0
    tau_v = float(np.mean(np.asarray(tau_flags, dtype=np.float64))) if tau_flags else 0.0
    mean_width = float(np.mean(np.asarray(widths, dtype=np.float64))) if widths else 0.0
    return rul_cov, tau_v, mean_width, int(n_tau)


def _write_md(path: Path, out: dict[str, Any]) -> None:
    lines = [
        "# Seed-Ensemble External Baselines",
        "",
        f"- matrix_report: `{out['inputs']['matrix_report']}`",
        f"- alpha={out['inputs']['alpha']:.4f}, alpha_conservative={out['inputs']['alpha_conservative']:.4f}",
        "",
    ]
    for m in out["methods"]:
        lines.append(f"## {m['name']}")
        lines.append(f"- {m.get('description', '')}")
        for row in m["per_fd"]:
            lines.append(
                f"- FD{int(row['fd']):03d}: rmse={float(row['rmse']):.3f}, "
                f"rul_cov={float(row['rul_cov']):.3f}, tau_v={float(row['tau_v']):.3f}, "
                f"mean_width={float(row['mean_width']):.3f}"
            )
        lines.append("")
    lines.append("## Notes")
    for n in out.get("notes", []):
        lines.append(f"- {n}")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    matrix_path = Path(args.matrix_report).resolve()
    matrix = _load_json(matrix_path)

    base_rows = list(matrix.get("baseline", []))
    seed_blob = dict(matrix.get("seed_repro", {}))
    seed_rows = list(seed_blob.get("rows", []))
    if not base_rows:
        raise ValueError("matrix baseline rows missing.")
    if not seed_rows:
        raise ValueError("matrix seed_repro rows missing.")

    base_by_fd = {int(r["fd"]): r for r in base_rows}
    seed_by_fd: dict[int, list[dict[str, Any]]] = {}
    for r in seed_rows:
        seed_by_fd.setdefault(int(r["fd"]), []).append(r)
    fds = sorted(base_by_fd.keys())
    if any(fd not in seed_by_fd for fd in fds):
        missing = [fd for fd in fds if fd not in seed_by_fd]
        raise ValueError(f"Missing seed rows for FDs: {missing}")

    methods_rows: dict[str, list[dict[str, Any]]] = {
        "deep_ensemble_gaussian_95": [],
        "deep_ensemble_gaussian_99": [],
        "deep_ensemble_conformalized_a0p05": [],
        "deep_ensemble_conformalized_a0p01": [],
    }
    diagnostics: list[dict[str, Any]] = []

    for fd in fds:
        strict_row = base_by_fd[fd]
        strict_run_dir = Path(str(strict_row["run_dir"])).resolve()
        strict_tem = _load_json(strict_run_dir / f"tem_metrics_fd{fd:03d}.json")
        r_max = int(strict_tem.get("config", {}).get("r_max", 125))

        train_metrics = _load_json(strict_run_dir / f"train_metrics_fd{fd:03d}.json")
        bundle_path = Path(str(train_metrics.get("calibration_bundle_path", "")))
        if not bundle_path.exists():
            bundle_path = strict_run_dir / f"calibration_bundle_fd{fd:03d}.npz"
        if not bundle_path.exists():
            raise FileNotFoundError(f"Missing calibration bundle for FD{fd:03d}: {bundle_path}")
        bundle = np.load(bundle_path)
        cal_res = np.asarray(bundle["residuals"], dtype=np.float64).reshape(-1)
        q_cal = _conformal_q(cal_res, alpha=float(args.alpha))
        q_cal_cons = _conformal_q(cal_res, alpha=float(args.alpha_conservative))

        seed_entries = sorted(seed_by_fd[fd], key=lambda r: str(r["run_dir"]))
        pred_stack: list[np.ndarray] = []
        true_ref: np.ndarray | None = None
        run_lengths_ref: np.ndarray | None = None
        for s in seed_entries:
            run_dir = Path(str(s["run_dir"])).resolve()
            cache_path = run_dir / f"audit_cache_fd{fd:03d}.npz"
            pred, true, run_lengths = _load_cache(cache_path)
            if true_ref is None:
                true_ref = true
                run_lengths_ref = run_lengths
            else:
                if not np.array_equal(true, true_ref):
                    raise ValueError(f"Seed true mismatch for FD{fd:03d}: {run_dir}")
                if not np.array_equal(run_lengths, run_lengths_ref):
                    raise ValueError(f"Seed run_lengths mismatch for FD{fd:03d}: {run_dir}")
            pred_stack.append(pred)

        stack = np.stack(pred_stack, axis=0)
        pred_mean = np.mean(stack, axis=0)
        pred_std = np.std(stack, axis=0, ddof=1) if stack.shape[0] > 1 else np.zeros_like(pred_mean)
        true = np.asarray(true_ref, dtype=np.float64)
        run_lengths = np.asarray(run_lengths_ref, dtype=np.int64)

        pred_runs = _split_runs(pred_mean, run_lengths)
        true_runs = _split_runs(true, run_lengths)
        std_runs = _split_runs(pred_std, run_lengths)

        q_gauss_95 = [float(args.z_alpha) * s for s in std_runs]
        q_gauss_99 = [float(args.z_alpha_conservative) * s for s in std_runs]
        q_conf_95 = [float(args.z_alpha) * s + float(q_cal) for s in std_runs]
        q_conf_99 = [float(args.z_alpha_conservative) * s + float(q_cal_cons) for s in std_runs]

        rmse = float(np.sqrt(np.mean((pred_mean - true) ** 2)))
        cov95, tau95, w95, n_tau95 = _eval_interval(pred_runs, true_runs, q_gauss_95, r_max=r_max)
        cov99, tau99, w99, n_tau99 = _eval_interval(pred_runs, true_runs, q_gauss_99, r_max=r_max)
        covc95, tauc95, wc95, n_tauc95 = _eval_interval(pred_runs, true_runs, q_conf_95, r_max=r_max)
        covc99, tauc99, wc99, n_tauc99 = _eval_interval(pred_runs, true_runs, q_conf_99, r_max=r_max)

        methods_rows["deep_ensemble_gaussian_95"].append(
            {"fd": fd, "rmse": rmse, "rul_cov": cov95, "tau_v": tau95, "mean_width": w95, "run_dir": ""}
        )
        methods_rows["deep_ensemble_gaussian_99"].append(
            {"fd": fd, "rmse": rmse, "rul_cov": cov99, "tau_v": tau99, "mean_width": w99, "run_dir": ""}
        )
        methods_rows["deep_ensemble_conformalized_a0p05"].append(
            {"fd": fd, "rmse": rmse, "rul_cov": covc95, "tau_v": tauc95, "mean_width": wc95, "run_dir": ""}
        )
        methods_rows["deep_ensemble_conformalized_a0p01"].append(
            {"fd": fd, "rmse": rmse, "rul_cov": covc99, "tau_v": tauc99, "mean_width": wc99, "run_dir": ""}
        )
        diagnostics.append(
            {
                "fd": fd,
                "num_seed_models": int(stack.shape[0]),
                "num_points": int(pred_mean.shape[0]),
                "num_runs": int(run_lengths.shape[0]),
                "num_tau_diag_gauss95": int(n_tau95),
                "num_tau_diag_gauss99": int(n_tau99),
                "num_tau_diag_conf95": int(n_tauc95),
                "num_tau_diag_conf99": int(n_tauc99),
                "q_cal_alpha": float(q_cal),
                "q_cal_alpha_conservative": float(q_cal_cons),
            }
        )

    out = {
        "inputs": {
            "matrix_report": str(matrix_path),
            "alpha": float(args.alpha),
            "alpha_conservative": float(args.alpha_conservative),
            "z_alpha": float(args.z_alpha),
            "z_alpha_conservative": float(args.z_alpha_conservative),
        },
        "methods": [
            {
                "name": "deep_ensemble_gaussian_95",
                "comparator_type": "external",
                "description": "External-style baseline: deep-ensemble predictive std interval using Gaussian z=1.96.",
                "per_fd": methods_rows["deep_ensemble_gaussian_95"],
            },
            {
                "name": "deep_ensemble_gaussian_99",
                "comparator_type": "external",
                "description": "External-style baseline: deep-ensemble predictive std interval using Gaussian z=2.576.",
                "per_fd": methods_rows["deep_ensemble_gaussian_99"],
            },
            {
                "name": "deep_ensemble_conformalized_a0p05",
                "comparator_type": "external",
                "description": "External-style baseline: deep-ensemble Gaussian interval plus split-conformal residual quantile (alpha=0.05).",
                "per_fd": methods_rows["deep_ensemble_conformalized_a0p05"],
            },
            {
                "name": "deep_ensemble_conformalized_a0p01",
                "comparator_type": "external",
                "description": "External-style baseline: deep-ensemble Gaussian interval plus split-conformal residual quantile (alpha=0.01).",
                "per_fd": methods_rows["deep_ensemble_conformalized_a0p01"],
            },
        ],
        "diagnostics": diagnostics,
        "notes": [
            "Seed ensemble is formed from cached seed_repro predictions for FD001-004.",
            "No retraining is performed by this script; it is artifact-replay only.",
            "run_dir is empty because these are derived interval baselines, not standalone TEM runs.",
        ],
    }

    out_json = Path(args.out_json).resolve()
    out_md = Path(args.out_md).resolve()
    _save_json(out_json, out)
    _write_md(out_md, out)
    print(f"Saved JSON: {out_json}")
    print(f"Saved Markdown: {out_md}")


if __name__ == "__main__":
    main()

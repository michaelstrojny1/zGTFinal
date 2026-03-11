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

from tem.calibration import ConditionalResidualCalibrator
from tem.evidence import infer_true_tau_from_true_rul


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build external baseline package from strict_main cached predictions.")
    p.add_argument("--matrix-report", type=str, default="outputs/publication_full_rtx4050_report.json")
    p.add_argument("--alpha", type=float, default=0.05, help="Conformal miscoverage level for interval baselines.")
    p.add_argument(
        "--alpha-conservative",
        type=float,
        default=0.01,
        help="Second, more conservative global split-conformal baseline alpha.",
    )
    p.add_argument("--out-json", type=str, default="outputs/external_baselines_generated.json")
    p.add_argument("--out-md", type=str, default="outputs/external_baselines_generated.md")
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


def _save_json_strict(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_sanitize_json(obj), indent=2, allow_nan=False), encoding="utf-8")


def _conformal_quantile(residuals: np.ndarray, alpha: float) -> float:
    vals = np.asarray(residuals, dtype=np.float64).reshape(-1)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0
    vals = np.sort(vals)
    n = int(vals.size)
    k = int(math.ceil((n + 1) * (1.0 - float(alpha))))
    idx = int(np.clip(k - 1, 0, n - 1))
    return float(vals[idx])


def _split_runs_from_cache(cache_path: Path) -> tuple[list[np.ndarray], list[np.ndarray]]:
    blob = np.load(cache_path)
    pred_flat = np.asarray(blob["pred_flat"], dtype=np.float64).reshape(-1)
    true_flat = np.asarray(blob["true_flat"], dtype=np.float64).reshape(-1)
    lengths = np.asarray(blob["run_lengths"], dtype=np.int64).reshape(-1)
    if int(np.sum(lengths)) != int(pred_flat.shape[0]) or pred_flat.shape[0] != true_flat.shape[0]:
        raise ValueError(f"Malformed cache file: {cache_path}")
    cuts = np.cumsum(lengths[:-1], dtype=np.int64)
    pred_runs = [np.ascontiguousarray(x) for x in np.split(pred_flat, cuts)]
    true_runs = [np.ascontiguousarray(x) for x in np.split(true_flat, cuts)]
    return pred_runs, true_runs


def _evaluate_interval_method(
    pred_runs: list[np.ndarray],
    true_runs: list[np.ndarray],
    r_max: int,
    q_t_runs: list[np.ndarray],
) -> tuple[float, float, int, int]:
    cov_per_engine: list[float] = []
    tau_violation_flags: list[bool] = []
    n_tau = 0
    n_total = 0
    for pred, true, q_t in zip(pred_runs, true_runs, q_t_runs):
        p = np.asarray(pred, dtype=np.float64).reshape(-1)
        y = np.asarray(true, dtype=np.float64).reshape(-1)
        q = np.asarray(q_t, dtype=np.float64).reshape(-1)
        if p.shape[0] != y.shape[0] or q.shape[0] != y.shape[0]:
            raise ValueError("Shape mismatch in run arrays.")
        n_total += 1
        in_set_r = np.abs(p - y) <= q
        cov_per_engine.append(float(np.mean(in_set_r)))

        true_tau = infer_true_tau_from_true_rul(y, r_max=int(r_max))
        if true_tau is None:
            continue
        n_tau += 1
        t_idx = np.arange(1, y.shape[0] + 1, dtype=np.int64)
        implied_true_rul = np.minimum(np.maximum(true_tau - (t_idx - 1), 1), int(r_max)).astype(np.float64)
        in_set_tau = np.abs(p - implied_true_rul) <= q
        tau_violation_flags.append(bool(np.any(~in_set_tau)))

    rul_cov = float(np.mean(np.asarray(cov_per_engine, dtype=np.float64))) if cov_per_engine else 0.0
    tau_v = float(np.mean(np.asarray(tau_violation_flags, dtype=np.float64))) if tau_violation_flags else 0.0
    return rul_cov, tau_v, n_tau, n_total


def _build_conditional_q_runs(
    pred_runs: list[np.ndarray],
    calibrator: ConditionalResidualCalibrator,
    alpha: float,
    r_max: int,
) -> list[np.ndarray]:
    global_q = _conformal_quantile(calibrator.global_sorted_residuals, alpha=float(alpha))
    q_per_bin: dict[int, float] = {}
    for b, residuals in enumerate(calibrator.per_bin_sorted_residuals):
        arr = np.asarray(residuals, dtype=np.float64).reshape(-1)
        if arr.size == 0:
            continue
        q_per_bin[int(b)] = _conformal_quantile(arr, alpha=float(alpha))

    if calibrator.rul_bin_lookup is None:
        return [np.full_like(np.asarray(p, dtype=np.float64), global_q, dtype=np.float64) for p in pred_runs]

    out: list[np.ndarray] = []
    for pred in pred_runs:
        p = np.asarray(pred, dtype=np.float64).reshape(-1)
        p_int = np.clip(np.rint(p).astype(np.int64), 1, int(r_max))
        bin_ids = calibrator.rul_bin_lookup[p_int - 1]
        q = np.empty_like(p, dtype=np.float64)
        for i, b in enumerate(bin_ids):
            q[i] = float(q_per_bin.get(int(b), global_q))
        out.append(q)
    return out


def _write_md(path: Path, obj: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# External Baselines (Generated)")
    lines.append("")
    lines.append(f"- Matrix report: `{obj['inputs']['matrix_report']}`")
    lines.append(f"- Alpha: {obj['inputs']['alpha']}")
    lines.append("")
    for method in obj["methods"]:
        lines.append(f"## {method['name']}")
        lines.append(f"- Comparator type: {method['comparator_type']}")
        lines.append(f"- Description: {method.get('description', '')}")
        for row in method["per_fd"]:
            lines.append(
                f"- FD{int(row['fd']):03d}: rmse={float(row['rmse']):.3f}, "
                f"rul_cov={float(row['rul_cov']):.3f}, tau_v={float(row['tau_v']):.3f}"
            )
        lines.append("")
    lines.append("## Notes")
    for n in obj.get("notes", []):
        lines.append(f"- {n}")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    matrix_path = Path(args.matrix_report).resolve()
    matrix = _load_json(matrix_path)
    baseline_rows = list(matrix.get("baseline", []))
    if not baseline_rows:
        raise ValueError("Matrix report has no baseline rows.")

    fds = sorted({int(r["fd"]) for r in baseline_rows})
    base_by_fd = {int(r["fd"]): r for r in baseline_rows}

    global_rows: list[dict[str, Any]] = []
    conservative_rows: list[dict[str, Any]] = []
    conditional_rows: list[dict[str, Any]] = []
    conditional_conservative_rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for fd in fds:
        row = base_by_fd[fd]
        run_dir = Path(str(row["run_dir"])).resolve()
        cache_path = run_dir / f"audit_cache_fd{fd:03d}.npz"
        train_metrics_path = run_dir / f"train_metrics_fd{fd:03d}.json"
        tem_path = run_dir / f"tem_metrics_fd{fd:03d}.json"
        if not cache_path.exists():
            raise FileNotFoundError(f"Missing cache for FD{fd:03d}: {cache_path}")
        if not tem_path.exists():
            raise FileNotFoundError(f"Missing TEM metrics for FD{fd:03d}: {tem_path}")
        if not train_metrics_path.exists():
            raise FileNotFoundError(f"Missing train metrics for FD{fd:03d}: {train_metrics_path}")

        train_metrics = _load_json(train_metrics_path)
        tem = _load_json(tem_path)
        r_max = int(tem.get("config", {}).get("r_max", 125))
        n_bins = int(tem.get("config", {}).get("calibration_bins", 8))
        min_bin = int(tem.get("config", {}).get("calibration_min_bin_size", 128))

        bundle_path = Path(str(train_metrics.get("calibration_bundle_path", "")))
        if not bundle_path.exists():
            bundle_path = run_dir / f"calibration_bundle_fd{fd:03d}.npz"
        if not bundle_path.exists():
            raise FileNotFoundError(f"Missing calibration bundle for FD{fd:03d}: {bundle_path}")
        bundle = np.load(bundle_path)
        cal_res = np.asarray(bundle["residuals"], dtype=np.float64).reshape(-1)
        cal_true = np.asarray(bundle["true_rul"], dtype=np.float64).reshape(-1)
        if cal_res.size == 0:
            raise ValueError(f"Empty calibration residuals for FD{fd:03d}")

        pred_runs, true_runs = _split_runs_from_cache(cache_path)
        q_global = _conformal_quantile(cal_res, alpha=float(args.alpha))
        q_conservative = _conformal_quantile(cal_res, alpha=float(args.alpha_conservative))
        calibrator = ConditionalResidualCalibrator.from_arrays(
            residuals=cal_res,
            true_rul=cal_true,
            r_max=int(r_max),
            n_bins=int(n_bins),
            min_bin_size=int(min_bin),
            pvalue_safety_margin=0.0,
        )

        # Method 1: global split-conformal intervals.
        q_runs_global = [np.full_like(np.asarray(p, dtype=np.float64), q_global, dtype=np.float64) for p in pred_runs]
        rul_cov_g, tau_v_g, n_tau_g, n_total_g = _evaluate_interval_method(
            pred_runs=pred_runs,
            true_runs=true_runs,
            r_max=r_max,
            q_t_runs=q_runs_global,
        )

        # Method 2: conservative global split-conformal intervals.
        q_runs_conservative = [
            np.full_like(np.asarray(p, dtype=np.float64), q_conservative, dtype=np.float64) for p in pred_runs
        ]
        rul_cov_c, tau_v_c, n_tau_c, n_total_c = _evaluate_interval_method(
            pred_runs=pred_runs,
            true_runs=true_runs,
            r_max=r_max,
            q_t_runs=q_runs_conservative,
        )

        # Method 3/4: conditional (Mondrian-style) split-conformal intervals using predicted-RUL bin assignment.
        q_runs_conditional = _build_conditional_q_runs(
            pred_runs=pred_runs,
            calibrator=calibrator,
            alpha=float(args.alpha),
            r_max=r_max,
        )
        rul_cov_cond, tau_v_cond, n_tau_cond, n_total_cond = _evaluate_interval_method(
            pred_runs=pred_runs,
            true_runs=true_runs,
            r_max=r_max,
            q_t_runs=q_runs_conditional,
        )
        q_runs_conditional_conservative = _build_conditional_q_runs(
            pred_runs=pred_runs,
            calibrator=calibrator,
            alpha=float(args.alpha_conservative),
            r_max=r_max,
        )
        rul_cov_cond_c, tau_v_cond_c, n_tau_cond_c, n_total_cond_c = _evaluate_interval_method(
            pred_runs=pred_runs,
            true_runs=true_runs,
            r_max=r_max,
            q_t_runs=q_runs_conditional_conservative,
        )

        global_rows.append(
            {
                "fd": int(fd),
                "rmse": float(row["rmse"]),
                "rul_cov": float(rul_cov_g),
                "tau_v": float(tau_v_g),
                "run_dir": "",
            }
        )
        conservative_rows.append(
            {
                "fd": int(fd),
                "rmse": float(row["rmse"]),
                "rul_cov": float(rul_cov_c),
                "tau_v": float(tau_v_c),
                "run_dir": "",
            }
        )
        conditional_rows.append(
            {
                "fd": int(fd),
                "rmse": float(row["rmse"]),
                "rul_cov": float(rul_cov_cond),
                "tau_v": float(tau_v_cond),
                "run_dir": "",
            }
        )
        conditional_conservative_rows.append(
            {
                "fd": int(fd),
                "rmse": float(row["rmse"]),
                "rul_cov": float(rul_cov_cond_c),
                "tau_v": float(tau_v_cond_c),
                "run_dir": "",
            }
        )
        diagnostics.append(
            {
                "fd": int(fd),
                "run_dir": str(run_dir),
                "r_max": int(r_max),
                "n_bins": int(n_bins),
                "min_bin_size": int(min_bin),
                "q_global": float(q_global),
                "q_conservative": float(q_conservative),
                "num_conditional_bins": int(len(calibrator.per_bin_sorted_residuals)),
                "num_tau_diagnostics_global": int(n_tau_g),
                "num_runs_global": int(n_total_g),
                "num_tau_diagnostics_conservative": int(n_tau_c),
                "num_runs_conservative": int(n_total_c),
                "num_tau_diagnostics_conditional": int(n_tau_cond),
                "num_runs_conditional": int(n_total_cond),
                "num_tau_diagnostics_conditional_conservative": int(n_tau_cond_c),
                "num_runs_conditional_conservative": int(n_total_cond_c),
            }
        )

    out = {
        "inputs": {
            "matrix_report": str(matrix_path),
            "alpha": float(args.alpha),
        },
        "methods": [
            {
                "name": f"split_conformal_global_a{str(args.alpha).replace('.', 'p')}",
                "comparator_type": "external",
                "description": (
                    "External baseline: classical split-conformal absolute-residual interval with global quantile "
                    f"(alpha={float(args.alpha):.4f})."
                ),
                "per_fd": global_rows,
            },
            {
                "name": f"split_conformal_global_a{str(args.alpha_conservative).replace('.', 'p')}",
                "comparator_type": "external",
                "description": (
                    "External baseline: conservative global split-conformal absolute-residual interval "
                    f"(alpha={float(args.alpha_conservative):.4f})."
                ),
                "per_fd": conservative_rows,
            },
            {
                "name": f"split_conformal_conditional_a{str(args.alpha).replace('.', 'p')}",
                "comparator_type": "external",
                "description": (
                    "External baseline: conditional (Mondrian-style) split-conformal intervals by RUL bin, "
                    f"with predicted-RUL bin assignment (alpha={float(args.alpha):.4f})."
                ),
                "per_fd": conditional_rows,
            },
            {
                "name": f"split_conformal_conditional_a{str(args.alpha_conservative).replace('.', 'p')}",
                "comparator_type": "external",
                "description": (
                    "External baseline: conservative conditional (Mondrian-style) split-conformal intervals by RUL bin, "
                    f"with predicted-RUL bin assignment (alpha={float(args.alpha_conservative):.4f})."
                ),
                "per_fd": conditional_conservative_rows,
            },
        ],
        "diagnostics": diagnostics,
        "notes": [
            "These baselines are generated from strict_main prediction/caching artifacts and use non-sequential split-conformal intervals.",
            "Includes global and conditional (RUL-bin Mondrian) split-conformal ladders at two alpha levels.",
            "run_dir is intentionally empty because these baselines do not emit full TEM per-run artifacts.",
        ],
    }
    out = _sanitize_json(out)

    out_json = Path(args.out_json).resolve()
    _save_json_strict(out_json, out)
    out_md = Path(args.out_md).resolve()
    _write_md(out_md, out)

    print(f"Saved JSON: {out_json}")
    print(f"Saved Markdown: {out_md}")
    print("Generated methods: split_conformal_global + split_conformal_conditional (two alpha levels each)")


if __name__ == "__main__":
    main()

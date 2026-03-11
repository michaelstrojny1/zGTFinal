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
from tem.evidence import TemConfig, infer_true_tau_from_true_rul, run_tem_single_engine, summarize_fleet_tem  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit TEM setup for leakage and validity diagnostics.")
    parser.add_argument("--fd", type=int, default=1)
    parser.add_argument("--data-root", type=str, default="data/rul_datasets")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--calibration", type=str, required=True)
    parser.add_argument("--predict-batch-size", type=int, default=8192)
    parser.add_argument("--tau-max", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--lambda-bet", type=float, default=0.07)
    parser.add_argument("--evidence-mode", type=str, choices=["marginal_rul", "fixed_tau"], default="fixed_tau")
    parser.add_argument("--gamma-crit", type=float, default=1.5)
    parser.add_argument("--width-crit", type=int, default=25)
    parser.add_argument("--min-persistence", type=float, default=0.5)
    parser.add_argument("--alert-patience", type=int, default=3)
    parser.add_argument("--use-conditional-calibration", action="store_true", default=True)
    parser.add_argument("--no-use-conditional-calibration", action="store_true")
    parser.add_argument("--cap-implied-rul", action="store_true", default=True)
    parser.add_argument("--no-cap-implied-rul", action="store_true")
    parser.add_argument("--compute-tau-diagnostics", action="store_true", default=True)
    parser.add_argument("--no-compute-tau-diagnostics", action="store_true")
    parser.add_argument("--calibration-bins", type=int, default=8)
    parser.add_argument("--calibration-min-bin-size", type=int, default=128)
    parser.add_argument(
        "--pvalue-safety-margin",
        type=float,
        default=0.02,
        help="Additive conservative p-value offset in [0,1] to hedge residual-distribution shift.",
    )
    parser.add_argument("--healthy-rul-floor", type=float, default=100.0)
    parser.add_argument("--cache", type=str, default="", help="Optional TEM cache (.npz with pred_flat/true_flat/run_lengths).")
    parser.add_argument("--tem-metrics", type=str, default="", help="Optional tem_metrics JSON path for fleet-summary reuse.")
    parser.add_argument("--out-dir", type=str, default="outputs/audit")
    return parser.parse_args()


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_json(obj: Any, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def get_device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(checkpoint_path: str | Path) -> dict:
    import torch

    return torch.load(checkpoint_path, map_location="cpu")


def load_model(ckpt: dict, device) -> Any:
    from tem.model import FastRULNet

    model = FastRULNet(
        in_channels=int(ckpt["in_channels"]),
        hidden=int(ckpt["hidden"]),
        depth=int(ckpt["depth"]),
        dropout=float(ckpt["dropout"]),
    )
    state_dict = ckpt["state_dict"]
    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k.replace("_orig_mod.", "", 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model


def _empirical_cdf_checks(pvals: np.ndarray) -> dict:
    checks = {
        "n": int(pvals.size),
        "mean_p": float(np.mean(pvals)) if pvals.size else float("nan"),
        "frac_le_0.1": float(np.mean(pvals <= 0.1)) if pvals.size else float("nan"),
        "frac_le_0.2": float(np.mean(pvals <= 0.2)) if pvals.size else float("nan"),
        "frac_le_0.5": float(np.mean(pvals <= 0.5)) if pvals.size else float("nan"),
    }
    if pvals.size:
        checks["superuniform_margin_0.1"] = 0.1 - checks["frac_le_0.1"]
        checks["superuniform_margin_0.2"] = 0.2 - checks["frac_le_0.2"]
        checks["superuniform_margin_0.5"] = 0.5 - checks["frac_le_0.5"]
    return checks


def load_calibration(path: str | Path) -> tuple[np.ndarray, np.ndarray | None]:
    p = Path(path)
    if p.suffix.lower() == ".npy" and "calibration_residuals_" in p.name:
        bundle_name = p.name.replace("calibration_residuals_", "calibration_bundle_").replace(".npy", ".npz")
        bundle = p.with_name(bundle_name)
        if bundle.exists():
            print(f"[AUDIT] Found calibration bundle {bundle}; using it instead of residual-only .npy file.")
            p = bundle
    if p.suffix.lower() == ".npz":
        blob = np.load(p)
        residuals = blob["residuals"].astype(np.float64, copy=False)
        true_rul = blob["true_rul"].astype(np.float64, copy=False) if "true_rul" in blob else None
        return residuals, true_rul
    return np.load(p).astype(np.float64, copy=False), None


def _split_flat_runs(flat: np.ndarray, run_lengths: np.ndarray) -> list[np.ndarray]:
    lengths = np.asarray(run_lengths, dtype=np.int64).reshape(-1)
    if lengths.size == 0:
        return []
    if np.any(lengths < 0):
        raise ValueError("run_lengths must be nonnegative.")
    total = int(np.sum(lengths))
    if total != int(flat.shape[0]):
        raise ValueError(
            f"cache mismatch: sum(run_lengths)={total} but flat length={int(flat.shape[0])}"
        )
    split_idx = np.cumsum(lengths[:-1], dtype=np.int64)
    return [np.asarray(x, dtype=np.float64) for x in np.split(flat, split_idx)]


def load_cache(path: str | Path) -> tuple[list[np.ndarray], list[np.ndarray]]:
    blob = np.load(path)
    pred_flat = np.asarray(blob["pred_flat"], dtype=np.float64).reshape(-1)
    true_flat = np.asarray(blob["true_flat"], dtype=np.float64).reshape(-1)
    run_lengths = np.asarray(blob["run_lengths"], dtype=np.int64).reshape(-1)
    pred_runs = _split_flat_runs(pred_flat, run_lengths)
    true_runs = _split_flat_runs(true_flat, run_lengths)
    if len(pred_runs) != len(true_runs):
        raise ValueError("cache mismatch: pred and true run counts differ.")
    return pred_runs, true_runs


def load_tem_metrics(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)

    calibration, calibration_true_rul = load_calibration(args.calibration)
    ckpt: dict | None = None
    ckpt_window: int | None = None
    ckpt_max_rul: int | None = None
    checkpoint_leakage_flag: bool | None = None

    tem_metrics_path = Path(args.tem_metrics) if args.tem_metrics else None
    tem_metrics_obj: dict | None = None
    if tem_metrics_path is not None and tem_metrics_path.exists():
        try:
            tem_metrics_obj = load_tem_metrics(tem_metrics_path)
            cfg_obj = tem_metrics_obj.get("config", {})
            if "r_max" in cfg_obj:
                ckpt_max_rul = int(cfg_obj["r_max"])
            if "checkpoint_leakage_flag" in tem_metrics_obj:
                checkpoint_leakage_flag = bool(tem_metrics_obj["checkpoint_leakage_flag"])
        except Exception as err:
            print(f"[AUDIT] failed to parse tem-metrics ({tem_metrics_path}): {err}")

    pred_runs: list[np.ndarray] | None = None
    true_runs: list[np.ndarray] | None = None
    pred_source = "model_inference"
    cache_path = Path(args.cache) if args.cache else None
    if cache_path is not None:
        if cache_path.exists():
            pred_runs, true_runs = load_cache(cache_path)
            pred_source = f"cache:{cache_path}"
        else:
            print(f"[AUDIT] cache file not found ({cache_path}); falling back to model inference.")

    if pred_runs is None or true_runs is None:
        if ckpt is None:
            ckpt = load_checkpoint(args.checkpoint)
            ckpt_window = int(ckpt["window_size"])
            ckpt_max_rul = int(ckpt["max_rul"])
            checkpoint_leakage_flag = bool(ckpt.get("leakage_check_passed", True))
        if ckpt_window is None or ckpt_max_rul is None:
            raise RuntimeError("checkpoint metadata missing; cannot run model inference.")
        from tem.data import load_cmapss_splits, predict_runs  # noqa: E402

        device = get_device()
        model = load_model(ckpt, device)
        splits = load_cmapss_splits(
            fd=args.fd,
            data_root=args.data_root,
            window_size=ckpt_window,
            max_rul=ckpt_max_rul,
        )
        pred_runs = predict_runs(
            model,
            splits.test_seq_features,
            device=device,
            batch_size=args.predict_batch_size,
            amp=True,
            flatten_batch_runs=True,
        )
        true_runs = [np.asarray(x, dtype=np.float64) for x in splits.test_seq_targets]

    if ckpt_max_rul is None:
        ckpt = load_checkpoint(args.checkpoint)
        ckpt_max_rul = int(ckpt["max_rul"])
        checkpoint_leakage_flag = bool(ckpt.get("leakage_check_passed", True))

    if len(pred_runs) != len(true_runs):
        raise ValueError("pred_runs and true_runs length mismatch.")

    use_cond = args.use_conditional_calibration and not args.no_use_conditional_calibration
    cap_implied_rul = args.cap_implied_rul and not args.no_cap_implied_rul
    compute_tau_diagnostics = args.compute_tau_diagnostics and not args.no_compute_tau_diagnostics
    if use_cond and calibration_true_rul is None:
        print("[AUDIT] calibration true_rul missing; disabling conditional calibration.")
        use_cond = False
    calibrator = ConditionalResidualCalibrator.from_arrays(
        residuals=calibration,
        true_rul=calibration_true_rul if use_cond else None,
        r_max=ckpt_max_rul,
        n_bins=args.calibration_bins,
        min_bin_size=args.calibration_min_bin_size,
        pvalue_safety_margin=args.pvalue_safety_margin,
    )

    all_pvals = []
    healthy_pvals = []
    for pred, true in zip(pred_runs, true_runs):
        pred = np.asarray(pred, dtype=np.float64).reshape(-1)
        true = np.asarray(true, dtype=np.float64).reshape(-1)
        if pred.shape[0] != true.shape[0]:
            raise ValueError("pred and true run lengths differ in audit inputs.")
        scores = np.abs(pred - true)
        pvals = calibrator.p_values(scores, implied_rul=true if use_cond else None)
        all_pvals.append(pvals)
        mask = true >= args.healthy_rul_floor
        if np.any(mask):
            healthy_pvals.append(pvals[mask])

    all_p = np.concatenate(all_pvals) if all_pvals else np.zeros(0, dtype=np.float64)
    healthy_p = np.concatenate(healthy_pvals) if healthy_pvals else np.zeros(0, dtype=np.float64)

    cfg = TemConfig(
        r_max=ckpt_max_rul,
        alpha=args.alpha,
        lambda_bet=args.lambda_bet,
        evidence_mode=args.evidence_mode,
        gamma_crit=args.gamma_crit,
        width_crit=args.width_crit,
        min_persistence=args.min_persistence,
        alert_patience=args.alert_patience,
        cap_implied_rul=cap_implied_rul,
        compute_tau_diagnostics=compute_tau_diagnostics,
        use_conditional_calibration=use_cond,
        calibration_bins=args.calibration_bins,
        calibration_min_bin_size=args.calibration_min_bin_size,
        pvalue_safety_margin=args.pvalue_safety_margin,
    )

    fleet = None
    fleet_source = "audit_recompute"
    if tem_metrics_obj is not None:
        try:
            maybe_fleet = tem_metrics_obj.get("fleet_summary")
            if isinstance(maybe_fleet, dict) and maybe_fleet:
                fleet = maybe_fleet
                fleet_source = f"tem_metrics:{tem_metrics_path}"
        except Exception as err:
            print(f"[AUDIT] failed to load fleet from tem-metrics ({tem_metrics_path}): {err}")

    if fleet is None:
        run_results = []
        for pred, true in zip(pred_runs, true_runs):
            true_tau = infer_true_tau_from_true_rul(true, r_max=ckpt_max_rul)
            run_results.append(
                run_tem_single_engine(
                    pred,
                    true,
                    calibration,
                    cfg,
                    tau_max=args.tau_max,
                    true_tau=true_tau,
                    calibration_true_rul=calibration_true_rul,
                    calibrator=calibrator,
                    store_log_k_hist=False,
                )
            )
        fleet = summarize_fleet_tem(run_results)

    leakage_check_passed = bool(checkpoint_leakage_flag) if checkpoint_leakage_flag is not None else None
    leakage_detected = (not leakage_check_passed) if leakage_check_passed is not None else None

    report = {
        "fd": args.fd,
        "checkpoint": str(args.checkpoint),
        "calibration_file": str(args.calibration),
        # Legacy name kept for backward compatibility; True means leakage check passed.
        "checkpoint_leakage_flag": leakage_check_passed,
        "checkpoint_leakage_check_passed": leakage_check_passed,
        "checkpoint_leakage_detected": leakage_detected,
        "calibration_size": int(calibration.size),
        "pred_source": pred_source,
        "tem_fleet_source": fleet_source,
        "num_runs": int(len(pred_runs)),
        "num_points": int(all_p.size),
        "use_conditional_calibration": bool(use_cond),
        "cap_implied_rul": bool(cap_implied_rul),
        "compute_tau_diagnostics": bool(compute_tau_diagnostics),
        "predict_batch_size": int(args.predict_batch_size),
        "pvalue_safety_margin": float(args.pvalue_safety_margin),
        "pvalue_all": _empirical_cdf_checks(all_p),
        "pvalue_healthy_prefix": _empirical_cdf_checks(healthy_p),
        "tem_fleet": fleet,
        "notes": {
            "superuniform_reference": {
                "frac_le_0.1_should_be_<=~": 0.1,
                "frac_le_0.2_should_be_<=~": 0.2,
                "frac_le_0.5_should_be_<=~": 0.5,
            }
        },
    }

    out_path = out_dir / f"audit_fd{args.fd:03d}.json"
    save_json(report, out_path)
    print(f"Saved audit: {out_path}")
    print(
        f"FD{args.fd:03d} audit | healthy mean_p={report['pvalue_healthy_prefix']['mean_p']:.3f} | "
        f"tau_violation_rate={fleet['tau_anytime_violation_rate']:.3f}"
    )


if __name__ == "__main__":
    main()

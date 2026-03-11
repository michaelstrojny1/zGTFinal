from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tem.data import load_cmapss_splits, predict_runs  # noqa: E402
from tem.calibration import ConditionalResidualCalibrator  # noqa: E402
from tem.evidence import TemConfig, infer_true_tau_from_true_rul, run_tem_single_engine, summarize_fleet_tem  # noqa: E402
from tem.marginal_topology import analyze_marginal_evidence_topology  # noqa: E402
from tem.model import FastRULNet  # noqa: E402
from tem.utils import configure_torch_fast_math, ensure_dir, get_device, save_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Topological Evidence Monitoring on C-MAPSS.")
    parser.add_argument("--fd", type=int, default=1)
    parser.add_argument("--data-root", type=str, default="data/rul_datasets")
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--max-rul", type=int, default=125)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--calibration", type=str, required=True)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--lambda-bet", type=float, default=0.07)
    parser.add_argument("--evidence-mode", type=str, choices=["marginal_rul", "fixed_tau"], default="fixed_tau")
    parser.add_argument("--gamma-crit", type=float, default=1.5)
    parser.add_argument("--width-crit", type=int, default=25)
    parser.add_argument("--min-persistence", type=float, default=0.5)
    parser.add_argument("--alert-patience", type=int, default=3)
    parser.add_argument("--tau-max", type=int, default=1000, help="Fixed hypothesis horizon for tau_h.")
    parser.add_argument("--compute-tau-diagnostics", action="store_true", default=True)
    parser.add_argument("--no-compute-tau-diagnostics", action="store_true")
    parser.add_argument("--cap-implied-rul", action="store_true", default=True)
    parser.add_argument("--no-cap-implied-rul", action="store_true")
    parser.add_argument("--use-conditional-calibration", action="store_true", default=True)
    parser.add_argument("--no-use-conditional-calibration", action="store_true")
    parser.add_argument("--calibration-bins", type=int, default=8)
    parser.add_argument("--calibration-min-bin-size", type=int, default=128)
    parser.add_argument(
        "--pvalue-safety-margin",
        type=float,
        default=0.02,
        help="Additive conservative p-value offset in [0,1] to hedge residual-distribution shift.",
    )
    parser.add_argument("--predict-batch-size", type=int, default=8192)
    parser.add_argument("--topology-level", type=str, choices=["lite", "full"], default="lite")
    parser.add_argument(
        "--surface-topology-scope",
        type=str,
        choices=["none", "plot_run", "all"],
        default="none",
        help="Compute 2D surface persistence for none, only plotted run, or all runs.",
    )
    parser.add_argument("--save-plots", action="store_true", help="Save diagnostic figures (disabled by default for speed).")
    parser.add_argument("--skip-plots", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--audit-cache", type=str, default="", help="Optional path to save pred/true cache for fast audit.")
    parser.add_argument("--plot-run-index", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default="outputs/fd001")
    return parser.parse_args()


def load_model(checkpoint_path: str | Path, device: torch.device) -> tuple[FastRULNet, dict[str, Any]]:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt["state_dict"]
    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k.replace("_orig_mod.", "", 1): v for k, v in state_dict.items()}
    model = FastRULNet(
        in_channels=int(ckpt["in_channels"]),
        hidden=int(ckpt["hidden"]),
        depth=int(ckpt["depth"]),
        dropout=float(ckpt["dropout"]),
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, ckpt


def load_calibration(path: str | Path) -> tuple[np.ndarray, np.ndarray | None]:
    p = Path(path)
    if p.suffix.lower() == ".npy" and "calibration_residuals_" in p.name:
        bundle_name = p.name.replace("calibration_residuals_", "calibration_bundle_").replace(".npy", ".npz")
        bundle = p.with_name(bundle_name)
        if bundle.exists():
            print(f"[TEM] Found calibration bundle {bundle}; using it instead of residual-only .npy file.")
            p = bundle
    if p.suffix.lower() == ".npz":
        blob = np.load(p)
        residuals = blob["residuals"].astype(np.float32, copy=False)
        true_rul = blob["true_rul"].astype(np.float32, copy=False) if "true_rul" in blob else None
        return residuals, true_rul
    return np.load(p).astype(np.float32, copy=False), None


def _lite_topology_from_streamed_hist(
    res: dict[str, Any],
    min_persistence: float,
    r_max: int,
) -> dict[str, Any]:
    max_p = np.asarray(res["max_p_hist"], dtype=np.float64).reshape(-1)
    second_p = np.asarray(res["second_p_hist"], dtype=np.float64).reshape(-1)
    gamma = np.asarray(res["gamma_hist"], dtype=np.float64).reshape(-1)
    r_star = np.asarray(res["r_star_hist"], dtype=np.int64).reshape(-1)
    local_min = np.asarray(res.get("local_minima_hist", np.zeros_like(max_p)), dtype=np.float64).reshape(-1)
    t_steps = int(max_p.shape[0])

    persistent_count_hist = (max_p >= float(min_persistence)).astype(np.float64)
    persistent_count_hist += (second_p >= float(min_persistence)).astype(np.float64)

    if r_star.size > 1:
        jumps = np.abs(np.diff(r_star).astype(np.float64))
        total_variation = float(np.sum(jumps))
        mean_jump = float(np.mean(jumps))
        p90_jump = float(np.quantile(jumps, 0.9))
    else:
        total_variation = 0.0
        mean_jump = 0.0
        p90_jump = 0.0

    return {
        "n_steps": t_steps,
        "r_max": int(r_max),
        "curve_backend": "lite_stream",
        "curve": {
            "mean_max_h0": float(np.mean(max_p)) if t_steps else 0.0,
            "mean_second_h0": float(np.mean(second_p)) if t_steps else 0.0,
            "mean_gamma": float(np.mean(gamma)) if t_steps else 1.0,
            "max_gamma": float(np.max(gamma)) if t_steps else 1.0,
            "mean_persistent_valleys": float(np.mean(persistent_count_hist)) if t_steps else 0.0,
            "mean_local_minima": float(np.mean(local_min)) if t_steps else 0.0,
        },
        "ridge": {
            "r_star_start": int(r_star[0]) if r_star.size else -1,
            "r_star_end": int(r_star[-1]) if r_star.size else -1,
            "total_variation_l1": total_variation,
            "mean_jump": mean_jump,
            "p90_jump": p90_jump,
        },
        "surface": {"backend": "skipped"},
    }


def main() -> None:
    args = parse_args()
    configure_torch_fast_math()
    device = get_device()
    out_dir = ensure_dir(args.out_dir)

    calibration_residuals, calibration_true_rul = load_calibration(args.calibration)
    model, ckpt = load_model(args.checkpoint, device)

    ckpt_window = int(ckpt.get("window_size", args.window_size))
    ckpt_max_rul = int(ckpt.get("max_rul", args.max_rul))
    if ckpt_window != int(args.window_size):
        print(
            f"[TEM] checkpoint window_size={ckpt_window} overrides CLI --window-size={args.window_size} "
            "for preprocessing consistency."
        )
    if ckpt_max_rul != int(args.max_rul):
        print(
            f"[TEM] checkpoint max_rul={ckpt_max_rul} overrides CLI --max-rul={args.max_rul} "
            "for hypothesis-grid consistency."
        )

    splits = load_cmapss_splits(
        fd=args.fd,
        data_root=args.data_root,
        window_size=ckpt_window,
        max_rul=ckpt_max_rul,
    )
    test_pred_runs = predict_runs(
        model,
        splits.test_seq_features,
        device=device,
        batch_size=args.predict_batch_size,
        amp=True,
        flatten_batch_runs=True,
    )
    tau_max = int(args.tau_max)

    cfg = TemConfig(
        r_max=ckpt_max_rul,
        alpha=args.alpha,
        lambda_bet=args.lambda_bet,
        gamma_crit=args.gamma_crit,
        width_crit=args.width_crit,
        min_persistence=args.min_persistence,
        alert_patience=args.alert_patience,
        cap_implied_rul=args.cap_implied_rul and not args.no_cap_implied_rul,
        evidence_mode=args.evidence_mode,
        compute_tau_diagnostics=args.compute_tau_diagnostics and not args.no_compute_tau_diagnostics,
        use_conditional_calibration=args.use_conditional_calibration and not args.no_use_conditional_calibration,
        calibration_bins=args.calibration_bins,
        calibration_min_bin_size=args.calibration_min_bin_size,
        pvalue_safety_margin=args.pvalue_safety_margin,
    )
    if cfg.use_conditional_calibration and calibration_true_rul is None:
        print(
            "[TEM] calibration true_rul not found in file; disabling conditional calibration and using global residuals."
        )
        cfg.use_conditional_calibration = False
    shared_calibrator = ConditionalResidualCalibrator.from_arrays(
        residuals=calibration_residuals,
        true_rul=calibration_true_rul if cfg.use_conditional_calibration else None,
        r_max=cfg.r_max,
        n_bins=cfg.calibration_bins,
        min_bin_size=cfg.calibration_min_bin_size,
        pvalue_safety_margin=cfg.pvalue_safety_margin,
    )

    run_results = []
    run_summaries = []
    cache_pred_runs: list[np.ndarray] = []
    cache_true_runs: list[np.ndarray] = []
    plot_idx = int(np.clip(args.plot_run_index, 0, len(test_pred_runs) - 1)) if test_pred_runs else 0
    save_plots = bool(args.save_plots and not args.skip_plots)
    need_log_k_hist = bool(args.topology_level == "full" or args.surface_topology_scope != "none" or save_plots)
    for run_idx, (pred_run, true_run) in enumerate(zip(test_pred_runs, splits.test_seq_targets)):
        cache_pred_runs.append(np.asarray(pred_run, dtype=np.float32))
        cache_true_runs.append(np.asarray(true_run, dtype=np.float32))
        true_tau = infer_true_tau_from_true_rul(true_run, r_max=cfg.r_max)
        if true_tau is not None and true_tau > tau_max:
            raise ValueError(
                f"Run {run_idx} has true_tau={true_tau} > tau_max={tau_max}. "
                "Increase --tau-max to avoid truncating hypotheses."
            )
        res = run_tem_single_engine(
            pred_run,
            true_run,
            calibration_residuals,
            cfg,
            tau_max=tau_max,
            true_tau=true_tau,
            calibration_true_rul=calibration_true_rul,
            calibrator=shared_calibrator,
            store_log_k_hist=need_log_k_hist,
        )
        run_results.append(res)
        if res.get("log_k_hist") is None:
            topology = _lite_topology_from_streamed_hist(res, min_persistence=args.min_persistence, r_max=cfg.r_max)
        else:
            surface = np.asarray(res["log_k_hist"], dtype=np.float64)
            compute_surface = (
                args.surface_topology_scope == "all" or (args.surface_topology_scope == "plot_run" and run_idx == plot_idx)
            )
            topology = analyze_marginal_evidence_topology(
                surface,
                min_persistence=args.min_persistence,
                max_p_hist=np.asarray(res["max_p_hist"], dtype=np.float64),
                second_p_hist=np.asarray(res["second_p_hist"], dtype=np.float64),
                gamma_hist=np.asarray(res["gamma_hist"], dtype=np.float64),
                topology_level=args.topology_level,
                compute_surface=compute_surface,
            )
        tau_available = bool(res["tau_diagnostics_available"])
        run_summaries.append(
            {
                "run_index": run_idx,
                "num_steps": int(len(true_run)),
                "first_alert_step": int(res["first_alert_step"]),
                "final_r_star": int(np.asarray(res["r_star_hist"])[-1]),
                "final_true_rul": float(np.asarray(true_run)[-1]),
                "mean_gamma": float(np.mean(np.asarray(res["gamma_hist"], dtype=np.float64))),
                "mean_width": float(np.mean(np.asarray(res["width_hist"], dtype=np.float64))),
                "temporal_rul_coverage": float(np.mean(np.asarray(res["true_r_in_set_hist"], dtype=np.float64))),
                "true_tau_estimate": int(res["true_tau"]) if tau_available else None,
                "tau_diagnostics_available": tau_available,
                "temporal_tau_coverage": (
                    float(np.mean(np.asarray(res["true_tau_in_set_hist"], dtype=np.float64))) if tau_available else None
                ),
                "tau_anytime_violation": bool(res["tau_anytime_violation"]) if tau_available else None,
                "marginal_evidence_topology": topology,
                "surface_topology": topology.get("surface"),
            }
        )

    fleet = summarize_fleet_tem(run_results)
    leakage_check_passed = bool(ckpt.get("leakage_check_passed", True))
    tem_report = {
        "fd": args.fd,
        # Legacy name kept for backward compatibility; True means leakage check passed.
        "checkpoint_leakage_flag": leakage_check_passed,
        "checkpoint_leakage_check_passed": leakage_check_passed,
        "checkpoint_leakage_detected": bool(not leakage_check_passed),
        "config": {
            "alpha": args.alpha,
            "lambda_bet": args.lambda_bet,
            "evidence_mode": args.evidence_mode,
            "gamma_crit": args.gamma_crit,
            "width_crit": args.width_crit,
            "min_persistence": args.min_persistence,
            "alert_patience": args.alert_patience,
            "compute_tau_diagnostics": cfg.compute_tau_diagnostics,
            "cap_implied_rul": cfg.cap_implied_rul,
            "use_conditional_calibration": cfg.use_conditional_calibration,
            "calibration_bins": cfg.calibration_bins,
            "calibration_min_bin_size": cfg.calibration_min_bin_size,
            "pvalue_safety_margin": cfg.pvalue_safety_margin,
            "r_max": ckpt_max_rul,
            "window_size": ckpt_window,
            "predict_batch_size": args.predict_batch_size,
            "topology_level": args.topology_level,
            "surface_topology_scope": args.surface_topology_scope,
            "save_plots": save_plots,
            "tau_max": tau_max,
            "checkpoint": str(args.checkpoint),
            "calibration": str(args.calibration),
        },
        "fleet_summary": fleet,
        "per_run": run_summaries,
    }

    save_json(tem_report, out_dir / f"tem_metrics_fd{args.fd:03d}.json")
    with open(out_dir / f"tem_metrics_fd{args.fd:03d}.txt", "w", encoding="utf-8") as f:
        f.write(json.dumps(tem_report, indent=2))

    cache_path = Path(args.audit_cache) if args.audit_cache else out_dir / f"audit_cache_fd{args.fd:03d}.npz"
    if cache_pred_runs:
        run_lengths = np.asarray([x.shape[0] for x in cache_pred_runs], dtype=np.int32)
        pred_flat = np.ascontiguousarray(np.concatenate(cache_pred_runs, axis=0), dtype=np.float32)
        true_flat = np.ascontiguousarray(np.concatenate(cache_true_runs, axis=0), dtype=np.float32)
        np.savez_compressed(cache_path, pred_flat=pred_flat, true_flat=true_flat, run_lengths=run_lengths)

    if run_results and save_plots:
        from tem.plots import plot_evidence_snapshots, plot_evidence_surface, plot_tem_trajectories  # noqa: E402

        idx = plot_idx
        chosen = run_results[idx]
        if chosen.get("log_k_hist") is None:
            raise RuntimeError("Internal error: plotting requested but log_k_hist was not stored.")
        true_rul = np.asarray(splits.test_seq_targets[idx], dtype=np.float64)
        plot_evidence_snapshots(
            np.asarray(chosen["log_k_hist"]),
            true_rul=true_rul,
            out_path=out_dir / f"tem_run_{idx:03d}_evidence_snapshots.png",
        )
        plot_tem_trajectories(
            gamma_hist=np.asarray(chosen["gamma_hist"]),
            width_hist=np.asarray(chosen["width_hist"]),
            r_star_hist=np.asarray(chosen["r_star_hist"]),
            true_rul=true_rul,
            out_path=out_dir / f"tem_run_{idx:03d}_trajectories.png",
        )
        plot_evidence_surface(
            log_k_hist=np.asarray(chosen["log_k_hist"]),
            out_path=out_dir / f"tem_run_{idx:03d}_evidence_surface.png",
        )

    print(
        f"FD{args.fd:03d} TEM complete | engines={fleet['num_engines']} | "
        f"alert_rate={fleet['alert_rate']:.3f} | "
        f"tau_violation_rate={fleet['tau_anytime_violation_rate']:.3f} | "
        f"rul_coverage={fleet['mean_temporal_rul_coverage']:.3f}"
    )
    print(f"Saved: {out_dir / f'tem_metrics_fd{args.fd:03d}.json'}")
    print(f"Saved audit cache: {cache_path}")


if __name__ == "__main__":
    main()

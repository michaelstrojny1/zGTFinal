from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full TEM pipeline (download -> train -> monitor).")
    parser.add_argument("--fd", type=int, default=1)
    parser.add_argument("--data-root", type=str, default="data/rul_datasets")
    parser.add_argument("--out-dir", type=str, default="outputs/fd001")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--healthy-rul-floor", type=float, default=1.0)
    parser.add_argument("--lambda-bet", type=float, default=0.07)
    parser.add_argument("--evidence-mode", type=str, choices=["marginal_rul", "fixed_tau"], default="fixed_tau")
    parser.add_argument("--min-persistence", type=float, default=0.5)
    parser.add_argument("--alert-patience", type=int, default=3)
    parser.add_argument("--tau-max", type=int, default=1000)
    parser.add_argument("--predict-batch-size", type=int, default=8192)
    parser.add_argument("--pvalue-safety-margin", type=float, default=0.02)
    parser.add_argument("--calibration-bins", type=int, default=8)
    parser.add_argument("--calibration-min-bin-size", type=int, default=128)
    parser.add_argument("--topology-level", type=str, choices=["lite", "full"], default="lite")
    parser.add_argument("--surface-topology-scope", type=str, choices=["none", "plot_run", "all"], default="none")
    parser.add_argument("--save-plots", action="store_true")
    parser.add_argument("--calibration-source", type=str, choices=["dev_holdout", "val"], default="val")
    parser.add_argument("--calibration-fraction", type=float, default=0.2)
    parser.add_argument("--skip-audit", action="store_true")
    parser.add_argument("--use-subprocess", action="store_true", help="Run stages as subprocesses (slower, stricter isolation).")
    return parser.parse_args()


def run_script(script: Path, args: list[str], use_subprocess: bool) -> None:
    if use_subprocess:
        import subprocess

        cmd = [sys.executable, str(script), *args]
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
        return

    # Faster path: reuse the current Python process, avoiding repeated interpreter/package startup.
    print(f"{script} {' '.join(args)}", flush=True)
    old_argv = sys.argv[:]
    try:
        sys.argv = [str(script), *args]
        runpy.run_path(str(script), run_name="__main__")
    finally:
        sys.argv = old_argv


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    run_script(
        root / "scripts" / "download_cmapss.py",
        [
            "--fds",
            str(args.fd),
            "--data-root",
            args.data_root,
        ],
        use_subprocess=args.use_subprocess,
    )

    run_script(
        root / "scripts" / "train_fast_cmapss.py",
        [
            "--fd",
            str(args.fd),
            "--data-root",
            args.data_root,
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--seed",
            str(args.seed),
            "--healthy-rul-floor",
            str(args.healthy_rul_floor),
            "--calibration-source",
            str(args.calibration_source),
            "--calibration-fraction",
            str(args.calibration_fraction),
            "--out-dir",
            args.out_dir,
        ],
        use_subprocess=args.use_subprocess,
    )

    ckpt = Path(args.out_dir) / f"model_fd{args.fd:03d}.pt"
    cal_bundle = Path(args.out_dir) / f"calibration_bundle_fd{args.fd:03d}.npz"
    cal = cal_bundle if cal_bundle.exists() else Path(args.out_dir) / f"calibration_residuals_fd{args.fd:03d}.npy"
    tem_metrics_path = Path(args.out_dir) / f"tem_metrics_fd{args.fd:03d}.json"
    audit_cache_path = Path(args.out_dir) / f"audit_cache_fd{args.fd:03d}.npz"
    tem_cmd = [
        "--fd",
        str(args.fd),
        "--data-root",
        args.data_root,
        "--checkpoint",
        str(ckpt),
        "--calibration",
        str(cal),
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
        str(args.pvalue_safety_margin),
        "--calibration-bins",
        str(args.calibration_bins),
        "--calibration-min-bin-size",
        str(args.calibration_min_bin_size),
        "--topology-level",
        str(args.topology_level),
        "--surface-topology-scope",
        str(args.surface_topology_scope),
        "--audit-cache",
        str(audit_cache_path),
        "--out-dir",
        args.out_dir,
    ]
    if args.save_plots:
        tem_cmd.append("--save-plots")
    run_script(root / "scripts" / "run_tem_cmapss.py", tem_cmd, use_subprocess=args.use_subprocess)
    if not args.skip_audit:
        run_script(
            root / "scripts" / "audit_tem.py",
            [
                "--fd",
                str(args.fd),
                "--data-root",
                args.data_root,
                "--checkpoint",
                str(ckpt),
                "--calibration",
                str(cal),
                "--lambda-bet",
                str(args.lambda_bet),
                "--evidence-mode",
                str(args.evidence_mode),
                "--tau-max",
                str(args.tau_max),
                "--predict-batch-size",
                str(args.predict_batch_size),
                "--pvalue-safety-margin",
                str(args.pvalue_safety_margin),
                "--calibration-bins",
                str(args.calibration_bins),
                "--calibration-min-bin-size",
                str(args.calibration_min_bin_size),
                "--cache",
                str(audit_cache_path),
                "--tem-metrics",
                str(tem_metrics_path),
                "--out-dir",
                args.out_dir,
            ],
            use_subprocess=args.use_subprocess,
        )


if __name__ == "__main__":
    main()

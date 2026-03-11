from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


STRICT_POLICY: dict[int, dict[str, float | int]] = {
    1: {"pvalue_safety_margin": 0.08, "calibration_bins": 8, "calibration_min_bin_size": 128},
    2: {"pvalue_safety_margin": 0.11, "calibration_bins": 16, "calibration_min_bin_size": 128},
    3: {"pvalue_safety_margin": 0.18, "calibration_bins": 12, "calibration_min_bin_size": 128},
    4: {"pvalue_safety_margin": 0.25, "calibration_bins": 12, "calibration_min_bin_size": 128},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full pipeline with strict per-FD calibration policy.")
    parser.add_argument("--fd", type=int, required=True, choices=[1, 2, 3, 4])
    parser.add_argument("--data-root", type=str, default="data/rul_datasets")
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--healthy-rul-floor", type=float, default=1.0)
    parser.add_argument("--lambda-bet", type=float, default=0.06)
    parser.add_argument("--evidence-mode", type=str, choices=["fixed_tau", "marginal_rul"], default="fixed_tau")
    parser.add_argument("--min-persistence", type=float, default=0.5)
    parser.add_argument("--alert-patience", type=int, default=3)
    parser.add_argument("--tau-max", type=int, default=1000)
    parser.add_argument("--predict-batch-size", type=int, default=8192)
    parser.add_argument("--surface-topology-scope", type=str, choices=["none", "plot_run", "all"], default="all")
    parser.add_argument("--calibration-source", type=str, choices=["dev_holdout", "val"], default="dev_holdout")
    parser.add_argument("--calibration-fraction", type=float, default=0.2)
    parser.add_argument("--save-plots", action="store_true")
    parser.add_argument("--skip-audit", action="store_true")
    parser.add_argument("--use-subprocess", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    pol = STRICT_POLICY[int(args.fd)]

    cmd = [
        sys.executable,
        str(root / "scripts" / "run_full_pipeline.py"),
        "--fd",
        str(args.fd),
        "--data-root",
        str(args.data_root),
        "--out-dir",
        str(args.out_dir),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--seed",
        str(args.seed),
        "--healthy-rul-floor",
        str(args.healthy_rul_floor),
        "--lambda-bet",
        str(args.lambda_bet),
        "--evidence-mode",
        str(args.evidence_mode),
        "--min-persistence",
        str(args.min_persistence),
        "--alert-patience",
        str(args.alert_patience),
        "--tau-max",
        str(args.tau_max),
        "--predict-batch-size",
        str(args.predict_batch_size),
        "--calibration-source",
        str(args.calibration_source),
        "--calibration-fraction",
        str(args.calibration_fraction),
        "--pvalue-safety-margin",
        str(pol["pvalue_safety_margin"]),
        "--calibration-bins",
        str(pol["calibration_bins"]),
        "--calibration-min-bin-size",
        str(pol["calibration_min_bin_size"]),
        "--surface-topology-scope",
        str(args.surface_topology_scope),
        "--topology-level",
        "full",
    ]
    # run_full_pipeline forwards strict calibration policy to both TEM and audit.
    cmd.extend(["--save-plots"] if args.save_plots else [])
    cmd.extend(["--skip-audit"] if args.skip_audit else [])
    cmd.extend(["--use-subprocess"] if args.use_subprocess else [])
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(root), check=True)


if __name__ == "__main__":
    main()

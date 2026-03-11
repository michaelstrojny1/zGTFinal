from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tem.data import (  # noqa: E402
    RULTensorDataset,
    build_calibration_bundle,
    build_calibration_residuals,
    flatten_runs,
    load_cmapss_splits,
    make_loader,
    predict_numpy,
    predict_runs,
)
from tem.model import FastRULNet  # noqa: E402
from tem.train import evaluate, train_model  # noqa: E402
from tem.utils import configure_torch_fast_math, ensure_dir, get_device, save_json, seed_everything  # noqa: E402


def _sanitize_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if any(k.startswith("_orig_mod.") for k in state_dict):
        return {k.replace("_orig_mod.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def _split_dev_for_calibration(
    features: list[np.ndarray],
    targets: list[np.ndarray],
    cal_fraction: float,
    seed: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray], list[int], list[int]]:
    if not (0.0 < cal_fraction < 0.5):
        raise ValueError("calibration_fraction must be in (0, 0.5).")
    n_runs = len(features)
    if n_runs < 3:
        raise ValueError("Need at least 3 runs in dev split to create train/cal partitions.")
    n_cal = int(round(n_runs * cal_fraction))
    n_cal = min(max(1, n_cal), n_runs - 2)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_runs)
    cal_idx = set(int(i) for i in perm[:n_cal])

    train_x, train_y, cal_x, cal_y = [], [], [], []
    train_indices: list[int] = []
    cal_indices: list[int] = []
    for i in range(n_runs):
        if i in cal_idx:
            cal_x.append(features[i])
            cal_y.append(targets[i])
            cal_indices.append(i)
        else:
            train_x.append(features[i])
            train_y.append(targets[i])
            train_indices.append(i)
    return train_x, train_y, cal_x, cal_y, train_indices, cal_indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast RUL training on C-MAPSS.")
    parser.add_argument("--fd", type=int, default=1)
    parser.add_argument("--data-root", type=str, default="data/rul_datasets")
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--max-rul", type=int, default=125)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=max(2, (os.cpu_count() or 8) // 2))
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--compile", action="store_true", default=False)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument(
        "--healthy-rul-floor",
        type=float,
        default=1.0,
        help="Minimum RUL used for calibration export. Use 1.0 for full-lifecycle validity; "
        "use higher values (e.g., 100) only for healthy-prefix specific analyses.",
    )
    parser.add_argument(
        "--calibration-source",
        type=str,
        choices=["dev_holdout", "val"],
        default="dev_holdout",
        help="Use independent dev holdout (recommended) or val for calibration residuals.",
    )
    parser.add_argument("--calibration-fraction", type=float, default=0.2, help="Used for dev_holdout mode.")
    parser.add_argument("--out-dir", type=str, default="outputs/fd001")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    use_amp = args.amp and not args.no_amp
    use_compile = args.compile and not args.no_compile

    seed_everything(args.seed)
    configure_torch_fast_math()
    device = get_device()
    out_dir = ensure_dir(args.out_dir)

    splits = load_cmapss_splits(
        fd=args.fd,
        data_root=args.data_root,
        window_size=args.window_size,
        max_rul=args.max_rul,
    )

    if args.calibration_source == "dev_holdout":
        train_runs_x, train_runs_y, cal_runs_x, cal_runs_y, train_run_indices, cal_run_indices = _split_dev_for_calibration(
            splits.dev_features,
            splits.dev_targets,
            cal_fraction=args.calibration_fraction,
            seed=args.seed,
        )
    else:
        train_runs_x, train_runs_y = splits.dev_features, splits.dev_targets
        cal_runs_x, cal_runs_y = splits.val_features, splits.val_targets
        train_run_indices = list(range(len(splits.dev_features)))
        cal_run_indices = list(range(len(splits.val_features)))
    leakage_check_passed = (
        set(train_run_indices).isdisjoint(set(cal_run_indices))
        if args.calibration_source == "dev_holdout"
        else True
    )

    train_x, train_y = flatten_runs(train_runs_x, train_runs_y)
    val_x, val_y = flatten_runs(splits.val_features, splits.val_targets)
    test_last_x, test_last_y = flatten_runs(splits.test_last_features, splits.test_last_targets)

    train_ds = RULTensorDataset(train_x, train_y)
    val_ds = RULTensorDataset(val_x, val_y)
    test_last_ds = RULTensorDataset(test_last_x, test_last_y)

    train_loader = make_loader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        prefetch_factor=args.prefetch_factor,
    )
    val_loader = make_loader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        prefetch_factor=args.prefetch_factor,
    )
    test_loader = make_loader(
        test_last_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        prefetch_factor=args.prefetch_factor,
    )

    in_channels = train_x.shape[1]
    model = FastRULNet(
        in_channels=in_channels,
        hidden=args.hidden,
        depth=args.depth,
        dropout=args.dropout,
    )
    result = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        amp=use_amp,
        compile_model=use_compile,
    )
    model.load_state_dict(result.best_state_dict)
    model.to(device)

    val_rmse = evaluate(model, val_loader, device=device, amp=use_amp)
    test_rmse = evaluate(model, test_loader, device=device, amp=use_amp)

    test_pred = predict_numpy(model, test_last_x, device=device, amp=use_amp)
    test_mae = float(np.mean(np.abs(test_pred - test_last_y)))

    val_pred_runs = predict_runs(model, cal_runs_x, device=device, amp=use_amp)
    calibration_residuals = build_calibration_residuals(
        pred_runs=val_pred_runs,
        true_runs=cal_runs_y,
        healthy_rul_floor=args.healthy_rul_floor,
    )
    calibration_bundle = build_calibration_bundle(
        pred_runs=val_pred_runs,
        true_runs=cal_runs_y,
        healthy_rul_floor=args.healthy_rul_floor,
    )

    ckpt_path = out_dir / f"model_fd{args.fd:03d}.pt"
    cal_path = out_dir / f"calibration_residuals_fd{args.fd:03d}.npy"
    cal_bundle_path = out_dir / f"calibration_bundle_fd{args.fd:03d}.npz"
    clean_state_dict = _sanitize_state_dict(model.state_dict())
    torch.save(
        {
            "fd": args.fd,
            "window_size": args.window_size,
            "max_rul": args.max_rul,
            "calibration_source": args.calibration_source,
            "calibration_fraction": args.calibration_fraction,
            "calibration_rul_floor": float(args.healthy_rul_floor),
            "train_run_indices": train_run_indices,
            "calibration_run_indices": cal_run_indices,
            "leakage_check_passed": leakage_check_passed,
            "in_channels": in_channels,
            "hidden": args.hidden,
            "depth": args.depth,
            "dropout": args.dropout,
            "state_dict": clean_state_dict,
        },
        ckpt_path,
    )
    np.save(cal_path, calibration_residuals)
    np.savez_compressed(cal_bundle_path, residuals=calibration_bundle["residuals"], true_rul=calibration_bundle["true_rul"])

    metrics = {
        "device": str(device),
        "epochs": args.epochs,
        "best_val_rmse": result.best_val_rmse,
        "final_val_rmse": val_rmse,
        "test_last_rmse": test_rmse,
        "test_last_mae": test_mae,
        "training_seconds": result.total_seconds,
        "num_train_samples": int(train_x.shape[0]),
        "num_val_samples": int(val_x.shape[0]),
        "num_test_samples": int(test_last_x.shape[0]),
        "calibration_size": int(calibration_residuals.shape[0]),
        "calibration_bundle_path": str(cal_bundle_path),
        "calibration_source": args.calibration_source,
        "calibration_fraction": args.calibration_fraction,
        "calibration_rul_floor": float(args.healthy_rul_floor),
        "leakage_check_passed": bool(leakage_check_passed),
        "num_train_runs": int(len(train_runs_x)),
        "num_calibration_runs": int(len(cal_runs_x)),
        "train_run_indices": train_run_indices,
        "calibration_run_indices": cal_run_indices,
        "history": result.history,
    }
    save_json(metrics, out_dir / f"train_metrics_fd{args.fd:03d}.json")

    print(f"Saved checkpoint: {ckpt_path}")
    print(f"Saved calibration residuals: {cal_path}")
    print(f"Saved calibration bundle: {cal_bundle_path}")
    print(
        f"FD{args.fd:03d} | Val RMSE={val_rmse:.3f} | Test(last) RMSE={test_rmse:.3f} | "
        f"Test(last) MAE={test_mae:.3f} | Train sec={result.total_seconds:.1f}"
    )


if __name__ == "__main__":
    main()

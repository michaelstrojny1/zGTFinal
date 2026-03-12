from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import traceback
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tem.calibration import ConditionalResidualCalibrator  # noqa: E402
from tem.data import (  # noqa: E402
    RULTensorDataset,
    build_calibration_bundle,
    flatten_runs,
    make_loader,
    predict_numpy,
    predict_runs,
)
from tem.evidence import TemConfig, infer_true_tau_from_true_rul, run_tem_single_engine, summarize_fleet_tem  # noqa: E402
from tem.marginal_topology import analyze_marginal_evidence_topology  # noqa: E402
from tem.model import FastRULNet  # noqa: E402
from tem.train import evaluate, train_model  # noqa: E402
from tem.utils import configure_torch_fast_math, ensure_dir, get_device, seed_everything  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run real external dataset evaluations and emit external performance report.")
    p.add_argument("--data-root", type=str, default="data/rul_datasets")
    p.add_argument("--datasets", type=str, default="femto,xjtu_sy", help="Comma-separated: femto,xjtu_sy,cmapss,ncmapss")
    p.add_argument("--fd", type=int, default=1)
    p.add_argument("--max-rul", type=int, default=125)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=max(2, (os.cpu_count() or 8) // 2))
    p.add_argument("--prefetch-factor", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument(
        "--low-rul-loss-weight",
        type=float,
        default=1.0,
        help="Sample-loss upweight factor for low-RUL targets during training (1.0 disables weighting).",
    )
    p.add_argument(
        "--low-rul-threshold",
        type=float,
        default=30.0,
        help="RUL threshold where low-RUL weighting starts to increase toward end-of-life.",
    )
    p.add_argument(
        "--low-rul-weight-power",
        type=float,
        default=1.0,
        help="Exponent controlling low-RUL weighting ramp shape.",
    )
    p.add_argument(
        "--low-rul-weight-datasets",
        type=str,
        default="",
        help="Comma-separated dataset keys to apply low-RUL weighting (empty means all datasets when weight>1).",
    )
    p.add_argument(
        "--dataset-overrides-json",
        type=str,
        default="",
        help=(
            "Optional JSON mapping per dataset key to override training params. "
            "Example: {'femto': {'epochs': 10, 'low_rul_loss_weight': 4.0}, "
            "'xjtu_sy': {'epochs': 15, 'low_rul_loss_weight': 2.0}}"
        ),
    )
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--lambda-bet", type=float, default=0.07)
    p.add_argument("--gamma-crit", type=float, default=1.5)
    p.add_argument("--width-crit", type=int, default=25)
    p.add_argument("--min-persistence", type=float, default=0.5)
    p.add_argument("--alert-patience", type=int, default=3)
    p.add_argument("--calibration-source", type=str, choices=["dev_holdout", "val"], default="val")
    p.add_argument("--calibration-fraction", type=float, default=0.2)
    p.add_argument("--calibration-bins", type=int, default=8)
    p.add_argument("--calibration-min-bin-size", type=int, default=128)
    p.add_argument("--pvalue-safety-margin", type=float, default=0.02)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--out-root", type=str, default="outputs/external_real_eval")
    p.add_argument("--out-json", type=str, default="outputs/external_performance_report.json")
    p.add_argument("--out-md", type=str, default="outputs/external_performance_report.md")
    p.add_argument(
        "--reuse-artifacts-root",
        type=str,
        default="",
        help=(
            "Optional root containing pre-trained per-dataset artifacts "
            "(checkpoint + calibration_bundle). If provided, training is skipped "
            "and policy is replayed with the loaded model/calibration."
        ),
    )
    return p.parse_args()


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


def _set_data_root(root: Path) -> None:
    from rul_datasets.reader.data_root import set_data_root

    root.mkdir(parents=True, exist_ok=True)
    os.environ["RUL_DATASETS_DATA_ROOT"] = str(root)
    set_data_root(str(root))


def _bind_reader_roots(root: Path) -> None:
    from rul_datasets.reader import CmapssReader, FemtoReader, NCmapssReader, XjtuSyReader

    CmapssReader._CMAPSS_ROOT = str(root / "CMAPSS")
    FemtoReader._FEMTO_ROOT = str(root / "FEMTOBearingDataSet")
    XjtuSyReader._XJTU_SY_ROOT = str(root / "XJTU-SY")
    NCmapssReader._NCMAPSS_ROOT = str(root / "NCMAPSS")


def _build_reader(dataset: str, fd: int, max_rul: int):
    from rul_datasets.reader import CmapssReader, FemtoReader, NCmapssReader, XjtuSyReader

    key = dataset.lower().strip()
    if key == "cmapss":
        return CmapssReader(fd=fd, window_size=30, max_rul=max_rul), 30
    if key == "femto":
        return FemtoReader(fd=fd, window_size=1024, max_rul=max_rul), 1024
    if key in {"xjtu_sy", "xjtu-sy"}:
        return XjtuSyReader(fd=fd, window_size=4096, max_rul=max_rul), 4096
    if key == "ncmapss":
        return NCmapssReader(fd=fd, resolution_seconds=10, window_size=300, max_rul=65), 300
    raise ValueError(f"Unknown dataset key: {dataset}")


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
        raise ValueError("Need at least 3 runs in dev split for dev_holdout calibration.")
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


def _load_splits(reader, dataset_key: str) -> dict[str, list[np.ndarray]]:
    reader.prepare_data()
    dev_x, dev_y = reader.load_split("dev")
    val_x, val_y = reader.load_split("val")
    test_x, test_y = reader.load_split("test")

    key = dataset_key.lower().strip()
    if key == "cmapss":
        test_seq_x, test_seq_y = reader.load_split("test", alias="dev")
        test_last_x, test_last_y = test_x, test_y
    else:
        test_seq_x, test_seq_y = test_x, test_y
        test_last_x = [np.ascontiguousarray(x[-1:, :, :], dtype=np.float32) for x in test_seq_x]
        test_last_y = [np.asarray([float(y[-1])], dtype=np.float32) for y in test_seq_y]

    return {
        "dev_x": dev_x,
        "dev_y": dev_y,
        "val_x": val_x,
        "val_y": val_y,
        "test_last_x": test_last_x,
        "test_last_y": test_last_y,
        "test_seq_x": test_seq_x,
        "test_seq_y": test_seq_y,
    }


def _save_json_strict(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_sanitize_json(obj), indent=2, allow_nan=False), encoding="utf-8")


def _is_triton_missing(err: Exception) -> bool:
    txt = f"{type(err).__name__}: {err}".lower()
    return ("tritonmissing" in txt) or ("cannot find a working triton installation" in txt)


def _clip_rul(arr: np.ndarray, max_rul: int) -> np.ndarray:
    # External readers cap labels at max_rul, so clipping predictions keeps metrics
    # and evidence hypotheses on the same support.
    return np.clip(np.asarray(arr, dtype=np.float64), 1.0, float(max_rul))


def _write_md(path: Path, report: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# Real External Performance Report")
    lines.append("")
    lines.append(f"- Data root: `{report['settings']['data_root']}`")
    lines.append(f"- Datasets: `{','.join(report['settings']['datasets'])}`")
    lines.append(f"- FD: {report['settings']['fd']}")
    lines.append(
        f"- alpha={report['settings']['alpha']}, lambda_bet={report['settings']['lambda_bet']}, "
        f"pvalue_safety_margin={report['settings']['pvalue_safety_margin']}"
    )
    lines.append(
        f"- calibration_source={report['settings']['calibration_source']}, "
        f"bins={report['settings']['calibration_bins']}, min_bin_size={report['settings']['calibration_min_bin_size']}"
    )
    lines.append("")
    lines.append("## Dataset Results")
    for row in report["datasets"]:
        status = str(row.get("status", "unknown")).upper()
        lines.append(f"- {row.get('dataset', 'unknown')}: {status}")
        if str(row.get("status", "")).lower() != "ok":
            lines.append(f"  error: {row.get('error', 'n/a')}")
            continue
        m = row.get("metrics", {})
        lines.append(
            f"  rmse={float(m.get('rmse', float('nan'))):.3f}, "
            f"rul_cov={float(m.get('rul_cov', float('nan'))):.3f}, "
            f"tau_v={float(m.get('tau_v', float('nan'))):.3f}, "
            f"tau_ident={float(m.get('tau_identifiability_ratio', float('nan'))):.3f}"
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _evaluate_one_dataset(
    dataset: str,
    dataset_index: int,
    args: argparse.Namespace,
    out_root: Path,
    device: torch.device,
    use_amp: bool,
    use_compile: bool,
    dataset_overrides: dict[str, dict[str, Any]] | None,
) -> dict[str, Any]:
    key = dataset.lower().strip()
    # Dataset-name-based seeding to avoid order-dependent runs.
    seed_offset = int(zlib.adler32(key.encode("utf-8")) % 10000)
    seed = int(args.seed) + seed_offset
    seed_everything(seed)

    reader, window_size = _build_reader(key, fd=int(args.fd), max_rul=int(args.max_rul))
    splits = _load_splits(reader, key)

    overrides = dataset_overrides.get(key, {}) if dataset_overrides else {}

    def _ov(name: str, default: Any, cast):
        if name in overrides:
            try:
                return cast(overrides[name])
            except Exception:
                return default
        return default

    epochs = _ov("epochs", int(args.epochs), int)
    lr = _ov("lr", float(args.lr), float)
    weight_decay = _ov("weight_decay", float(args.weight_decay), float)
    hidden = _ov("hidden", int(args.hidden), int)
    depth = _ov("depth", int(args.depth), int)
    dropout = _ov("dropout", float(args.dropout), float)
    low_rul_loss_weight = _ov("low_rul_loss_weight", float(args.low_rul_loss_weight), float)
    low_rul_threshold = _ov("low_rul_threshold", float(args.low_rul_threshold), float)
    low_rul_weight_power = _ov("low_rul_weight_power", float(args.low_rul_weight_power), float)

    low_weight_datasets_raw = [d.strip().lower() for d in str(args.low_rul_weight_datasets).split(",") if d.strip()]
    low_weight_datasets = set(low_weight_datasets_raw)
    override_has_lowrul = any(k in overrides for k in ("low_rul_loss_weight", "low_rul_threshold", "low_rul_weight_power"))
    apply_low_rul_weight = bool(low_rul_loss_weight > 1.0) and (
        override_has_lowrul or (not low_weight_datasets) or (key in low_weight_datasets)
    )
    loss_weight_used = float(low_rul_loss_weight) if apply_low_rul_weight else 1.0

    cal_mode_used = str(args.calibration_source)
    cal_fallback_reason = ""
    if cal_mode_used == "dev_holdout":
        try:
            train_runs_x, train_runs_y, cal_runs_x, cal_runs_y, train_run_indices, cal_run_indices = _split_dev_for_calibration(
                splits["dev_x"], splits["dev_y"], cal_fraction=float(args.calibration_fraction), seed=seed
            )
            leakage_check_passed = bool(set(train_run_indices).isdisjoint(set(cal_run_indices)))
        except ValueError:
            cal_mode_used = "val"
            cal_fallback_reason = "dev split has too few runs for dev_holdout; fell back to val."

    if cal_mode_used == "val":
        train_runs_x, train_runs_y = splits["dev_x"], splits["dev_y"]
        cal_runs_x, cal_runs_y = splits["val_x"], splits["val_y"]
        train_run_indices = list(range(len(splits["dev_x"])))
        cal_run_indices = list(range(len(splits["val_x"])))
        leakage_check_passed = True

    train_x, train_y = flatten_runs(train_runs_x, train_runs_y)
    val_x, val_y = flatten_runs(splits["val_x"], splits["val_y"])
    test_last_x, test_last_y = flatten_runs(splits["test_last_x"], splits["test_last_y"])

    ds_dir = ensure_dir(out_root / f"{key}_fd{int(args.fd):03d}")

    train_ds = RULTensorDataset(train_x, train_y)
    val_ds = RULTensorDataset(val_x, val_y)
    test_ds = RULTensorDataset(test_last_x, test_last_y)

    train_loader = make_loader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=device.type == "cuda",
        prefetch_factor=int(args.prefetch_factor),
    )
    val_loader = make_loader(
        val_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=device.type == "cuda",
        prefetch_factor=int(args.prefetch_factor),
    )
    test_loader = make_loader(
        test_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=device.type == "cuda",
        prefetch_factor=int(args.prefetch_factor),
    )

    ckpt_path = ds_dir / f"model_fd{int(args.fd):03d}.pt"
    cal_bundle_path = ds_dir / f"calibration_bundle_fd{int(args.fd):03d}.npz"
    train_metrics_path = ds_dir / f"train_metrics_fd{int(args.fd):03d}.json"
    tem_metrics_path = ds_dir / f"tem_metrics_fd{int(args.fd):03d}.json"

    reuse_root_raw = str(getattr(args, "reuse_artifacts_root", "")).strip()
    reuse_mode = bool(reuse_root_raw)
    compile_requested = bool(use_compile)
    compile_used = bool(use_compile)
    compile_fallback_reason = ""
    training_seconds = 0.0
    history: list[dict[str, Any]] = []

    if reuse_mode:
        source_root = Path(reuse_root_raw).resolve()
        source_dir = source_root / f"{key}_fd{int(args.fd):03d}"
        source_ckpt = source_dir / f"model_fd{int(args.fd):03d}.pt"
        source_cal = source_dir / f"calibration_bundle_fd{int(args.fd):03d}.npz"
        if not source_ckpt.exists():
            raise FileNotFoundError(f"{key}: reuse checkpoint missing at {source_ckpt}")
        if not source_cal.exists():
            raise FileNotFoundError(f"{key}: reuse calibration bundle missing at {source_cal}")

        payload = torch.load(source_ckpt, map_location=device, weights_only=False)
        if not isinstance(payload, dict):
            raise RuntimeError(f"{key}: unexpected checkpoint payload type: {type(payload).__name__}")
        state_dict = payload.get("state_dict")
        if not isinstance(state_dict, dict):
            raise RuntimeError(f"{key}: checkpoint missing state_dict in {source_ckpt}")
        in_channels = int(payload.get("in_channels", int(train_x.shape[1])))
        hidden = int(payload.get("hidden", int(hidden)))
        depth = int(payload.get("depth", int(depth)))
        dropout = float(payload.get("dropout", float(dropout)))
        cal_mode_used = str(payload.get("calibration_source", cal_mode_used))
        leakage_check_passed = bool(payload.get("leakage_check_passed", True))

        model = FastRULNet(
            in_channels=in_channels,
            hidden=int(hidden),
            depth=int(depth),
            dropout=float(dropout),
        )
        model.load_state_dict(state_dict)
        model.to(device)
        compile_requested = False
        compile_used = False
        compile_fallback_reason = ""
        epochs = 0
        lr = float("nan")
        weight_decay = float("nan")
        loss_weight_used = float("nan")
        train_run_indices = []
        cal_run_indices = []

        # Snapshot reuse inputs into the output directory for fully-contained artifacts.
        shutil.copy2(source_ckpt, ckpt_path)
        shutil.copy2(source_cal, cal_bundle_path)
        cal_loaded = np.load(source_cal)
        cal_residuals = np.asarray(cal_loaded["residuals"], dtype=np.float32)
        cal_true_rul = np.asarray(cal_loaded["true_rul"], dtype=np.float32)
    else:
        model = FastRULNet(
            in_channels=int(train_x.shape[1]),
            hidden=int(hidden),
            depth=int(depth),
            dropout=float(dropout),
        )
        try:
            result = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                epochs=int(epochs),
                lr=float(lr),
                weight_decay=float(weight_decay),
                amp=use_amp,
                compile_model=compile_used,
                low_rul_loss_weight=float(loss_weight_used),
                low_rul_threshold=float(low_rul_threshold),
                low_rul_weight_power=float(low_rul_weight_power),
            )
        except Exception as err:
            if compile_requested and _is_triton_missing(err):
                compile_used = False
                compile_fallback_reason = (
                    "compile disabled after TritonMissing; reran training with eager mode (no torch.compile)."
                )
                result = train_model(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    device=device,
                    epochs=int(epochs),
                    lr=float(lr),
                    weight_decay=float(weight_decay),
                    amp=use_amp,
                    compile_model=False,
                    low_rul_loss_weight=float(loss_weight_used),
                    low_rul_threshold=float(low_rul_threshold),
                    low_rul_weight_power=float(low_rul_weight_power),
                )
            else:
                raise
        model.load_state_dict(result.best_state_dict)
        model.to(device)
        training_seconds = float(result.total_seconds)
        history = result.history
        torch.save(
            {
                "dataset": key,
                "fd": int(args.fd),
                "window_size": int(window_size),
                "max_rul": int(args.max_rul),
                "calibration_source": cal_mode_used,
                "calibration_fraction": float(args.calibration_fraction),
                "in_channels": int(train_x.shape[1]),
                "hidden": int(hidden),
                "depth": int(depth),
                "dropout": float(dropout),
                "leakage_check_passed": bool(leakage_check_passed),
                "state_dict": model.state_dict(),
            },
            ckpt_path,
        )

    val_rmse = evaluate(model, val_loader, device=device, amp=use_amp)
    test_rmse_last_raw = evaluate(model, test_loader, device=device, amp=use_amp)
    test_pred_last_raw = predict_numpy(model, test_last_x, device=device, amp=use_amp, batch_size=int(args.batch_size))
    test_pred_last = _clip_rul(test_pred_last_raw, max_rul=int(args.max_rul))
    test_rmse_last = float(np.sqrt(np.mean((test_pred_last - test_last_y) ** 2)))
    test_mae_last = float(np.mean(np.abs(test_pred_last - test_last_y)))

    if not reuse_mode:
        cal_pred_runs_raw = predict_runs(
            model,
            cal_runs_x,
            device=device,
            batch_size=int(args.batch_size),
            amp=use_amp,
            flatten_batch_runs=True,
        )
        cal_pred_runs = [
            _clip_rul(np.asarray(x, dtype=np.float64), max_rul=int(args.max_rul)).astype(np.float32)
            for x in cal_pred_runs_raw
        ]
        cal_bundle = build_calibration_bundle(
            pred_runs=cal_pred_runs,
            true_runs=cal_runs_y,
            healthy_rul_floor=1.0,
        )
        cal_residuals = np.asarray(cal_bundle["residuals"], dtype=np.float32)
        cal_true_rul = np.asarray(cal_bundle["true_rul"], dtype=np.float32)
        if cal_residuals.size == 0:
            raise RuntimeError(f"{key}: empty calibration residuals.")
        np.savez_compressed(cal_bundle_path, residuals=cal_residuals, true_rul=cal_true_rul)
    else:
        if cal_residuals.size == 0:
            raise RuntimeError(f"{key}: loaded empty calibration residuals from reuse artifacts.")

    calibrator = ConditionalResidualCalibrator.from_arrays(
        residuals=cal_residuals,
        true_rul=cal_true_rul,
        r_max=int(args.max_rul),
        n_bins=int(args.calibration_bins),
        min_bin_size=int(args.calibration_min_bin_size),
        pvalue_safety_margin=float(args.pvalue_safety_margin),
    )
    cfg = TemConfig(
        r_max=int(args.max_rul),
        alpha=float(args.alpha),
        lambda_bet=float(args.lambda_bet),
        gamma_crit=float(args.gamma_crit),
        width_crit=int(args.width_crit),
        min_persistence=float(args.min_persistence),
        alert_patience=int(args.alert_patience),
        cap_implied_rul=True,
        evidence_mode="fixed_tau",
        compute_tau_diagnostics=True,
        use_conditional_calibration=True,
        calibration_bins=int(args.calibration_bins),
        calibration_min_bin_size=int(args.calibration_min_bin_size),
        pvalue_safety_margin=float(args.pvalue_safety_margin),
    )

    pred_seq_runs_raw = predict_runs(
        model,
        splits["test_seq_x"],
        device=device,
        batch_size=int(args.batch_size),
        amp=use_amp,
        flatten_batch_runs=True,
    )
    pred_seq_runs = [_clip_rul(np.asarray(x, dtype=np.float64), max_rul=int(args.max_rul)) for x in pred_seq_runs_raw]
    y_seq_flat = np.concatenate([np.asarray(y, dtype=np.float64).reshape(-1) for y in splits["test_seq_y"]], axis=0)
    p_seq_flat = np.concatenate([np.asarray(p, dtype=np.float64).reshape(-1) for p in pred_seq_runs], axis=0)
    test_rmse_seq = float(np.sqrt(np.mean((p_seq_flat - y_seq_flat) ** 2)))
    test_mae_seq = float(np.mean(np.abs(p_seq_flat - y_seq_flat)))
    max_len = max(int(np.asarray(y).shape[0]) for y in splits["test_seq_y"]) if splits["test_seq_y"] else 1
    tau_max = int(max_len + int(args.max_rul) + 8)

    run_results: list[dict[str, Any]] = []
    run_summaries: list[dict[str, Any]] = []
    for run_idx, (pred_run, true_run) in enumerate(zip(pred_seq_runs, splits["test_seq_y"])):
        true_arr = np.asarray(true_run, dtype=np.float64).reshape(-1)
        pred_arr = np.asarray(pred_run, dtype=np.float64).reshape(-1)
        true_tau = infer_true_tau_from_true_rul(true_arr, r_max=int(args.max_rul))
        res = run_tem_single_engine(
            pred_rul=pred_arr,
            true_rul=true_arr,
            calibration_residuals=cal_residuals,
            cfg=cfg,
            tau_max=tau_max,
            true_tau=true_tau,
            calibration_true_rul=cal_true_rul,
            calibrator=calibrator,
            store_log_k_hist=True,
        )
        run_results.append(res)
        tau_available = bool(res.get("tau_diagnostics_available", False))
        gamma_hist = np.asarray(res.get("gamma_hist", []), dtype=np.float64)
        width_hist = np.asarray(res.get("width_hist", []), dtype=np.float64)
        topology = analyze_marginal_evidence_topology(
            log_k_hist=np.asarray(res["log_k_hist"], dtype=np.float64),
            min_persistence=float(cfg.min_persistence),
            max_p_hist=np.asarray(res.get("max_p_hist", []), dtype=np.float64),
            second_p_hist=np.asarray(res.get("second_p_hist", []), dtype=np.float64),
            gamma_hist=np.asarray(res.get("gamma_hist", []), dtype=np.float64),
            topology_level="lite",
            compute_surface=True,
        )
        run_summaries.append(
            {
                "run_index": int(run_idx),
                "num_steps": int(true_arr.shape[0]),
                "tau_diagnostics_available": tau_available,
                "first_alert_step": int(res.get("first_alert_step", -1)),
                "temporal_rul_coverage": float(np.mean(np.asarray(res["true_r_in_set_hist"], dtype=np.float64))),
                "tau_anytime_violation": bool(res["tau_anytime_violation"]) if tau_available else None,
                "mean_gamma": float(np.mean(gamma_hist)) if gamma_hist.size else None,
                "mean_width": float(np.mean(width_hist)) if width_hist.size else None,
                "marginal_evidence_topology": topology,
                "surface_topology": topology.get("surface"),
            }
        )

    fleet = summarize_fleet_tem(run_results)
    num_eng = int(fleet.get("num_engines", 0))
    num_tau = int(fleet.get("num_tau_diagnostics_engines", 0))
    tau_ident_ratio = float(num_tau / num_eng) if num_eng > 0 else float("nan")

    train_metrics = {
        "dataset": key,
        "fd": int(args.fd),
        "device": str(device),
        "epochs": int(epochs),
        "best_val_rmse": float(result.best_val_rmse) if not reuse_mode else float(val_rmse),
        "final_val_rmse": float(val_rmse),
        "test_last_rmse_raw": float(test_rmse_last_raw),
        "test_last_rmse": float(test_rmse_last),
        "test_last_mae": float(test_mae_last),
        "test_seq_rmse": float(test_rmse_seq),
        "test_seq_mae": float(test_mae_seq),
        "training_seconds": float(training_seconds),
        "num_train_samples": int(train_x.shape[0]),
        "num_val_samples": int(val_x.shape[0]),
        "num_test_samples": int(test_last_x.shape[0]),
        "num_train_runs": int(len(train_runs_x)),
        "num_calibration_runs": int(len(cal_runs_x)),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "hidden": int(hidden),
        "depth": int(depth),
        "dropout": float(dropout),
        "calibration_source_requested": str(args.calibration_source),
        "calibration_source_used": cal_mode_used,
        "calibration_fallback_reason": cal_fallback_reason,
        "calibration_fraction": float(args.calibration_fraction),
        "leakage_check_passed": bool(leakage_check_passed),
        "train_run_indices": train_run_indices,
        "calibration_run_indices": cal_run_indices,
        "compile_requested": bool(compile_requested),
        "compile_used": bool(compile_used),
        "compile_fallback_reason": compile_fallback_reason,
        "reuse_mode": bool(reuse_mode),
        "reuse_artifacts_root": str(Path(reuse_root_raw).resolve()) if reuse_mode else "",
        "amp_enabled": bool(use_amp),
        "low_rul_loss_weight_requested": float(low_rul_loss_weight),
        "low_rul_loss_weight_used": float(loss_weight_used),
        "low_rul_threshold": float(low_rul_threshold),
        "low_rul_weight_power": float(low_rul_weight_power),
        "low_rul_weight_datasets": low_weight_datasets_raw,
        "pvalue_safety_margin": float(args.pvalue_safety_margin),
        "calibration_bins": int(args.calibration_bins),
        "calibration_min_bin_size": int(args.calibration_min_bin_size),
        "history": history,
    }
    tem_metrics = {
        "dataset": key,
        "fd": int(args.fd),
        "fleet_summary": {
            "num_engines": num_eng,
            "num_tau_diagnostics_engines": num_tau,
            "num_alerted": int(fleet.get("num_alerted", 0)),
            "alert_rate": float(fleet.get("alert_rate", 0.0)),
            "mean_first_alert_step": float(fleet.get("mean_first_alert_step", -1.0)),
            "mean_temporal_rul_coverage": float(fleet.get("mean_temporal_rul_coverage", 0.0)),
            "tau_anytime_violation_rate": float(fleet.get("tau_anytime_violation_rate", 0.0)),
            "mean_temporal_tau_coverage": float(fleet.get("mean_temporal_tau_coverage", 0.0)),
        },
        "config": {
            "alpha": float(args.alpha),
            "lambda_bet": float(args.lambda_bet),
            "r_max": int(args.max_rul),
            "tau_max": int(tau_max),
            "calibration_bins": int(args.calibration_bins),
            "calibration_min_bin_size": int(args.calibration_min_bin_size),
            "pvalue_safety_margin": float(args.pvalue_safety_margin),
            "evidence_mode": "fixed_tau",
            "topology_level": "lite",
            "surface_topology_scope": "all",
            "window_size": int(window_size),
            "seed_offset": int(seed_offset),
        },
        "per_run": run_summaries,
    }
    _save_json_strict(train_metrics_path, train_metrics)
    _save_json_strict(tem_metrics_path, tem_metrics)

    row: dict[str, Any] = {
        "dataset": key,
        "status": "ok",
        "fd": int(args.fd),
        "num_runs": num_eng,
        "metrics": {
            # Use sequence-level metrics as primary external performance signal.
            "rmse": float(test_rmse_seq),
            "mae": float(test_mae_seq),
            "rmse_last": float(test_rmse_last),
            "mae_last": float(test_mae_last),
            "rul_cov": float(fleet.get("mean_temporal_rul_coverage", 0.0)),
            "tau_v": float(fleet.get("tau_anytime_violation_rate", 0.0)),
            "tau_identifiability_ratio": float(tau_ident_ratio),
        },
        "effective_settings": {
            "epochs": int(epochs),
            "lr": float(lr),
            "weight_decay": float(weight_decay),
            "hidden": int(hidden),
            "depth": int(depth),
            "dropout": float(dropout),
            "low_rul_loss_weight": float(loss_weight_used),
            "low_rul_threshold": float(low_rul_threshold),
            "low_rul_weight_power": float(low_rul_weight_power),
        },
        "artifacts": {
            "run_dir": str(ds_dir.resolve()),
            "checkpoint": str(ckpt_path.resolve()),
            "calibration_bundle": str(cal_bundle_path.resolve()),
            "train_metrics": str(train_metrics_path.resolve()),
            "tem_metrics": str(tem_metrics_path.resolve()),
        },
    }
    if cal_fallback_reason:
        row["notes"] = [cal_fallback_reason]
    if compile_fallback_reason:
        row.setdefault("notes", []).append(compile_fallback_reason)
    if reuse_mode:
        row.setdefault("notes", []).append("Policy replay mode: loaded checkpoint and calibration bundle from reuse_artifacts_root.")
    if apply_low_rul_weight:
        row.setdefault("notes", []).append(
            (
                "Applied low-RUL weighted training: "
                f"weight={float(loss_weight_used):.3f}, threshold={float(low_rul_threshold):.3f}, "
                f"power={float(low_rul_weight_power):.3f}."
            )
        )
    return row


def main() -> None:
    args = parse_args()
    configure_torch_fast_math()
    device = get_device()
    use_amp = not bool(args.no_amp)
    use_compile = not bool(args.no_compile)

    dataset_overrides: dict[str, dict[str, Any]] | None = None
    if str(args.dataset_overrides_json).strip():
        try:
            override_path = Path(str(args.dataset_overrides_json)).resolve()
            if override_path.exists():
                # utf-8-sig handles BOM if the file was written by Windows tools.
                raw = json.loads(override_path.read_text(encoding="utf-8-sig"))
                if isinstance(raw, dict):
                    dataset_overrides = {str(k).lower(): v for k, v in raw.items() if isinstance(v, dict)}
        except Exception:
            dataset_overrides = None

    data_root = Path(args.data_root).resolve()
    _set_data_root(data_root)
    _bind_reader_roots(data_root)

    dataset_list = [d.strip() for d in str(args.datasets).split(",") if d.strip()]
    out_root = ensure_dir(Path(args.out_root).resolve())

    rows: list[dict[str, Any]] = []
    for idx, dataset in enumerate(dataset_list):
        try:
            rows.append(
                _evaluate_one_dataset(
                    dataset=dataset,
                    dataset_index=idx,
                    args=args,
                    out_root=out_root,
                    device=device,
                    use_amp=use_amp,
                    use_compile=use_compile,
                    dataset_overrides=dataset_overrides,
                )
            )
        except Exception as err:
            rows.append(
                {
                    "dataset": dataset.lower().strip(),
                    "status": "error",
                    "error": f"{type(err).__name__}: {err}",
                    "traceback_tail": traceback.format_exc(limit=3),
                }
            )

    report = {
        "settings": {
            "data_root": str(data_root),
            "datasets": dataset_list,
            "fd": int(args.fd),
            "seed": int(args.seed),
            "seed_mode": "dataset_adler32_mod_10000",
            "max_rul": int(args.max_rul),
            "batch_size": int(args.batch_size),
            "num_workers": int(args.num_workers),
            "prefetch_factor": int(args.prefetch_factor),
            "epochs": int(args.epochs),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "hidden": int(args.hidden),
            "depth": int(args.depth),
            "dropout": float(args.dropout),
            "low_rul_loss_weight": float(args.low_rul_loss_weight),
            "low_rul_threshold": float(args.low_rul_threshold),
            "low_rul_weight_power": float(args.low_rul_weight_power),
            "low_rul_weight_datasets": [d.strip() for d in str(args.low_rul_weight_datasets).split(",") if d.strip()],
            "dataset_overrides_json": str(args.dataset_overrides_json),
            "dataset_overrides": dataset_overrides or {},
            "reuse_artifacts_root": str(args.reuse_artifacts_root),
            "alpha": float(args.alpha),
            "lambda_bet": float(args.lambda_bet),
            "calibration_source": str(args.calibration_source),
            "calibration_fraction": float(args.calibration_fraction),
            "calibration_bins": int(args.calibration_bins),
            "calibration_min_bin_size": int(args.calibration_min_bin_size),
            "pvalue_safety_margin": float(args.pvalue_safety_margin),
            "amp_enabled": bool(use_amp),
            "compile_requested": bool(use_compile),
        },
        "datasets": rows,
    }
    report = _sanitize_json(report)

    out_json = Path(args.out_json).resolve()
    _save_json_strict(out_json, report)
    out_md = Path(args.out_md).resolve()
    _write_md(out_md, report)

    num_ok = int(sum(1 for r in rows if str(r.get("status", "")).lower() == "ok"))
    print(f"Saved JSON: {out_json}")
    print(f"Saved Markdown: {out_md}")
    print(f"Datasets OK: {num_ok}/{len(rows)}")


if __name__ == "__main__":
    main()

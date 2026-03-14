from __future__ import annotations

import argparse
import json
import os
import sys
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tem.calibration import ConditionalResidualCalibrator  # noqa: E402
from tem.data import RULTensorDataset, build_calibration_bundle, flatten_runs, make_loader, predict_runs  # noqa: E402
from tem.evidence import TemConfig, infer_true_tau_from_true_rul, run_tem_single_engine  # noqa: E402
from tem.model import FastRULNet  # noqa: E402
from tem.train import train_model  # noqa: E402
from tem.utils import configure_torch_fast_math, get_device, seed_everything  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run repeated held-out external crossfit evaluation for small-run datasets. "
            "Each fold holds out one run for evaluation, one run for calibration, and trains on the rest."
        )
    )
    p.add_argument("--data-root", type=str, default="data/rul_datasets")
    p.add_argument("--datasets", type=str, default="femto,xjtu_sy")
    p.add_argument("--fd", type=int, default=1)
    p.add_argument("--max-rul", type=int, default=125)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--prefetch-factor", type=int, default=4)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--low-rul-loss-weight", type=float, default=1.0)
    p.add_argument("--low-rul-threshold", type=float, default=30.0)
    p.add_argument("--low-rul-weight-power", type=float, default=1.0)
    p.add_argument("--dataset-overrides-json", type=str, default="outputs/external_dataset_overrides.json")
    p.add_argument("--alpha", type=float, default=0.01)
    p.add_argument("--lambda-bet", type=float, default=0.1)
    p.add_argument("--calibration-bins", type=int, default=8)
    p.add_argument("--calibration-min-bin-size", type=int, default=128)
    p.add_argument("--pvalue-safety-margin", type=float, default=0.19)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--out-root", type=str, default="outputs/external_small_sample_crossfit")
    p.add_argument("--out-json", type=str, default="outputs/external_small_sample_crossfit/report.json")
    p.add_argument("--out-md", type=str, default="outputs/external_small_sample_crossfit/report.md")
    return p.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    # Some generated repo JSON files are BOM-prefixed; accept both plain UTF-8 and UTF-8-SIG.
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


def _set_data_root(root: Path) -> None:
    from rul_datasets.reader import CmapssReader, FemtoReader, NCmapssReader, XjtuSyReader
    from rul_datasets.reader.data_root import set_data_root

    root.mkdir(parents=True, exist_ok=True)
    os.environ["RUL_DATASETS_DATA_ROOT"] = str(root)
    set_data_root(str(root))
    CmapssReader._CMAPSS_ROOT = str(root / "CMAPSS")
    FemtoReader._FEMTO_ROOT = str(root / "FEMTOBearingDataSet")
    XjtuSyReader._XJTU_SY_ROOT = str(root / "XJTU-SY")
    NCmapssReader._NCMAPSS_ROOT = str(root / "NCMAPSS")


def _build_reader(dataset: str, fd: int, max_rul: int):
    from rul_datasets.reader import FemtoReader, XjtuSyReader

    key = dataset.lower().strip()
    if key == "femto":
        return FemtoReader(fd=fd, window_size=1024, max_rul=max_rul)
    if key in {"xjtu_sy", "xjtu-sy"}:
        return XjtuSyReader(fd=fd, window_size=4096, max_rul=max_rul)
    raise ValueError(f"Crossfit script currently supports only femto/xjtu_sy, got `{dataset}`.")


def _load_all_runs(reader) -> tuple[list[np.ndarray], list[np.ndarray], list[str]]:
    reader.prepare_data()
    all_x: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    split_tags: list[str] = []
    for split in ("dev", "val", "test"):
        feats, targs = reader.load_split(split)
        for x, y in zip(feats, targs):
            all_x.append(np.asarray(x, dtype=np.float32))
            all_y.append(np.asarray(y, dtype=np.float32))
            split_tags.append(split)
    return all_x, all_y, split_tags


def _parse_overrides(path_raw: str) -> dict[str, dict[str, Any]]:
    raw = str(path_raw).strip()
    if not raw:
        return {}
    path = Path(raw).resolve()
    if not path.exists():
        return {}
    obj = _load_json(path)
    if not isinstance(obj, dict):
        return {}
    return {str(k).lower(): v for k, v in obj.items() if isinstance(v, dict)}


def _ov(overrides: dict[str, Any], name: str, default: Any, cast) -> Any:
    if name in overrides:
        try:
            return cast(overrides[name])
        except Exception:
            return default
    return default


def _clip_rul(arr: np.ndarray, max_rul: int) -> np.ndarray:
    return np.clip(np.asarray(arr, dtype=np.float64), 1.0, float(max_rul))


def _fold_seed(base_seed: int, dataset_key: str, fold_idx: int) -> int:
    return int(base_seed) + int(zlib.adler32(dataset_key.encode("utf-8")) % 10000) + 97 * int(fold_idx)


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_sanitize(obj), indent=2, allow_nan=False), encoding="utf-8")


def _write_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _write_md(path: Path, report: dict[str, Any]) -> None:
    lines = ["# Small-Sample External Crossfit Report", ""]
    lines.append(
        "Repeated held-out evaluation across all available benchmark runs for small external datasets. "
        "Each fold uses one calibration run, one held-out test run, and the remaining runs for training."
    )
    lines.append("")
    for row in list(report.get("datasets", [])):
        if str(row.get("status", "")).lower() != "ok":
            lines.append(f"- {row.get('dataset', 'unknown')}: ERROR ({row.get('error', 'n/a')})")
            continue
        summ = row.get("summary", {}) if isinstance(row.get("summary", {}), dict) else {}
        lines.append(
            f"- {row['dataset']}: folds={int(summ.get('num_folds', 0))}, "
            f"seq_rmse={float(summ.get('rmse_mean', np.nan)):.3f}, "
            f"last_rmse={float(summ.get('rmse_last_mean', np.nan)):.3f}, "
            f"rul_cov={float(summ.get('rul_cov_mean', np.nan)):.3f}, "
            f"tau_v={float(summ.get('tau_v_mean', np.nan)):.3f}, "
            f"mean_width={float(summ.get('mean_width_mean', np.nan)):.3f}"
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_fold(
    *,
    dataset_key: str,
    fold_idx: int,
    holdout_idx: int,
    cal_idx: int,
    all_x: list[np.ndarray],
    all_y: list[np.ndarray],
    split_tags: list[str],
    args: argparse.Namespace,
    overrides: dict[str, Any],
    out_dir: Path,
    device: torch.device,
    use_amp: bool,
    use_compile: bool,
) -> dict[str, Any]:
    seed_everything(_fold_seed(args.seed, dataset_key, fold_idx))

    train_indices = [i for i in range(len(all_x)) if i not in {holdout_idx, cal_idx}]
    train_runs_x = [all_x[i] for i in train_indices]
    train_runs_y = [all_y[i] for i in train_indices]
    cal_runs_x = [all_x[cal_idx]]
    cal_runs_y = [all_y[cal_idx]]
    test_runs_x = [all_x[holdout_idx]]
    test_runs_y = [all_y[holdout_idx]]

    train_x, train_y = flatten_runs(train_runs_x, train_runs_y)
    val_x, val_y = flatten_runs(cal_runs_x, cal_runs_y)

    hidden = _ov(overrides, "hidden", int(args.hidden), int)
    depth = _ov(overrides, "depth", int(args.depth), int)
    dropout = _ov(overrides, "dropout", float(args.dropout), float)
    epochs = _ov(overrides, "epochs", int(args.epochs), int)
    lr = _ov(overrides, "lr", float(args.lr), float)
    weight_decay = _ov(overrides, "weight_decay", float(args.weight_decay), float)
    low_rul_loss_weight = _ov(overrides, "low_rul_loss_weight", float(args.low_rul_loss_weight), float)
    low_rul_threshold = _ov(overrides, "low_rul_threshold", float(args.low_rul_threshold), float)
    low_rul_weight_power = _ov(overrides, "low_rul_weight_power", float(args.low_rul_weight_power), float)

    train_ds = RULTensorDataset(train_x, train_y)
    val_ds = RULTensorDataset(val_x, val_y)
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

    model = FastRULNet(
        in_channels=int(train_x.shape[1]),
        hidden=int(hidden),
        depth=int(depth),
        dropout=float(dropout),
    )
    train_result = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=int(epochs),
        lr=float(lr),
        weight_decay=float(weight_decay),
        amp=use_amp,
        compile_model=use_compile,
        low_rul_loss_weight=float(low_rul_loss_weight),
        low_rul_threshold=float(low_rul_threshold),
        low_rul_weight_power=float(low_rul_weight_power),
    )
    model.load_state_dict(train_result.best_state_dict)
    model.to(device).eval()

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
    cal_bundle = build_calibration_bundle(pred_runs=cal_pred_runs, true_runs=cal_runs_y, healthy_rul_floor=1.0)
    cal_residuals = np.asarray(cal_bundle["residuals"], dtype=np.float32)
    cal_true = np.asarray(cal_bundle["true_rul"], dtype=np.float32)
    calibrator = ConditionalResidualCalibrator.from_arrays(
        residuals=cal_residuals,
        true_rul=cal_true,
        r_max=int(args.max_rul),
        n_bins=int(args.calibration_bins),
        min_bin_size=int(args.calibration_min_bin_size),
        pvalue_safety_margin=float(args.pvalue_safety_margin),
    )
    cfg = TemConfig(
        r_max=int(args.max_rul),
        alpha=float(args.alpha),
        lambda_bet=float(args.lambda_bet),
        evidence_mode="fixed_tau",
        compute_tau_diagnostics=True,
        use_conditional_calibration=True,
        calibration_bins=int(args.calibration_bins),
        calibration_min_bin_size=int(args.calibration_min_bin_size),
        pvalue_safety_margin=float(args.pvalue_safety_margin),
        cap_implied_rul=True,
    )

    pred_test_raw = predict_runs(
        model,
        test_runs_x,
        device=device,
        batch_size=int(args.batch_size),
        amp=use_amp,
        flatten_batch_runs=True,
    )[0]
    pred_test = _clip_rul(np.asarray(pred_test_raw, dtype=np.float64), max_rul=int(args.max_rul))
    true_test = np.asarray(test_runs_y[0], dtype=np.float64).reshape(-1)
    rmse = float(np.sqrt(np.mean((pred_test - true_test) ** 2)))
    mae = float(np.mean(np.abs(pred_test - true_test)))
    rmse_last = float(np.sqrt(np.mean((pred_test[-1:] - true_test[-1:]) ** 2)))
    mae_last = float(np.mean(np.abs(pred_test[-1:] - true_test[-1:])))
    true_tau = infer_true_tau_from_true_rul(true_test, r_max=int(args.max_rul))
    tau_max = int(true_test.shape[0] + int(args.max_rul) + 8)
    tem_res = run_tem_single_engine(
        pred_rul=pred_test,
        true_rul=true_test,
        calibration_residuals=cal_residuals,
        cfg=cfg,
        tau_max=tau_max,
        true_tau=true_tau,
        calibration_true_rul=cal_true,
        calibrator=calibrator,
        store_log_k_hist=False,
    )
    width_hist = np.asarray(tem_res.get("width_hist", []), dtype=np.float64)
    artifacts_npz = out_dir / f"fold_{fold_idx:02d}_artifacts.npz"
    _write_npz(
        artifacts_npz,
        calibration_residuals=np.asarray(cal_residuals, dtype=np.float32),
        calibration_true_rul=np.asarray(cal_true, dtype=np.float32),
        pred_test=np.asarray(pred_test, dtype=np.float32),
        true_test=np.asarray(true_test, dtype=np.float32),
    )

    fold = {
        "fold_index": int(fold_idx),
        "holdout_index": int(holdout_idx),
        "calibration_index": int(cal_idx),
        "train_indices": train_indices,
        "holdout_split": split_tags[holdout_idx],
        "calibration_split": split_tags[cal_idx],
        "num_train_runs": int(len(train_indices)),
        "num_steps": int(true_test.shape[0]),
        "rmse": rmse,
        "mae": mae,
        "rmse_last": rmse_last,
        "mae_last": mae_last,
        "rul_cov": float(np.mean(np.asarray(tem_res["true_r_in_set_hist"], dtype=np.float64))),
        "tau_v": float(bool(tem_res["tau_anytime_violation"])),
        "mean_width": float(np.mean(width_hist)) if width_hist.size else None,
        "true_tau": int(true_tau),
        "tau_max": int(tau_max),
        "tau_diagnostics_available": bool(tem_res.get("tau_diagnostics_available", False)),
        "train_seconds": float(train_result.total_seconds),
        "best_val_rmse": float(train_result.best_val_rmse),
        "artifacts_npz": str(artifacts_npz),
        "effective_settings": {
            "epochs": int(epochs),
            "lr": float(lr),
            "weight_decay": float(weight_decay),
            "hidden": int(hidden),
            "depth": int(depth),
            "dropout": float(dropout),
            "low_rul_loss_weight": float(low_rul_loss_weight),
            "low_rul_threshold": float(low_rul_threshold),
            "low_rul_weight_power": float(low_rul_weight_power),
        },
    }
    _write_json(out_dir / f"fold_{fold_idx:02d}.json", fold)
    return fold


def _summary_from_folds(folds: list[dict[str, Any]]) -> dict[str, Any]:
    arr = lambda key: np.asarray([float(f[key]) for f in folds if f.get(key) is not None], dtype=np.float64)
    rmse = arr("rmse")
    rmse_last = arr("rmse_last")
    cov = arr("rul_cov")
    tau = arr("tau_v")
    width = arr("mean_width")
    return {
        "num_folds": int(len(folds)),
        "rmse_mean": float(np.mean(rmse)) if rmse.size else None,
        "rmse_last_mean": float(np.mean(rmse_last)) if rmse_last.size else None,
        "rul_cov_mean": float(np.mean(cov)) if cov.size else None,
        "tau_v_mean": float(np.mean(tau)) if tau.size else None,
        "mean_width_mean": float(np.mean(width)) if width.size else None,
        "rmse_std": float(np.std(rmse)) if rmse.size else None,
        "rmse_last_std": float(np.std(rmse_last)) if rmse_last.size else None,
        "rul_cov_std": float(np.std(cov)) if cov.size else None,
        "tau_v_std": float(np.std(tau)) if tau.size else None,
        "mean_width_std": float(np.std(width)) if width.size else None,
    }


def main() -> None:
    args = parse_args()
    configure_torch_fast_math()
    device = get_device()
    use_amp = not bool(args.no_amp)
    use_compile = not bool(args.no_compile)

    data_root = Path(args.data_root).resolve()
    _set_data_root(data_root)
    overrides_all = _parse_overrides(args.dataset_overrides_json)

    datasets = [d.strip() for d in str(args.datasets).split(",") if d.strip()]
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        key = dataset.lower().strip()
        ds_out = out_root / key
        ds_out.mkdir(parents=True, exist_ok=True)
        try:
            reader = _build_reader(key, fd=int(args.fd), max_rul=int(args.max_rul))
            all_x, all_y, split_tags = _load_all_runs(reader)
            n_runs = len(all_x)
            if n_runs < 4:
                raise ValueError(f"{key}: need at least 4 runs for crossfit, got {n_runs}.")
            folds: list[dict[str, Any]] = []
            overrides = overrides_all.get(key, {})
            for holdout_idx in range(n_runs):
                cal_idx = (holdout_idx + 1) % n_runs
                fold = _run_fold(
                    dataset_key=key,
                    fold_idx=holdout_idx,
                    holdout_idx=holdout_idx,
                    cal_idx=cal_idx,
                    all_x=all_x,
                    all_y=all_y,
                    split_tags=split_tags,
                    args=args,
                    overrides=overrides,
                    out_dir=ds_out,
                    device=device,
                    use_amp=use_amp,
                    use_compile=use_compile,
                )
                folds.append(fold)
            row = {
                "dataset": key,
                "status": "ok",
                "fd": int(args.fd),
                "all_run_splits": split_tags,
                "summary": _summary_from_folds(folds),
                "folds": folds,
            }
        except Exception as err:
            row = {
                "dataset": key,
                "status": "error",
                "error": f"{type(err).__name__}: {err}",
            }
        rows.append(row)

    report = {
        "settings": {
            "data_root": str(data_root),
            "datasets": datasets,
            "fd": int(args.fd),
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
            "dataset_overrides_json": str(args.dataset_overrides_json),
            "dataset_overrides": overrides_all,
            "alpha": float(args.alpha),
            "lambda_bet": float(args.lambda_bet),
            "calibration_bins": int(args.calibration_bins),
            "calibration_min_bin_size": int(args.calibration_min_bin_size),
            "pvalue_safety_margin": float(args.pvalue_safety_margin),
            "amp_enabled": bool(use_amp),
            "compile_requested": bool(use_compile),
            "fold_scheme": "holdout=i, calibration=(i+1) mod n, train=rest",
        },
        "datasets": rows,
    }
    out_json = Path(args.out_json).resolve()
    _write_json(out_json, report)
    out_md = Path(args.out_md).resolve()
    _write_md(out_md, report)
    print(f"Saved JSON: {out_json}")
    print(f"Saved Markdown: {out_md}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

@dataclass
class CmapssSplits:
    dev_features: List[np.ndarray]
    dev_targets: List[np.ndarray]
    val_features: List[np.ndarray]
    val_targets: List[np.ndarray]
    test_last_features: List[np.ndarray]
    test_last_targets: List[np.ndarray]
    test_seq_features: List[np.ndarray]
    test_seq_targets: List[np.ndarray]


def _cache_path(
    data_root: str | Path,
    fd: int,
    window_size: int,
    max_rul: int,
    feature_select: list[int] | None,
) -> Path:
    root = Path(data_root).resolve()
    feat_tag = "all" if not feature_select else "-".join(str(x) for x in feature_select)
    name = f"tem_cache_fd{fd:03d}_w{window_size}_r{max_rul}_feat{feat_tag}.npz"
    return root / "CMAPSS" / name


def _to_object_array(seq: Sequence[np.ndarray]) -> np.ndarray:
    out = np.empty(len(seq), dtype=object)
    for i, arr in enumerate(seq):
        out[i] = np.asarray(arr)
    return out


def _serialize_splits(path: Path, splits: CmapssSplits) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        dev_features=_to_object_array(splits.dev_features),
        dev_targets=_to_object_array(splits.dev_targets),
        val_features=_to_object_array(splits.val_features),
        val_targets=_to_object_array(splits.val_targets),
        test_last_features=_to_object_array(splits.test_last_features),
        test_last_targets=_to_object_array(splits.test_last_targets),
        test_seq_features=_to_object_array(splits.test_seq_features),
        test_seq_targets=_to_object_array(splits.test_seq_targets),
    )


def _load_cached_splits(path: Path) -> CmapssSplits:
    blob = np.load(path, allow_pickle=True)

    def _list(key: str) -> List[np.ndarray]:
        arrs = list(blob[key])
        return [np.ascontiguousarray(np.asarray(x, dtype=np.float32)) for x in arrs]

    return CmapssSplits(
        dev_features=_list("dev_features"),
        dev_targets=_list("dev_targets"),
        val_features=_list("val_features"),
        val_targets=_list("val_targets"),
        test_last_features=_list("test_last_features"),
        test_last_targets=_list("test_last_targets"),
        test_seq_features=_list("test_seq_features"),
        test_seq_targets=_list("test_seq_targets"),
    )


def _import_rul_reader():
    from rul_datasets.reader import CmapssReader
    from rul_datasets.reader import cmapss as cmapss_module
    from rul_datasets.reader.data_root import set_data_root

    return CmapssReader, cmapss_module, set_data_root


def _prepare_data_root(data_root: str | Path) -> Path:
    _, cmapss_module, set_data_root = _import_rul_reader()
    root = Path(data_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    os.environ["RUL_DATASETS_DATA_ROOT"] = str(root)
    set_data_root(str(root))
    cmapss_module.CmapssReader._CMAPSS_ROOT = str(root / "CMAPSS")
    return root


def load_cmapss_splits(
    fd: int,
    data_root: str | Path,
    window_size: int,
    max_rul: int,
    feature_select: list[int] | None = None,
) -> CmapssSplits:
    cache = _cache_path(data_root, fd, window_size, max_rul, feature_select)
    if cache.exists():
        return _load_cached_splits(cache)

    _prepare_data_root(data_root)
    CmapssReader, _, _ = _import_rul_reader()
    reader = CmapssReader(
        fd=fd,
        window_size=window_size,
        max_rul=max_rul,
        feature_select=feature_select,
    )
    reader.prepare_data()

    dev_x, dev_y = reader.load_split("dev")
    val_x, val_y = reader.load_split("val")
    test_last_x, test_last_y = reader.load_split("test")
    # Load full test trajectories by aliasing test as dev preprocessing.
    test_seq_x, test_seq_y = reader.load_split("test", alias="dev")

    splits = CmapssSplits(
        dev_features=dev_x,
        dev_targets=dev_y,
        val_features=val_x,
        val_targets=val_y,
        test_last_features=test_last_x,
        test_last_targets=test_last_y,
        test_seq_features=test_seq_x,
        test_seq_targets=test_seq_y,
    )
    _serialize_splits(cache, splits)
    return splits


def flatten_runs(features: Sequence[np.ndarray], targets: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    x = np.concatenate(features, axis=0).astype(np.float32, copy=False)
    y = np.concatenate(targets, axis=0).astype(np.float32, copy=False)
    # Conv1D expects [N, C, T].
    x = np.ascontiguousarray(np.transpose(x, (0, 2, 1)))
    y = np.ascontiguousarray(y.reshape(-1))
    return x, y


class RULTensorDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, features: np.ndarray, targets: np.ndarray) -> None:
        self.features = torch.from_numpy(features)
        self.targets = torch.from_numpy(targets)

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.targets[index]


def make_loader(
    dataset: Dataset[Tuple[torch.Tensor, torch.Tensor]],
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    prefetch_factor: int,
) -> DataLoader[Tuple[torch.Tensor, torch.Tensor]]:
    kwargs: Dict[str, object] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
        "drop_last": False,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(dataset, **kwargs)


@torch.inference_mode()
def predict_numpy(
    model: torch.nn.Module,
    features_nct: np.ndarray,
    device: torch.device,
    batch_size: int = 4096,
    amp: bool = True,
) -> np.ndarray:
    model.eval()
    feats = torch.from_numpy(features_nct)
    preds: list[np.ndarray] = []
    use_amp = amp and device.type == "cuda"
    for i in range(0, feats.shape[0], batch_size):
        batch = feats[i : i + batch_size].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            out = model(batch).float().squeeze(-1)
        preds.append(out.detach().cpu().numpy())
    return np.concatenate(preds, axis=0)


@torch.inference_mode()
def predict_runs(
    model: torch.nn.Module,
    run_features: Sequence[np.ndarray],
    device: torch.device,
    batch_size: int = 4096,
    amp: bool = True,
    flatten_batch_runs: bool = True,
) -> List[np.ndarray]:
    if not run_features:
        return []

    if flatten_batch_runs:
        lengths = [int(run.shape[0]) for run in run_features]
        stacked_nct = np.concatenate(
            [
                np.ascontiguousarray(np.transpose(run.astype(np.float32, copy=False), (0, 2, 1)))
                for run in run_features
            ],
            axis=0,
        )
        flat_pred = predict_numpy(model, stacked_nct, device, batch_size=batch_size, amp=amp)
        splits = np.cumsum(lengths[:-1], dtype=np.int64)
        return [np.ascontiguousarray(x) for x in np.split(flat_pred, splits)]

    outputs: List[np.ndarray] = []
    for run in run_features:
        run_nct = np.ascontiguousarray(np.transpose(run.astype(np.float32, copy=False), (0, 2, 1)))
        outputs.append(predict_numpy(model, run_nct, device, batch_size=batch_size, amp=amp))
    return outputs


def build_calibration_residuals(
    pred_runs: Sequence[np.ndarray],
    true_runs: Sequence[np.ndarray],
    healthy_rul_floor: float,
) -> np.ndarray:
    pred = np.concatenate(pred_runs, axis=0)
    true = np.concatenate(true_runs, axis=0).astype(np.float32, copy=False)
    residuals = np.abs(pred - true)
    mask = true >= healthy_rul_floor
    if np.any(mask):
        return np.ascontiguousarray(residuals[mask].astype(np.float32, copy=False))
    return np.ascontiguousarray(residuals.astype(np.float32, copy=False))


def build_calibration_bundle(
    pred_runs: Sequence[np.ndarray],
    true_runs: Sequence[np.ndarray],
    healthy_rul_floor: float,
) -> dict[str, np.ndarray]:
    pred = np.concatenate(pred_runs, axis=0).astype(np.float32, copy=False)
    true = np.concatenate(true_runs, axis=0).astype(np.float32, copy=False)
    residuals = np.abs(pred - true).astype(np.float32, copy=False)
    mask = true >= healthy_rul_floor
    if np.any(mask):
        residuals = residuals[mask]
        true = true[mask]
    return {
        "residuals": np.ascontiguousarray(residuals),
        "true_rul": np.ascontiguousarray(true),
    }

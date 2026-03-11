from __future__ import annotations

import argparse
import json
import os
import traceback
import warnings
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize real datasets via rul_datasets readers.")
    p.add_argument("--data-root", type=str, default="data/rul_datasets")
    p.add_argument(
        "--datasets",
        type=str,
        default="cmapss,femto,xjtu_sy",
        help="Comma-separated: cmapss,femto,xjtu_sy,ncmapss",
    )
    p.add_argument("--fd", type=int, default=1, help="Sub-dataset id (FD) to summarize.")
    p.add_argument("--out-json", type=str, default="outputs/rul_dataset_summary.json")
    p.add_argument("--out-md", type=str, default="outputs/rul_dataset_summary.md")
    p.add_argument("--include-ncmapss", action="store_true", help="Convenience flag to append ncmapss.")
    p.add_argument(
        "--rebuild-scalers",
        action="store_true",
        help="Delete scaler cache .pkl files for the selected readers before summarization.",
    )
    p.add_argument(
        "--no-auto-rebuild-on-warning",
        action="store_true",
        help="Disable automatic scaler-cache rebuild when sklearn pickle-version warnings are detected.",
    )
    return p.parse_args()


def _set_data_root(root: Path) -> None:
    from rul_datasets.reader.data_root import set_data_root

    root.mkdir(parents=True, exist_ok=True)
    os.environ["RUL_DATASETS_DATA_ROOT"] = str(root)
    set_data_root(str(root))


def _bind_reader_roots(root: Path) -> None:
    # Reader classes cache their root paths at import time. Rebind to requested root
    # so summaries are reproducible and path provenance is explicit.
    from rul_datasets.reader import CmapssReader, FemtoReader, NCmapssReader, XjtuSyReader

    CmapssReader._CMAPSS_ROOT = str(root / "CMAPSS")
    FemtoReader._FEMTO_ROOT = str(root / "FEMTOBearingDataSet")
    XjtuSyReader._XJTU_SY_ROOT = str(root / "XJTU-SY")
    NCmapssReader._NCMAPSS_ROOT = str(root / "NCMAPSS")


def _split_stats(features: list[np.ndarray], targets: list[np.ndarray]) -> dict[str, Any]:
    if not features:
        return {
            "num_runs": 0,
            "num_windows_total": 0,
            "mean_windows_per_run": 0.0,
            "window_size_mean": 0.0,
            "num_channels_mean": 0.0,
            "target_min": float("nan"),
            "target_max": float("nan"),
            "target_mean": float("nan"),
        }
    run_windows = np.asarray([int(f.shape[0]) for f in features], dtype=np.float64)
    window_sizes = np.asarray([int(f.shape[1]) for f in features], dtype=np.float64)
    channels = np.asarray([int(f.shape[2]) for f in features], dtype=np.float64)
    tflat = np.concatenate([np.asarray(t, dtype=np.float64).reshape(-1) for t in targets], axis=0)
    return {
        "num_runs": int(len(features)),
        "num_windows_total": int(np.sum(run_windows)),
        "mean_windows_per_run": float(np.mean(run_windows)),
        "window_size_mean": float(np.mean(window_sizes)),
        "num_channels_mean": float(np.mean(channels)),
        "target_min": float(np.min(tflat)),
        "target_max": float(np.max(tflat)),
        "target_mean": float(np.mean(tflat)),
    }


def _summarize_reader_once(reader, name: str) -> dict[str, Any]:
    out: dict[str, Any] = {
        "dataset": name,
        "status": "ok",
        "reader_class": reader.__class__.__name__,
        "dataset_name": getattr(reader, "dataset_name", "unknown"),
        "fd": int(getattr(reader, "fd", -1)),
        "hparams": dict(getattr(reader, "hparams", {})),
        "splits": {},
    }
    reader.prepare_data()
    for split in ["dev", "val", "test"]:
        feats, targs = reader.load_split(split)
        out["splits"][split] = _split_stats(feats, targs)
    return out


def _format_warning(w: warnings.WarningMessage) -> dict[str, Any]:
    return {
        "category": getattr(w.category, "__name__", str(w.category)),
        "message": str(w.message),
        "filename": str(getattr(w, "filename", "")),
        "lineno": int(getattr(w, "lineno", -1)),
    }


def _is_sklearn_pickle_warning(w: warnings.WarningMessage) -> bool:
    category_name = getattr(w.category, "__name__", "")
    msg = str(w.message)
    return bool(
        category_name == "InconsistentVersionWarning"
        or ("Trying to unpickle estimator" in msg and "from version" in msg)
    )


def _safe_root(root: Any) -> Path | None:
    if not isinstance(root, str):
        return None
    root = root.strip()
    if not root:
        return None
    return Path(root)


def _collect_scaler_paths(reader, include_dataset_glob: bool = False) -> list[Path]:
    paths: set[Path] = set()

    for owner in (reader, getattr(reader, "_preparator", None)):
        getter = getattr(owner, "_get_scaler_path", None)
        if not callable(getter):
            continue
        try:
            p = Path(str(getter()))
        except Exception:
            continue
        if p.suffix.lower() == ".pkl":
            paths.add(p.resolve())

    if not include_dataset_glob:
        return sorted(paths)

    dataset_name = str(getattr(reader, "dataset_name", "")).lower()
    fd = int(getattr(reader, "fd", -1))

    if dataset_name == "cmapss":
        root = _safe_root(getattr(reader, "_CMAPSS_ROOT", None))
        if root is not None:
            for p in root.glob(f"FD{fd:03d}_scaler_*.pkl"):
                paths.add(p.resolve())
    elif dataset_name == "femto":
        root = _safe_root(getattr(reader, "_FEMTO_ROOT", None))
        if root is not None:
            for p in root.glob(f"scaler_{fd}_*.pkl"):
                paths.add(p.resolve())
    elif dataset_name in {"xjtu_sy", "xjtu-sy"}:
        root = _safe_root(getattr(reader, "_XJTU_SY_ROOT", None))
        if root is not None:
            for p in root.glob("**/scaler_*.pkl"):
                paths.add(p.resolve())
    elif dataset_name == "ncmapss":
        root = _safe_root(getattr(reader, "_NCMAPSS_ROOT", None))
        if root is not None:
            for p in root.glob(f"scaler_{fd}_*.pkl"):
                paths.add(p.resolve())

    return sorted(paths)


def _delete_scaler_paths(paths: list[Path]) -> list[str]:
    deleted: list[str] = []
    for p in paths:
        try:
            if p.exists():
                p.unlink()
                deleted.append(str(p))
        except OSError:
            continue
    return sorted(set(deleted))


def _run_summary_with_warnings(reader, name: str) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = _summarize_reader_once(reader, name)
    warning_rows = [_format_warning(w) for w in caught]
    pickle_warning_count = int(sum(1 for w in caught if _is_sklearn_pickle_warning(w)))
    return result, warning_rows, pickle_warning_count


def _summarize_reader(
    reader,
    name: str,
    rebuild_scalers: bool = False,
    auto_rebuild_on_warning: bool = True,
) -> dict[str, Any]:
    deleted_scalers: list[str] = []
    if rebuild_scalers:
        deleted_scalers.extend(_delete_scaler_paths(_collect_scaler_paths(reader, include_dataset_glob=True)))

    first_result, first_warnings, first_pickle_warning_count = _run_summary_with_warnings(reader, name)
    result = first_result
    final_warnings = first_warnings
    recovery_triggered = False
    final_pickle_warning_count = first_pickle_warning_count

    if first_pickle_warning_count > 0 and auto_rebuild_on_warning:
        recovery_triggered = True
        deleted_scalers.extend(_delete_scaler_paths(_collect_scaler_paths(reader, include_dataset_glob=True)))
        second_result, second_warnings, second_pickle_warning_count = _run_summary_with_warnings(reader, name)
        result = second_result
        final_warnings = second_warnings
        final_pickle_warning_count = second_pickle_warning_count
        result["warning_recovery"] = {
            "triggered": True,
            "pickle_warning_count_before_rebuild": int(first_pickle_warning_count),
            "pickle_warning_count_after_rebuild": int(second_pickle_warning_count),
        }
    elif first_pickle_warning_count > 0:
        result["warning_recovery"] = {
            "triggered": False,
            "pickle_warning_count": int(first_pickle_warning_count),
            "note": "Automatic rebuild disabled by --no-auto-rebuild-on-warning.",
        }

    result["scaler_rebuilt"] = bool(rebuild_scalers or recovery_triggered)
    if deleted_scalers:
        result["deleted_scaler_paths"] = sorted(set(deleted_scalers))
    if final_warnings:
        result["warnings"] = final_warnings
    if final_pickle_warning_count > 0:
        result["status"] = "warning"

    return result


def _build_reader(dataset: str, fd: int):
    from rul_datasets.reader import CmapssReader, FemtoReader, NCmapssReader, XjtuSyReader

    key = dataset.lower().strip()
    if key == "cmapss":
        return CmapssReader(fd=fd, window_size=30, max_rul=125)
    if key == "femto":
        return FemtoReader(fd=fd, window_size=1024, max_rul=125)
    if key in {"xjtu_sy", "xjtu-sy"}:
        return XjtuSyReader(fd=fd, window_size=4096, max_rul=125)
    if key == "ncmapss":
        # Keep memory bounded via temporal downsampling and explicit window size.
        return NCmapssReader(fd=fd, resolution_seconds=10, window_size=300, max_rul=65)
    raise ValueError(f"Unknown dataset key: {dataset}")


def _write_md(path: Path, obj: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# RUL Datasets Summary")
    lines.append("")
    lines.append(f"- Data root: `{obj['settings']['data_root']}`")
    lines.append(f"- Requested datasets: `{','.join(obj['settings']['datasets'])}`")
    lines.append(f"- FD: {obj['settings']['fd']}")
    lines.append(f"- Rebuild scalers requested: {obj['settings']['rebuild_scalers']}")
    lines.append(f"- Auto rebuild on warning: {obj['settings']['auto_rebuild_on_warning']}")
    lines.append("")
    lines.append("## Results")
    for item in obj["datasets"]:
        status = str(item.get("status", "unknown")).upper()
        lines.append(f"- {item['dataset']}: {status}")
        if item.get("status") != "ok":
            lines.append(f"  error: {item.get('error', 'n/a')}")
            continue
        for split, stats in item.get("splits", {}).items():
            lines.append(
                f"  {split}: runs={stats['num_runs']}, windows={stats['num_windows_total']}, "
                f"win/run={stats['mean_windows_per_run']:.1f}, "
                f"w={stats['window_size_mean']:.1f}, ch={stats['num_channels_mean']:.1f}, "
                f"target=[{stats['target_min']:.3f},{stats['target_max']:.3f}]"
            )
        if item.get("scaler_rebuilt"):
            lines.append("  scaler_cache_rebuilt: true")
        if item.get("deleted_scaler_paths"):
            lines.append(f"  deleted_scalers: {len(item['deleted_scaler_paths'])}")
        if item.get("warnings"):
            lines.append(f"  warnings: {len(item['warnings'])}")
        if item.get("warning_recovery"):
            wr = item["warning_recovery"]
            if bool(wr.get("triggered", False)):
                lines.append(
                    "  warning_recovery: triggered "
                    f"(before={wr.get('pickle_warning_count_before_rebuild', 0)}, "
                    f"after={wr.get('pickle_warning_count_after_rebuild', 0)})"
                )
            else:
                lines.append("  warning_recovery: not triggered")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    _set_data_root(data_root)
    _bind_reader_roots(data_root)

    datasets = [d.strip() for d in str(args.datasets).split(",") if d.strip()]
    if args.include_ncmapss and "ncmapss" not in {d.lower() for d in datasets}:
        datasets.append("ncmapss")
    auto_rebuild_on_warning = not bool(args.no_auto_rebuild_on_warning)

    results: list[dict[str, Any]] = []
    for d in datasets:
        try:
            reader = _build_reader(d, fd=int(args.fd))
            res = _summarize_reader(
                reader,
                name=d,
                rebuild_scalers=bool(args.rebuild_scalers),
                auto_rebuild_on_warning=auto_rebuild_on_warning,
            )
            results.append(res)
        except Exception as err:
            results.append(
                {
                    "dataset": d,
                    "status": "error",
                    "error": f"{type(err).__name__}: {err}",
                    "traceback_tail": traceback.format_exc(limit=3),
                }
            )

    from rul_datasets.reader import CmapssReader, FemtoReader, NCmapssReader, XjtuSyReader

    out = {
        "settings": {
            "data_root": str(data_root),
            "datasets": datasets,
            "fd": int(args.fd),
            "rebuild_scalers": bool(args.rebuild_scalers),
            "auto_rebuild_on_warning": auto_rebuild_on_warning,
            "reader_roots": {
                "cmapss": str(CmapssReader._CMAPSS_ROOT),
                "femto": str(FemtoReader._FEMTO_ROOT),
                "xjtu_sy": str(XjtuSyReader._XJTU_SY_ROOT),
                "ncmapss": str(NCmapssReader._NCMAPSS_ROOT),
            },
        },
        "datasets": results,
    }

    out_json = Path(args.out_json).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")

    out_md = Path(args.out_md).resolve()
    _write_md(out_md, out)

    print(f"Saved JSON: {out_json}")
    print(f"Saved Markdown: {out_md}")


if __name__ == "__main__":
    main()

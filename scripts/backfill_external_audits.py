from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rul_datasets.reader.data_root import set_data_root  # noqa: E402
from tem.calibration import ConditionalResidualCalibrator  # noqa: E402
from tem.data import predict_runs  # noqa: E402
from tem.evidence import TemConfig, infer_true_tau_from_true_rul, run_tem_single_engine, summarize_fleet_tem  # noqa: E402
from tem.marginal_topology import analyze_marginal_evidence_topology  # noqa: E402
from tem.model import FastRULNet  # noqa: E402
from tem.utils import get_device  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Backfill external audit_*.json and audit_cache_*.npz from existing external checkpoints."
    )
    p.add_argument("--external-performance-report", type=str, default="outputs/external_performance_report.json")
    p.add_argument("--data-root", type=str, default="", help="Optional override. Defaults to report settings.data_root.")
    p.add_argument("--healthy-rul-floor", type=float, default=100.0)
    p.add_argument("--predict-batch-size", type=int, default=128)
    p.add_argument("--out-json", type=str, default="outputs/external_audit_backfill_report.json")
    p.add_argument("--out-md", type=str, default="outputs/external_audit_backfill_report.md")
    return p.parse_args()


def _safe_float(x: Any) -> float | None:
    try:
        v = float(x)
    except Exception:
        return None
    if not np.isfinite(v):
        return None
    return v


def _empirical_cdf_checks(pvals: np.ndarray) -> dict[str, Any]:
    p = np.asarray(pvals, dtype=np.float64).reshape(-1)
    out: dict[str, Any] = {
        "n": int(p.size),
        "mean_p": float(np.mean(p)) if p.size else None,
        "frac_le_0.1": float(np.mean(p <= 0.1)) if p.size else None,
        "frac_le_0.2": float(np.mean(p <= 0.2)) if p.size else None,
        "frac_le_0.5": float(np.mean(p <= 0.5)) if p.size else None,
    }
    if p.size:
        out["superuniform_margin_0.1"] = 0.1 - float(out["frac_le_0.1"])
        out["superuniform_margin_0.2"] = 0.2 - float(out["frac_le_0.2"])
        out["superuniform_margin_0.5"] = 0.5 - float(out["frac_le_0.5"])
    else:
        out["superuniform_margin_0.1"] = None
        out["superuniform_margin_0.2"] = None
        out["superuniform_margin_0.5"] = None
    return out


def _set_data_root(root: Path) -> None:
    from rul_datasets.reader import CmapssReader, FemtoReader, NCmapssReader, XjtuSyReader

    root.mkdir(parents=True, exist_ok=True)
    os.environ["RUL_DATASETS_DATA_ROOT"] = str(root)
    set_data_root(str(root))
    CmapssReader._CMAPSS_ROOT = str(root / "CMAPSS")
    FemtoReader._FEMTO_ROOT = str(root / "FEMTOBearingDataSet")
    XjtuSyReader._XJTU_SY_ROOT = str(root / "XJTU-SY")
    NCmapssReader._NCMAPSS_ROOT = str(root / "NCMAPSS")


def _build_reader(dataset: str, fd: int, max_rul: int):
    from rul_datasets.reader import CmapssReader, FemtoReader, NCmapssReader, XjtuSyReader

    key = dataset.lower().strip()
    if key == "cmapss":
        return CmapssReader(fd=fd, window_size=30, max_rul=max_rul)
    if key == "femto":
        return FemtoReader(fd=fd, window_size=1024, max_rul=max_rul)
    if key in {"xjtu_sy", "xjtu-sy"}:
        return XjtuSyReader(fd=fd, window_size=4096, max_rul=max_rul)
    if key == "ncmapss":
        return NCmapssReader(fd=fd, resolution_seconds=10, window_size=300, max_rul=65)
    raise ValueError(f"Unknown dataset key: {dataset}")


def _load_test_runs(dataset: str, reader) -> tuple[list[np.ndarray], list[np.ndarray]]:
    key = dataset.lower().strip()
    reader.prepare_data()
    test_x, test_y = reader.load_split("test")
    if key == "cmapss":
        # Keep compatibility with C-MAPSS monitoring convention.
        test_x, test_y = reader.load_split("test", alias="dev")
    true_runs = [np.asarray(y, dtype=np.float64).reshape(-1) for y in test_y]
    return test_x, true_runs


def _load_model(checkpoint_path: Path, device: torch.device) -> tuple[FastRULNet, dict[str, Any]]:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
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
    return model, ckpt


def _save_cache(path: Path, pred_runs: list[np.ndarray], true_runs: list[np.ndarray]) -> None:
    run_lengths = np.asarray([int(len(x)) for x in true_runs], dtype=np.int64)
    pred_flat = np.concatenate([np.asarray(x, dtype=np.float64).reshape(-1) for x in pred_runs], axis=0)
    true_flat = np.concatenate([np.asarray(x, dtype=np.float64).reshape(-1) for x in true_runs], axis=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, pred_flat=pred_flat, true_flat=true_flat, run_lengths=run_lengths)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")


def _write_md(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = ["# External Audit Backfill Report", ""]
    for r in rows:
        if r.get("status") != "ok":
            lines.append(f"- {r.get('dataset', 'unknown')}: ERROR ({r.get('error', 'n/a')})")
            continue
        lines.append(
            f"- {r['dataset']}: OK | audit={r['audit_path']} | "
            f"healthy_mean_p={r['healthy_mean_p']:.4f} | tau_v={r['tau_anytime_violation_rate']:.4f}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    report_path = Path(args.external_performance_report).resolve()
    report = _load_json(report_path)
    settings = report.get("settings", {}) if isinstance(report.get("settings", {}), dict) else {}

    data_root_str = str(args.data_root).strip() or str(settings.get("data_root", "")).strip()
    if not data_root_str:
        raise ValueError("Could not infer data_root from args/report.")
    data_root = Path(data_root_str).resolve()
    _set_data_root(data_root)

    alpha = float(settings.get("alpha", 0.05))
    lambda_bet = float(settings.get("lambda_bet", 0.07))
    cal_bins = int(settings.get("calibration_bins", 8))
    cal_min_bin = int(settings.get("calibration_min_bin_size", 128))
    margin = float(settings.get("pvalue_safety_margin", 0.02))

    device = get_device()
    use_amp = bool(device.type == "cuda")

    rows: list[dict[str, Any]] = []
    ds_rows = report.get("datasets", [])
    if not isinstance(ds_rows, list):
        ds_rows = []

    for ds in ds_rows:
        try:
            if str(ds.get("status", "")).lower() != "ok":
                continue
            dataset = str(ds.get("dataset", "unknown")).lower()
            fd = int(ds.get("fd", settings.get("fd", 1)))
            artifacts = ds.get("artifacts", {}) if isinstance(ds.get("artifacts", {}), dict) else {}
            run_dir = Path(str(artifacts.get("run_dir", ""))).resolve()
            ckpt_path = Path(str(artifacts.get("checkpoint", ""))).resolve()
            cal_path = Path(str(artifacts.get("calibration_bundle", ""))).resolve()
            tem_path = Path(str(artifacts.get("tem_metrics", ""))).resolve()
            if not run_dir.exists():
                raise FileNotFoundError(f"run_dir missing: {run_dir}")
            if not ckpt_path.exists():
                raise FileNotFoundError(f"checkpoint missing: {ckpt_path}")
            if not cal_path.exists():
                raise FileNotFoundError(f"calibration missing: {cal_path}")
            if not tem_path.exists():
                raise FileNotFoundError(f"tem_metrics missing: {tem_path}")

            model, ckpt = _load_model(ckpt_path, device=device)
            max_rul = int(ckpt["max_rul"])
            reader = _build_reader(dataset, fd=fd, max_rul=max_rul)
            test_x, true_runs = _load_test_runs(dataset, reader)
            pred_runs_raw = predict_runs(
                model,
                test_x,
                device=device,
                batch_size=int(args.predict_batch_size),
                amp=use_amp,
                flatten_batch_runs=True,
            )
            pred_runs = [np.clip(np.asarray(p, dtype=np.float64), 1.0, float(max_rul)) for p in pred_runs_raw]

            cache_path = run_dir / f"audit_cache_fd{fd:03d}.npz"
            _save_cache(cache_path, pred_runs=pred_runs, true_runs=true_runs)

            cal = np.load(cal_path)
            cal_res = np.asarray(cal["residuals"], dtype=np.float64).reshape(-1)
            cal_true = np.asarray(cal["true_rul"], dtype=np.float64).reshape(-1) if "true_rul" in cal else None
            use_cond = cal_true is not None
            calibrator = ConditionalResidualCalibrator.from_arrays(
                residuals=cal_res,
                true_rul=cal_true if use_cond else None,
                r_max=max_rul,
                n_bins=cal_bins,
                min_bin_size=cal_min_bin,
                pvalue_safety_margin=margin,
            )

            all_pvals = []
            healthy_pvals = []
            for pred, true in zip(pred_runs, true_runs):
                scores = np.abs(np.asarray(pred, dtype=np.float64).reshape(-1) - np.asarray(true, dtype=np.float64).reshape(-1))
                implied = np.asarray(true, dtype=np.float64).reshape(-1) if use_cond else None
                pvals = calibrator.p_values(scores, implied_rul=implied)
                all_pvals.append(pvals)
                mask = np.asarray(true, dtype=np.float64).reshape(-1) >= float(args.healthy_rul_floor)
                if np.any(mask):
                    healthy_pvals.append(np.asarray(pvals, dtype=np.float64).reshape(-1)[mask])

            all_p = np.concatenate(all_pvals) if all_pvals else np.zeros(0, dtype=np.float64)
            healthy_p = np.concatenate(healthy_pvals) if healthy_pvals else np.zeros(0, dtype=np.float64)

            tem_obj = _load_json(tem_path)
            cfg_raw = tem_obj.get("config", {}) if isinstance(tem_obj.get("config", {}), dict) else {}
            cfg_raw = dict(cfg_raw)
            cfg_raw.setdefault("topology_level", "lite")
            cfg_raw.setdefault("surface_topology_scope", "all")
            tem_obj["config"] = cfg_raw
            cfg = TemConfig(
                r_max=max_rul,
                alpha=float(cfg_raw.get("alpha", alpha)),
                lambda_bet=float(cfg_raw.get("lambda_bet", lambda_bet)),
                evidence_mode=str(cfg_raw.get("evidence_mode", "fixed_tau")),
                gamma_crit=float(cfg_raw.get("gamma_crit", 1.5)),
                width_crit=int(cfg_raw.get("width_crit", 25)),
                min_persistence=float(cfg_raw.get("min_persistence", 0.5)),
                alert_patience=int(cfg_raw.get("alert_patience", 3)),
                cap_implied_rul=bool(cfg_raw.get("cap_implied_rul", True)),
                compute_tau_diagnostics=bool(cfg_raw.get("compute_tau_diagnostics", True)),
                use_conditional_calibration=bool(cfg_raw.get("use_conditional_calibration", use_cond)),
                calibration_bins=int(cfg_raw.get("calibration_bins", cal_bins)),
                calibration_min_bin_size=int(cfg_raw.get("calibration_min_bin_size", cal_min_bin)),
                pvalue_safety_margin=float(cfg_raw.get("pvalue_safety_margin", margin)),
            )

            fleet = tem_obj.get("fleet_summary", {})
            per_run = tem_obj.get("per_run", []) if isinstance(tem_obj.get("per_run", []), list) else []
            topology_missing = bool(per_run) and any(
                (not isinstance(r, dict)) or ("marginal_evidence_topology" not in r)
                for r in per_run
            )
            surface_missing = bool(per_run) and any(
                (not isinstance(r, dict))
                or (not isinstance(r.get("surface_topology"), dict))
                or (str(r.get("surface_topology", {}).get("backend", "")) == "skipped")
                for r in per_run
            )
            need_fleet = not (isinstance(fleet, dict) and bool(fleet))
            need_recompute = bool(need_fleet or topology_missing or surface_missing)

            run_results: list[dict[str, Any]] = []
            if need_recompute:
                max_len = max((len(x) for x in true_runs), default=1)
                tau_max = int(cfg_raw.get("tau_max", int(max_len + max_rul + 8)))
                for pred, true in zip(pred_runs, true_runs):
                    true_arr = np.asarray(true, dtype=np.float64).reshape(-1)
                    true_tau = infer_true_tau_from_true_rul(true_arr, r_max=max_rul)
                    run_results.append(
                        run_tem_single_engine(
                            pred_rul=np.asarray(pred, dtype=np.float64).reshape(-1),
                            true_rul=true_arr,
                            calibration_residuals=cal_res,
                            cfg=cfg,
                            tau_max=tau_max,
                            true_tau=true_tau,
                            calibration_true_rul=cal_true if use_cond else None,
                            calibrator=calibrator,
                            store_log_k_hist=bool(topology_missing or surface_missing),
                        )
                    )

            if need_fleet and run_results:
                fleet = summarize_fleet_tem(run_results)

            if (topology_missing or surface_missing) and run_results and len(per_run) == len(run_results):
                for i, (summary, res) in enumerate(zip(per_run, run_results)):
                    if not isinstance(summary, dict):
                        summary = {"run_index": int(i)}
                        per_run[i] = summary
                    topology = analyze_marginal_evidence_topology(
                        log_k_hist=np.asarray(res["log_k_hist"], dtype=np.float64),
                        min_persistence=float(cfg.min_persistence),
                        max_p_hist=np.asarray(res.get("max_p_hist", []), dtype=np.float64),
                        second_p_hist=np.asarray(res.get("second_p_hist", []), dtype=np.float64),
                        gamma_hist=np.asarray(res.get("gamma_hist", []), dtype=np.float64),
                        topology_level="lite",
                        compute_surface=True,
                    )
                    gamma_hist = np.asarray(res.get("gamma_hist", []), dtype=np.float64)
                    width_hist = np.asarray(res.get("width_hist", []), dtype=np.float64)
                    summary["mean_gamma"] = float(np.mean(gamma_hist)) if gamma_hist.size else None
                    summary["mean_width"] = float(np.mean(width_hist)) if width_hist.size else None
                    summary["marginal_evidence_topology"] = topology
                    summary["surface_topology"] = topology.get("surface")
                tem_obj["per_run"] = per_run
                tem_obj["fleet_summary"] = fleet
                _write_json(tem_path, tem_obj)

            audit_obj = {
                "fd": int(fd),
                "dataset": dataset,
                "checkpoint": str(ckpt_path),
                "calibration_file": str(cal_path),
                "checkpoint_leakage_flag": bool(ckpt.get("leakage_check_passed", True)),
                "checkpoint_leakage_check_passed": bool(ckpt.get("leakage_check_passed", True)),
                "checkpoint_leakage_detected": bool(not bool(ckpt.get("leakage_check_passed", True))),
                "calibration_size": int(cal_res.size),
                "pred_source": f"model_inference:{dataset}",
                "tem_fleet_source": f"tem_metrics:{tem_path}",
                "num_runs": int(len(true_runs)),
                "num_points": int(all_p.size),
                "use_conditional_calibration": bool(use_cond),
                "cap_implied_rul": True,
                "compute_tau_diagnostics": True,
                "predict_batch_size": int(args.predict_batch_size),
                "pvalue_safety_margin": float(margin),
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
            audit_path = run_dir / f"audit_fd{fd:03d}.json"
            _write_json(audit_path, audit_obj)

            rows.append(
                {
                    "dataset": dataset,
                    "status": "ok",
                    "fd": int(fd),
                    "run_dir": str(run_dir),
                    "audit_path": str(audit_path),
                    "cache_path": str(cache_path),
                    "healthy_mean_p": _safe_float(audit_obj["pvalue_healthy_prefix"].get("mean_p")),
                    "tau_anytime_violation_rate": _safe_float(fleet.get("tau_anytime_violation_rate")),
                    "topology_backfilled": bool(topology_missing),
                }
            )
        except Exception as err:
            rows.append(
                {
                    "dataset": str(ds.get("dataset", "unknown")).lower(),
                    "status": "error",
                    "error": f"{type(err).__name__}: {err}",
                }
            )

    out = {
        "inputs": {
            "external_performance_report": str(report_path),
            "data_root": str(data_root),
        },
        "settings_used": {
            "alpha": float(alpha),
            "lambda_bet": float(lambda_bet),
            "calibration_bins": int(cal_bins),
            "calibration_min_bin_size": int(cal_min_bin),
            "pvalue_safety_margin": float(margin),
            "healthy_rul_floor": float(args.healthy_rul_floor),
            "predict_batch_size": int(args.predict_batch_size),
        },
        "datasets": rows,
    }

    out_json = Path(args.out_json).resolve()
    out_md = Path(args.out_md).resolve()
    _write_json(out_json, out)
    _write_md(out_md, rows)
    print(f"Saved JSON: {out_json}")
    print(f"Saved Markdown: {out_md}")


if __name__ == "__main__":
    main()

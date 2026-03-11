from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from tem.calibration import ConditionalResidualCalibrator
from tem.topology import persistence_summary


@dataclass
class TemConfig:
    r_max: int = 125
    alpha: float = 0.05
    lambda_bet: float = 0.07
    gamma_crit: float = 1.5
    width_crit: int = 25
    min_persistence: float = 0.5
    alert_patience: int = 3
    cap_implied_rul: bool = True
    evidence_mode: str = "marginal_rul"  # "marginal_rul" or "fixed_tau"
    compute_tau_diagnostics: bool = True
    use_conditional_calibration: bool = True
    calibration_bins: int = 8
    calibration_min_bin_size: int = 128
    pvalue_safety_margin: float = 0.02


def conformal_p_values(scores: np.ndarray, sorted_calibration_residuals: np.ndarray) -> np.ndarray:
    n_cal = sorted_calibration_residuals.shape[0]
    idx = np.searchsorted(sorted_calibration_residuals, scores, side="left")
    count_ge = n_cal - idx
    return (1.0 + count_ge.astype(np.float64)) / (n_cal + 1.0)


def _implied_rul_from_tau(tau_grid: np.ndarray, t_one_based: int, cfg: TemConfig) -> np.ndarray:
    implied = tau_grid - (t_one_based - 1)
    if cfg.cap_implied_rul:
        implied = np.minimum(implied, float(cfg.r_max))
    return implied


def _alert_from_raw(raw_alert_hist: np.ndarray, patience: int) -> np.ndarray:
    raw = np.asarray(raw_alert_hist, dtype=bool).reshape(-1)
    if patience <= 1:
        return raw.copy()
    if raw.size == 0 or patience > raw.size:
        return np.zeros_like(raw)
    hits = np.convolve(raw.astype(np.int32), np.ones(int(patience), dtype=np.int32), mode="valid") == int(patience)
    alert = np.zeros_like(raw)
    alert[int(patience) - 1 :] = hits
    return alert


def _count_local_minima_1d(curve: np.ndarray) -> int:
    vals = np.asarray(curve, dtype=np.float64).reshape(-1)
    n = vals.shape[0]
    if n == 0:
        return 0
    if n == 1:
        return 1
    count = int(vals[0] < vals[1]) + int(vals[-1] < vals[-2])
    if n > 2:
        mid = vals[1:-1]
        left = vals[:-2]
        right = vals[2:]
        mask = (mid <= left) & (mid <= right) & ((mid < left) | (mid < right))
        count += int(np.sum(mask))
    return count


def infer_true_tau_from_true_rul(true_rul: np.ndarray, r_max: int) -> int | None:
    """
    Infer a fixed failure-time index tau from observed true RUL path.

    Returns:
      - integer tau when identifiable from at least one uncapped RUL point.
      - None when the path is fully capped at r_max and tau is not identifiable.
    """
    true = np.asarray(true_rul, dtype=np.float64).reshape(-1)
    if true.size == 0:
        return None
    t_idx = np.arange(1, true.shape[0] + 1, dtype=np.int64)
    tau_seq = t_idx + np.rint(true).astype(np.int64) - 1
    uncapped = true < (float(r_max) - 0.5)
    if np.any(uncapped):
        return int(np.max(tau_seq[uncapped]))
    return None


def run_tem_single_engine(
    pred_rul: np.ndarray,
    true_rul: np.ndarray,
    calibration_residuals: np.ndarray,
    cfg: TemConfig,
    tau_max: int | None = None,
    true_tau: int | None = None,
    calibration_true_rul: np.ndarray | None = None,
    calibrator: ConditionalResidualCalibrator | None = None,
    store_log_k_hist: bool = True,
) -> Dict[str, np.ndarray | int | float | bool | str | None]:
    pred = np.asarray(pred_rul, dtype=np.float64).reshape(-1)
    true = np.asarray(true_rul, dtype=np.float64).reshape(-1)
    cal_res = np.asarray(calibration_residuals, dtype=np.float64).reshape(-1)
    if cal_res.size == 0:
        raise ValueError("Calibration residuals must not be empty.")
    if not (0.0 <= cfg.lambda_bet < 1.0):
        raise ValueError("lambda_bet must satisfy 0 <= lambda_bet < 1 to keep betting factors nonnegative.")
    if cfg.alert_patience < 1:
        raise ValueError("alert_patience must be >= 1.")
    if cfg.evidence_mode not in {"marginal_rul", "fixed_tau"}:
        raise ValueError("evidence_mode must be 'marginal_rul' or 'fixed_tau'.")

    t_steps = pred.shape[0]
    tau_max = int(tau_max or 1000)
    tau_state_required = bool(cfg.evidence_mode == "fixed_tau")
    required_tau_max = t_steps + cfg.r_max - 1 if tau_state_required else 1
    if tau_max < required_tau_max:
        raise ValueError(
            f"tau_max must be >= {required_tau_max} for evidence_mode='{cfg.evidence_mode}'. "
            f"Received tau_max={tau_max}, t_steps={t_steps}, r_max={cfg.r_max}."
        )
    true_tau_known = true_tau is not None
    if true_tau_known and tau_state_required:
        true_tau = int(true_tau)
        if true_tau < 1 or true_tau > tau_max:
            raise ValueError("true_tau must satisfy 1 <= true_tau <= tau_max.")
    else:
        true_tau_known = False
        true_tau = -1

    if calibrator is None:
        calibrator = ConditionalResidualCalibrator.from_arrays(
            residuals=cal_res,
            true_rul=calibration_true_rul if cfg.use_conditional_calibration else None,
            r_max=cfg.r_max,
            n_bins=cfg.calibration_bins,
            min_bin_size=cfg.calibration_min_bin_size,
            pvalue_safety_margin=cfg.pvalue_safety_margin,
        )

    threshold = np.log(1.0 / cfg.alpha)
    rul_grid = np.arange(1, cfg.r_max + 1, dtype=np.float64)
    tau_grid = np.arange(1, tau_max + 1, dtype=np.float64) if tau_state_required else None

    log_k_hist = np.empty((t_steps, cfg.r_max), dtype=np.float64) if store_log_k_hist else None
    r_star_hist = np.empty(t_steps, dtype=np.int32)
    width_hist = np.empty(t_steps, dtype=np.int32)
    gamma_hist = np.empty(t_steps, dtype=np.float64)
    max_p_hist = np.empty(t_steps, dtype=np.float64)
    sec_p_hist = np.empty(t_steps, dtype=np.float64)
    local_minima_hist = np.empty(t_steps, dtype=np.int32)
    true_r_in_set_hist = np.empty(t_steps, dtype=bool)
    true_tau_in_set_hist = np.ones(t_steps, dtype=bool)
    raw_alert_hist = np.empty(t_steps, dtype=bool)

    tau_diagnostics_available = bool(cfg.compute_tau_diagnostics and true_tau_known)
    log_k_tau = np.zeros(tau_max, dtype=np.float64) if tau_state_required else None
    log_k_rul = np.zeros(cfg.r_max, dtype=np.float64) if not tau_state_required else None
    conditional_bins_active = bool(cfg.use_conditional_calibration and len(calibrator.per_bin_sorted_residuals) > 1)
    if tau_state_required and log_k_tau is None:
        raise RuntimeError("Internal error: tau recursion state missing.")
    if (not tau_state_required) and log_k_rul is None:
        raise RuntimeError("Internal error: RUL recursion state missing.")
    rul_bin_ids_const = None
    if conditional_bins_active and calibrator.rul_bin_lookup is not None:
        rul_bin_ids_const = calibrator.rul_bin_lookup[np.arange(cfg.r_max, dtype=np.int64)].astype(np.int64, copy=False)
    for t_idx in range(1, t_steps + 1):
        if tau_state_required:
            implied_tau = _implied_rul_from_tau(tau_grid, t_idx, cfg)
            scores_tau = np.abs(pred[t_idx - 1] - implied_tau)
            if conditional_bins_active and calibrator.rul_bin_lookup is not None:
                implied_tau_int = np.clip(implied_tau.astype(np.int64, copy=False), 1, cfg.r_max)
                tau_bin_ids = calibrator.rul_bin_lookup[implied_tau_int - 1].astype(np.int64, copy=False)
                pvals_tau = calibrator.p_values_with_bin_ids(scores_tau, tau_bin_ids)
            else:
                pvals_tau = calibrator.p_values(
                    scores_tau,
                    implied_rul=implied_tau if cfg.use_conditional_calibration else None,
                )
            factors_tau = 1.0 + cfg.lambda_bet * (1.0 - 2.0 * pvals_tau)
            factors_tau = np.clip(factors_tau, 1e-12, None)
            log_k_tau += np.log(factors_tau)
            if tau_diagnostics_available:
                true_tau_in_set_hist[t_idx - 1] = bool(log_k_tau[true_tau - 1] < threshold)

            # Current RUL hypotheses map to contiguous tau indices at time t.
            tau_curve = t_idx + np.arange(cfg.r_max, dtype=np.int64)
            curve = log_k_tau[tau_curve - 1]
        else:
            scores_r = np.abs(pred[t_idx - 1] - rul_grid)
            if rul_bin_ids_const is not None:
                pvals_r = calibrator.p_values_with_bin_ids(scores_r, rul_bin_ids_const)
            else:
                pvals_r = calibrator.p_values(
                    scores_r,
                    implied_rul=rul_grid if cfg.use_conditional_calibration else None,
                )
            factors_r = 1.0 + cfg.lambda_bet * (1.0 - 2.0 * pvals_r)
            factors_r = np.clip(factors_r, 1e-12, None)
            log_k_rul += np.log(factors_r)
            curve = log_k_rul

        if log_k_hist is not None:
            log_k_hist[t_idx - 1] = curve
        r_star = int(np.argmin(curve)) + 1
        conf_mask_r = curve < threshold
        width = int(np.sum(conf_mask_r))
        max_p, sec_p, gamma = persistence_summary(curve)
        local_minima = _count_local_minima_1d(curve)
        true_r = int(np.clip(round(true[t_idx - 1]), 1, cfg.r_max))
        true_r_in_set = bool(conf_mask_r[true_r - 1])
        raw_alert = bool(
            gamma > cfg.gamma_crit and width < cfg.width_crit and max_p >= cfg.min_persistence
        )

        r_star_hist[t_idx - 1] = r_star
        width_hist[t_idx - 1] = width
        gamma_hist[t_idx - 1] = gamma
        max_p_hist[t_idx - 1] = max_p
        sec_p_hist[t_idx - 1] = sec_p
        local_minima_hist[t_idx - 1] = local_minima
        true_r_in_set_hist[t_idx - 1] = true_r_in_set
        raw_alert_hist[t_idx - 1] = raw_alert

    alert_hist = _alert_from_raw(raw_alert_hist, cfg.alert_patience)
    alert_idx = np.where(alert_hist)[0]
    first_alert_step = int(alert_idx[0]) if alert_idx.size > 0 else -1
    tau_violation = bool(np.any(~true_tau_in_set_hist)) if tau_diagnostics_available else False

    return {
        "evidence_mode": cfg.evidence_mode,
        "tau_diagnostics_computed": tau_state_required,
        "tau_diagnostics_available": tau_diagnostics_available,
        "log_k_hist": log_k_hist,
        "r_star_hist": r_star_hist,
        "width_hist": width_hist,
        "gamma_hist": gamma_hist,
        "max_p_hist": max_p_hist,
        "second_p_hist": sec_p_hist,
        "local_minima_hist": local_minima_hist,
        "true_r_in_set_hist": true_r_in_set_hist,
        "true_tau_in_set_hist": true_tau_in_set_hist,
        "tau_anytime_violation": tau_violation,
        "raw_alert_hist": raw_alert_hist,
        "alert_hist": alert_hist,
        "first_alert_step": first_alert_step,
        "tau_max": tau_max,
        "true_tau": true_tau,
    }


def summarize_fleet_tem(results: list[Dict[str, np.ndarray | int | float | bool | str]]) -> dict:
    first_alerts = [int(r["first_alert_step"]) for r in results]
    valid_first = [x for x in first_alerts if x >= 0]
    rul_coverage_per_engine = [float(np.mean(np.asarray(r["true_r_in_set_hist"], dtype=np.float64))) for r in results]
    tau_coverage_per_engine = [
        float(np.mean(np.asarray(r["true_tau_in_set_hist"], dtype=np.float64)))
        for r in results
        if bool(r.get("tau_diagnostics_available", False))
    ]
    tau_violation_flags = [bool(r["tau_anytime_violation"]) for r in results if bool(r.get("tau_diagnostics_available", False))]
    tau_violation_rate = float(np.mean(tau_violation_flags)) if tau_violation_flags else 0.0
    return {
        "num_engines": len(results),
        "num_tau_diagnostics_engines": len(tau_coverage_per_engine),
        "num_alerted": len(valid_first),
        "alert_rate": len(valid_first) / max(1, len(results)),
        "mean_first_alert_step": float(np.mean(valid_first)) if valid_first else -1.0,
        "mean_temporal_rul_coverage": float(np.mean(rul_coverage_per_engine)) if rul_coverage_per_engine else 0.0,
        "mean_temporal_tau_coverage": float(np.mean(tau_coverage_per_engine)) if tau_coverage_per_engine else 0.0,
        "tau_anytime_violation_rate": tau_violation_rate,
    }

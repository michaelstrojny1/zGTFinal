from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tem.calibration import ConditionalResidualCalibrator  # noqa: E402
from tem.evidence import TemConfig, infer_true_tau_from_true_rul, run_tem_single_engine  # noqa: E402
from tem.utils import ensure_dir, save_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthetic validation for TEM theorems.")
    parser.add_argument("--n-engines", type=int, default=500)
    parser.add_argument("--life-min", type=int, default=180)
    parser.add_argument("--life-max", type=int, default=260)
    parser.add_argument("--r-max", type=int, default=125)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--lambda-bet", type=float, default=0.07)
    parser.add_argument("--evidence-mode", type=str, choices=["marginal_rul", "fixed_tau"], default="fixed_tau")
    parser.add_argument("--sigma", type=float, default=5.0)
    parser.add_argument("--onset-frac", type=float, default=0.5)
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--gamma-crit", type=float, default=1.5)
    parser.add_argument("--width-crit", type=int, default=25)
    parser.add_argument("--min-persistence", type=float, default=0.5)
    parser.add_argument("--alert-patience", type=int, default=3)
    parser.add_argument("--cap-rul-at-rmax", action="store_true", default=True)
    parser.add_argument("--no-cap-rul-at-rmax", action="store_true")
    parser.add_argument("--clip-predictions-to-rmax", action="store_true", default=True)
    parser.add_argument("--no-clip-predictions-to-rmax", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default="outputs/synthetic")
    return parser.parse_args()


def simulate_engine(
    rng: np.random.Generator,
    life: int,
    r_max: int,
    sigma: float,
    onset_frac: float,
    rho: float,
    cap_rul_at_rmax: bool,
    clip_predictions_to_rmax: bool,
) -> tuple[np.ndarray, np.ndarray]:
    raw_true_rul = np.arange(life, 0, -1, dtype=np.float64)
    true_rul = np.minimum(raw_true_rul, float(r_max)) if cap_rul_at_rmax else raw_true_rul
    onset = int(round(life * onset_frac))
    noise = rng.normal(0.0, sigma, size=life)
    drift = np.zeros(life, dtype=np.float64)
    if onset < life:
        drift[onset:] = rho * np.arange(1, life - onset + 1, dtype=np.float64)
    pred = true_rul + noise + drift
    if clip_predictions_to_rmax:
        pred = np.clip(pred, 1.0, float(r_max))
    return pred, true_rul


def oracle_superuniform_violation_prob(
    rng: np.random.Generator,
    n_engines: int,
    life_min: int,
    life_max: int,
    alpha: float,
    lambda_bet: float = 0.5,
) -> float:
    threshold = np.log(1.0 / alpha)
    violations = 0
    for _ in range(n_engines):
        life = int(rng.integers(life_min, life_max + 1))
        p_true = rng.uniform(0.0, 1.0, size=life)
        factors = 1.0 + lambda_bet * (1.0 - 2.0 * p_true)
        factors = np.clip(factors, 1e-12, None)
        logk = np.cumsum(np.log(factors))
        if np.any(logk >= threshold):
            violations += 1
    return violations / max(1, n_engines)


def evaluate_cohort(
    rng: np.random.Generator,
    n_engines: int,
    life_min: int,
    life_max: int,
    sigma: float,
    onset_frac: float,
    rho: float,
    r_max: int,
    cap_rul_at_rmax: bool,
    clip_predictions_to_rmax: bool,
    calibration: np.ndarray,
    cfg: TemConfig,
    tau_max: int,
) -> tuple[dict, np.ndarray]:
    tau_viol = 0
    rul_any_viol = 0
    alert_count = 0
    first_alerts: list[int] = []
    mean_rul_cov: list[float] = []
    mean_tau_cov: list[float] = []

    norm_time = np.linspace(0.0, 1.0, 200)
    interp_gammas = []
    calibrator = ConditionalResidualCalibrator.from_arrays(
        residuals=calibration,
        true_rul=None,
        r_max=cfg.r_max,
        n_bins=cfg.calibration_bins,
        min_bin_size=cfg.calibration_min_bin_size,
    )
    for _ in range(n_engines):
        life = int(rng.integers(life_min, life_max + 1))
        pred, true = simulate_engine(
            rng=rng,
            life=life,
            r_max=r_max,
            sigma=sigma,
            onset_frac=onset_frac,
            rho=rho,
            cap_rul_at_rmax=cap_rul_at_rmax,
            clip_predictions_to_rmax=clip_predictions_to_rmax,
        )
        res = run_tem_single_engine(
            pred,
            true,
            calibration,
            cfg,
            tau_max=tau_max,
            true_tau=infer_true_tau_from_true_rul(true, r_max=cfg.r_max),
            calibrator=calibrator,
        )
        rul_cov = float(np.mean(np.asarray(res["true_r_in_set_hist"], dtype=np.float64)))
        tau_cov = float(np.mean(np.asarray(res["true_tau_in_set_hist"], dtype=np.float64)))
        mean_rul_cov.append(rul_cov)
        mean_tau_cov.append(tau_cov)

        tau_viol += int(bool(res["tau_anytime_violation"]))
        rul_any_viol += int(np.any(~np.asarray(res["true_r_in_set_hist"], dtype=bool)))
        first_alert = int(res["first_alert_step"])
        if first_alert >= 0:
            alert_count += 1
            first_alerts.append(first_alert)

        gamma = np.asarray(res["gamma_hist"], dtype=np.float64)
        x_src = np.linspace(0.0, 1.0, gamma.shape[0])
        interp_gammas.append(np.interp(norm_time, x_src, gamma))

    summary = {
        "n_engines": n_engines,
        "rho": rho,
        "tau_anytime_violation_rate": tau_viol / max(1, n_engines),
        "dynamic_rul_any_violation_rate": rul_any_viol / max(1, n_engines),
        "mean_temporal_rul_coverage": float(np.mean(mean_rul_cov)) if mean_rul_cov else 0.0,
        "mean_temporal_tau_coverage": float(np.mean(mean_tau_cov)) if mean_tau_cov else 0.0,
        "alert_rate": alert_count / max(1, n_engines),
        "mean_first_alert_step": float(np.mean(first_alerts)) if first_alerts else -1.0,
    }
    gamma_curve = np.mean(np.stack(interp_gammas, axis=0), axis=0)
    return summary, gamma_curve


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)
    rng = np.random.default_rng(args.seed)
    cap_rul_at_rmax = args.cap_rul_at_rmax and not args.no_cap_rul_at_rmax
    clip_predictions_to_rmax = args.clip_predictions_to_rmax and not args.no_clip_predictions_to_rmax

    calibration = np.abs(rng.normal(0.0, args.sigma, size=4000)).astype(np.float32)
    cfg = TemConfig(
        r_max=args.r_max,
        alpha=args.alpha,
        lambda_bet=args.lambda_bet,
        evidence_mode=args.evidence_mode,
        gamma_crit=args.gamma_crit,
        width_crit=args.width_crit,
        min_persistence=args.min_persistence,
        alert_patience=args.alert_patience,
        cap_implied_rul=cap_rul_at_rmax,
    )
    tau_max = args.life_max + args.r_max
    null_summary, null_gamma = evaluate_cohort(
        rng=rng,
        n_engines=args.n_engines,
        life_min=args.life_min,
        life_max=args.life_max,
        sigma=args.sigma,
        onset_frac=args.onset_frac,
        rho=0.0,
        r_max=args.r_max,
        cap_rul_at_rmax=cap_rul_at_rmax,
        clip_predictions_to_rmax=clip_predictions_to_rmax,
        calibration=calibration,
        cfg=cfg,
        tau_max=tau_max,
    )
    degraded_summary, degraded_gamma = evaluate_cohort(
        rng=rng,
        n_engines=args.n_engines,
        life_min=args.life_min,
        life_max=args.life_max,
        sigma=args.sigma,
        onset_frac=args.onset_frac,
        rho=args.rho,
        r_max=args.r_max,
        cap_rul_at_rmax=cap_rul_at_rmax,
        clip_predictions_to_rmax=clip_predictions_to_rmax,
        calibration=calibration,
        cfg=cfg,
        tau_max=tau_max,
    )
    oracle_violation = oracle_superuniform_violation_prob(
        rng=rng,
        n_engines=args.n_engines,
        life_min=args.life_min,
        life_max=args.life_max,
        alpha=args.alpha,
        lambda_bet=args.lambda_bet,
    )
    norm_time = np.linspace(0.0, 1.0, 200)

    summary = {
        "n_engines": args.n_engines,
        "alpha": args.alpha,
        "config": {
            "sigma": args.sigma,
            "rho_degraded": args.rho,
            "onset_frac": args.onset_frac,
            "lambda_bet": args.lambda_bet,
            "evidence_mode": args.evidence_mode,
            "gamma_crit": args.gamma_crit,
            "width_crit": args.width_crit,
            "min_persistence": args.min_persistence,
            "alert_patience": args.alert_patience,
            "tau_max": tau_max,
            "cap_rul_at_rmax": cap_rul_at_rmax,
            "clip_predictions_to_rmax": clip_predictions_to_rmax,
        },
        "oracle_superuniform_violation_prob": oracle_violation,
        "null_cohort": null_summary,
        "degraded_cohort": degraded_summary,
    }
    save_json(summary, out_dir / "synthetic_summary.json")

    fig, ax = plt.subplots(figsize=(9, 4), dpi=140)
    ax.plot(norm_time, null_gamma, color="tab:gray", lw=1.8, label="mean gamma_t (null)")
    ax.plot(norm_time, degraded_gamma, color="tab:blue", lw=2.0, label="mean gamma_t (degraded)")
    ax.axvline(args.onset_frac, color="tab:red", linestyle="--", lw=1.5, label="degradation onset")
    ax.axhline(args.gamma_crit, color="tab:orange", linestyle=":", lw=1.5, label="gamma_crit")
    ax.set_xlabel("normalized life")
    ax.set_ylabel("gamma_t")
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "synthetic_gamma_phase_transition.png")
    plt.close(fig)

    print(
        f"Synthetic done | oracle_superuniform_violation={oracle_violation:.4f} (target <= {args.alpha:.4f}) | "
        f"null_tau_violation={null_summary['tau_anytime_violation_rate']:.4f} | "
        f"null_alert_rate={null_summary['alert_rate']:.3f} | "
        f"degraded_alert_rate={degraded_summary['alert_rate']:.3f}"
    )
    print(f"Saved: {out_dir / 'synthetic_summary.json'}")


if __name__ == "__main__":
    main()

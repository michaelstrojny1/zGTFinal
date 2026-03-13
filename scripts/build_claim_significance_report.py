from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build claim-level significance/effect report with multiplicity correction.")
    p.add_argument("--baseline-json", type=str, default="outputs/baseline_comparison.json")
    p.add_argument("--policy-sweep-json", type=str, default="outputs/external_policy_replay_sweep_retrain_v3/summary.json")
    p.add_argument("--out-json", type=str, default="outputs/publication_full_rtx4050/claim_significance_report.json")
    p.add_argument("--out-md", type=str, default="outputs/publication_full_rtx4050/claim_significance_report.md")
    p.add_argument("--bootstrap", type=int, default=4000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sanitize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        fv = float(obj)
        if not np.isfinite(fv):
            return None
        return fv
    return obj


def _fmt(v: Any, nd: int = 4) -> str:
    try:
        return f"{float(v):.{nd}f}"
    except Exception:
        return "n/a"


def _holm(pvals: list[float]) -> list[float]:
    n = len(pvals)
    order = np.argsort(np.asarray(pvals, dtype=np.float64))
    out = np.zeros(n, dtype=np.float64)
    prev = 0.0
    for rank, idx in enumerate(order, start=1):
        adj = (n - rank + 1) * float(pvals[idx])
        adj = max(prev, adj)
        out[idx] = min(1.0, adj)
        prev = out[idx]
    return out.tolist()


def _bh_fdr(pvals: list[float]) -> list[float]:
    n = len(pvals)
    order = np.argsort(np.asarray(pvals, dtype=np.float64))
    out = np.zeros(n, dtype=np.float64)
    nxt = 1.0
    for rank in range(n, 0, -1):
        idx = int(order[rank - 1])
        raw = float(pvals[idx]) * n / rank
        nxt = min(nxt, raw)
        out[idx] = min(1.0, nxt)
    return out.tolist()


def _bootstrap_ci_mean_diff(diffs: np.ndarray, n_boot: int, seed: int) -> dict[str, float]:
    x = np.asarray(diffs, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    rng = np.random.default_rng(int(seed))
    if x.size == 1:
        m = float(x[0])
        return {"mean": m, "ci_low": m, "ci_high": m}
    idx = rng.integers(0, int(x.size), size=(int(n_boot), int(x.size)), endpoint=False)
    means = np.mean(x[idx], axis=1)
    return {
        "mean": float(np.mean(x)),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
    }


def _baseline_effects_with_ci(rep: dict[str, Any], n_boot: int, seed: int) -> list[dict[str, Any]]:
    methods = list(rep.get("methods", []))
    main = next((m for m in methods if str(m.get("name")) == str(rep.get("main_method", "strict_main"))), None)
    if not main:
        return []
    main_by_fd = {int(r["fd"]): r for r in list(main.get("per_fd", []))}
    out: list[dict[str, Any]] = []
    for m in methods:
        if str(m.get("name")) == str(main.get("name")):
            continue
        comp_by_fd = {int(r["fd"]): r for r in list(m.get("per_fd", []))}
        fds = sorted(set(main_by_fd.keys()) & set(comp_by_fd.keys()))
        if not fds:
            continue
        dif_rmse = np.asarray([float(main_by_fd[fd]["rmse"]) - float(comp_by_fd[fd]["rmse"]) for fd in fds], dtype=np.float64)
        dif_cov = np.asarray([float(main_by_fd[fd]["rul_cov"]) - float(comp_by_fd[fd]["rul_cov"]) for fd in fds], dtype=np.float64)
        dif_tau = np.asarray([float(main_by_fd[fd]["tau_v"]) - float(comp_by_fd[fd]["tau_v"]) for fd in fds], dtype=np.float64)
        out.append(
            {
                "comparator": str(m.get("name")),
                "comparator_type": str(m.get("comparator_type", "")),
                "fds": fds,
                "rmse_diff_main_minus_comp": _bootstrap_ci_mean_diff(dif_rmse, n_boot=n_boot, seed=seed + 1),
                "coverage_diff_main_minus_comp": _bootstrap_ci_mean_diff(dif_cov, n_boot=n_boot, seed=seed + 2),
                "tau_diff_main_minus_comp": _bootstrap_ci_mean_diff(dif_tau, n_boot=n_boot, seed=seed + 3),
            }
        )
    return out


def _paired_tests_with_corrections(rep: dict[str, Any]) -> list[dict[str, Any]]:
    rows = list(rep.get("paired_results", []))
    tests: list[dict[str, Any]] = []
    for r in rows:
        comp = str(r.get("comparator", "unknown"))
        p_cov = r.get("coverage_sign_test_p")
        if p_cov is not None:
            tests.append(
                {
                    "comparator": comp,
                    "metric": "coverage",
                    "p_raw": float(p_cov),
                    "effect_mean_diff": float(r.get("coverage_diff_mean")),
                    "win_rate_main_better": float(r.get("coverage_win_rate")),
                    "n_pairs": int(r.get("n_coverage_pairs", 0)),
                }
            )
        p_tau = r.get("tau_violation_sign_test_p")
        if p_tau is not None:
            tests.append(
                {
                    "comparator": comp,
                    "metric": "tau_violation",
                    "p_raw": float(p_tau),
                    "effect_mean_diff": float(r.get("tau_violation_diff_mean")),
                    "win_rate_main_better": float(r.get("tau_violation_win_rate")),
                    "n_pairs": int(r.get("n_tau_pairs", 0)),
                }
            )
    if tests:
        pvals = [float(t["p_raw"]) for t in tests]
        holm = _holm(pvals)
        fdr = _bh_fdr(pvals)
        for t, ph, pf in zip(tests, holm, fdr):
            t["p_holm"] = float(ph)
            t["p_fdr_bh"] = float(pf)
            t["significant_holm_0p05"] = bool(ph <= 0.05)
    return tests


def _policy_margin_summary(sweep: dict[str, Any]) -> list[dict[str, Any]]:
    rows = list(sweep.get("rows_sorted", []))
    by_margin: dict[float, list[dict[str, Any]]] = {}
    for r in rows:
        m = float(r.get("pvalue_safety_margin"))
        by_margin.setdefault(m, []).append(r)
    out: list[dict[str, Any]] = []
    for m in sorted(by_margin.keys()):
        grp = by_margin[m]
        valid = [1.0 if bool(x.get("validity_ok")) else 0.0 for x in grp]
        cov = [float(x.get("cov_min", np.nan)) for x in grp]
        tau = [float(x.get("tau_max", np.nan)) for x in grp]
        out.append(
            {
                "margin": float(m),
                "num_points": int(len(grp)),
                "valid_fraction": float(np.mean(valid)) if valid else float("nan"),
                "cov_min_mean": float(np.nanmean(np.asarray(cov, dtype=np.float64))),
                "tau_max_mean": float(np.nanmean(np.asarray(tau, dtype=np.float64))),
            }
        )
    return out


def _write_md(path: Path, out: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# Claim Significance Report")
    lines.append("")
    lines.append(f"- baseline_json: `{out['inputs']['baseline_json']}`")
    lines.append(f"- policy_sweep_json: `{out['inputs']['policy_sweep_json']}`")
    lines.append("")
    lines.append("## Paired Tests")
    for t in out.get("paired_tests", []):
        lines.append(
            f"- {t['comparator']} [{t['metric']}]: p={_fmt(t['p_raw'])}, "
            f"p_holm={_fmt(t['p_holm'])}, effect={_fmt(t['effect_mean_diff'])}, "
            f"win_rate={_fmt(t['win_rate_main_better'])}, n={int(t['n_pairs'])}"
        )
    if not out.get("paired_tests"):
        lines.append("- no paired tests available")
    lines.append("")
    lines.append("## FD Bootstrap CI (Main - Comparator)")
    for e in out.get("fd_bootstrap_effects", []):
        c_cov = e["coverage_diff_main_minus_comp"]
        c_tau = e["tau_diff_main_minus_comp"]
        lines.append(
            f"- {e['comparator']}: "
            f"cov_diff={_fmt(c_cov['mean'])} [{_fmt(c_cov['ci_low'])},{_fmt(c_cov['ci_high'])}], "
            f"tau_diff={_fmt(c_tau['mean'])} [{_fmt(c_tau['ci_low'])},{_fmt(c_tau['ci_high'])}]"
        )
    lines.append("")
    lines.append("## Policy Margin Summary")
    for r in out.get("policy_margin_summary", []):
        lines.append(
            f"- margin={_fmt(r['margin'],3)}: valid_fraction={_fmt(r['valid_fraction'],3)}, "
            f"cov_min_mean={_fmt(r['cov_min_mean'],3)}, tau_max_mean={_fmt(r['tau_max_mean'],3)}"
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    baseline_path = Path(args.baseline_json).resolve()
    sweep_path = Path(args.policy_sweep_json).resolve()
    baseline = _load(baseline_path)
    sweep = _load(sweep_path)

    out = {
        "inputs": {
            "baseline_json": str(baseline_path),
            "policy_sweep_json": str(sweep_path),
            "bootstrap": int(args.bootstrap),
            "seed": int(args.seed),
        },
        "paired_tests": _paired_tests_with_corrections(baseline),
        "fd_bootstrap_effects": _baseline_effects_with_ci(baseline, n_boot=int(args.bootstrap), seed=int(args.seed)),
        "policy_margin_summary": _policy_margin_summary(sweep),
        "policy_best": dict(sweep.get("best_policy", {})),
    }
    out = _sanitize(out)

    out_json = Path(args.out_json).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2, allow_nan=False), encoding="utf-8")
    out_md = Path(args.out_md).resolve()
    _write_md(out_md, out)
    print(f"Saved JSON: {out_json}")
    print(f"Saved Markdown: {out_md}")


if __name__ == "__main__":
    main()

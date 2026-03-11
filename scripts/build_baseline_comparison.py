from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build baseline comparison artifact from matrix outputs.")
    p.add_argument("--matrix-report", type=str, default="outputs/publication_full_rtx4050_report.json")
    p.add_argument(
        "--external-baselines-json",
        type=str,
        default="",
        help=(
            "Optional external baseline package JSON with schema "
            "{'methods':[{'name','comparator_type':'external','per_fd':[{'fd','rmse','rul_cov','tau_v','run_dir?'}]}]}."
        ),
    )
    p.add_argument("--out-json", type=str, default="outputs/baseline_comparison.json")
    p.add_argument("--out-md", type=str, default="outputs/baseline_comparison.md")
    return p.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _tem_path(run_dir: str | Path, fd: int) -> Path:
    return Path(run_dir) / f"tem_metrics_fd{int(fd):03d}.json"


def _sign_test_two_sided(diffs: np.ndarray) -> float:
    vals = np.asarray(diffs, dtype=np.float64).reshape(-1)
    vals = vals[np.isfinite(vals)]
    vals = vals[vals != 0.0]
    n = int(vals.size)
    if n == 0:
        return float("nan")
    k = int(np.sum(vals > 0.0))
    lo = min(k, n - k)
    num = 0
    for i in range(0, lo + 1):
        num += math.comb(n, i)
    p = 2.0 * num / (2.0**n)
    return float(min(1.0, p))


def _safe_float(v: float | np.floating | None) -> float | None:
    if v is None:
        return None
    fv = float(v)
    if not np.isfinite(fv):
        return None
    return fv


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


def _fmt(v: float | None, digits: int = 4) -> str:
    if v is None:
        return "n/a"
    return f"{float(v):.{digits}f}"


def _method_rows_by_fd(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for r in rows:
        out[int(r["fd"])] = r
    return out


def _find_row(rows: list[dict[str, Any]], fd: int, pred) -> dict[str, Any]:
    cands = [r for r in rows if int(r["fd"]) == int(fd) and pred(r)]
    if not cands:
        raise ValueError(f"Missing comparator row for FD{fd:03d}")
    if len(cands) > 1:
        # Deterministic pick by run-dir path to keep reproducibility.
        cands = sorted(cands, key=lambda x: str(x["run_dir"]))
    return cands[0]


def _paired_metric_diffs(main_t: dict[str, Any], comp_t: dict[str, Any], key: str) -> np.ndarray:
    a = np.asarray([r.get(key) for r in main_t["per_run"]], dtype=object)
    b = np.asarray([r.get(key) for r in comp_t["per_run"]], dtype=object)
    if a.shape[0] != b.shape[0]:
        raise ValueError(f"per_run length mismatch for key={key}: {a.shape[0]} vs {b.shape[0]}")
    diffs: list[float] = []
    for x, y in zip(a, b):
        if x is None or y is None:
            continue
        fx = float(x)
        fy = float(y)
        if np.isfinite(fx) and np.isfinite(fy):
            diffs.append(fx - fy)
    return np.asarray(diffs, dtype=np.float64)


def _summ(v: list[float]) -> dict[str, float]:
    arr = np.asarray(v, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _write_md(path: Path, rep: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# Baseline Comparison")
    lines.append("")
    lines.append(f"- Matrix report: `{rep['inputs']['matrix_report']}`")
    lines.append(f"- Main method: `{rep['main_method']}`")
    lines.append(f"- Number of FDs: {len(rep['fds'])}")
    lines.append(
        f"- Comparator mix: external={rep['comparator_summary']['num_external']}, "
        f"internal={rep['comparator_summary']['num_internal']}"
    )
    lines.append("")
    lines.append("## Aggregate")
    for m in rep["methods"]:
        s = m["summary"]
        lines.append(
            f"- {m['name']} ({m['comparator_type']}): rmse={s['rmse_mean']:.3f}, rul_cov={s['rul_cov_mean']:.3f}, "
            f"tau_v={s['tau_v_mean']:.3f}"
        )
    lines.append("")
    lines.append("## Paired Run-Level Stats")
    for r in rep["paired_results"]:
        lines.append(f"- {r['comparator']} vs {rep['main_method']}:")
        if int(r["n_coverage_pairs"]) > 0:
            lines.append(
                f"  coverage diff mean={_fmt(r['coverage_diff_mean'], 4)}, "
                f"win_rate={_fmt(r['coverage_win_rate'], 3)}, "
                f"p_sign={_fmt(r['coverage_sign_test_p'], 4)}"
            )
        else:
            lines.append("  coverage diff: N/A (comparator lacks per-run TEM artifacts)")
        if int(r["n_tau_pairs"]) > 0:
            lines.append(
                f"  tau-violation diff mean={_fmt(r['tau_violation_diff_mean'], 4)}, "
                f"win_rate={_fmt(r['tau_violation_win_rate'], 3)}, p_sign={_fmt(r['tau_violation_sign_test_p'], 4)}"
            )
        else:
            lines.append("  tau-violation diff: N/A (comparator lacks tau diagnostics)")
        if r.get("pairing_note"):
            lines.append(f"  note: {r['pairing_note']}")
    lines.append("")
    lines.append("## Notes")
    lines.append("- Comparators are internal alternatives from the same codebase, not external published baselines.")
    lines.append("- Use this as a rigorous ablation/comparator package; add external methods for final submission strength.")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_external_methods(path: Path, fds: list[int]) -> tuple[list[dict[str, Any]], list[str]]:
    blob = _load_json(path)
    raw_methods = list(blob.get("methods", []))
    methods: list[dict[str, Any]] = []
    notes: list[str] = []
    for raw in raw_methods:
        name = str(raw.get("name", "")).strip()
        if not name:
            continue
        ctype = str(raw.get("comparator_type", "external")).strip().lower() or "external"
        per_fd = list(raw.get("per_fd", []))
        by_fd: dict[int, dict[str, Any]] = {}
        for row in per_fd:
            try:
                fd = int(row["fd"])
            except Exception:
                continue
            by_fd[fd] = row

        missing = [fd for fd in fds if fd not in by_fd]
        if missing:
            notes.append(f"Skipped external method `{name}` due to missing FD rows: {missing}")
            continue

        rows_by_fd: dict[int, dict[str, Any]] = {}
        try:
            for fd in fds:
                row = by_fd[fd]
                rows_by_fd[fd] = {
                    "fd": int(fd),
                    "run_dir": str(row.get("run_dir", "")),
                    "rmse": float(row["rmse"]),
                    "rul_cov": float(row["rul_cov"]),
                    "tau_v": float(row["tau_v"]),
                    "source": str(path),
                }
        except Exception as err:
            notes.append(f"Skipped external method `{name}` due to parse error: {type(err).__name__}: {err}")
            continue

        methods.append(
            {
                "name": name,
                "comparator_type": ctype,
                "rows_by_fd": rows_by_fd,
                "metadata": {
                    "description": str(raw.get("description", "")),
                    "source": str(path),
                },
            }
        )
    return methods, notes


def main() -> None:
    args = parse_args()
    matrix_path = Path(args.matrix_report).resolve()
    matrix = _load_json(matrix_path)

    baseline_rows = list(matrix.get("baseline", []))
    evidence_rows = list(matrix.get("evidence_mode", []))
    alpha_rows = list(matrix.get("alpha_sweep", []))
    policy_rows = list(matrix.get("policy_sweep", []))
    if not baseline_rows:
        raise ValueError("Matrix report has no baseline rows.")

    fds = sorted({int(r["fd"]) for r in baseline_rows})
    base_by_fd = _method_rows_by_fd(baseline_rows)

    methods: list[dict[str, Any]] = [
        {
            "name": "strict_main",
            "comparator_type": "main",
            "rows_by_fd": {fd: base_by_fd[fd] for fd in fds},
        },
        {
            "name": "marginal_rul",
            "comparator_type": "internal",
            "rows_by_fd": {
                fd: _find_row(evidence_rows, fd, lambda r: "\\marginal_rul\\" in str(r["run_dir"]).replace("/", "\\").lower())
                for fd in fds
            },
        },
        {
            "name": "alpha_0p1",
            "comparator_type": "internal",
            "rows_by_fd": {
                fd: _find_row(alpha_rows, fd, lambda r: "\\a0p1\\" in str(r["run_dir"]).replace("/", "\\").lower()) for fd in fds
            },
        },
        {
            "name": "no_margin_matched_bins",
            "comparator_type": "internal",
            "rows_by_fd": {
                fd: _find_row(
                    policy_rows,
                    fd,
                    lambda r: (
                        ("\\m0p0_" in str(r["run_dir"]).replace("/", "\\").lower())
                        and (
                            f"_b{int(_load_json(_tem_path(base_by_fd[fd]['run_dir'], fd))['config']['calibration_bins'])}_"
                            in str(r["run_dir"]).replace("/", "\\").lower()
                        )
                        and (
                            f"_mb{int(_load_json(_tem_path(base_by_fd[fd]['run_dir'], fd))['config']['calibration_min_bin_size'])}\\"
                            in (str(r["run_dir"]).replace("/", "\\").lower() + "\\")
                        )
                    ),
                )
                for fd in fds
            },
        },
    ]
    notes: list[str] = []
    if args.external_baselines_json:
        ext_path = Path(args.external_baselines_json).resolve()
        if ext_path.exists():
            ext_methods, ext_notes = _load_external_methods(ext_path, fds)
            methods.extend(ext_methods)
            notes.extend(ext_notes)
        else:
            notes.append(f"external-baselines-json not found: {ext_path}")

    # Preload tem metrics for paired run-level tests.
    tem_cache: dict[tuple[str, int], dict[str, Any]] = {}
    for m in methods:
        for fd in fds:
            row = m["rows_by_fd"][fd]
            run_dir = str(row.get("run_dir", "")).strip()
            if not run_dir:
                continue
            tpath = _tem_path(run_dir, fd)
            if tpath.exists():
                tem_cache[(m["name"], fd)] = _load_json(tpath)

    method_summaries: list[dict[str, Any]] = []
    for m in methods:
        rmse = [float(m["rows_by_fd"][fd]["rmse"]) for fd in fds]
        cov = [float(m["rows_by_fd"][fd]["rul_cov"]) for fd in fds]
        tau = [float(m["rows_by_fd"][fd]["tau_v"]) for fd in fds]
        method_summaries.append(
            {
                "name": m["name"],
                "comparator_type": m["comparator_type"],
                "fds": fds,
                "summary": {
                    "rmse_mean": float(np.mean(rmse)),
                    "rmse_std": float(np.std(np.asarray(rmse), ddof=1)) if len(rmse) > 1 else 0.0,
                    "rul_cov_mean": float(np.mean(cov)),
                    "rul_cov_std": float(np.std(np.asarray(cov), ddof=1)) if len(cov) > 1 else 0.0,
                    "tau_v_mean": float(np.mean(tau)),
                    "tau_v_std": float(np.std(np.asarray(tau), ddof=1)) if len(tau) > 1 else 0.0,
                },
                "per_fd": [
                    {
                        "fd": int(fd),
                        "run_dir": str(m["rows_by_fd"][fd]["run_dir"]),
                        "rmse": float(m["rows_by_fd"][fd]["rmse"]),
                        "rul_cov": float(m["rows_by_fd"][fd]["rul_cov"]),
                        "tau_v": float(m["rows_by_fd"][fd]["tau_v"]),
                    }
                    for fd in fds
                ],
            }
        )

    main_name = "strict_main"
    paired_results: list[dict[str, Any]] = []
    for m in methods:
        if m["name"] == main_name:
            continue
        cov_diffs_all: list[float] = []
        tau_diffs_all: list[float] = []
        pairing_note = ""
        can_pair = all((main_name, fd) in tem_cache and (m["name"], fd) in tem_cache for fd in fds)
        if not can_pair:
            pairing_note = "No complete per-run TEM artifacts across all FD for this comparator."
        for fd in fds:
            if (main_name, fd) not in tem_cache or (m["name"], fd) not in tem_cache:
                continue
            main_t = tem_cache[(main_name, fd)]
            comp_t = tem_cache[(m["name"], fd)]
            cov_diffs = _paired_metric_diffs(main_t, comp_t, "temporal_rul_coverage")
            tau_diffs = _paired_metric_diffs(main_t, comp_t, "tau_anytime_violation")
            if cov_diffs.size:
                cov_diffs_all.extend(cov_diffs.tolist())
            if tau_diffs.size:
                tau_diffs_all.extend(tau_diffs.tolist())
        cov_arr = np.asarray(cov_diffs_all, dtype=np.float64)
        tau_arr = np.asarray(tau_diffs_all, dtype=np.float64)
        tau_diff_mean = _safe_float(float(np.mean(tau_arr)) if tau_arr.size else None)
        tau_win_rate = _safe_float(float(np.mean(tau_arr < 0.0)) if tau_arr.size else None)
        tau_sign_p = _safe_float(_sign_test_two_sided(-tau_arr)) if tau_arr.size else None
        paired_results.append(
            {
                "comparator": m["name"],
                "comparator_type": m["comparator_type"],
                "n_coverage_pairs": int(cov_arr.size),
                "coverage_diff_mean": _safe_float(float(np.mean(cov_arr)) if cov_arr.size else None),
                "coverage_win_rate": _safe_float(float(np.mean(cov_arr > 0.0)) if cov_arr.size else None),
                "coverage_sign_test_p": _safe_float(_sign_test_two_sided(cov_arr)),
                "n_tau_pairs": int(tau_arr.size),
                "tau_violation_diff_mean": tau_diff_mean,
                # For tau violation, negative diff means main has fewer violations (better).
                "tau_violation_win_rate": tau_win_rate,
                "tau_violation_sign_test_p": tau_sign_p,
                "pairing_note": pairing_note,
            }
        )

    num_external = int(sum(1 for m in method_summaries if m.get("comparator_type") == "external"))
    num_internal = int(sum(1 for m in method_summaries if m.get("comparator_type") == "internal"))
    out = {
        "inputs": {"matrix_report": str(matrix_path)},
        "fds": fds,
        "main_method": main_name,
        "comparator_summary": {
            "num_external": num_external,
            "num_internal": num_internal,
            "has_external": bool(num_external >= 1),
            "num_methods_total": int(len(method_summaries)),
        },
        "methods": method_summaries,
        "paired_results": paired_results,
        "notes": [
            "Comparators are internal alternatives (ablation-grade), not external published baselines.",
            "This artifact is intended to make comparative evidence explicit and reproducible.",
            *notes,
        ],
    }
    out = _sanitize_json(out)

    out_json = Path(args.out_json).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2, allow_nan=False), encoding="utf-8")

    out_md = Path(args.out_md).resolve()
    _write_md(out_md, out)

    print(f"Saved JSON: {out_json}")
    print(f"Saved Markdown: {out_md}")


if __name__ == "__main__":
    main()

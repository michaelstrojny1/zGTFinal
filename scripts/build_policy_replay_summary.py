from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize canonical vs policy-replay validity metrics.")
    p.add_argument("--canonical", type=str, default="outputs/external_performance_report.json")
    p.add_argument("--balanced", type=str, default="outputs/external_performance_report_policy_replay_balanced_v2.json")
    p.add_argument("--aggressive", type=str, default="outputs/external_performance_report_policy_replay_aggressive_v1.json")
    p.add_argument("--best-valid", type=str, default="outputs/external_performance_report_policy_replay_best_valid_v1.json")
    p.add_argument(
        "--retrain-policy-sweep-json",
        type=str,
        default="outputs/external_policy_replay_sweep_retrain_v3/summary.json",
        help="Optional replay sweep summary used to resolve a selected best-valid report path.",
    )
    p.add_argument(
        "--robust",
        type=str,
        default="",
        help="Deprecated fallback robust policy replay report JSON path.",
    )
    p.add_argument("--out-json", type=str, default="outputs/publication_full_rtx4050/policy_replay_summary.json")
    p.add_argument("--out-md", type=str, default="outputs/publication_full_rtx4050/policy_replay_summary.md")
    return p.parse_args()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(v: Any) -> str:
    try:
        return f"{float(v):.3f}"
    except Exception:
        return "n/a"


def _rows(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in list(report.get("datasets", [])):
        out[str(row.get("dataset", "")).lower()] = row
    return out


def _resolve_aux_policy(
    *,
    best_valid_path: str,
    retrain_policy_sweep_json: str,
    robust_path: str,
) -> tuple[str | None, Path | None]:
    candidates: list[tuple[str, Path]] = []
    if str(best_valid_path).strip():
        candidates.append(("best_valid", Path(best_valid_path).resolve()))

    sweep_path = Path(retrain_policy_sweep_json).resolve()
    if sweep_path.exists():
        try:
            sweep = _load(sweep_path)
        except Exception:
            sweep = {}
        top_level = str(sweep.get("best_policy_report_json", "")).strip()
        if top_level:
            candidates.append(("best_valid", Path(top_level).resolve()))
        nested = sweep.get("best_policy", {}) if isinstance(sweep.get("best_policy", {}), dict) else {}
        nested_path = str(nested.get("report_json", "")).strip()
        if nested_path:
            candidates.append(("best_valid", Path(nested_path).resolve()))

    if str(robust_path).strip():
        candidates.append(("robust", Path(robust_path).resolve()))

    seen: set[tuple[str, str]] = set()
    for label, path in candidates:
        key = (label, str(path))
        if key in seen:
            continue
        seen.add(key)
        if path.exists():
            return label, path
    return None, None


def main() -> None:
    args = parse_args()
    canonical = _load(Path(args.canonical).resolve())
    balanced = _load(Path(args.balanced).resolve())
    aggressive = _load(Path(args.aggressive).resolve())
    aux_label, aux_path = _resolve_aux_policy(
        best_valid_path=args.best_valid,
        retrain_policy_sweep_json=args.retrain_policy_sweep_json,
        robust_path=args.robust,
    )
    aux_report = _load(aux_path) if aux_path is not None else None

    can = _rows(canonical)
    bal = _rows(balanced)
    agg = _rows(aggressive)
    all_sets = set(can.keys()) | set(bal.keys()) | set(agg.keys())
    if aux_report is not None:
        all_sets |= set(_rows(aux_report).keys())
    preferred = ["femto", "xjtu_sy", "cmapss"]
    datasets = [d for d in preferred if d in all_sets] + sorted(d for d in all_sets if d not in preferred)
    aux_rows = _rows(aux_report) if aux_report is not None else {}

    summary_rows: list[dict[str, Any]] = []
    for ds in datasets:
        entry: dict[str, Any] = {"dataset": ds}
        for tag, src in (("canonical", can), ("balanced", bal), ("aggressive", agg)):
            row = src.get(ds)
            if not row or str(row.get("status", "")).lower() != "ok":
                entry[tag] = {"status": "missing"}
                continue
            m = row.get("metrics", {})
            entry[tag] = {
                "status": "ok",
                "rmse": float(m.get("rmse", float("nan"))),
                "rul_cov": float(m.get("rul_cov", float("nan"))),
                "tau_v": float(m.get("tau_v", float("nan"))),
            }
        if aux_report is not None and aux_label is not None:
            row = aux_rows.get(ds)
            if not row or str(row.get("status", "")).lower() != "ok":
                entry[aux_label] = {"status": "missing"}
            else:
                m = row.get("metrics", {})
                entry[aux_label] = {
                    "status": "ok",
                    "rmse": float(m.get("rmse", float("nan"))),
                    "rul_cov": float(m.get("rul_cov", float("nan"))),
                    "tau_v": float(m.get("tau_v", float("nan"))),
                }
        summary_rows.append(entry)

    out = {
        "reports": {
            "canonical": str(Path(args.canonical).resolve()),
            "balanced": str(Path(args.balanced).resolve()),
            "aggressive": str(Path(args.aggressive).resolve()),
            "retrain_policy_sweep_json": str(Path(args.retrain_policy_sweep_json).resolve()) if Path(args.retrain_policy_sweep_json).exists() else "",
            "aux_policy_label": aux_label or "",
            "aux_policy_report": str(aux_path) if aux_path is not None else "",
        },
        "rows": summary_rows,
    }

    out_json = Path(args.out_json).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2, allow_nan=False), encoding="utf-8")

    lines = [
        "# Policy Replay Summary",
        "",
        "| Dataset | Policy | RMSE | RUL cov | Tau v |",
        "|---|---|---:|---:|---:|",
    ]
    policies = ["canonical", "balanced", "aggressive"] + ([aux_label] if aux_label is not None and aux_report is not None else [])
    for r in summary_rows:
        ds = r["dataset"]
        for policy in policies:
            pr = r[policy]
            if pr.get("status") != "ok":
                lines.append(f"| {ds} | {policy} | n/a | n/a | n/a |")
            else:
                lines.append(
                    f"| {ds} | {policy} | {_fmt(pr.get('rmse'))} | {_fmt(pr.get('rul_cov'))} | {_fmt(pr.get('tau_v'))} |"
                )
    any_missing = any(
        r[policy].get("status") != "ok" for r in summary_rows for policy in policies
    )
    lines += [
        "",
        "Notes:",
        "- Balanced/aggressive rows are replayed from fixed canonical checkpoints/calibration bundles.",
    ]
    if any_missing:
        lines.append("- Missing rows indicate unavailable replay artifacts in this run.")
    if aux_report is not None and aux_label == "best_valid":
        lines.append("- best_valid is the sweep-selected width-optimal point among target-valid replay settings.")
    elif aux_report is not None and aux_label is not None:
        lines.append(f"- {aux_label} is an additional replay point selected from policy sweep criteria.")
    lines.append("")
    out_md = Path(args.out_md).resolve()
    out_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved JSON: {out_json}")
    print(f"Saved Markdown: {out_md}")


if __name__ == "__main__":
    main()

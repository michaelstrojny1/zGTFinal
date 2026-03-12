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


def main() -> None:
    args = parse_args()
    canonical = _load(Path(args.canonical).resolve())
    balanced = _load(Path(args.balanced).resolve())
    aggressive = _load(Path(args.aggressive).resolve())

    can = _rows(canonical)
    bal = _rows(balanced)
    agg = _rows(aggressive)
    datasets = ["femto", "xjtu_sy", "cmapss"]

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
        summary_rows.append(entry)

    out = {
        "reports": {
            "canonical": str(Path(args.canonical).resolve()),
            "balanced": str(Path(args.balanced).resolve()),
            "aggressive": str(Path(args.aggressive).resolve()),
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
    for r in summary_rows:
        ds = r["dataset"]
        for policy in ("canonical", "balanced", "aggressive"):
            pr = r[policy]
            if pr.get("status") != "ok":
                lines.append(f"| {ds} | {policy} | n/a | n/a | n/a |")
            else:
                lines.append(
                    f"| {ds} | {policy} | {_fmt(pr.get('rmse'))} | {_fmt(pr.get('rul_cov'))} | {_fmt(pr.get('tau_v'))} |"
                )
    lines += [
        "",
        "Notes:",
        "- Balanced/aggressive rows are replayed from fixed canonical checkpoints/calibration bundles.",
        "- Missing rows indicate unavailable replay artifacts in this run (e.g., interrupted external download).",
        "",
    ]
    out_md = Path(args.out_md).resolve()
    out_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved JSON: {out_json}")
    print(f"Saved Markdown: {out_md}")


if __name__ == "__main__":
    main()


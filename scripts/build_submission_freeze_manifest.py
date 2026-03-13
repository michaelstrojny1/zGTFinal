from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a frozen submission manifest with SHA256 checksums.")
    p.add_argument("--out-json", type=str, default="outputs/publication_full_rtx4050/submission_freeze_manifest.json")
    p.add_argument(
        "--paths",
        type=str,
        default=(
            "paper/main.pdf,"
            "outputs/publication_full_rtx4050/publication_gate_summary.json,"
            "outputs/publication_full_rtx4050/paper_release/reports/paper_honesty_report.json,"
            "outputs/publication_full_rtx4050/policy_replay_summary.json,"
            "outputs/publication_full_rtx4050/claim_significance_report.json,"
            "outputs/publication_full_rtx4050/policy_sharpness_report.json"
        ),
        help="Comma-separated artifact paths to include.",
    )
    return p.parse_args()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    args = parse_args()
    entries: list[dict[str, Any]] = []
    for tok in str(args.paths).split(","):
        p = Path(tok.strip()).resolve()
        if not tok.strip():
            continue
        if not p.exists():
            entries.append({"path": str(p), "exists": False, "sha256": "", "size_bytes": 0})
            continue
        entries.append(
            {
                "path": str(p),
                "exists": True,
                "sha256": _sha256(p),
                "size_bytes": int(p.stat().st_size),
            }
        )
    out = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "entries": entries,
    }
    out_path = Path(args.out_json).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Saved manifest: {out_path}")


if __name__ == "__main__":
    main()

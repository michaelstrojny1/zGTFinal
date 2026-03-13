from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge multiple external baseline package JSON files.")
    p.add_argument(
        "--inputs",
        type=str,
        required=True,
        help="Comma-separated JSON file paths with {'methods':[...]} schema.",
    )
    p.add_argument("--out-json", type=str, default="outputs/external_baselines_merged.json")
    p.add_argument("--out-md", type=str, default="outputs/external_baselines_merged.md")
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


def main() -> None:
    args = parse_args()
    srcs = [Path(x.strip()).resolve() for x in str(args.inputs).split(",") if x.strip()]
    if not srcs:
        raise ValueError("No input files provided.")

    methods: list[dict[str, Any]] = []
    notes: list[str] = []
    seen: set[str] = set()
    for p in srcs:
        blob = _load(p)
        for m in list(blob.get("methods", [])):
            name = str(m.get("name", "")).strip()
            if not name:
                continue
            if name in seen:
                notes.append(f"Skipped duplicate method name: {name}")
                continue
            seen.add(name)
            methods.append(m)
        for n in list(blob.get("notes", [])):
            notes.append(str(n))

    out = {
        "inputs": {"sources": [str(p) for p in srcs]},
        "methods": methods,
        "notes": notes,
    }
    out = _sanitize(out)

    out_json = Path(args.out_json).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2, allow_nan=False), encoding="utf-8")

    lines = ["# Merged External Baselines", "", f"- num_sources: {len(srcs)}", f"- num_methods: {len(methods)}", ""]
    for m in methods:
        lines.append(f"- {m.get('name','unknown')} ({m.get('comparator_type','external')})")
    lines.append("")
    out_md = Path(args.out_md).resolve()
    out_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved JSON: {out_json}")
    print(f"Saved Markdown: {out_md}")


if __name__ == "__main__":
    main()

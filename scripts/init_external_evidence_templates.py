from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Initialize template JSON files for external evidence artifacts.")
    p.add_argument("--fds", nargs="+", type=int, default=[1, 2, 3, 4], help="FD ids to include in template rows.")
    p.add_argument("--out-dir", type=str, default="outputs/templates")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    fds = sorted(set(int(x) for x in args.fds))
    baseline_template = {
        "notes": [
            "Fill with external baseline method results measured under the same protocol as strict_main.",
            "Set comparator_type='external' for methods that are truly external to this codebase.",
        ],
        "methods": [
            {
                "name": "external_method_1",
                "comparator_type": "external",
                "description": "e.g., conformal residual baseline from external implementation",
                "per_fd": [
                    {
                        "fd": fd,
                        "rmse": 0.0,
                        "rul_cov": 0.0,
                        "tau_v": 0.0,
                        "run_dir": "",
                    }
                    for fd in fds
                ],
            },
            {
                "name": "external_method_2",
                "comparator_type": "external",
                "description": "e.g., CUSUM/Page-Hinkley external baseline",
                "per_fd": [
                    {
                        "fd": fd,
                        "rmse": 0.0,
                        "rul_cov": 0.0,
                        "tau_v": 0.0,
                        "run_dir": "",
                    }
                    for fd in fds
                ],
            },
        ],
    }

    external_perf_template = {
        "notes": [
            "Fill with real external dataset model evaluation outputs.",
            "Readiness gate expects at least one dataset row with status='ok'.",
        ],
        "datasets": [
            {
                "dataset": "femto",
                "status": "pending",
                "num_runs": 0,
                "metrics": {
                    "rul_cov": None,
                    "tau_v": None,
                    "rmse": None,
                },
            },
            {
                "dataset": "xjtu_sy",
                "status": "pending",
                "num_runs": 0,
                "metrics": {
                    "rul_cov": None,
                    "tau_v": None,
                    "rmse": None,
                },
            },
        ],
    }

    baseline_path = out_dir / "external_baselines_template.json"
    perf_path = out_dir / "external_performance_template.json"
    baseline_path.write_text(json.dumps(baseline_template, indent=2, allow_nan=False), encoding="utf-8")
    perf_path.write_text(json.dumps(external_perf_template, indent=2, allow_nan=False), encoding="utf-8")

    print(f"Saved template: {baseline_path}")
    print(f"Saved template: {perf_path}")


if __name__ == "__main__":
    main()

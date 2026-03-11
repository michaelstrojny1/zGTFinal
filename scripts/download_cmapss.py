from __future__ import annotations

import argparse
from pathlib import Path

from rul_datasets.reader import CmapssReader
from rul_datasets.reader import cmapss as cmapss_module
from rul_datasets.reader.data_root import set_data_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and prepare C-MAPSS via rul-datasets.")
    parser.add_argument("--fds", type=int, nargs="+", default=[1, 2, 3, 4], help="FD subsets to prepare.")
    parser.add_argument("--data-root", type=str, default="data/rul_datasets", help="Dataset root directory.")
    parser.add_argument("--window-size", type=int, default=30, help="Window size for preparation.")
    parser.add_argument("--max-rul", type=int, default=125, help="Maximum capped RUL.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    data_root.mkdir(parents=True, exist_ok=True)
    set_data_root(str(data_root))
    cmapss_module.CmapssReader._CMAPSS_ROOT = str(data_root / "CMAPSS")

    for fd in args.fds:
        reader = CmapssReader(fd=fd, window_size=args.window_size, max_rul=args.max_rul)
        reader.prepare_data()
        print(f"FD{fd:03d} ready at {data_root}")


if __name__ == "__main__":
    main()

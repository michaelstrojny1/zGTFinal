from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compile paper/main.tex with XeLaTeX + BibTeX.")
    p.add_argument("--paper-dir", type=str, default="paper")
    p.add_argument("--tex", type=str, default="main.tex")
    return p.parse_args()


def _run(cmd: list[str], cwd: Path) -> None:
    print("RUN:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> None:
    args = parse_args()
    paper_dir = Path(args.paper_dir).resolve()
    tex = str(args.tex)
    _run(["xelatex", "-interaction=nonstopmode", "-halt-on-error", tex], paper_dir)
    _run(["bibtex", Path(tex).stem], paper_dir)
    _run(["xelatex", "-interaction=nonstopmode", "-halt-on-error", tex], paper_dir)
    _run(["xelatex", "-interaction=nonstopmode", "-halt-on-error", tex], paper_dir)
    print(f"Built PDF: {paper_dir / (Path(tex).stem + '.pdf')}")


if __name__ == "__main__":
    main()


"""Regenerate every approach's assets and metrics.

Usage
-----
    uv run python research/run_all.py
    uv run python research/run_all.py --image limestone

Each approach is executed in its own process with the current interpreter, so
one failing approach does not hide the others. Exit code is non-zero if any of
them failed.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def approach_dirs() -> list[Path]:
    """Approach directories, i.e. every subdirectory holding a ``run.py``."""
    return sorted(p.parent for p in HERE.glob("*/run.py"))


def main(argv: list[str] | None = None) -> int:
    """Run every approach; forward ``--image`` to each of them."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--image", action="append", default=None,
                        help="substring of a file name in data/; repeatable")
    args = parser.parse_args(argv)

    forwarded: list[str] = []
    for pattern in args.image or []:
        forwarded += ["--image", pattern]

    failures = 0
    for directory in approach_dirs():
        print(f"\n=== {directory.name} " + "=" * (60 - len(directory.name)))
        command = [sys.executable, str(directory / "run.py"), *forwarded]
        result = subprocess.run(command, check=False)
        if result.returncode:
            failures += 1
            print(f"!! {directory.name} exited with {result.returncode}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

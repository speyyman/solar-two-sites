"""Check that every number in a piece of writing comes from results/summary.json.

Usage: python scripts/check_numbers.py <file>... [--allow 2024 2025 ...]

Prints each number found in the files that is not a value in summary.json (after
normalising signs, thousands separators and percent signs) and is not in the allow
list. Exit status 1 if any remain. Dates, plant IDs and definitions (an 80% interval,
a 25 degree tilt) are expected in the output; pass them with --allow once read.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NUMBER = re.compile(r"(?<![\w.])[+\-−]?\d[\d,]*(?:\.\d+)?%?")


def normalise(token: str) -> str:
    t = token.replace("−", "-").replace(",", "").rstrip("%")
    return t.lstrip("+")


def known_values() -> set[str]:
    summary = json.loads((ROOT / "results" / "summary.json").read_text(encoding="utf-8"))
    return {normalise(str(v)) for v in summary.values()}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+", type=Path)
    p.add_argument("--allow", nargs="*", default=[])
    a = p.parse_args()
    known = known_values() | {normalise(x) for x in a.allow}
    missing: list[tuple[str, int, str]] = []
    for f in a.files:
        for n, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            for m in NUMBER.finditer(line):
                if normalise(m.group()) not in known:
                    missing.append((f.name, n, m.group()))
    for name, n, tok in missing:
        print(f"{name}:{n}: {tok}")
    print(f"{len(missing)} number(s) not in results/summary.json or the allow list")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())

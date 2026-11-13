#!/usr/bin/env python
"""Regenerate golden images for the plot tests. EXPLICIT USE ONLY.

    python tools/regenerate_goldens.py --reason "why the expected look changed"

Golden images are the expected appearance of every plot type and style preset. Only
regenerate them after deliberately changing how figures look, check the new images by
eye, and commit them together with the change. Every run is appended to
tests/golden/REGENERATION_LOG.md (and should be mentioned in DECISIONS.md).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--reason", required=True, help="Why the goldens are being regenerated")
    parser.add_argument("--only", nargs="*", help="Regenerate only these case names")
    args = parser.parse_args()

    import matplotlib

    from tests.golden.cases import DPI, build_cases
    from torusplot.export import figure_bytes

    out = ROOT / "tests" / "golden" / "images"
    out.mkdir(parents=True, exist_ok=True)
    names = []
    for name, make in build_cases().items():
        if args.only and name not in args.only:
            continue
        (out / f"{name}.png").write_bytes(figure_bytes(make(), "png", dpi=DPI))
        names.append(name)
        print(f"wrote {name}.png")
    log = ROOT / "tests" / "golden" / "REGENERATION_LOG.md"
    if not log.exists():
        log.write_text("# Golden image regeneration log\n\n| Date | matplotlib | Cases | Reason |\n|---|---|---|---|\n", encoding="utf-8")
    with log.open("a", encoding="utf-8") as fh:
        what = "all" if not args.only else ", ".join(names)
        fh.write(f"| {_dt.date.today().isoformat()} | {matplotlib.__version__} | {what} ({len(names)}) | {args.reason} |\n")
    print(f"Logged in {log.relative_to(ROOT)}. Review the images before committing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

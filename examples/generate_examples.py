#!/usr/bin/env python
"""Regenerate the SYNTHETIC sample datasets in this folder.

    python examples/generate_examples.py

All numbers are simulated with fixed random seeds by torusplot/data/samples.py.
They are not real measurements and must not be cited as data.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from torusplot.data.samples import write_examples  # noqa: E402

if __name__ == "__main__":
    for path in write_examples(Path(__file__).resolve().parent):
        print(f"wrote {path.name}")

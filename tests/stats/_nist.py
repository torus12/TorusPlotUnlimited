"""Loader for the NIST StRD one-way ANOVA files kept verbatim in ``tests/fixtures/nist``.

Each file is the unmodified NIST/ITL text file: a header stating which lines hold the
certified values and which hold the data, the certified ANOVA table, then one
``treatment  response`` pair per line. Source and attribution:
``tests/fixtures/sources/oneway.md``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import numpy as np

NIST_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "nist"


@dataclass(frozen=True)
class NistAnova:
    name: str
    difficulty: str
    #: treatment label -> response values exactly as printed (decimal strings)
    text: dict[str, list[str]]
    df_between: int
    df_within: int
    ss_between: float
    ss_within: float
    ms_between: float
    ms_within: float
    f: float
    r_squared: float
    residual_sd: float

    def groups(self) -> dict[str, np.ndarray]:
        """The data as ordinary double-precision numbers (what an analysis receives)."""
        return {k: np.array([float(v) for v in vals]) for k, vals in self.text.items()}

    def groups_offset_removed(self) -> dict[str, np.ndarray]:
        """The data with the smallest value subtracted in exact decimal arithmetic.

        ANOVA is invariant to this shift; it removes the constant leading digits that
        cannot be represented exactly in double precision.
        """
        low = min(Decimal(v) for vals in self.text.values() for v in vals)
        return {k: np.array([float(Decimal(v) - low) for v in vals])
                for k, vals in self.text.items()}


def _span(header: str, label: str) -> tuple[int, int]:
    match = re.search(label + r"\s+\(lines\s+(\d+)\s+to\s+(\d+)\)", header)
    assert match, f"cannot find the '{label}' line range"
    return int(match.group(1)), int(match.group(2))


def load_nist_anova(name: str) -> NistAnova:
    lines = (NIST_DIR / f"{name}.dat").read_text(encoding="utf-8").splitlines()
    header = "\n".join(lines[:30])
    cert_start, cert_end = _span(header, "Certified Values")
    data_start, data_end = _span(header, "Data")
    difficulty = re.search(r"(Lower|Average|Higher) Level of Difficulty", header).group(1).lower()
    values: dict[str, float] = {}
    # AtmWtAg's header says lines 41-47 but its table sits one line lower, so scan every
    # line between the stated start and the data block.
    assert cert_end < data_start
    for line in lines[cert_start - 1:data_start - 1]:
        tokens = line.split()
        if not tokens:
            continue
        if tokens[0] == "Between":
            values.update(df_between=int(tokens[2]), ss_between=float(tokens[3]),
                          ms_between=float(tokens[4]), f=float(tokens[5]))
        elif tokens[0] == "Within":
            values.update(df_within=int(tokens[2]), ss_within=float(tokens[3]),
                          ms_within=float(tokens[4]))
        elif "R-Squared" in line:
            values["r_squared"] = float(tokens[-1])
        elif "Standard Deviation" in line:
            values["residual_sd"] = float(tokens[-1])
    text: dict[str, list[str]] = {}
    for line in lines[data_start - 1:data_end]:
        tokens = line.split()
        text.setdefault(tokens[0], []).append(tokens[1])
    return NistAnova(name=name, difficulty=difficulty, text=text, **values)

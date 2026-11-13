"""Loader for the NIST StRD regression data files stored next to this module.

Source of the data and certified values: NIST Statistical Reference Datasets (StRD),
National Institute of Standards and Technology, Information Technology Laboratory,
https://www.itl.nist.gov/div898/strd/ (a work of the US government, public domain).
The ``.dat`` files are kept exactly as NIST distributes them (their headers carry the
certified parameter values, standard deviations, residual sum of squares and the two
official starting points). See ``tests/fixtures/sources/regression.md`` for how each file
was obtained.

The model equations below were typed from the "Model:" section of each file header, in the
syntax of :mod:`torusplot.stats.equation` (round brackets, ``^`` for powers).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

#: Nonlinear least-squares datasets: file stem -> model equation.
NLS_MODELS: dict[str, str] = {
    "Misra1a": "y = b1*(1 - exp(-b2*x))",
    "Chwirut2": "y = exp(-b1*x) / (b2 + b3*x)",
    "Chwirut1": "y = exp(-b1*x) / (b2 + b3*x)",
    "Lanczos3": "y = b1*exp(-b2*x) + b3*exp(-b4*x) + b5*exp(-b6*x)",
    "Gauss1": "y = b1*exp(-b2*x) + b3*exp(-(x - b4)^2 / b5^2) + b6*exp(-(x - b7)^2 / b8^2)",
    "Gauss2": "y = b1*exp(-b2*x) + b3*exp(-(x - b4)^2 / b5^2) + b6*exp(-(x - b7)^2 / b8^2)",
    "DanWood": "y = b1*x^b2",
    "Misra1b": "y = b1*(1 - (1 + b2*x/2)^(-2))",
    "Kirby2": "y = (b1 + b2*x + b3*x^2) / (1 + b4*x + b5*x^2)",
    "Hahn1": "y = (b1 + b2*x + b3*x^2 + b4*x^3) / (1 + b5*x + b6*x^2 + b7*x^3)",
    "MGH17": "y = b1 + b2*exp(-x*b4) + b3*exp(-x*b5)",
    "Lanczos1": "y = b1*exp(-b2*x) + b3*exp(-b4*x) + b5*exp(-b6*x)",
    "Lanczos2": "y = b1*exp(-b2*x) + b3*exp(-b4*x) + b5*exp(-b6*x)",
    "Gauss3": "y = b1*exp(-b2*x) + b3*exp(-(x - b4)^2 / b5^2) + b6*exp(-(x - b7)^2 / b8^2)",
    "Misra1c": "y = b1*(1 - (1 + 2*b2*x)^(-0.5))",
    "Misra1d": "y = b1*b2*x*((1 + b2*x)^(-1))",
    "Roszman1": "y = b1 - b2*x - atan(b3 / (x - b4)) / pi",
    "ENSO": ("y = b1 + b2*cos(2*pi*x/12) + b3*sin(2*pi*x/12) + b5*cos(2*pi*x/b4) "
             "+ b6*sin(2*pi*x/b4) + b8*cos(2*pi*x/b7) + b9*sin(2*pi*x/b7)"),
    "MGH09": "y = b1*(x^2 + x*b2) / (x^2 + x*b3 + b4)",
    "Thurber": "y = (b1 + b2*x + b3*x^2 + b4*x^3) / (1 + b5*x + b6*x^2 + b7*x^3)",
    "BoxBOD": "y = b1*(1 - exp(-b2*x))",
    "Rat42": "y = b1 / (1 + exp(b2 - b3*x))",
    "MGH10": "y = b1 * exp(b2 / (x + b3))",
    "Eckerle4": "y = (b1/b2) * exp(-0.5*((x - b3)/b2)^2)",
    "Rat43": "y = b1 / ((1 + exp(b2 - b3*x))^(1/b4))",
    "Bennett5": "y = b1 * (b2 + x)^(-1/b3)",
}

#: NIST's own difficulty classification of each nonlinear dataset.
NLS_DIFFICULTY: dict[str, str] = {
    "Misra1a": "lower", "Chwirut2": "lower", "Chwirut1": "lower", "Lanczos3": "lower",
    "Gauss1": "lower", "Gauss2": "lower", "DanWood": "lower", "Misra1b": "lower",
    "Kirby2": "average", "Hahn1": "average", "MGH17": "average", "Lanczos1": "average",
    "Lanczos2": "average", "Gauss3": "average", "Misra1c": "average", "Misra1d": "average",
    "Roszman1": "average", "ENSO": "average",
    "MGH09": "higher", "Thurber": "higher", "BoxBOD": "higher", "Rat42": "higher",
    "MGH10": "higher", "Eckerle4": "higher", "Rat43": "higher", "Bennett5": "higher",
}

_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
_NLS_PARAM = re.compile(
    rf"^\s*(b\d+)\s*=\s*({_NUM})\s+({_NUM})\s+({_NUM})\s+({_NUM})\s*$"
)
_LLS_PARAM = re.compile(rf"^\s*(B\d+)\s+({_NUM})\s+({_NUM})\s*$")


@dataclass(frozen=True)
class NlsDataset:
    """One NIST StRD nonlinear regression problem."""

    name: str
    equation: str
    difficulty: str
    x: np.ndarray
    y: np.ndarray
    parameters: tuple[str, ...]
    start1: dict[str, float]
    start2: dict[str, float]
    certified: dict[str, float]
    certified_sd: dict[str, float]
    residual_ss: float
    residual_sd: float
    df: int
    n: int

    def start(self, which: int) -> dict[str, float]:
        return dict(self.start1 if which == 1 else self.start2)


@dataclass(frozen=True)
class LlsDataset:
    """One NIST StRD linear regression problem (y, then one or more predictors)."""

    name: str
    x: np.ndarray            # shape (n, number of predictors)
    y: np.ndarray
    certified: dict[str, float]
    certified_sd: dict[str, float]
    residual_sd: float
    r_squared: float
    f_statistic: float
    ss_regression: float
    ss_residual: float
    df_regression: int
    df_residual: int


def _lines(name: str) -> list[str]:
    path = HERE / f"{name}.dat"
    return path.read_text(encoding="utf-8").splitlines()


def _after(lines: list[str], label: str) -> float:
    for line in lines:
        if line.strip().startswith(label):
            return float(line.split(":")[1].split()[0])
    raise ValueError(f"'{label}' not found")


def _data_rows(lines: list[str]) -> np.ndarray:
    starts = [i for i, line in enumerate(lines) if line.startswith("Data:")]
    rows = []
    for line in lines[starts[-1] + 1:]:
        parts = line.split()
        if not parts:
            continue
        rows.append([float(p) for p in parts])
    return np.asarray(rows, dtype=float)


def load_nls(name: str) -> NlsDataset:
    """Parse ``<name>.dat`` (NIST nonlinear least-squares format)."""
    lines = _lines(name)
    names, s1, s2, cert, sd = [], {}, {}, {}, {}
    for line in lines:
        m = _NLS_PARAM.match(line)
        if m:
            key = m.group(1)
            names.append(key)
            s1[key], s2[key] = float(m.group(2)), float(m.group(3))
            cert[key], sd[key] = float(m.group(4)), float(m.group(5))
    data = _data_rows(lines)
    n = int(_after(lines, "Number of Observations"))
    if data.shape != (n, 2):
        raise ValueError(f"{name}: expected {n} rows of (y, x), found shape {data.shape}")
    return NlsDataset(
        name=name, equation=NLS_MODELS[name], difficulty=NLS_DIFFICULTY[name],
        x=data[:, 1].copy(), y=data[:, 0].copy(), parameters=tuple(names),
        start1=s1, start2=s2, certified=cert, certified_sd=sd,
        residual_ss=_after(lines, "Residual Sum of Squares"),
        residual_sd=_after(lines, "Residual Standard Deviation"),
        df=int(_after(lines, "Degrees of Freedom")), n=n,
    )


def load_lls(name: str) -> LlsDataset:
    """Parse ``<name>.dat`` (NIST linear least-squares format: Norris, Longley...)."""
    lines = _lines(name)
    cert, sd = {}, {}
    residual_sd = r2 = None
    reg = res = None
    for i, line in enumerate(lines):
        m = _LLS_PARAM.match(line)
        if m:
            cert[m.group(1)], sd[m.group(1)] = float(m.group(2)), float(m.group(3))
        text = line.strip()
        if text.startswith("Standard Deviation") and residual_sd is None and i > 0 \
                and lines[i - 1].strip().startswith("Residual"):
            residual_sd = float(text.split()[-1])
        elif text.startswith("R-Squared"):
            r2 = float(text.split()[-1])
        elif text.startswith("Regression") and len(text.split()) == 5:
            reg = [float(v) for v in text.split()[1:]]
        elif text.startswith("Residual") and len(text.split()) == 4:
            res = [float(v) for v in text.split()[1:]]
    data = _data_rows(lines)
    return LlsDataset(
        name=name, x=data[:, 1:].copy(), y=data[:, 0].copy(), certified=cert, certified_sd=sd,
        residual_sd=float(residual_sd), r_squared=float(r2), f_statistic=reg[3],
        ss_regression=reg[1], ss_residual=res[1], df_regression=int(reg[0]),
        df_residual=int(res[0]),
    )

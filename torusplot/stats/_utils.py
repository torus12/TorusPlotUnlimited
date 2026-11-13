"""Shared helpers for the statistics engine (input cleaning, wording, formatting)."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import numpy as np

from .result import format_p


class StatsError(ValueError):
    """Raised when an analysis cannot be computed (too few values, wrong design...).

    The message is written for end users and says what to fix.
    """


class ApproximateDataError(StatsError):
    """Raised when a test is requested on digitized / approximate data without override."""


def clean(values: Iterable[Any] | None, name: str = "data") -> tuple[np.ndarray, list[str]]:
    """Return (1-D float array without NaN/inf, notes about what was dropped)."""
    if values is None:
        raise StatsError(f"No data given for {name}.")
    arr = np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=float).ravel()
    notes: list[str] = []
    n_nan = int(np.isnan(arr).sum())
    n_inf = int(np.isinf(arr).sum())
    if n_nan:
        notes.append(f"{name}: {n_nan} missing value(s) ignored.")
    if n_inf:
        notes.append(f"{name}: {n_inf} infinite value(s) ignored.")
    return arr[np.isfinite(arr)], notes


def clean_groups(groups: dict[str, Iterable[Any]]) -> tuple[dict[str, np.ndarray], list[str]]:
    """Clean every group; keeps order; returns (groups, notes)."""
    out: dict[str, np.ndarray] = {}
    notes: list[str] = []
    for name, values in groups.items():
        arr, n = clean(values, str(name))
        out[str(name)] = arr
        notes.extend(n)
    return out, notes


def clean_paired(
    x: Iterable[Any], y: Iterable[Any], names: tuple[str, str] = ("x", "y")
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Pairwise-complete cleaning for paired / XY data."""
    a = np.asarray(x, dtype=float).ravel()
    b = np.asarray(y, dtype=float).ravel()
    if a.shape != b.shape:
        raise StatsError(
            f"Paired data must have the same number of values ({names[0]} has {a.size}, "
            f"{names[1]} has {b.size})."
        )
    ok = np.isfinite(a) & np.isfinite(b)
    notes = []
    dropped = int((~ok).sum())
    if dropped:
        notes.append(f"{dropped} pair(s) with a missing value were ignored.")
    return a[ok], b[ok], notes


def require_n(arr: np.ndarray, minimum: int, name: str, what: str = "this analysis") -> None:
    if arr.size < minimum:
        raise StatsError(
            f"{name} has {arr.size} value(s); {what} needs at least {minimum}."
        )


#: Spreads below this are treated as zero: their squares/cubes underflow in double precision,
#: so variances and higher moments cannot be computed reliably.
_TINY_SPREAD = 1e-75


def is_constant(arr: np.ndarray) -> bool:
    """True if all values are identical (zero variance), within floating-point rounding.

    The tolerance is a few units in the last place of the largest magnitude, so data with
    a huge offset and a tiny but real spread (e.g. 1e12 + 0.1, 1e12 + 0.3) are *not*
    mistaken for a constant. A spread so small that its powers underflow (below 1e-75) is
    also treated as constant.
    """
    if arr.size < 2:
        return True
    scale = float(np.max(np.abs(arr)))
    return bool(np.ptp(arr) <= max(4.0 * np.spacing(scale), _TINY_SPREAD))


def fmt(value: float | None, digits: int = 4) -> str:
    """Compact number formatting with ``digits`` significant figures."""
    if value is None:
        return "n/a"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(v):
        return "n/a"
    if math.isinf(v):
        return "infinity" if v > 0 else "-infinity"
    if v == 0:
        return "0"
    if abs(v) >= 10 ** digits or abs(v) < 1e-4:
        return f"{v:.{max(digits - 1, 1)}e}"
    return f"{v:.{digits}g}"


def p_phrase(p: float | None) -> str:
    """'P = 0.0123' / 'P < 0.0001'."""
    text = format_p(p)
    if text == "n/a":
        return "P not available"
    return f"P < {text[1:]}" if text.startswith("<") else f"P = {text}"


def sig_phrase(p: float | None, alpha: float = 0.05) -> str:
    """'statistically significant (P = 0.012)' or 'not statistically significant (P = 0.34)'."""
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "could not be evaluated"
    word = "statistically significant" if p < alpha else "not statistically significant"
    return f"{word} ({p_phrase(p)}, alpha = {alpha:g})"


def ci_phrase(low: float | None, high: float | None, level: float = 0.95) -> str:
    return f"{level * 100:g}% CI {fmt(low)} to {fmt(high)}"


def alternative_label(alternative: str) -> str:
    return {
        "two-sided": "two-tailed",
        "greater": "one-tailed (greater)",
        "less": "one-tailed (less)",
    }.get(alternative, alternative)


def check_alternative(alternative: str) -> str:
    if alternative not in {"two-sided", "greater", "less"}:
        raise StatsError("alternative must be 'two-sided', 'greater' or 'less'.")
    return alternative


def check_alpha(alpha: float) -> float:
    if not (0 < float(alpha) < 1):
        raise StatsError("alpha must be between 0 and 1 (typically 0.05).")
    return float(alpha)


def jsonable(obj: Any) -> Any:
    """Recursively convert numpy types to plain Python so results serialise to JSON."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [jsonable(v) for v in obj.tolist()]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj

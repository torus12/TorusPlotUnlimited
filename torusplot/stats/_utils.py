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


def is_constant(arr: np.ndarray) -> bool:
    """True if all values are identical (zero variance), within floating-point rounding.

    The only criterion is the spread relative to the size of the numbers: a range of at
    most a few units in the last place of the largest magnitude is rounding noise. Data
    with a huge offset and a tiny but real spread (e.g. 1e12 + 0.1, 1e12 + 0.3) are *not*
    constant, and neither are very small numbers that differ by a real fraction of their
    size (1e-80, 2e-80, 3e-80). There is no absolute floor: computations that could
    under- or overflow for extreme magnitudes use :func:`rescale` instead.
    """
    arr = np.asarray(arr, dtype=float)
    if arr.size < 2:
        return True
    lo, hi = float(np.min(arr)), float(np.max(arr))
    scale = max(abs(lo), abs(hi))
    # halves: the range of e.g. [-1.7e308, 1.7e308] is not representable
    return bool(0.5 * hi - 0.5 * lo <= 2.0 * np.spacing(scale))


_TOO_LARGE = (
    "The values are so large that this result cannot be represented as a number (numbers on a "
    "computer end at about 1.8e308). Express the data in a larger unit and run the analysis again."
)


class Rescaled:
    """Data mapped to z = (x - location) / scale with ``scale`` an exact power of two.

    Test statistics, P values and standardised effect sizes are unchanged by a change of
    origin and unit, so they can be computed from ``arrays`` (values of order 1) without
    the overflow (|x| above about 1e154) or underflow (spread below about 1e-154, or 1e-75
    for third and fourth moments) that squaring the raw numbers causes. Quantities in the
    units of the data are mapped back with :meth:`location` (means, medians, limits of a
    CI of a mean), :meth:`spread` (SDs, SEs, differences of means and their CI limits)
    and :meth:`square` (variances, sums of squares).

    ``noise`` is the size, in rescaled units, of one unit in the last place of the largest
    original magnitude: quantities of the order of a few ``noise`` are rounding error.
    """

    __slots__ = ("arrays", "_centre", "_e1", "_e2", "noise")

    def __init__(self, arrays: list[np.ndarray], centre: float, e1: int, e2: int, noise: float) -> None:
        self.arrays = arrays
        self._centre = centre
        self._e1 = e1
        self._e2 = e2
        self.noise = noise

    def location(self, value: float | None) -> float | None:
        """Map a location-type quantity (mean, median, quantile, CI limit of a mean) back."""
        if value is None:
            return None
        v = float(value)
        if not math.isfinite(v):
            return v
        with np.errstate(over="ignore", under="ignore"):
            out = float(np.ldexp(self._centre + float(np.ldexp(v, self._e2)), self._e1))
        if not math.isfinite(out):
            raise StatsError(_TOO_LARGE)
        return out

    def spread(self, value: float | None) -> float | None:
        """Map a scale-type quantity (SD, SE, difference, half-width) back; StatsError if
        the result exceeds the largest representable number."""
        if value is None:
            return None
        v = float(value)
        if not math.isfinite(v):
            return v
        with np.errstate(over="ignore", under="ignore"):
            out = float(np.ldexp(v, self._e1 + self._e2))
        if not math.isfinite(out):
            raise StatsError(_TOO_LARGE)
        return out

    def spread_array(self, values: np.ndarray) -> np.ndarray:
        """:meth:`spread` for a whole array (e.g. residuals); StatsError on overflow."""
        with np.errstate(over="ignore", under="ignore"):
            out = np.ldexp(np.asarray(values, dtype=float), self._e1 + self._e2)
        if not np.all(np.isfinite(out)):
            raise StatsError(_TOO_LARGE)
        return out

    def square(self, value: float | None) -> float | None:
        """Map a squared-unit quantity (variance, sum of squares, mean square) back.

        Returns ``None`` when the result cannot be represented: it overflows, or a
        non-zero value underflows to zero."""
        if value is None:
            return None
        v = float(value)
        if not math.isfinite(v):
            return v
        with np.errstate(over="ignore", under="ignore"):
            out = float(np.ldexp(v, 2 * (self._e1 + self._e2)))
        if not math.isfinite(out) or (out == 0.0 and v != 0.0):
            return None
        return out

    @property
    def exponents(self) -> tuple[int, int]:
        """(e1, e2): data = (centre + z * 2**e2) * 2**e1; ``x / 2**e1`` is the data scaled
        by a power of two only (no centring)."""
        return self._e1, self._e2

    def to_scaled_location(self, value: float) -> float:
        """Map a location in data units (e.g. a hypothetical mean) into rescaled units."""
        with np.errstate(over="ignore", under="ignore"):
            return float(np.ldexp(float(np.ldexp(float(value), -self._e1)) - self._centre, -self._e2))


def rescale(*arrays: np.ndarray) -> Rescaled:
    """Jointly centre and scale one or more samples (see :class:`Rescaled`).

    All arrays share one location (the mid-range of the pooled values) and one scale (a
    power of two just above the largest absolute deviation from that location), so
    differences between groups are preserved. Dividing by a power of two is exact, and
    subtracting the location is exact whenever the values lie within a factor of two of
    it (the large-offset case), so no accuracy is lost.
    """
    arrs = [np.asarray(a, dtype=float) for a in arrays]
    sizes = [a.size for a in arrs]
    pooled = np.concatenate([a.ravel() for a in arrs]) if arrs else np.empty(0)
    if pooled.size == 0:
        return Rescaled(arrs, 0.0, 0, 0, 0.0)
    amax = float(np.max(np.abs(pooled)))
    if amax == 0.0 or not math.isfinite(amax):
        return Rescaled(arrs, 0.0, 0, 0, 0.0)
    e1 = math.frexp(amax)[1]
    with np.errstate(under="ignore"):
        y = np.ldexp(pooled, -e1)  # |y| < 1, exact
    centre = 0.5 * float(np.min(y)) + 0.5 * float(np.max(y))
    dev = y - centre
    dmax = float(np.max(np.abs(dev)))
    e2 = math.frexp(dmax)[1] if dmax > 0.0 else 0
    z = np.ldexp(dev, -e2) if e2 else dev
    # one ulp of the largest magnitude (0.5 <= amax / 2**e1 < 1) expressed in z units
    noise = float(np.ldexp(np.finfo(float).eps / 2.0, -e2))
    out: list[np.ndarray] = []
    start = 0
    for a, size in zip(arrs, sizes):
        out.append(z[start:start + size].reshape(a.shape))
        start += size
    return Rescaled(out, centre, e1, e2, noise)


def pow2_scaled(arr: np.ndarray) -> tuple[np.ndarray, int]:
    """(arr / 2**e, e) with e chosen so that the largest magnitude lies in [0.5, 1).

    Scaling by a power of two is exact, so statistics computed from the scaled values and
    mapped back with :func:`pow2_back` are bit-for-bit those of the original values, but
    without overflow for huge numbers or underflow for tiny ones.
    """
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        return arr, 0
    amax = float(np.max(np.abs(arr)))
    if amax == 0.0 or not math.isfinite(amax):
        return arr, 0
    e = math.frexp(amax)[1]
    with np.errstate(under="ignore"):
        return np.ldexp(arr, -e), e


def pow2_back(value: float, e: int, power: int = 1) -> float:
    """value * 2**(power * e); raises StatsError when that exceeds the largest double."""
    with np.errstate(over="ignore", under="ignore"):
        out = float(np.ldexp(float(value), power * e))
    if math.isfinite(value) and not math.isfinite(out):
        raise StatsError(_TOO_LARGE)
    return out


def difference_tolerance(a: np.ndarray, b: np.ndarray | float, factor: float = 2.0) -> np.ndarray:
    """Rounding tolerance of each difference ``a - b``.

    A difference inherits the rounding error of the two numbers it was computed from, so
    its uncertainty is a few units in the last place of the *larger of |a| and |b|*, not
    of the (possibly much smaller) difference itself: 100.1 - 100 is 0.0999999999999943.
    """
    a = np.asarray(a, dtype=float)
    b = np.broadcast_to(np.asarray(b, dtype=float), a.shape)
    return factor * np.spacing(np.maximum(np.abs(a), np.abs(b)))


def differences_constant(d: np.ndarray, tol: np.ndarray) -> bool:
    """True when all differences are equal to within their rounding tolerances, i.e. the
    intervals ``d_i +/- tol_i`` have a point in common."""
    d = np.asarray(d, dtype=float)
    if d.size < 2:
        return True
    return bool(np.max(d - tol) <= np.min(d + tol))


def snap_differences(d: np.ndarray, tol: np.ndarray) -> np.ndarray:
    """Differences with rounding noise removed, for sign and rank procedures.

    * a difference no larger than its tolerance becomes exactly 0;
    * absolute differences that agree to within their tolerances (neighbours in sorted
      order closer than the sum of their tolerances) are made exactly equal, so they are
      recognised as ties when ranked. The common value is that of the most precisely
      known member of the tie group (smallest tolerance).
    """
    d = np.asarray(d, dtype=float)
    tol = np.broadcast_to(np.asarray(tol, dtype=float), d.shape)
    out = np.where(np.abs(d) <= tol, 0.0, d)
    idx = np.nonzero(out != 0.0)[0]
    if idx.size < 2:
        return out
    mag = np.abs(out[idx])
    order = np.argsort(mag, kind="stable")
    sorted_mag = mag[order]
    sorted_tol = tol[idx][order]
    new_group = np.concatenate([[True], np.diff(sorted_mag) > sorted_tol[1:] + sorted_tol[:-1]])
    if bool(np.all(new_group)):
        return out
    group = np.cumsum(new_group) - 1
    starts = np.nonzero(new_group)[0]
    ends = np.concatenate([starts[1:], [sorted_mag.size]])
    rep = np.empty(starts.size)
    for g, (s, e) in enumerate(zip(starts, ends)):
        rep[g] = sorted_mag[s + int(np.argmin(sorted_tol[s:e]))]
    snapped = np.empty(mag.size)
    snapped[order] = rep[group]
    out[idx] = np.copysign(snapped, out[idx])
    return out


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

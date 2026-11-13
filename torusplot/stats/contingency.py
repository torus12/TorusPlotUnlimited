"""Contingency tables: chi-square test, Fisher's exact test, chi-square test for trend,
and the 2x2 measures (odds ratio, relative risk, risk difference, NNT, diagnostic
accuracy) with confidence intervals.

Table orientation used throughout
---------------------------------
Rows are the groups being compared (exposure / treatment), with the **first row** the
exposed / treated group. Columns are the outcome, with the **first column** "outcome
present". For a 2x2 table ``[[a, b], [c, d]]``::

                     outcome present   outcome absent
    first row  (1)          a                b            risk1 = a / (a + b)
    second row (2)          c                d            risk2 = c / (c + d)

    odds ratio = (a d) / (b c)    relative risk = risk1 / risk2    risk difference = risk1 - risk2

Cells must be **counts of subjects** (whole numbers), never percentages or means.

References
----------
* Pearson K (1900) On the criterion that a given system of deviations from the probable
  ... *Philos Mag* 50:157-175 (chi-square test).
* Yates F (1934) Contingency tables involving small numbers and the chi-square test.
  *J R Stat Soc Suppl* 1:217-235 (continuity correction for 2x2 tables).
* Cochran WG (1954) Some methods for strengthening the common chi-square tests.
  *Biometrics* 10:417-451 (expected-count rule; test for linear trend).
* Armitage P (1955) Tests for linear trends in proportions and frequencies.
  *Biometrics* 11:375-386.
* Fisher RA (1922) On the interpretation of chi-square from contingency tables, and the
  calculation of P. *J R Stat Soc* 85:87-94; Fisher RA (1935) *The Design of Experiments*.
* Freeman GH, Halton JH (1951) Note on an exact treatment of contingency, goodness of
  fit and other problems of significance. *Biometrika* 38:141-149 (r x c exact test).
* Cramér H (1946) *Mathematical Methods of Statistics*; Bergsma W (2013) A bias-correction
  for Cramér's V and Tschuprow's T. *J Korean Stat Soc* 42:323-328.
* Woolf B (1955) On estimating the relation between blood group and disease.
  *Ann Hum Genet* 19:251-253 (logit CI of the odds ratio).
* Haldane JBS (1956) *Ann Hum Genet* 20:309-311; Anscombe FJ (1956) *Biometrika*
  43:461-464 (adding 0.5 to each cell when a cell is zero).
* Katz D, Baptista J, Azen SP, Pike MC (1978) Obtaining confidence intervals for the
  risk ratio in cohort studies. *Biometrics* 34:469-474.
* Wilson EB (1927) Probable inference, the law of succession, and statistical
  inference. *J Am Stat Assoc* 22:209-212 (score interval for a proportion).
* Newcombe RG (1998) Interval estimation for the difference between independent
  proportions: comparison of eleven methods. *Stat Med* 17:873-890 (method 10).
* Altman DG (1998) Confidence intervals for the number needed to treat.
  *BMJ* 317:1309-1312.
"""

from __future__ import annotations

import math
import re
import warnings as _warnings
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
from scipy import optimize, stats

from ._utils import (
    StatsError,
    alternative_label,
    check_alpha,
    check_alternative,
    ci_phrase,
    fmt,
    p_phrase,
    sig_phrase,
)
from .effect_sizes import magnitude_r
from .registry import ParamSpec, register
from .result import AssumptionCheck, EffectSize, Parameter, StatResult

ORIENTATION = (
    "Table orientation: rows are the groups (first row = exposed / treated), columns are the "
    "outcome (first column = outcome present)."
)

CITATIONS = {
    "pearson": "Pearson K (1900) Philos Mag 50:157-175",
    "yates": "Yates F (1934) J R Stat Soc Suppl 1:217-235",
    "cochran": "Cochran WG (1954) Biometrics 10:417-451",
    "armitage": "Armitage P (1955) Biometrics 11:375-386",
    "fisher": "Fisher RA (1922) J R Stat Soc 85:87-94",
    "freeman": "Freeman GH, Halton JH (1951) Biometrika 38:141-149",
    "bergsma": "Bergsma W (2013) J Korean Stat Soc 42:323-328",
    "woolf": "Woolf B (1955) Ann Hum Genet 19:251-253",
    "katz": "Katz D, Baptista J, Azen SP, Pike MC (1978) Biometrics 34:469-474",
    "wilson": "Wilson EB (1927) J Am Stat Assoc 22:209-212",
    "newcombe": "Newcombe RG (1998) Stat Med 17:873-890",
    "altman": "Altman DG (1998) BMJ 317:1309-1312",
}

#: Largest number of tables enumerated for the exact r x c test before falling back to
#: Monte Carlo sampling.
MAX_EXACT_TABLES = 1_000_000
#: Largest total count for which the exact CI of the conditional odds ratio is computed.
EXACT_CI_MAX_N = 5_000
MONTE_CARLO_RESAMPLES = 100_000


# ============================================================================ input
def _finite_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    v = float(value)
    return v if math.isfinite(v) else None


def _check_level(ci_level: float) -> float:
    if not (0 < float(ci_level) < 1):
        raise StatsError("ci_level must be between 0 and 1 (typically 0.95).")
    return float(ci_level)


def _validate_counts(
    counts: Any,
    row_names: Iterable[Any] | None,
    column_names: Iterable[Any] | None,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Return (float array of whole-number counts, row names, column names)."""
    if isinstance(counts, pd.DataFrame):
        if row_names is None:
            row_names = [str(v) for v in counts.index]
        if column_names is None:
            column_names = [str(v) for v in counts.columns]
        counts = counts.to_numpy()
    try:
        arr = np.asarray(counts, dtype=float)
    except (TypeError, ValueError):
        raise StatsError(
            "A contingency table must contain numbers only (counts of subjects in each cell)."
        ) from None
    if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 2:
        raise StatsError(
            "A contingency table needs at least 2 rows and 2 columns of counts "
            f"(got shape {tuple(arr.shape)})."
        )
    if not np.all(np.isfinite(arr)):
        raise StatsError("The table has missing or non-finite cells. Every cell needs a count (0 is allowed).")
    if (arr < 0).any():
        raise StatsError("Counts cannot be negative. Each cell is the number of subjects in that category.")
    if np.any(np.abs(arr - np.round(arr)) > 1e-9):
        raise StatsError(
            "Counts must be whole numbers of subjects. Percentages, proportions, means or "
            "normalised values are not valid input for this test: enter the actual number of "
            "subjects in each cell."
        )
    arr = np.round(arr)
    rows = [f"Row {i + 1}" for i in range(arr.shape[0])] if row_names is None else [str(v) for v in row_names]
    cols = ([f"Column {j + 1}" for j in range(arr.shape[1])] if column_names is None
            else [str(v) for v in column_names])
    if len(rows) != arr.shape[0] or len(cols) != arr.shape[1]:
        raise StatsError("The number of row / column names does not match the shape of the table.")
    if arr.sum() <= 0:
        raise StatsError("The table is empty (all counts are zero).")
    empty_rows = [rows[i] for i in np.flatnonzero(arr.sum(axis=1) == 0)]
    empty_cols = [cols[j] for j in np.flatnonzero(arr.sum(axis=0) == 0)]
    if empty_rows or empty_cols:
        parts = []
        if empty_rows:
            parts.append(f"row(s) {', '.join(empty_rows)}")
        if empty_cols:
            parts.append(f"column(s) {', '.join(empty_cols)}")
        raise StatsError(
            f"No subjects at all in {' and '.join(parts)} (total = 0), so the test cannot be "
            "computed. Remove the empty row / column and run again."
        )
    return arr, rows, cols


# ================================================================ interval helpers
def wilson_interval(successes: float, n: float, level: float = 0.95) -> tuple[float, float]:
    """Wilson (1927) score interval for a binomial proportion ``successes / n``."""
    if n <= 0:
        return (math.nan, math.nan)
    z = float(stats.norm.ppf(0.5 + level / 2.0))
    p = successes / n
    z2 = z * z
    centre = p + z2 / (2.0 * n)
    half = z * math.sqrt(max(p * (1.0 - p) / n + z2 / (4.0 * n * n), 0.0))
    denom = 1.0 + z2 / n
    low = 0.0 if successes <= 0 else max(0.0, (centre - half) / denom)
    high = 1.0 if successes >= n else min(1.0, (centre + half) / denom)
    return (low, high)


def risk_difference_newcombe(
    x1: float, n1: float, x2: float, n2: float, level: float = 0.95
) -> tuple[float, float, float]:
    """Risk difference ``x1/n1 - x2/n2`` with Newcombe's (1998) hybrid score interval.

    Method 10 of Newcombe (1998), without continuity correction: with Wilson limits
    ``(l1, u1)`` and ``(l2, u2)`` of the two proportions,
    ``lower = d - sqrt((p1 - l1)^2 + (u2 - p2)^2)``,
    ``upper = d + sqrt((u1 - p1)^2 + (p2 - l2)^2)``.
    """
    p1, p2 = x1 / n1, x2 / n2
    l1, u1 = wilson_interval(x1, n1, level)
    l2, u2 = wilson_interval(x2, n2, level)
    diff = p1 - p2
    low = diff - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    high = diff + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return (diff, max(low, -1.0), min(high, 1.0))


def _ncx2_ncp_limits(chi2: float, df: float, level: float) -> tuple[float, float]:
    """Noncentrality parameters that put ``chi2`` at the upper / lower tail of the
    noncentral chi-square (lower limit 0 when the test is not significant at that level)."""
    if not math.isfinite(chi2) or chi2 < 0 or df <= 0:
        return (math.nan, math.nan)
    alpha = 1.0 - level

    def cdf(ncp: float) -> float:
        if ncp <= 0:
            return float(stats.chi2.cdf(chi2, df))
        return float(stats.ncx2.cdf(chi2, df, ncp))

    def solve(target: float) -> float:
        if cdf(0.0) <= target:
            return 0.0
        hi = max(10.0, 3.0 * chi2 + 10.0)
        tries = 0
        while cdf(hi) > target and tries < 60:
            hi *= 2.0
            tries += 1
        if cdf(hi) > target:
            return math.nan
        try:
            return float(optimize.brentq(lambda v: cdf(v) - target, 0.0, hi, xtol=1e-10, maxiter=300))
        except (ValueError, RuntimeError):
            return math.nan

    return solve(1.0 - alpha / 2.0), solve(alpha / 2.0)


# ================================================================== 2x2 measures
def two_by_two_measures(
    counts: Any, *, ci_level: float = 0.95, diagnostic: bool = False
) -> dict[str, Any]:
    """Odds ratio, relative risk, risk difference and NNT of a 2x2 table, with CIs.

    ``counts = [[a, b], [c, d]]`` in the orientation described in the module docstring.

    * **Odds ratio** ``ad / bc`` with the Woolf (1955) logit interval
      ``exp(ln OR +/- z sqrt(1/a + 1/b + 1/c + 1/d))``. If any cell is zero, 0.5 is added
      to all four cells first (Haldane-Anscombe) and ``corrected`` is set.
    * **Relative risk** ``risk1 / risk2`` with the Katz et al. (1978) log interval,
      ``SE(ln RR) = sqrt(1/a - 1/(a+b) + 1/c - 1/(c+d))``.
    * **Risk difference** ``risk1 - risk2`` with the Newcombe (1998) hybrid score
      interval (method 10); the simple Wald interval is returned for comparison.
    * **NNT** ``1 / |risk difference|``; its CI is the reciprocal of the risk-difference
      limits and is only a single interval when that CI excludes zero (Altman 1998).
    * ``diagnostic=True``: sensitivity ``a/(a+c)``, specificity ``d/(b+d)``, positive and
      negative predictive values ``a/(a+b)``, ``d/(c+d)`` with Wilson (1927) intervals
      (rows = test positive / negative, columns = condition present / absent).

    Returns a dict of dicts (``odds_ratio, relative_risk, risk_difference, nnt`` and,
    if requested, ``diagnostic``) plus ``notes``. Undefined quantities are ``None``.
    """
    level = _check_level(ci_level)
    arr, _, _ = _validate_counts(counts, None, None)
    if arr.shape != (2, 2):
        raise StatsError("Odds ratio, relative risk and risk difference need a 2x2 table.")
    return _two_by_two(arr, level, diagnostic)


def _two_by_two(arr: np.ndarray, level: float, diagnostic: bool) -> dict[str, Any]:
    a, b = float(arr[0, 0]), float(arr[0, 1])
    c, d = float(arr[1, 0]), float(arr[1, 1])
    n1, n2 = a + b, c + d
    z = float(stats.norm.ppf(0.5 + level / 2.0))
    notes: list[str] = []
    risk1, risk2 = a / n1, c / n2

    # ---- odds ratio (Woolf), Haldane-Anscombe when a cell is zero
    corrected = bool(min(a, b, c, d) == 0)
    if b * c > 0:
        raw_or: float | None = (a * d) / (b * c)
    else:
        raw_or = None  # infinite or 0/0
    aa, bb, cc, dd = (a + 0.5, b + 0.5, c + 0.5, d + 0.5) if corrected else (a, b, c, d)
    log_or = math.log(aa) + math.log(dd) - math.log(bb) - math.log(cc)
    se_or = math.sqrt(1.0 / aa + 1.0 / bb + 1.0 / cc + 1.0 / dd)
    odds = {
        "value": math.exp(log_or),
        "ci_low": math.exp(log_or - z * se_or),
        "ci_high": math.exp(log_or + z * se_or),
        "log_value": log_or,
        "se_log": se_or,
        "corrected": corrected,
        "uncorrected": raw_or,
        "method": "Woolf logit interval" + (" after adding 0.5 to every cell (Haldane-Anscombe)" if corrected else ""),
    }
    if corrected:
        plain = "0" if raw_or == 0 else "infinite or undefined"
        notes.append(
            f"A cell is zero, so the plain odds ratio is {plain}. The reported odds ratio and its "
            "CI were computed after adding 0.5 to every cell (Haldane-Anscombe correction); "
            "treat them as rough."
        )

    # ---- relative risk (Katz)
    rr: dict[str, Any] = {"value": None, "ci_low": None, "ci_high": None, "log_value": None,
                          "se_log": None, "method": "Katz log interval"}
    if a > 0 and c > 0:
        log_rr = math.log(risk1) - math.log(risk2)
        se_rr = math.sqrt(max(1.0 / a - 1.0 / n1 + 1.0 / c - 1.0 / n2, 0.0))
        rr.update(value=math.exp(log_rr), ci_low=math.exp(log_rr - z * se_rr),
                  ci_high=math.exp(log_rr + z * se_rr), log_value=log_rr, se_log=se_rr)
    elif c > 0:
        rr.update(value=0.0)
        notes.append("The relative risk is 0 (no outcomes in the first row); its log-based CI is undefined.")
    else:
        notes.append(
            "The relative risk is undefined (infinite) because the outcome never occurred in the "
            "second row."
        )

    # ---- risk difference (Newcombe hybrid score) + Wald for comparison
    diff, rd_low, rd_high = risk_difference_newcombe(a, n1, c, n2, level)
    se_wald = math.sqrt(risk1 * (1.0 - risk1) / n1 + risk2 * (1.0 - risk2) / n2)
    rd = {
        "value": diff, "ci_low": rd_low, "ci_high": rd_high,
        "method": "Newcombe hybrid score interval (method 10)",
        "wald_se": se_wald, "wald_ci_low": diff - z * se_wald, "wald_ci_high": diff + z * se_wald,
    }

    # ---- number needed to treat
    nnt: dict[str, Any] = {"value": None, "ci_low": None, "ci_high": None, "ci_crosses_zero": None,
                           "bound_same_direction": None, "bound_opposite_direction": None}
    if diff != 0:
        nnt["value"] = 1.0 / abs(diff)
        crosses = rd_low < 0 < rd_high
        nnt["ci_crosses_zero"] = bool(crosses)
        if not crosses and rd_low != 0 and rd_high != 0:
            limits = sorted((1.0 / abs(rd_low), 1.0 / abs(rd_high)))
            nnt["ci_low"], nnt["ci_high"] = limits
        elif crosses:
            same, opposite = (rd_high, rd_low) if diff > 0 else (rd_low, rd_high)
            nnt["bound_same_direction"] = 1.0 / abs(same)
            nnt["bound_opposite_direction"] = 1.0 / abs(opposite)
            notes.append(
                "The confidence interval of the risk difference includes zero, so the NNT has no "
                f"single interval: it runs from {fmt(1.0 / abs(same))} in the observed direction, "
                f"through infinity (no difference), to {fmt(1.0 / abs(opposite))} in the opposite "
                "direction."
            )
    else:
        notes.append("The risks are identical, so the number needed to treat is infinite (not defined).")

    out: dict[str, Any] = {
        "risk1": risk1, "risk2": risk2,
        "risk1_ci": list(wilson_interval(a, n1, level)), "risk2_ci": list(wilson_interval(c, n2, level)),
        "odds_ratio": odds, "relative_risk": rr, "risk_difference": rd, "nnt": nnt,
        "ci_level": level, "notes": notes,
    }
    if diagnostic:
        diag = {}
        for name, num, den in (("sensitivity", a, a + c), ("specificity", d, b + d),
                               ("positive predictive value", a, a + b),
                               ("negative predictive value", d, c + d)):
            low, high = wilson_interval(num, den, level)
            diag[name] = {"value": num / den if den > 0 else None,
                          "ci_low": _finite_or_none(low), "ci_high": _finite_or_none(high),
                          "numerator": num, "denominator": den}
        out["diagnostic"] = diag
    return out


def _two_by_two_parameters(
    measures: dict[str, Any], rows: list[str], cols: list[str], level: float
) -> list[Parameter]:
    r1, r2 = rows
    c1 = cols[0]
    odds, rr, rd, nnt = (measures[k] for k in ("odds_ratio", "relative_risk", "risk_difference", "nnt"))
    params = [
        Parameter(name=f"Risk of {c1} in {r1}", value=measures["risk1"], ci_low=measures["risk1_ci"][0],
                  ci_high=measures["risk1_ci"][1], ci_level=level, unit_note="Proportion; Wilson score interval."),
        Parameter(name=f"Risk of {c1} in {r2}", value=measures["risk2"], ci_low=measures["risk2_ci"][0],
                  ci_high=measures["risk2_ci"][1], ci_level=level, unit_note="Proportion; Wilson score interval."),
        Parameter(name="Odds ratio", value=odds["value"], ci_low=odds["ci_low"], ci_high=odds["ci_high"],
                  ci_level=level,
                  unit_note=f"Odds of {c1} in {r1} relative to {r2}; {odds['method']}."),
        Parameter(name="Relative risk", value=rr["value"], ci_low=rr["ci_low"], ci_high=rr["ci_high"],
                  ci_level=level,
                  unit_note=f"Risk of {c1} in {r1} divided by the risk in {r2}; {rr['method']}."),
        Parameter(name="Risk difference", value=rd["value"], ci_low=rd["ci_low"], ci_high=rd["ci_high"],
                  ci_level=level,
                  unit_note=f"Risk of {c1} in {r1} minus the risk in {r2} (attributable risk), as a "
                            f"proportion; {rd['method']}."),
        Parameter(name="Number needed to treat (NNT)", value=nnt["value"], ci_low=nnt["ci_low"],
                  ci_high=nnt["ci_high"], ci_level=level,
                  unit_note="1 / |risk difference|: subjects in one group rather than the other per "
                            f"one additional (or one fewer) {c1}."
                            + (" The CI includes 'no difference' (infinity), see notes."
                               if nnt["ci_crosses_zero"] else "")),
    ]
    for name, item in measures.get("diagnostic", {}).items():
        params.append(Parameter(
            name=name[0].upper() + name[1:], value=item["value"], ci_low=item["ci_low"],
            ci_high=item["ci_high"], ci_level=level, unit_note="Wilson score interval.",
        ))
    return params


def _two_by_two_sentence(measures: dict[str, Any], arr: np.ndarray, rows: list[str], cols: list[str],
                         level: float) -> str:
    a, c = arr[0, 0], arr[1, 0]
    n1, n2 = arr[0].sum(), arr[1].sum()
    odds, rr, rd = measures["odds_ratio"], measures["relative_risk"], measures["risk_difference"]
    text = (
        f"{cols[0]} occurred in {a:.0f} of {n1:.0f} ({fmt(measures['risk1'] * 100, 3)}%) in {rows[0]} "
        f"and {c:.0f} of {n2:.0f} ({fmt(measures['risk2'] * 100, 3)}%) in {rows[1]}: risk difference "
        f"{fmt(rd['value'] * 100, 3)} percentage points ({level * 100:g}% CI {fmt(rd['ci_low'] * 100, 3)} "
        f"to {fmt(rd['ci_high'] * 100, 3)})"
    )
    if rr["value"] is not None and rr["ci_low"] is not None:
        text += f", relative risk {fmt(rr['value'])} ({ci_phrase(rr['ci_low'], rr['ci_high'], level)})"
    text += (f", odds ratio {fmt(odds['value'])} ({ci_phrase(odds['ci_low'], odds['ci_high'], level)}"
             + ("; 0.5 added to each cell" if odds["corrected"] else "") + f"), all for {rows[0]} relative to {rows[1]}.")
    return text


def _diagnostic_note(rows: list[str], cols: list[str]) -> str:
    return (
        f"Diagnostic measures read the table as: rows = test result ('{rows[0]}' = positive, "
        f"'{rows[1]}' = negative), columns = true condition ('{cols[0]}' = present, '{cols[1]}' = "
        "absent). Predictive values apply only if the table reflects the real prevalence."
    )


def _descriptive_rows(arr: np.ndarray, rows: list[str], cols: list[str]) -> list[dict[str, Any]]:
    out = []
    for i, name in enumerate(rows):
        total = float(arr[i].sum())
        row: dict[str, Any] = {"group": name, "n": int(total)}
        for j, col in enumerate(cols):
            row[col] = int(arr[i, j])
            row[f"% {col}"] = float(arr[i, j] / total * 100.0)
        out.append(row)
    return out


def _base_extra(arr: np.ndarray, rows: list[str], cols: list[str], row_label: str, column_label: str) -> dict[str, Any]:
    return {
        "observed": arr.tolist(),
        "row_names": rows,
        "column_names": cols,
        "row_label": row_label,
        "column_label": column_label,
        "row_totals": arr.sum(axis=1).tolist(),
        "column_totals": arr.sum(axis=0).tolist(),
        "orientation": ORIENTATION,
    }


def _counts_assumptions() -> list[AssumptionCheck]:
    return [
        AssumptionCheck(
            name="Independent observations", status="info",
            detail="Each subject must be counted once, in exactly one cell. Paired or repeated "
                   "measurements on the same subjects need McNemar's test instead.",
        ),
        AssumptionCheck(
            name="Cells are counts", status="pass",
            detail="All cells are whole numbers. They must be actual counts of subjects, not "
                   "percentages or averages.",
        ),
    ]


# =================================================================== chi-square
def _pearson(arr: np.ndarray) -> tuple[float, np.ndarray]:
    n = arr.sum()
    expected = np.outer(arr.sum(axis=1), arr.sum(axis=0)) / n
    return float(((arr - expected) ** 2 / expected).sum()), expected


def chi_square_test(
    counts: Any,
    *,
    row_names: Iterable[Any] | None = None,
    column_names: Iterable[Any] | None = None,
    row_label: str = "Group",
    column_label: str = "Outcome",
    yates: bool | None = None,
    alpha: float = 0.05,
    ci_level: float = 0.95,
    diagnostic: bool = False,
) -> StatResult:
    """Pearson's chi-square test of independence for an r x c table of counts.

    ``chi2 = sum (O - E)^2 / E`` with ``E = row total * column total / N`` and
    ``(r - 1)(c - 1)`` degrees of freedom (Pearson 1900).

    ``yates``: Yates' (1934) continuity correction, which only exists for 2x2 tables:
    each ``|O - E|`` is reduced by 0.5 (never past zero) before squaring. ``None``
    (default) applies it to 2x2 tables, as is conventional, and not otherwise; it makes
    the test conservative. The uncorrected statistic is always in ``extra``.

    Also reported: expected counts and Pearson / adjusted standardized residuals
    (``extra``), Cochran's (1954) expected-count rule as an assumption check, phi (2x2)
    and Cramér's V (plus Bergsma's 2013 bias-corrected V) as effect sizes, and for 2x2
    tables the measures of :func:`two_by_two_measures` as ``parameters``.
    """
    alpha = check_alpha(alpha)
    level = _check_level(ci_level)
    arr, rows, cols = _validate_counts(counts, row_names, column_names)
    r, c = arr.shape
    n = float(arr.sum())
    is_2x2 = (r, c) == (2, 2)
    warnings: list[str] = []
    notes: list[str] = []
    if yates is None:
        use_yates = is_2x2
    else:
        use_yates = bool(yates)
        if use_yates and not is_2x2:
            use_yates = False
            notes.append("Yates' continuity correction exists only for 2x2 tables and was not applied.")

    chi2_plain, expected = _pearson(arr)
    df = (r - 1) * (c - 1)
    if use_yates:
        shrink = np.minimum(0.5, np.abs(arr - expected))
        chi2 = float(((np.abs(arr - expected) - shrink) ** 2 / expected).sum())
    else:
        chi2 = chi2_plain
    p_value = float(stats.chi2.sf(chi2, df))
    p_plain = float(stats.chi2.sf(chi2_plain, df))

    row_frac = arr.sum(axis=1, keepdims=True) / n
    col_frac = arr.sum(axis=0, keepdims=True) / n
    pearson_resid = (arr - expected) / np.sqrt(expected)
    with np.errstate(divide="ignore", invalid="ignore"):
        adjusted = pearson_resid / np.sqrt((1.0 - row_frac) * (1.0 - col_frac))

    # ---- Cochran's rule
    small5 = int((expected < 5).sum())
    frac5 = small5 / expected.size
    min_expected = float(expected.min())
    alt = "Fisher's exact test" if is_2x2 else "Fisher's exact test (or merge sparse categories)"
    if min_expected < 1 or frac5 > 0.2:
        status = "fail"
        detail = (f"{small5} of {expected.size} expected counts are below 5 (smallest "
                  f"{fmt(min_expected, 3)}). The chi-square approximation is unreliable here: "
                  f"use {alt}.")
        warnings.append(
            f"Expected counts are too small for the chi-square approximation (smallest expected "
            f"count {fmt(min_expected, 3)}; {small5} of {expected.size} cells below 5). Use {alt}."
        )
    elif small5:
        status = "warn"
        detail = (f"{small5} of {expected.size} expected counts are below 5 (smallest "
                  f"{fmt(min_expected, 3)}), within Cochran's rule (no more than 20% below 5, none "
                  "below 1), but the P value is approximate.")
    else:
        status = "pass"
        detail = f"All expected counts are at least 5 (smallest {fmt(min_expected, 3)})."
    assumptions = [
        AssumptionCheck(name="Expected counts large enough (Cochran's rule)", statistic=min_expected,
                        status=status, detail=detail),
        *_counts_assumptions(),
    ]

    # ---- effect sizes
    m = min(r - 1, c - 1)
    v = math.sqrt(chi2_plain / (n * m))
    ncp_lo, ncp_hi = _ncx2_ncp_limits(chi2_plain, df, level)
    v_lo = math.sqrt(ncp_lo / (n * m)) if math.isfinite(ncp_lo) else math.nan
    v_hi = min(1.0, math.sqrt(ncp_hi / (n * m))) if math.isfinite(ncp_hi) else math.nan
    phi2_adj = max(0.0, chi2_plain / n - df / (n - 1.0)) if n > 1 else math.nan
    r_adj = r - (r - 1) ** 2 / (n - 1.0) if n > 1 else math.nan
    c_adj = c - (c - 1) ** 2 / (n - 1.0) if n > 1 else math.nan
    m_adj = min(r_adj - 1.0, c_adj - 1.0) if n > 1 else math.nan
    v_bc = math.sqrt(phi2_adj / m_adj) if n > 1 and m_adj > 0 else math.nan
    effect_sizes: list[EffectSize] = []
    if is_2x2:
        a, b, cc, d = arr.ravel()
        phi = float((a * d - b * cc) / math.sqrt(arr[0].sum() * arr[1].sum() * arr[:, 0].sum() * arr[:, 1].sum()))
        effect_sizes.append(EffectSize(name="phi", value=phi, ci_level=level, magnitude=magnitude_r(phi)))
    effect_sizes.append(EffectSize(
        name="Cramér's V", value=v, ci_low=_finite_or_none(v_lo), ci_high=_finite_or_none(v_hi),
        ci_level=level, ci_method="noncentral chi-square (approximate)",
        magnitude=magnitude_r(v * math.sqrt(m)),
    ))
    effect_sizes.append(EffectSize(
        name="Cramér's V (bias-corrected)", value=_finite_or_none(v_bc), ci_level=level,
        ci_method=None, magnitude=magnitude_r(v_bc * math.sqrt(m)) if math.isfinite(v_bc) else None,
    ))

    extra = _base_extra(arr, rows, cols, row_label, column_label)
    extra.update({
        "expected": expected.tolist(),
        "pearson_residuals": pearson_resid.tolist(),
        "adjusted_residuals": [[_finite_or_none(x) for x in row] for row in adjusted.tolist()],
        "chi2_contributions": ((arr - expected) ** 2 / expected).tolist(),
        "chi2_uncorrected": chi2_plain,
        "p_uncorrected": p_plain,
        "yates": use_yates,
    })

    parameters: list[Parameter] = []
    detail_text = ""
    citations = [CITATIONS["pearson"], CITATIONS["cochran"], CITATIONS["bergsma"]]
    if use_yates:
        citations.insert(1, CITATIONS["yates"])
    if is_2x2:
        measures = _two_by_two(arr, level, diagnostic)
        parameters = _two_by_two_parameters(measures, rows, cols, level)
        warnings.extend(measures["notes"])
        extra["measures"] = {k: val for k, val in measures.items() if k != "notes"}
        detail_text = " " + _two_by_two_sentence(measures, arr, rows, cols, level)
        notes.append(ORIENTATION + f" Here: rows '{rows[0]}' vs '{rows[1]}'; first column '{cols[0]}'.")
        if diagnostic:
            notes.append(_diagnostic_note(rows, cols))
            citations.append(CITATIONS["wilson"])
        citations += [CITATIONS["woolf"], CITATIONS["katz"], CITATIONS["newcombe"], CITATIONS["altman"]]
    elif diagnostic:
        notes.append("Sensitivity / specificity need a 2x2 table and were not computed.")

    corr_words = " with Yates' continuity correction" if use_yates else ""
    largest = ""
    if not is_2x2 and np.all(np.isfinite(adjusted)):
        i, j = np.unravel_index(int(np.argmax(np.abs(adjusted))), adjusted.shape)
        word = "more" if adjusted[i, j] > 0 else "fewer"
        largest = (f" The largest departure from independence is {rows[i]} / {cols[j]}: {word} "
                   f"subjects than expected ({arr[i, j]:.0f} observed, {fmt(expected[i, j], 3)} expected; "
                   f"adjusted residual {fmt(adjusted[i, j], 3)}).")
    interpretation = (
        f"The association between {row_label} ({', '.join(rows)}) and {column_label} "
        f"({', '.join(cols)}) in {n:.0f} subjects was {sig_phrase(p_value, alpha)} by the chi-square "
        f"test{corr_words} (chi-square = {fmt(chi2)}, df = {df}). The strength of association was "
        f"Cramér's V = {fmt(v, 3)}.{detail_text}{largest}"
    )
    if status == "fail":
        interpretation += f" Expected counts are small, so prefer {alt}."
    return StatResult(
        analysis="chi_square",
        test_name="Chi-square test of independence" + (" (Yates' continuity correction)" if use_yates else ""),
        why_chosen="The chi-square test asks whether two categorical variables are associated, i.e. "
                   "whether the proportions differ between groups, when every subject is counted in "
                   "exactly one cell and the expected counts are not small.",
        assumptions=assumptions,
        statistic_name="chi2",
        statistic=chi2,
        df=df,
        p_value=p_value,
        alpha=alpha,
        effect_sizes=effect_sizes,
        parameters=parameters,
        n={name: int(arr[i].sum()) for i, name in enumerate(rows)},
        descriptives=_descriptive_rows(arr, rows, cols),
        warnings=warnings,
        data_notes=notes,
        interpretation=interpretation,
        methods_text=(
            f"The association between {row_label} and {column_label} was tested with Pearson's "
            f"chi-square test{corr_words} (alpha = {alpha:g})"
            + ("; odds ratio (Woolf), relative risk (Katz) and risk difference (Newcombe hybrid score) "
               f"are given with {level * 100:g}% confidence intervals." if is_2x2 else ".")
        ),
        citations=citations,
        extra=extra,
    )


# ============================================================= chi-square for trend
def _trend_scores(scores: Any, names: list[str]) -> tuple[np.ndarray, bool]:
    """Scores of the ordered categories: (array, True when the user supplied them).

    ``scores`` may be ``None`` / empty text (equally spaced 1, 2, ..., k), a sequence of
    numbers, or text such as ``"0, 1, 5"`` (separated by commas, semicolons or spaces).
    """
    k = len(names)
    listing = ", ".join(names)
    if scores is None or (isinstance(scores, str) and not scores.strip()):
        return np.arange(1, k + 1, dtype=float), False
    if isinstance(scores, str):
        parts = [part for part in re.split(r"[,;\s]+", scores.strip()) if part]
    else:
        try:
            parts = list(scores)
        except TypeError:
            raise StatsError(
                "The category scores must be a list of numbers, for example '0, 1, 5'."
            ) from None
    values = []
    for part in parts:
        try:
            values.append(float(part))
        except (TypeError, ValueError):
            raise StatsError(
                f"The category score '{part}' is not a number. Enter one number per ordered category, "
                f"separated by commas, for example '0, 1, 5' (leave the box empty for equally spaced "
                "scores 1, 2, 3, ...)."
            ) from None
    if len(values) != k:
        raise StatsError(
            f"The trend test needs one numeric score for each of the {k} ordered categories ({listing}), "
            f"but {len(values)} score(s) were given. Enter exactly one number per category, in the same "
            "order, or leave the box empty for equally spaced scores."
        )
    arr = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise StatsError("The category scores must be ordinary finite numbers (no 'inf' or 'nan').")
    if np.ptp(arr) == 0:
        raise StatsError("All category scores are identical, so a trend cannot be tested. Give the "
                         "categories different scores, such as the doses themselves.")
    return arr, True


def chi_square_trend(
    counts: Any,
    *,
    scores: Any = None,
    row_names: Iterable[Any] | None = None,
    column_names: Iterable[Any] | None = None,
    row_label: str = "Group",
    column_label: str = "Outcome",
    alpha: float = 0.05,
) -> StatResult:
    """Chi-square test for linear trend in proportions (Cochran 1954; Armitage 1955).

    For a ``k x 2`` table (k ordered groups in rows, a yes/no outcome in columns) or a
    ``2 x k`` table (two groups, k ordered categories). With ordered-category scores
    ``s_i`` (default 1, 2, ..., k), category totals ``n_i``, counts ``x_i`` in the first
    level of the two-level variable, ``R = sum x_i`` and ``N = sum n_i``::

        chi2_trend = N (N sum x_i s_i - R sum n_i s_i)^2
                     / (R (N - R) (N sum n_i s_i^2 - (sum n_i s_i)^2))        (1 df)

    which equals ``N r^2`` for the correlation r between outcome and score. The
    remainder of the ordinary (uncorrected) chi-square, with k - 2 df, tests departure
    from a straight-line trend.

    ``scores``: one number per ordered category, as a sequence or as text such as
    ``"0, 1, 5"`` (for example the doses themselves when they are not evenly spaced).
    ``None`` or empty text means equally spaced scores 1, 2, ..., k. The statistic does not
    change when all scores are shifted or multiplied by a constant; only their relative
    spacing matters. Wrong counts of scores, non-numeric or identical scores raise
    ``StatsError``.
    """
    alpha = check_alpha(alpha)
    arr, rows, cols = _validate_counts(counts, row_names, column_names)
    r, c = arr.shape
    if c == 2:
        ordered_names, binary_names = rows, cols
        x, totals = arr[:, 0], arr.sum(axis=1)
        ordered_label, binary_label = row_label, column_label
    elif r == 2:
        ordered_names, binary_names = cols, rows
        x, totals = arr[0, :], arr.sum(axis=0)
        ordered_label, binary_label = column_label, row_label
    else:
        raise StatsError(
            "The chi-square test for trend needs a table with exactly two columns (ordered groups "
            "in rows) or exactly two rows (ordered categories in columns)."
        )
    k = len(ordered_names)
    s, custom_scores = _trend_scores(scores, [str(v) for v in ordered_names])
    n = float(totals.sum())
    big_r = float(x.sum())
    num = n * (n * float((x * s).sum()) - big_r * float((totals * s).sum())) ** 2
    spread = n * float((totals * s * s).sum()) - float((totals * s).sum()) ** 2
    if spread <= 0:
        raise StatsError("All category scores are identical, so a trend cannot be tested.")
    chi2 = num / (big_r * (n - big_r) * spread)
    p_value = float(stats.chi2.sf(chi2, 1))
    s_bar = float((totals * s).sum()) / n
    p_bar = big_r / n
    props = x / totals
    slope = float((totals * (s - s_bar) * (props - p_bar)).sum() / (totals * (s - s_bar) ** 2).sum())
    corr = math.copysign(math.sqrt(chi2 / n), slope) if slope != 0 else 0.0
    chi2_total, expected = _pearson(arr)
    warnings: list[str] = []
    steps = np.diff(s)
    if custom_scores and not (np.all(steps > 0) or np.all(steps < 0)):
        warnings.append(
            "The scores you entered are not in steadily increasing (or decreasing) order across the "
            f"categories as listed ({', '.join(f'{name}: {fmt(v)}' for name, v in zip(ordered_names, s))}). "
            "The trend is tested along the scores, not along the order of the table; check that each "
            "score sits against the right category."
        )
    departure = None
    if k > 2:
        dep = max(chi2_total - chi2, 0.0)
        departure = {"statistic": dep, "df": k - 2, "p_value": float(stats.chi2.sf(dep, k - 2))}
        if departure["p_value"] < 0.05:
            warnings.append(
                f"The proportions depart from a straight-line trend (chi-square = {fmt(dep)}, "
                f"df = {k - 2}, {p_phrase(departure['p_value'])}); a linear trend may "
                "not describe them well."
            )
    small = int((expected < 5).sum())
    extra = _base_extra(arr, rows, cols, row_label, column_label)
    extra.update({
        "scores": s.tolist(), "scores_source": "entered by the user" if custom_scores else "equally spaced (default)",
        "ordered_categories": ordered_names,
        "proportions": props.tolist(), "proportion_of": binary_names[0],
        "chi2_total": chi2_total, "df_total": k - 1, "departure_from_trend": departure,
        "expected": expected.tolist(), "z": math.copysign(math.sqrt(chi2), slope) if slope != 0 else 0.0,
    })
    direction = "increased" if slope > 0 else "decreased" if slope < 0 else "did not change"
    prop_text = ", ".join(f"{name}: {fmt(pv * 100, 3)}%" for name, pv in zip(ordered_names, props))
    return StatResult(
        analysis="chi_square_trend",
        test_name="Chi-square test for trend (Cochran-Armitage)",
        why_chosen="When one variable has two outcomes and the other has ordered categories (dose "
                   "levels, stages, age bands), this test asks whether the proportion rises or falls "
                   "steadily across the order. It is more powerful than the ordinary chi-square test "
                   "for that question.",
        assumptions=[
            AssumptionCheck(
                name="Expected counts large enough", status="warn" if small else "pass",
                statistic=float(expected.min()),
                detail=(f"{small} expected count(s) are below 5; the P value is approximate."
                        if small else "All expected counts are at least 5."),
            ),
            AssumptionCheck(
                name="Ordered categories with meaningful scores" if custom_scores
                else "Ordered, evenly spaced categories",
                status="info",
                detail=(
                    f"The categories were given the scores you entered ({', '.join(fmt(v) for v in s)}, in "
                    "the order of the table). The test assumes these scores reflect the real spacing "
                    "between the categories."
                    if custom_scores else
                    f"The categories were scored {', '.join(fmt(v) for v in s)} in the order given "
                    "(equally spaced). The test assumes these scores reflect the real spacing; if the "
                    "categories are not evenly spaced (for example doses 0, 1 and 5), enter the real "
                    "values as category scores."
                ),
            ),
            *_counts_assumptions(),
        ],
        statistic_name="chi2",
        statistic=chi2,
        df=1,
        p_value=p_value,
        alpha=alpha,
        effect_sizes=[EffectSize(name="r (linear trend)", value=corr, magnitude=magnitude_r(corr))],
        parameters=[Parameter(
            name="Slope", value=slope,
            unit_note=f"Change in the proportion of {binary_names[0]} per "
                      + ("one-unit increase in the score you entered." if custom_scores
                         else "one-step increase in score."),
        )],
        n={name: int(totals[i]) for i, name in enumerate(ordered_names)},
        descriptives=_descriptive_rows(arr, rows, cols),
        warnings=warnings,
        data_notes=[f"Trend tested across {ordered_label} ({' < '.join(ordered_names)}); proportion of "
                    f"{binary_label} = {binary_names[0]}."]
        + ([f"Category scores entered by the user: "
            f"{', '.join(f'{name} = {fmt(v)}' for name, v in zip(ordered_names, s))}."] if custom_scores else []),
        interpretation=(
            f"The proportion of {binary_names[0]} {direction} across {ordered_label} ({prop_text}); the "
            f"linear trend was {sig_phrase(p_value, alpha)} (chi-square for trend = {fmt(chi2)}, df = 1; "
            f"slope {fmt(slope * 100, 3)} percentage points per "
            + (f"unit of the entered scores, {', '.join(fmt(v) for v in s)})." if custom_scores else "step).")
        ),
        methods_text=(
            f"A linear trend in proportions across {ordered_label} was tested with the Cochran-Armitage "
            f"chi-square test for trend ("
            + (f"category scores {', '.join(fmt(v) for v in s)}" if custom_scores
               else "equally spaced category scores")
            + f"; alpha = {alpha:g})."
        ),
        citations=[CITATIONS["cochran"], CITATIONS["armitage"]],
        extra=extra,
    )


# ============================================================== Fisher's exact test
class _TooManyTables(Exception):
    pass


def _rxc_exact_p(arr: np.ndarray, max_tables: int) -> tuple[float, int]:
    """Exact Fisher-Freeman-Halton P value by enumerating every table with the observed
    margins: the sum of the probabilities of all tables no more probable than the
    observed one. Raises :class:`_TooManyTables` beyond ``max_tables``."""
    table = arr.astype(int)
    if table.shape[0] > table.shape[1]:
        table = table.T
    n_rows, n_cols = table.shape
    row_tot = [int(v) for v in table.sum(axis=1)]
    col_tot = [int(v) for v in table.sum(axis=0)]
    n = sum(row_tot)
    lf = [0.0] * (n + 1)
    for i in range(2, n + 1):
        lf[i] = lf[i - 1] + math.log(i)
    const = sum(lf[v] for v in row_tot) + sum(lf[v] for v in col_tot) - lf[n]
    obs_log = const - sum(lf[int(v)] for v in table.ravel())
    threshold = obs_log + 1e-7  # relative tolerance for ties, as other implementations use
    remaining = row_tot[:]
    state = {"count": 0, "total": 0.0}

    def fill(col: int, row: int, left: int, acc: float) -> None:
        if col == n_cols - 1:
            # the last column is whatever remains in each row
            log_p = const + acc - sum(lf[v] for v in remaining)
            state["count"] += 1
            if state["count"] > max_tables:
                raise _TooManyTables
            if log_p <= threshold:
                state["total"] += math.exp(log_p)
            return
        if row == n_rows - 1:
            if left > remaining[row]:
                return
            remaining[row] -= left
            fill(col + 1, 0, col_tot[col + 1], acc - lf[left])
            remaining[row] += left
            return
        room_below = sum(remaining[row + 1:])
        lo = max(0, left - room_below)
        hi = min(remaining[row], left)
        for x in range(lo, hi + 1):
            remaining[row] -= x
            fill(col, row + 1, left - x, acc - lf[x])
            remaining[row] += x

    fill(0, 0, col_tot[0], 0.0)
    return min(1.0, state["total"]), state["count"]


def fisher_exact_test(
    counts: Any,
    *,
    row_names: Iterable[Any] | None = None,
    column_names: Iterable[Any] | None = None,
    row_label: str = "Group",
    column_label: str = "Outcome",
    alternative: str = "two-sided",
    alpha: float = 0.05,
    ci_level: float = 0.95,
    diagnostic: bool = False,
    max_tables: int = MAX_EXACT_TABLES,
    n_resamples: int = MONTE_CARLO_RESAMPLES,
    seed: int = 20240229,
) -> StatResult:
    """Fisher's exact test (Fisher 1922, 1935).

    **2x2 tables**: with all margins fixed, the top-left count follows a hypergeometric
    distribution under the null hypothesis. The two-sided P value is the total
    probability of all tables that are no more probable than the observed one (the
    standard method; computed by ``scipy.stats.fisher_exact``). One-sided alternatives:
    ``"greater"`` = the outcome in the first column is *more* likely in the first row
    than in the second (odds ratio > 1), ``"less"`` = less likely. The conditional
    maximum-likelihood odds ratio with its exact confidence interval is reported next to
    the usual 2x2 measures.

    **Larger tables** (Freeman & Halton 1951): the same definition, computed here by
    enumerating every table with the observed margins. If more than ``max_tables``
    tables would be needed, the P value is instead *estimated* from ``n_resamples``
    random tables (Monte Carlo, fixed ``seed`` so results are reproducible) and a
    warning says so. Only the two-sided test exists for larger tables.
    """
    alpha = check_alpha(alpha)
    level = _check_level(ci_level)
    alternative = check_alternative(alternative)
    arr, rows, cols = _validate_counts(counts, row_names, column_names)
    r, c = arr.shape
    n = float(arr.sum())
    is_2x2 = (r, c) == (2, 2)
    warnings: list[str] = []
    notes: list[str] = []
    extra = _base_extra(arr, rows, cols, row_label, column_label)
    _, expected = _pearson(arr)
    extra["expected"] = expected.tolist()
    parameters: list[Parameter] = []
    effect_sizes: list[EffectSize] = []
    citations = [CITATIONS["fisher"]]
    statistic_name = None
    statistic = None
    detail_text = ""
    method_words = "Fisher's exact test"

    if is_2x2:
        table = arr.astype(np.int64) if arr.max() < 2**62 else arr
        p_value = float(stats.fisher_exact(table, alternative=alternative).pvalue)
        p_value = min(max(p_value, 0.0), 1.0)
        measures = _two_by_two(arr, level, diagnostic)
        parameters = _two_by_two_parameters(measures, rows, cols, level)
        warnings.extend(measures["notes"])
        extra["measures"] = {k: val for k, val in measures.items() if k != "notes"}
        # conditional MLE of the odds ratio with its exact CI
        cond: dict[str, Any] = {"value": None, "ci_low": None, "ci_high": None}
        try:
            if n > 2**31 - 1:
                raise OverflowError
            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore")
                res = stats.contingency.odds_ratio(table, kind="conditional")
                cond["value"] = _finite_or_none(res.statistic)
                infinite = not math.isfinite(float(res.statistic))
                if n <= EXACT_CI_MAX_N:
                    ci = res.confidence_interval(confidence_level=level)
                    cond["ci_low"], cond["ci_high"] = _finite_or_none(ci.low), _finite_or_none(ci.high)
                    infinite = infinite or not math.isfinite(float(ci.high))
                else:
                    notes.append(
                        "The exact confidence interval of the conditional odds ratio is not computed "
                        f"for tables with more than {EXACT_CI_MAX_N:,} subjects (it would take very "
                        "long); with this many subjects the Woolf interval of the odds ratio is accurate."
                    )
            if infinite:
                notes.append(
                    "The conditional odds ratio or its upper confidence limit is infinite (a cell is "
                    "zero); infinite values are shown as empty."
                )
        except (ValueError, RuntimeError, OverflowError, FloatingPointError):
            notes.append(
                "The conditional (exact) odds ratio could not be computed for this table; use the "
                "ordinary odds ratio with its Woolf interval."
            )
        extra["conditional_odds_ratio"] = cond
        parameters.insert(3, Parameter(
            name="Odds ratio (conditional MLE)", value=cond["value"], ci_low=cond["ci_low"],
            ci_high=cond["ci_high"], ci_level=level,
            unit_note="Conditional maximum-likelihood estimate with the exact (Fisher) confidence "
                      "interval; an empty limit means infinity.",
        ))
        odds = measures["odds_ratio"]
        statistic_name = "odds ratio"
        statistic = odds["uncorrected"]
        effect_sizes.append(EffectSize(
            name="Odds ratio", value=odds["value"], ci_low=odds["ci_low"], ci_high=odds["ci_high"],
            ci_level=level, ci_method=odds["method"],
        ))
        a, b, cc, d = arr.ravel()
        phi = float((a * d - b * cc) / math.sqrt(arr[0].sum() * arr[1].sum() * arr[:, 0].sum() * arr[:, 1].sum()))
        effect_sizes.append(EffectSize(name="phi", value=phi, ci_level=level, magnitude=magnitude_r(phi)))
        detail_text = " " + _two_by_two_sentence(measures, arr, rows, cols, level)
        notes.append(ORIENTATION + f" Here: rows '{rows[0]}' vs '{rows[1]}'; first column '{cols[0]}'.")
        if alternative != "two-sided":
            word = "more" if alternative == "greater" else "less"
            notes.append(
                f"One-sided test: the alternative hypothesis is that '{cols[0]}' is {word} likely in "
                f"'{rows[0]}' than in '{rows[1]}'. A one-sided test is only valid if the direction "
                "was chosen before seeing the data."
            )
        if diagnostic:
            notes.append(_diagnostic_note(rows, cols))
            citations.append(CITATIONS["wilson"])
        citations += [CITATIONS["woolf"], CITATIONS["katz"], CITATIONS["newcombe"], CITATIONS["altman"]]
        extra["p_method"] = "exact"
    else:
        if alternative != "two-sided":
            raise StatsError(
                "One-sided Fisher's exact tests exist only for 2x2 tables. Use the two-sided test "
                "for larger tables."
            )
        if diagnostic:
            notes.append("Sensitivity / specificity need a 2x2 table and were not computed.")
        citations.append(CITATIONS["freeman"])
        method_words = "Fisher's exact test for r x c tables (Fisher-Freeman-Halton)"
        try:
            if n > 5_000_000:
                raise _TooManyTables
            p_value, n_tables = _rxc_exact_p(arr, int(max_tables))
            extra["p_method"] = "exact"
            extra["tables_enumerated"] = n_tables
        except (_TooManyTables, RecursionError):
            method = stats.MonteCarloMethod(n_resamples=int(n_resamples), rng=np.random.default_rng(seed))
            try:
                with _warnings.catch_warnings():
                    _warnings.simplefilter("ignore")
                    p_value = float(stats.fisher_exact(arr.astype(np.int64), method=method).pvalue)
            except (ValueError, MemoryError, OverflowError) as exc:
                raise StatsError(
                    "This table is too large for Fisher's exact test. Use the chi-square test, "
                    "which is accurate for large counts."
                ) from exc
            se = math.sqrt(max(p_value * (1.0 - p_value), 0.0) / n_resamples)
            extra["p_method"] = "monte_carlo"
            extra["monte_carlo"] = {"n_resamples": int(n_resamples), "seed": int(seed), "p_standard_error": se}
            warnings.append(
                f"The table is too large to enumerate exactly, so the P value is a Monte Carlo "
                f"estimate from {n_resamples:,} random tables with the same margins (standard error "
                f"about {fmt(se, 2)}). With counts this large the chi-square test is also appropriate."
            )
        m = min(r - 1, c - 1)
        chi2_plain, _ = _pearson(arr)
        v = math.sqrt(chi2_plain / (n * m))
        effect_sizes.append(EffectSize(name="Cramér's V", value=v, ci_level=level,
                                       magnitude=magnitude_r(v * math.sqrt(m))))

    tail = "" if alternative == "two-sided" else f" ({alternative_label(alternative)})"
    interpretation = (
        f"The association between {row_label} ({', '.join(rows)}) and {column_label} "
        f"({', '.join(cols)}) in {n:.0f} subjects was {sig_phrase(p_value, alpha)} by "
        f"{method_words}{tail}.{detail_text}"
    )
    if not is_2x2:
        interpretation += f" The strength of association was Cramér's V = {fmt(effect_sizes[0].value, 3)}."
    return StatResult(
        analysis="fisher_exact",
        test_name="Fisher's exact test" + ("" if is_2x2 else " (r x c, Fisher-Freeman-Halton)"),
        why_chosen="Fisher's exact test asks whether two categorical variables are associated without "
                   "relying on a large-sample approximation, so it is the right choice when counts "
                   "(or expected counts) are small. It is valid for any sample size.",
        assumptions=[
            AssumptionCheck(
                name="Fixed margins", status="info",
                detail="The exact P value is calculated as if the row and column totals were fixed in "
                       "advance. When they are not, the test is still valid but somewhat conservative.",
            ),
            *_counts_assumptions(),
        ],
        statistic_name=statistic_name,
        statistic=statistic,
        p_value=p_value,
        alpha=alpha,
        alternative=alternative,
        effect_sizes=effect_sizes,
        parameters=parameters,
        n={name: int(arr[i].sum()) for i, name in enumerate(rows)},
        descriptives=_descriptive_rows(arr, rows, cols),
        warnings=warnings,
        data_notes=notes,
        interpretation=interpretation,
        methods_text=(
            f"The association between {row_label} and {column_label} was tested with {method_words}"
            f"{tail or ' (two-sided)'} (alpha = {alpha:g})"
            + ("; odds ratio (Woolf), relative risk (Katz) and risk difference (Newcombe hybrid score) "
               f"are given with {level * 100:g}% confidence intervals." if is_2x2 else ".")
        ),
        citations=citations,
        extra=extra,
    )


# =================================================================== table adapters
def _table_inputs(table: Any) -> tuple[dict[str, Any], list[str]]:
    counts = table.counts
    notes = []
    blanks = int(table.data.iloc[:, 1:].isna().to_numpy().sum())
    if blanks:
        notes.append(f"{blanks} empty cell(s) in the table were counted as 0.")
    return {
        "counts": counts.to_numpy(dtype=float),
        "row_names": table.row_levels,
        "column_names": table.column_levels,
        "row_label": table.row_label,
        "column_label": table.column_label,
    }, notes


def _with_notes(result: StatResult, notes: list[str]) -> StatResult:
    result.data_notes = notes + result.data_notes
    return result


_ALPHA = ParamSpec("alpha", "Significance level (alpha)", kind="float", default=0.05,
                   help="The P value threshold for calling a result significant. 0.05 is the usual choice.")
_DIAGNOSTIC = ParamSpec(
    "diagnostic", "Diagnostic test measures", kind="bool", default=False,
    help="Tick this if the rows are a test result (positive first) and the columns the true condition "
         "(present first). Adds sensitivity, specificity and predictive values for 2x2 tables.",
)


@register(
    key="chi_square",
    label="Chi-square test",
    category="Contingency tables",
    tables=("contingency",),
    params=(
        ParamSpec("yates", "Yates' continuity correction", kind="choice", default="auto",
                  choices=("auto", "yes", "no"),
                  help="A small adjustment that makes the test more cautious for 2x2 tables. 'auto' "
                       "applies it to 2x2 tables only, which is the usual convention."),
        _ALPHA,
        _DIAGNOSTIC,
    ),
    description="Tests whether two categorical variables are related, e.g. whether the proportion of "
                "responders differs between treatment groups. Needs counts of subjects in each cell.",
    citation="Pearson (1900) Philos Mag 50:157-175; Yates (1934) J R Stat Soc Suppl 1:217-235",
)
def _run_chi_square(table: Any, yates: Any = "auto", alpha: float = 0.05, diagnostic: bool = False,
                    **_: Any) -> StatResult:
    if isinstance(yates, str):
        key = yates.strip().lower()
        if key not in {"auto", "yes", "no", "true", "false"}:
            raise StatsError("yates must be 'auto', 'yes' or 'no'.")
        flag = None if key == "auto" else key in {"yes", "true"}
    else:
        flag = None if yates is None else bool(yates)
    inputs, notes = _table_inputs(table)
    return _with_notes(chi_square_test(**inputs, yates=flag, alpha=alpha, diagnostic=bool(diagnostic)), notes)


@register(
    key="fisher_exact",
    label="Fisher's exact test",
    category="Contingency tables",
    tables=("contingency",),
    params=(
        ParamSpec("alternative", "Alternative hypothesis", kind="choice", default="two-sided",
                  choices=("two-sided", "greater", "less"),
                  help="'two-sided' asks whether the groups differ in either direction (the usual "
                       "choice). 'greater' / 'less' ask only whether the first-column outcome is more / "
                       "less likely in the first row; 2x2 tables only."),
        _ALPHA,
        _DIAGNOSTIC,
    ),
    description="Tests whether two categorical variables are related using an exact calculation, so it "
                "stays valid when counts are small. Best known for 2x2 tables; larger tables are supported.",
    citation="Fisher (1922) J R Stat Soc 85:87-94; Freeman & Halton (1951) Biometrika 38:141-149",
)
def _run_fisher_exact(table: Any, alternative: str = "two-sided", alpha: float = 0.05,
                      diagnostic: bool = False, **_: Any) -> StatResult:
    inputs, notes = _table_inputs(table)
    return _with_notes(
        fisher_exact_test(**inputs, alternative=alternative, alpha=alpha, diagnostic=bool(diagnostic)), notes
    )


@register(
    key="chi_square_trend",
    label="Chi-square test for trend",
    category="Contingency tables",
    tables=("contingency",),
    params=(
        ParamSpec("scores", "Category scores (optional)", kind="text", default="", optional=True,
                  help="Numbers that say how far apart the ordered groups are, one per group in the order "
                       "of the table, separated by commas: e.g. 0, 1, 5 for doses of 0, 1 and 5 mg. Leave "
                       "empty to treat the groups as equally spaced (scores 1, 2, 3, ...), which is right "
                       "for evenly spaced doses or for ranks such as mild / moderate / severe."),
        _ALPHA,
    ),
    description="For a yes/no outcome across ordered groups (such as increasing doses): tests whether "
                "the proportion rises or falls steadily. Needs a table with two columns or two rows.",
    citation="Cochran (1954) Biometrics 10:417-451; Armitage (1955) Biometrics 11:375-386",
)
def _run_chi_square_trend(table: Any, scores: Any = "", alpha: float = 0.05, **_: Any) -> StatResult:
    inputs, notes = _table_inputs(table)
    return _with_notes(chi_square_trend(**inputs, scores=scores, alpha=alpha), notes)

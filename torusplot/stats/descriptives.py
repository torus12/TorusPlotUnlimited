"""Descriptive statistics (one summary row per group / cell / variable).

Definitions used
----------------
* ``sd``: sample standard deviation (n - 1 in the denominator); ``sem`` = sd / sqrt(n).
* ``ci_low`` / ``ci_high``: confidence interval of the mean from the t distribution with
  n - 1 degrees of freedom: mean +/- t(1 - (1 - level) / 2, n - 1) * sem.
* Quartiles (``q1``, ``median``, ``q3``): linear interpolation between order statistics,
  quantile definition 7 of Hyndman RJ, Fan Y (1996) Sample quantiles in statistical
  packages. *Am Stat* 50:361-365. This is the default of numpy and of R's ``quantile``.
  Other programs use other definitions, so quartiles of small samples can differ slightly
  between packages; ``iqr`` = q3 - q1.
* ``skewness``: the adjusted Fisher-Pearson coefficient G1 = g1 * sqrt(n (n - 1)) / (n - 2)
  (needs n >= 3); ``kurtosis``: the sample *excess* kurtosis
  G2 = (n - 1) / ((n - 2)(n - 3)) * ((n + 1) g2 + 6) (needs n >= 4; 0 for a normal
  distribution). Joanes DN, Gill CA (1998) Comparing measures of sample skewness and
  kurtosis. *The Statistician* 47:183-189. Both are ``None`` when undefined (too few
  values or zero spread).
* ``cv_percent``: 100 * sd / mean (``None`` when the mean is 0).
* ``geometric_mean``: exp(mean(ln x)); only defined (else ``None``) when every value > 0.

Numerical range: all moments are computed on the values divided by a power of two near
their largest magnitude (an exact operation) and scaled back, so the summary is correct
for any finite numbers, from 1e-300 to 1e300. A group counts as constant (SD = 0) only
when its values agree to within floating-point rounding. ``sum`` is ``None`` when the
total exceeds the largest representable number; an SD or confidence limit that cannot be
represented raises ``StatsError``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import numpy as np
from scipy import stats

from ..data.tables import DataTable
from ._utils import StatsError, clean, fmt, is_constant, pow2_back, pow2_scaled
from .registry import ParamSpec, register
from .result import AssumptionCheck, StatResult

DESCRIBE_KEYS = (
    "n", "mean", "sd", "sem", "median", "q1", "q3", "iqr", "min", "max", "ci_low", "ci_high",
    "cv_percent", "skewness", "kurtosis", "geometric_mean", "sum",
)

_CITATIONS = [
    "Hyndman RJ, Fan Y (1996) Sample quantiles in statistical packages. Am Stat 50:361-365 "
    "(quartiles: definition 7).",
    "Joanes DN, Gill CA (1998) Comparing measures of sample skewness and kurtosis. "
    "The Statistician 47:183-189.",
]


def _check_level(ci_level: float) -> float:
    try:
        level = float(ci_level)
    except (TypeError, ValueError):
        raise StatsError("The confidence level must be a number between 0 and 1 (e.g. 0.95).") from None
    if not (0.0 < level < 1.0):
        raise StatsError("The confidence level must be between 0 and 1 (e.g. 0.95 for a 95% CI).")
    return level


def _name_list(names: list[str], limit: int = 8) -> str:
    """Comma-separated names, shortened when there are many."""
    if len(names) <= limit:
        return ", ".join(names)
    return ", ".join(names[:limit]) + f" and {len(names) - limit} more"


def _empty_row() -> dict[str, Any]:
    row: dict[str, Any] = dict.fromkeys(DESCRIBE_KEYS)
    row["n"] = 0
    return row


def _describe_clean(arr: np.ndarray, level: float) -> dict[str, Any]:
    """``describe`` for an already-cleaned, non-empty 1-D float array."""
    n = int(arr.size)
    y, e = pow2_scaled(arr)  # exact scaling: no overflow / underflow in the moments below
    mean_y = float(y.mean())
    mean = pow2_back(mean_y, e)
    q1_y, median_y, q3_y = (float(v) for v in np.percentile(y, [25.0, 50.0, 75.0]))
    total_y = float(y.sum())
    with np.errstate(over="ignore", under="ignore"):
        total = float(np.ldexp(total_y, e))
    out: dict[str, Any] = {
        "n": n,
        "mean": mean,
        "sd": None,
        "sem": None,
        "median": pow2_back(median_y, e),
        "q1": pow2_back(q1_y, e),
        "q3": pow2_back(q3_y, e),
        "iqr": pow2_back(q3_y - q1_y, e),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "ci_low": None,
        "ci_high": None,
        "cv_percent": None,
        "skewness": None,
        "kurtosis": None,
        "geometric_mean": None,
        "sum": total if math.isfinite(total) else None,
    }
    if n >= 2:
        constant = is_constant(arr)
        sd_y = 0.0 if constant else float(y.std(ddof=1))
        sem_y = sd_y / math.sqrt(n)
        half_y = float(stats.t.ppf(0.5 + level / 2.0, n - 1)) * sem_y
        out.update(sd=pow2_back(sd_y, e), sem=pow2_back(sem_y, e),
                   ci_low=pow2_back(mean_y - half_y, e), ci_high=pow2_back(mean_y + half_y, e))
        if mean_y != 0.0:
            cv = 100.0 * sd_y / mean_y
            out["cv_percent"] = cv if math.isfinite(cv) else None
        if not constant:
            dev = y - mean_y
            m2 = float(np.mean(dev**2))
            if m2 > 0.0:
                if n >= 3:
                    g1 = float(np.mean(dev**3)) / m2**1.5
                    out["skewness"] = g1 * math.sqrt(n * (n - 1.0)) / (n - 2.0)
                if n >= 4:
                    g2 = float(np.mean(dev**4)) / m2**2 - 3.0
                    out["kurtosis"] = (n - 1.0) / ((n - 2.0) * (n - 3.0)) * ((n + 1.0) * g2 + 6.0)
    if bool(np.all(arr > 0.0)):
        out["geometric_mean"] = float(math.exp(float(np.mean(np.log(arr)))))
    return out


def describe(values: Iterable[Any], *, ci_level: float = 0.95) -> dict[str, Any]:
    """Summary statistics of one sample (see the module docstring for every definition).

    Missing (NaN) and infinite values are ignored; ``n`` counts the values actually used
    and the extra key ``n_missing`` says how many were ignored. With a single value the
    SD, SEM and confidence interval are ``None``. An empty sample raises ``StatsError``.
    """
    level = _check_level(ci_level)
    if values is None:
        raise StatsError("No data given.")
    raw = np.asarray(values if isinstance(values, np.ndarray) else list(values), dtype=float).ravel()
    arr, _ = clean(raw, "data")
    raw_size = int(raw.size)
    if arr.size == 0:
        raise StatsError("There are no numeric values to summarise (the data are empty or all missing).")
    out = _describe_clean(arr, level)
    out["n_missing"] = raw_size - int(arr.size)
    return out


def descriptive_statistics(groups: dict[str, Iterable[Any]], *, ci_level: float = 0.95) -> StatResult:
    """Descriptive statistics for several groups: one row per group in ``descriptives``.

    Empty groups are kept as a row with ``n = 0`` (and a warning) so the layout of the
    summary matches the data table. There is no hypothesis test and no P value.
    """
    level = _check_level(ci_level)
    if not groups:
        raise StatsError("No groups were given. Enter at least one column of numbers.")
    rows: list[dict[str, Any]] = []
    n: dict[str, int] = {}
    warnings: list[str] = []
    notes: list[str] = []
    empty: list[str] = []
    single: list[str] = []
    for name, values in groups.items():
        label = str(name)
        arr, group_notes = clean(values, label)
        notes.extend(group_notes)
        if arr.size == 0:
            row = _empty_row()
            empty.append(label)
        else:
            row = _describe_clean(arr, level)
            if arr.size == 1:
                single.append(label)
        rows.append({"group": label, **row})
        n[label] = int(arr.size)
    if sum(n.values()) == 0:
        raise StatsError("There are no numeric values to summarise (every group is empty or all missing).")
    if empty:
        warnings.append(f"No numeric values, so the summary is empty for: {_name_list(empty)}.")
    if single:
        warnings.append(
            "Only one value, so SD, SEM and the confidence interval cannot be computed for: "
            f"{_name_list(single)}."
        )

    skewed = [
        r["group"] for r in rows if r["skewness"] is not None and abs(r["skewness"]) > 1.0
    ]
    if skewed:
        detail = (
            f"Markedly skewed (|skewness| > 1): {', '.join(skewed)}. For these, the median and "
            "quartiles describe a typical value better than the mean and SD."
        )
        status = "warn"
    else:
        detail = (
            "The mean, SD and the CI of the mean describe the data well when values are roughly "
            "symmetric; for skewed data prefer the median and quartiles."
        )
        status = "info"
    assumptions = [
        AssumptionCheck(name="Roughly symmetric data (for mean-based summaries)", status=status, detail=detail)
    ]

    used = [r for r in rows if r["n"] > 0]
    pct = f"{level * 100:g}%"
    if len(used) == 1:
        r = used[0]
        spread = f", SD {fmt(r['sd'])}" if r["sd"] is not None else ""
        ci = (
            f" ({pct} CI of the mean {fmt(r['ci_low'])} to {fmt(r['ci_high'])})"
            if r["ci_low"] is not None else ""
        )
        interpretation = (
            f"{r['group']}: n = {r['n']}, mean {fmt(r['mean'])}{spread}{ci}; median {fmt(r['median'])} "
            f"(quartiles {fmt(r['q1'])} to {fmt(r['q3'])}), range {fmt(r['min'])} to {fmt(r['max'])}."
        )
    else:
        means = [r["mean"] for r in used]
        lo, hi = used[int(np.argmin(means))], used[int(np.argmax(means))]
        sizes = sorted({r["n"] for r in used})
        size_text = f"n = {sizes[0]} each" if len(sizes) == 1 else f"n from {sizes[0]} to {sizes[-1]}"
        interpretation = (
            f"Summary of {len(used)} groups ({size_text}). Means range from {fmt(lo['mean'])} "
            f"({lo['group']}) to {fmt(hi['mean'])} ({hi['group']})."
        )
    interpretation += (
        f" The {pct} confidence interval of each mean is the range likely to contain the true "
        "(population) mean, assuming the values are a random sample. These are descriptions only: "
        "no groups were compared and there is no P value."
    )
    return StatResult(
        analysis="descriptives",
        test_name="Descriptive statistics",
        why_chosen=(
            "Descriptive statistics summarise each group (centre, spread and shape) before any "
            "comparison is made. They do not test a hypothesis."
        ),
        assumptions=assumptions,
        n=n,
        descriptives=rows,
        warnings=warnings,
        data_notes=notes,
        interpretation=interpretation,
        methods_text=(
            f"Data are summarised as mean, SD, SEM and {pct} confidence interval of the mean "
            "(t distribution), and as median with quartiles (linear interpolation, Hyndman & Fan type 7)."
        ),
        citations=list(_CITATIONS),
        extra={"ci_level": level, "quartile_method": "Hyndman & Fan (1996) type 7 (linear interpolation)"},
    )


# ------------------------------------------------------------------ table adapters


def _column_groups(table: DataTable) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    return dict(table.groups()), {}


def _grouped_groups(table: DataTable) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    groups: dict[str, np.ndarray] = {}
    keys: dict[str, dict[str, Any]] = {}
    for row in table.row_levels:
        for column in table.column_levels:
            label = f"{row} | {column}"
            groups[label] = table.cell(row, column)
            keys[label] = {"row": row, "column": column}
    return groups, keys


def _xy_groups(table: DataTable) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    groups: dict[str, np.ndarray] = {}
    keys: dict[str, dict[str, Any]] = {}
    x = table.x
    for series in table.series_names:
        values = table.series(series).to_numpy(dtype=float)
        for i in range(values.shape[0]):
            if not np.isfinite(x[i]):
                continue
            row = values[i][np.isfinite(values[i])]
            if row.size == 0:
                continue
            label = f"{series} | {table.x_name} = {fmt(x[i], 6)}"
            if label in groups:  # repeated X value: keep rows apart
                label = f"{label} (row {i + 1})"
            groups[label] = row
            keys[label] = {"series": series, "x": float(x[i])}
    if not groups:
        raise StatsError("The XY table has no Y values to summarise.")
    return groups, keys


def _multiple_variables_groups(
    table: DataTable, columns: Any, by: Any
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]], list[str]]:
    numeric = table.numeric_columns
    if isinstance(columns, str):
        columns = [columns]
    chosen = list(columns) if columns else list(numeric)
    if by is not None and by in chosen and not columns:
        chosen.remove(by)
    if not chosen:
        raise StatsError("This table has no numeric columns to summarise.")
    unknown = [c for c in chosen if c not in table.data.columns]
    if unknown:
        raise StatsError(f"Unknown column(s): {unknown}. Available columns: {list(table.data.columns)}.")
    not_numeric = [c for c in chosen if c not in numeric]
    if not_numeric:
        raise StatsError(
            f"Column(s) {not_numeric} hold text, not numbers, so they cannot be summarised. "
            f"Numeric columns: {numeric}."
        )
    frame = table.analysis_frame()
    groups: dict[str, np.ndarray] = {}
    keys: dict[str, dict[str, Any]] = {}
    notes: list[str] = []
    if by is None or by == "":
        for col in chosen:
            groups[col] = frame[col].to_numpy(dtype=float)
            keys[col] = {"variable": col}
        return groups, keys, notes
    if by not in frame.columns:
        raise StatsError(f"Unknown grouping column '{by}'. Available columns: {list(table.data.columns)}.")
    labels = frame[by]
    missing = labels.isna()
    if bool(missing.any()):
        notes.append(f"{int(missing.sum())} row(s) with no value in '{by}' were left out of the grouped summary.")
    level_names = [str(v) for v in labels[~missing].tolist()]
    levels = list(dict.fromkeys(level_names))
    if not levels:
        raise StatsError(f"The grouping column '{by}' is empty.")
    as_text = labels.astype(object).where(missing, labels.astype(str))
    for col in chosen:
        for level in levels:
            label = f"{col} | {level}"
            mask = (~missing) & (as_text == level)
            groups[label] = frame.loc[mask, col].to_numpy(dtype=float)
            keys[label] = {"variable": col, "by": level}
    return groups, keys, notes


@register(
    key="descriptives",
    label="Descriptive statistics",
    category="Descriptive statistics",
    tables=("column", "grouped", "xy", "multiple_variables"),
    params=(
        ParamSpec(
            "ci_level", "Confidence level", kind="float", default=0.95,
            help="How sure you want to be that the interval contains the true mean. 0.95 gives the "
                 "usual 95% confidence interval.",
        ),
        ParamSpec(
            "columns", "Variables to summarise", kind="columns", default=None, optional=True,
            help="Multiple-variables tables only: which numeric columns to summarise. Leave empty "
                 "to summarise every numeric column.",
        ),
        ParamSpec(
            "by", "Split by", kind="column", default=None, optional=True,
            help="Multiple-variables tables only: a column of category labels (e.g. Treatment). "
                 "Each variable is then summarised separately for every category.",
        ),
    ),
    description=(
        "Summarises each group with its count, mean, SD, SEM, confidence interval of the mean, "
        "median, quartiles, range, and measures of shape. Nothing is compared or tested."
    ),
    citation="Hyndman & Fan (1996) Am Stat 50:361-365 (quartile definition 7)",
)
def _run_descriptives(table: DataTable, ci_level: float = 0.95, columns: Any = None, by: Any = None,
                      **_: Any) -> StatResult:
    extra_notes: list[str] = []
    if table.kind == "column":
        groups, keys = _column_groups(table)
        layout = "one row per column"
    elif table.kind == "grouped":
        groups, keys = _grouped_groups(table)
        layout = f"one row per cell ({table.row_label} | {table.column_label})"
    elif table.kind == "xy":
        groups, keys = _xy_groups(table)
        layout = f"one row per series at each {table.x_name} value (replicates pooled)"
    elif table.kind == "multiple_variables":
        groups, keys, extra_notes = _multiple_variables_groups(table, columns, by)
        layout = "one row per variable" + (f" and level of {by}" if by else "")
    else:  # pragma: no cover - run_analysis refuses other table kinds
        raise StatsError("Descriptive statistics are not available for this table type.")
    result = descriptive_statistics(groups, ci_level=ci_level)
    for row in result.descriptives:
        row.update(keys.get(row["group"], {}))
    result.data_notes.extend(extra_notes)
    result.extra["layout"] = layout
    return result

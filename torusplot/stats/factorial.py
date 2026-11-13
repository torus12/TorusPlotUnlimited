"""Factorial designs: two-way / three-way / N-way ANOVA and ANCOVA.

What is computed
----------------
* The linear model is an ordinary least-squares fit (``statsmodels`` OLS) with
  **sum-to-zero (deviation) coding** for every factor, so that each main effect is the
  effect averaged over the levels of the other factors, also when interactions are in
  the model. The ANOVA table comes from ``statsmodels.stats.anova.anova_lm`` with the
  requested type of sums of squares:

  - **Type III** (default): every term is adjusted for all other terms, including the
    interactions that contain it. Main effects compare *unweighted* (estimated)
    marginal means. This is what most biomedical users get from other statistics
    packages and is the usual choice for unbalanced designs with an interaction.
  - **Type II**: every term is adjusted for all terms that do not contain it. More
    powerful for main effects when there truly is no interaction.
  - With equal cell sizes (balanced data) all types give the same table.

  When the residual variation is exactly zero (a perfect fit, or constant data) the
  Wald-test route of ``anova_lm`` is undefined; the same sums of squares are then
  obtained by direct model comparison (difference of residual SS between nested
  least-squares fits), implemented in :func:`_ss_model_comparison`.

* Multiple comparisons use the **pooled error of the whole model** (residual mean
  square and residual df). Without covariates the compared means are the observed
  cell means (marginal means = unweighted averages of the cell means); with covariates
  they are the covariate-adjusted (least-squares) means from the model, evaluated at
  the mean of each covariate.

  - Tukey (Tukey-Kramer for unequal n): P = P(Q[k, df] > |t| * sqrt(2)), where ``k``
    is the number of means compared in one family: the number of levels of the factor
    being compared for simple-effect and main-effect comparisons (each combination of
    the conditioning factors is its own Tukey family), and the number of cells for
    ``all_cells``.
  - Bonferroni / Sidak / Holm / Holm-Sidak / Benjamini-Hochberg: one family = all the
    comparisons that were requested, together.
  - Confidence intervals: Tukey uses the studentized-range critical value; Bonferroni
    and Holm use the Bonferroni critical value; Sidak and Holm-Sidak use the Sidak
    critical value (single-step simultaneous intervals: conservative companions to the
    step-down tests); ``none`` and ``fdr_bh`` give unadjusted intervals.
  - Standardized effect: Hedges' g = difference / sqrt(MS residual) times Hedges'
    correction J(df residual), with a noncentral-t confidence interval.

References
----------
* Fisher RA (1925) *Statistical Methods for Research Workers*. Oliver & Boyd.
* Yates F (1934) The analysis of multiple classifications with unequal numbers in the
  different classes. *J Am Stat Assoc* 29:51-66.
* Searle SR (1987) *Linear Models for Unbalanced Data*. Wiley (Type II / III
  hypotheses, unweighted means, least-squares means).
* Tukey JW (1949) Comparing individual means in the analysis of variance.
  *Biometrics* 5:99-114; Kramer CY (1956) Extension of multiple range tests to group
  means with unequal numbers of replications. *Biometrics* 12:307-310.
* Cochran WG (1957) Analysis of covariance: its nature and uses. *Biometrics*
  13:261-281.
* Shapiro SS, Wilk MB (1965) *Biometrika* 52:591-611; Brown MB, Forsythe AB (1974)
  *J Am Stat Assoc* 69:364-367 (assumption checks).
* Hedges LV (1981) *J Educ Stat* 6:107-128; Steiger JH (2004) *Psychol Methods*
  9:164-182 (effect sizes and their intervals).
"""

from __future__ import annotations

import itertools
import math
import re
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import optimize, stats
from statsmodels.stats.anova import anova_lm

from . import effect_sizes as es
from ._utils import StatsError, check_alpha, fmt, is_constant, p_phrase, sig_phrase
from .multiplicity import METHOD_LABELS, adjust_pvalues
from .registry import ParamSpec, register
from .result import AssumptionCheck, Comparison, EffectSize, Parameter, StatResult

POSTHOC_METHODS = ("none", "tukey", "bonferroni", "sidak", "holm", "holm-sidak", "fdr_bh")

_CITE_ANOVA = "Fisher RA (1925) Statistical Methods for Research Workers. Oliver & Boyd, Edinburgh"
_CITE_TYPES = (
    "Yates F (1934) The analysis of multiple classifications with unequal numbers in the "
    "different classes. J Am Stat Assoc 29:51-66"
)
_CITE_TUKEY = (
    "Tukey JW (1949) Biometrics 5:99-114; Kramer CY (1956) Biometrics 12:307-310 "
    "(Tukey-Kramer multiple comparisons)"
)
_CITE_SW = "Shapiro SS, Wilk MB (1965) Biometrika 52:591-611"
_CITE_BF = "Brown MB, Forsythe AB (1974) J Am Stat Assoc 69:364-367"
_CITE_ANCOVA = "Cochran WG (1957) Analysis of covariance: its nature and uses. Biometrics 13:261-281"
_CITE_ADJUST = {
    "bonferroni": "Dunn OJ (1961) J Am Stat Assoc 56:52-64",
    "sidak": "Sidak Z (1967) J Am Stat Assoc 62:626-633",
    "holm": "Holm S (1979) Scand J Stat 6:65-70",
    "holm-sidak": "Holm S (1979) Scand J Stat 6:65-70; Sidak Z (1967) J Am Stat Assoc 62:626-633",
    "fdr_bh": "Benjamini Y, Hochberg Y (1995) J R Stat Soc B 57:289-300",
    "tukey": _CITE_TUKEY,
}


# =============================================================================
# Small shared helpers (also used by torusplot.stats.repeated)
# =============================================================================
@contextmanager
def _quiet():
    """Silence expected numerical warnings (they are turned into result warnings)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with np.errstate(all="ignore"):
            yield


def _none_if_nan(value: Any) -> float | None:
    if value is None:
        return None
    value = float(value)
    return None if math.isnan(value) else value


def _level_label(value: Any) -> str:
    """Readable text for a factor level (4.0 -> '4')."""
    if isinstance(value, bool | np.bool_):
        return str(bool(value))
    if isinstance(value, float | np.floating) and math.isfinite(float(value)):
        return f"{float(value):g}"
    if isinstance(value, int | np.integer):
        return str(int(value))
    return str(value).strip()


def _encode_factor(series: pd.Series) -> tuple[np.ndarray, list[str]]:
    """Integer codes + ordered level labels for one factor column.

    Order: the category order for categorical columns, ascending for numeric columns,
    otherwise order of first appearance.
    """
    if isinstance(series.dtype, pd.CategoricalDtype):
        order = [_level_label(c) for c in series.cat.categories]
        labels = series.astype(object).map(_level_label)
    elif pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        order = [_level_label(v) for v in np.sort(pd.unique(series))]
        labels = series.map(_level_label)
    else:
        labels = series.map(_level_label)
        order = list(pd.unique(labels))
    present = set(labels.tolist())
    levels: list[str] = []
    for lab in order:
        if lab in present and lab not in levels:
            levels.append(lab)
    lookup = {lab: i for i, lab in enumerate(levels)}
    return np.array([lookup[v] for v in labels.tolist()], dtype=int), levels


def _check_posthoc(posthoc: str | None, allowed: tuple[str, ...] = POSTHOC_METHODS) -> str:
    key = "none" if posthoc is None else str(posthoc).strip().lower().replace("_", "-")
    key = {"fdr-bh": "fdr_bh", "bh": "fdr_bh", "fdr": "fdr_bh", "tukey-kramer": "tukey",
           "holmsidak": "holm-sidak", "": "none"}.get(key, key)
    if key not in allowed:
        raise StatsError(
            f"Unknown multiple-comparison method '{posthoc}'. Choose one of: {', '.join(allowed)}."
        )
    return key


def _check_ss_type(ss_type: Any) -> int:
    key = str(ss_type).strip().upper()
    value = {"2": 2, "II": 2, "3": 3, "III": 3}.get(key)
    if value is None:
        raise StatsError("ss_type must be 2 (Type II) or 3 (Type III) sums of squares.")
    return value


def _posthoc_label(method: str, balanced: bool = True) -> str:
    if method == "tukey":
        return "Tukey HSD" if balanced else "Tukey-Kramer"
    return METHOD_LABELS[method]


def _describe(values: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=float)
    n = int(arr.size)
    sd = float(arr.std(ddof=1)) if n > 1 else None
    return {
        "n": n,
        "mean": float(arr.mean()) if n else None,
        "sd": sd,
        "sem": sd / math.sqrt(n) if sd is not None else None,
        "median": float(np.median(arr)) if n else None,
        "min": float(arr.min()) if n else None,
        "max": float(arr.max()) if n else None,
    }


def _f_text(f: float | None, df1: float, df2: float) -> str:
    return f"F({fmt(df1)}, {fmt(df2)}) = {fmt(f)}"


def _normality_check(residuals: np.ndarray, target: str = "model residuals") -> AssumptionCheck:
    """Shapiro-Wilk test on residuals (alpha 0.05)."""
    name = "Normality of residuals (Shapiro-Wilk)"
    arr = np.asarray(residuals, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 3:
        return AssumptionCheck(name=name, target=target, status="warn",
                               detail="Too few values to test normality (need at least 3).")
    scale = max(1.0, float(np.max(np.abs(arr))))
    if float(np.ptp(arr)) <= 1e-10 * scale:
        return AssumptionCheck(name=name, target=target, status="warn",
                               detail="The residuals are all (nearly) identical, so normality cannot be tested.")
    with _quiet():
        w, p = stats.shapiro(arr)
    if not (math.isfinite(float(w)) and math.isfinite(float(p))):
        return AssumptionCheck(name=name, target=target, status="warn",
                               detail="The normality test could not be computed for these residuals.")
    note = " (with more than 5000 values this P value is approximate)" if arr.size > 5000 else ""
    if p < 0.05:
        return AssumptionCheck(
            name=name, target=target, statistic=float(w), p_value=float(p), status="fail",
            detail=f"The residuals deviate from a normal distribution ({p_phrase(float(p))}){note}. "
                   "ANOVA is fairly robust to this with similar group sizes; otherwise consider "
                   "transforming the data (e.g. logarithms) or a rank-based analysis.",
        )
    return AssumptionCheck(
        name=name, target=target, statistic=float(w), p_value=float(p), status="pass",
        detail=f"No evidence against normally distributed residuals ({p_phrase(float(p))}){note}.",
    )


def _brown_forsythe(groups: dict[str, np.ndarray], target: str,
                    name: str = "Equal variances (Brown-Forsythe)") -> AssumptionCheck:
    """Brown-Forsythe (median-centred Levene) test across groups with n >= 2."""
    usable = {k: np.asarray(v, dtype=float) for k, v in groups.items() if np.asarray(v).size >= 2}
    skipped = len(groups) - len(usable)
    skip_note = f" {skipped} group(s) with a single value were left out." if skipped else ""
    if len(usable) < 2:
        return AssumptionCheck(
            name=name, target=target, status="warn",
            detail="Equal variances cannot be checked: fewer than two groups have 2 or more values.",
        )
    with _quiet():
        stat, p = stats.levene(*usable.values(), center="median")
    if not (math.isfinite(float(stat)) and math.isfinite(float(p))):
        return AssumptionCheck(
            name=name, target=target, status="warn",
            detail="The variance test could not be computed (no spread within the groups)." + skip_note,
        )
    if p < 0.05:
        return AssumptionCheck(
            name=name, target=target, statistic=float(stat), p_value=float(p), status="fail",
            detail=f"The spread differs between groups ({p_phrase(float(p))}). With unequal group "
                   "sizes the F tests can be unreliable; consider transforming the data." + skip_note,
        )
    return AssumptionCheck(
        name=name, target=target, statistic=float(stat), p_value=float(p), status="pass",
        detail=f"No evidence that the spread differs between groups ({p_phrase(float(p))})." + skip_note,
    )


def _nct_limits(t_obs: float, df: float, level: float) -> tuple[float, float]:
    """Noncentrality parameters that put ``t_obs`` at the upper / lower tail of the
    noncentral t distribution (Steiger & Fouladi 1997): limits of a CI for the ncp."""
    if not (math.isfinite(t_obs) and df > 0):
        return (math.nan, math.nan)
    tail = (1.0 - level) / 2.0
    span = abs(t_obs) + 10.0 + 4.0 * math.sqrt(1.0 + t_obs * t_obs / max(df, 1.0))

    def solve(target: float) -> float:
        def func(ncp: float) -> float:
            value = float(stats.nct.cdf(t_obs, df, ncp))
            if math.isnan(value):
                value = 0.0 if ncp > t_obs else 1.0
            return value - target

        lo, hi = t_obs - span, t_obs + span
        try:
            if func(lo) * func(hi) > 0:
                return math.nan
            return float(optimize.brentq(func, lo, hi, xtol=1e-10, rtol=1e-10, maxiter=200))
        except (ValueError, RuntimeError):  # pragma: no cover - defensive
            return math.nan

    with _quiet():
        return solve(1.0 - tail), solve(tail)


def _smd_pooled(estimate: float, se: float, sd: float, df: float, level: float = 0.95,
                target: str | None = None, name: str = "Hedges' g", correct: bool = True,
                ci_method: str = "noncentral t; standardized by the pooled residual SD of the model",
                ) -> EffectSize | None:
    """Standardized mean difference for a contrast of means: estimate / ``sd``.

    The t statistic of the contrast is d / scale with scale = SE / sd, so the CI comes
    from the noncentral t distribution with the error df (Steiger & Fouladi 1997).
    ``correct=True`` applies Hedges' small-sample correction J(df).
    """
    if not (sd > 0 and se > 0 and df > 1 and math.isfinite(estimate)):
        return None
    scale = se / sd
    j = es.hedges_correction(df) if correct else 1.0
    lo, hi = _nct_limits(estimate / se, df, level)
    value = estimate / sd * j
    return EffectSize(
        name=name, value=value, ci_low=_none_if_nan(lo * scale * j),
        ci_high=_none_if_nan(hi * scale * j), ci_level=level, ci_method=ci_method,
        magnitude=es.magnitude_d(value), target=target,
    )


def _hedges_g_pooled(estimate: float, se: float, mse: float, df: float, level: float = 0.95,
                     target: str | None = None) -> EffectSize | None:
    """Hedges' g = estimate / sqrt(MSE) * J(df), CI by noncentral t (pooled residual SD)."""
    if not mse > 0:
        return None
    return _smd_pooled(estimate, se, math.sqrt(mse), df, level, target)


def _tukey_p(t_value: float, k: int, df: float) -> float:
    """Tukey / Tukey-Kramer P value: P(Q[k, df] > |t| * sqrt(2))."""
    if not math.isfinite(t_value):
        return math.nan
    with _quiet():
        p = float(stats.studentized_range.sf(abs(t_value) * math.sqrt(2.0), k, df))
    return min(max(p, 0.0), 1.0) if math.isfinite(p) else math.nan


def _ci_multiplier(method: str, n_comparisons: int, df: float, alpha: float, k: int | None = None) -> float:
    """Critical value multiplying the SE for the confidence interval of one comparison."""
    m = max(int(n_comparisons), 1)
    with _quiet():
        if method == "tukey" and k is not None and k >= 2:
            return float(stats.studentized_range.ppf(1.0 - alpha, k, df) / math.sqrt(2.0))
        if method in {"bonferroni", "holm"}:
            return float(stats.t.ppf(1.0 - alpha / (2.0 * m), df))
        if method in {"sidak", "holm-sidak"}:
            return float(stats.t.ppf(1.0 - (1.0 - (1.0 - alpha) ** (1.0 / m)) / 2.0, df))
        return float(stats.t.ppf(1.0 - alpha / 2.0, df))


def _ci_note(method: str) -> str:
    if method == "tukey":
        return "Confidence intervals are simultaneous Tukey intervals (studentized range)."
    if method in {"bonferroni", "holm"}:
        extra = " (a conservative companion to the Holm step-down P values)" if method == "holm" else ""
        return f"Confidence intervals are Bonferroni-adjusted simultaneous intervals{extra}."
    if method in {"sidak", "holm-sidak"}:
        extra = " (a conservative companion to the Holm-Sidak step-down P values)" if method == "holm-sidak" else ""
        return f"Confidence intervals are Sidak-adjusted simultaneous intervals{extra}."
    return "Confidence intervals are not adjusted for multiple comparisons."


# =============================================================================
# Design preparation and model fitting
# =============================================================================
@dataclass
class _Design:
    work: pd.DataFrame  # columns y, f0.., c0.. (safe internal names)
    value: str
    factors: list[str]  # factor keys (column names in the user's frame)
    labels: list[str]  # human-readable factor names
    levels: list[list[str]]
    covariates: list[str]
    codes: np.ndarray  # (N, n_factors) integer level codes
    y: np.ndarray
    cov: np.ndarray  # (N, n_covariates)
    notes: list[str] = field(default_factory=list)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(len(lv) for lv in self.levels)

    @property
    def cells(self) -> list[tuple[int, ...]]:
        return list(itertools.product(*[range(k) for k in self.shape]))

    @property
    def cell_index(self) -> np.ndarray:
        return np.ravel_multi_index(tuple(self.codes.T), self.shape)

    @property
    def cell_n(self) -> np.ndarray:
        return np.bincount(self.cell_index, minlength=int(np.prod(self.shape)))

    def cell_name(self, cell: tuple[int, ...]) -> str:
        return ": ".join(self.levels[i][c] for i, c in enumerate(cell))

    def cell_dict(self, cell: tuple[int, ...]) -> dict[str, str]:
        return {self.factors[i]: self.levels[i][c] for i, c in enumerate(cell)}

    def term_name(self, combo: tuple[int, ...]) -> str:
        return " x ".join(self.labels[i] for i in combo)


def _prepare(df: pd.DataFrame, value: str, factors: Any, covariates: Any,
             factor_labels: dict[str, str] | None) -> _Design:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame in long (tidy) format.")
    factors = [factors] if isinstance(factors, str) else [str(f) for f in factors]
    covariates = [] if covariates is None else (
        [covariates] if isinstance(covariates, str) else [str(c) for c in covariates])
    value = str(value)
    if not factors:
        raise StatsError("Choose at least one grouping factor.")
    columns = [value, *factors, *covariates]
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise StatsError(f"Column(s) not found in the data: {missing}. Available: {list(df.columns)}.")
    if len(set(columns)) != len(columns):
        raise StatsError("Each column can play only one role (outcome, factor or covariate).")

    y = pd.to_numeric(df[value], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(y)
    cov_arrays = []
    for c in covariates:
        arr = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
        cov_arrays.append(arr)
        ok &= np.isfinite(arr)
    for f in factors:
        ok &= df[f].notna().to_numpy()
    notes: list[str] = []
    dropped = int((~ok).sum())
    if dropped:
        notes.append(f"{dropped} row(s) with a missing or non-numeric value were ignored.")
    if int(ok.sum()) < 3:
        raise StatsError("There are fewer than 3 complete rows of data; nothing can be analysed.")

    sub = df.loc[ok]
    codes, levels = [], []
    for f in factors:
        c, lv = _encode_factor(sub[f])
        if len(lv) < 2:
            raise StatsError(
                f"Factor '{f}' has only one level ('{lv[0]}') in the data; a factor needs at least "
                "two levels to be tested."
            )
        codes.append(c)
        levels.append(lv)
    codes_arr = np.column_stack(codes)
    y = y[ok]
    cov = np.column_stack([a[ok] for a in cov_arrays]) if cov_arrays else np.empty((y.size, 0))
    for j, c in enumerate(covariates):
        if is_constant(cov[:, j]):
            raise StatsError(f"Covariate '{c}' has the same value in every row, so it cannot be used.")

    labels = [str((factor_labels or {}).get(f, f)) for f in factors]
    if len(set(labels)) != len(labels):
        labels = [f"{lab} ({key})" for lab, key in zip(labels, factors)]

    work = pd.DataFrame({"y": y})
    for i in range(len(factors)):
        work[f"f{i}"] = pd.Categorical(codes_arr[:, i], categories=list(range(len(levels[i]))))
    for j in range(len(covariates)):
        work[f"c{j}"] = cov[:, j]

    design = _Design(work=work, value=value, factors=factors, labels=labels, levels=levels,
                     covariates=covariates, codes=codes_arr, y=y, cov=cov, notes=notes)
    counts = design.cell_n
    if (counts == 0).any():
        empty = [design.cell_name(cell) for cell, n in zip(design.cells, counts) if n == 0]
        shown = "; ".join(empty[:6]) + ("; ..." if len(empty) > 6 else "")
        raise StatsError(
            f"{len(empty)} combination(s) of factor levels have no data ({shown}). A factorial ANOVA "
            "needs at least one value in every cell; remove a level or collect the missing data."
        )
    return design


def _term_key(name: str) -> tuple[str, tuple[int, ...]]:
    """Map a design-matrix / ANOVA-table name to ('i', ()), ('c', (j,)), ('f', (i, ...))
    or, for a factor x covariate column, ('fc', (factor indices..., covariate indices...))."""
    if name == "Intercept":
        return ("i", ())
    facs = tuple(sorted({int(i) for i in re.findall(r"\bf(\d+)\b", name)}))
    covs = tuple(sorted({int(i) for i in re.findall(r"\bc(\d+)\b", name)}))
    if facs and covs:
        return ("fc", facs + covs)
    if covs:
        return ("c", covs)
    return ("f", facs)


def _rss(y: np.ndarray, x: np.ndarray) -> float:
    if x.shape[1] == 0:
        return float(y @ y)
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ beta
    return float(resid @ resid)


def _ss_model_comparison(y: np.ndarray, exog: np.ndarray, col_keys: list[tuple],
                         terms: list[tuple], ss_type: int) -> dict[tuple, float]:
    """Type II / III sums of squares by comparing nested least-squares fits.

    Type III: RSS(all columns except the term's) - RSS(full model); valid because the
    factors use sum-to-zero coding. Type II: the term is added to the model that holds
    every term not containing it (principle of marginality).
    """
    col_keys = list(col_keys)
    all_cols = np.arange(exog.shape[1])
    out: dict[tuple, float] = {}
    rss_full = _rss(y, exog)
    for term in terms:
        own = np.array([k == term for k in col_keys])
        if ss_type == 3:
            out[term] = _rss(y, exog[:, all_cols[~own]]) - rss_full
            continue
        contains = np.array([
            k != term and k[0] == "f" and term[0] == "f" and set(term[1]) < set(k[1]) for k in col_keys
        ])
        base = ~own & ~contains
        out[term] = _rss(y, exog[:, all_cols[base]]) - _rss(y, exog[:, all_cols[base | own]])
    return out


@dataclass
class _Fit:
    result: Any  # statsmodels OLS results
    terms: list[tuple]  # ('f', combo) / ('c', (j,)) in table order
    ss: dict[tuple, float]
    df: dict[tuple, int]
    ss_resid: float
    df_resid: int
    mse: float
    ss_total: float
    n_total: int
    max_order: int
    degenerate: bool


def _fit(design: _Design, ss_type: int, max_order: int, extra_terms: tuple[str, ...] = ()) -> _Fit:
    nf = len(design.factors)
    combos = [c for order in range(1, max_order + 1) for c in itertools.combinations(range(nf), order)]
    rhs = [":".join(f"C(f{i}, Sum)" for i in combo) for combo in combos]
    rhs += [f"c{j}" for j in range(len(design.covariates))]
    rhs += list(extra_terms)
    formula = "y ~ " + " + ".join(rhs)
    with _quiet():
        res = smf.ols(formula, data=design.work).fit()
    exog = np.asarray(res.model.exog, dtype=float)
    names = list(res.model.exog_names)
    y = design.y
    n_total = int(y.size)
    df_resid = n_total - exog.shape[1]
    if df_resid < 1:
        raise StatsError(
            "There are no residual degrees of freedom left to estimate the error: the model has as "
            "many parameters as there are values. Add replicates or leave out the interaction."
        )
    if np.linalg.matrix_rank(exog) < exog.shape[1]:
        raise StatsError(
            "The model cannot be estimated: a covariate is completely determined by the factors "
            "(or by another covariate). Remove it or choose a different covariate."
        )
    ss_total = float(((y - y.mean()) ** 2).sum())
    constant = is_constant(y)
    if constant:
        ss_total = 0.0
    ss_resid = float(res.ssr)
    degenerate = constant or ss_resid <= 1e-12 * ss_total
    col_keys = [_term_key(nm) for nm in names]
    terms = [("f", c) for c in combos] + [("c", (j,)) for j in range(len(design.covariates))]
    df_term = {t: int(sum(k == t for k in col_keys)) for t in terms}
    ss: dict[tuple, float] = {}
    if not degenerate:
        with _quiet():
            table = anova_lm(res, typ=ss_type)
        for idx in table.index:
            key = _term_key(str(idx))
            if key in df_term:
                ss[key] = float(table.loc[idx, "sum_sq"])
        if len(ss) != len(terms) or not all(math.isfinite(v) for v in ss.values()):
            degenerate = True  # pragma: no cover - defensive
    if degenerate:
        if constant:
            ss = dict.fromkeys(terms, 0.0)
        else:
            ss = _ss_model_comparison(y, exog, col_keys, terms, ss_type)
        ss_resid = 0.0
    ss = {t: max(float(v), 0.0) for t, v in ss.items()}
    return _Fit(result=res, terms=terms, ss=ss, df=df_term, ss_resid=ss_resid, df_resid=df_resid,
                mse=ss_resid / df_resid, ss_total=ss_total, n_total=n_total, max_order=max_order,
                degenerate=degenerate)


def _term_rows(design: _Design, fit: _Fit, level: float = 0.95) -> tuple[list[dict[str, Any]], list[EffectSize]]:
    """ANOVA-table rows (terms, Residual, Total) and per-term effect sizes."""
    rows: list[dict[str, Any]] = []
    effects: list[EffectSize] = []
    for term in fit.terms:
        kind, idx = term
        if kind == "c":
            name, row_kind = design.covariates[idx[0]], "covariate"
        else:
            name, row_kind = design.term_name(idx), ("main" if len(idx) == 1 else "interaction")
        ss, df = fit.ss[term], fit.df[term]
        ms = ss / df
        f = p = None
        if fit.mse > 0:
            f = ms / fit.mse
            p = float(stats.f.sf(f, df, fit.df_resid))
        peta = ss / (ss + fit.ss_resid) if (ss + fit.ss_resid) > 0 else None
        rows.append({
            "source": name, "kind": row_kind, "ss": ss, "df": df, "ms": ms, "f": f, "p": p,
            "partial_eta_sq": peta,
            "percent_of_variation": 100.0 * ss / fit.ss_total if fit.ss_total > 0 else None,
        })
        eta = es.partial_eta_squared(ss, fit.ss_resid, df, fit.df_resid, level=level, target=name)
        eta.value = _none_if_nan(eta.value)
        eta.ci_low, eta.ci_high = _none_if_nan(eta.ci_low), _none_if_nan(eta.ci_high)
        effects.append(eta)
        if f is not None:
            omega = es.partial_omega_squared(f, df, fit.df_resid, fit.n_total, level=level, target=name)
            omega.ci_low, omega.ci_high = _none_if_nan(omega.ci_low), _none_if_nan(omega.ci_high)
            effects.append(omega)
    rows.append({
        "source": "Residual", "kind": "residual", "ss": fit.ss_resid, "df": fit.df_resid,
        "ms": fit.mse, "f": None, "p": None, "partial_eta_sq": None,
        "percent_of_variation": 100.0 * fit.ss_resid / fit.ss_total if fit.ss_total > 0 else None,
    })
    rows.append({
        "source": "Total", "kind": "total", "ss": fit.ss_total, "df": fit.n_total - 1,
        "ms": None, "f": None, "p": None, "partial_eta_sq": None,
        "percent_of_variation": 100.0 if fit.ss_total > 0 else None,
    })
    return rows, effects


def _cell_means_cov(design: _Design, fit: _Fit) -> tuple[np.ndarray, np.ndarray]:
    """Means of every cell and their covariance matrix (already scaled by MS residual).

    Without covariates: observed cell means, covariance MSE * diag(1 / n). With
    covariates: least-squares (adjusted) means at the covariate means, L b with
    covariance L Cov(b) L'.
    """
    index = design.cell_index
    counts = design.cell_n
    if not design.covariates:
        means = np.bincount(index, weights=design.y, minlength=counts.size) / counts
        return means, np.diag(fit.mse / counts)
    exog = np.asarray(fit.result.model.exog, dtype=float)
    names = list(fit.result.model.exog_names)
    rows = np.empty((counts.size, exog.shape[1]))
    for cell in range(counts.size):
        rows[cell] = exog[int(np.flatnonzero(index == cell)[0])]
    for j in range(len(design.covariates)):
        rows[:, names.index(f"c{j}")] = design.cov[:, j].mean()
    beta = np.asarray(fit.result.params, dtype=float)
    cov_beta = np.asarray(fit.result.normalized_cov_params, dtype=float) * fit.mse
    return rows @ beta, rows @ cov_beta @ rows.T


# =============================================================================
# Multiple comparisons
# =============================================================================
@dataclass
class _Contrast:
    group_a: str
    group_b: str
    label: str | None
    cell_a: dict[str, str]
    cell_b: dict[str, str]
    weights: np.ndarray  # over the cells (A minus B)
    family: int = 0


def _parse_compare(mode: str, n_factors: int) -> tuple[int | None, tuple[int, ...]]:
    """Return (target factor index or None for all cells, conditioning factor indices)."""
    key = str(mode).strip().lower()
    letters = "abcdefghijklmnopqrstuvwxyz"[:n_factors]
    if key in {"all_cells", "cells", "all"}:
        return None, ()
    match = re.fullmatch(r"main_([a-z])", key)
    if match and match.group(1) in letters:
        return letters.index(match.group(1)), ()
    match = re.fullmatch(r"simple_([a-z])(?:_within_([a-z]+))?", key)
    if match and match.group(1) in letters:
        target = letters.index(match.group(1))
        others = match.group(2)
        if others is None:
            return target, tuple(i for i in range(n_factors) if i != target)
        if all(ch in letters for ch in others) and match.group(1) not in others and len(set(others)) == len(others):
            return target, tuple(sorted(letters.index(ch) for ch in others))
    valid = ["all_cells"] + [f"main_{ch}" for ch in letters]
    if n_factors >= 2:
        valid += [f"simple_{ch}_within_{''.join(o for o in letters if o != ch)}" for ch in letters]
    raise StatsError(
        f"Unknown comparison set '{mode}' for a design with {n_factors} factor(s). "
        f"Examples of valid choices: {', '.join(valid)}."
    )


def _build_contrasts(design: _Design, target: int | None, cond: tuple[int, ...]) -> tuple[list[_Contrast], int, str]:
    """Contrasts for one comparison mode, the Tukey ``k`` and a description of the family."""
    cells = design.cells
    n_cells = len(cells)
    contrasts: list[_Contrast] = []
    if target is None:
        for a, b in itertools.combinations(range(n_cells), 2):
            w = np.zeros(n_cells)
            w[a], w[b] = 1.0, -1.0
            contrasts.append(_Contrast(design.cell_name(cells[a]), design.cell_name(cells[b]), None,
                                       design.cell_dict(cells[a]), design.cell_dict(cells[b]), w))
        text = (f"all {len(contrasts)} pairwise comparisons among the {n_cells} cell means "
                f"(every combination of {' and '.join(design.labels)})")
        return contrasts, n_cells, text

    levels_t = design.levels[target]
    cond_combos = list(itertools.product(*[range(len(design.levels[i])) for i in cond]))
    for fam, combo in enumerate(cond_combos):
        fixed = dict(zip(cond, combo))
        prefix = ": ".join(design.levels[i][c] for i, c in fixed.items())
        weights = []
        for lv in range(len(levels_t)):
            member = np.array([
                cell[target] == lv and all(cell[i] == c for i, c in fixed.items()) for cell in cells
            ])
            weights.append(member / member.sum())
        base_cell = {design.factors[i]: design.levels[i][c] for i, c in fixed.items()}
        if cond:
            label = ", ".join(f"{design.labels[i]} = {design.levels[i][c]}" for i, c in fixed.items())
        else:
            label = f"Main effect of {design.labels[target]}"
        for a, b in itertools.combinations(range(len(levels_t)), 2):
            name_a = f"{prefix}: {levels_t[a]}" if cond else levels_t[a]
            name_b = f"{prefix}: {levels_t[b]}" if cond else levels_t[b]
            contrasts.append(_Contrast(
                name_a, name_b, label,
                {**base_cell, design.factors[target]: levels_t[a]},
                {**base_cell, design.factors[target]: levels_t[b]},
                weights[a] - weights[b], fam,
            ))
    averaged = [design.labels[i] for i in range(len(design.factors)) if i != target and i not in cond]
    avg_note = f", averaged (unweighted) over {' and '.join(averaged)}" if averaged else ""
    if cond:
        text = (f"{len(contrasts)} comparisons: the levels of {design.labels[target]} compared with "
                f"each other within each level of {' x '.join(design.labels[i] for i in cond)}{avg_note}")
    else:
        text = (f"all {len(contrasts)} pairwise comparisons among the {len(levels_t)} marginal means of "
                f"{design.labels[target]}{avg_note}")
    return contrasts, len(levels_t), text


def _comparisons(contrasts: list[_Contrast], means: np.ndarray, cov: np.ndarray, mse: float, df: float,
                 method: str, alpha: float, tukey_k: int) -> list[Comparison]:
    """Turn contrasts of means into Comparison objects (pooled-error t / Tukey tests)."""
    est = np.array([float(c.weights @ means) for c in contrasts])
    se = np.array([math.sqrt(max(float(c.weights @ cov @ c.weights), 0.0)) for c in contrasts])
    t_val = np.full(est.size, np.nan)
    good = se > 0
    t_val[good] = est[good] / se[good]
    with _quiet():
        p_raw = 2.0 * stats.t.sf(np.abs(t_val), df)
    if method == "tukey":
        p_adj = np.array([_tukey_p(t, tukey_k, df) for t in t_val])
    else:
        p_adj = adjust_pvalues(p_raw, method)
    crit = _ci_multiplier(method, int(np.isfinite(p_raw).sum()), df, alpha, tukey_k)
    out: list[Comparison] = []
    for i, c in enumerate(contrasts):
        ok = bool(good[i])
        stat_name, stat = "t", (float(t_val[i]) if ok else None)
        if method == "tukey":
            stat_name, stat = "q", (abs(float(t_val[i])) * math.sqrt(2.0) if ok else None)
        out.append(Comparison(
            id="", group_a=c.group_a, group_b=c.group_b, label=c.label, cell_a=c.cell_a, cell_b=c.cell_b,
            estimate=float(est[i]), se=float(se[i]),
            ci_low=float(est[i] - crit * se[i]) if ok else None,
            ci_high=float(est[i] + crit * se[i]) if ok else None,
            statistic_name=stat_name, statistic=stat, df=float(df),
            p_value=_none_if_nan(p_raw[i]), p_adjusted=_none_if_nan(p_adj[i]),
            effect_size=_hedges_g_pooled(float(est[i]), float(se[i]), mse, df,
                                         target=f"{c.group_a} vs {c.group_b}") if ok else None,
        ))
    return out


# =============================================================================
# N-way ANOVA
# =============================================================================
_WAY_NAMES = {1: "One-way", 2: "Two-way", 3: "Three-way"}


def anova_factorial(
    df: pd.DataFrame,
    value: str,
    factors: list[str] | tuple[str, ...],
    *,
    ss_type: int = 3,
    interactions: bool = True,
    alpha: float = 0.05,
    posthoc: str | None = None,
    posthoc_compare: str = "simple_b_within_a",
    covariates: list[str] | None = None,
    factor_labels: dict[str, str] | None = None,
) -> StatResult:
    """Fixed-effects N-way (factorial) ANOVA on long-format data.

    Parameters
    ----------
    df : tidy frame, one row per measured value.
    value : name of the numeric outcome column.
    factors : names of the grouping columns (2 for two-way, 3 for three-way, ...).
    ss_type : 3 (default) or 2. Type III adjusts every effect for all others including
        interactions and is the usual choice for unbalanced designs; Type II adjusts
        only for terms that do not contain the effect. Identical for balanced data.
    interactions : include all interactions between the factors. When every cell holds
        a single value the highest-order interaction cannot be separated from the error
        and is dropped automatically (with a warning).
    posthoc : ``None``/"none", "tukey", "bonferroni", "sidak", "holm", "holm-sidak",
        "fdr_bh".
    posthoc_compare : which means to compare. Factors are referred to by letter in the
        order given (a = first factor ...):

        * ``"simple_a_within_b"``: levels of A compared within each level of B
          (``"simple_a_within_bc"`` etc. for more factors; unlisted factors are
          averaged over). Tukey k = number of levels of A, one Tukey family per level
          of B; other methods treat all listed comparisons as one family.
        * ``"simple_b_within_a"``: the same with the roles swapped.
        * ``"main_a"``, ``"main_b"``...: marginal means of one factor (unweighted
          averages of the cell means). Tukey k = number of levels of that factor.
        * ``"all_cells"``: every cell against every other cell. Tukey k = number of cells.
    covariates : optional continuous covariates (common slopes; makes it an ANCOVA and
        the compared means become covariate-adjusted means).
    factor_labels : optional ``{factor column: readable name}`` used in the output.

    Comparisons always use the residual mean square and residual df of the fitted
    model (pooled error); see the module docstring for details.
    """
    alpha = check_alpha(alpha)
    ss_type = _check_ss_type(ss_type)
    method = _check_posthoc(posthoc)
    design = _prepare(df, value, factors, covariates, factor_labels)
    nf = len(design.factors)
    target = cond = None
    if method != "none":
        target, cond = _parse_compare(posthoc_compare, nf)

    counts = design.cell_n
    warnings_out: list[str] = []
    unreplicated = bool((counts == 1).all())
    balanced = bool(counts.min() == counts.max())
    max_order = nf if interactions else 1
    dropped_term = None
    if interactions and unreplicated and nf >= 2:
        max_order = nf - 1
        dropped_term = design.term_name(tuple(range(nf)))
        warnings_out.append(
            f"Every cell holds a single value, so the {dropped_term} interaction cannot be separated "
            "from random error. It was left out of the model (its variation is used as the error "
            "term). The tests are only valid if that interaction is truly absent."
        )
    fit = _fit(design, ss_type, max_order)
    rows, effects = _term_rows(design, fit)

    if fit.ss_total == 0:
        warnings_out.append("All values are identical: there is no variation to analyse, so no F tests are possible.")
    elif fit.mse <= 0:
        warnings_out.append(
            "The model fits the data perfectly (zero residual variation), so F ratios and P values "
            "cannot be computed. Check that the data are real replicates."
        )
    if not balanced and nf >= 2 and max_order >= 2:
        warnings_out.append(
            f"The design is unbalanced (cell sizes range from {int(counts.min())} to {int(counts.max())}). "
            f"Type {'III' if ss_type == 3 else 'II'} sums of squares were used; with unbalanced data the "
            "sums of squares do not add up to the total and Type II and Type III results differ."
        )

    # ---------------------------------------------------------------- assumptions
    resid = design.y - np.asarray(fit.result.fittedvalues, dtype=float)
    index = design.cell_index
    cells = design.cells
    assumptions = [
        AssumptionCheck(
            name="Independent observations", status="info",
            detail="Each value must come from a different, independent subject or sample. If the same "
                   "subjects were measured repeatedly, use a repeated-measures or mixed design instead.",
        ),
        _normality_check(resid),
        _brown_forsythe({design.cell_name(c): resid[index == i] if design.covariates else design.y[index == i]
                         for i, c in enumerate(cells)}, target="cells"),
    ]
    if balanced:
        assumptions.append(AssumptionCheck(
            name="Balanced design", status="pass",
            detail=f"Every cell has {int(counts[0])} value(s); Type II and Type III sums of squares are identical.",
        ))
    else:
        assumptions.append(AssumptionCheck(
            name="Balanced design", status="warn",
            detail=f"Cell sizes differ ({int(counts.min())} to {int(counts.max())}). The result depends on the "
                   "type of sums of squares, and the F tests are more sensitive to unequal variances.",
        ))
    if unreplicated:
        assumptions.append(AssumptionCheck(
            name="Replication", status="warn",
            detail="One value per cell: the highest-order interaction is assumed to be absent and serves as error.",
        ))
    else:
        assumptions.append(AssumptionCheck(
            name="Replication", status="pass" if counts.min() >= 2 else "warn",
            detail="Every cell has at least 2 values." if counts.min() >= 2
            else "Some cells hold a single value; their spread cannot be estimated separately.",
        ))
    if design.covariates:
        assumptions.append(AssumptionCheck(
            name="Covariate model", status="info",
            detail="Covariates are assumed to be linearly related to the outcome with the same slope in "
                   "every cell, and not to be affected by the factors.",
        ))

    # ---------------------------------------------------------------- descriptives
    means, cov = _cell_means_cov(design, fit)
    descriptives, cell_means, n_map = [], [], {}
    for i, cell in enumerate(cells):
        desc = _describe(design.y[index == i])
        name = design.cell_name(cell)
        n_map[name] = desc["n"]
        descriptives.append({**design.cell_dict(cell), "group": name, **desc})
        entry = {**design.cell_dict(cell), "mean": desc["mean"], "sd": desc["sd"], "sem": desc["sem"],
                 "n": desc["n"]}
        if design.covariates:
            entry["adjusted_mean"] = float(means[i])
            entry["adjusted_se"] = float(math.sqrt(max(cov[i, i], 0.0)))
        cell_means.append(entry)
    t_crit = float(stats.t.ppf(1 - alpha / 2, fit.df_resid))
    marginal = []
    for fi in range(nf):
        for lv, lv_name in enumerate(design.levels[fi]):
            member = np.array([cell[fi] == lv for cell in cells])
            w = member / member.sum()
            est = float(w @ means)
            se = math.sqrt(max(float(w @ cov @ w), 0.0))
            obs = design.y[design.codes[:, fi] == lv]
            marginal.append({
                "factor": design.factors[fi], "level": lv_name, "mean": est, "se": se,
                "ci_low": est - t_crit * se, "ci_high": est + t_crit * se,
                "n": int(obs.size), "observed_mean": float(obs.mean()),
            })

    # ---------------------------------------------------------------- post hoc tests
    comparisons: list[Comparison] = []
    family_text = correction = None
    extra: dict[str, Any] = {}
    if method != "none":
        if fit.mse <= 0:
            warnings_out.append("Multiple comparisons were skipped because the residual variation is zero.")
        else:
            contrasts, k, family_text = _build_contrasts(design, target, cond)
            comparisons = _comparisons(contrasts, means, cov, fit.mse, fit.df_resid, method, alpha, k)
            correction = _posthoc_label(method, balanced)
            n_fam = len({c.family for c in contrasts})
            if method == "tukey":
                family_text += (
                    f". Tukey's test with k = {k} means per family"
                    + (f" ({n_fam} families, one per conditioning level)" if n_fam > 1 else "")
                )
            elif method != "none":
                family_text += f"; all {len(contrasts)} treated as one family"
            family_text += f". Pooled error: MS residual = {fmt(fit.mse)} with {fit.df_resid} df."
            extra["comparisons_note"] = (
                _ci_note(method) + " Means compared: "
                + ("covariate-adjusted (least-squares) means." if design.covariates
                   else "observed cell means (marginal means are unweighted averages of cell means).")
            )
            extra["tukey_k"] = k if method == "tukey" else None

    # ---------------------------------------------------------------- headline + text
    term_rows = [r for r in rows if r["kind"] in {"main", "interaction"}]
    headline = term_rows[-1] if max_order >= 2 else term_rows[0]
    way = _WAY_NAMES.get(nf, f"{nf}-way")
    test_name = f"{way} ANOVA" if not design.covariates else f"{way} ANCOVA"
    type_text = f"Type {'III' if ss_type == 3 else 'II'} sums of squares"
    sentences = [
        f"{test_name} of {design.value} by {' and '.join(design.labels)} "
        f"({fit.n_total} values, {type_text})."
    ]
    for r in reversed(term_rows) if max_order >= 2 else term_rows:
        what = "Interaction" if r["kind"] == "interaction" else "Main effect of"
        if r["f"] is None:
            sentences.append(f"{what} {r['source']}: could not be tested.")
            continue
        sentences.append(
            f"{what} {r['source']}: {sig_phrase(r['p'], alpha)}, {_f_text(r['f'], r['df'], fit.df_resid)}, "
            f"partial eta squared = {fmt(r['partial_eta_sq'], 3)}."
        )
    inter_sig = [r for r in term_rows if r["kind"] == "interaction" and r["p"] is not None and r["p"] < alpha]
    if inter_sig:
        sentences.append(
            "Because an interaction is significant there is evidence that the effect of one factor "
            "depends on the level of the other; interpret the main effects with caution and look at "
            "the comparisons within each level."
        )
        if ss_type == 2:
            warnings_out.append(
                "Type II main-effect tests assume there is no interaction, but an interaction is "
                "significant here. Consider Type III sums of squares."
            )
    if comparisons:
        n_sig = sum(1 for c in comparisons if c.p_adjusted is not None and c.p_adjusted < alpha)
        sentences.append(
            f"Multiple comparisons ({correction}): {n_sig} of {len(comparisons)} comparisons were "
            f"significant at alpha = {alpha:g}."
        )
    methods = (
        f"Data were analysed by {way.lower()} ANOVA with the factors {' and '.join(design.labels)}"
        + (" and their interaction" + ("s" if nf > 2 else "") if max_order >= 2 else " (no interaction term)")
        + (f", adjusting for the covariate(s) {', '.join(design.covariates)}" if design.covariates else "")
        + f" ({type_text}; alpha = {alpha:g})"
        + (f", followed by {correction} multiple comparisons using the pooled residual error" if comparisons else "")
        + "."
    )
    citations = [_CITE_ANOVA, _CITE_TYPES, _CITE_SW, _CITE_BF]
    if method != "none":
        citations.append(_CITE_ADJUST[method])
    if design.covariates:
        citations.append(_CITE_ANCOVA)

    extra.update({
        "cell_means": cell_means,
        "factors": list(design.factors),
        "factor_labels": dict(zip(design.factors, design.labels)),
        "levels": {f: list(lv) for f, lv in zip(design.factors, design.levels)},
        "marginal_means": marginal,
        "ss_type": ss_type,
        "balanced": balanced,
        "interaction_in_model": max_order >= 2,
        "dropped_term": dropped_term,
        "headline_term": headline["source"],
        "residuals": {"fitted": [float(v) for v in np.asarray(fit.result.fittedvalues)],
                      "residual": [float(v) for v in resid]},
    })
    if design.covariates:
        extra["covariate_means"] = {c: float(design.cov[:, j].mean()) for j, c in enumerate(design.covariates)}

    parameters = []
    if design.covariates:
        parameters = _slope_parameters(design, fit, alpha)

    return StatResult(
        analysis="anova_factorial",
        test_name=test_name,
        why_chosen=(
            f"There are {nf} grouping factors and every value comes from a different subject, so a "
            f"{way.lower()} ANOVA can test each factor and whether the factors interact (whether the "
            "effect of one depends on the other)."
        ),
        assumptions=assumptions,
        statistic_name="F", statistic=headline["f"], df=headline["df"], df2=fit.df_resid,
        p_value=headline["p"], alpha=alpha,
        effect_sizes=effects, parameters=parameters, n=n_map, n_total=fit.n_total,
        descriptives=descriptives, anova_table=rows,
        comparisons=comparisons, comparison_family=family_text, correction=correction,
        warnings=warnings_out, data_notes=list(design.notes),
        interpretation=" ".join(sentences), methods_text=methods, citations=citations, extra=extra,
    )


def anova_twoway(df: pd.DataFrame, value: str, factor_a: str, factor_b: str, **kwargs: Any) -> StatResult:
    """Two-way ANOVA: thin wrapper around :func:`anova_factorial` (same keyword options)."""
    return anova_factorial(df, value, [factor_a, factor_b], **kwargs)


def anova_threeway(df: pd.DataFrame, value: str, factor_a: str, factor_b: str, factor_c: str,
                   **kwargs: Any) -> StatResult:
    """Three-way ANOVA: thin wrapper around :func:`anova_factorial` (same keyword options)."""
    kwargs.setdefault("posthoc_compare", "simple_a_within_bc")
    return anova_factorial(df, value, [factor_a, factor_b, factor_c], **kwargs)


# =============================================================================
# ANCOVA
# =============================================================================
def _slope_parameters(design: _Design, fit: _Fit, alpha: float) -> list[Parameter]:
    res = fit.result
    names = list(res.model.exog_names)
    beta = np.asarray(res.params, dtype=float)
    cov_beta = np.asarray(res.normalized_cov_params, dtype=float) * fit.mse
    t_crit = float(stats.t.ppf(1 - alpha / 2, fit.df_resid))
    out = []
    for j, cname in enumerate(design.covariates):
        i = names.index(f"c{j}")
        b, se = float(beta[i]), math.sqrt(max(float(cov_beta[i, i]), 0.0))
        t = b / se if se > 0 else None
        out.append(Parameter(
            name=f"Slope: {cname}", value=b, se=se,
            ci_low=b - t_crit * se if se > 0 else None, ci_high=b + t_crit * se if se > 0 else None,
            ci_level=1 - alpha, statistic=t,
            p_value=float(2 * stats.t.sf(abs(t), fit.df_resid)) if t is not None else None,
            unit_note=f"Change in {design.value} per unit of {cname} (common slope within groups)",
        ))
    return out


def ancova(
    df: pd.DataFrame,
    value: str,
    factor: str,
    covariates: str | list[str],
    *,
    ss_type: int = 3,
    alpha: float = 0.05,
    posthoc: str | None = None,
    factor_label: str | None = None,
) -> StatResult:
    """One-factor analysis of covariance (ANCOVA) with one or more continuous covariates.

    Model: value = group effect + common slope(s) * covariate(s) + error. Reports the
    group effect adjusted for the covariates, the covariate slopes, and the adjusted
    (least-squares) group means evaluated at the overall mean of each covariate.

    The assumption of equal slopes is tested with a separate model that adds the
    group x covariate interaction (one F test per covariate). Pairwise comparisons of
    the adjusted means use the model's residual error; ``posthoc="tukey"`` applies the
    Tukey-Kramer studentized-range adjustment to these t statistics, which is exact
    only when the covariate means are equal across groups and otherwise an
    approximation (usually slightly conservative). Because the model has no interactions,
    Type II and Type III sums of squares coincide.
    """
    alpha = check_alpha(alpha)
    ss_type = _check_ss_type(ss_type)
    method = _check_posthoc(posthoc)
    cov_list = [covariates] if isinstance(covariates, str) else [str(c) for c in (covariates or [])]
    if not cov_list:
        raise StatsError("ANCOVA needs at least one covariate (a numeric column to adjust for).")
    if not isinstance(factor, str):
        raise StatsError("ANCOVA takes exactly one grouping factor.")
    design = _prepare(df, value, [factor], cov_list, {factor: factor_label} if factor_label else None)
    label = design.labels[0]
    levels = design.levels[0]
    k = len(levels)
    fit = _fit(design, ss_type, 1)
    rows, effects = _term_rows(design, fit)
    means, cov = _cell_means_cov(design, fit)
    codes = design.codes[:, 0]
    warnings_out: list[str] = []
    if fit.ss_total == 0:
        warnings_out.append("All values are identical: there is no variation to analyse.")
    elif fit.mse <= 0:
        warnings_out.append("The model fits the data perfectly (zero residual variation); F tests are not possible.")

    # ------------------------------------------------ homogeneity of regression slopes
    assumptions = [AssumptionCheck(
        name="Independent observations", status="info",
        detail="Each row must come from a different, independent subject or sample.",
    )]
    slope_fail = False
    inter_terms = tuple(f"C(f0, Sum):c{j}" for j in range(len(cov_list)))
    for j, cname in enumerate(cov_list):
        check_name = "Equal slopes (group x covariate interaction)"
        try:
            full = _fit(design, 3, 1, extra_terms=inter_terms)
            reduced = _fit(design, 3, 1, extra_terms=tuple(t for i, t in enumerate(inter_terms) if i != j))
        except StatsError:
            assumptions.append(AssumptionCheck(
                name=check_name, target=cname, status="warn",
                detail="Too few values to test whether the slopes are equal across groups.",
            ))
            continue
        if full.mse <= 0:
            assumptions.append(AssumptionCheck(
                name=check_name, target=cname, status="warn",
                detail="The separate-slopes model fits perfectly, so equality of slopes cannot be tested.",
            ))
            continue
        f_slopes = ((float(reduced.result.ssr) - float(full.result.ssr)) / (k - 1)) / full.mse
        f_slopes = max(f_slopes, 0.0)
        p_slopes = float(stats.f.sf(f_slopes, k - 1, full.df_resid))
        if p_slopes < 0.05:
            slope_fail = True
            assumptions.append(AssumptionCheck(
                name=check_name, target=cname, statistic=f_slopes, p_value=p_slopes, status="fail",
                detail=f"The slope of {cname} differs between groups ({_f_text(f_slopes, k - 1, full.df_resid)}, "
                       f"{p_phrase(p_slopes)}). A standard ANCOVA is not appropriate: the group difference "
                       "depends on the covariate value. Fit separate regression lines per group instead.",
            ))
        else:
            assumptions.append(AssumptionCheck(
                name=check_name, target=cname, statistic=f_slopes, p_value=p_slopes, status="pass",
                detail=f"No evidence that the slope of {cname} differs between groups "
                       f"({_f_text(f_slopes, k - 1, full.df_resid)}, {p_phrase(p_slopes)}).",
            ))
    if slope_fail:
        warnings_out.append(
            "The regression slopes differ between groups, so adjusted means and their comparisons can "
            "be misleading (the group difference changes with the covariate)."
        )
    resid = design.y - np.asarray(fit.result.fittedvalues, dtype=float)
    assumptions.append(AssumptionCheck(
        name="Linear relationship with the covariate", status="info",
        detail="ANCOVA assumes a straight-line relationship between each covariate and the outcome within "
               "every group. Check a scatter plot of the outcome against the covariate.",
    ))
    assumptions.append(_normality_check(resid))
    assumptions.append(_brown_forsythe({lv: resid[codes == i] for i, lv in enumerate(levels)},
                                       target="residuals by group"))
    for j, cname in enumerate(cov_list):
        with _quiet():
            f_cov, p_cov = stats.f_oneway(*[design.cov[codes == i, j] for i in range(k)])
        if math.isfinite(float(p_cov)) and p_cov < 0.05:
            assumptions.append(AssumptionCheck(
                name="Covariate independent of group", target=cname, statistic=float(f_cov),
                p_value=float(p_cov), status="warn",
                detail=f"The mean of {cname} differs between groups ({p_phrase(float(p_cov))}). Adjusted means "
                       "then rely on extrapolation, and the adjustment is not valid if the treatment could "
                       "have changed the covariate.",
            ))
        else:
            assumptions.append(AssumptionCheck(
                name="Covariate independent of group", target=cname,
                statistic=_none_if_nan(f_cov), p_value=_none_if_nan(p_cov), status="info",
                detail=f"The covariate {cname} should be measured before treatment or be unaffected by it.",
            ))

    # ------------------------------------------------ descriptives, adjusted means
    t_crit = float(stats.t.ppf(1 - alpha / 2, fit.df_resid))
    descriptives, adjusted, cell_means, n_map = [], [], [], {}
    for i, lv in enumerate(levels):
        desc = _describe(design.y[codes == i])
        n_map[lv] = desc["n"]
        se = math.sqrt(max(float(cov[i, i]), 0.0))
        cov_means = {f"mean_{c}": float(design.cov[codes == i, j].mean()) for j, c in enumerate(cov_list)}
        descriptives.append({"group": lv, **desc, "adjusted_mean": float(means[i]), **cov_means})
        adjusted.append({
            "group": lv, "n": desc["n"], "mean": desc["mean"], "adjusted_mean": float(means[i]), "se": se,
            "ci_low": float(means[i] - t_crit * se), "ci_high": float(means[i] + t_crit * se),
        })
        cell_means.append({design.factors[0]: lv, "mean": desc["mean"], "sd": desc["sd"], "sem": desc["sem"],
                           "n": desc["n"], "adjusted_mean": float(means[i]), "adjusted_se": se})

    # ------------------------------------------------ pairwise comparisons
    comparisons: list[Comparison] = []
    family_text = correction = None
    extra: dict[str, Any] = {}
    if method != "none":
        if fit.mse <= 0:
            warnings_out.append("Multiple comparisons were skipped because the residual variation is zero.")
        else:
            contrasts, _, _ = _build_contrasts(design, 0, ())
            for c in contrasts:
                c.label = None
            comparisons = _comparisons(contrasts, means, cov, fit.mse, fit.df_resid, method, alpha, k)
            correction = "Tukey-Kramer (approximate for adjusted means)" if method == "tukey" else METHOD_LABELS[method]
            family_text = (
                f"all {len(comparisons)} pairwise comparisons among the {k} covariate-adjusted group means "
                f"(pooled error: MS residual = {fmt(fit.mse)} with {fit.df_resid} df)"
            )
            extra["comparisons_note"] = _ci_note(method) + (
                " The Tukey-Kramer adjustment of adjusted means is exact only when the covariate means "
                "are equal across groups; otherwise it is an approximation (usually slightly conservative)."
                if method == "tukey" else ""
            )

    factor_row = rows[0]
    parameters = _slope_parameters(design, fit, alpha)
    sentences = [
        f"ANCOVA of {design.value} by {label}, adjusting for {', '.join(cov_list)} ({fit.n_total} values)."
    ]
    if factor_row["f"] is None:
        sentences.append(f"The effect of {label} could not be tested.")
    else:
        sentences.append(
            f"After adjusting for the covariate(s), the difference between the {k} groups was "
            f"{sig_phrase(factor_row['p'], alpha)}, {_f_text(factor_row['f'], factor_row['df'], fit.df_resid)}, "
            f"partial eta squared = {fmt(factor_row['partial_eta_sq'], 3)}."
        )
    for par, row in zip(parameters, rows[1:1 + len(cov_list)]):
        sentences.append(
            f"{par.name.removeprefix('Slope: ')}: slope = {fmt(par.value)} "
            f"({p_phrase(row['p'])})."
        )
    sentences.append(
        "Adjusted means: " + "; ".join(f"{a['group']} = {fmt(a['adjusted_mean'])}" for a in adjusted)
        + " (estimated at the average covariate value)."
    )
    if comparisons:
        n_sig = sum(1 for c in comparisons if c.p_adjusted is not None and c.p_adjusted < alpha)
        sentences.append(f"Multiple comparisons ({correction}): {n_sig} of {len(comparisons)} were significant.")
    methods = (
        f"Groups ({label}) were compared by analysis of covariance (ANCOVA) with {', '.join(cov_list)} as "
        f"covariate(s) (Type {'III' if ss_type == 3 else 'II'} sums of squares; alpha = {alpha:g}); "
        "homogeneity of regression slopes was checked with the group x covariate interaction"
        + (f"; adjusted means were compared with {correction} tests" if comparisons else "") + "."
    )
    extra.update({
        "adjusted_means": adjusted,
        "covariate_means": {c: float(design.cov[:, j].mean()) for j, c in enumerate(cov_list)},
        "cell_means": cell_means,
        "factors": list(design.factors),
        "factor_labels": dict(zip(design.factors, design.labels)),
        "covariates": cov_list,
        "ss_type": ss_type,
        "residuals": {"fitted": [float(v) for v in np.asarray(fit.result.fittedvalues)],
                      "residual": [float(v) for v in resid]},
    })
    if len(cov_list) == 1:
        slope = float(parameters[0].value)
        x_mean = float(design.cov[:, 0].mean())
        extra["group_lines"] = {
            lv: {"slope": slope, "intercept": float(means[i] - slope * x_mean),
                 "x": [float(v) for v in design.cov[codes == i, 0]],
                 "y": [float(v) for v in design.y[codes == i]]}
            for i, lv in enumerate(levels)
        }
    citations = [_CITE_ANCOVA, _CITE_ANOVA, _CITE_SW, _CITE_BF]
    if method != "none":
        citations.append(_CITE_ADJUST[method])
    return StatResult(
        analysis="ancova",
        test_name="Analysis of covariance (ANCOVA)",
        why_chosen=(
            "Groups are compared on an outcome that also depends on a continuous covariate (for example "
            "a baseline value). ANCOVA removes the variation explained by the covariate, which makes the "
            "group comparison fairer and usually more sensitive."
        ),
        assumptions=assumptions,
        statistic_name="F", statistic=factor_row["f"], df=factor_row["df"], df2=fit.df_resid,
        p_value=factor_row["p"], alpha=alpha,
        effect_sizes=effects, parameters=parameters, n=n_map, n_total=fit.n_total,
        descriptives=descriptives, anova_table=rows,
        comparisons=comparisons, comparison_family=family_text, correction=correction,
        warnings=warnings_out, data_notes=list(design.notes),
        interpretation=" ".join(sentences), methods_text=methods, citations=citations, extra=extra,
    )


# =============================================================================
# Registered (table-level) analyses
# =============================================================================
_HELP_SS = (
    "How effects are adjusted for each other when group sizes are unequal. Type III (the default, and "
    "what most other statistics packages report) tests each factor after accounting for everything "
    "else, including the interaction. Type II is a little more powerful when there is no interaction. "
    "With equal group sizes both give the same answer."
)
_HELP_POSTHOC = (
    "Follow-up tests that say which groups differ. Tukey is the usual choice for comparing every pair "
    "of means; Bonferroni, Sidak and Holm control the chance of any false positive; fdr_bh controls "
    "the false discovery rate. Choose 'none' to skip."
)
_HELP_ALPHA = "The P value threshold for calling a result significant. 0.05 is the usual choice."


def _mv_pick(table: Any, value: str | None, factors: list[str | None],
             roles: list[str]) -> tuple[str, list[str], list[str]]:
    """Resolve outcome / factor columns of a Multiple variables table, with defaults."""
    notes: list[str] = []
    columns = list(table.data.columns)
    for name in [value, *factors]:
        if name is not None and name not in columns:
            raise StatsError(f"Column '{name}' is not in the table. Available columns: {columns}.")
    if value is None:
        candidates = [c for c in table.numeric_columns if c not in factors]
        if not candidates:
            raise StatsError("The table has no numeric column to analyse. Choose the outcome column.")
        value = candidates[0]
        notes.append(f"Outcome column chosen automatically: '{value}' (the first numeric column).")
    chosen: list[str] = []
    for role, name in zip(roles, factors):
        if name is None:
            taken = {value, *chosen, *[f for f in factors if f is not None]}
            candidates = [c for c in table.categorical_columns if c not in taken]
            if not candidates:
                raise StatsError(
                    f"Could not choose a column for {role} automatically (no unused text column). "
                    "Pick the grouping columns yourself."
                )
            name = candidates[0]
            notes.append(f"{role} chosen automatically: '{name}' (the next text column).")
        chosen.append(name)
    return value, chosen, notes


@register(
    key="anova_twoway",
    label="Two-way ANOVA",
    category="Factorial designs",
    tables=("grouped", "multiple_variables"),
    params=(
        ParamSpec("value", "Outcome column", kind="column", default=None, optional=True,
                  help="Multiple variables tables only: the numeric column that was measured. Leave empty "
                       "to use the first numeric column. Ignored for Grouped tables."),
        ParamSpec("factor_a", "Factor A column", kind="column", default=None, optional=True,
                  help="Multiple variables tables only: the first grouping column. Leave empty to use the "
                       "first text column. In a Grouped table, factor A is always the rows."),
        ParamSpec("factor_b", "Factor B column", kind="column", default=None, optional=True,
                  help="Multiple variables tables only: the second grouping column. Leave empty to use the "
                       "second text column. In a Grouped table, factor B is always the column groups."),
        ParamSpec("ss_type", "Sums of squares type", kind="choice", default=3, choices=(3, 2), help=_HELP_SS),
        ParamSpec("interaction", "Include the interaction", kind="bool", default=True,
                  help="Tests whether the effect of one factor depends on the level of the other. Keep this "
                       "on unless you have only one value per cell or a strong reason to assume no interaction."),
        ParamSpec("posthoc", "Multiple comparisons", kind="choice", default="none",
                  choices=POSTHOC_METHODS, help=_HELP_POSTHOC),
        ParamSpec("posthoc_compare", "What to compare", kind="choice", default="simple_b_within_a",
                  choices=("simple_b_within_a", "simple_a_within_b", "main_a", "main_b", "all_cells"),
                  help="simple_b_within_a: compare the levels of factor B (column groups) within each level "
                       "of factor A (each row), e.g. treatments at each time point. simple_a_within_b: the "
                       "reverse. main_a / main_b: compare the overall means of one factor. all_cells: "
                       "compare every cell with every other cell."),
        ParamSpec("alpha", "Significance level (alpha)", kind="float", default=0.05, help=_HELP_ALPHA),
    ),
    description=(
        "Tests how two grouping factors (for example treatment and time point) affect a measurement, and "
        "whether the effect of one factor depends on the other (interaction). Every value must come from "
        "a different subject."
    ),
    citation=_CITE_ANOVA,
)
def _run_anova_twoway(table: Any, value: str | None = None, factor_a: str | None = None,
                      factor_b: str | None = None, ss_type: Any = 3, interaction: bool = True,
                      posthoc: str | None = "none", posthoc_compare: str = "simple_b_within_a",
                      alpha: float = 0.05) -> StatResult:
    if table.kind == "grouped":
        long = table.to_long()
        labels = {"row": table.row_label, "column": table.column_label}
        if labels["row"] == labels["column"]:
            labels = {"row": f"{table.row_label} (rows)", "column": f"{table.column_label} (columns)"}
        result = anova_factorial(long, "value", ["row", "column"], ss_type=ss_type, interactions=bool(interaction),
                                 alpha=alpha, posthoc=posthoc, posthoc_compare=posthoc_compare,
                                 factor_labels=labels)
        result.data_notes.insert(0, f"Factor A = rows ({table.row_label}); factor B = column groups "
                                    f"({table.column_label}); each replicate is an independent value.")
        return result
    value, (fa, fb), notes = _mv_pick(table, value, [factor_a, factor_b], ["Factor A", "Factor B"])
    result = anova_factorial(table.frame(dropna=False), value, [fa, fb], ss_type=ss_type,
                             interactions=bool(interaction), alpha=alpha, posthoc=posthoc,
                             posthoc_compare=posthoc_compare)
    result.data_notes[:0] = notes
    return result


@register(
    key="anova_threeway",
    label="Three-way ANOVA",
    category="Factorial designs",
    tables=("multiple_variables",),
    params=(
        ParamSpec("value", "Outcome column", kind="column", default=None, optional=True,
                  help="The numeric column that was measured. Leave empty to use the first numeric column."),
        ParamSpec("factor_a", "Factor A column", kind="column", default=None, optional=True,
                  help="The first grouping column. Leave empty to use the first text column."),
        ParamSpec("factor_b", "Factor B column", kind="column", default=None, optional=True,
                  help="The second grouping column. Leave empty to use the second text column."),
        ParamSpec("factor_c", "Factor C column", kind="column", default=None, optional=True,
                  help="The third grouping column. Leave empty to use the third text column."),
        ParamSpec("ss_type", "Sums of squares type", kind="choice", default=3, choices=(3, 2), help=_HELP_SS),
        ParamSpec("interaction", "Include interactions", kind="bool", default=True,
                  help="Tests all two-factor interactions and the three-factor interaction (whether effects "
                       "depend on the levels of the other factors)."),
        ParamSpec("posthoc", "Multiple comparisons", kind="choice", default="none",
                  choices=POSTHOC_METHODS, help=_HELP_POSTHOC),
        ParamSpec("posthoc_compare", "What to compare", kind="choice", default="simple_a_within_bc",
                  choices=("simple_a_within_bc", "simple_b_within_ac", "simple_c_within_ab",
                           "main_a", "main_b", "main_c", "all_cells"),
                  help="simple_a_within_bc: compare the levels of factor A within each combination of B and "
                       "C (and likewise for the others). main_a / main_b / main_c: compare the overall means "
                       "of one factor. all_cells: compare every cell with every other cell."),
        ParamSpec("alpha", "Significance level (alpha)", kind="float", default=0.05, help=_HELP_ALPHA),
    ),
    description=(
        "Tests how three grouping factors affect a measurement, including all their interactions. Every "
        "value must come from a different subject."
    ),
    citation=_CITE_ANOVA,
)
def _run_anova_threeway(table: Any, value: str | None = None, factor_a: str | None = None,
                        factor_b: str | None = None, factor_c: str | None = None, ss_type: Any = 3,
                        interaction: bool = True, posthoc: str | None = "none",
                        posthoc_compare: str = "simple_a_within_bc", alpha: float = 0.05) -> StatResult:
    value, facs, notes = _mv_pick(table, value, [factor_a, factor_b, factor_c],
                                  ["Factor A", "Factor B", "Factor C"])
    result = anova_factorial(table.frame(dropna=False), value, facs, ss_type=ss_type,
                             interactions=bool(interaction), alpha=alpha, posthoc=posthoc,
                             posthoc_compare=posthoc_compare)
    result.data_notes[:0] = notes
    return result


@register(
    key="ancova",
    label="ANCOVA (analysis of covariance)",
    category="Factorial designs",
    tables=("multiple_variables",),
    params=(
        ParamSpec("value", "Outcome column", kind="column", default=None, optional=True,
                  help="The numeric column that was measured. Leave empty to use the first numeric column."),
        ParamSpec("factor", "Group column", kind="column", default=None, optional=True,
                  help="The column that says which group each row belongs to. Leave empty to use the first "
                       "text column."),
        ParamSpec("covariates", "Covariate column(s)", kind="columns", default=None, optional=True,
                  help="One or more numeric columns to adjust for, such as a baseline measurement, age or "
                       "body weight. Leave empty to use the second numeric column."),
        ParamSpec("ss_type", "Sums of squares type", kind="choice", default=3, choices=(3, 2),
                  help="For ANCOVA without interactions, Type II and Type III give the same result; the "
                       "option is here for consistency with the ANOVA analyses."),
        ParamSpec("posthoc", "Multiple comparisons", kind="choice", default="none",
                  choices=POSTHOC_METHODS,
                  help="Follow-up tests comparing every pair of covariate-adjusted group means. " + _HELP_POSTHOC),
        ParamSpec("alpha", "Significance level (alpha)", kind="float", default=0.05, help=_HELP_ALPHA),
    ),
    description=(
        "Compares group means after adjusting for a continuous variable (a covariate, such as a baseline "
        "value) that also influences the measurement. It reports adjusted means and checks that the "
        "covariate has the same slope in every group."
    ),
    citation=_CITE_ANCOVA,
)
def _run_ancova(table: Any, value: str | None = None, factor: str | None = None,
                covariates: Any = None, ss_type: Any = 3, posthoc: str | None = "none",
                alpha: float = 0.05) -> StatResult:
    cov_list = [] if covariates is None else ([covariates] if isinstance(covariates, str) else list(covariates))
    columns = list(table.data.columns)
    for name in cov_list:
        if name not in columns:
            raise StatsError(f"Column '{name}' is not in the table. Available columns: {columns}.")
    if value is not None and value in cov_list:
        raise StatsError("The outcome column cannot also be a covariate.")
    notes: list[str] = []
    if value is None:
        candidates = [c for c in table.numeric_columns if c not in cov_list and c != factor]
        if not candidates:
            raise StatsError("The table has no numeric column to analyse. Choose the outcome column.")
        value = candidates[0]
        notes.append(f"Outcome column chosen automatically: '{value}' (the first numeric column).")
    value, (fac,), more = _mv_pick(table, value, [factor], ["Group column"])
    notes += more
    if not cov_list:
        candidates = [c for c in table.numeric_columns if c not in {value, fac}]
        if not candidates:
            raise StatsError("ANCOVA needs a numeric covariate column, but no unused numeric column was found.")
        cov_list = [candidates[0]]
        notes.append(f"Covariate chosen automatically: '{cov_list[0]}' (the next numeric column).")
    result = ancova(table.frame(dropna=False), value, fac, cov_list, ss_type=ss_type, alpha=alpha,
                    posthoc=posthoc)
    result.data_notes[:0] = notes
    return result

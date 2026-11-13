"""Replicate-aware ("SuperPlot") analysis of nested data.

Technical replicates (cells, wells, fields of view) are nested inside biological
replicates (independent experiments, animals, donors). Measurements from the same
biological replicate are not independent, so counting every cell as n inflates
significance (pseudoreplication). The simple, correct analysis is to average the
technical replicates within each biological replicate and run the test on those
means: n is the number of biological replicates.

References
----------
* Lord SJ, Velle KB, Mullins RD, Fritz-Laylin LK (2020) SuperPlots: communicating
  reproducibility and variability in cell biology. *J Cell Biol* 219:e202001064.
* Hurlbert SH (1984) Pseudoreplication and the design of ecological field
  experiments. *Ecol Monogr* 54:187-211.
* Lazic SE (2010) The problem of pseudoreplication in neuroscientific studies: is it
  affecting your analysis? *BMC Neurosci* 11:5.
* Student (1908) The probable error of a mean. *Biometrika* 6:1-25 (t tests).
* Welch BL (1947) The generalization of 'Student's' problem when several different
  population variances are involved. *Biometrika* 34:28-35; Satterthwaite FE (1946)
  *Biometrics Bull* 2:110-114 (unequal-variance t test and its degrees of freedom).
* Fisher RA (1925) *Statistical Methods for Research Workers* (analysis of variance,
  randomized blocks).
* Holm S (1979) A simple sequentially rejective multiple test procedure.
  *Scand J Stat* 6:65-70.
* Laird NM, Ware JH (1982) Random-effects models for longitudinal data.
  *Biometrics* 38:963-974 (linear mixed model used as a cross-check).
"""

from __future__ import annotations

import math
import warnings as _warnings
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from . import effect_sizes as es
from . import multiplicity
from ._utils import StatsError, check_alpha, ci_phrase, fmt, is_constant, p_phrase, sig_phrase
from .registry import ParamSpec, register
from .result import AssumptionCheck, Comparison, EffectSize, Parameter, StatResult

SUMMARIES = ("mean", "median")
TESTS = ("auto", "t", "welch", "paired_t", "anova", "rm_anova")
_TEST_NAMES = {
    "t": "Unpaired t test on replicate means",
    "welch": "Welch's t test on replicate means",
    "paired_t": "Paired t test on replicate means",
    "anova": "One-way ANOVA on replicate means",
    "rm_anova": "Randomized-block (repeated-measures) one-way ANOVA on replicate means",
}
_TEST_PHRASES = {
    "t": "an unpaired Student's t test",
    "welch": "Welch's unpaired t test",
    "paired_t": "a paired t test",
    "anova": "a one-way ANOVA",
    "rm_anova": "a randomized-block (repeated-measures) one-way ANOVA",
}
CITATIONS = {
    "lord": "Lord SJ, Velle KB, Mullins RD, Fritz-Laylin LK (2020) J Cell Biol 219:e202001064",
    "hurlbert": "Hurlbert SH (1984) Ecol Monogr 54:187-211",
    "lazic": "Lazic SE (2010) BMC Neurosci 11:5",
    "student": "Student (1908) Biometrika 6:1-25",
    "welch": "Welch BL (1947) Biometrika 34:28-35",
    "fisher": "Fisher RA (1925) Statistical Methods for Research Workers",
    "holm": "Holm S (1979) Scand J Stat 6:65-70",
}


# ============================================================================ input
def _label(value: Any) -> str | None:
    """Label as text (``1.0`` -> ``'1'``); ``None`` when missing."""
    if value is None:
        return None
    if isinstance(value, bool | np.bool_):
        return str(bool(value))
    if isinstance(value, float | np.floating):
        if math.isnan(value):
            return None
        return str(int(value)) if float(value).is_integer() else f"{float(value):g}"
    if isinstance(value, int | np.integer):
        return str(int(value))
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "<na>", "nat"}:
        return None
    return text


def _none_if_nan(value: Any) -> float | None:
    if value is None:
        return None
    v = float(value)
    return v if math.isfinite(v) else None


def _prepare(df: pd.DataFrame, value: str, group: str, replicate: str) -> tuple[pd.DataFrame, list[str]]:
    """Long frame with columns ``group, replicate, value`` (text labels, float values)."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame with one row per measurement.")
    roles = {"value": value, "group": group, "replicate": replicate}
    for role, col in roles.items():
        if col not in df.columns:
            raise StatsError(f"The {role} column '{col}' is not in the data. Available columns: {list(df.columns)}.")
    if len({value, group, replicate}) < 3:
        raise StatsError("The value, group and replicate columns must be three different columns.")
    values = pd.to_numeric(df[value], errors="coerce").to_numpy(dtype=float)
    raw_missing = int(df[value].isna().sum())
    non_numeric = int(np.isnan(values).sum()) - raw_missing
    if non_numeric > 0 and non_numeric == len(df) - raw_missing:
        raise StatsError(f"The value column '{value}' does not contain numbers.")
    values = np.where(np.isfinite(values), values, np.nan)
    long = pd.DataFrame({
        "group": [_label(v) for v in df[group].tolist()],
        "replicate": [_label(v) for v in df[replicate].tolist()],
        "value": values,
    })
    keep = long["group"].notna() & long["replicate"].notna() & long["value"].notna()
    notes: list[str] = []
    dropped = int((~keep).sum())
    if dropped:
        notes.append(
            f"{dropped} row(s) with a missing or non-numeric value, group or replicate label were ignored."
        )
    long = long[keep].reset_index(drop=True)
    if long.empty:
        raise StatsError("No complete rows: every measurement needs a value, a group and a replicate label.")
    return long, notes


def _replicate_table(long: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (g, r), sub in long.groupby(["group", "replicate"], sort=False):
        v = sub["value"].to_numpy(dtype=float)
        rows.append({
            "group": g,
            "replicate": r,
            "mean": float(v.mean()),
            "median": float(np.median(v)),
            "n_technical": int(v.size),
            "sd_technical": float(v.std(ddof=1)) if v.size > 1 else math.nan,
        })
    table = pd.DataFrame(rows)
    order = {g: i for i, g in enumerate(dict.fromkeys(long["group"]))}
    table["_order"] = table["group"].map(order)
    return table.sort_values("_order", kind="stable").drop(columns="_order").reset_index(drop=True)


def _check_summary(summary: str) -> str:
    key = str(summary).lower()
    if key not in SUMMARIES:
        raise StatsError("summary must be 'mean' or 'median'.")
    return key


def replicate_means(
    df: pd.DataFrame, group: str, replicate: str, value: str, *, summary: str = "mean"
) -> pd.DataFrame:
    """Collapse technical replicates to one number per biological replicate.

    Returns a DataFrame with columns ``group, replicate, mean`` (or ``median`` when
    ``summary="median"``), ``n_technical`` (number of technical replicates averaged)
    and ``sd_technical`` (their SD; NaN when there is only one). Groups and replicates
    keep their order of first appearance. Rows with a missing value or label are ignored.
    """
    summary = _check_summary(summary)
    long, _ = _prepare(df, value, group, replicate)
    table = _replicate_table(long)
    return table[["group", "replicate", summary, "n_technical", "sd_technical"]]


# =============================================================== small test helpers
def _quiet(func: Any, *args: Any, **kwargs: Any) -> Any:
    """Call ``func`` with numpy / scipy warnings silenced (degenerate input is handled by callers)."""
    with _warnings.catch_warnings(), np.errstate(all="ignore"):
        _warnings.simplefilter("ignore")
        return func(*args, **kwargs)


def _effect(func: Any, *args: Any, **kwargs: Any) -> EffectSize:
    """Shared effect-size helper with warnings silenced and NaN turned into ``None``
    (e.g. Hedges' correction does not exist for 1 degree of freedom)."""
    effect = _quiet(func, *args, **kwargs)
    value = _none_if_nan(effect.value)
    low, high = _none_if_nan(effect.ci_low), _none_if_nan(effect.ci_high)
    if value is None or low is None or high is None:
        low = high = None
    return effect.model_copy(update={
        "value": value, "ci_low": low, "ci_high": high,
        "magnitude": effect.magnitude if value is not None else None,
        "ci_method": effect.ci_method if low is not None else None,
    })


def _t_summary(diff: float, se: float, df: float, level: float) -> dict[str, Any]:
    """t statistic, two-sided P and CI of a difference; ``None`` where undefined."""
    out: dict[str, Any] = {"estimate": diff, "se": se, "df": df, "statistic": None, "p_value": None,
                           "ci_low": None, "ci_high": None}
    if not (math.isfinite(se) and math.isfinite(df)) or se <= 0 or df <= 0:
        out["se"] = _none_if_nan(se)
        out["df"] = _none_if_nan(df)
        return out
    t = diff / se
    crit = float(stats.t.ppf(0.5 + level / 2.0, df))
    out.update(statistic=t, p_value=float(2.0 * stats.t.sf(abs(t), df)),
               ci_low=diff - crit * se, ci_high=diff + crit * se)
    return out


def _unpaired_t(a: np.ndarray, b: np.ndarray, equal_var: bool, level: float) -> dict[str, Any]:
    n1, n2 = a.size, b.size
    diff = float(a.mean() - b.mean())
    v1 = 0.0 if is_constant(a) else float(a.var(ddof=1))
    v2 = 0.0 if is_constant(b) else float(b.var(ddof=1))
    if equal_var:
        df = float(n1 + n2 - 2)
        se = math.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / df * (1.0 / n1 + 1.0 / n2))
    else:
        q1, q2 = v1 / n1, v2 / n2
        se = math.sqrt(q1 + q2)
        denom = q1 * q1 / (n1 - 1) + q2 * q2 / (n2 - 1)
        df = (q1 + q2) ** 2 / denom if denom > 0 else math.nan
    return _t_summary(diff, se, df, level)


def _paired_t(a: np.ndarray, b: np.ndarray, level: float) -> dict[str, Any]:
    d = a - b
    n = d.size
    sd = 0.0 if is_constant(d) else float(d.std(ddof=1))
    return _t_summary(float(d.mean()), sd / math.sqrt(n), float(n - 1), level)


def _shapiro(values: np.ndarray, target: str, what: str) -> AssumptionCheck:
    name = "Normality (Shapiro-Wilk)"
    n = values.size
    if n < 3:
        return AssumptionCheck(name=name, target=target, status="warn",
                               detail=f"Cannot be checked with {n} {what} (at least 3 are needed).")
    if is_constant(values):
        return AssumptionCheck(name=name, target=target, status="warn",
                               detail=f"Cannot be checked: all {what} are identical.")
    res = _quiet(stats.shapiro, values)
    w, p = float(res.statistic), float(res.pvalue)
    weak = " With so few replicates this check has little power." if n < 8 else ""
    if p < 0.05:
        return AssumptionCheck(
            name=name, target=target, statistic=w, p_value=p, status="fail",
            detail=f"The {what} depart from a normal distribution ({p_phrase(p)}). Consider analysing "
                   "log-transformed values or a nonparametric test on the replicate means.",
        )
    return AssumptionCheck(name=name, target=target, statistic=w, p_value=p, status="pass",
                           detail=f"No evidence against normality of the {what} ({p_phrase(p)}).{weak}")


def _brown_forsythe(arrays: list[np.ndarray], fallback: str) -> AssumptionCheck:
    name = "Equal variances (Brown-Forsythe)"
    if any(a.size < 3 for a in arrays):
        return AssumptionCheck(name=name, status="warn",
                               detail="Cannot be checked reliably with fewer than 3 replicates per group.")
    if all(is_constant(a) for a in arrays):
        return AssumptionCheck(name=name, status="warn", detail="Cannot be checked: no variation within groups.")
    res = _quiet(stats.levene, *arrays, center="median")
    f, p = float(res.statistic), float(res.pvalue)
    if not math.isfinite(p):
        return AssumptionCheck(name=name, status="warn", detail="Could not be computed for these data.")
    if p < 0.05:
        return AssumptionCheck(name=name, statistic=f, p_value=p, status="fail",
                               detail=f"The groups' variances differ ({p_phrase(p)}). {fallback}")
    return AssumptionCheck(name=name, statistic=f, p_value=p, status="pass",
                           detail=f"No evidence of unequal variances ({p_phrase(p)}).")


def _oneway_anova(arrays: list[np.ndarray]) -> dict[str, Any]:
    allv = np.concatenate(arrays)
    grand = float(allv.mean())
    ss_between = float(sum(a.size * (a.mean() - grand) ** 2 for a in arrays))
    ss_within = float(sum(((a - a.mean()) ** 2).sum() for a in arrays))
    if all(is_constant(a) for a in arrays):
        ss_within = 0.0
    df1, df2 = len(arrays) - 1, allv.size - len(arrays)
    out: dict[str, Any] = {"ss_effect": ss_between, "ss_error": ss_within, "ss_total": ss_between + ss_within,
                           "df1": df1, "df2": df2, "f": None, "p_value": None}
    if df2 > 0 and ss_within > 0:
        f = (ss_between / df1) / (ss_within / df2)
        out.update(f=f, p_value=float(stats.f.sf(f, df1, df2)))
    return out


def _blocked_anova(matrix: np.ndarray) -> dict[str, Any]:
    """Randomized-block one-way ANOVA; ``matrix`` rows = blocks (replicates), columns = groups.

    SS_total = SS_treatment + SS_blocks + SS_error with
    SS_treatment = b * sum_j (mean_j - grand)^2, SS_blocks = k * sum_i (mean_i - grand)^2,
    F = MS_treatment / MS_error on (k - 1, (k - 1)(b - 1)) degrees of freedom.
    """
    b, k = matrix.shape
    grand = float(matrix.mean())
    ss_total = float(((matrix - grand) ** 2).sum())
    ss_treat = float(b * ((matrix.mean(axis=0) - grand) ** 2).sum())
    ss_blocks = float(k * ((matrix.mean(axis=1) - grand) ** 2).sum())
    resid = matrix - matrix.mean(axis=0, keepdims=True) - matrix.mean(axis=1, keepdims=True) + grand
    ss_error = float((resid ** 2).sum())
    scale = max(ss_total, float(np.abs(matrix).max()) ** 2, 1e-300)
    if ss_error <= 1e-22 * scale * matrix.size:
        ss_error = 0.0
    df1, df_blocks, df2 = k - 1, b - 1, (k - 1) * (b - 1)
    out: dict[str, Any] = {"ss_effect": ss_treat, "ss_blocks": ss_blocks, "ss_error": ss_error,
                           "ss_total": ss_total, "df1": df1, "df_blocks": df_blocks, "df2": df2,
                           "f": None, "p_value": None, "f_blocks": None, "p_blocks": None,
                           "residuals": resid}
    if df2 > 0 and ss_error > 0:
        ms_error = ss_error / df2
        f = (ss_treat / df1) / ms_error
        f_blocks = (ss_blocks / df_blocks) / ms_error
        out.update(f=f, p_value=float(stats.f.sf(f, df1, df2)), f_blocks=f_blocks,
                   p_blocks=float(stats.f.sf(f_blocks, df_blocks, df2)))
    return out


def _mixed_model(long: pd.DataFrame, groups: list[str], matched: bool, df_design: float) -> dict[str, Any]:
    """Linear mixed model on the individual measurements, as a cross-check only.

    Unmatched design: random intercept for every biological replicate (nested in group).
    Matched design: random intercept per replicate (experiment) plus a group-within-
    replicate variance component. Returns a Wald chi-square test of the group effect
    (REML fit). In a balanced design this chi-square equals the squared t (or
    (k - 1) * F) of the replicate-means test, but it is referred to a chi-square
    instead of a t / F distribution, hence the smaller P value. ``p_value_design_df``
    refers chi-square / (k - 1) to an F distribution whose denominator degrees of
    freedom are those of the replicate-level design (``df_design``).
    """
    note = ("Cross-check only: a linear mixed model fitted to the individual measurements. "
            "'p_value' is the large-sample (Wald chi-square) P value, which is too small when there "
            "are few biological replicates; 'p_value_design_df' uses an F distribution with the "
            "replicate-level degrees of freedom instead. Report the replicate-means test.")
    out: dict[str, Any] = {"converged": False, "p_value": None, "p_value_design_df": None,
                           "statistic": None, "df": len(groups) - 1, "df_design": df_design, "note": note}
    try:
        import statsmodels.formula.api as smf

        data = pd.DataFrame({
            "y": long["value"].to_numpy(dtype=float),
            "g": pd.Categorical(long["group"], categories=groups),
            "r": long["replicate"].astype(str),
        })
        data["unit"] = data["g"].astype(str) + "␟" + data["r"]
        if len(data) <= data["unit"].nunique():
            out["note"] = "Not fitted: there is no technical replication (one value per biological replicate)."
            return out
        with _warnings.catch_warnings(), np.errstate(all="ignore"):
            _warnings.simplefilter("ignore")
            if matched:
                model = smf.mixedlm("y ~ C(g)", data, groups=data["r"], re_formula="1",
                                    vc_formula={"condition": "0 + C(g)"})
                out["model"] = "value ~ group + (1 | replicate) + (1 | replicate:group)"
            else:
                model = smf.mixedlm("y ~ C(g)", data, groups=data["unit"])
                out["model"] = "value ~ group + (1 | replicate within group)"
            # Powell first: the default gradient optimisers often stall on these small designs
            fit = model.fit(reml=True, method=["powell", "lbfgs"])
            k = len(groups)
            beta = np.asarray(fit.fe_params, dtype=float)[1:k]
            cov = np.asarray(fit.cov_params(), dtype=float)[1:k, 1:k]
            if not (np.all(np.isfinite(beta)) and np.all(np.isfinite(cov))):
                out["note"] = "The mixed model did not give usable estimates. " + note
                return out
            stat = float(beta @ np.linalg.solve(cov, beta))
        if not math.isfinite(stat) or stat < 0:
            out["note"] = "The mixed model did not give usable estimates. " + note
            return out
        out.update(converged=bool(getattr(fit, "converged", False)), statistic=stat,
                   p_value=float(stats.chi2.sf(stat, k - 1)))
        if df_design > 0:
            out["p_value_design_df"] = float(stats.f.sf(stat / (k - 1), k - 1, df_design))
        if not out["converged"]:
            out["note"] = "The optimiser did not report convergence; treat this value with caution. " + note
    except Exception as exc:  # noqa: BLE001 - optional cross-check must never break the analysis
        out["note"] = f"The mixed model could not be fitted ({type(exc).__name__}). " + note
    return out


def _naive_pooled(long: pd.DataFrame, groups: list[str]) -> tuple[float | None, str, list[int]]:
    """P value of the WRONG analysis that treats every measurement as independent.

    Computed only so the size of the pseudoreplication error can be shown in a warning.
    """
    cells = [long.loc[long["group"] == g, "value"].to_numpy(dtype=float) for g in groups]
    sizes = [int(c.size) for c in cells]
    if any(c.size < 2 for c in cells) or all(is_constant(c) for c in cells):
        return None, "", sizes
    if len(groups) == 2:
        p = float(_quiet(stats.ttest_ind, cells[0], cells[1], equal_var=False).pvalue)
        name = "Welch's t test"
    else:
        p = float(_quiet(stats.f_oneway, *cells).pvalue)
        name = "one-way ANOVA"
    return (p if math.isfinite(p) else None), name, sizes


def _fmt_p_any(p: float | None) -> str:
    if p is None:
        return "not available"
    return f"{p:.2g}" if p < 1e-4 else f"{p:.4f}"


# ================================================================== main analysis
def nested_analysis(
    df: pd.DataFrame,
    value: str,
    group: str,
    replicate: str,
    *,
    paired: bool | None = None,
    test: str = "auto",
    summary: str = "mean",
    alpha: float = 0.05,
    ci_level: float = 0.95,
    mixed_model: bool = True,
) -> StatResult:
    """Compare groups using biological replicates as the unit of analysis.

    ``df`` has one row per measurement (technical replicate) with a numeric ``value``
    column, a ``group`` column (condition) and a ``replicate`` column (which biological
    replicate, e.g. experiment or animal, the measurement came from).

    1. Technical replicates are summarised (``summary``: mean or median) within each
       biological replicate of each group.
    2. The test runs on those replicate summaries, so n = number of biological replicates:

       * two groups: Welch's t test (default for unmatched replicates), Student's t test
         (``test="t"``) or a paired t test when replicates are matched;
       * three or more groups: one-way ANOVA, or a randomized-block (repeated-measures)
         one-way ANOVA when replicates are matched, followed by all pairwise t tests
         (Welch or paired) with Holm-adjusted P values.

    ``paired=None`` treats replicates as matched when exactly the same replicate labels
    occur in every group (e.g. experiments 1, 2, 3 each containing all conditions), as
    Lord et al. (2020) recommend. Set ``paired=False`` if equal labels denote different
    animals or samples in each group.

    The result's ``n`` is the number of biological replicates per group;
    ``extra["replicate_means"]`` holds the per-replicate rows for drawing a SuperPlot.
    The naive P value from pooling all measurements is computed only to show, inside a
    warning, how misleading that analysis would be. ``extra["mixed_model"]`` holds a
    linear mixed model cross-check when it can be fitted.
    """
    alpha = check_alpha(alpha)
    if not (0 < float(ci_level) < 1):
        raise StatsError("ci_level must be between 0 and 1 (typically 0.95).")
    level = float(ci_level)
    summary = _check_summary(summary)
    test = str(test).lower().replace("-", "_")
    if test not in TESTS:
        raise StatsError(f"test must be one of {TESTS}.")
    if paired is not None and not isinstance(paired, bool | np.bool_):
        raise StatsError("paired must be True, False or None (automatic).")

    long, notes = _prepare(df, value, group, replicate)
    table = _replicate_table(long)
    table["value"] = table[summary]
    groups = list(dict.fromkeys(table["group"]))
    k = len(groups)
    if k < 2:
        raise StatsError(
            f"Only one group ('{groups[0]}') was found in '{group}'. Comparing groups needs at least two."
        )
    by_group = {g: table[table["group"] == g].reset_index(drop=True) for g in groups}
    n_bio = {g: int(len(by_group[g])) for g in groups}
    single = [g for g in groups if n_bio[g] < 2]
    if single:
        raise StatsError(
            f"Group(s) {', '.join(single)} have only one biological replicate (n = 1). The sample "
            "size is the number of biological replicates (independent experiments, animals), not "
            "the number of technical replicates (cells, wells) inside them, so with n = 1 no "
            "statistical comparison is possible. Collect at least 3 biological replicates per group."
        )

    # ---------------------------------------------------- matched (paired) design?
    label_sets = [set(by_group[g]["replicate"]) for g in groups]
    same_labels = all(s == label_sets[0] for s in label_sets)
    common = [r for r in by_group[groups[0]]["replicate"] if all(r in s for s in label_sets)]
    if test in {"paired_t", "rm_anova"}:
        if paired is False:
            raise StatsError(f"test='{test}' needs matched replicates, but paired was set to False.")
        want_paired = True
    elif test in {"t", "welch", "anova"}:
        if paired is True:
            raise StatsError(f"test='{test}' is an unpaired test, but paired was set to True.")
        want_paired = False
    else:
        want_paired = same_labels if paired is None else bool(paired)
    warnings: list[str] = []
    if want_paired:
        if len(common) < 2:
            raise StatsError(
                "A paired analysis needs at least 2 biological replicates that occur in every "
                f"group, but only {len(common)} replicate label(s) are shared by all groups. Use "
                "the same replicate labels (e.g. experiment 1, 2, 3) in every group, or set paired = no."
            )
        left_out = sorted({r for s in label_sets for r in s} - set(common))
        if left_out:
            warnings.append(
                f"Replicate(s) {', '.join(left_out)} do not occur in every group and were left out "
                "of the paired analysis."
            )
    if test == "auto":
        test = ("paired_t" if want_paired else "welch") if k == 2 else ("rm_anova" if want_paired else "anova")
    if k == 2 and test in {"anova", "rm_anova"}:
        raise StatsError("ANOVA is for three or more groups; with two groups use a t test (test='auto').")
    if k > 2 and test in {"t", "welch", "paired_t"}:
        raise StatsError(f"A t test compares two groups, but {k} groups were found. Use test='auto' or an ANOVA.")

    if want_paired:
        used = {g: by_group[g].set_index("replicate").loc[common].reset_index() for g in groups}
    else:
        used = by_group
    arrays = {g: used[g]["value"].to_numpy(dtype=float) for g in groups}
    n_used = {g: int(arrays[g].size) for g in groups}

    # -------------------------------------------------------------- descriptives
    descriptives = []
    parameters: list[Parameter] = []
    for g in groups:
        a = arrays[g]
        sd = float(a.std(ddof=1)) if a.size > 1 else math.nan
        sem = sd / math.sqrt(a.size) if a.size > 1 else math.nan
        crit = float(stats.t.ppf(0.5 + level / 2.0, a.size - 1)) if a.size > 1 else math.nan
        tech = [int(v) for v in by_group[g]["n_technical"]]
        descriptives.append({
            "group": g, "n": n_used[g], "mean": float(a.mean()), "sd": _none_if_nan(sd),
            "sem": _none_if_nan(sem), "median": float(np.median(a)), "min": float(a.min()),
            "max": float(a.max()), "ci_low": _none_if_nan(a.mean() - crit * sem),
            "ci_high": _none_if_nan(a.mean() + crit * sem),
            "n_biological": n_bio[g], "n_technical_total": int(sum(tech)),
            "n_technical_per_replicate": tech,
        })
        parameters.append(Parameter(
            name=f"Mean of replicate {summary}s: {g}", value=float(a.mean()), se=_none_if_nan(sem),
            ci_low=_none_if_nan(a.mean() - crit * sem), ci_high=_none_if_nan(a.mean() + crit * sem),
            ci_level=level, unit_note=f"n = {n_used[g]} biological replicates.",
        ))
    rep_rows = [
        {"group": row["group"], "replicate": row["replicate"], "mean": float(row["mean"]),
         "median": float(row["median"]), "value": float(row["value"]),
         "n_technical": int(row["n_technical"]), "sd_technical": _none_if_nan(row["sd_technical"])}
        for row in table.to_dict("records")
    ]
    extra: dict[str, Any] = {
        "replicate_means": rep_rows,
        "summary": summary,
        "design": "paired" if want_paired else "unpaired",
        "matched_replicates": common if want_paired else [],
        "columns": {"value": value, "group": group, "replicate": replicate},
    }

    n_tech_total = int(len(long))
    n_bio_total = int(sum(n_bio.values()))
    tech_min = int(table["n_technical"].min())
    tech_max = int(table["n_technical"].max())
    per_rep = f"{tech_min}" if tech_min == tech_max else f"{tech_min} to {tech_max}"
    notes.append(
        f"{n_tech_total} measurements (technical replicates) were reduced to {n_bio_total} biological "
        f"replicate {summary}s ({per_rep} measurement(s) per replicate); the test used the replicate "
        f"{summary}s."
    )
    if tech_max == 1:
        notes.append("Every biological replicate has a single measurement, so there was nothing to average.")
    few = [g for g in groups if n_used[g] < 3]
    if few:
        warnings.append(
            f"Fewer than 3 biological replicates in {', '.join(few)} (n = "
            f"{', '.join(str(n_used[g]) for g in few)}). With n = 2 a test has almost no power and the "
            "result is fragile; at least 3 independent biological replicates are recommended."
        )

    # ------------------------------------------------------------------ the test
    assumptions: list[AssumptionCheck] = [
        AssumptionCheck(
            name="Independent biological replicates", status="info",
            detail="Each biological replicate (experiment, animal) must be independent of the others. "
                   "Technical replicates inside one biological replicate are not independent, which is "
                   "why they were averaged first.",
        ),
    ]
    effect_sizes: list[EffectSize] = []
    comparisons: list[Comparison] = []
    anova_table: list[dict[str, Any]] = []
    family = None
    correction = None
    statistic_name = "t"
    statistic: float | None = None
    df1: float | None = None
    df2: float | None = None
    p_value: float | None = None
    citations = [CITATIONS["lord"], CITATIONS["hurlbert"], CITATIONS["lazic"]]
    labels_text = ", ".join(common[:6]) + (", ..." if len(common) > 6 else "")

    if want_paired:
        if paired is None and test in {"paired_t", "rm_anova"} and same_labels:
            pair_why = (f"The same replicate labels ({labels_text}) occur in every group, so replicates "
                        "were treated as matched (each experiment contains all conditions) and compared "
                        "within replicate, which removes experiment-to-experiment variation.")
            notes.append(
                "Replicates were matched automatically because the same labels occur in every group. "
                "If equal labels denote different animals or samples in each group, set paired = no."
            )
        else:
            pair_why = (f"Replicates were analysed as matched ({labels_text}): each group's value is "
                        "compared with the other groups' values from the same replicate.")
        assumptions.append(AssumptionCheck(
            name="Matched replicates", status="info",
            detail=f"Values with the same replicate label ({labels_text}) are assumed to come from the "
                   "same experiment / batch across groups.",
        ))
    elif same_labels and paired is False:
        pair_why = ("Replicates were analysed as independent between groups (pairing switched off), "
                    "even though the same replicate labels occur in every group.")
    else:
        pair_why = ("The groups do not share the same replicate labels, so their biological replicates "
                    "were analysed as independent (unpaired)." if not same_labels else
                    "Replicates were analysed as independent between groups (unpaired).")

    degenerate = False
    mean_diff_text = ""
    if k == 2:
        ga, gb = groups
        a, b = arrays[ga], arrays[gb]
        if test == "paired_t":
            res = _paired_t(a, b, level)
            assumptions.append(_shapiro(a - b, f"{ga} - {gb}", "paired differences"))
            effect_sizes = [_effect(es.cohen_d_paired, a, b, level), _effect(es.cohen_d_paired, a, b, level, True)]
            citations.append(CITATIONS["student"])
            test_why = "A paired t test compares the mean of the within-replicate differences with zero."
        else:
            res = _unpaired_t(a, b, test == "t", level)
            assumptions.append(_shapiro(a, ga, "replicate means"))
            assumptions.append(_shapiro(b, gb, "replicate means"))
            if test == "t":
                assumptions.append(_brown_forsythe([a, b], "Use Welch's t test (test = welch) instead."))
                citations.append(CITATIONS["student"])
                test_why = "Student's unpaired t test assumes both groups have the same variance."
            else:
                assumptions.append(AssumptionCheck(
                    name="Equal variances", status="info",
                    detail="Not required: Welch's t test allows the groups to have different variances.",
                ))
                citations.append(CITATIONS["welch"])
                test_why = ("Welch's t test was used because it does not assume equal variances and "
                            "loses almost nothing when they are equal.")
            effect_sizes = [_effect(es.cohen_d_independent, a, b, level, True),
                            _effect(es.cohen_d_independent, a, b, level, False)]
        statistic, df1, p_value = res["statistic"], res["df"], res["p_value"]
        degenerate = statistic is None
        parameters.insert(0, Parameter(
            name=f"Mean difference ({ga} - {gb})", value=res["estimate"], se=res["se"],
            ci_low=res["ci_low"], ci_high=res["ci_high"], ci_level=level, statistic=statistic,
            p_value=p_value, unit_note=f"Difference between the means of the replicate {summary}s.",
        ))
        if not degenerate:
            mean_diff_text = (f" The mean difference ({ga} - {gb}) was {fmt(res['estimate'])} "
                              f"({ci_phrase(res['ci_low'], res['ci_high'], level)}).")
    else:
        statistic_name = "F"
        citations.append(CITATIONS["fisher"])
        if test == "rm_anova":
            matrix = np.column_stack([arrays[g] for g in groups])
            res = _blocked_anova(matrix)
            assumptions.append(_shapiro(res["residuals"].ravel(), "residuals", "residuals"))
            assumptions.append(AssumptionCheck(
                name="Sphericity", status="info",
                detail="The blocked F test assumes the differences between every pair of groups vary "
                       "equally (sphericity). It was not tested or corrected for; with few replicates it "
                       "cannot be checked reliably. The pairwise paired t tests do not rely on it.",
            ))
            anova_table = [
                {"source": group, "ss": res["ss_effect"], "df": res["df1"], "ms": res["ss_effect"] / res["df1"],
                 "f": res["f"], "p": res["p_value"]},
                {"source": f"{replicate} (blocks)", "ss": res["ss_blocks"], "df": res["df_blocks"],
                 "ms": res["ss_blocks"] / res["df_blocks"], "f": res["f_blocks"], "p": res["p_blocks"]},
                {"source": "Residual", "ss": res["ss_error"], "df": res["df2"],
                 "ms": res["ss_error"] / res["df2"], "f": None, "p": None},
                {"source": "Total", "ss": res["ss_total"], "df": matrix.size - 1, "ms": None, "f": None, "p": None},
            ]
            if res["f"] is not None:
                pes = _effect(es.partial_eta_squared, res["ss_effect"], res["ss_error"], res["df1"], res["df2"], level)
                anova_table[0]["partial_eta_sq"] = pes.value
                effect_sizes = [pes]
            test_why = ("A randomized-block (repeated-measures) one-way ANOVA separates the variation "
                        "between replicates (blocks) from the residual before testing the groups.")
        else:
            arrs = [arrays[g] for g in groups]
            res = _oneway_anova(arrs)
            for g in groups:
                assumptions.append(_shapiro(arrays[g], g, "replicate means"))
            assumptions.append(_brown_forsythe(
                arrs, "The pairwise Welch t tests below do not assume equal variances."))
            anova_table = [
                {"source": group, "ss": res["ss_effect"], "df": res["df1"], "ms": res["ss_effect"] / res["df1"],
                 "f": res["f"], "p": res["p_value"]},
                {"source": "Residual", "ss": res["ss_error"], "df": res["df2"],
                 "ms": res["ss_error"] / res["df2"] if res["df2"] > 0 else None, "f": None, "p": None},
                {"source": "Total", "ss": res["ss_total"], "df": res["df1"] + res["df2"], "ms": None,
                 "f": None, "p": None},
            ]
            if res["f"] is not None:
                eta = _effect(es.eta_squared, res["ss_effect"], res["ss_total"], res["f"], res["df1"], res["df2"], level)
                omega = _effect(es.omega_squared, res["ss_effect"], res["df1"], res["ss_error"] / res["df2"],
                               res["ss_total"], res["f"], res["df2"], level)
                anova_table[0]["eta_sq"] = eta.value
                effect_sizes = [eta, omega]
            test_why = "A one-way ANOVA compares the means of three or more independent groups."
        statistic, df1, df2, p_value = res["f"], float(res["df1"]), float(res["df2"]), res["p_value"]
        degenerate = statistic is None

        # pairwise comparisons on the replicate means
        rows = []
        for ga, gb in combinations(groups, 2):
            a, b = arrays[ga], arrays[gb]
            if want_paired:
                pr = _paired_t(a, b, level)
                effect = _effect(es.cohen_d_paired, a, b, level)
            else:
                pr = _unpaired_t(a, b, False, level)
                effect = _effect(es.cohen_d_independent, a, b, level, True)
            effect.target = f"{ga} vs {gb}"
            rows.append((ga, gb, pr, effect))
        adjusted = multiplicity.adjust_pvalues(
            [math.nan if r[2]["p_value"] is None else r[2]["p_value"] for r in rows], "holm")
        for (ga, gb, pr, effect), p_adj in zip(rows, adjusted):
            comparisons.append(Comparison(
                id="", group_a=ga, group_b=gb, estimate=pr["estimate"], se=pr["se"], ci_low=pr["ci_low"],
                ci_high=pr["ci_high"], statistic_name="t", statistic=pr["statistic"], df=pr["df"],
                p_value=pr["p_value"], p_adjusted=_none_if_nan(p_adj), effect_size=effect,
            ))
        pair_name = "paired t tests" if want_paired else "Welch's t tests"
        family = (f"all {len(rows)} pairwise comparisons among {k} groups ({pair_name} on the replicate "
                  f"{summary}s; confidence intervals are not adjusted)")
        correction = multiplicity.METHOD_LABELS["holm"]
        citations.append(CITATIONS["holm"])
        if not want_paired:
            citations.append(CITATIONS["welch"])

    for comp in comparisons:
        if comp.statistic is None and comp.effect_size is not None:
            comp.effect_size = EffectSize(name=comp.effect_size.name, value=None, ci_level=level,
                                          target=comp.effect_size.target)
    if degenerate:
        effect_sizes = [EffectSize(name=e.name, value=None, ci_level=level) for e in effect_sizes]
        warnings.append(
            "The test statistic could not be computed because the biological replicates show no "
            "variability (identical replicate values or identical differences). No P value is given."
        )

    # ----------------------------------------- pseudoreplication warning (naive P)
    if tech_max > 1:
        naive_p, naive_name, sizes = _naive_pooled(long, groups)
        if naive_p is not None:
            size_text = " vs ".join(str(s) for s in sizes)
            bio_text = " vs ".join(str(n_used[g]) for g in groups)
            if p_value is not None and naive_p < p_value:
                verdict = "The naive analysis overstates the evidence"
            else:
                verdict = "The naive analysis uses the wrong sample size"
            warnings.append(
                f"Pseudoreplicated, do not report: treating all {n_tech_total} individual measurements "
                f"as independent (n = {size_text}; {naive_name}) would give P = {_fmt_p_any(naive_p)}. "
                f"The correct analysis of the biological replicates (n = {bio_text}) gives "
                f"P = {_fmt_p_any(p_value)}. {verdict}, because measurements from the same biological "
                "replicate are not independent."
            )

    if mixed_model:
        if want_paired:
            mm_long = long[long["replicate"].isin(common)]
            df_design = float((k - 1) * (len(common) - 1))
        else:
            mm_long = long
            df_design = float(sum(n_used.values()) - k)
        extra["mixed_model"] = _mixed_model(mm_long, groups, want_paired, df_design)

    # ---------------------------------------------------------------------- text
    test_name = _TEST_NAMES[test]
    group_text = ", ".join(f"{g} (n = {n_used[g]})" for g in groups)
    means_text = ", ".join(f"{g}: {fmt(float(arrays[g].mean()))}" for g in groups)
    if degenerate:
        result_text = "The comparison could not be evaluated (no variability among the replicates)."
    elif k == 2:
        result_text = (f"The difference between {groups[0]} and {groups[1]} was {sig_phrase(p_value, alpha)} "
                       f"(t = {fmt(statistic)}, df = {fmt(df1)}).")
    else:
        result_text = (f"The difference among the groups was {sig_phrase(p_value, alpha)} "
                       f"(F({fmt(df1)}, {fmt(df2)}) = {fmt(statistic)}).")
    effect_text = ""
    if effect_sizes and effect_sizes[0].value is not None:
        e0 = effect_sizes[0]
        ci = (f" ({ci_phrase(e0.ci_low, e0.ci_high, level)})"
              if e0.ci_low is not None and e0.ci_high is not None else "")
        mag = f", a {e0.magnitude} effect by the usual benchmarks" if e0.magnitude else ""
        effect_text = f" Effect size: {e0.name} = {fmt(e0.value, 3)}{ci}{mag}."
    pair_text = ""
    if comparisons:
        sig = [c for c in comparisons if c.p_adjusted is not None and c.p_adjusted < alpha]
        if sig:
            listed = "; ".join(f"{c.group_a} vs {c.group_b} ({p_phrase(c.p_adjusted)})" for c in sig[:6])
            pair_text = f" After Holm adjustment these pairs differed: {listed}."
        else:
            pair_text = " After Holm adjustment no pair of groups differed significantly."
    interpretation = (
        f"Technical replicates were averaged within each biological replicate, and the {n_bio_total} "
        f"biological replicate {summary}s were compared between {group_text} with "
        f"{_TEST_PHRASES[test]}. "
        f"{result_text} Group means of the replicate {summary}s: {means_text}.{mean_diff_text}"
        f"{effect_text}{pair_text}"
    )
    methods = (
        f"Technical replicates were summarised by their {summary} within each biological replicate, and "
        f"the replicate {summary}s (n = number of biological replicates) were compared with "
        + {"t": "an unpaired two-tailed Student's t test",
           "welch": "an unpaired two-tailed Welch's t test",
           "paired_t": "a paired two-tailed t test (replicates matched across groups)",
           "anova": "a one-way ANOVA followed by pairwise Welch's t tests with Holm adjustment",
           "rm_anova": "a randomized-block (repeated-measures) one-way ANOVA followed by pairwise "
                       "paired t tests with Holm adjustment"}[test]
        + f" (alpha = {alpha:g}; Lord et al. 2020)."
    )
    why = (
        "The data are nested: several technical replicates (cells, wells) were measured inside each "
        "biological replicate (experiment, animal). Technical replicates are not independent, so each "
        "biological replicate was reduced to one value and the test was run on those values "
        f"(n = number of biological replicates). {pair_why} {test_why}"
    )
    return StatResult(
        analysis="nested_analysis",
        test_name=test_name,
        why_chosen=why,
        assumptions=assumptions,
        statistic_name=statistic_name,
        statistic=statistic,
        df=df1 if statistic is not None else None,
        df2=df2 if statistic is not None else None,
        p_value=p_value,
        alpha=alpha,
        effect_sizes=effect_sizes,
        parameters=parameters,
        n=n_used,
        descriptives=descriptives,
        anova_table=anova_table,
        comparisons=comparisons,
        comparison_family=family,
        correction=correction,
        warnings=warnings,
        data_notes=notes,
        interpretation=interpretation,
        methods_text=methods,
        citations=list(dict.fromkeys(citations)),
        extra=extra,
    )


# ===================================================================== table adapter
_VALUE_WORDS = ("value", "measure", "response", "signal", "intensity")
_GROUP_WORDS = ("group", "condition", "treatment", "genotype")
_REPLICATE_WORDS = ("replicate", "experiment", "batch", "biorep", "animal", "mouse", "donor")


def _match(columns: list[str], words: tuple[str, ...], taken: set[str]) -> str | None:
    for word in words:
        for col in columns:
            if col not in taken and word in col.lower():
                return col
    return None


def _pick_nested_columns(
    table: Any, value: str | None, group: str | None, replicate: str | None
) -> tuple[str, str, str, list[str]]:
    columns = list(table.data.columns)
    numeric = list(table.numeric_columns)
    categorical = list(table.categorical_columns)
    for role, name in (("value", value), ("group", group), ("replicate", replicate)):
        if name is not None and name not in columns:
            raise StatsError(f"The {role} column '{name}' is not in the table. Available columns: {columns}.")
    taken = {c for c in (value, group, replicate) if c is not None}
    notes: list[str] = []
    if replicate is None:
        replicate = _match(columns, _REPLICATE_WORDS, taken)
        if replicate is not None:
            taken.add(replicate)
    if group is None:
        group = _match(columns, _GROUP_WORDS, taken)
        if group is None:
            group = next((c for c in categorical if c not in taken), None)
        if group is None:
            raise StatsError(
                "No column could be identified as the group (condition). Choose the group column."
            )
        taken.add(group)
        notes.append(f"Group column chosen automatically: '{group}'.")
    if replicate is None:
        replicate = next((c for c in categorical if c not in taken), None)
        if replicate is None:
            raise StatsError(
                "No column could be identified as the biological replicate (experiment, animal). "
                "Choose the replicate column."
            )
        taken.add(replicate)
    if value is None:
        value = _match(numeric, _VALUE_WORDS, taken)
        if value is None:
            value = next((c for c in numeric if c not in taken), None)
        if value is None:
            raise StatsError("No numeric column is available for the measured values. Choose the value column.")
        notes.append(f"Value column chosen automatically: '{value}'.")
    return value, group, replicate, notes


@register(
    key="nested_analysis",
    label="Replicate-aware comparison (SuperPlot analysis)",
    category="Replicate-aware (nested)",
    tables=("multiple_variables",),
    params=(
        ParamSpec("value", "Measured value column", kind="column", default=None, optional=True,
                  help="The column with the measurements (one row per cell, well or field). Leave empty "
                       "to use a column named like 'value', or the first numeric column."),
        ParamSpec("group", "Group column", kind="column", default=None, optional=True,
                  help="The column that says which condition each measurement belongs to (e.g. control "
                       "or treated). Leave empty to use a column named like 'group'."),
        ParamSpec("replicate", "Biological replicate column", kind="column", default=None, optional=True,
                  help="The column that says which biological replicate (independent experiment, animal, "
                       "donor) each measurement came from. Biological replicates are independent repeats "
                       "of the whole experiment; technical replicates are repeated measurements (cells, "
                       "wells) inside one of them and do not count towards n."),
        ParamSpec("paired", "Match replicates across groups", kind="choice", default="auto",
                  choices=("auto", "yes", "no"),
                  help="Choose 'yes' when each experiment contains every condition (experiment 1 control "
                       "is compared with experiment 1 treated). 'auto' pairs when the same replicate "
                       "labels appear in every group. Choose 'no' if the groups used different animals."),
        ParamSpec("summary", "Summarise technical replicates by", kind="choice", default="mean",
                  choices=SUMMARIES,
                  help="How the measurements inside one biological replicate are reduced to a single "
                       "number. The mean is usual; the median resists outlying cells."),
        ParamSpec("alpha", "Significance level (alpha)", kind="float", default=0.05,
                  help="The P value threshold for calling a result significant. 0.05 is the usual choice."),
        ParamSpec("test", "Test", kind="choice", default="auto", choices=TESTS,
                  help="'auto' picks the test from the design: Welch's t test or a paired t test for two "
                       "groups, a one-way or randomized-block ANOVA for three or more."),
    ),
    description="For data with technical replicates (cells, wells) nested inside biological replicates "
                "(independent experiments, animals): averages within each biological replicate, then "
                "compares groups using the biological replicates as n, avoiding pseudoreplication.",
    citation="Lord SJ et al. (2020) J Cell Biol 219:e202001064",
)
def _run_nested(
    table: Any,
    value: str | None = None,
    group: str | None = None,
    replicate: str | None = None,
    paired: Any = "auto",
    summary: str = "mean",
    alpha: float = 0.05,
    test: str = "auto",
    **_: Any,
) -> StatResult:
    if isinstance(paired, str):
        key = paired.strip().lower()
        if key not in {"auto", "yes", "no", "true", "false"}:
            raise StatsError("paired must be 'auto', 'yes' or 'no'.")
        flag = None if key == "auto" else key in {"yes", "true"}
    else:
        flag = None if paired is None else bool(paired)
    auto_replicate = replicate is None
    value, group, replicate, notes = _pick_nested_columns(table, value, group, replicate)
    if auto_replicate:
        notes.append(f"Biological replicate column chosen automatically: '{replicate}'.")
    df = table.frame(None, dropna=False)
    result = nested_analysis(df, value, group, replicate, paired=flag, test=test, summary=summary, alpha=alpha)
    result.data_notes = notes + result.data_notes
    return result

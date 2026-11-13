"""Comparing two groups (or one group against a hypothetical value).

Parametric tests
----------------
* Unpaired (Student) t test, pooled variance: Student (1908) The probable error of a
  mean. *Biometrika* 6:1-25; Fisher RA (1925) *Statistical Methods for Research Workers*.
* Welch's t test, unequal variances, Welch-Satterthwaite degrees of freedom: Welch BL
  (1947) The generalization of 'Student's' problem when several different population
  variances are involved. *Biometrika* 34:28-35; Satterthwaite FE (1946) *Biometrics
  Bull* 2:110-114.
* Paired t test and one-sample t test: Student (1908).

Rank-based tests (all computed here from first principles)
-----------------------------------------------------------
* Mann-Whitney U test: Mann HB, Whitney DR (1947) On a test of whether one of two
  random variables is stochastically larger than the other. *Ann Math Stat* 18:50-60;
  Wilcoxon F (1945) Individual comparisons by ranking methods. *Biometrics Bull* 1:80-83.
* Wilcoxon signed-rank test: Wilcoxon (1945); zero differences either discarded
  (Wilcoxon's method) or ranked and then dropped (Pratt JW (1959) Remarks on zeros and
  ties in the Wilcoxon signed rank procedures. *J Am Stat Assoc* 54:655-667).
* Hodges-Lehmann estimators: Hodges JL, Lehmann EL (1963) Estimates of location based
  on rank tests. *Ann Math Stat* 34:598-611. Distribution-free confidence intervals from
  order statistics of the pairwise differences / Walsh averages: Bauer DF (1972)
  Constructing confidence sets using rank statistics. *J Am Stat Assoc* 67:687-690;
  Hollander M, Wolfe DA (1999) *Nonparametric Statistical Methods*, 2nd ed.

Conventions (identical for every test in this module)
------------------------------------------------------
* Differences are always "first group minus second group" (A - B), or "sample minus
  hypothetical value".
* ``alternative="greater"`` means A tends to be larger than B (or the sample larger than
  the hypothetical value); ``"less"`` the opposite.
* **Confidence intervals are always two-sided** at ``ci_level``, also when the P value is
  one-tailed. The result says so in ``warnings`` when a one-tailed test is requested.
* Exact P values for the rank tests are used when there are no ties and no more than
  50 values per group (50 non-zero differences for the signed-rank test); otherwise the
  normal approximation with tie correction and continuity correction is used. The
  result states which (``extra["p_method"]`` and the interpretation).
"""

from __future__ import annotations

import math
import warnings as _warnings
from collections.abc import Iterable
from typing import Any

import numpy as np
from scipy import stats

from ..data.tables import DataTable
from . import effect_sizes as es
from ._utils import (
    StatsError,
    alternative_label,
    check_alpha,
    check_alternative,
    ci_phrase,
    clean,
    clean_paired,
    fmt,
    is_constant,
    p_phrase,
    require_n,
    sig_phrase,
)
from .assumptions import brown_forsythe_test
from .descriptives import _check_level, _describe_clean
from .registry import ParamSpec, register
from .result import AssumptionCheck, EffectSize, Parameter, StatResult

#: Largest sample size (per group / number of non-zero differences) for exact P values.
EXACT_MAX_N = 50
#: Largest number of pairwise differences / Walsh averages held in memory for the
#: Hodges-Lehmann estimate.
_MAX_PAIRS = 10_000_000

_CITE_STUDENT = "Student (1908) The probable error of a mean. Biometrika 6:1-25."
_CITE_WELCH = ("Welch BL (1947) The generalization of 'Student's' problem when several different "
               "population variances are involved. Biometrika 34:28-35.")
_CITE_MWU = ("Mann HB, Whitney DR (1947) On a test of whether one of two random variables is "
             "stochastically larger than the other. Ann Math Stat 18:50-60.")
_CITE_WILCOXON = "Wilcoxon F (1945) Individual comparisons by ranking methods. Biometrics Bull 1:80-83."
_CITE_PRATT = ("Pratt JW (1959) Remarks on zeros and ties in the Wilcoxon signed rank procedures. "
               "J Am Stat Assoc 54:655-667.")
_CITE_HL = ("Hodges JL, Lehmann EL (1963) Estimates of location based on rank tests. "
            "Ann Math Stat 34:598-611.")
_CITE_BAUER = ("Bauer DF (1972) Constructing confidence sets using rank statistics. "
               "J Am Stat Assoc 67:687-690.")
_CITE_D = "Cohen J (1988) Statistical Power Analysis for the Behavioral Sciences, 2nd ed."
_CITE_G = ("Hedges LV (1981) Distribution theory for Glass's estimator of effect size and related "
           "estimators. J Educ Stat 6:107-128.")
_CITE_RB = "Kerby DS (2014) The simple difference formula. Compr Psychol 3:11.IT.3.1 (rank-biserial r)."


# ======================================================================= helpers


def _names(names: tuple[str, str] | list[str]) -> tuple[str, str]:
    if len(names) != 2:
        raise TypeError("names must hold exactly two group names.")
    a, b = str(names[0]), str(names[1])
    if a == b:
        b = f"{b} (2)"
    return a, b


def _float_param(value: Any, what: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        raise StatsError(f"{what} must be a number.") from None
    if not math.isfinite(out):
        raise StatsError(f"{what} must be a finite number.")
    return out


def _row(name: str, arr: np.ndarray, level: float, **more: Any) -> dict[str, Any]:
    return {"group": name, **_describe_clean(arr, level), **more}


def _t_p(t: float, df: float, alternative: str) -> float:
    if alternative == "greater":
        return float(stats.t.sf(t, df))
    if alternative == "less":
        return float(stats.t.cdf(t, df))
    return float(min(1.0, 2.0 * stats.t.sf(abs(t), df)))


def _normality_check(arr: np.ndarray, target: str, alternative_test: str) -> AssumptionCheck:
    """Shapiro-Wilk check (alpha 0.05) of one sample, as an :class:`AssumptionCheck`."""
    name = "Normality (Shapiro-Wilk)"
    if arr.size < 3:
        return AssumptionCheck(name=name, target=target, status="warn",
                               detail=f"Only {arr.size} values: too few to check normality.")
    if is_constant(arr):
        return AssumptionCheck(name=name, target=target, status="warn",
                               detail="All values are identical, so normality cannot be checked.")
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")  # scipy warns that P is approximate for n > 5000
        res = stats.shapiro(arr)
    w, p = float(res.statistic), float(res.pvalue)
    if p < 0.05:
        return AssumptionCheck(
            name=name, target=target, statistic=w, p_value=p, status="fail",
            detail=f"Values deviate from a normal distribution ({p_phrase(p)}). Consider the "
                   f"{alternative_test} instead, or transform the data.",
        )
    detail = f"No evidence against normality ({p_phrase(p)})."
    if arr.size < 8:
        detail += " With so few values this check has little power."
    return AssumptionCheck(name=name, target=target, statistic=w, p_value=p, status="pass", detail=detail)


def _spread_check(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Brown-Forsythe test of equal spread for two samples (statistic/p may be None)."""
    return brown_forsythe_test({"a": x, "b": y})


_INDEPENDENCE = AssumptionCheck(
    name="Independent observations", status="info",
    detail="Each value must come from a different, independently sampled subject or experimental unit. "
           "This cannot be checked from the numbers.",
)
_INDEPENDENT_PAIRS = AssumptionCheck(
    name="Independent pairs", status="info",
    detail="Each pair (row) must be a different subject or matched set, independent of the other pairs. "
           "This cannot be checked from the numbers.",
)


def _one_sided_note(alternative: str, level: float) -> list[str]:
    if alternative == "two-sided":
        return []
    return [
        f"The P value is one-tailed, but every confidence interval shown is the usual two-sided "
        f"{level * 100:g}% interval, so an interval can include 'no difference' although the one-tailed "
        "P value is below alpha."
    ]


def _direction(value: float) -> str:
    return "higher" if value > 0 else ("lower" if value < 0 else "the same as")


def _tail_text(alternative: str, a: str, b: str) -> str:
    if alternative == "greater":
        return f" (one-tailed test of {a} > {b})"
    if alternative == "less":
        return f" (one-tailed test of {a} < {b})"
    return ""


def _evidence(significant: bool, alternative: str, claims: tuple[str, str, str]) -> str:
    """Sentence saying what a (non-)significant result is evidence for.

    ``claims`` = (two-sided claim, claim for "greater", claim for "less").
    """
    claim = claims[{"two-sided": 0, "greater": 1, "less": 2}[alternative]]
    if significant:
        return f"This is evidence that {claim}. "
    return (f"The data do not provide evidence that {claim}; this is not the same as evidence of "
            "no difference. ")


def _tidy(effect: EffectSize) -> EffectSize:
    """Turn NaN in an effect size into ``None`` (undefined), e.g. Hedges' g with 1 df."""
    for field in ("value", "ci_low", "ci_high"):
        v = getattr(effect, field)
        if v is not None and math.isnan(v):
            setattr(effect, field, None)
    if effect.value is None:
        effect.ci_low = effect.ci_high = effect.magnitude = None
    return effect


def _effect_sentence(effect: EffectSize) -> str:
    if effect.value is None:
        return f"{effect.name} could not be computed."
    text = f"{effect.name[0].upper()}{effect.name[1:]} = {fmt(effect.value, 3)}"
    if effect.ci_low is not None and effect.ci_high is not None:
        text += f" ({ci_phrase(effect.ci_low, effect.ci_high, effect.ci_level)})"
    if effect.magnitude:
        text += f", a {effect.magnitude} effect by Cohen's conventions"
    return text + "."


# ================================================================ independent t tests


def _independent_t(a: Iterable[Any], b: Iterable[Any], names: tuple[str, str], alpha: float,
                   alternative: str, ci_level: float, welch: bool) -> StatResult:
    alpha = check_alpha(alpha)
    alternative = check_alternative(alternative)
    level = _check_level(ci_level)
    na, nb = _names(names)
    x, notes_a = clean(a, na)
    y, notes_b = clean(b, nb)
    test_name = "Welch's t test" if welch else "Unpaired t test (Student, pooled variance)"
    require_n(x, 2, na, "a t test")
    require_n(y, 2, nb, "a t test")
    n1, n2 = int(x.size), int(y.size)
    m1, m2 = float(x.mean()), float(y.mean())
    v1 = 0.0 if is_constant(x) else float(x.var(ddof=1))
    v2 = 0.0 if is_constant(y) else float(y.var(ddof=1))
    diff = m1 - m2
    if welch:
        se = math.sqrt(v1 / n1 + v2 / n2)
        df = (
            (v1 / n1 + v2 / n2) ** 2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
            if se > 0 else math.nan
        )
    else:
        df = float(n1 + n2 - 2)
        se = math.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / df * (1.0 / n1 + 1.0 / n2))
    warnings: list[str] = []
    t: float | None
    p: float | None
    if se > 0:
        t = diff / se
        p = _t_p(t, df, alternative)
        half = float(stats.t.ppf(0.5 + level / 2.0, df)) * se
        ci_low, ci_high = diff - half, diff + half
        df_out: float | None = float(df)
        se_out: float | None = se
    else:
        t = p = ci_low = ci_high = df_out = se_out = None
        warnings.append(
            "Both groups have no variability at all (every value within a group is identical), so "
            "the t statistic and P value cannot be computed."
        )
    if min(n1, n2) < 3:
        warnings.append(
            "At least one group has only 2 values: the test has very little power and normality "
            "cannot be checked."
        )
    elif (v1 == 0.0) != (v2 == 0.0) and se > 0:
        warnings.append("One group has no variability (all its values are identical); interpret with care.")
    warnings.extend(_one_sided_note(alternative, level))

    # ---- assumptions
    alt_test = "Mann-Whitney test"
    assumptions = [_normality_check(x, na, alt_test), _normality_check(y, nb, alt_test)]
    bf = _spread_check(x, y)
    if welch:
        detail = "Welch's t test does not assume equal variances."
        if bf["p_value"] is not None:
            detail += f" (Brown-Forsythe test of equal variances: {p_phrase(bf['p_value'])}.)"
        assumptions.append(AssumptionCheck(name="Equal variances (not required)", statistic=bf["statistic"],
                                           p_value=bf["p_value"], status="info", detail=detail))
    elif bf["p_value"] is None:
        assumptions.append(AssumptionCheck(
            name="Equal variances (Brown-Forsythe)", status="warn",
            detail="Could not be checked with these data. If in doubt, use Welch's t test, which does "
                   "not assume equal variances.",
        ))
    elif bf["p_value"] < 0.05:
        assumptions.append(AssumptionCheck(
            name="Equal variances (Brown-Forsythe)", statistic=bf["statistic"], p_value=bf["p_value"],
            status="fail",
            detail=f"The two groups differ in spread ({p_phrase(bf['p_value'])}; SD {fmt(math.sqrt(v1))} "
                   f"vs {fmt(math.sqrt(v2))}). Use Welch's t test instead.",
        ))
    else:
        assumptions.append(AssumptionCheck(
            name="Equal variances (Brown-Forsythe)", statistic=bf["statistic"], p_value=bf["p_value"],
            status="pass", detail=f"No evidence that the spread differs ({p_phrase(bf['p_value'])}).",
        ))
    assumptions.append(_INDEPENDENCE)

    # ---- effect sizes
    d = _tidy(es.cohen_d_independent(x, y, level=level))
    g = _tidy(es.cohen_d_independent(x, y, level=level, hedges=True))
    if welch:
        for effect in (d, g):
            if effect.ci_method:
                effect.ci_method += (
                    "; standardised by the pooled SD of the two groups (as for the unpaired t test), "
                    "not by a Welch-specific SD"
                )

    parameters = [
        Parameter(name=f"Mean difference ({na} - {nb})", value=diff, se=se_out, ci_low=ci_low,
                  ci_high=ci_high, ci_level=level, statistic=t, p_value=p,
                  unit_note="Two-sided confidence interval" + (" (Welch-Satterthwaite df)." if welch else ".")),
    ]
    for name, arr, mean, var in ((na, x, m1, v1), (nb, y, m2, v2)):
        sem = math.sqrt(var / arr.size)
        half = float(stats.t.ppf(0.5 + level / 2.0, arr.size - 1)) * sem
        parameters.append(Parameter(name=f"Mean of {name}", value=mean, se=sem, ci_low=mean - half,
                                    ci_high=mean + half, ci_level=level))

    kind = "Welch's t test" if welch else "unpaired t test"
    by_kind = "By Welch's t test" if welch else "By the unpaired t test"
    if t is None:
        interpretation = (
            f"The mean of {na} was {fmt(m1)} (n = {n1}) and the mean of {nb} was {fmt(m2)} (n = {n2}), "
            f"a difference of {fmt(diff)}. Because neither group shows any variability, the {kind} "
            "cannot be computed."
        )
    else:
        interpretation = (
            f"The mean of {na} ({fmt(m1)}, n = {n1}) was {fmt(abs(diff))} {_direction(diff)} than the mean "
            f"of {nb} ({fmt(m2)}, n = {n2}); difference {na} - {nb} = {fmt(diff)} "
            f"({ci_phrase(ci_low, ci_high, level)}). "
            if diff != 0 else
            f"The means of {na} (n = {n1}) and {nb} (n = {n2}) were identical ({fmt(m1)}). "
        )
        interpretation += (
            f"{by_kind}{_tail_text(alternative, na, nb)} this difference is {sig_phrase(p, alpha)}; "
            f"t({fmt(df)}) = {fmt(t)}. "
        )
        interpretation += _evidence(
            p < alpha, alternative,
            ("the population means differ", f"the population mean of {na} is higher than that of {nb}",
             f"the population mean of {na} is lower than that of {nb}"),
        )
        interpretation += _effect_sentence(d)
    if welch:
        methods = (f"The two groups were compared with Welch's unequal-variance t test "
                   f"({alternative_label(alternative)}, alpha = {alpha:g}).")
    else:
        methods = (f"The two groups were compared with an unpaired Student's t test "
                   f"({alternative_label(alternative)}, alpha = {alpha:g}).")
    why = (
        "Welch's t test compares the means of two independent groups without assuming that they have "
        "the same spread. It is the safer default whenever the SDs or the group sizes differ."
        if welch else
        "The unpaired t test compares the means of two independent groups when both are roughly "
        "normally distributed with similar spread (SD)."
    )
    return StatResult(
        analysis="ttest_welch" if welch else "ttest_unpaired",
        test_name=test_name,
        why_chosen=why,
        assumptions=assumptions,
        statistic_name="t",
        statistic=t,
        df=df_out,
        p_value=p,
        alpha=alpha,
        alternative=alternative,
        effect_sizes=[d, g],
        parameters=parameters,
        n={na: n1, nb: n2},
        descriptives=[_row(na, x, level), _row(nb, y, level)],
        warnings=warnings,
        data_notes=notes_a + notes_b,
        interpretation=interpretation,
        methods_text=methods,
        citations=[_CITE_WELCH if welch else _CITE_STUDENT, _CITE_D, _CITE_G],
        extra={"mean_difference": diff, "se_difference": se_out,
               "variance_assumption": "unequal (Welch)" if welch else "equal (pooled)"},
    )


def ttest_unpaired(a: Iterable[Any], b: Iterable[Any], *, names: tuple[str, str] = ("A", "B"),
                   alpha: float = 0.05, alternative: str = "two-sided", ci_level: float = 0.95) -> StatResult:
    """Student's unpaired t test (pooled variance, df = n1 + n2 - 2).

    t = (mean_a - mean_b) / sqrt(s_p^2 (1/n1 + 1/n2)). Reports the mean difference with
    its CI, Cohen's d and Hedges' g (noncentral-t CIs), and checks normality of each group
    (Shapiro-Wilk) and equality of variances (Brown-Forsythe); when the variances differ
    the assumption is marked ``fail`` and Welch's test is recommended.
    """
    return _independent_t(a, b, names, alpha, alternative, ci_level, welch=False)


def ttest_welch(a: Iterable[Any], b: Iterable[Any], *, names: tuple[str, str] = ("A", "B"),
                alpha: float = 0.05, alternative: str = "two-sided", ci_level: float = 0.95) -> StatResult:
    """Welch's (1947) unequal-variance t test.

    t = (mean_a - mean_b) / sqrt(s_a^2/n_a + s_b^2/n_b) with the Welch-Satterthwaite
    degrees of freedom, which are also used for the CI of the difference. Cohen's d and
    Hedges' g are standardised by the pooled SD (noted in their ``ci_method``).
    """
    return _independent_t(a, b, names, alpha, alternative, ci_level, welch=True)


# ============================================================ paired / one-sample t


def _pearson(x: np.ndarray, y: np.ndarray, level: float) -> dict[str, Any]:
    """Pearson r between pairs with two-sided P, one-sided P (r > 0) and Fisher-z CI."""
    out: dict[str, Any] = {"r": None, "p": None, "p_positive": None, "ci_low": None, "ci_high": None}
    n = x.size
    if n < 3 or is_constant(x) or is_constant(y):
        return out
    r = float(np.clip(np.corrcoef(x, y)[0, 1], -1.0, 1.0))
    out["r"] = r
    if abs(r) >= 1.0:
        out.update(p=0.0, p_positive=0.0 if r > 0 else 1.0, ci_low=r, ci_high=r)
        return out
    t = r * math.sqrt((n - 2) / (1.0 - r * r))
    out["p"] = float(min(1.0, 2.0 * stats.t.sf(abs(t), n - 2)))
    out["p_positive"] = float(stats.t.sf(t, n - 2))
    if n >= 4:
        z, half = math.atanh(r), float(stats.norm.ppf(0.5 + level / 2.0)) / math.sqrt(n - 3)
        out.update(ci_low=math.tanh(z - half), ci_high=math.tanh(z + half))
    return out


def _pairing_check(corr: dict[str, Any], label: str = "Pearson r") -> AssumptionCheck:
    name = "Effective pairing (correlation between pairs)"
    r, p1 = corr["r"], corr["p_positive"]
    if r is None:
        return AssumptionCheck(name=name, status="warn",
                               detail="The correlation between the paired values could not be computed "
                                      "(too few pairs, or one column is constant).")
    if r > 0 and p1 is not None and p1 < 0.05:
        return AssumptionCheck(
            name=name, statistic=r, p_value=p1, status="pass",
            detail=f"The paired values are positively correlated ({label} = {fmt(r, 3)}, one-tailed "
                   f"{p_phrase(p1)}), so pairing was effective.",
        )
    if r > 0:
        return AssumptionCheck(
            name=name, statistic=r, p_value=p1, status="info",
            detail=f"The paired values are only weakly correlated ({label} = {fmt(r, 3)}, one-tailed "
                   f"{p_phrase(p1)}); pairing gained little.",
        )
    return AssumptionCheck(
        name=name, statistic=r, p_value=p1, status="warn",
        detail=f"The paired values are not positively correlated ({label} = {fmt(r, 3)}). Pairing did "
               "not help here; check that the rows really are matched pairs.",
    )


def ttest_paired(a: Iterable[Any], b: Iterable[Any], *, names: tuple[str, str] = ("A", "B"),
                 alpha: float = 0.05, alternative: str = "two-sided", ci_level: float = 0.95) -> StatResult:
    """Paired t test: a one-sample t test on the within-pair differences (a - b).

    ``a`` and ``b`` must have the same length (row i of each is one subject); pairs with a
    missing value are dropped and counted in ``data_notes``. Checks normality of the
    differences, reports the correlation between pairs (how effective the pairing was),
    and Cohen's d_z / Hedges' g_z (mean difference / SD of the differences).
    """
    alpha = check_alpha(alpha)
    alternative = check_alternative(alternative)
    level = _check_level(ci_level)
    na, nb = _names(names)
    x, y, notes = clean_paired(a, b, (na, nb))
    if x.size < 2:
        raise StatsError(f"A paired t test needs at least 2 complete pairs; there are {x.size}.")
    n = int(x.size)
    d = x - y
    diff_name = f"Difference ({na} - {nb})"
    md = float(d.mean())
    sd = 0.0 if is_constant(d) else float(d.std(ddof=1))
    se = sd / math.sqrt(n)
    df = float(n - 1)
    warnings: list[str] = []
    t: float | None
    p: float | None
    if se > 0:
        t = md / se
        p = _t_p(t, df, alternative)
        half = float(stats.t.ppf(0.5 + level / 2.0, df)) * se
        ci_low, ci_high = md - half, md + half
        se_out: float | None = se
    else:
        t = p = ci_low = ci_high = se_out = None
        warnings.append(
            "Every pair has exactly the same difference, so there is no variability and the t "
            "statistic and P value cannot be computed."
        )
    if n < 3:
        warnings.append("Only 2 pairs: the test has very little power and normality cannot be checked.")
    warnings.extend(_one_sided_note(alternative, level))

    corr = _pearson(x, y, level)
    assumptions = [
        _normality_check(d, diff_name, "Wilcoxon matched-pairs signed-rank test"),
        _pairing_check(corr),
        _INDEPENDENT_PAIRS,
    ]
    dz = _tidy(es.cohen_d_paired(x, y, level=level))
    gz = _tidy(es.cohen_d_paired(x, y, level=level, hedges=True))
    parameters = [
        Parameter(name=f"Mean of differences ({na} - {nb})", value=md, se=se_out, ci_low=ci_low,
                  ci_high=ci_high, ci_level=level, statistic=t, p_value=p,
                  unit_note="Two-sided confidence interval."),
        Parameter(name="Correlation between pairs (Pearson r)", value=corr["r"], ci_low=corr["ci_low"],
                  ci_high=corr["ci_high"], ci_level=level, p_value=corr["p"],
                  unit_note="Two-sided P value; CI by the Fisher z transformation."),
    ]
    if t is None:
        interpretation = (
            f"In {n} pairs the difference {na} - {nb} was always {fmt(md)}. With no variability "
            "between pairs the paired t test cannot be computed."
        )
    else:
        interpretation = (
            f"Across {n} pairs, {na} was on average {fmt(abs(md))} {_direction(md)} than {nb} "
            f"(mean difference {na} - {nb} = {fmt(md)}, {ci_phrase(ci_low, ci_high, level)}). "
            if md != 0 else f"Across {n} pairs the mean difference between {na} and {nb} was exactly 0. "
        )
        interpretation += (
            f"By the paired t test{_tail_text(alternative, na, nb)} this is {sig_phrase(p, alpha)}; "
            f"t({fmt(df)}) = {fmt(t)}. "
        )
        interpretation += _evidence(
            p < alpha, alternative,
            ("there is a systematic difference within pairs", f"{na} is systematically higher than {nb}",
             f"{na} is systematically lower than {nb}"),
        )
        interpretation += _effect_sentence(dz)
    return StatResult(
        analysis="ttest_paired",
        test_name="Paired t test",
        why_chosen=(
            "The paired t test compares two measurements made on the same subjects (or matched "
            "pairs) by testing whether the mean of the within-pair differences is zero. It assumes "
            "those differences are roughly normally distributed."
        ),
        assumptions=assumptions,
        statistic_name="t",
        statistic=t,
        df=df,
        p_value=p,
        alpha=alpha,
        alternative=alternative,
        effect_sizes=[dz, gz],
        parameters=parameters,
        n={na: n, nb: n},
        n_total=n,
        descriptives=[_row(na, x, level), _row(nb, y, level), _row(diff_name, d, level)],
        warnings=warnings,
        data_notes=notes,
        interpretation=interpretation,
        methods_text=(f"Paired measurements were compared with a paired t test "
                      f"({alternative_label(alternative)}, alpha = {alpha:g})."),
        citations=[_CITE_STUDENT, _CITE_D, _CITE_G],
        extra={"differences": [float(v) for v in d], "n_pairs": n, "pearson_r": corr["r"]},
    )


def ttest_one_sample(x: Iterable[Any], popmean: float = 0.0, *, name: str = "Sample", alpha: float = 0.05,
                     alternative: str = "two-sided", ci_level: float = 0.95) -> StatResult:
    """One-sample t test: is the mean different from a hypothetical value ``popmean``?

    t = (mean - popmean) / (sd / sqrt(n)), df = n - 1. Reports the mean and the
    difference from the hypothetical value with CIs, and Cohen's d / Hedges' g.
    """
    alpha = check_alpha(alpha)
    alternative = check_alternative(alternative)
    level = _check_level(ci_level)
    mu = _float_param(popmean, "The hypothetical value")
    name = str(name)
    arr, notes = clean(x, name)
    require_n(arr, 2, name, "a one-sample t test")
    n = int(arr.size)
    mean = float(arr.mean())
    sd = 0.0 if is_constant(arr) else float(arr.std(ddof=1))
    se = sd / math.sqrt(n)
    df = float(n - 1)
    diff = mean - mu
    warnings: list[str] = []
    t: float | None
    p: float | None
    if se > 0:
        t = diff / se
        p = _t_p(t, df, alternative)
        half = float(stats.t.ppf(0.5 + level / 2.0, df)) * se
        se_out: float | None = se
        mean_ci = (mean - half, mean + half)
        diff_ci = (diff - half, diff + half)
    else:
        t = p = se_out = None
        mean_ci = diff_ci = (None, None)
        warnings.append("All values are identical, so there is no variability and the t statistic "
                        "and P value cannot be computed.")
    if n < 3:
        warnings.append("Only 2 values: the test has very little power and normality cannot be checked.")
    warnings.extend(_one_sided_note(alternative, level))
    d = _tidy(es.cohen_d_one_sample(arr, mu, level=level))
    g = _tidy(es.cohen_d_one_sample(arr, mu, level=level, hedges=True))
    assumptions = [_normality_check(arr, name, "Wilcoxon signed-rank test"), _INDEPENDENCE]
    parameters = [
        Parameter(name=f"Mean of {name}", value=mean, se=se_out, ci_low=mean_ci[0], ci_high=mean_ci[1],
                  ci_level=level, unit_note="Two-sided confidence interval."),
        Parameter(name="Difference from hypothetical value", value=diff, se=se_out, ci_low=diff_ci[0],
                  ci_high=diff_ci[1], ci_level=level, statistic=t, p_value=p,
                  unit_note=f"Mean minus {fmt(mu)}."),
    ]
    if t is None:
        interpretation = (
            f"Every value of {name} equals {fmt(mean)} (n = {n}). With no variability the one-sample "
            "t test cannot be computed."
        )
    else:
        interpretation = (
            f"The mean of {name} was {fmt(mean)} (n = {n}, {ci_phrase(mean_ci[0], mean_ci[1], level)}), "
            f"which is {fmt(abs(diff))} {_direction(diff)} than the hypothetical value {fmt(mu)}. "
            if diff != 0 else
            f"The mean of {name} (n = {n}) equals the hypothetical value {fmt(mu)} exactly. "
        )
        interpretation += (
            f"By the one-sample t test{_tail_text(alternative, 'mean', fmt(mu))} this difference is "
            f"{sig_phrase(p, alpha)}; t({fmt(df)}) = {fmt(t)}. "
        )
        interpretation += _evidence(
            p < alpha, alternative,
            (f"the population mean differs from {fmt(mu)}", f"the population mean is higher than {fmt(mu)}",
             f"the population mean is lower than {fmt(mu)}"),
        )
        interpretation += _effect_sentence(d)
    return StatResult(
        analysis="ttest_one_sample",
        test_name="One-sample t test",
        why_chosen=(
            "The one-sample t test compares the mean of one group with a fixed hypothetical value "
            "(for example 0, 1 or 100%), assuming the values are roughly normally distributed."
        ),
        assumptions=assumptions,
        statistic_name="t",
        statistic=t,
        df=df,
        p_value=p,
        alpha=alpha,
        alternative=alternative,
        effect_sizes=[d, g],
        parameters=parameters,
        n={name: n},
        descriptives=[_row(name, arr, level)],
        warnings=warnings,
        data_notes=notes,
        interpretation=interpretation,
        methods_text=(f"The mean was compared with the hypothetical value {fmt(mu)} using a one-sample "
                      f"t test ({alternative_label(alternative)}, alpha = {alpha:g})."),
        citations=[_CITE_STUDENT, _CITE_D, _CITE_G],
        extra={"hypothetical_value": mu},
    )


# ================================================================== Mann-Whitney


def mann_whitney_null_pmf(n1: int, n2: int) -> np.ndarray:
    """Exact null distribution of the Mann-Whitney U statistic without ties.

    Returns ``pmf`` with ``pmf[u] = P(U = u)`` for u = 0 .. n1*n2. Under the null
    hypothesis every choice of n1 ranks out of n1 + n2 is equally likely; the number of
    subsets with a given rank sum is counted by dynamic programming, and
    U = rank sum - n1 (n1 + 1) / 2.
    """
    if n1 < 1 or n2 < 1:
        raise ValueError("n1 and n2 must be at least 1.")
    total = n1 + n2
    max_sum = n1 * total
    counts = np.zeros((n1 + 1, max_sum + 1))
    counts[0, 0] = 1.0
    for rank in range(1, total + 1):
        for size in range(min(rank, n1), 0, -1):
            counts[size, rank:] += counts[size - 1, : max_sum + 1 - rank]
    low = n1 * (n1 + 1) // 2
    pmf = counts[n1, low : low + n1 * n2 + 1]
    return pmf / pmf.sum()


def signed_rank_null_pmf(ranks: Iterable[int]) -> np.ndarray:
    """Exact null distribution of W+ (sum of the positive ranks) for integer ``ranks``.

    Under the null hypothesis each rank is positive or negative with probability 1/2,
    independently. ``pmf[w] = P(W+ = w)`` for w = 0 .. sum(ranks).
    """
    rank_list = [int(r) for r in ranks]
    total = int(sum(rank_list))
    counts = np.zeros(total + 1)
    counts[0] = 1.0
    for r in rank_list:
        shifted = np.zeros_like(counts)
        shifted[r:] = counts[: total + 1 - r]
        counts = counts + shifted
    return counts / counts.sum()


def _exact_tail_p(pmf: np.ndarray, observed: float, alternative: str) -> float:
    k = int(round(observed))
    p_le = float(pmf[: k + 1].sum())
    p_ge = float(pmf[k:].sum())
    if alternative == "greater":
        return min(1.0, p_ge)
    if alternative == "less":
        return min(1.0, p_le)
    return min(1.0, 2.0 * min(p_le, p_ge))


def _normal_tail_p(stat: float, mean: float, sd: float, alternative: str) -> tuple[float, float]:
    """(z, p) with a continuity correction of 0.5 towards the mean."""
    if alternative == "greater":
        z = (stat - mean - 0.5) / sd
        return z, float(stats.norm.sf(z))
    if alternative == "less":
        z = (stat - mean + 0.5) / sd
        return z, float(stats.norm.cdf(z))
    z_abs = max(abs(stat - mean) - 0.5, 0.0) / sd
    p = float(min(1.0, 2.0 * stats.norm.sf(z_abs)))
    return math.copysign(z_abs, stat - mean), p


def _order_stat_ci(sorted_values: np.ndarray, pmf: np.ndarray | None, normal_mean: float,
                   normal_sd: float, level: float) -> tuple[float | None, float | None, float | None, str]:
    """Distribution-free CI from order statistics of pairwise differences / Walsh averages.

    With M sorted values and null statistic T (U or W+): the interval
    [v_(k), v_(M + 1 - k)] has coverage P(k <= T <= M - k) = 1 - 2 P(T <= k - 1). ``k`` is
    the largest integer with P(T <= k - 1) <= (1 - level) / 2, from the exact null
    distribution ``pmf`` when given, otherwise from the normal approximation with
    continuity correction: k = floor(mean + 0.5 - z * sd).
    Returns (low, high, achieved level or None, method text).
    """
    m = int(sorted_values.size)
    tail = (1.0 - level) / 2.0
    if pmf is not None:
        cdf = np.cumsum(pmf)
        below = np.nonzero(cdf <= tail + 1e-12)[0]
        if below.size == 0:
            return None, None, None, "exact"
        k = int(below[-1]) + 1
        achieved: float | None = float(1.0 - 2.0 * cdf[k - 1])
        method = "exact, from order statistics of the pairwise differences (Bauer 1972)"
    else:
        if not normal_sd > 0:
            return None, None, None, "normal approximation"
        k = int(math.floor(normal_mean + 0.5 - float(stats.norm.ppf(1.0 - tail)) * normal_sd))
        achieved = None
        method = ("normal approximation to the rank of the order statistics (Hollander & Wolfe 1999), "
                  "with continuity correction; approximate when there are ties")
    if k < 1 or k > m:
        return None, None, achieved, method
    return float(sorted_values[k - 1]), float(sorted_values[m - k]), achieved, method


def mann_whitney(a: Iterable[Any], b: Iterable[Any], *, names: tuple[str, str] = ("A", "B"),
                 alpha: float = 0.05, alternative: str = "two-sided", ci_level: float = 0.95,
                 method: str = "auto") -> StatResult:
    """Mann-Whitney U test (Wilcoxon rank-sum test) for two independent groups.

    * ``statistic`` is **U = the smaller of U_A and U_B** (the conventional tabulated
      value, unaffected by which group is listed first). ``extra`` holds ``u_a`` (number of
      (a, b) pairs with a > b, ties counting 1/2), ``u_b`` = n_a n_b - u_a, the rank sums
      and the mean ranks.
    * P value: exact (``method="exact"``, the default choice of ``"auto"`` when there are
      no ties and both groups have at most 50 values) or the normal approximation with tie
      correction and continuity correction (``"asymptotic"``). ``extra["p_method"]`` says
      which was used.
    * Location shift: the Hodges-Lehmann estimator (median of all n_a * n_b pairwise
      differences a_i - b_j) with a distribution-free CI from the order statistics of those
      differences (exact without ties, otherwise a normal approximation for the ranks).
      Note that this is not the same number as the difference between the two medians.
    * Effect size: rank-biserial r = 2 U_A / (n_a n_b) - 1.

    The test compares the two distributions as a whole (it asks whether values in one
    group tend to be larger than in the other, i.e. it compares mean ranks). It is a
    test of medians only if the two distributions have the same shape and spread.
    """
    alpha = check_alpha(alpha)
    alternative = check_alternative(alternative)
    level = _check_level(ci_level)
    if method not in {"auto", "exact", "asymptotic"}:
        raise StatsError("method must be 'auto', 'exact' or 'asymptotic'.")
    na, nb = _names(names)
    x, notes_a = clean(a, na)
    y, notes_b = clean(b, nb)
    require_n(x, 2, na, "the Mann-Whitney test")
    require_n(y, 2, nb, "the Mann-Whitney test")
    n1, n2 = int(x.size), int(y.size)
    n_pairs = n1 * n2
    total = n1 + n2
    pooled = np.concatenate([x, y])
    ranks = stats.rankdata(pooled)
    r1, r2 = float(ranks[:n1].sum()), float(ranks[n1:].sum())
    u_a = r1 - n1 * (n1 + 1) / 2.0
    u_b = n_pairs - u_a
    u = min(u_a, u_b)
    _, tie_counts = np.unique(pooled, return_counts=True)
    has_ties = bool(np.any(tie_counts > 1))
    tie_term = float(np.sum(tie_counts.astype(float) ** 3 - tie_counts))
    mean_u = n_pairs / 2.0
    var_u = n_pairs / 12.0 * ((total + 1.0) - tie_term / (total * (total - 1.0)))
    sd_u = math.sqrt(var_u) if var_u > 0 else 0.0

    warnings: list[str] = []
    can_exact = (not has_ties) and max(n1, n2) <= EXACT_MAX_N
    use_exact = can_exact and method in {"auto", "exact"}
    if method == "exact" and not can_exact:
        warnings.append(
            "An exact P value was requested but is not available "
            + ("because there are tied values" if has_ties else f"for groups larger than {EXACT_MAX_N}")
            + "; the normal approximation was used instead."
        )
    pmf: np.ndarray | None = None
    z: float | None = None
    p: float | None
    if use_exact:
        pmf = mann_whitney_null_pmf(n1, n2)
        p = _exact_tail_p(pmf, u_a, alternative)
        p_method = "exact"
    elif sd_u > 0:
        z, p = _normal_tail_p(u_a, mean_u, sd_u, alternative)
        p_method = "asymptotic (normal approximation with tie correction and continuity correction)"
    else:
        p = 1.0
        p_method = "all values identical"
        warnings.append(
            "Every value in both groups is identical, so the groups cannot be told apart (P = 1)."
        )
    if has_ties and sd_u > 0:
        n_tied = int(tie_counts[tie_counts > 1].sum())
        warnings.append(
            f"There are tied values ({n_tied} of {total} values share a value with another), so an "
            "exact P value is not available; the normal approximation with tie correction was used."
            + (" With samples this small the approximation can be rough." if min(n1, n2) < 8 else "")
        )
    # smallest attainable exact two-sided P value, 2 / C(N, n1); only relevant for tiny samples
    min_p = 2.0 / math.comb(total, n1) if total <= 60 else 0.0
    if alternative == "two-sided" and min_p > alpha:
        warnings.append(
            f"With {n1} and {n2} values the smallest possible two-tailed exact P value is "
            f"{fmt(min_p, 3)}, so this test can never reach significance at alpha = {alpha:g}."
        )
    warnings.extend(_one_sided_note(alternative, level))

    # ---- Hodges-Lehmann shift estimate and CI
    hl: float | None = None
    hl_low = hl_high = achieved = None
    ci_method = None
    if n_pairs <= _MAX_PAIRS:
        diffs = np.sort(np.subtract.outer(x, y).ravel())
        hl = float(np.median(diffs))
        hl_low, hl_high, achieved, ci_method = _order_stat_ci(
            diffs, pmf, mean_u, math.sqrt(n_pairs * (total + 1.0) / 12.0), level
        )
        if hl_low is None:
            warnings.append(
                f"The samples are too small for a {level * 100:g}% confidence interval of the "
                "Hodges-Lehmann difference."
            )
    else:
        warnings.append("The groups are too large to compute the Hodges-Lehmann estimate "
                        f"(more than {_MAX_PAIRS:,} pairs).")
    med_a, med_b = float(np.median(x)), float(np.median(y))

    rb = _tidy(es.rank_biserial_independent(x, y, level=level))
    bf = _spread_check(x, y)
    shape_name = "Similar distribution shape (needed to interpret the result as a difference in medians)"
    if bf["p_value"] is None:
        shape = AssumptionCheck(name=shape_name, status="info",
                                detail="Could not be checked. Without it, read the result as 'values in one "
                                       "group tend to be larger', not as a difference in medians.")
    elif bf["p_value"] < 0.05:
        shape = AssumptionCheck(
            name=shape_name, statistic=bf["statistic"], p_value=bf["p_value"], status="warn",
            detail=f"The spread of the two groups differs (Brown-Forsythe {p_phrase(bf['p_value'])}). The "
                   "test is still valid as a comparison of the two distributions, but do not interpret "
                   "it as a difference in medians.",
        )
    else:
        shape = AssumptionCheck(
            name=shape_name, statistic=bf["statistic"], p_value=bf["p_value"], status="pass",
            detail=f"No evidence that the spread differs (Brown-Forsythe {p_phrase(bf['p_value'])}).",
        )
    assumptions = [
        _INDEPENDENCE,
        AssumptionCheck(name="Ordinal or continuous data", status="info",
                        detail="Values only need to be rankable; no normal distribution is assumed."),
        shape,
    ]
    parameters = [
        Parameter(name=f"Hodges-Lehmann difference ({na} - {nb})", value=hl, ci_low=hl_low, ci_high=hl_high,
                  ci_level=level,
                  unit_note="Median of all pairwise differences; " + (f"CI: {ci_method}" if ci_method else "no CI")
                            + (f" (actual coverage {achieved * 100:.2f}%)." if achieved is not None else ".")),
        Parameter(name=f"Difference between medians ({na} - {nb})", value=med_a - med_b,
                  unit_note="Shown for reference; it has no confidence interval and is not what the test examines."),
    ]
    mean_rank_a, mean_rank_b = r1 / n1, r2 / n2
    if u_a > u_b:
        tendency = f"values in {na} tended to be larger than in {nb}"
    elif u_a < u_b:
        tendency = f"values in {na} tended to be smaller than in {nb}"
    else:
        tendency = f"values in {na} and {nb} were evenly interleaved"
    interpretation = (
        f"By the Mann-Whitney test{_tail_text(alternative, na, nb)}, {tendency} (median {fmt(med_a)} vs "
        f"{fmt(med_b)}; mean rank {fmt(mean_rank_a)} vs {fmt(mean_rank_b)}; n = {n1} and {n2}); the "
        f"difference between the two distributions is {sig_phrase(p, alpha)}; U = {fmt(u)}, "
        f"{'exact' if use_exact else 'approximate'} P value. "
    )
    if hl is not None:
        interpretation += f"The Hodges-Lehmann estimate of the shift {na} - {nb} is {fmt(hl)}"
        interpretation += (f" ({ci_phrase(hl_low, hl_high, level)}). " if hl_low is not None else ". ")
    interpretation += _effect_sentence(rb)
    interpretation += (
        " The test compares the distributions (mean ranks); it compares medians only if both groups "
        "have distributions of the same shape."
    )
    return StatResult(
        analysis="mann_whitney",
        test_name="Mann-Whitney U test",
        why_chosen=(
            "The Mann-Whitney test compares two independent groups using ranks, so it does not "
            "assume a normal distribution. It asks whether values in one group tend to be larger "
            "than in the other (it compares the distributions via mean ranks); it is a comparison "
            "of medians only when the two distributions have the same shape."
        ),
        assumptions=assumptions,
        statistic_name="U",
        statistic=u,
        p_value=p,
        alpha=alpha,
        alternative=alternative,
        effect_sizes=[rb],
        parameters=parameters,
        n={na: n1, nb: n2},
        descriptives=[
            _row(na, x, level, rank_sum=r1, mean_rank=mean_rank_a),
            _row(nb, y, level, rank_sum=r2, mean_rank=mean_rank_b),
        ],
        warnings=warnings,
        data_notes=notes_a + notes_b,
        interpretation=interpretation,
        methods_text=(
            f"The two groups were compared with the Mann-Whitney U test ({alternative_label(alternative)}, "
            f"alpha = {alpha:g}; {'exact P value' if use_exact else 'normal approximation with tie and continuity correction'}); "
            "the shift was estimated with the Hodges-Lehmann estimator."
        ),
        citations=[_CITE_MWU, _CITE_HL, _CITE_BAUER, _CITE_RB],
        extra={
            "u_a": u_a, "u_b": u_b, "u_min": u, "statistic_reported": "U = smaller of U_A and U_B",
            "rank_sum_a": r1, "rank_sum_b": r2, "mean_rank_a": mean_rank_a, "mean_rank_b": mean_rank_b,
            "z": z, "p_method": p_method, "exact": bool(use_exact), "ties": has_ties,
            "probability_of_superiority": u_a / n_pairs,
            "hodges_lehmann_ci_method": ci_method, "hodges_lehmann_ci_coverage": achieved,
        },
    )


# ============================================================ Wilcoxon signed-rank


def _signed_rank_result(
    d: np.ndarray, *, analysis: str, test_name: str, why: str, label: str, estimate_name: str,
    estimate_offset: float, alpha: float, alternative: str, level: float, zero_method: str, method: str,
    tail_names: tuple[str, str], n_label: dict[str, int], rows: list[dict[str, Any]], notes: list[str],
    assumptions: list[AssumptionCheck], parameters_extra: list[Parameter], methods_prefix: str,
    extra_more: dict[str, Any],
) -> StatResult:
    """Shared engine of the paired and one-sample Wilcoxon signed-rank tests.

    ``d`` are the differences (a - b, or x - hypothetical median).
    """
    if zero_method not in {"wilcox", "pratt"}:
        raise StatsError("zero_method must be 'wilcox' (discard zero differences) or 'pratt'.")
    if method not in {"auto", "exact", "asymptotic"}:
        raise StatsError("method must be 'auto', 'exact' or 'asymptotic'.")
    n_all = int(d.size)
    nonzero = d != 0
    n_zero = int(n_all - nonzero.sum())
    dn = d[nonzero]
    n_r = int(dn.size)
    if zero_method == "wilcox":
        ranks = stats.rankdata(np.abs(dn)) if n_r else np.array([])
    else:
        ranks = stats.rankdata(np.abs(d))[nonzero]
    w_plus = float(ranks[dn > 0].sum())
    w_minus = float(ranks[dn < 0].sum())
    w = w_plus - w_minus
    _, tie_counts = np.unique(np.abs(dn), return_counts=True)
    has_ties = bool(np.any(tie_counts > 1))
    tie_term = float(np.sum(tie_counts.astype(float) ** 3 - tie_counts))
    if zero_method == "wilcox":
        mean_w = n_r * (n_r + 1) / 4.0
        var_w = n_r * (n_r + 1) * (2 * n_r + 1) / 24.0 - tie_term / 48.0
    else:
        mean_w = (n_all * (n_all + 1) - n_zero * (n_zero + 1)) / 4.0
        var_w = (n_all * (n_all + 1) * (2 * n_all + 1) - n_zero * (n_zero + 1) * (2 * n_zero + 1)) / 24.0 \
            - tie_term / 48.0
    sd_w = math.sqrt(var_w) if var_w > 0 else 0.0

    warnings: list[str] = []
    can_exact = n_r >= 1 and not has_ties and n_r <= EXACT_MAX_N
    use_exact = can_exact and method in {"auto", "exact"}
    if method == "exact" and not can_exact and n_r >= 1:
        warnings.append(
            "An exact P value was requested but is not available "
            + ("because some differences are tied in size" if has_ties
               else f"for more than {EXACT_MAX_N} non-zero differences")
            + "; the normal approximation was used instead."
        )
    pmf: np.ndarray | None = None
    z: float | None = None
    p: float | None
    statistic: float | None = w
    if n_r == 0:
        p = None
        statistic = None
        p_method = "not computed"
        warnings.append(
            "Every difference is exactly zero, so there is nothing to rank and the test cannot be computed."
        )
    elif use_exact:
        pmf = signed_rank_null_pmf(np.rint(ranks).astype(int))
        p = _exact_tail_p(pmf, w_plus, alternative)
        p_method = "exact"
        if n_zero and zero_method == "wilcox":
            p_method = "exact (zero differences discarded)"
        elif n_zero:
            p_method = "exact (conditional on the ranks of the non-zero differences, Pratt's method)"
    else:
        z, p = _normal_tail_p(w_plus, mean_w, sd_w, alternative)
        p_method = "asymptotic (normal approximation with tie correction and continuity correction)"
    if has_ties and n_r:
        warnings.append(
            "Some differences are tied in size, so an exact P value is not available; the normal "
            "approximation with tie correction was used."
            + (" With so few pairs the approximation can be rough." if n_r < 10 else "")
        )
    if n_zero and n_r:
        warnings.append(
            f"{n_zero} difference(s) are exactly zero. "
            + ("They were discarded before ranking (Wilcoxon's method)." if zero_method == "wilcox"
               else "They were ranked with the others and their ranks then dropped (Pratt's method).")
        )
    if 0 < n_r < 6 and alternative == "two-sided":
        warnings.append(
            f"With only {n_r} non-zero difference(s) the smallest possible two-tailed exact P value is "
            f"{fmt(2.0 / 2 ** n_r, 3)}; the test cannot reach significance at alpha = 0.05."
        )
    warnings.extend(_one_sided_note(alternative, level))

    # ---- Hodges-Lehmann pseudo-median of the differences (median of Walsh averages)
    hl = hl_low = hl_high = achieved = None
    ci_method = None
    n_walsh = n_all * (n_all + 1) // 2
    if n_walsh <= _MAX_PAIRS:
        i, j = np.triu_indices(n_all)
        walsh = np.sort((d[i] + d[j]) / 2.0)
        hl = float(np.median(walsh))
        ci_pmf = None
        if n_zero == 0 and not has_ties and n_all <= EXACT_MAX_N:
            ci_pmf = pmf if pmf is not None else signed_rank_null_pmf(range(1, n_all + 1))
        hl_low, hl_high, achieved, ci_method = _order_stat_ci(
            walsh, ci_pmf, n_all * (n_all + 1) / 4.0,
            math.sqrt(n_all * (n_all + 1) * (2 * n_all + 1) / 24.0), level,
        )
        if ci_method:
            ci_method = ci_method.replace("pairwise differences", "Walsh averages")
        if hl_low is None:
            warnings.append(
                f"There are too few values for a {level * 100:g}% confidence interval of the "
                "Hodges-Lehmann estimate."
            )
    else:
        warnings.append("There are too many values to compute the Hodges-Lehmann estimate.")

    rb = _tidy(es.rank_biserial_paired(d, level=level))
    skew = _describe_clean(d, level)["skewness"] if n_all else None
    sym_detail = (
        "The test assumes the differences are distributed symmetrically around their median. "
        "This cannot be verified from a small sample"
    )
    sym_detail += f" (sample skewness of the differences: {fmt(skew, 3)})." if skew is not None else "."
    assumptions = [
        AssumptionCheck(name="Symmetric distribution of the differences", status="info", detail=sym_detail),
        *assumptions,
    ]
    shift = None if hl is None else hl + estimate_offset
    parameters = [
        Parameter(
            name=estimate_name, value=shift,
            ci_low=None if hl_low is None else hl_low + estimate_offset,
            ci_high=None if hl_high is None else hl_high + estimate_offset, ci_level=level,
            unit_note="Hodges-Lehmann estimator (median of the Walsh averages, all values included); "
                      + (f"CI: {ci_method}" if ci_method else "no CI")
                      + (f" (actual coverage {achieved * 100:.2f}%)." if achieved is not None else "."),
        ),
        *parameters_extra,
    ]
    first, second = tail_names
    if p is None:
        interpretation = (
            f"All {n_all} differences between {first} and {second} are exactly zero, so the Wilcoxon "
            "signed-rank test cannot be computed; there is no sign of any difference."
        )
    else:
        if w > 0:
            tendency = f"{first} tended to be larger than {second}"
        elif w < 0:
            tendency = f"{first} tended to be smaller than {second}"
        else:
            tendency = f"positive and negative differences between {first} and {second} balanced out"
        interpretation = (
            f"By the {label}{_tail_text(alternative, first, second)}, {tendency} "
            f"(sum of positive ranks {fmt(w_plus)}, sum of negative ranks {fmt(w_minus)}, "
            f"{n_r} non-zero differences); this is {sig_phrase(p, alpha)}; W = {fmt(w)}, "
            f"{'exact' if use_exact else 'approximate'} P value. "
        )
        if shift is not None:
            interpretation += f"{estimate_name}: {fmt(shift)}"
            interpretation += (
                f" ({ci_phrase(hl_low + estimate_offset, hl_high + estimate_offset, level)}). "
                if hl_low is not None else ". "
            )
        interpretation += _effect_sentence(rb)
    citations = [_CITE_WILCOXON, _CITE_HL, _CITE_BAUER, _CITE_RB]
    if zero_method == "pratt":
        citations.insert(1, _CITE_PRATT)
    return StatResult(
        analysis=analysis,
        test_name=test_name,
        why_chosen=why,
        assumptions=assumptions,
        statistic_name="W",
        statistic=statistic,
        p_value=p,
        alpha=alpha,
        alternative=alternative,
        effect_sizes=[rb],
        parameters=parameters,
        n=n_label,
        n_total=n_all,
        descriptives=rows,
        warnings=warnings,
        data_notes=notes,
        interpretation=interpretation,
        methods_text=(
            methods_prefix
            + ("; exact P value" if use_exact else "; normal approximation with tie and continuity correction")
            + ("; zero differences discarded)." if zero_method == "wilcox"
               else "; zero differences handled by Pratt's method).")
        ),
        citations=citations,
        extra={
            "w_plus": w_plus, "w_minus": w_minus, "w": w, "t_min": min(w_plus, w_minus),
            "statistic_reported": "W = sum of signed ranks (W+ minus W-)",
            "n_nonzero": n_r, "n_zero": n_zero, "z": z, "p_method": p_method, "exact": bool(use_exact),
            "ties": has_ties, "zero_method": zero_method,
            "hodges_lehmann_ci_method": ci_method, "hodges_lehmann_ci_coverage": achieved,
            **extra_more,
        },
    )


def _spearman(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {"r": None, "p": None, "p_positive": None}
    if x.size < 3 or is_constant(x) or is_constant(y):
        return out
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    r = float(np.clip(np.corrcoef(rx, ry)[0, 1], -1.0, 1.0))
    out["r"] = r
    if abs(r) >= 1.0:
        out.update(p=0.0, p_positive=0.0 if r > 0 else 1.0)
        return out
    t = r * math.sqrt((x.size - 2) / (1.0 - r * r))
    out["p"] = float(min(1.0, 2.0 * stats.t.sf(abs(t), x.size - 2)))
    out["p_positive"] = float(stats.t.sf(t, x.size - 2))
    return out


def wilcoxon_signed_rank(a: Iterable[Any], b: Iterable[Any], *, names: tuple[str, str] = ("A", "B"),
                         alpha: float = 0.05, alternative: str = "two-sided", ci_level: float = 0.95,
                         zero_method: str = "wilcox", method: str = "auto") -> StatResult:
    """Wilcoxon matched-pairs signed-rank test on the differences a - b.

    * ``statistic`` is **W = W+ - W-**, the sum of the signed ranks; ``extra`` holds
      ``w_plus``, ``w_minus`` and ``t_min`` (the smaller of the two, the value found in
      classical tables).
    * ``zero_method``: ``"wilcox"`` (default) discards pairs whose difference is exactly
      zero; ``"pratt"`` ranks them with the others and then drops their ranks (Pratt 1959).
    * P value: exact when no two non-zero differences have the same size and there are at
      most 50 of them; otherwise the normal approximation with tie correction and
      continuity correction. ``extra["p_method"]`` says which.
    * Estimate: the Hodges-Lehmann pseudo-median of the differences (median of all Walsh
      averages (d_i + d_j) / 2, i <= j, using every pair including zero differences) with a
      distribution-free CI (exact without ties or zeros, else normal approximation).
    * Effect size: matched-pairs rank-biserial r = (W+ - W-) / (W+ + W-).
    """
    alpha = check_alpha(alpha)
    alternative = check_alternative(alternative)
    level = _check_level(ci_level)
    na, nb = _names(names)
    x, y, notes = clean_paired(a, b, (na, nb))
    if x.size < 2:
        raise StatsError(f"The Wilcoxon signed-rank test needs at least 2 complete pairs; there are {x.size}.")
    d = x - y
    n = int(x.size)
    diff_name = f"Difference ({na} - {nb})"
    corr = _spearman(x, y)
    return _signed_rank_result(
        d,
        analysis="wilcoxon_signed_rank",
        test_name="Wilcoxon matched-pairs signed-rank test",
        why="The Wilcoxon matched-pairs signed-rank test compares two measurements on the same subjects "
            "(or matched pairs) using the ranks of the within-pair differences. It does not assume "
            "normality, only that the differences are distributed symmetrically.",
        label="Wilcoxon matched-pairs signed-rank test",
        estimate_name=f"Hodges-Lehmann median difference ({na} - {nb})",
        estimate_offset=0.0,
        alpha=alpha, alternative=alternative, level=level, zero_method=zero_method, method=method,
        tail_names=(na, nb),
        n_label={na: n, nb: n},
        rows=[_row(na, x, level), _row(nb, y, level), _row(diff_name, d, level)],
        notes=notes,
        assumptions=[_pairing_check(corr, "Spearman r"), _INDEPENDENT_PAIRS],
        parameters_extra=[
            Parameter(name="Median of differences", value=float(np.median(d)),
                      unit_note="Plain sample median of the differences, for reference."),
            Parameter(name="Correlation between pairs (Spearman r)", value=corr["r"], p_value=corr["p"],
                      unit_note="Two-sided P value (t approximation)."),
        ],
        methods_prefix=(f"Paired measurements were compared with the Wilcoxon matched-pairs signed-rank test "
                        f"({alternative_label(alternative)}, alpha = {alpha:g}"),
        extra_more={"differences": [float(v) for v in d], "n_pairs": n, "spearman_r": corr["r"]},
    )


def wilcoxon_one_sample(x: Iterable[Any], hypothetical_median: float = 0.0, *, name: str = "Sample",
                        alpha: float = 0.05, alternative: str = "two-sided", ci_level: float = 0.95,
                        zero_method: str = "wilcox", method: str = "auto") -> StatResult:
    """One-sample Wilcoxon signed-rank test against a hypothetical median.

    The signed-rank test is applied to x - ``hypothetical_median`` (see
    :func:`wilcoxon_signed_rank` for the statistic, P value and zero handling). The
    reported estimate is the Hodges-Lehmann pseudo-median of the sample with its CI; it
    equals the median when the distribution is symmetric.
    """
    alpha = check_alpha(alpha)
    alternative = check_alternative(alternative)
    level = _check_level(ci_level)
    m0 = _float_param(hypothetical_median, "The hypothetical median")
    name = str(name)
    arr, notes = clean(x, name)
    require_n(arr, 2, name, "the Wilcoxon signed-rank test")
    d = arr - m0
    median = float(np.median(arr))
    return _signed_rank_result(
        d,
        analysis="wilcoxon_one_sample",
        test_name="Wilcoxon signed-rank test (one sample)",
        why="The one-sample Wilcoxon signed-rank test compares the median of one group with a fixed "
            "hypothetical value without assuming normality; it assumes the values are distributed "
            "symmetrically around their median.",
        label="one-sample Wilcoxon signed-rank test",
        estimate_name=f"Hodges-Lehmann pseudo-median of {name}",
        estimate_offset=m0,
        alpha=alpha, alternative=alternative, level=level, zero_method=zero_method, method=method,
        tail_names=(name, f"the hypothetical median {fmt(m0)}"),
        n_label={name: int(arr.size)},
        rows=[_row(name, arr, level)],
        notes=notes,
        assumptions=[_INDEPENDENCE],
        parameters_extra=[
            Parameter(name=f"Median of {name}", value=median,
                      unit_note="Plain sample median, for reference."),
            Parameter(name="Median minus hypothetical value", value=median - m0),
        ],
        methods_prefix=(f"The median was compared with the hypothetical value {fmt(m0)} using the Wilcoxon "
                        f"signed-rank test ({alternative_label(alternative)}, alpha = {alpha:g}"),
        extra_more={"hypothetical_value": m0},
    )


# ------------------------------------------------------------------ table adapters

_ALPHA = ParamSpec(
    "alpha", "Significance level (alpha)", kind="float", default=0.05,
    help="The P value threshold for calling a result statistically significant. 0.05 is the usual choice.",
)
_ALTERNATIVE = ParamSpec(
    "alternative", "Tails", kind="choice", default="two-sided", choices=("two-sided", "greater", "less"),
    help="'two-sided' asks whether the groups differ in either direction (the usual choice). 'greater' "
         "asks only whether the first group is larger; 'less' only whether it is smaller. Choose a "
         "one-sided test only if you decided the direction before seeing the data.",
)
_CI_LEVEL = ParamSpec(
    "ci_level", "Confidence level", kind="float", default=0.95,
    help="Confidence level of the intervals shown. 0.95 gives the usual 95% confidence intervals.",
)
_GROUP_A = ParamSpec(
    "group_a", "First group", kind="group", default=None, optional=True,
    help="Which column is the first group (A). Leave empty to use the first column.",
)
_GROUP_B = ParamSpec(
    "group_b", "Second group", kind="group", default=None, optional=True,
    help="Which column is the second group (B). Leave empty to use the second column. Differences "
         "are reported as A minus B.",
)
_GROUP = ParamSpec(
    "group", "Group", kind="group", default=None, optional=True,
    help="Which column to test. Leave empty to use the first column.",
)
_HYPOTHETICAL = ParamSpec(
    "hypothetical_value", "Hypothetical value", kind="float", default=0.0,
    help="The fixed value to compare your data with, for example 0 (no change), 1 (a ratio of one) "
         "or 100 (percent of control).",
)
_ZERO_METHOD = ParamSpec(
    "zero_method", "Zero differences", kind="choice", default="wilcox", choices=("wilcox", "pratt"),
    help="What to do with differences that are exactly zero. 'wilcox' ignores them (the classic "
         "method); 'pratt' keeps them when ranking, which is slightly more conservative.",
)


def _pick_two(table: DataTable, group_a: Any, group_b: Any) -> tuple[str, str]:
    names = table.group_names
    for chosen in (group_a, group_b):
        if chosen not in (None, "") and str(chosen) not in names:
            raise StatsError(f"Unknown group '{chosen}'. Available groups: {names}.")
    a = str(group_a) if group_a not in (None, "") else None
    b = str(group_b) if group_b not in (None, "") else None
    if a is None:
        a = next((n for n in names if n != b), None)
    if b is None:
        b = next((n for n in names if n != a), None)
    if a is None or b is None:
        raise StatsError("This analysis compares two groups, but the table has only one column. "
                         "Add a second column of values.")
    if a == b:
        raise StatsError("Choose two different columns to compare.")
    return a, b


def _pick_one(table: DataTable, group: Any) -> str:
    names = table.group_names
    if group in (None, ""):
        return names[0]
    if str(group) not in names:
        raise StatsError(f"Unknown group '{group}'. Available groups: {names}.")
    return str(group)


def _paired_columns(table: DataTable, a: str, b: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    wide = table.analysis_frame()[[a, b]]
    n_any = int(wide.notna().any(axis=1).sum())
    frame = table.paired_frame([a, b])
    n_complete = int(len(frame))
    dropped = n_any - n_complete
    if dropped:
        note = (f"{dropped} incomplete pair(s) (rows with a value in only one of {a} and {b}) were "
                f"left out; {n_complete} complete pair(s) were analysed.")
    else:
        note = f"All {n_complete} pair(s) were complete; no incomplete pairs were dropped."
    return frame[a].to_numpy(dtype=float), frame[b].to_numpy(dtype=float), [note]


@register(
    key="ttest_unpaired", label="Unpaired t test", category="Compare two groups", tables=("column",),
    params=(_GROUP_A, _GROUP_B, _ALPHA, _ALTERNATIVE, _CI_LEVEL),
    description="Compares the means of two independent groups, assuming both are roughly normal with "
                "similar spread.",
    citation="Student (1908) Biometrika 6:1-25",
)
def _run_ttest_unpaired(table: DataTable, group_a: Any = None, group_b: Any = None, alpha: float = 0.05,
                        alternative: str = "two-sided", ci_level: float = 0.95, **_: Any) -> StatResult:
    a, b = _pick_two(table, group_a, group_b)
    groups = table.groups()
    return ttest_unpaired(groups[a], groups[b], names=(a, b), alpha=alpha, alternative=alternative,
                          ci_level=ci_level)


@register(
    key="ttest_welch", label="Welch's t test", category="Compare two groups", tables=("column",),
    params=(_GROUP_A, _GROUP_B, _ALPHA, _ALTERNATIVE, _CI_LEVEL),
    description="Compares the means of two independent groups without assuming that they have the same "
                "spread. A safe default when the SDs or group sizes differ.",
    citation="Welch (1947) Biometrika 34:28-35",
)
def _run_ttest_welch(table: DataTable, group_a: Any = None, group_b: Any = None, alpha: float = 0.05,
                     alternative: str = "two-sided", ci_level: float = 0.95, **_: Any) -> StatResult:
    a, b = _pick_two(table, group_a, group_b)
    groups = table.groups()
    return ttest_welch(groups[a], groups[b], names=(a, b), alpha=alpha, alternative=alternative,
                       ci_level=ci_level)


@register(
    key="ttest_paired", label="Paired t test", category="Compare two groups", tables=("column",),
    params=(_GROUP_A, _GROUP_B, _ALPHA, _ALTERNATIVE, _CI_LEVEL),
    description="Compares two measurements made on the same subjects (each row is one subject), such as "
                "before and after a treatment.",
    citation="Student (1908) Biometrika 6:1-25",
)
def _run_ttest_paired(table: DataTable, group_a: Any = None, group_b: Any = None, alpha: float = 0.05,
                      alternative: str = "two-sided", ci_level: float = 0.95, **_: Any) -> StatResult:
    a, b = _pick_two(table, group_a, group_b)
    x, y, notes = _paired_columns(table, a, b)
    result = ttest_paired(x, y, names=(a, b), alpha=alpha, alternative=alternative, ci_level=ci_level)
    result.data_notes = notes + result.data_notes
    return result


@register(
    key="ttest_one_sample", label="One-sample t test", category="Compare two groups", tables=("column",),
    params=(_GROUP, _HYPOTHETICAL, _ALPHA, _ALTERNATIVE, _CI_LEVEL),
    description="Compares the mean of one group with a fixed hypothetical value such as 0 or 100.",
    citation="Student (1908) Biometrika 6:1-25",
)
def _run_ttest_one_sample(table: DataTable, group: Any = None, hypothetical_value: float = 0.0,
                          alpha: float = 0.05, alternative: str = "two-sided", ci_level: float = 0.95,
                          **_: Any) -> StatResult:
    name = _pick_one(table, group)
    return ttest_one_sample(table.groups()[name], hypothetical_value, name=name, alpha=alpha,
                            alternative=alternative, ci_level=ci_level)


@register(
    key="mann_whitney", label="Mann-Whitney test", category="Compare two groups", tables=("column",),
    params=(_GROUP_A, _GROUP_B, _ALPHA, _ALTERNATIVE, _CI_LEVEL),
    description="Compares two independent groups using ranks instead of the raw values, so it does not "
                "need normally distributed data. It asks whether values in one group tend to be larger.",
    citation="Mann & Whitney (1947) Ann Math Stat 18:50-60",
)
def _run_mann_whitney(table: DataTable, group_a: Any = None, group_b: Any = None, alpha: float = 0.05,
                      alternative: str = "two-sided", ci_level: float = 0.95, **_: Any) -> StatResult:
    a, b = _pick_two(table, group_a, group_b)
    groups = table.groups()
    return mann_whitney(groups[a], groups[b], names=(a, b), alpha=alpha, alternative=alternative,
                        ci_level=ci_level)


@register(
    key="wilcoxon_signed_rank", label="Wilcoxon matched-pairs signed-rank test",
    category="Compare two groups", tables=("column",),
    params=(_GROUP_A, _GROUP_B, _ALPHA, _ALTERNATIVE, _CI_LEVEL, _ZERO_METHOD),
    description="Compares two measurements made on the same subjects (each row is one subject) using "
                "ranks, so the differences do not need to be normally distributed.",
    citation="Wilcoxon (1945) Biometrics Bull 1:80-83",
)
def _run_wilcoxon_signed_rank(table: DataTable, group_a: Any = None, group_b: Any = None,
                              alpha: float = 0.05, alternative: str = "two-sided", ci_level: float = 0.95,
                              zero_method: str = "wilcox", **_: Any) -> StatResult:
    a, b = _pick_two(table, group_a, group_b)
    x, y, notes = _paired_columns(table, a, b)
    result = wilcoxon_signed_rank(x, y, names=(a, b), alpha=alpha, alternative=alternative,
                                  ci_level=ci_level, zero_method=zero_method)
    result.data_notes = notes + result.data_notes
    return result


@register(
    key="wilcoxon_one_sample", label="Wilcoxon signed-rank test (one sample)",
    category="Compare two groups", tables=("column",),
    params=(_GROUP, _HYPOTHETICAL, _ALPHA, _ALTERNATIVE, _CI_LEVEL, _ZERO_METHOD),
    description="Compares the median of one group with a fixed hypothetical value such as 0 or 100, "
                "without assuming normally distributed data.",
    citation="Wilcoxon (1945) Biometrics Bull 1:80-83",
)
def _run_wilcoxon_one_sample(table: DataTable, group: Any = None, hypothetical_value: float = 0.0,
                             alpha: float = 0.05, alternative: str = "two-sided", ci_level: float = 0.95,
                             zero_method: str = "wilcox", **_: Any) -> StatResult:
    name = _pick_one(table, group)
    return wilcoxon_one_sample(table.groups()[name], hypothetical_value, name=name, alpha=alpha,
                               alternative=alternative, ci_level=ci_level, zero_method=zero_method)

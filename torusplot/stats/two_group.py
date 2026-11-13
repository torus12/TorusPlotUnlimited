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
  50 values per group (50 non-zero differences for the signed-rank test). With ties the
  exact *conditional* (permutation) distribution of the tied mid-ranks is used for very
  small samples (at most 7 values per group / 7 non-zero differences, ``EXACT_TIES_MAX_N``),
  where the normal approximation is poorest and can even fall below the smallest P value
  that is attainable; for larger tied samples, and above 50, the normal approximation
  with tie correction and continuity correction is used. An approximate P value is never
  reported below the smallest value the exact test could give for these sample sizes.
  The result states which method was used (``extra["p_method"]`` and the interpretation).
* Numerical range: test statistics are computed on internally rescaled data (they do not
  depend on the unit or origin of the measurements), so results are correct from 1e-300
  to 1e300; "no variability" means equal to within floating-point rounding, nothing else.
* Paired differences inherit the rounding error of the two values they are computed
  from. Whether differences are equal to each other, equal to zero, or tied in size is
  therefore judged with a tolerance of a few units in the last place of the *original*
  values (100.1 - 100 and 1.1 - 1 are the same difference).
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
    _TOO_LARGE,
    StatsError,
    alternative_label,
    check_alpha,
    check_alternative,
    ci_phrase,
    clean,
    clean_paired,
    difference_tolerance,
    differences_constant,
    fmt,
    is_constant,
    p_phrase,
    pow2_back,
    pow2_scaled,
    require_n,
    rescale,
    sig_phrase,
    snap_differences,
)
from .assumptions import brown_forsythe_test
from .descriptives import _check_level, _describe_clean
from .registry import ParamSpec, register
from .result import AssumptionCheck, EffectSize, Parameter, StatResult

#: Largest sample size (per group / number of non-zero differences) for exact P values.
EXACT_MAX_N = 50
#: Largest sample size (per group / number of non-zero differences) for which the exact
#: permutation distribution of the *tied* mid-ranks is used automatically. Up to this size
#: there are at most C(14, 7) = 3432 splits (2^7 = 128 sign patterns) and the normal
#: approximation is at its worst; larger tied samples use the normal approximation.
EXACT_TIES_MAX_N = 7
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
_CITE_NEWCOMBE = ("Newcombe RG (2006) Confidence intervals for an effect size measure based on the "
                  "Mann-Whitney statistic. Part 2: asymptotic methods and evaluation. Stat Med 25:559-573 "
                  "(confidence interval of the rank-biserial r).")


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
        res = stats.shapiro(rescale(arr).arrays[0])  # W does not depend on origin or unit
    w, p = float(res.statistic), float(res.pvalue)
    if not (math.isfinite(w) and math.isfinite(p)):
        return AssumptionCheck(name=name, target=target, status="warn",
                               detail="The Shapiro-Wilk test could not be computed for these values.")
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
    row_a, row_b = _row(na, x, level), _row(nb, y, level)
    m1, m2 = row_a["mean"], row_b["mean"]
    # t, df and P do not depend on the origin or unit of the data: compute them on jointly
    # centred values of order 1 (no overflow, no underflow) and scale the estimates back.
    rs = rescale(x, y)
    zx, zy = rs.arrays
    v1 = 0.0 if is_constant(x) else float(zx.var(ddof=1))
    v2 = 0.0 if is_constant(y) else float(zy.var(ddof=1))
    diff_z = float(zx.mean()) - float(zy.mean())
    if welch:
        se_z = math.sqrt(v1 / n1 + v2 / n2)
        df = (
            (v1 / n1 + v2 / n2) ** 2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
            if se_z > 0 else math.nan
        )
    else:
        df = float(n1 + n2 - 2)
        se_z = math.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / df * (1.0 / n1 + 1.0 / n2))
    diff = rs.spread(diff_z)
    se = rs.spread(se_z)
    warnings: list[str] = []
    t: float | None
    p: float | None
    if se_z > 0:
        t = diff_z / se_z
        p = _t_p(t, df, alternative)
        half = float(stats.t.ppf(0.5 + level / 2.0, df)) * se_z
        ci_low, ci_high = rs.spread(diff_z - half), rs.spread(diff_z + half)
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
    elif (v1 == 0.0) != (v2 == 0.0) and se_z > 0:
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
            detail=f"The two groups differ in spread ({p_phrase(bf['p_value'])}; SD {fmt(row_a['sd'])} "
                   f"vs {fmt(row_b['sd'])}). Use Welch's t test instead.",
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
    for name, row in ((na, row_a), (nb, row_b)):
        parameters.append(Parameter(name=f"Mean of {name}", value=row["mean"], se=row["sem"],
                                    ci_low=row["ci_low"], ci_high=row["ci_high"], ci_level=level))

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
        descriptives=[row_a, row_b],
        warnings=warnings,
        data_notes=notes_a + notes_b,
        interpretation=interpretation,
        methods_text=methods,
        citations=[_CITE_WELCH if welch else _CITE_STUDENT, _CITE_D, _CITE_G],
        extra={"mean_difference": diff, "se_difference": se_out,
               "variance_assumption": "unequal (Welch)" if welch else "equal (pooled)"},
    ).finalize()


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
    r = float(np.clip(np.corrcoef(rescale(x).arrays[0], rescale(y).arrays[0])[0, 1], -1.0, 1.0))
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
    """Was pairing effective? One-tailed test of a *positive* correlation between pairs.

    The P value stored with this check is one-tailed (H1: r > 0), because only a positive
    correlation makes pairing worthwhile. The estimates table shows the two-sided P value
    of the same correlation; the text names both so the two numbers are not confused.
    """
    name = "Effective pairing (correlation between pairs)"
    r, p1, p2 = corr["r"], corr["p_positive"], corr["p"]
    if r is None:
        return AssumptionCheck(name=name, status="warn",
                               detail="The correlation between the paired values could not be computed "
                                      "(too few pairs, or one column is constant).")
    tails = f"one-tailed {p_phrase(p1)} for a positive correlation"
    if p2 is not None:
        tails += f"; two-tailed {p_phrase(p2)}, as shown with the estimate"
    if r > 0 and p1 is not None and p1 < 0.05:
        return AssumptionCheck(
            name=name, statistic=r, p_value=p1, status="pass",
            detail=f"The paired values are positively correlated ({label} = {fmt(r, 3)}; {tails}), so "
                   "pairing was effective.",
        )
    if r > 0:
        return AssumptionCheck(
            name=name, statistic=r, p_value=p1, status="info",
            detail=f"The paired values are only weakly correlated ({label} = {fmt(r, 3)}; {tails}); "
                   "pairing gained little.",
        )
    return AssumptionCheck(
        name=name, statistic=r, p_value=p1, status="warn",
        detail=f"The paired values are not positively correlated ({label} = {fmt(r, 3)}; {tails}). "
               "Pairing did not help here; check that the rows really are matched pairs.",
    )


def _paired_differences(x: np.ndarray, y: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
    """(d, tol): the differences x - y and the rounding tolerance of each one.

    ``tol`` is a few units in the last place of the larger of abs(x) and abs(y), i.e. the
    rounding error a difference inherits from the two values it was computed from.
    """
    with np.errstate(over="ignore"):
        d = x - y
    if not np.all(np.isfinite(d)):
        raise StatsError(_TOO_LARGE)
    return d, difference_tolerance(x, y)


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
    d, tol = _paired_differences(x, y)
    diff_name = f"Difference ({na} - {nb})"
    # "Every pair has the same difference" is judged at the rounding error of the ORIGINAL
    # values: 100.1 - 100 and 1.1 - 1 differ in the 14th digit but are the same difference.
    constant = differences_constant(d, tol)
    rs = rescale(d)
    dz = rs.arrays[0]
    if constant:
        md = float(snap_differences(d, tol)[int(np.argmin(tol))])
        md_z = sd_z = 0.0
    else:
        md_z = float(dz.mean())
        md = rs.location(md_z)
        sd_z = float(dz.std(ddof=1))
    se_z = sd_z / math.sqrt(n)
    df = float(n - 1)
    warnings: list[str] = []
    t: float | None
    p: float | None
    if se_z > 0:
        zero_z = rs.to_scaled_location(0.0)
        t = (md_z - zero_z) / se_z
        p = _t_p(t, df, alternative)
        half = float(stats.t.ppf(0.5 + level / 2.0, df)) * se_z
        ci_low, ci_high = rs.location(md_z - half), rs.location(md_z + half)
        se_out: float | None = rs.spread(se_z)
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
    ).finalize()


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
    row = _row(name, arr, level)
    mean = row["mean"]
    rs = rescale(arr)
    z = rs.arrays[0]
    mean_z = float(z.mean())
    mu_z = rs.to_scaled_location(mu)
    if not math.isfinite(mu_z):
        raise StatsError("The hypothetical value is too far from the data (by more than 300 orders of "
                         "magnitude) for the difference to be computed.")
    sd_z = 0.0 if is_constant(arr) else float(z.std(ddof=1))
    se_z = sd_z / math.sqrt(n)
    df = float(n - 1)
    with np.errstate(over="ignore"):
        diff = float(np.float64(mean) - np.float64(mu))
    if not math.isfinite(diff):
        raise StatsError(_TOO_LARGE)
    warnings: list[str] = []
    t: float | None
    p: float | None
    if se_z > 0:
        t = (mean_z - mu_z) / se_z
        p = _t_p(t, df, alternative)
        half = rs.spread(float(stats.t.ppf(0.5 + level / 2.0, df)) * se_z)
        se_out: float | None = rs.spread(se_z)
        mean_ci = (mean - half, mean + half)
        diff_ci = (diff - half, diff + half)
        if not all(math.isfinite(v) for v in (*mean_ci, *diff_ci)):
            raise StatsError(_TOO_LARGE)
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
        descriptives=[row],
        warnings=warnings,
        data_notes=notes,
        interpretation=interpretation,
        methods_text=(f"The mean was compared with the hypothetical value {fmt(mu)} using a one-sample "
                      f"t test ({alternative_label(alternative)}, alpha = {alpha:g})."),
        citations=[_CITE_STUDENT, _CITE_D, _CITE_G],
        extra={"hypothetical_value": mu},
    ).finalize()


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


def rank_sum_null_pmf(weights: Iterable[int], n1: int) -> np.ndarray:
    """Exact permutation distribution of the sum of ``n1`` of the integer ``weights``.

    Every choice of ``n1`` of the N weights is equally likely under the null hypothesis
    (this is the conditional distribution of the rank sum given the observed pattern of
    ties when the weights are the mid-ranks, doubled so that they are integers).
    ``pmf[s] = P(sum = s)`` for s = 0 .. sum of all weights; subsets are counted by
    dynamic programming over (subset size, sum).
    """
    w = [int(v) for v in weights]
    if n1 < 1 or n1 > len(w):
        raise ValueError("n1 must be between 1 and the number of weights.")
    max_sum = int(sum(sorted(w)[-n1:]))
    counts = np.zeros((n1 + 1, max_sum + 1))
    counts[0, 0] = 1.0
    for seen, weight in enumerate(w, start=1):
        for size in range(min(seen, n1), 0, -1):
            if weight <= max_sum:
                counts[size, weight:] += counts[size - 1, : max_sum + 1 - weight]
    pmf = counts[n1]
    return pmf / pmf.sum()


def _attainable_min_p(pmf: np.ndarray, alternative: str) -> float:
    """Smallest P value the exact test with null distribution ``pmf`` can produce."""
    support = np.nonzero(pmf > 0)[0]
    low, high = float(pmf[support[0]]), float(pmf[support[-1]])
    if alternative == "greater":
        return high
    if alternative == "less":
        return low
    return min(1.0, 2.0 * min(low, high))


def _p_kind(use_exact: bool, has_ties: bool) -> str:
    if not use_exact:
        return "approximate"
    return "exact (permutation distribution of the tied ranks)" if has_ties else "exact"


def _p_methods_phrase(use_exact: bool, has_ties: bool) -> str:
    if not use_exact:
        return "normal approximation with tie and continuity correction"
    return "exact permutation P value for the tied ranks" if has_ties else "exact P value"


def _tails_word(alternative: str) -> str:
    return "two-tailed" if alternative == "two-sided" else "one-tailed"


_P_ASYMPTOTIC = "asymptotic (normal approximation with tie correction and continuity correction)"
_P_EXACT_TIES = "exact (permutation distribution of the tied mid-ranks)"
_P_FLOORED = (_P_ASYMPTOTIC[:-1] + "; raised to the smallest P value the exact test can attain)")


def _safe_diff(a: float, b: float) -> float:
    """a - b, or a StatsError when the difference exceeds the largest representable number."""
    with np.errstate(over="ignore"):
        out = float(np.float64(a) - np.float64(b))
    if not math.isfinite(out):
        raise StatsError(_TOO_LARGE)
    return out


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
      no ties and both groups have at most 50 values, or when there are ties and both
      groups have at most 7 values: then the permutation distribution of the tied
      mid-ranks is enumerated) or the normal approximation with tie correction and
      continuity correction (``"asymptotic"``; never reported below the smallest P value
      the exact test can attain). ``extra["p_method"]`` says which was used and
      ``extra["smallest_attainable_p"]`` gives that smallest value.
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
    # Exact P values: without ties up to EXACT_MAX_N per group (tabulated null distribution);
    # with ties, the permutation distribution of the mid-ranks for very small samples.
    ties_exact = has_ties and sd_u > 0 and max(n1, n2) <= EXACT_TIES_MAX_N
    can_exact = ((not has_ties) and max(n1, n2) <= EXACT_MAX_N) or ties_exact
    use_exact = can_exact and method in {"auto", "exact"}
    if method == "exact" and not can_exact and sd_u > 0:
        warnings.append(
            "An exact P value was requested but is not available "
            + (f"because there are tied values in groups larger than {EXACT_TIES_MAX_N}" if has_ties
               else f"for groups larger than {EXACT_MAX_N}")
            + "; the normal approximation was used instead."
        )
    # smallest P value any arrangement of N values into these two groups can give
    tails = 2.0 if alternative == "two-sided" else 1.0
    log_min = math.log(tails) - (math.lgamma(total + 1) - math.lgamma(n1 + 1) - math.lgamma(n2 + 1))
    min_p = min(1.0, tails / math.comb(total, n1)) if total <= 60 else min(1.0, math.exp(log_min))
    pmf: np.ndarray | None = None
    z: float | None = None
    p: float | None
    if sd_u <= 0:
        p = 1.0
        p_method = "all values identical"
        use_exact = False
        min_p = 1.0
        warnings.append(
            "Every value in both groups is identical, so the groups cannot be told apart (P = 1)."
        )
    elif use_exact and not has_ties:
        pmf = mann_whitney_null_pmf(n1, n2)
        p = _exact_tail_p(pmf, u_a, alternative)
        p_method = "exact"
    elif use_exact:
        doubled = np.rint(2.0 * ranks).astype(int)
        tied_pmf = rank_sum_null_pmf(doubled, n1)
        p = _exact_tail_p(tied_pmf, float(doubled[:n1].sum()), alternative)
        min_p = _attainable_min_p(tied_pmf, alternative)
        p_method = _P_EXACT_TIES
    else:
        z, p = _normal_tail_p(u_a, mean_u, sd_u, alternative)
        p_method = _P_ASYMPTOTIC
        if p < min_p:  # the approximation cannot be more extreme than the exact test allows
            p, p_method = min_p, _P_FLOORED
    if has_ties and sd_u > 0:
        n_tied = int(tie_counts[tie_counts > 1].sum())
        if use_exact:
            warnings.append(
                f"There are tied values ({n_tied} of {total} values share a value with another). The P "
                "value is exact for these tied ranks: it comes from the permutation distribution of "
                "the mid-ranks (every way of dividing the pooled values into two groups of these "
                "sizes), not from the normal approximation."
            )
        else:
            warnings.append(
                f"There are tied values ({n_tied} of {total} values share a value with another), so "
                "the tabulated exact distribution does not apply; the normal approximation with tie "
                "correction was used."
                + (" With samples this small the approximation can be rough." if min(n1, n2) < 8 else "")
            )
    if min_p > alpha and sd_u > 0:
        warnings.append(
            f"With {n1} and {n2} values the smallest possible {_tails_word(alternative)} P value is "
            f"{fmt(min_p, 3)}, so this test can never reach significance at alpha = {alpha:g}."
        )
    warnings.extend(_one_sided_note(alternative, level))

    # ---- Hodges-Lehmann shift estimate and CI
    hl: float | None = None
    hl_low = hl_high = achieved = None
    ci_method = None
    if n_pairs <= _MAX_PAIRS:
        # scaled by a common power of two (exact), so a_i - b_j cannot overflow
        scaled, exponent = pow2_scaled(pooled)
        diffs = np.sort(np.subtract.outer(scaled[:n1], scaled[n1:]).ravel())
        hl = pow2_back(float(np.median(diffs)), exponent)
        hl_low, hl_high, achieved, ci_method = _order_stat_ci(
            diffs, pmf, mean_u, math.sqrt(n_pairs * (total + 1.0) / 12.0), level
        )
        if hl_low is None:
            warnings.append(
                f"The samples are too small for a {level * 100:g}% confidence interval of the "
                "Hodges-Lehmann difference."
            )
        else:
            hl_low, hl_high = pow2_back(hl_low, exponent), pow2_back(hl_high, exponent)
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
        Parameter(name=f"Difference between medians ({na} - {nb})", value=_safe_diff(med_a, med_b),
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
        f"{_p_kind(use_exact, has_ties)} P value. "
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
            f"alpha = {alpha:g}; {_p_methods_phrase(use_exact, has_ties)}); "
            "the shift was estimated with the Hodges-Lehmann estimator."
        ),
        citations=[_CITE_MWU, _CITE_HL, _CITE_BAUER, _CITE_RB, _CITE_NEWCOMBE],
        extra={
            "u_a": u_a, "u_b": u_b, "u_min": u, "statistic_reported": "U = smaller of U_A and U_B",
            "rank_sum_a": r1, "rank_sum_b": r2, "mean_rank_a": mean_rank_a, "mean_rank_b": mean_rank_b,
            "z": z, "p_method": p_method, "exact": bool(use_exact), "ties": has_ties,
            "probability_of_superiority": u_a / n_pairs,
            "smallest_attainable_p": min_p,
            "hodges_lehmann_ci_method": ci_method, "hodges_lehmann_ci_coverage": achieved,
        },
    ).finalize()


# ============================================================ Wilcoxon signed-rank


def _signed_rank_core(d: np.ndarray, *, zero_method: str = "wilcox", method: str = "auto",
                      alternative: str = "two-sided") -> dict[str, Any]:
    """Statistic and P value of the Wilcoxon signed-rank test for differences ``d``.

    This is the single engine behind :func:`wilcoxon_signed_rank`,
    :func:`wilcoxon_one_sample` and the Wilcoxon comparisons after the Friedman test
    (``posthoc.friedman_posthoc``), so all of them give identical P values.

    ``d`` must already be free of rounding noise (see ``snap_differences``): exact zeros
    are zero differences and exactly equal absolute values are ties.

    P value: exact when no two non-zero differences are tied and there are at most
    ``EXACT_MAX_N`` of them (null distribution of W+ over all sign patterns); exact also
    with ties when there are at most ``EXACT_TIES_MAX_N`` non-zero differences (the same
    enumeration over the tied mid-ranks); otherwise the normal approximation with tie
    correction and continuity correction, never reported below the smallest attainable
    P value 2 / 2^n (1 / 2^n one-tailed).
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

    ties_exact = has_ties and 1 <= n_r <= EXACT_TIES_MAX_N
    can_exact = n_r >= 1 and ((not has_ties and n_r <= EXACT_MAX_N) or ties_exact)
    use_exact = can_exact and method in {"auto", "exact"}
    tails = 2.0 if alternative == "two-sided" else 1.0
    min_p = min(1.0, tails * 0.5 ** n_r) if n_r else None
    pmf: np.ndarray | None = None
    z: float | None = None
    p: float | None
    if n_r == 0:
        p = None
        p_method = "not computed"
    elif use_exact:
        if has_ties:  # mid-ranks are multiples of 1/2: enumerate on doubled (integer) ranks
            pmf = signed_rank_null_pmf(np.rint(2.0 * ranks).astype(int))
            p = _exact_tail_p(pmf, 2.0 * w_plus, alternative)
            p_method = _P_EXACT_TIES
        else:
            pmf = signed_rank_null_pmf(np.rint(ranks).astype(int))
            p = _exact_tail_p(pmf, w_plus, alternative)
            p_method = "exact"
        if n_zero and zero_method == "wilcox":
            p_method += " (zero differences discarded)"
        elif n_zero:
            p_method += " (conditional on the ranks of the non-zero differences, Pratt's method)"
    else:
        z, p = _normal_tail_p(w_plus, mean_w, sd_w, alternative)
        p_method = _P_ASYMPTOTIC
        if p < min_p:  # the approximation cannot be more extreme than the exact test allows
            p, p_method = min_p, _P_FLOORED
    return {
        "n_all": n_all, "n_zero": n_zero, "n_nonzero": n_r, "w_plus": w_plus, "w_minus": w_minus,
        "w": w_plus - w_minus, "has_ties": has_ties, "can_exact": can_exact, "use_exact": bool(use_exact),
        "p": p, "z": z, "p_method": p_method, "min_p": min_p,
        "pmf": pmf if (use_exact and not has_ties) else None,
    }


def _signed_rank_result(
    d: np.ndarray, *, tol: np.ndarray, analysis: str, test_name: str, why: str, label: str, estimate_name: str,
    estimate_offset: float, alpha: float, alternative: str, level: float, zero_method: str, method: str,
    tail_names: tuple[str, str], n_label: dict[str, int], rows: list[dict[str, Any]], notes: list[str],
    assumptions: list[AssumptionCheck], parameters_extra: list[Parameter], methods_prefix: str,
    extra_more: dict[str, Any],
) -> StatResult:
    """Shared engine of the paired and one-sample Wilcoxon signed-rank tests.

    ``d`` are the differences (a - b, or x - hypothetical median) and ``tol`` their
    rounding tolerances (a few units in the last place of the values they were computed
    from). Differences within tolerance of zero count as zero and absolute differences
    within tolerance of each other count as ties, so that floating-point noise such as
    5.4 - 5.1 != 5.2 - 4.9 is not ranked as if it were data.
    """
    raw = d
    d = snap_differences(raw, tol)
    core = _signed_rank_core(d, zero_method=zero_method, method=method, alternative=alternative)
    n_all, n_zero, n_r = core["n_all"], core["n_zero"], core["n_nonzero"]
    w_plus, w_minus, w = core["w_plus"], core["w_minus"], core["w"]
    has_ties, use_exact = core["has_ties"], core["use_exact"]
    p, z, p_method, pmf, min_p = core["p"], core["z"], core["p_method"], core["pmf"], core["min_p"]

    warnings: list[str] = []
    if method == "exact" and not core["can_exact"] and n_r >= 1:
        warnings.append(
            "An exact P value was requested but is not available "
            + (f"because some differences are tied in size and there are more than {EXACT_TIES_MAX_N} "
               "of them" if has_ties else f"for more than {EXACT_MAX_N} non-zero differences")
            + "; the normal approximation was used instead."
        )
    statistic: float | None = w
    if n_r == 0:
        statistic = None
        warnings.append(
            "Every difference is exactly zero, so there is nothing to rank and the test cannot be computed."
        )
    if has_ties and n_r:
        if use_exact:
            warnings.append(
                "Some differences are tied in size. The P value is exact for these tied ranks: it comes "
                f"from the distribution of the mid-ranks over all {2 ** n_r} possible patterns of signs, "
                "not from the normal approximation."
            )
        else:
            warnings.append(
                "Some differences are tied in size, so the tabulated exact distribution does not apply; "
                "the normal approximation with tie correction was used."
                + (" With so few pairs the approximation can be rough." if n_r < 10 else "")
            )
    if n_zero and n_r:
        warnings.append(
            f"{n_zero} difference(s) are exactly zero. "
            + ("They were discarded before ranking (Wilcoxon's method)." if zero_method == "wilcox"
               else "They were ranked with the others and their ranks then dropped (Pratt's method).")
        )
    if min_p is not None and min_p > alpha:
        warnings.append(
            f"With only {n_r} non-zero difference(s) the smallest possible {_tails_word(alternative)} P "
            f"value is {fmt(min_p, 3)}; the test cannot reach significance at alpha = {alpha:g}."
        )
    warnings.extend(_one_sided_note(alternative, level))

    # ---- Hodges-Lehmann pseudo-median of the differences (median of Walsh averages)
    hl = hl_low = hl_high = achieved = None
    ci_method = None
    n_walsh = n_all * (n_all + 1) // 2
    if n_walsh <= _MAX_PAIRS:
        i, j = np.triu_indices(n_all)
        walsh = np.sort(d[i] / 2.0 + d[j] / 2.0)  # halves first: the sum itself could overflow
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
    skew = _describe_clean(raw, level)["skewness"] if n_all else None
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
            f"{_p_kind(use_exact, has_ties)} P value. "
        )
        if shift is not None:
            interpretation += f"{estimate_name}: {fmt(shift)}"
            interpretation += (
                f" ({ci_phrase(hl_low + estimate_offset, hl_high + estimate_offset, level)}). "
                if hl_low is not None else ". "
            )
        interpretation += _effect_sentence(rb)
    citations = [_CITE_WILCOXON, _CITE_HL, _CITE_BAUER, _CITE_RB, _CITE_NEWCOMBE]
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
            + "; " + _p_methods_phrase(use_exact, has_ties)
            + ("; zero differences discarded)." if zero_method == "wilcox"
               else "; zero differences handled by Pratt's method).")
        ),
        citations=citations,
        extra={
            "w_plus": w_plus, "w_minus": w_minus, "w": w, "t_min": min(w_plus, w_minus),
            "statistic_reported": "W = sum of signed ranks (W+ minus W-)",
            "n_nonzero": n_r, "n_zero": n_zero, "z": z, "p_method": p_method, "exact": bool(use_exact),
            "ties": has_ties, "zero_method": zero_method, "smallest_attainable_p": min_p,
            "hodges_lehmann_ci_method": ci_method, "hodges_lehmann_ci_coverage": achieved,
            **extra_more,
        },
    ).finalize()


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
      most 50 of them; exact also with tied differences when there are at most 7 non-zero
      differences (enumeration of the tied mid-ranks over all sign patterns); otherwise
      the normal approximation with tie correction and continuity correction (never
      reported below the smallest attainable value 2 / 2^n). ``extra["p_method"]`` says
      which. Differences are compared with a tolerance of a few units in the last place
      of the original values, so 5.4 - 5.1 and 5.2 - 4.9 count as tied.
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
    d, tol = _paired_differences(x, y)
    n = int(x.size)
    diff_name = f"Difference ({na} - {nb})"
    corr = _spearman(x, y)
    return _signed_rank_result(
        d, tol=tol,
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
            Parameter(name="Median of differences", value=_describe_clean(d, level)["median"],
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
    d, tol = _paired_differences(arr, m0)
    median = _describe_clean(arr, level)["median"]
    return _signed_rank_result(
        d, tol=tol,
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
            Parameter(name="Median minus hypothetical value", value=_safe_diff(median, m0)),
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

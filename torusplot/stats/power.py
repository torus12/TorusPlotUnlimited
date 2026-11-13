"""Power and sample-size calculations for t tests and one-way ANOVA.

Power is the probability that a study finds a statistically significant result when the
true effect has the stated size. All calculations use the exact noncentral t / noncentral
F distributions through :mod:`statsmodels.stats.power` (``TTestIndPower``, ``TTestPower``,
``FTestAnovaPower``); the unknown quantity is found by root-finding on that power function.

Effect sizes
------------
* t tests: Cohen's d = (difference between means) / SD. For a paired design d is the
  mean of the within-pair differences divided by the SD *of the differences* (d_z).
  Conventional benchmarks: 0.2 small, 0.5 medium, 0.8 large.
* ANOVA: Cohen's f = (SD of the group means) / (common within-group SD);
  f^2 = eta^2 / (1 - eta^2). Benchmarks: 0.10 small, 0.25 medium, 0.40 large.

Sample sizes are *per group* (pairs for a paired design, values for a one-sample design;
for an unpaired design with unequal groups, ``n`` is the size of group 1 and group 2 has
``ratio`` x n). Solved sample sizes are reported both exactly (fractional) and rounded up
to the next whole subject; sample sizes entered by the user are reported as entered.

References: Cohen J (1988) *Statistical Power Analysis for the Behavioral Sciences*,
2nd ed.; Cohen J (1992) A power primer. *Psychol Bull* 112:155-159.

These functions are not registered analyses (they need no data table).
"""

from __future__ import annotations

import math
import warnings as _warnings
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from scipy import optimize, stats
from statsmodels.stats.power import FTestAnovaPower, TTestIndPower, TTestPower

from ._utils import StatsError, alternative_label, check_alpha, check_alternative, fmt
from .effect_sizes import magnitude_d
from .result import AssumptionCheck, Parameter, StatResult

_CITATIONS = [
    "Cohen J (1988) Statistical Power Analysis for the Behavioral Sciences, 2nd ed. Lawrence Erlbaum.",
    "Cohen J (1992) A power primer. Psychol Bull 112:155-159.",
]

_DESIGNS = {
    "unpaired": "unpaired t test",
    "paired": "paired t test",
    "one-sample": "one-sample t test",
}
_SM_ALTERNATIVE = {"two-sided": "two-sided", "greater": "larger", "less": "smaller"}
_N_LIMIT = 1e9


# ======================================================================= helpers


def effect_size_from_means(mean_diff: float, sd: float) -> float:
    """Cohen's d from a raw difference and a standard deviation: d = mean_diff / sd.

    For an unpaired design ``sd`` is the (common) SD within each group; for a paired
    design it is the SD of the within-pair differences.
    """
    try:
        diff, spread = float(mean_diff), float(sd)
    except (TypeError, ValueError):
        raise StatsError("The difference and the SD must be numbers.") from None
    if not (math.isfinite(diff) and math.isfinite(spread)) or spread <= 0:
        raise StatsError("The SD must be a positive number (and the difference a finite number).")
    return diff / spread


def _number(value: Any, what: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        raise StatsError(f"{what} must be a number.") from None
    if not math.isfinite(out):
        raise StatsError(f"{what} must be a finite number.")
    return out


def _which_unknown(effect_size: Any, n: Any, power: Any) -> str:
    missing = [name for name, value in (("effect_size", effect_size), ("n", n), ("power", power)) if value is None]
    if len(missing) != 1:
        raise StatsError(
            "Give exactly two of: effect size, sample size (n) and power. The one you leave empty is "
            "calculated."
        )
    return missing[0]


def _check_power(power: Any, alpha: float) -> float:
    value = _number(power, "Power")
    if not (0.0 < value < 1.0):
        raise StatsError("Power must be between 0 and 1 (for example 0.8 for 80%).")
    if value <= alpha:
        raise StatsError(
            f"Power ({value:g}) must be larger than alpha ({alpha:g}); a test always has at least "
            "alpha chance of a significant result."
        )
    return value


def _call(func: Callable[..., float], *args: Any, **kwargs: Any) -> float:
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        with np.errstate(all="ignore"):
            value = float(np.squeeze(func(*args, **kwargs)))
    return value


def _t_power(design: str, d: float, n: float, alpha: float, alternative: str, ratio: float) -> float:
    """Power of a t test from the noncentral t distribution (statsmodels)."""
    alt = _SM_ALTERNATIVE[alternative]
    if design == "unpaired":
        value = _call(TTestIndPower().power, d, n, alpha, ratio=ratio, alternative=alt)
    else:
        value = _call(TTestPower().power, d, n, alpha, alternative=alt)
    if not math.isfinite(value):
        value = _t_power_guarded(design, d, n, alpha, alternative, ratio)
    return min(max(value, 0.0), 1.0) if math.isfinite(value) else math.nan


def _t_power_guarded(design: str, d: float, n: float, alpha: float, alternative: str, ratio: float) -> float:
    """The same power computed tail by tail, for the cases where scipy's noncentral t
    returns NaN: that happens in the far tail *opposite* to the noncentrality parameter,
    where the true probability is negligibly small (it underflows) and is taken as 0."""
    if design == "unpaired":
        nobs, df = n * ratio / (1.0 + ratio), n * (1.0 + ratio) - 2.0
    else:
        nobs, df = n, n - 1.0
    if df <= 0:
        return math.nan
    nc = d * math.sqrt(nobs)
    tail_alpha = alpha / 2.0 if alternative == "two-sided" else alpha
    crit = float(stats.t.isf(tail_alpha, df))
    total = 0.0
    if alternative in {"two-sided", "greater"}:
        upper = _call(stats.nct.sf, crit, df, nc)
        if not math.isfinite(upper):
            if nc >= 0:
                return math.nan
            upper = 0.0
        total += upper
    if alternative in {"two-sided", "less"}:
        lower = _call(stats.nct.cdf, -crit, df, nc)
        if not math.isfinite(lower):
            if nc <= 0:
                return math.nan
            lower = 0.0
        total += lower
    return total


def _anova_power(f: float, n: float, alpha: float, k: int) -> float:
    """Power of the one-way ANOVA F test; ``n`` is per group (statsmodels wants the total)."""
    value = _call(FTestAnovaPower().power, f, n * k, alpha, k_groups=k)
    if not math.isfinite(value):
        # statsmodels evaluates 1 - CDF, and scipy's noncentral F CDF returns NaN where it
        # underflows (large noncentrality); the survival function itself is well behaved.
        df1, df2, nc = k - 1.0, n * k - k, f * f * n * k
        if df2 > 0:
            value = _call(stats.ncf.sf, float(stats.f.isf(alpha, df1, df2)), df1, df2, nc)
        if not math.isfinite(value) and nc > 1e4:
            value = 1.0  # enormous noncentrality: the power is 1 to machine precision
    return min(max(value, 0.0), 1.0) if math.isfinite(value) else math.nan


def _solve_increasing(func: Callable[[float], float], target: float, lo: float, what: str,
                      impossible: str) -> float:
    """Smallest value >= ``lo`` at which the increasing function ``func`` reaches ``target``."""
    start = func(lo)
    if not math.isfinite(start):
        raise StatsError(f"The {what} could not be calculated for these settings.")
    if start >= target:
        return lo
    hi = max(2.0 * lo, 4.0)
    while True:
        current = func(hi)
        if not math.isfinite(current):
            raise StatsError(f"The {what} could not be calculated for these settings.")
        if current >= target:
            break
        if hi > _N_LIMIT:
            raise StatsError(impossible)
        hi *= 2.0
    try:
        return float(optimize.brentq(lambda v: func(v) - target, lo, hi, xtol=1e-10, rtol=1e-12, maxiter=500))
    except (ValueError, RuntimeError):
        raise StatsError(f"The {what} could not be calculated for these settings.") from None


def _ceil(value: float) -> int:
    return int(math.ceil(value - 1e-9))


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


_PLANNING_ASSUMPTIONS = [
    AssumptionCheck(
        name="The effect size is a planning value", status="info",
        detail="The calculation is only as good as the effect size (and SD) you enter. Base it on pilot "
               "data, published work, or the smallest effect that would matter scientifically.",
    ),
    AssumptionCheck(
        name="Normal data, equal SDs, independent observations", status="info",
        detail="The calculation assumes the data will meet the assumptions of the planned test.",
    ),
]


# ======================================================================== t tests


def power_ttest(
    *,
    design: str = "unpaired",
    effect_size: float | None = None,
    n: float | None = None,
    power: float | None = None,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    ratio: float = 1.0,
) -> StatResult:
    """Power, sample size or detectable effect for a t test.

    Leave exactly one of ``effect_size`` (Cohen's d), ``n`` (per group; number of pairs
    for ``design="paired"``; number of values for ``"one-sample"``) and ``power`` as
    ``None``: it is solved for. ``ratio`` = n2 / n1 for unequal group sizes (unpaired
    design only; ``n`` is then the size of group 1).

    For one-sided alternatives the sign of the effect matters: ``"greater"`` needs a
    positive d, ``"less"`` a negative d. Every solved value is in ``parameters`` and in
    ``extra`` (``n_exact``, ``n`` rounded up, ``n2``, ``n_total``, ``power``,
    ``achieved_power`` at the rounded n, ``effect_size``).
    """
    if design not in _DESIGNS:
        raise StatsError(f"design must be one of {list(_DESIGNS)}.")
    alpha = check_alpha(alpha)
    alternative = check_alternative(alternative)
    unknown = _which_unknown(effect_size, n, power)
    ratio = _number(ratio, "The group size ratio")
    if design != "unpaired":
        ratio = 1.0
    if ratio <= 0:
        raise StatsError("The ratio of group sizes (n2 / n1) must be positive.")
    n_min = max(2.0, 2.0 / ratio) if design == "unpaired" else 2.0
    unit = {"unpaired": "per group" if ratio == 1.0 else "group 1", "paired": "pairs",
            "one-sample": "values"}[design]
    warnings: list[str] = []

    def pw(d: float, size: float) -> float:
        return _t_power(design, d, size, alpha, alternative, ratio)

    if unknown != "effect_size":
        d = _number(effect_size, "The effect size")
        if d == 0 and unknown == "n":
            raise StatsError("An effect size of zero can never be detected; enter the smallest effect "
                             "you care about.")
    if unknown != "n":
        n_given = _number(n, "The sample size")
        if n_given < n_min:
            raise StatsError(f"The sample size must be at least {fmt(n_min)} ({unit}) for this design.")
    if unknown != "power":
        target = _check_power(power, alpha)

    if unknown == "power":
        n_exact = n_given
        solved_power = pw(d, n_exact)
    elif unknown == "n":
        if (alternative == "greater" and d < 0) or (alternative == "less" and d > 0):
            raise StatsError(
                "The sign of the effect size contradicts the one-sided alternative, so the desired "
                "power can never be reached. Use a positive effect with 'greater' and a negative "
                "effect with 'less'."
            )
        n_exact = _solve_increasing(
            lambda size: pw(d, size), target, n_min, "sample size",
            "The required sample size is larger than a billion; the effect is too small to detect.",
        )
        solved_power = target
    else:
        sign = -1.0 if alternative == "less" else 1.0
        n_exact = n_given
        magnitude = _solve_increasing(
            lambda size_d: pw(sign * size_d, n_exact), target, 1e-9, "effect size",
            "No effect size reaches the desired power with this sample size.",
        )
        d = sign * magnitude
        solved_power = target

    integer_n = unknown != "n" and abs(n_exact - round(n_exact)) < 1e-9
    n1 = _ceil(n_exact)
    if design == "unpaired":
        n2 = _ceil(ratio * n1) if (unknown == "n" or integer_n) else None
        n2_exact = ratio * n_exact
    else:
        n2, n2_exact = None, None
    if unknown == "n":
        achieved = (_t_power(design, d, n1, alpha, alternative, (n2 / n1) if n2 else 1.0))
    else:
        achieved = solved_power
    if design == "unpaired":
        n_total: float = (n1 + n2) if n2 is not None else n_exact * (1.0 + ratio)
    else:
        n_total = n1 if (unknown == "n" or integer_n) else n_exact

    if unknown != "n" and not integer_n:
        warnings.append("The sample size entered is not a whole number; results are for that exact value.")
    if unknown == "n" and n_exact <= n_min + 1e-9:
        warnings.append(
            f"The smallest allowed sample size ({fmt(n_min)} {unit}) already gives the desired power; "
            "an effect this large is rarely realistic."
        )
    if solved_power < 0.5 and unknown == "power":
        warnings.append("Power is below 50%: the study is more likely to miss this effect than to find it.")
    if n1 < 5 and unknown == "n":
        warnings.append("With so few subjects the normality assumption cannot be checked and matters most.")

    parameters = [
        Parameter(name="Effect size (Cohen's d)", value=d,
                  unit_note=("Solved." if unknown == "effect_size" else "As entered.")
                            + (" For paired designs d = mean difference / SD of the differences." if design == "paired" else "")),
        Parameter(name=f"Sample size, exact ({unit})" if unknown == "n" else f"Sample size ({unit})",
                  value=n_exact,
                  unit_note="Solved (fractional)." if unknown == "n" else "As entered."),
    ]
    if unknown == "n":
        parameters.append(Parameter(name=f"Sample size, rounded up ({unit})", value=n1,
                                    unit_note="Use this number when planning the study."))
    if design == "unpaired":
        parameters.append(Parameter(name="Sample size of group 2", value=n2 if n2 is not None else n2_exact,
                                    unit_note=f"n2 = {fmt(ratio)} x n1."))
        parameters.append(Parameter(name="Total sample size", value=n_total))
    parameters.append(Parameter(name="Power", value=achieved if unknown == "n" else solved_power,
                                unit_note="Power at the rounded-up sample size." if unknown == "n"
                                else ("Solved." if unknown == "power" else "As entered.")))
    parameters.append(Parameter(name="Alpha", value=alpha, unit_note=alternative_label(alternative)))

    test = _DESIGNS[design]
    tails = alternative_label(alternative)
    size_label = magnitude_d(d)
    d_text = f"d = {fmt(d, 3)}" + (f" (conventionally a {size_label} effect)" if size_label else "")
    if design == "unpaired":
        n_text = (f"{n1} subjects per group ({n_total} in total)" if n2 == n1
                  else f"{n1} subjects in group 1 and {n2} in group 2 ({n_total} in total)") \
            if n2 is not None else f"{fmt(n_exact)} subjects in group 1 and {fmt(n2_exact)} in group 2"
    elif design == "paired":
        n_text = f"{n1} pairs" if (unknown == "n" or integer_n) else f"{fmt(n_exact)} pairs"
    else:
        n_text = f"{n1} values" if (unknown == "n" or integer_n) else f"{fmt(n_exact)} values"
    if unknown == "n":
        interpretation = (
            f"To have {_pct(target)} power to detect an effect of {d_text} with a {tails} {test} at "
            f"alpha = {alpha:g}, you need {n_text}. The exact solution is {fmt(n_exact, 5)} ({unit}); "
            f"rounding up gives an actual power of {_pct(achieved)}. If the true effect is smaller than "
            "assumed, the power will be lower."
        )
    elif unknown == "power":
        interpretation = (
            f"With {n_text}, a {tails} {test} at alpha = {alpha:g} has {_pct(solved_power)} power to detect "
            f"an effect of {d_text}. That is the chance of obtaining a statistically significant result "
            f"if the true effect is exactly that large; the chance of missing it is {_pct(1 - solved_power)}."
        )
    else:
        if alternative == "less":
            reach = (f"effects of size {d_text} or more negative (a decrease of at least "
                     f"{fmt(abs(d), 3)} SD). True effects closer to zero will usually be missed")
        elif alternative == "greater":
            reach = (f"effects of size {d_text} or larger. Smaller true effects will usually be "
                     "missed")
        else:
            reach = (f"effects of size {d_text} or larger in magnitude, in either direction. True "
                     "effects closer to zero will usually be missed")
        interpretation = (
            f"With {n_text}, a {tails} {test} at alpha = {alpha:g} has {_pct(target)} power to detect "
            f"{reach} with this sample size."
        )
    n_dict: dict[str, int] = {}
    if unknown == "n" or integer_n:
        if design == "unpaired" and n2 is not None:
            n_dict = {"Group 1": n1, "Group 2": n2}
        elif design == "paired":
            n_dict = {"Pairs": n1}
        elif design == "one-sample":
            n_dict = {"Sample": n1}
    return StatResult(
        analysis="power_ttest",
        test_name=f"Power analysis: {test}",
        why_chosen=(
            "A power analysis links four quantities: the effect size, the sample size, alpha and "
            "power. Fixing three of them determines the fourth. Use it before an experiment to "
            "choose a sample size that gives a good chance of detecting the effect you care about."
        ),
        assumptions=list(_PLANNING_ASSUMPTIONS),
        alpha=alpha,
        alternative=alternative,
        parameters=parameters,
        n=n_dict,
        warnings=warnings,
        interpretation=interpretation,
        methods_text=(
            f"Sample size and power were calculated for a {tails} {test} (alpha = {alpha:g}) from the "
            "noncentral t distribution."
        ),
        citations=list(_CITATIONS),
        extra={
            "solved_for": unknown, "design": design, "effect_size": d, "n_exact": n_exact,
            "n": n1 if (unknown == "n" or integer_n) else n_exact,
            "n2": n2 if n2 is not None else n2_exact, "n_total": n_total, "ratio": ratio,
            "power": solved_power, "achieved_power": achieved, "alpha": alpha,
        },
    )


# =========================================================================== ANOVA


def power_anova(
    *,
    k_groups: int,
    effect_size: float | None = None,
    n: float | None = None,
    power: float | None = None,
    alpha: float = 0.05,
) -> StatResult:
    """Power, sample size or detectable effect for a one-way ANOVA with ``k_groups`` groups.

    ``effect_size`` is Cohen's f; ``n`` is the sample size **per group** (equal group
    sizes are assumed; the total is n x k_groups). Leave exactly one of ``effect_size``,
    ``n`` and ``power`` as ``None``. Uses the noncentral F distribution with
    df = (k - 1, N - k) and noncentrality f^2 N.
    """
    try:
        k = int(k_groups)
    except (TypeError, ValueError):
        raise StatsError("The number of groups must be a whole number.") from None
    if k < 2 or k != float(k_groups):
        raise StatsError("The number of groups must be a whole number of at least 2.")
    alpha = check_alpha(alpha)
    unknown = _which_unknown(effect_size, n, power)
    n_min = 2.0
    warnings: list[str] = []
    if unknown != "effect_size":
        f = _number(effect_size, "The effect size")
        if f < 0:
            raise StatsError("Cohen's f cannot be negative.")
        if f == 0 and unknown == "n":
            raise StatsError("An effect size of zero can never be detected; enter the smallest effect "
                             "you care about.")
    if unknown != "n":
        n_given = _number(n, "The sample size per group")
        if n_given < n_min:
            raise StatsError("The sample size must be at least 2 per group.")
    if unknown != "power":
        target = _check_power(power, alpha)

    if unknown == "power":
        n_exact = n_given
        solved_power = _anova_power(f, n_exact, alpha, k)
    elif unknown == "n":
        n_exact = _solve_increasing(
            lambda size: _anova_power(f, size, alpha, k), target, n_min, "sample size",
            "The required sample size is larger than a billion; the effect is too small to detect.",
        )
        solved_power = target
    else:
        n_exact = n_given
        f = _solve_increasing(
            lambda size_f: _anova_power(size_f, n_exact, alpha, k), target, 1e-9, "effect size",
            "No effect size reaches the desired power with this sample size.",
        )
        solved_power = target
    integer_n = unknown != "n" and abs(n_exact - round(n_exact)) < 1e-9
    n_round = _ceil(n_exact)
    achieved = _anova_power(f, n_round, alpha, k) if unknown == "n" else solved_power
    whole = unknown == "n" or integer_n
    n_total: float = n_round * k if whole else n_exact * k
    if unknown != "n" and not integer_n:
        warnings.append("The sample size entered is not a whole number; results are for that exact value.")
    if unknown == "n" and n_exact <= n_min + 1e-9:
        warnings.append("The smallest allowed sample size (2 per group) already gives the desired power; "
                        "an effect this large is rarely realistic.")
    if solved_power < 0.5 and unknown == "power":
        warnings.append("Power is below 50%: the study is more likely to miss this effect than to find it.")
    eta2 = f * f / (1.0 + f * f)
    size_label = "negligible" if f < 0.1 else "small" if f < 0.25 else "medium" if f < 0.4 else "large"
    parameters = [
        Parameter(name="Effect size (Cohen's f)", value=f,
                  unit_note=("Solved." if unknown == "effect_size" else "As entered.")
                            + f" Equivalent to eta squared = {fmt(eta2, 3)}."),
        Parameter(name="Sample size per group, exact" if unknown == "n" else "Sample size per group",
                  value=n_exact,
                  unit_note="Solved (fractional)." if unknown == "n" else "As entered."),
    ]
    if unknown == "n":
        parameters.append(Parameter(name="Sample size per group, rounded up", value=n_round,
                                    unit_note="Use this number when planning the study."))
    parameters += [
        Parameter(name="Total sample size", value=n_total, unit_note=f"{k} groups of equal size."),
        Parameter(name="Power", value=achieved if unknown == "n" else solved_power,
                  unit_note="Power at the rounded-up sample size." if unknown == "n"
                  else ("Solved." if unknown == "power" else "As entered.")),
        Parameter(name="Alpha", value=alpha),
    ]
    f_text = f"f = {fmt(f, 3)} (conventionally a {size_label} effect)"
    n_text = (f"{n_round} subjects per group ({int(n_total)} in total)" if whole
              else f"{fmt(n_exact)} subjects per group")
    if unknown == "n":
        interpretation = (
            f"To have {_pct(target)} power to detect an effect of {f_text} with a one-way ANOVA of {k} "
            f"groups at alpha = {alpha:g}, you need {n_text}. The exact solution is {fmt(n_exact, 5)} per "
            f"group; rounding up gives an actual power of {_pct(achieved)}."
        )
    elif unknown == "power":
        interpretation = (
            f"With {n_text}, a one-way ANOVA of {k} groups at alpha = {alpha:g} has {_pct(solved_power)} "
            f"power to detect an effect of {f_text}: that is the chance of a statistically significant "
            "overall F test if the true differences between group means are that large."
        )
    else:
        interpretation = (
            f"With {n_text}, a one-way ANOVA of {k} groups at alpha = {alpha:g} has {_pct(target)} power "
            f"to detect effects of {f_text} or larger."
        )
    interpretation += (
        " This is the power of the overall F test only; comparisons between specific pairs of "
        "groups need their own calculation."
    )
    return StatResult(
        analysis="power_anova",
        test_name="Power analysis: one-way ANOVA",
        why_chosen=(
            "A power analysis for one-way ANOVA links the effect size (Cohen's f), the sample size "
            "per group, alpha and power; fixing three determines the fourth."
        ),
        assumptions=list(_PLANNING_ASSUMPTIONS),
        alpha=alpha,
        parameters=parameters,
        n={f"Group {i + 1}": n_round for i in range(k)} if whole else {},
        warnings=warnings,
        interpretation=interpretation,
        methods_text=(
            f"Sample size and power were calculated for a one-way ANOVA with {k} groups (alpha = {alpha:g}) "
            "from the noncentral F distribution."
        ),
        citations=list(_CITATIONS),
        extra={
            "solved_for": unknown, "design": "anova", "k_groups": k, "effect_size": f, "eta_squared": eta2,
            "n_exact": n_exact, "n": n_round if whole else n_exact, "n_total": n_total,
            "power": solved_power, "achieved_power": achieved, "alpha": alpha,
        },
    )


# =========================================================================== curve


def power_curve(
    *,
    effect_size: float,
    design: str = "unpaired",
    alpha: float = 0.05,
    alternative: str = "two-sided",
    ratio: float = 1.0,
    k_groups: int | None = None,
    n_values: Sequence[float] | None = None,
    n_max: int | None = None,
    target_power: float = 0.8,
) -> StatResult:
    """Power as a function of the sample size, for plotting.

    The x axis is the sample size per group (ANOVA; unpaired design with equal groups),
    the size of group 1 (unpaired design with ``ratio`` != 1; group 2 has ``ratio`` x n),
    the number of pairs (paired) or the number of values (one-sample).

    ``design`` is ``"unpaired"``, ``"paired"``, ``"one-sample"`` or ``"anova"`` (then
    ``k_groups`` is required and ``effect_size`` is Cohen's f). ``extra["curve"]`` is
    ``{"x": [n ...], "y": [power ...]}``; ``extra["target"]`` gives the n needed for
    ``target_power``. By default the curve runs from n = 2 up to the n that gives about
    99% power (override with ``n_max`` or explicit ``n_values``).
    """
    alpha = check_alpha(alpha)
    es_value = _number(effect_size, "The effect size")
    if es_value == 0:
        raise StatsError("An effect size of zero has no power curve; enter a non-zero effect size.")
    target_power = _check_power(target_power, alpha)
    if design == "anova":
        if k_groups is None:
            raise StatsError("Give the number of groups (k_groups) for an ANOVA power curve.")
        solved = power_anova(k_groups=k_groups, effect_size=es_value, power=target_power, alpha=alpha)
        k = int(k_groups)

        def pw(size: float) -> float:
            return _anova_power(es_value, size, alpha, k)

        label = f"one-way ANOVA ({k} groups)"
        lo = 2.0
    else:
        solved = power_ttest(design=design, effect_size=es_value, power=target_power, alpha=alpha,
                             alternative=alternative, ratio=ratio)
        ratio_used = solved.extra["ratio"]

        def pw(size: float) -> float:
            return _t_power(design, es_value, size, alpha, alternative, ratio_used)

        label = _DESIGNS[design]
        lo = max(2.0, 2.0 / ratio_used) if design == "unpaired" else 2.0
    if n_values is not None:
        xs = sorted({_number(v, "Each sample size") for v in n_values})
        if not xs or xs[0] < lo:
            raise StatsError(f"Sample sizes for the curve must be at least {fmt(lo)}.")
    else:
        if n_max is None:
            high = _solve_increasing(pw, max(0.99, target_power), lo, "sample size",
                                     "The effect is too small for a power curve.")
            top = max(int(math.ceil(high)), int(math.ceil(lo)) + 8)
        else:
            top = int(_number(n_max, "The largest sample size"))
            if top <= lo:
                raise StatsError(f"The largest sample size must be greater than {fmt(lo)}.")
        start = int(math.ceil(lo))
        count = top - start + 1
        if count <= 300:
            xs = [float(v) for v in range(start, top + 1)]
        else:
            xs = sorted({float(round(v)) for v in np.linspace(start, top, 300)})
    ys = [pw(v) for v in xs]
    n_needed = solved.extra["n"]
    if design == "paired":
        x_what, x_label, needs = "the number of pairs", "Number of pairs", f"{n_needed} pairs"
    elif design == "one-sample":
        x_what, x_label, needs = "the number of values", "Number of values", f"{n_needed} values"
    elif design == "unpaired" and solved.extra["ratio"] != 1.0:
        n2_needed = solved.extra["n2"]
        x_what = (f"the size of group 1 (group 2 has {fmt(solved.extra['ratio'])} times as many "
                  "subjects)")
        x_label = "Sample size of group 1"
        needs = (f"{n_needed} subjects in group 1 and {n2_needed} in group 2 "
                 f"({solved.extra['n_total']} in total)")
    else:
        x_what, x_label, needs = "the sample size per group", "Sample size per group", f"{n_needed} per group"
    interpretation = (
        f"Power of the {label} at alpha = {alpha:g} for an effect size of {fmt(es_value, 3)}, as a function "
        f"of {x_what} (from {fmt(xs[0])} to {fmt(xs[-1])}). Power rises from "
        f"{_pct(ys[0])} to {_pct(ys[-1])}; {_pct(target_power)} power needs {needs}."
    )
    return StatResult(
        analysis="power_curve",
        test_name=f"Power curve: {label}",
        why_chosen=(
            "A power curve shows how the chance of detecting the assumed effect grows with the "
            "sample size, which helps to choose a sample size that balances cost against power."
        ),
        assumptions=list(_PLANNING_ASSUMPTIONS),
        alpha=alpha,
        alternative=alternative if design != "anova" else "two-sided",
        parameters=[
            Parameter(name="Effect size", value=es_value,
                      unit_note="Cohen's f." if design == "anova" else "Cohen's d."),
            Parameter(name=f"{x_label} for {_pct(target_power)} power", value=n_needed),
        ],
        interpretation=interpretation,
        methods_text=solved.methods_text,
        citations=list(_CITATIONS),
        extra={
            "curve": {"x": xs, "y": ys},
            "x_label": x_label,
            "y_label": "Power",
            "target": {"power": target_power, "n": n_needed, "n_exact": solved.extra["n_exact"]},
            "design": design, "effect_size": es_value, "alpha": alpha,
        },
    )

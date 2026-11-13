"""Assumption checks: normality tests and equal-variance (homoscedasticity) tests.

Normality
---------
* Shapiro-Wilk: Shapiro SS, Wilk MB (1965) An analysis of variance test for normality
  (complete samples). *Biometrika* 52:591-611; P value by Royston P (1995) Remark AS R94.
  *Appl Stat* 44:547-551 (as implemented in scipy). The P value is approximate for
  n > 5000.
* D'Agostino-Pearson omnibus K2: D'Agostino RB, Pearson ES (1973) Tests for departure
  from normality. *Biometrika* 60:613-622; D'Agostino RB, Belanger A, D'Agostino RB Jr
  (1990) A suggestion for using powerful and informative tests of normality.
  *Am Stat* 44:316-321. Combines a skewness and a kurtosis z score; needs n >= 8 and
  the kurtosis part is only accurate from about n = 20.
* Kolmogorov-Smirnov distance with the **Lilliefors** P value: Lilliefors HW (1967) On
  the Kolmogorov-Smirnov test for normality with mean and variance unknown.
  *J Am Stat Assoc* 62:399-402. The classical KS P value assumes the mean and SD of the
  reference normal distribution were fixed in advance. When they are estimated from the
  same sample (which is what everybody does) the classical P value is far too large, i.e.
  **an uncorrected KS P value is invalid**. We therefore always report the Lilliefors-
  corrected P value (statsmodels' simulation-based table, bounded to 0.001 - 0.99).
* Anderson-Darling: Anderson TW, Darling DA (1952) *Ann Math Stat* 23:193-212;
  Stephens MA (1974) EDF statistics for goodness of fit and some comparisons.
  *J Am Stat Assoc* 69:730-737; P value from the modified statistic
  A* = A2 (1 + 0.75/n + 2.25/n^2) with the formulas of D'Agostino RB, Stephens MA (1986)
  *Goodness-of-Fit Techniques*, Table 4.9. Computed here from first principles.

Equal variances
---------------
* Levene H (1960) Robust tests for equality of variances. In: *Contributions to
  Probability and Statistics*, pp 278-292: one-way ANOVA on the absolute deviations from
  each group's mean.
* Brown MB, Forsythe AB (1974) Robust tests for the equality of variances.
  *J Am Stat Assoc* 69:364-367: the same with deviations from each group's **median**
  (more robust for skewed data; our default recommendation).
* Bartlett MS (1937) Properties of sufficiency and statistical tests. *Proc R Soc A*
  160:268-282. Powerful for normal data but very sensitive to non-normality.
* Two-sided F test for the ratio of two variances (Snedecor & Cochran, *Statistical
  Methods*). Also very sensitive to non-normality.

Levene and Brown-Forsythe are implemented here from the defining formulas (and
cross-checked against scipy in the tests).
"""

from __future__ import annotations

import math
import warnings as _warnings
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
from scipy import stats
from statsmodels.stats.diagnostic import lilliefors as _sm_lilliefors

from ..data.tables import DataTable
from ._utils import StatsError, check_alpha, clean, fmt, is_constant, p_phrase, sig_phrase
from .descriptives import _describe_clean, _name_list
from .registry import ParamSpec, register
from .result import AssumptionCheck, Parameter, StatResult

NORMALITY_TESTS = ("shapiro", "dagostino", "ks", "anderson")

NORMALITY_LABELS = {
    "shapiro": "Shapiro-Wilk",
    "dagostino": "D'Agostino-Pearson",
    "ks": "Kolmogorov-Smirnov (Lilliefors P value)",
    "anderson": "Anderson-Darling",
}

_NORMALITY_MIN_N = {"shapiro": 3, "dagostino": 8, "ks": 5, "anderson": 8}

_NORMALITY_CITATIONS = {
    "shapiro": "Shapiro SS, Wilk MB (1965) Biometrika 52:591-611; Royston P (1995) Appl Stat 44:547-551.",
    "dagostino": "D'Agostino RB, Pearson ES (1973) Biometrika 60:613-622; D'Agostino RB, Belanger A, "
                 "D'Agostino RB Jr (1990) Am Stat 44:316-321.",
    "ks": "Lilliefors HW (1967) J Am Stat Assoc 62:399-402 (Kolmogorov-Smirnov test with estimated "
          "mean and variance).",
    "anderson": "Anderson TW, Darling DA (1952) Ann Math Stat 23:193-212; Stephens MA (1974) "
                "J Am Stat Assoc 69:730-737; D'Agostino RB, Stephens MA (1986) Goodness-of-Fit Techniques.",
}

_VARIANCE_CITATIONS = [
    "Brown MB, Forsythe AB (1974) Robust tests for the equality of variances. J Am Stat Assoc 69:364-367.",
    "Levene H (1960) Robust tests for equality of variances. In: Contributions to Probability and "
    "Statistics, Stanford Univ Press, pp 278-292.",
    "Bartlett MS (1937) Properties of sufficiency and statistical tests. Proc R Soc A 160:268-282.",
]

#: 5% critical value of the modified Anderson-Darling statistic A* (mean and SD estimated).
ANDERSON_CRITICAL_5PCT = 0.752


# ====================================================================== normality


def _prepare(x: Iterable[Any], minimum: int, test: str) -> np.ndarray:
    arr, _ = clean(x, "data")
    if arr.size < minimum:
        raise StatsError(
            f"The {test} test needs at least {minimum} values; there are only {arr.size}."
        )
    return arr


def _normality_dict(key: str, statistic_name: str, statistic: float | None, p_value: float | None,
                    n: int, alpha: float, notes: list[str], **more: Any) -> dict[str, Any]:
    normal = None if p_value is None else bool(p_value >= alpha)
    return {
        "key": key,
        "test": NORMALITY_LABELS[key],
        "statistic_name": statistic_name,
        "statistic": statistic,
        "p_value": p_value,
        "n": n,
        "alpha": alpha,
        "normal": normal,
        "warnings": notes,
        **more,
    }


_CONSTANT_NOTE = "All values are identical, so a normality test cannot be computed."


def shapiro_wilk(x: Iterable[Any], *, alpha: float = 0.05) -> dict[str, Any]:
    """Shapiro-Wilk W test of normality (Shapiro & Wilk 1965; Royston 1995).

    Needs n >= 3. W close to 1 means the ordered values line up with what a normal
    distribution predicts. For n > 5000 the P value is only approximate.
    """
    alpha = check_alpha(alpha)
    arr = _prepare(x, 3, "Shapiro-Wilk")
    notes: list[str] = []
    if is_constant(arr):
        return _normality_dict("shapiro", "W", None, None, int(arr.size), alpha, [_CONSTANT_NOTE])
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        res = stats.shapiro(arr)
    if arr.size > 5000:
        notes.append("With more than 5000 values the Shapiro-Wilk P value is only approximate.")
    return _normality_dict("shapiro", "W", float(res.statistic), float(res.pvalue), int(arr.size), alpha, notes)


def _skewness_z(arr: np.ndarray) -> float:
    """z score of the sample skewness under normality (D'Agostino 1970; D'Agostino et al. 1990)."""
    n = arr.size
    dev = arr - arr.mean()
    m2 = float(np.mean(dev**2))
    b1 = float(np.mean(dev**3)) / m2**1.5
    y = b1 * math.sqrt((n + 1.0) * (n + 3.0) / (6.0 * (n - 2.0)))
    beta2 = 3.0 * (n * n + 27.0 * n - 70.0) * (n + 1.0) * (n + 3.0) / ((n - 2.0) * (n + 5.0) * (n + 7.0) * (n + 9.0))
    w2 = -1.0 + math.sqrt(2.0 * (beta2 - 1.0))
    delta = 1.0 / math.sqrt(0.5 * math.log(w2))
    alpha = math.sqrt(2.0 / (w2 - 1.0))
    return delta * math.asinh(y / alpha)


def _kurtosis_z(arr: np.ndarray) -> float:
    """z score of the sample kurtosis under normality (Anscombe & Glynn 1983); NaN if undefined."""
    n = arr.size
    dev = arr - arr.mean()
    m2 = float(np.mean(dev**2))
    b2 = float(np.mean(dev**4)) / m2**2
    mean_b2 = 3.0 * (n - 1.0) / (n + 1.0)
    var_b2 = 24.0 * n * (n - 2.0) * (n - 3.0) / ((n + 1.0) ** 2 * (n + 3.0) * (n + 5.0))
    x = (b2 - mean_b2) / math.sqrt(var_b2)
    sqrt_beta1 = (6.0 * (n * n - 5.0 * n + 2.0) / ((n + 7.0) * (n + 9.0))
                  * math.sqrt(6.0 * (n + 3.0) * (n + 5.0) / (n * (n - 2.0) * (n - 3.0))))
    a = 6.0 + 8.0 / sqrt_beta1 * (2.0 / sqrt_beta1 + math.sqrt(1.0 + 4.0 / sqrt_beta1**2))
    denom = 1.0 + x * math.sqrt(2.0 / (a - 4.0))
    if denom == 0.0:
        return math.nan
    return ((1.0 - 2.0 / (9.0 * a)) - float(np.cbrt((1.0 - 2.0 / a) / denom))) / math.sqrt(2.0 / (9.0 * a))


def dagostino_pearson(x: Iterable[Any], *, alpha: float = 0.05) -> dict[str, Any]:
    """D'Agostino-Pearson omnibus K2 test (D'Agostino & Pearson 1973; D'Agostino et al. 1990).

    K2 = Z(skewness)^2 + Z(kurtosis)^2, compared with chi-square (2 df). Needs n >= 8.
    The result also carries the two z scores (``z_skewness``, ``z_kurtosis``).

    Computed here from the published formulas (the skewness transformation of D'Agostino
    1970 and the kurtosis transformation of Anscombe & Glynn 1983) and cross-checked
    against ``scipy.stats.normaltest`` in the tests. We do not call scipy because its
    skewness test returns a non-zero z score when the sample skewness is *exactly* zero
    (perfectly symmetric data), where the correct value is 0.
    """
    alpha = check_alpha(alpha)
    arr = _prepare(x, 8, "D'Agostino-Pearson")
    n = int(arr.size)
    if is_constant(arr):
        return _normality_dict("dagostino", "K2", None, None, n, alpha, [_CONSTANT_NOTE])
    notes: list[str] = []
    with np.errstate(all="ignore"):
        z_skew = _skewness_z(arr)
        z_kurt = _kurtosis_z(arr)
    if n < 20:
        notes.append(
            "With fewer than 20 values the kurtosis part of the D'Agostino-Pearson test is only "
            "roughly calibrated."
        )
    statistic = z_skew * z_skew + z_kurt * z_kurt
    if not math.isfinite(statistic):
        return _normality_dict(
            "dagostino", "K2", None, None, n, alpha,
            ["The D'Agostino-Pearson statistic could not be computed for these values."],
        )
    p_value = float(stats.chi2.sf(statistic, 2))
    return _normality_dict("dagostino", "K2", float(statistic), p_value, n, alpha, notes,
                           z_skewness=z_skew, z_kurtosis=z_kurt)


def kolmogorov_smirnov_normal(x: Iterable[Any], *, alpha: float = 0.05) -> dict[str, Any]:
    """Kolmogorov-Smirnov distance from a normal distribution, with the Lilliefors P value.

    The reference normal distribution uses the sample mean and SD, so the classical KS P
    value would be invalid (far too large); the P value reported here is Lilliefors' (1967)
    correction, taken from statsmodels' simulation-based table. That table is bounded:
    P values below 0.001 are reported as 0.001 and above 0.99 as 0.99. Needs n >= 5.
    """
    alpha = check_alpha(alpha)
    arr = _prepare(x, 5, "Kolmogorov-Smirnov")
    n = int(arr.size)
    if is_constant(arr):
        return _normality_dict("ks", "D", None, None, n, alpha, [_CONSTANT_NOTE])
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        with np.errstate(all="ignore"):
            d, p = _sm_lilliefors(arr, dist="norm", pvalmethod="table")
    d, p = float(d), float(p)
    if not (math.isfinite(d) and math.isfinite(p)):
        return _normality_dict(
            "ks", "D", None, None, n, alpha,
            ["The Kolmogorov-Smirnov statistic could not be computed for these values."],
        )
    notes: list[str] = []
    p_note = "Lilliefors-corrected (mean and SD estimated from the data)"
    if p <= 0.001 + 1e-12:
        p_note += "; the true P value is 0.001 or smaller (limit of the table)"
    elif p >= 0.99 - 1e-12:
        p_note += "; the true P value is 0.99 or larger (limit of the table)"
    return _normality_dict("ks", "D", d, min(max(p, 0.0), 1.0), n, alpha, notes, p_note=p_note)


def _anderson_p(a_star: float) -> float:
    """P value for the modified statistic A* (D'Agostino & Stephens 1986, Table 4.9)."""
    if a_star < 0.2:
        p = 1.0 - math.exp(-13.436 + 101.14 * a_star - 223.73 * a_star**2)
    elif a_star < 0.34:
        p = 1.0 - math.exp(-8.318 + 42.796 * a_star - 59.938 * a_star**2)
    elif a_star < 0.6:
        p = math.exp(0.9177 - 4.279 * a_star - 1.38 * a_star**2)
    elif a_star < 153.0:
        p = math.exp(1.2937 - 5.709 * a_star + 0.0186 * a_star**2)
    else:  # the quadratic term would turn upwards far outside the formula's range
        p = 0.0
    return min(max(p, 0.0), 1.0)


def anderson_darling(x: Iterable[Any], *, alpha: float = 0.05) -> dict[str, Any]:
    """Anderson-Darling test of normality with estimated mean and SD (Stephens 1974).

    ``statistic`` is A2; ``statistic_modified`` is A* = A2 (1 + 0.75/n + 2.25/n^2), which
    is compared with the 5% critical value 0.752 (``exceeds_5pct_critical``) and converted
    to a P value with the D'Agostino & Stephens (1986) formulas. Needs n >= 8.
    """
    alpha = check_alpha(alpha)
    arr = _prepare(x, 8, "Anderson-Darling")
    n = int(arr.size)
    if is_constant(arr):
        return _normality_dict("anderson", "A2", None, None, n, alpha, [_CONSTANT_NOTE])
    z = np.sort((arr - arr.mean()) / arr.std(ddof=1))
    i = np.arange(1, n + 1)
    with np.errstate(all="ignore"):
        a2 = float(-n - np.sum((2 * i - 1) * (stats.norm.logcdf(z) + stats.norm.logsf(z[::-1]))) / n)
    if not math.isfinite(a2):
        return _normality_dict(
            "anderson", "A2", None, None, n, alpha,
            ["The Anderson-Darling statistic could not be computed for these values."],
        )
    a_star = a2 * (1.0 + 0.75 / n + 2.25 / n**2)
    return _normality_dict(
        "anderson", "A2", a2, _anderson_p(a_star), n, alpha, [],
        statistic_modified=a_star,
        critical_5pct=ANDERSON_CRITICAL_5PCT,
        exceeds_5pct_critical=bool(a_star > ANDERSON_CRITICAL_5PCT),
    )


_NORMALITY_FUNCS = {
    "shapiro": shapiro_wilk,
    "dagostino": dagostino_pearson,
    "ks": kolmogorov_smirnov_normal,
    "anderson": anderson_darling,
}


def qq_data(x: Iterable[Any]) -> dict[str, Any]:
    """Coordinates of a normal quantile-quantile (QQ) plot.

    ``theoretical``: expected normal order statistics (Filliben's estimate, as in
    scipy.stats.probplot); ``ordered``: the sorted data; ``slope`` / ``intercept``: the
    least-squares line through the points; ``r``: their correlation (``None`` when all
    values are identical). Points close to the line suggest a normal distribution.
    """
    arr, _ = clean(x, "data")
    if arr.size < 2:
        raise StatsError(f"A QQ plot needs at least 2 values; there are only {arr.size}.")
    constant = is_constant(arr)
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        with np.errstate(all="ignore"):
            (theoretical, ordered), (slope, intercept, r) = stats.probplot(arr, dist="norm", fit=True)
    if constant:
        slope, intercept, r = 0.0, float(arr[0]), None
    else:
        slope, intercept = float(slope), float(intercept)
        r = float(r) if math.isfinite(float(r)) else None
    return {
        "theoretical": [float(v) for v in theoretical],
        "ordered": [float(v) for v in ordered],
        "slope": slope,
        "intercept": intercept,
        "r": r,
    }


def _check_tests(tests: Sequence[str] | str) -> list[str]:
    if isinstance(tests, str):
        tests = [tests]
    aliases = {
        "shapiro-wilk": "shapiro", "sw": "shapiro", "dagostino-pearson": "dagostino",
        "d'agostino": "dagostino", "k2": "dagostino", "omnibus": "dagostino",
        "kolmogorov-smirnov": "ks", "lilliefors": "ks", "anderson-darling": "anderson", "ad": "anderson",
    }
    out: list[str] = []
    for t in tests:
        key = aliases.get(str(t).strip().lower(), str(t).strip().lower())
        if key not in NORMALITY_TESTS:
            raise StatsError(f"Unknown normality test '{t}'. Choose from {list(NORMALITY_TESTS)}.")
        if key not in out:
            out.append(key)
    if not out:
        raise StatsError(f"Choose at least one normality test from {list(NORMALITY_TESTS)}.")
    return out


def normality_tests(
    groups: dict[str, Iterable[Any]],
    *,
    tests: Sequence[str] | str = ("shapiro", "dagostino", "ks"),
    alpha: float = 0.05,
) -> StatResult:
    """Run normality tests on every group.

    One :class:`AssumptionCheck` per (group, test) goes in ``assumptions``; the same
    numbers are in ``extra["table"]``; ``extra["qq"]`` holds QQ-plot coordinates per
    group. A test that cannot be run on a group (too few values, identical values) is
    reported with status ``warn`` instead of failing the whole analysis.
    """
    alpha = check_alpha(alpha)
    test_keys = _check_tests(tests)
    if not groups:
        raise StatsError("No groups were given. Enter at least one column of numbers.")
    checks: list[AssumptionCheck] = []
    table: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    qq: dict[str, Any] = {}
    n: dict[str, int] = {}
    notes: list[str] = []
    warnings: list[str] = []
    failed: dict[str, list[str]] = {}
    untestable: list[str] = []
    n_done = 0
    for name, values in groups.items():
        label = str(name)
        arr, group_notes = clean(values, label)
        notes.extend(group_notes)
        n[label] = int(arr.size)
        if arr.size:
            rows.append({"group": label, **_describe_clean(arr, 0.95)})
        if arr.size >= 2:
            qq[label] = qq_data(arr)
        group_done = 0
        for key in test_keys:
            test_label = NORMALITY_LABELS[key]
            check_name = f"Normality ({test_label})"
            if arr.size < _NORMALITY_MIN_N[key]:
                detail = (
                    f"Needs at least {_NORMALITY_MIN_N[key]} values; this group has {arr.size}. "
                    "Not tested."
                )
                checks.append(AssumptionCheck(name=check_name, target=label, status="warn", detail=detail))
                table.append({"group": label, "test": test_label, "statistic_name": None, "statistic": None,
                              "p_value": None, "n": int(arr.size), "passed": None})
                continue
            res = _NORMALITY_FUNCS[key](arr, alpha=alpha)
            for w in res["warnings"]:
                text = f"{label}: {w}"
                if text not in warnings:
                    warnings.append(text)
            if res["p_value"] is None:
                status, detail = "warn", " ".join(res["warnings"]) or "Could not be computed."
            elif res["normal"]:
                status = "pass"
                detail = f"No evidence against normality ({p_phrase(res['p_value'])})."
                group_done += 1
            else:
                status = "fail"
                detail = f"The values deviate from a normal distribution ({p_phrase(res['p_value'])})."
                failed.setdefault(label, []).append(test_label)
                group_done += 1
            if res.get("p_note") and res["p_value"] is not None:
                detail += f" P value: {res['p_note']}."
            checks.append(AssumptionCheck(name=check_name, target=label, statistic=res["statistic"],
                                          p_value=res["p_value"], status=status, detail=detail))
            table.append({"group": label, "test": test_label, "statistic_name": res["statistic_name"],
                          "statistic": res["statistic"], "p_value": res["p_value"], "n": res["n"],
                          "passed": res["normal"]})
        n_done += group_done
        if group_done == 0:
            untestable.append(label)
    if sum(n.values()) == 0:
        raise StatsError("There are no numeric values to test (every group is empty or all missing).")

    tested = [g for g in n if g not in untestable]
    test_names = ", ".join(NORMALITY_LABELS[k] for k in test_keys)
    parts: list[str] = [
        f"Normality was checked in {len(n)} group{'s' if len(n) != 1 else ''} with: {test_names} "
        f"(alpha = {alpha:g})."
    ]
    if not tested:
        parts.append(
            "None of the groups could be tested (too few values, or all values identical), so "
            "normality cannot be judged from these data."
        )
    elif not failed:
        parts.append(
            f"No test found evidence against a normal distribution in {_name_list(tested)}. This "
            "does not show that the data are normal, only that a deviation was not detected."
        )
    else:
        desc = "; ".join(f"{g} ({', '.join(t)})" for g, t in failed.items())
        ok = [g for g in tested if g not in failed]
        parts.append(f"The values deviate from a normal distribution in: {desc}.")
        if ok:
            parts.append(f"No deviation was detected in {_name_list(ok)}.")
        parts.append(
            "Consider a nonparametric test, or a transformation (for example logarithms) if the "
            "data are skewed."
        )
    if untestable and tested:
        parts.append(f"Could not be tested: {_name_list(untestable)}.")
    parts.append(
        "Keep the sample size in mind: with very few values these tests can miss even strong "
        "non-normality, and with very many values they flag deviations too small to matter. "
        "Look at the QQ plot as well."
    )
    if untestable:
        warnings.append(
            f"Normality could not be tested for: {_name_list(untestable)} (too few values or no spread)."
        )
    if any(0 < size < 8 for size in n.values()):
        warnings.append(
            "Some groups have fewer than 8 values; normality tests have very little power at this size."
        )
    return StatResult(
        analysis="normality",
        test_name="Normality tests",
        why_chosen=(
            "Normality tests ask whether the values in each group could plausibly be a sample from "
            "a normal (Gaussian) distribution, which t tests, ANOVA and linear regression assume."
        ),
        assumptions=checks,
        alpha=alpha,
        n=n,
        descriptives=rows,
        warnings=warnings,
        data_notes=notes,
        interpretation=" ".join(parts),
        methods_text=(
            f"Normality was assessed in each group with the following test(s): {test_names} "
            f"(alpha = {alpha:g}); the Kolmogorov-Smirnov P value, where reported, uses the "
            "Lilliefors correction."
            if "ks" in test_keys else
            f"Normality was assessed in each group with the following test(s): {test_names} "
            f"(alpha = {alpha:g})."
        ),
        citations=[_NORMALITY_CITATIONS[k] for k in test_keys],
        extra={"table": table, "qq": qq, "tests": test_keys},
    )


# ================================================================== equal variances


def _variance_groups(groups: Any, minimum: int = 2, at_least: int = 2) -> tuple[dict[str, np.ndarray], list[str]]:
    if groups is None:
        raise StatsError("No groups were given.")
    if not isinstance(groups, dict):
        groups = {f"Group {i + 1}": g for i, g in enumerate(groups)}
    out: dict[str, np.ndarray] = {}
    notes: list[str] = []
    for name, values in groups.items():
        arr, group_notes = clean(values, str(name))
        notes.extend(group_notes)
        out[str(name)] = arr
    if len(out) < at_least:
        raise StatsError(f"Comparing variances needs at least {at_least} groups; {len(out)} given.")
    small = [f"{name} (n = {arr.size})" for name, arr in out.items() if arr.size < minimum]
    if small:
        raise StatsError(
            f"Every group needs at least {minimum} values to compare variances; too few in: "
            f"{', '.join(small)}."
        )
    return out, notes


def _variance_dict(test: str, statistic_name: str, statistic: float | None, df1: float | None,
                   df2: float | None, p_value: float | None, notes: list[str], **more: Any) -> dict[str, Any]:
    return {"test": test, "statistic_name": statistic_name, "statistic": statistic, "df1": df1,
            "df2": df2, "p_value": p_value, "warnings": notes, **more}


def levene_test(groups: Any, *, center: str = "mean") -> dict[str, Any]:
    """Levene's test for equal variances: one-way ANOVA F on absolute deviations.

    ``center="mean"`` is Levene's (1960) original test; ``center="median"`` is the
    Brown-Forsythe (1974) variant. With z_ij = abs(x_ij - center_i), N values and k groups:

        F = [(N - k) / (k - 1)] * sum_i n_i (zbar_i - zbar)^2 / sum_ij (z_ij - zbar_i)^2

    compared with the F distribution on (k - 1, N - k) degrees of freedom.
    ``groups`` is ``{name: values}`` (or a sequence of arrays). When the deviations have no
    spread within the groups (e.g. every group has only 2 values) F is undefined and the
    statistic and P value are ``None`` with an explanatory warning.
    """
    if center not in {"mean", "median"}:
        raise StatsError("center must be 'mean' (Levene) or 'median' (Brown-Forsythe).")
    data, _ = _variance_groups(groups)
    name = "Levene" if center == "mean" else "Brown-Forsythe"
    arrays = list(data.values())
    k = len(arrays)
    n_total = int(sum(a.size for a in arrays))
    df1, df2 = float(k - 1), float(n_total - k)
    z = [np.abs(a - (a.mean() if center == "mean" else np.median(a))) for a in arrays]
    z_means = np.array([zi.mean() for zi in z])
    sizes = np.array([zi.size for zi in z], dtype=float)
    grand = float(np.concatenate(z).mean())
    ss_between = float(np.sum(sizes * (z_means - grand) ** 2))
    ss_within = float(sum(np.sum((zi - zm) ** 2) for zi, zm in zip(z, z_means)))
    scale = max(float(np.max(np.abs(np.concatenate(arrays)))), 1e-300)
    tol = n_total * (1e-12 * scale) ** 2
    if ss_within <= tol:
        if ss_between <= tol:
            note = ("The spread is identical in every group (no variability at all, or exactly equal "
                    "deviations), so the test statistic is undefined.")
        else:
            note = ("Within each group all deviations from the centre are identical (this happens when "
                    "groups have only 2 values), so the test cannot be computed.")
        return _variance_dict(name, "F", None, df1, df2, None, [note], center=center)
    f = (ss_between / df1) / (ss_within / df2)
    p = float(stats.f.sf(f, df1, df2))
    return _variance_dict(name, "F", float(f), df1, df2, p, [], center=center)


def brown_forsythe_test(groups: Any) -> dict[str, Any]:
    """Brown-Forsythe (1974) test: Levene's test with deviations from the group medians."""
    return levene_test(groups, center="median")


def bartlett_test(groups: Any) -> dict[str, Any]:
    """Bartlett's (1937) test for equal variances (chi-square with k - 1 df).

    Most powerful when every group is normal, but very sensitive to non-normality: a
    small P value can reflect skewed or heavy-tailed data rather than unequal variances.
    """
    data, _ = _variance_groups(groups)
    arrays = list(data.values())
    df1 = float(len(arrays) - 1)
    if any(is_constant(a) for a in arrays):
        return _variance_dict(
            "Bartlett", "chi2", None, df1, None, None,
            ["At least one group has no variability, so Bartlett's test cannot be computed."],
        )
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        res = stats.bartlett(*arrays)
    statistic, p = float(res.statistic), float(res.pvalue)
    if not (math.isfinite(statistic) and math.isfinite(p)):
        return _variance_dict("Bartlett", "chi2", None, df1, None, None,
                              ["Bartlett's test could not be computed for these values."])
    return _variance_dict("Bartlett", "chi2", statistic, df1, None, p, [])


def f_test_two_variances(a: Iterable[Any], b: Iterable[Any], *, ci_level: float = 0.95) -> dict[str, Any]:
    """Two-sided F test for the ratio of two variances.

    F = var(a) / var(b) on (n_a - 1, n_b - 1) degrees of freedom; the two-sided P value is
    twice the smaller tail area. ``ratio_ci_low`` / ``ratio_ci_high`` bound the true
    variance ratio. Assumes both samples are normal and is very sensitive to that.
    """
    if not (0.0 < float(ci_level) < 1.0):
        raise StatsError("The confidence level must be between 0 and 1 (e.g. 0.95).")
    data, _ = _variance_groups({"a": a, "b": b})
    x, y = data["a"], data["b"]
    df1, df2 = float(x.size - 1), float(y.size - 1)
    if is_constant(x) or is_constant(y):
        return _variance_dict(
            "F test of two variances", "F", None, df1, df2, None,
            ["At least one group has no variability, so the variance ratio cannot be tested."],
            ratio_ci_low=None, ratio_ci_high=None, ci_level=float(ci_level),
        )
    f = float(x.var(ddof=1) / y.var(ddof=1))
    p = float(min(1.0, 2.0 * min(stats.f.cdf(f, df1, df2), stats.f.sf(f, df1, df2))))
    tail = (1.0 - float(ci_level)) / 2.0
    lo = f / float(stats.f.ppf(1.0 - tail, df1, df2))
    hi = f / float(stats.f.ppf(tail, df1, df2))
    return _variance_dict("F test of two variances", "F", f, df1, df2, p, [],
                          ratio_ci_low=lo, ratio_ci_high=hi, ci_level=float(ci_level))


def _variance_check(res: dict[str, Any], alpha: float, extra_detail: str = "") -> AssumptionCheck:
    name = f"Equal variances ({res['test']})"
    if res["p_value"] is None:
        return AssumptionCheck(name=name, status="warn",
                               detail=" ".join(res["warnings"]) or "Could not be computed.")
    if res["p_value"] < alpha:
        status = "fail"
        detail = f"The groups differ in spread ({p_phrase(res['p_value'])})."
    else:
        status = "pass"
        detail = f"No evidence that the groups differ in spread ({p_phrase(res['p_value'])})."
    if extra_detail:
        detail += " " + extra_detail
    return AssumptionCheck(name=name, statistic=res["statistic"], p_value=res["p_value"],
                           status=status, detail=detail)


def equal_variance_tests(groups: dict[str, Iterable[Any]], *, alpha: float = 0.05) -> StatResult:
    """Compare the variances (spread) of two or more groups.

    Runs Brown-Forsythe (the headline result: ``statistic``, ``df``, ``df2``, ``p_value``),
    Levene and Bartlett, plus the two-sided F test when there are exactly two groups.
    Every test is listed in ``assumptions`` and in ``extra["table"]``.
    """
    alpha = check_alpha(alpha)
    data, notes = _variance_groups(groups)
    names = list(data)
    results = [brown_forsythe_test(data), levene_test(data, center="mean"), bartlett_test(data)]
    if len(names) == 2:
        results.append(f_test_two_variances(data[names[0]], data[names[1]]))
    bf = results[0]
    details = {
        "Brown-Forsythe": "Recommended: robust when the data are not normal.",
        "Levene": "",
        "Bartlett": "Very sensitive to non-normal data.",
        "F test of two variances": "Very sensitive to non-normal data.",
    }
    checks = [_variance_check(r, alpha, details[r["test"]]) for r in results]
    checks.append(AssumptionCheck(
        name="Independent observations", status="info",
        detail="Each value must be an independent measurement (not repeated measures of one subject).",
    ))
    warnings: list[str] = []
    for r in results:
        for w in r["warnings"]:
            text = f"{r['test']}: {w}"
            if text not in warnings:
                warnings.append(text)
    if any(a.size < 5 for a in data.values()):
        warnings.append(
            "Some groups have fewer than 5 values; tests for equal variances have little power at this size."
        )
    rows = []
    for name, arr in data.items():
        d = _describe_clean(arr, 0.95)
        rows.append({"group": name, "n": d["n"], "mean": d["mean"], "median": d["median"], "sd": d["sd"],
                     "variance": None if d["sd"] is None else d["sd"] ** 2, "iqr": d["iqr"]})
    sds = [r["sd"] for r in rows]
    parameters: list[Parameter] = []
    lo_sd, hi_sd = min(sds), max(sds)
    ratio = hi_sd / lo_sd if lo_sd > 0 else None
    parameters.append(Parameter(name="SD ratio (largest / smallest)", value=ratio,
                                unit_note="Ratio of the largest to the smallest group SD."))
    if len(names) == 2:
        ft = results[-1]
        parameters.append(Parameter(
            name=f"Variance ratio ({names[0]} / {names[1]})", value=ft["statistic"],
            ci_low=ft["ratio_ci_low"], ci_high=ft["ratio_ci_high"], statistic=ft["statistic"],
            p_value=ft["p_value"], unit_note="CI from the F distribution; assumes normal data.",
        ))
    table = [{"test": r["test"], "statistic_name": r["statistic_name"], "statistic": r["statistic"],
              "df1": r["df1"], "df2": r["df2"], "p_value": r["p_value"],
              "equal_variances": None if r["p_value"] is None else bool(r["p_value"] >= alpha)}
             for r in results]

    hi_name = rows[int(np.argmax(sds))]["group"]
    lo_name = rows[int(np.argmin(sds))]["group"]
    ratio_text = (
        f"The SD ranges from {fmt(lo_sd)} ({lo_name}) to {fmt(hi_sd)} ({hi_name})"
        + (f", a {fmt(ratio, 3)}-fold difference." if ratio is not None else ".")
    )
    if bf["p_value"] is None:
        verdict = ("The Brown-Forsythe test could not be computed for these data, so equality of "
                   "variances cannot be judged formally.")
    elif bf["p_value"] < alpha:
        verdict = (
            f"The Brown-Forsythe test found evidence that the spread differs between the {len(names)} "
            f"groups: the difference is {sig_phrase(bf['p_value'], alpha)}; F({fmt(bf['df1'])}, "
            f"{fmt(bf['df2'])}) = {fmt(bf['statistic'])}. Prefer tests that do not assume equal variances "
            "(Welch's t test or Welch's ANOVA), or consider transforming the data."
        )
    else:
        verdict = (
            f"The Brown-Forsythe test found no evidence that the spread differs between the {len(names)} "
            f"groups: the difference is {sig_phrase(bf['p_value'], alpha)}; F({fmt(bf['df1'])}, "
            f"{fmt(bf['df2'])}) = {fmt(bf['statistic'])}. This does not prove the variances are equal, "
            "especially with small samples."
        )
    disagree = {r["p_value"] < alpha for r in results if r["p_value"] is not None}
    extra_sentence = ""
    if len(disagree) > 1:
        extra_sentence = (
            " The tests disagree; Bartlett's test and the F test react strongly to non-normal data, "
            "so rely on Brown-Forsythe unless you are confident the data are normal."
        )
    return StatResult(
        analysis="equal_variance",
        test_name="Tests for equal variances (Brown-Forsythe, Levene, Bartlett)",
        why_chosen=(
            "These tests ask whether the groups have the same spread (variance), which the ordinary "
            "t test and ANOVA assume. Brown-Forsythe is the headline result because it stays "
            "reliable when the data are not normal."
        ),
        assumptions=checks,
        statistic_name="F",
        statistic=bf["statistic"],
        df=bf["df1"],
        df2=bf["df2"],
        p_value=bf["p_value"],
        alpha=alpha,
        parameters=parameters,
        n={name: int(arr.size) for name, arr in data.items()},
        descriptives=rows,
        warnings=warnings,
        data_notes=notes,
        interpretation=f"{verdict} {ratio_text}{extra_sentence}",
        methods_text=(
            f"Homogeneity of variances was assessed with the Brown-Forsythe test (alpha = {alpha:g}); "
            "Levene's and Bartlett's tests are reported for comparison."
        ),
        citations=list(_VARIANCE_CITATIONS),
        extra={"table": table, "headline_test": "Brown-Forsythe"},
    )


# ------------------------------------------------------------------ table adapters


def select_groups(table: DataTable, groups: Any) -> dict[str, np.ndarray]:
    """Columns of a Column table chosen by a ``kind="groups"`` parameter (None/empty = all)."""
    available = table.groups()
    if groups is None or groups == "" or (isinstance(groups, list | tuple) and len(groups) == 0):
        return dict(available)
    names = [groups] if isinstance(groups, str) else [str(g) for g in groups]
    unknown = [g for g in names if g not in available]
    if unknown:
        raise StatsError(f"Unknown group(s): {unknown}. Available groups: {list(available)}.")
    return {g: available[g] for g in dict.fromkeys(names)}


def _multiple_variables_groups(table: DataTable, column: Any, by: Any) -> tuple[dict[str, np.ndarray], list[str]]:
    numeric = table.numeric_columns
    if column is None or column == "":
        candidates = [c for c in numeric if c != by]
        if not candidates:
            raise StatsError("This table has no numeric column to test.")
        column = candidates[0]
    if column not in table.data.columns:
        raise StatsError(f"Unknown column '{column}'. Available columns: {list(table.data.columns)}.")
    if column not in numeric:
        raise StatsError(f"Column '{column}' holds text, not numbers. Numeric columns: {numeric}.")
    frame = table.analysis_frame()
    notes: list[str] = []
    if by is None or by == "":
        return {column: frame[column].to_numpy(dtype=float)}, notes
    if by not in frame.columns:
        raise StatsError(f"Unknown grouping column '{by}'. Available columns: {list(table.data.columns)}.")
    labels = frame[by]
    missing = labels.isna()
    if bool(missing.any()):
        notes.append(f"{int(missing.sum())} row(s) with no value in '{by}' were left out.")
    text = labels.astype(object).where(missing, labels.astype(str))
    levels = list(dict.fromkeys(text[~missing].tolist()))
    if not levels:
        raise StatsError(f"The grouping column '{by}' is empty.")
    return {str(level): frame.loc[(~missing) & (text == level), column].to_numpy(dtype=float)
            for level in levels}, notes


_ALPHA_PARAM = ParamSpec(
    "alpha", "Significance level (alpha)", kind="float", default=0.05,
    help="The P value threshold below which a deviation is called significant. 0.05 is the usual choice.",
)

_GROUPS_PARAM = ParamSpec(
    "groups", "Groups to check", kind="groups", default=None, optional=True,
    help="Which columns to check. Leave empty to check every column.",
)


@register(
    key="normality",
    label="Normality tests",
    category="Assumption checks",
    tables=("column", "multiple_variables"),
    params=(
        _GROUPS_PARAM,
        ParamSpec(
            "column", "Variable to check", kind="column", default=None, optional=True,
            help="Multiple-variables tables only: the numeric column to test. Leave empty to use the "
                 "first numeric column.",
        ),
        ParamSpec(
            "by", "Split by", kind="column", default=None, optional=True,
            help="Multiple-variables tables only: a column of category labels. Normality is then "
                 "tested separately within each category.",
        ),
        _ALPHA_PARAM,
        ParamSpec(
            "include_anderson", "Also run Anderson-Darling", kind="bool", default=False,
            help="Adds the Anderson-Darling test, which pays extra attention to the tails of the "
                 "distribution. Needs at least 8 values per group.",
        ),
    ),
    description=(
        "Checks whether the values in each group look like a sample from a normal (bell-shaped) "
        "distribution, using the Shapiro-Wilk, D'Agostino-Pearson and Kolmogorov-Smirnov tests, and "
        "prepares a QQ plot."
    ),
    citation="Shapiro & Wilk (1965) Biometrika 52:591-611; D'Agostino & Pearson (1973) Biometrika "
             "60:613-622; Lilliefors (1967) J Am Stat Assoc 62:399-402",
)
def _run_normality(table: DataTable, groups: Any = None, column: Any = None, by: Any = None,
                   alpha: float = 0.05, include_anderson: bool = False, **_: Any) -> StatResult:
    extra_notes: list[str] = []
    if table.kind == "multiple_variables":
        data, extra_notes = _multiple_variables_groups(table, column, by)
    else:
        data = select_groups(table, groups)
    tests = ["shapiro", "dagostino", "ks"] + (["anderson"] if include_anderson else [])
    result = normality_tests(data, tests=tests, alpha=alpha)
    result.data_notes.extend(extra_notes)
    return result


@register(
    key="equal_variance",
    label="Equal variance tests",
    category="Assumption checks",
    tables=("column",),
    params=(_GROUPS_PARAM, _ALPHA_PARAM),
    description=(
        "Checks whether the groups have a similar spread (variance), which ordinary t tests and "
        "ANOVA assume, using the Brown-Forsythe, Levene and Bartlett tests."
    ),
    citation="Brown & Forsythe (1974) J Am Stat Assoc 69:364-367; Levene (1960); Bartlett (1937) "
             "Proc R Soc A 160:268-282",
)
def _run_equal_variance(table: DataTable, groups: Any = None, alpha: float = 0.05, **_: Any) -> StatResult:
    return equal_variance_tests(select_groups(table, groups), alpha=alpha)

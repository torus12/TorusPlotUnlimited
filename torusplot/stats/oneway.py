"""Comparing three or more groups: one-way ANOVA family and its rank-based relatives.

Array-level functions (numbers in, :class:`~torusplot.stats.result.StatResult` out):

* :func:`anova_oneway`          classic fixed-effects one-way ANOVA (Fisher 1925)
* :func:`welch_anova`           Welch's heteroscedastic ANOVA (Welch 1951)
* :func:`brown_forsythe_anova`  Brown-Forsythe F* test of means (Brown & Forsythe 1974)
* :func:`kruskal_wallis`        Kruskal-Wallis rank test (Kruskal & Wallis 1952)
* :func:`friedman_test`         Friedman test for related samples (Friedman 1937)

and the table-level adapters registered as ``anova_oneway``, ``welch_anova``,
``brown_forsythe_anova``, ``kruskal_wallis`` and ``friedman``.

References
----------
* Fisher RA (1925) *Statistical Methods for Research Workers*. Oliver & Boyd.
* Welch BL (1951) On the comparison of several mean values: an alternative approach.
  *Biometrika* 38:330-336.
* Brown MB, Forsythe AB (1974) The small sample behavior of some statistics which test
  the equality of several means. *Technometrics* 16:129-132.
* Mehrotra DV (1997) Improving the Brown-Forsythe solution to the generalized
  Behrens-Fisher problem. *Commun Stat Simul Comput* 26:1139-1145.
* Kruskal WH, Wallis WA (1952) Use of ranks in one-criterion variance analysis.
  *J Am Stat Assoc* 47:583-621.
* Friedman M (1937) The use of ranks to avoid the assumption of normality implicit in
  the analysis of variance. *J Am Stat Assoc* 32:675-701.
* Kendall MG, Babington Smith B (1939) The problem of m rankings.
  *Ann Math Stat* 10:275-287.
* Iman RL, Davenport JM (1980) Approximations of the critical region of the Friedman
  statistic. *Commun Stat Theory Methods* 9:571-595.
* Shapiro SS, Wilk MB (1965) An analysis of variance test for normality.
  *Biometrika* 52:591-611.
* Brown MB, Forsythe AB (1974) Robust tests for the equality of variances.
  *J Am Stat Assoc* 69:364-367.
* Bartlett MS (1937) Properties of sufficiency and statistical tests.
  *Proc R Soc A* 160:268-282.

Numerical range: F statistics, P values and effect sizes do not depend on the origin or
unit of the data, so they are computed from jointly rescaled values (``_utils.rescale``:
a common location is subtracted and the result divided by a power of two). Sums of
squares, mean squares, means, SDs and confidence limits are scaled back to the units of
the data; a quantity outside the representable range (about 1e-308 to 1.8e308, e.g. a sum of
squares of values near 1e200 or 1e-200) is reported as missing with a warning instead of
as infinity, zero or NaN.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy import stats

from . import assumptions as am
from . import effect_sizes as es
from . import posthoc as ph
from ._utils import (
    Rescaled,
    StatsError,
    check_alpha,
    ci_phrase,
    clean_groups,
    fmt,
    p_phrase,
    pow2_back,
    pow2_scaled,
    rescale,
    sig_phrase,
)
from .multiplicity import METHOD_LABELS
from .registry import ParamSpec, register
from .result import AssumptionCheck, Comparison, EffectSize, Parameter, StatResult

__all__ = [
    "anova_oneway",
    "brown_forsythe_anova",
    "friedman_test",
    "kruskal_wallis",
    "welch_anova",
]

CITE_FISHER = "Fisher RA (1925) Statistical Methods for Research Workers. Oliver & Boyd"
CITE_WELCH = "Welch BL (1951) Biometrika 38:330-336"
CITE_BF = "Brown MB, Forsythe AB (1974) Technometrics 16:129-132"
CITE_KW = "Kruskal WH, Wallis WA (1952) J Am Stat Assoc 47:583-621"
CITE_FRIEDMAN = "Friedman M (1937) J Am Stat Assoc 32:675-701"
CITE_KENDALL = "Kendall MG, Babington Smith B (1939) Ann Math Stat 10:275-287"

_EQUAL_VAR_POSTHOCS = ("tukey", "dunnett", "bonferroni", "sidak", "holm", "holm-sidak",
                       "fdr_bh", "lsd", "games-howell", "none")
_UNEQUAL_VAR_POSTHOCS = ("games-howell", "bonferroni", "sidak", "holm", "holm-sidak",
                         "fdr_bh", "lsd", "none")
_ADJUST_CHOICES = ("bonferroni", "sidak", "holm", "holm-sidak", "fdr_bh", "none")
_SHAPIRO_MAX_N = 5000


# ======================================================================== helpers


def _prepare(groups: Mapping[str, Any], what: str, min_each: int = 1,
             why_min: str = "") -> tuple[dict[str, np.ndarray], list[str]]:
    """Clean the groups and check there is enough data for ``what``."""
    if not isinstance(groups, Mapping):
        raise TypeError("groups must be a mapping {group name: values}.")
    cleaned, notes = clean_groups(dict(groups))
    if len(cleaned) < 2:
        raise StatsError(f"{what} compares groups: at least 2 groups are needed (3 or more is usual).")
    for name, arr in cleaned.items():
        if arr.size == 0:
            raise StatsError(
                f"Group '{name}' has no values. Add data or leave this group out of the analysis."
            )
        if arr.size < min_each:
            raise StatsError(
                f"Group '{name}' has {arr.size} value(s); {what} needs at least {min_each} in "
                f"every group{why_min}."
            )
    return cleaned, notes


def _two_group_note(k: int, equivalent: str) -> list[str]:
    if k == 2:
        return [f"Only two groups were compared. {equivalent}"]
    return []


def _sums_of_squares(arrays: Sequence[np.ndarray], constant: Sequence[bool] | None = None,
                     noise: float | None = None) -> dict[str, Any]:
    """Between / within / total sums of squares, computed to survive large offsets.

    The data are centred on the grand mean first (the subtraction of nearby numbers is
    exact in floating point), every sum uses exactly-rounded summation (``math.fsum``),
    and group means are refined with one correction pass. Groups whose values agree to
    rounding error contribute exactly zero to the within-group sum.

    For rescaled data (see :func:`_scaled_ss`) ``constant`` gives, per group, whether
    the ORIGINAL values are constant, and ``noise`` the size of one unit in the last
    place of the original values in rescaled units (group-mean differences of a few
    ``noise`` are rounding error).
    """
    if constant is None:
        constant = [ph._near_constant(a) for a in arrays]
    n_total = sum(a.size for a in arrays)
    pooled = np.concatenate(arrays)
    grand = math.fsum(pooled) / n_total
    grand += math.fsum(pooled - grand) / n_total
    deviations: list[float] = []  # group mean minus grand mean
    residuals: list[np.ndarray] = []
    for a, flat in zip(arrays, constant):
        if flat:
            deviations.append(float(a[0]) - grand)
            residuals.append(np.zeros(a.size))
            continue
        centred = a - grand
        dev = math.fsum(centred) / a.size
        res = centred - dev
        shift = math.fsum(res) / a.size
        deviations.append(dev + shift)
        residuals.append(res - shift)
    # re-centre the group deviations on their weighted mean (rounding-level correction)
    offset = math.fsum(a.size * d for a, d in zip(arrays, deviations)) / n_total
    deviations = [d - offset for d in deviations]
    if noise is None:
        scale = float(np.max(np.abs(pooled))) if n_total else 0.0
        tol = 8.0 * np.finfo(float).eps * scale
    else:
        tol = 16.0 * noise
    if all(abs(d) <= tol for d in deviations):
        deviations = [0.0] * len(arrays)
    ss_between = math.fsum(a.size * d * d for a, d in zip(arrays, deviations))
    ss_within = math.fsum(np.concatenate(residuals) ** 2)
    return {
        "grand_mean": grand + offset,
        "deviations": deviations,
        "residuals": residuals,
        "ss_between": ss_between,
        "ss_within": ss_within,
        "ss_total": ss_between + ss_within,
    }


def _scaled_ss(arrays: Sequence[np.ndarray]) -> tuple[Rescaled, dict[str, Any]]:
    """Jointly rescale the groups and compute the sums of squares in rescaled units.

    Everything returned in the dict is in rescaled units; map quantities back with the
    returned :class:`~torusplot.stats._utils.Rescaled` (``square`` for sums of squares
    and mean squares, ``spread`` for deviations / residuals, ``location`` for means)."""
    rs = rescale(*arrays)
    ss = _sums_of_squares(rs.arrays, [ph._near_constant(a) for a in arrays], rs.noise)
    return rs, ss


def _sd(a: np.ndarray) -> float:
    """Sample SD without overflow or underflow (power-of-two scaling is exact)."""
    if ph._near_constant(a):
        return 0.0
    scaled, e = pow2_scaled(a)
    return pow2_back(float(np.std(scaled, ddof=1)), e)


def _finite_or_none(value: float | None) -> float | None:
    return value if value is not None and math.isfinite(value) else None


def _descriptives(cleaned: Mapping[str, np.ndarray], level: float = 0.95) -> list[dict[str, Any]]:
    rows = []
    for name, a in cleaned.items():
        n = a.size
        mean = ph._mean(a)
        sd = _sd(a) if n > 1 else None
        sem = sd / math.sqrt(n) if sd is not None else None
        half = float(stats.t.ppf(0.5 + level / 2.0, n - 1)) * sem if sem is not None else None
        with np.errstate(over="ignore"):
            low = float(np.float64(mean) - half) if half is not None else None
            high = float(np.float64(mean) + half) if half is not None else None
        rows.append({
            "group": name, "n": int(n), "mean": mean, "sd": sd, "sem": sem,
            "median": float(np.median(a)), "min": float(np.min(a)), "max": float(np.max(a)),
            # a limit beyond the largest representable number is reported as missing
            "ci_low": _finite_or_none(low), "ci_high": _finite_or_none(high),
        })
    return rows


_UNREPRESENTABLE = (
    "Some quantities (sums of squares, mean squares, variances or confidence limits) lie outside "
    "the range of numbers a computer can represent (about 1e-308 to 1.8e308) and are shown as "
    "missing. The test statistic, P value and effect sizes are unaffected; express the data in "
    "another unit (e.g. thousands or millionths) to see them."
)


def _mean_parameters(rows: list[dict[str, Any]]) -> list[Parameter]:
    return [
        Parameter(name=f"Mean of {r['group']}", value=r["mean"], se=r["sem"],
                  ci_low=r["ci_low"], ci_high=r["ci_high"])
        for r in rows
    ]


def _median_parameters(rows: list[dict[str, Any]]) -> list[Parameter]:
    return [Parameter(name=f"Median of {r['group']}", value=r["median"]) for r in rows]


def _shapiro(values: np.ndarray) -> tuple[float, float] | None:
    """Shapiro-Wilk (W, P), or None when it cannot be computed."""
    if values.size < 3 or values.size > _SHAPIRO_MAX_N or ph._near_constant(values):
        return None
    values = rescale(values).arrays[0]  # W does not depend on origin or unit
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        w, p = stats.shapiro(values)
    if not (math.isfinite(w) and math.isfinite(p)):
        return None
    return float(w), float(p)


def _normality_checks(cleaned: Mapping[str, np.ndarray], residuals: np.ndarray,
                      alternative_test: str) -> list[AssumptionCheck]:
    """Shapiro-Wilk on the pooled residuals (primary) and on each group with n >= 3."""
    checks: list[AssumptionCheck] = []
    name = "Normality of residuals (Shapiro-Wilk)"
    res = _shapiro(residuals)
    if res is None:
        if residuals.size > _SHAPIRO_MAX_N:
            detail = (f"Not tested: with more than {_SHAPIRO_MAX_N} values the Shapiro-Wilk P "
                      "value is unreliable (and trivial departures become significant). Inspect "
                      "a QQ plot of the residuals instead.")
        elif residuals.size < 3:
            detail = "Too few values to test normality."
        else:
            detail = "Cannot be tested: the residuals have no variability."
        checks.append(AssumptionCheck(name=name, target="residuals", status="warn", detail=detail))
    else:
        w, p = res
        if p < 0.05:
            detail = (f"The residuals deviate from a normal distribution ({p_phrase(p)}). "
                      f"Consider {alternative_test}, or transforming the data.")
            status = "fail"
        else:
            detail = f"No evidence against normally distributed residuals ({p_phrase(p)})."
            status = "pass"
        checks.append(AssumptionCheck(name=name, target="residuals", statistic=w, p_value=p,
                                      status=status, detail=detail))
    for group, a in cleaned.items():
        name = "Normality within group (Shapiro-Wilk)"
        res = _shapiro(a)
        if res is None:
            if a.size < 3:
                detail = f"Only {a.size} value(s): too few to test normality."
            elif a.size > _SHAPIRO_MAX_N:
                detail = "Not tested (too many values for a reliable Shapiro-Wilk P value)."
            else:
                detail = "Cannot be tested: all values in this group are identical."
            checks.append(AssumptionCheck(name=name, target=group, status="warn", detail=detail))
            continue
        w, p = res
        if p < 0.05:
            checks.append(AssumptionCheck(
                name=name, target=group, statistic=w, p_value=p, status="warn",
                detail=f"This group deviates from a normal distribution ({p_phrase(p)}); "
                       "judge it together with the residual check above."))
        else:
            checks.append(AssumptionCheck(
                name=name, target=group, statistic=w, p_value=p, status="pass",
                detail=f"No evidence against normality ({p_phrase(p)})."))
    return checks


def _brown_forsythe_variance(arrays: Sequence[np.ndarray]) -> tuple[float, float] | None:
    """Brown-Forsythe (median-centred Levene) test of equal variances, or None.

    Uses :func:`torusplot.stats.assumptions.brown_forsythe_test` (computed on rescaled
    data, so large offsets and extreme magnitudes are handled)."""
    if any(a.size < 2 for a in arrays) or all(ph._near_constant(a) for a in arrays):
        return None
    res = am.brown_forsythe_test(list(arrays))
    if res["statistic"] is None or res["p_value"] is None:
        return None
    return float(res["statistic"]), float(res["p_value"])


def _variance_untestable_reason(arrays: Sequence[np.ndarray], any_constant: bool) -> str:
    """Why an equal-variance test could not be run, in words that match the data."""
    small = any(a.size < 2 for a in arrays)
    if small:
        return "Cannot be tested: at least one group has fewer than 2 values."
    if any_constant and any(ph._near_constant(a) for a in arrays):
        return "Cannot be tested: at least one group has no variability (all its values are identical)."
    if all(ph._near_constant(a) for a in arrays):
        return "Cannot be tested: no group shows any variability."
    return ("Cannot be tested: within every group the values are equally far from the group "
            "median (this happens when groups have only 2 values), so the test is undefined.")


def _bartlett(arrays: Sequence[np.ndarray]) -> tuple[float, float] | None:
    """Bartlett's test via :func:`torusplot.stats.assumptions.bartlett_test` (exactly 0,
    P = 1, when the variances are equal), or None when a group has n < 2 or no spread."""
    if any(a.size < 2 for a in arrays) or any(ph._near_constant(a) for a in arrays):
        return None
    res = am.bartlett_test(list(arrays))
    if res["statistic"] is None or res["p_value"] is None:
        return None
    return float(res["statistic"]), float(res["p_value"])


def _variance_checks(arrays: Sequence[np.ndarray], *, required: bool) -> list[AssumptionCheck]:
    """Homogeneity of variance. ``required=False`` reports it for information only."""
    checks: list[AssumptionCheck] = []
    bf = _brown_forsythe_variance(arrays)
    name = "Equal variances (Brown-Forsythe)"
    if bf is None:
        checks.append(AssumptionCheck(
            name=name, status="warn" if required else "info",
            detail=_variance_untestable_reason(arrays, any_constant=False)))
    else:
        stat, p = bf
        if not required:
            verdict = "differ" if p < 0.05 else "do not differ detectably"
            checks.append(AssumptionCheck(
                name=name, statistic=stat, p_value=p, status="info",
                detail=f"Not required by this test. For information: the group variances "
                       f"{verdict} ({p_phrase(p)})."))
        elif p < 0.05:
            checks.append(AssumptionCheck(
                name=name, statistic=stat, p_value=p, status="fail",
                detail=f"The groups have unequal spread ({p_phrase(p)}). Use Welch's ANOVA "
                       "with Games-Howell comparisons instead."))
        else:
            checks.append(AssumptionCheck(
                name=name, statistic=stat, p_value=p, status="pass",
                detail=f"No evidence that the group variances differ ({p_phrase(p)})."))
    if required:
        name = "Equal variances (Bartlett)"
        bart = _bartlett(arrays)
        if bart is None:
            checks.append(AssumptionCheck(
                name=name, status="warn",
                detail=_variance_untestable_reason(arrays, any_constant=True)))
        else:
            stat, p = bart
            if p < 0.05:
                checks.append(AssumptionCheck(
                    name=name, statistic=stat, p_value=p, status="warn",
                    detail=f"Bartlett's test suggests unequal variances ({p_phrase(p)}). It is "
                           "very sensitive to non-normal data, so rely on the Brown-Forsythe "
                           "check when the two disagree."))
            else:
                checks.append(AssumptionCheck(
                    name=name, statistic=stat, p_value=p, status="pass",
                    detail=f"No evidence that the group variances differ ({p_phrase(p)})."))
    return checks


def _independence_check(related: bool = False) -> AssumptionCheck:
    if related:
        return AssumptionCheck(
            name="Independent subjects", status="info",
            detail="Each row must be a different subject (or matched set), measured once under "
                   "every condition. This comes from the experimental design and cannot be "
                   "tested from the data.")
    return AssumptionCheck(
        name="Independent observations", status="info",
        detail="Every value must come from a different subject or independent sample. This "
               "comes from the experimental design and cannot be tested from the data.")


def _none_if_nan(effect: EffectSize) -> EffectSize:
    for field in ("value", "ci_low", "ci_high"):
        v = getattr(effect, field)
        if v is not None and math.isnan(v):
            setattr(effect, field, None)
    if effect.value is None:
        effect.magnitude = None
    if effect.ci_low is None or effect.ci_high is None:
        effect.ci_low = effect.ci_high = None
        effect.ci_method = None
    return effect


def _posthoc_key(posthoc: str | None, allowed: Sequence[str], test: str) -> str:
    key = "none" if posthoc is None else str(posthoc).strip().lower().replace("_", "-")
    key = {"tukey-hsd": "tukey", "fdr-bh": "fdr_bh", "gameshowell": "games-howell",
           "off": "none", "": "none"}.get(key, key)
    if key not in allowed:
        raise StatsError(
            f"Post-hoc method '{posthoc}' is not available after {test}. "
            f"Choose one of: {', '.join(allowed)}."
        )
    return key


def _mean_posthoc(
    cleaned: Mapping[str, np.ndarray], key: str, control: str | None, alpha: float, *,
    pooled: bool, mse: float | None = None, df_error: float | None = None,
    mse_in_pow2_units: float | None = None,
) -> tuple[ph.PosthocResult | None, list[str]]:
    """Run the chosen mean-comparison procedure; problems become warnings, not errors."""
    notes: list[str] = []
    names = list(cleaned)
    if control is not None and str(control) not in names:
        raise StatsError(
            f"The control group '{control}' is not one of the groups ({', '.join(names)})."
        )
    if key == "none":
        return None, notes
    if key == "dunnett" and control is None:
        control = names[0]
        notes.append(f"No control group was chosen for Dunnett's test, so the first group "
                     f"('{control}') was used as the control.")
    if key in {"tukey", "games-howell"} and control is not None:
        notes.append("The control group setting is ignored: this post-hoc method compares all "
                     "pairs of groups. Choose Dunnett (or a P value adjustment such as Holm) to "
                     "compare each group with the control only.")
        control = None
    # Mean comparisons are location-scale equivariant: run them on the data divided by a
    # common power of two (exact; no overflow or underflow of squares) and scale the
    # estimates, SEs and CI limits back. ``mse`` must then be given in those units too.
    scaled, e = pow2_scaled(np.concatenate(list(cleaned.values())))
    bounds = np.cumsum([0] + [a.size for a in cleaned.values()])
    scaled_groups = {name: scaled[bounds[i]:bounds[i + 1]] for i, name in enumerate(names)}
    mse_scaled = mse_in_pow2_units if mse_in_pow2_units is not None else (
        None if mse is None else float(np.ldexp(mse, -2 * e)))
    use_mse = key != "games-howell" and mse_scaled is not None and df_error is not None
    try:
        result = ph.run_posthoc(
            scaled_groups, key, control=control, alpha=alpha, pooled=pooled,
            mse=mse_scaled if use_mse else None, df_error=df_error if use_mse else None,
        )
        fields = ("estimate", "se", "ci_low", "ci_high")
        comparisons = [
            c.model_copy(update={f: None if getattr(c, f) is None else pow2_back(getattr(c, f), e)
                                 for f in fields})
            for c in result.comparisons
        ]
    except StatsError as exc:
        notes.append(f"Post-hoc comparisons could not be computed: {exc}")
        return None, notes
    result = ph.PosthocResult(comparisons, result.family, result.correction,
                              warnings=result.warnings, citations=result.citations,
                              method=result.method)
    notes.extend(result.warnings)
    if key == "lsd" and len(names) > 3:
        notes.append("Unadjusted (LSD-type) comparisons do not control the family-wise error "
                     "rate with more than 3 groups; prefer an adjusted method unless the "
                     "comparisons were planned in advance.")
    return result, notes


def _comparison_sentence(comparisons: Sequence[Comparison], correction: str | None,
                         alpha: float) -> str:
    """One sentence saying which pairs differ."""
    if not comparisons:
        return ""
    label = f"Post-hoc comparisons ({correction})" if correction else "Post-hoc comparisons"
    adjusted = "" if correction is None or "unadjusted" in correction else "adjusted "
    usable = [c for c in comparisons if c.p_for_display is not None]
    hits = [c for c in usable if c.p_for_display < alpha]
    total = len(comparisons)
    if not usable:
        return f"{label} could not be evaluated."
    if not hits:
        return (f"{label} found no pair that differed significantly "
                f"(all {total} {adjusted}P values >= {alpha:g}).")
    shown = ", ".join(
        f"{c.group_a} vs {c.group_b} ({adjusted}{p_phrase(c.p_for_display)})" for c in hits[:6])
    more = f", and {len(hits) - 6} more" if len(hits) > 6 else ""
    return (f"{label} found evidence of a difference for {len(hits)} of {total} "
            f"comparison(s): {shown}{more}.")


def _attach_posthoc(result: StatResult, post: ph.PosthocResult | None) -> None:
    if post is None:
        return
    result.comparisons = list(post.comparisons)
    result.comparison_family = post.family
    result.correction = post.correction
    for cite in post.citations:
        if cite not in result.citations:
            result.citations.append(cite)
    result.extra["posthoc_method"] = post.method


def _magnitude_phrase(effect: EffectSize) -> str:
    return f" (a {effect.magnitude} effect by Cohen's benchmarks)" if effect.magnitude else ""


def _f_and_p(numerator: float, denominator: float, df1: float, df2: float) -> tuple[float | None, float | None, str | None]:
    """F ratio with honest handling of a zero denominator."""
    if denominator > 0:
        f = numerator / denominator
        return f, float(stats.f.sf(f, df1, df2)), None
    if numerator > 0:
        return math.inf, 0.0, (
            "Every group's values are identical within the group, so the within-group "
            "variability is zero. The F ratio is infinite and P is reported as 0; this "
            "usually means the data were rounded or duplicated, and the test's model (random "
            "normal scatter within groups) does not describe them.")
    return None, None, (
        "All values are identical, so there is no variability to analyse; the test statistic "
        "and P value are undefined.")


def _group_list(names: Sequence[str]) -> str:
    return ", ".join(names) if len(names) <= 8 else ", ".join(names[:8]) + ", ..."


# ================================================================== one-way ANOVA


def anova_oneway(
    groups: Mapping[str, Any],
    *,
    posthoc: str | None = "tukey",
    control: str | None = None,
    alpha: float = 0.05,
) -> StatResult:
    """Classic fixed-effects one-way analysis of variance (Fisher 1925).

    ``F = MS_between / MS_within`` with ``k - 1`` and ``N - k`` degrees of freedom. The
    sums of squares are computed on centred data with exactly-rounded summation, so
    large offsets with tiny variances do not destroy the result (see NIST StRD tests).

    ``posthoc``: ``tukey`` (default), ``dunnett`` (each group vs ``control``; first group
    if none is given), ``bonferroni`` / ``sidak`` / ``holm`` / ``holm-sidak`` / ``fdr_bh``
    / ``lsd`` (pooled-SD t tests; all pairs, or each group vs ``control`` when given),
    ``games-howell``, or ``none`` (no post-hoc comparisons).
    """
    alpha = check_alpha(alpha)
    key = _posthoc_key(posthoc, _EQUAL_VAR_POSTHOCS, "a one-way ANOVA")
    cleaned, notes = _prepare(groups, "One-way ANOVA")
    names = list(cleaned)
    arrays = list(cleaned.values())
    k = len(arrays)
    n_total = sum(a.size for a in arrays)
    df1, df2 = k - 1, n_total - k
    if df2 < 1:
        raise StatsError(
            "One-way ANOVA needs replicates: every group has a single value, so the "
            "within-group variability cannot be estimated. Add more values per group."
        )
    # F, P and effect sizes from rescaled data (scale invariant); SS / MS mapped back
    rs, ss = _scaled_ss(arrays)
    ssb_z, ssw_z, sst_z = ss["ss_between"], ss["ss_within"], ss["ss_total"]
    msb_z, msw_z = ssb_z / df1, ssw_z / df2
    f, p, degenerate = _f_and_p(msb_z, msw_z, df1, df2)
    ssb, ssw, sst = rs.square(ssb_z), rs.square(ssw_z), rs.square(sst_z)
    msb, msw = rs.square(msb_z), rs.square(msw_z)

    warn: list[str] = []
    if degenerate:
        warn.append(degenerate)
    singles = [name for name, a in cleaned.items() if a.size == 1]
    if singles:
        warn.append(
            f"Group(s) with a single value ({_group_list(singles)}): their variability cannot "
            "be checked and the result leans entirely on the equal-variance assumption.")
    constant = [name for name, a in cleaned.items() if a.size > 1 and ph._near_constant(a)]
    if constant and not degenerate:
        warn.append(f"All values are identical in group(s) {_group_list(constant)}; the "
                    "equal-variance assumption is doubtful.")
    warn.extend(_two_group_note(
        k, "With two groups the one-way ANOVA is equivalent to the unpaired t test "
           "(F = t squared), which is the usual choice."))

    # ratios of sums of squares: identical in rescaled units
    if f is not None and math.isfinite(f):
        eta = es.eta_squared(ssb_z, sst_z, f, df1, df2)
        omega = es.omega_squared(ssb_z, df1, msw_z, sst_z, f, df2)
    else:
        eta = es.eta_squared(ssb_z, sst_z) if sst_z > 0 else EffectSize(name="eta squared", value=None)
        omega = (es.omega_squared(ssb_z, df1, msw_z, sst_z) if sst_z > 0
                 else EffectSize(name="omega squared", value=None))
    effects = [_none_if_nan(eta), _none_if_nan(omega)]

    residuals_z = np.concatenate(ss["residuals"])
    assumptions = [_independence_check()]
    assumptions += _normality_checks(cleaned, residuals_z, "the Kruskal-Wallis test")
    assumptions += _variance_checks(arrays, required=True)

    # the error mean square in the units of the data divided by 2**e1 (see _mean_posthoc)
    _, e2 = rs.exponents
    post, post_notes = _mean_posthoc(cleaned, key, control, alpha, pooled=True,
                                     df_error=float(df2),
                                     mse_in_pow2_units=float(np.ldexp(msw_z, 2 * e2)))
    warn.extend(post_notes)

    desc = _descriptives(cleaned)
    grand_mean = rs.location(ss["grand_mean"])
    root_mse = rs.spread(math.sqrt(msw_z))
    if any(v is None for v in (ssb, ssw, sst, msb, msw)) or any(
            r["ci_low"] is None and r["sem"] is not None for r in desc):
        warn.append(_UNREPRESENTABLE)
    result = StatResult(
        analysis="anova_oneway",
        test_name="One-way ANOVA",
        why_chosen=(
            "One-way ANOVA compares the means of three or more independent groups defined by "
            "a single factor, when the values in each group are roughly normally distributed "
            "with similar spread."),
        assumptions=assumptions,
        statistic_name="F", statistic=f, df=df1, df2=df2, p_value=p, alpha=alpha,
        effect_sizes=effects,
        parameters=_mean_parameters(desc),
        n={name: int(a.size) for name, a in cleaned.items()},
        descriptives=desc,
        anova_table=[
            {"source": "Between groups", "ss": ssb, "df": df1, "ms": msb, "f": f, "p": p},
            {"source": "Within groups (residual)", "ss": ssw, "df": df2, "ms": msw, "f": None, "p": None},
            {"source": "Total", "ss": sst, "df": n_total - 1, "ms": None, "f": None, "p": None},
        ],
        warnings=warn, data_notes=notes,
        citations=[CITE_FISHER],
        extra={
            "grand_mean": grand_mean,
            "r_squared": ssb_z / sst_z if sst_z > 0 else None,
            "root_mse": root_mse,
            "residuals": {
                "group": [name for name, a in cleaned.items() for _ in range(a.size)],
                "fitted": [rs.location(ss["grand_mean"] + d) for a, d in zip(arrays, ss["deviations"])
                           for _ in range(a.size)],
                "residual": rs.spread_array(residuals_z).tolist(),
            },
        },
    )
    _attach_posthoc(result, post)

    sentences = [f"A one-way ANOVA compared the means of {k} groups ({_group_list(names)}; "
                 f"N = {n_total})."]
    if f is None:
        sentences.append("All values are identical, so no test could be carried out.")
    else:
        sentences.append(
            f"The difference among the group means was {sig_phrase(p, alpha)}, "
            f"F({df1}, {df2}) = {fmt(f)}"
            + (", which is evidence that at least one group mean differs from the others."
               if p is not None and p < alpha else
               ", so the data give no convincing evidence that the group means differ."))
    if eta.value is not None:
        ci = (f", {ci_phrase(eta.ci_low, eta.ci_high)}" if eta.ci_low is not None else "")
        sentences.append(
            f"Group membership accounts for about {eta.value * 100:.1f}% of the variability "
            f"(eta squared = {fmt(eta.value)}{ci}){_magnitude_phrase(eta)}.")
    sentence = _comparison_sentence(result.comparisons, result.correction, alpha)
    if sentence:
        sentences.append(sentence)
    result.interpretation = " ".join(sentences)
    result.methods_text = (
        f"Group means were compared with a one-way ANOVA (alpha = {alpha:g})"
        + (f", followed by {post.method} ({post.family})." if post is not None else ".")
    )
    return result.finalize()


# ================================================================== Welch's ANOVA


def _heteroscedastic_common(
    groups: Mapping[str, Any], what: str, posthoc: str | None, control: str | None, alpha: float,
) -> dict[str, Any]:
    """Shared preparation for the Welch and Brown-Forsythe tests of means."""
    alpha = check_alpha(alpha)
    key = _posthoc_key(posthoc, _UNEQUAL_VAR_POSTHOCS, what)
    cleaned, notes = _prepare(
        groups, what, min_each=2,
        why_min=" because it estimates a separate variance for each group")
    arrays = list(cleaned.values())
    # statistics from rescaled data (scale invariant); "_z" quantities are in rescaled units
    rs, ss = _scaled_ss(arrays)
    n = np.array([a.size for a in arrays], dtype=float)
    variances_z = np.array([0.0 if ph._near_constant(a) else float(np.var(z, ddof=1))
                            for a, z in zip(arrays, rs.arrays)])
    variances = [rs.square(v) for v in variances_z]  # data units (None if unrepresentable)
    residuals = np.concatenate(ss["residuals"])
    assumptions = [_independence_check()]
    assumptions += _normality_checks(cleaned, residuals, "the Kruskal-Wallis test")
    assumptions += _variance_checks(arrays, required=False)
    post, post_notes = _mean_posthoc(cleaned, key, control, alpha, pooled=False)
    desc = _descriptives(cleaned)
    eta = EffectSize(
        name="eta squared (descriptive)",
        value=ss["ss_between"] / ss["ss_total"] if ss["ss_total"] > 0 else None,
    )
    eta.magnitude = es.magnitude_eta2(eta.value)
    if any(v is None for v in variances) or any(
            r["ci_low"] is None and r["sem"] is not None for r in desc):
        post_notes = [*post_notes, _UNREPRESENTABLE]
    return {
        "alpha": alpha, "cleaned": cleaned, "notes": notes, "arrays": arrays, "ss": ss, "n": n,
        "variances_z": variances_z, "variances": variances, "assumptions": assumptions, "post": post,
        "post_notes": post_notes, "desc": desc, "eta": eta, "k": len(arrays),
        "n_total": int(n.sum()), "residuals": residuals, "rs": rs,
        "grand_mean": rs.location(ss["grand_mean"]),
    }


_DESCRIPTIVE_ETA_NOTE = (
    "The effect size is eta squared from the ordinary sums of squares (between / total). It "
    "describes this sample only: no confidence interval is given because the usual interval "
    "assumes equal variances."
)


def _heteroscedastic_text(result: StatResult, c: dict[str, Any], label: str, f: float | None,
                          df1: float, df2: float | None, p: float | None) -> None:
    alpha, k, names = c["alpha"], c["k"], list(c["cleaned"])
    sentences = [f"{label[0].upper()}{label[1:]} compared the means of {k} groups ({_group_list(names)}; "
                 f"N = {c['n_total']}) without assuming equal variances."]
    if f is None:
        sentences.append("The test statistic could not be computed (see the warnings).")
    else:
        df2_text = fmt(df2) if df2 is not None else "n/a"
        sentences.append(
            f"The difference among the group means was {sig_phrase(p, alpha)}, "
            f"F({fmt(df1)}, {df2_text}) = {fmt(f)}"
            + (", which is evidence that at least one group mean differs from the others."
               if p is not None and p < alpha else
               ", so the data give no convincing evidence that the group means differ."))
    eta = c["eta"]
    if eta.value is not None:
        sentences.append(
            f"Descriptively, group membership accounts for about {eta.value * 100:.1f}% of the "
            f"variability in this sample (eta squared = {fmt(eta.value)}).")
    sentence = _comparison_sentence(result.comparisons, result.correction, alpha)
    if sentence:
        sentences.append(sentence)
    result.interpretation = " ".join(sentences)
    post = c["post"]
    result.methods_text = (
        f"Group means were compared with {label} (alpha = {alpha:g})"
        + (f", followed by {post.method} ({post.family})." if post is not None else "."))


def welch_anova(
    groups: Mapping[str, Any],
    *,
    posthoc: str | None = "games-howell",
    control: str | None = None,
    alpha: float = 0.05,
) -> StatResult:
    """Welch's (1951) one-way ANOVA for groups with unequal variances.

    With weights ``w_i = n_i / s_i^2``, ``W = sum w_i`` and the weighted grand mean
    ``xw = sum w_i xbar_i / W``::

        F = [sum w_i (xbar_i - xw)^2 / (k - 1)] / [1 + 2 (k - 2) / (k^2 - 1) * L]
        L = sum (1 - w_i / W)^2 / (n_i - 1),   df1 = k - 1,   df2 = (k^2 - 1) / (3 L)

    Every group needs at least 2 values and a non-zero variance. ``posthoc``:
    ``games-howell`` (default), a P value adjustment of pairwise Welch t tests
    (``bonferroni``, ``sidak``, ``holm``, ``holm-sidak``, ``fdr_bh``, ``lsd``; each group
    vs ``control`` when one is given) or ``none``.
    """
    c = _heteroscedastic_common(groups, "Welch's ANOVA", posthoc, control, alpha)
    k, n, variances = c["k"], c["n"], c["variances_z"]
    warn: list[str] = []
    df1 = float(k - 1)
    f: float | None = None
    df2: float | None = None
    p: float | None = None
    if np.any(variances <= 0):
        flat = [name for name, v in zip(c["cleaned"], variances) if v <= 0]
        warn.append(
            f"All values are identical in group(s) {_group_list(flat)}. Welch's ANOVA weights "
            "each group by 1 / variance, so it cannot be computed when a group has zero "
            "variance; the statistic and P value are undefined.")
    else:
        w = n / variances
        w_total = float(w.sum())
        dev = np.array(c["ss"]["deviations"])  # group means relative to the grand mean
        centre = float(np.sum(w * dev) / w_total)
        numerator = float(np.sum(w * (dev - centre) ** 2)) / df1
        lam = float(np.sum((1.0 - w / w_total) ** 2 / (n - 1.0)))
        f = numerator / (1.0 + 2.0 * (k - 2.0) / (k * k - 1.0) * lam)
        df2 = (k * k - 1.0) / (3.0 * lam)
        p = float(stats.f.sf(f, df1, df2))
    warn.extend(_two_group_note(
        k, "With two groups Welch's ANOVA is equivalent to Welch's unpaired t test "
           "(F = t squared), which is the usual choice."))
    warn.extend(c["post_notes"])
    warn.append(_DESCRIPTIVE_ETA_NOTE)
    result = StatResult(
        analysis="welch_anova",
        test_name="Welch's ANOVA",
        why_chosen=(
            "Welch's ANOVA compares the means of three or more independent groups without "
            "assuming that the groups have the same variance, so it is the safer choice when "
            "the spread or the sample size differs between groups."),
        assumptions=c["assumptions"],
        statistic_name="F", statistic=f, df=df1, df2=df2, p_value=p, alpha=c["alpha"],
        effect_sizes=[c["eta"]],
        parameters=_mean_parameters(c["desc"]),
        n={name: int(a.size) for name, a in c["cleaned"].items()},
        descriptives=c["desc"],
        anova_table=[
            {"source": "Between groups (Welch)", "ss": None, "df": df1, "ms": None, "f": f, "p": p},
            {"source": "Within groups (Welch-adjusted df)", "ss": None, "df": df2, "ms": None,
             "f": None, "p": None},
        ],
        warnings=warn, data_notes=c["notes"], citations=[CITE_WELCH],
        extra={"grand_mean": c["grand_mean"],
               "group_variances": dict(zip(c["cleaned"], c["variances"]))},
    )
    _attach_posthoc(result, c["post"])
    _heteroscedastic_text(result, c, "Welch's ANOVA", f, df1, df2, p)
    return result.finalize()


# ======================================================== Brown-Forsythe F* test


def brown_forsythe_anova(
    groups: Mapping[str, Any],
    *,
    posthoc: str | None = "games-howell",
    control: str | None = None,
    alpha: float = 0.05,
) -> StatResult:
    """Brown-Forsythe (1974) test of equal means under unequal variances.

    ``F* = SS_between / sum (1 - n_i / N) s_i^2`` with ``df1 = k - 1`` and the
    Satterthwaite ``df2 = 1 / sum c_i^2 / (n_i - 1)``,
    ``c_i = (1 - n_i/N) s_i^2 / sum (1 - n_j/N) s_j^2``. (This is the test of *means*,
    not the Brown-Forsythe test of equal variances.) Mehrotra's (1997) corrected
    numerator df and the corresponding P value are given in ``extra['mehrotra']``.

    Every group needs at least 2 values. ``posthoc`` as in :func:`welch_anova`.
    """
    c = _heteroscedastic_common(groups, "The Brown-Forsythe ANOVA", posthoc, control, alpha)
    k, n, variances, n_total = c["k"], c["n"], c["variances_z"], c["n_total"]
    rs = c["rs"]
    ssb = c["ss"]["ss_between"]  # rescaled units: F*, df and P are scale invariant
    warn: list[str] = []
    df1 = float(k - 1)
    terms = (1.0 - n / n_total) * variances
    denom = float(terms.sum())
    df2: float | None = None
    extra: dict[str, Any] = {"grand_mean": c["grand_mean"],
                             "group_variances": dict(zip(c["cleaned"], c["variances"]))}
    if denom > 0:
        f: float | None = ssb / denom
        df2 = denom ** 2 / float(np.sum(terms ** 2 / (n - 1.0)))
        p: float | None = float(stats.f.sf(f, df1, df2))
        df1_m = denom ** 2 / float(
            np.sum(variances ** 2) + np.sum(n / n_total * variances) ** 2
            - 2.0 * np.sum(n / n_total * variances ** 2))
        extra["mehrotra"] = {"df1": df1_m, "df2": df2, "p": float(stats.f.sf(f, df1_m, df2))}
    else:
        f, p, message = _f_and_p(ssb, 0.0, df1, 1.0)
        if message:
            warn.append(message)
    flat = [name for name, v in zip(c["cleaned"], variances) if v <= 0]
    if flat and denom > 0:
        warn.append(f"All values are identical in group(s) {_group_list(flat)}; they "
                    "contribute no variance to the test's denominator.")
    warn.extend(_two_group_note(
        k, "With two groups Welch's unpaired t test is the usual choice."))
    warn.extend(c["post_notes"])
    warn.append(_DESCRIPTIVE_ETA_NOTE)
    result = StatResult(
        analysis="brown_forsythe_anova",
        test_name="Brown-Forsythe ANOVA",
        why_chosen=(
            "The Brown-Forsythe ANOVA compares the means of three or more independent groups "
            "without assuming equal variances. It is an alternative to Welch's ANOVA that "
            "keeps the ordinary between-group sum of squares."),
        assumptions=c["assumptions"],
        statistic_name="F*", statistic=f, df=df1, df2=df2, p_value=p, alpha=c["alpha"],
        effect_sizes=[c["eta"]],
        parameters=_mean_parameters(c["desc"]),
        n={name: int(a.size) for name, a in c["cleaned"].items()},
        descriptives=c["desc"],
        anova_table=[
            {"source": "Between groups", "ss": rs.square(ssb), "df": df1, "ms": rs.square(ssb / df1),
             "f": f, "p": p},
            {"source": "Within groups (Brown-Forsythe denominator)", "ss": None, "df": df2,
             "ms": rs.square(denom / df1), "f": None, "p": None},
        ],
        warnings=warn, data_notes=c["notes"], citations=[CITE_BF], extra=extra,
    )
    _attach_posthoc(result, c["post"])
    _heteroscedastic_text(result, c, "a Brown-Forsythe ANOVA", f, df1, df2, p)
    return result.finalize()


# ================================================================ Kruskal-Wallis


def _rank_posthoc_key(posthoc: str | None, allowed: Sequence[str], test: str) -> str:
    key = "none" if posthoc is None else str(posthoc).strip().lower()
    key = {"off": "none", "": "none"}.get(key, key)
    if key not in allowed:
        raise StatsError(f"Post-hoc method '{posthoc}' is not available after {test}. "
                         f"Choose one of: {', '.join(allowed)}.")
    return key


def kruskal_wallis(
    groups: Mapping[str, Any],
    *,
    posthoc: str | None = "dunn",
    adjust: str = "bonferroni",
    control: str | None = None,
    alpha: float = 0.05,
) -> StatResult:
    """Kruskal-Wallis (1952) rank test for three or more independent groups.

    All values are ranked together (average ranks for ties) and

        H = [12 / (N (N + 1))] sum n_i (Rbar_i - (N + 1) / 2)^2 / C,
        C = 1 - sum (t^3 - t) / (N^3 - N)   (tie correction)

    is referred to the chi-square distribution with ``k - 1`` df (an approximation that
    is rough when every group has 5 or fewer values). ``posthoc='dunn'`` adds Dunn's
    comparisons with the ``adjust`` method (all pairs, or each group vs ``control``).
    """
    alpha = check_alpha(alpha)
    key = _rank_posthoc_key(posthoc, ("dunn", "none"), "the Kruskal-Wallis test")
    adjust_key = ph._normalise_adjust(adjust)
    cleaned, notes = _prepare(groups, "The Kruskal-Wallis test")
    names = list(cleaned)
    arrays = list(cleaned.values())
    if control is not None and str(control) not in names:
        raise StatsError(
            f"The control group '{control}' is not one of the groups ({', '.join(names)}).")
    k = len(arrays)
    n = np.array([a.size for a in arrays], dtype=float)
    n_total = int(n.sum())
    if n_total < 3:
        raise StatsError("The Kruskal-Wallis test needs at least 3 values in total.")
    ranks = stats.rankdata(np.concatenate(arrays))
    bounds = np.concatenate([[0], np.cumsum(n.astype(int))])
    rank_sums = np.array([ranks[bounds[g]:bounds[g + 1]].sum() for g in range(k)])
    mean_ranks = rank_sums / n
    _, counts = np.unique(ranks, return_counts=True)
    tie_sum = float(np.sum(counts.astype(float) ** 3 - counts))
    correction = 1.0 - tie_sum / (float(n_total) ** 3 - n_total)
    warn: list[str] = []
    df = k - 1
    if correction <= 1e-12:
        h: float | None = None
        p: float | None = None
        warn.append("All values are identical, so the ranks carry no information; the "
                    "Kruskal-Wallis statistic and P value are undefined.")
    else:
        h_raw = 12.0 / (n_total * (n_total + 1.0)) * float(
            np.sum(n * (mean_ranks - (n_total + 1.0) / 2.0) ** 2))
        h = h_raw / correction
        p = float(stats.chi2.sf(h, df))
    if np.all(n <= 5):
        warn.append(
            "Every group has 5 or fewer values. The P value uses the chi-square approximation, "
            "which can be inaccurate for samples this small; treat a borderline P value with "
            "caution.")
    warn.extend(_two_group_note(
        k, "With two groups the Kruskal-Wallis test is equivalent to the Mann-Whitney test "
           "(normal approximation without continuity correction), which is the usual choice."))

    spread = _brown_forsythe_variance(arrays)
    if spread is None:
        shape = AssumptionCheck(
            name="Similar distribution shape and spread (Brown-Forsythe)", status="warn",
            detail="Cannot be checked (a group has fewer than 2 values or no group varies). "
                   "Only if the groups have similarly shaped distributions can a significant "
                   "result be read as a difference in medians.")
    elif spread[1] < 0.05:
        shape = AssumptionCheck(
            name="Similar distribution shape and spread (Brown-Forsythe)", statistic=spread[0],
            p_value=spread[1], status="warn",
            detail=f"The groups differ in spread ({p_phrase(spread[1])}). The test is still "
                   "valid as a test that values in some group tend to be larger, but it should "
                   "not be read as a pure comparison of medians.")
    else:
        shape = AssumptionCheck(
            name="Similar distribution shape and spread (Brown-Forsythe)", statistic=spread[0],
            p_value=spread[1], status="pass",
            detail=f"No evidence that the groups differ in spread ({p_phrase(spread[1])}), so a "
                   "significant result can reasonably be read as a shift in typical values.")
    n_tied = int(np.sum(counts[counts > 1]))
    assumptions = [
        _independence_check(),
        AssumptionCheck(
            name="Ordinal or continuous measurements", status="info",
            detail="The values only need to be rankable; no normal distribution is assumed."
                   + (f" {n_tied} of {n_total} values are tied; the tie correction was applied."
                      if n_tied else "")),
        shape,
    ]

    post: ph.PosthocResult | None = None
    if key == "dunn":
        post = ph.dunn(cleaned, adjust=adjust_key, control=control, alpha=alpha)
        warn.extend(post.warnings)

    desc = _descriptives(cleaned)
    for row, rs, mr in zip(desc, rank_sums, mean_ranks):
        row["rank_sum"] = float(rs)
        row["mean_rank"] = float(mr)
    effect = (es.epsilon_squared_kw(h, n_total, k) if h is not None
              else EffectSize(name="epsilon squared (rank)", value=None))
    result = StatResult(
        analysis="kruskal_wallis",
        test_name="Kruskal-Wallis test",
        why_chosen=(
            "The Kruskal-Wallis test compares three or more independent groups using ranks, so "
            "it does not need normally distributed data; it suits skewed data, outliers or "
            "ordinal scores."),
        assumptions=assumptions,
        statistic_name="H", statistic=h, df=df, p_value=p, alpha=alpha,
        effect_sizes=[_none_if_nan(effect)],
        parameters=_median_parameters(desc),
        n={name: int(a.size) for name, a in cleaned.items()},
        descriptives=desc,
        warnings=warn, data_notes=notes, citations=[CITE_KW],
        extra={"tie_correction": correction, "mean_ranks": dict(zip(names, mean_ranks.tolist())),
               "p_method": "chi-square approximation"},
    )
    _attach_posthoc(result, post)

    sentences = [f"A Kruskal-Wallis test compared {k} groups ({_group_list(names)}; "
                 f"N = {n_total}) using ranks."]
    if h is None:
        sentences.append("All values are identical, so no test could be carried out.")
    else:
        sentences.append(
            f"The difference among the groups was {sig_phrase(p, alpha)}, "
            f"H({df}) = {fmt(h)}"
            + (", which is evidence that values in at least one group tend to be higher or "
               "lower than in the others." if p is not None and p < alpha else
               ", so the data give no convincing evidence that the groups differ."))
        if effect.value is not None:
            sentences.append(f"Rank epsilon squared = {fmt(effect.value)}"
                             f"{_magnitude_phrase(effect)}.")
    sentence = _comparison_sentence(result.comparisons, result.correction, alpha)
    if sentence:
        sentences.append(sentence)
    result.interpretation = " ".join(sentences)
    result.methods_text = (
        f"Groups were compared with the Kruskal-Wallis test (alpha = {alpha:g})"
        + (f", followed by Dunn's multiple comparisons with {METHOD_LABELS[adjust_key]} "
           f"adjustment ({post.family})." if post is not None and adjust_key != "none"
           else ", followed by Dunn's multiple comparisons without adjustment."
           if post is not None else "."))
    return result.finalize()


# ====================================================================== Friedman


def _as_matrix(data: Any, names: Sequence[str] | None) -> tuple[np.ndarray, list[str]]:
    """Rows = subjects, columns = conditions, from a mapping or a 2-D array."""
    if isinstance(data, Mapping):
        labels = [str(name) for name in data]
        columns = [np.asarray(list(v) if not isinstance(v, np.ndarray) else v, dtype=float).ravel()
                   for v in data.values()]
        if len({c.size for c in columns}) > 1:
            sizes = ", ".join(f"{name}: {c.size}" for name, c in zip(labels, columns))
            raise StatsError(
                "The Friedman test needs one value per subject in every condition, but the "
                f"conditions have different numbers of rows ({sizes}).")
        if not columns:
            raise StatsError("No data given for the Friedman test.")
        return np.column_stack(columns) if columns[0].size else np.empty((0, len(columns))), labels
    matrix = np.asarray(data, dtype=float)
    if matrix.ndim != 2:
        raise StatsError("The Friedman test needs a table with rows = subjects and columns = "
                         "conditions.")
    labels = ([str(name) for name in names] if names is not None
              else [f"Condition {j + 1}" for j in range(matrix.shape[1])])
    if len(labels) != matrix.shape[1]:
        raise ValueError("names must have one entry per column.")
    return matrix, labels


def friedman_test(
    data: Any,
    *,
    names: Sequence[str] | None = None,
    posthoc: str | None = "dunn",
    adjust: str = "bonferroni",
    alpha: float = 0.05,
) -> StatResult:
    """Friedman (1937) test for three or more related samples (repeated measures).

    ``data`` is either a mapping ``{condition: values}`` (equal lengths, position =
    subject) or a 2-D array with rows = subjects and columns = conditions (``names``
    labels the columns). Subjects with a missing value in any condition are left out and
    reported. Values are ranked within each subject (average ranks for ties) and

        Q = 12 sum_j (R_j - n (k + 1) / 2)^2 / [n k (k + 1) - sum_rows sum (t^3 - t) / (k - 1)]

    is referred to the chi-square distribution with ``k - 1`` df. Kendall's W =
    ``Q / (n (k - 1))`` is the effect size. ``posthoc``: ``dunn`` (rank-sum comparisons),
    ``wilcoxon`` (paired signed-rank tests) or ``none``; ``adjust`` is the P value
    adjustment. The Iman-Davenport (1980) F version is given in ``extra``.
    """
    alpha = check_alpha(alpha)
    key = _rank_posthoc_key(posthoc, ("dunn", "wilcoxon", "none"), "the Friedman test")
    adjust_key = ph._normalise_adjust(adjust)
    matrix, labels = _as_matrix(data, names)
    k = matrix.shape[1]
    if k < 3:
        raise StatsError(
            "The Friedman test needs at least 3 conditions (columns). For 2 related "
            "conditions use the Wilcoxon matched-pairs signed-rank test or a paired t test.")
    notes: list[str] = []
    n_inf = int(np.isinf(matrix).sum())
    complete = np.all(np.isfinite(matrix), axis=1)
    blank = np.all(np.isnan(matrix), axis=1)
    dropped = int(np.sum(~complete & ~blank))
    if dropped:
        notes.append(
            f"{dropped} subject(s) (rows) with a missing value in at least one condition were "
            "left out: the Friedman test needs a value in every condition for each subject.")
    if n_inf:
        notes.append(f"{n_inf} infinite value(s) were treated as missing.")
    matrix = matrix[complete]
    n = matrix.shape[0]
    if n < 2:
        raise StatsError(
            f"The Friedman test needs at least 2 subjects (rows) with a value in every "
            f"condition; {n} complete row(s) were found.")
    ranks = np.apply_along_axis(stats.rankdata, 1, matrix)
    rank_sums = ranks.sum(axis=0)
    tie_sum = 0.0
    for row in ranks:
        _, counts = np.unique(row, return_counts=True)
        tie_sum += float(np.sum(counts.astype(float) ** 3 - counts))
    denom = n * k * (k + 1.0) - tie_sum / (k - 1.0)
    df = k - 1
    warn: list[str] = []
    if denom <= 1e-9:
        q: float | None = None
        p: float | None = None
        warn.append("Within every subject all conditions have the same value, so the ranks "
                    "carry no information; the Friedman statistic and P value are undefined.")
    else:
        q = 12.0 * float(np.sum((rank_sums - n * (k + 1.0) / 2.0) ** 2)) / denom
        p = float(stats.chi2.sf(q, df))
    if n < 10 and k < 6:
        warn.append(
            f"With only {n} subjects and {k} conditions the chi-square approximation used for "
            "the P value is rough; treat a borderline P value with caution.")
    extra: dict[str, Any] = {
        "rank_sums": dict(zip(labels, rank_sums.tolist())),
        "mean_ranks": dict(zip(labels, (rank_sums / n).tolist())),
        "n_subjects": int(n), "p_method": "chi-square approximation",
    }
    if q is not None:
        rest = n * (k - 1.0) - q
        if rest > 1e-12:
            f_id = (n - 1.0) * q / rest
            extra["iman_davenport"] = {
                "f": f_id, "df1": df, "df2": (k - 1) * (n - 1),
                "p": float(stats.f.sf(f_id, df, (k - 1) * (n - 1)))}
        else:
            extra["iman_davenport"] = {"f": math.inf, "df1": df, "df2": (k - 1) * (n - 1), "p": 0.0}

    post: ph.PosthocResult | None = None
    if key != "none":
        post = ph.friedman_posthoc(matrix, labels, adjust=adjust_key, alpha=alpha, method=key)
        warn.extend(post.warnings)

    desc = []
    for j, label in enumerate(labels):
        col = matrix[:, j]
        sd = math.sqrt(ph._variance(col))
        desc.append({
            "group": label, "n": int(n), "mean": ph._mean(col), "sd": sd, "sem": sd / math.sqrt(n),
            "median": float(np.median(col)), "min": float(np.min(col)), "max": float(np.max(col)),
            "rank_sum": float(rank_sums[j]), "mean_rank": float(rank_sums[j] / n),
        })
    n_tied_rows = int(sum(np.unique(row).size < k for row in matrix))
    assumptions = [
        _independence_check(related=True),
        AssumptionCheck(
            name="Complete blocks", status="pass",
            detail=f"{n} subject(s) have a value in every one of the {k} conditions"
                   + (f"; {dropped} incomplete row(s) were left out." if dropped else ".")),
        AssumptionCheck(
            name="Ordinal or continuous measurements", status="info",
            detail="Values only need to be rankable within each subject; no normal distribution "
                   "is assumed."
                   + (f" {n_tied_rows} subject(s) have tied values; the tie correction was applied."
                      if n_tied_rows else "")),
    ]
    effect = (es.kendall_w(q, n, k) if q is not None
              else EffectSize(name="Kendall's W", value=None))
    result = StatResult(
        analysis="friedman",
        test_name="Friedman test",
        why_chosen=(
            "The Friedman test compares three or more conditions measured on the same subjects "
            "(or matched sets) using ranks within each subject, so it does not need normally "
            "distributed data. It is the rank-based counterpart of repeated-measures ANOVA."),
        assumptions=assumptions,
        statistic_name="chi2", statistic=q, df=df, p_value=p, alpha=alpha,
        effect_sizes=[_none_if_nan(effect)],
        parameters=_median_parameters(desc),
        n={label: int(n) for label in labels},
        n_total=int(n),
        descriptives=desc,
        warnings=warn, data_notes=notes, citations=[CITE_FRIEDMAN, CITE_KENDALL],
        extra=extra,
    )
    _attach_posthoc(result, post)

    sentences = [f"A Friedman test compared {k} related conditions ({_group_list(labels)}) "
                 f"measured on {n} subjects."]
    if q is None:
        sentences.append("No subject shows any difference between conditions, so no test "
                         "could be carried out.")
    else:
        sentences.append(
            f"The difference among the conditions was {sig_phrase(p, alpha)}, "
            f"chi-square({df}) = {fmt(q)}"
            + (", which is evidence that at least one condition tends to give higher or lower "
               "values than the others." if p is not None and p < alpha else
               ", so the data give no convincing evidence that the conditions differ."))
        if effect.value is not None:
            sentences.append(
                f"Kendall's W = {fmt(effect.value)} (0 = no agreement between subjects on the "
                "ordering of conditions, 1 = complete agreement).")
    sentence = _comparison_sentence(result.comparisons, result.correction, alpha)
    if sentence:
        sentences.append(sentence)
    result.interpretation = " ".join(sentences)
    result.methods_text = (
        f"Related conditions were compared with the Friedman test (alpha = {alpha:g})"
        + (f", followed by {post.method} ({METHOD_LABELS[adjust_key]} adjustment; "
           f"{post.family})." if post is not None else "."))
    return result.finalize()


# ============================================================ registered adapters

_GROUPS_PARAM = ParamSpec(
    "groups", "Groups to compare", kind="groups", default=None, optional=True,
    help="Which columns (groups) to include. Leave empty to compare all columns. Three or more "
         "groups is usual; with two, a t test gives the same answer.")
_GROUPS_HELP_START = "Which columns (groups) to include. Leave empty to compare all columns. Three or more groups is usual; "
_WELCH_GROUPS_PARAM = ParamSpec(
    "groups", "Groups to compare", kind="groups", default=None, optional=True,
    help=_GROUPS_HELP_START + "with two, Welch's t test gives the same answer.")
_BF_GROUPS_PARAM = ParamSpec(
    "groups", "Groups to compare", kind="groups", default=None, optional=True,
    help=_GROUPS_HELP_START + "with two groups, use Welch's t test instead.")
_KW_GROUPS_PARAM = ParamSpec(
    "groups", "Groups to compare", kind="groups", default=None, optional=True,
    help=_GROUPS_HELP_START + "with two groups, use the Mann-Whitney test, the two-group "
         "version of this rank test.")
_ALPHA_PARAM = ParamSpec(
    "alpha", "Significance level (alpha)", kind="float", default=0.05,
    help="The P value threshold for calling a result significant. 0.05 is the usual choice.")
_CONTROL_PARAM = ParamSpec(
    "control", "Control group", kind="group", default=None, optional=True,
    help="Optional. The reference group that every other group is compared with (used by "
         "Dunnett's test and by the Bonferroni/Holm-type options). Leave empty to compare all "
         "pairs; Dunnett's test then uses the first column as control.")
_UNEQUAL_CONTROL_PARAM = ParamSpec(
    "control", "Control group", kind="group", default=None, optional=True,
    help="Optional. A reference group: with the Bonferroni/Holm-type options every other group is "
         "compared with it only (Welch t tests), instead of all pairs. Games-Howell always compares "
         "all pairs and ignores it. (Dunnett's test is not offered here because it assumes equal "
         "variances.) Leave empty to compare all pairs.")
_KW_CONTROL_PARAM = ParamSpec(
    "control", "Control group", kind="group", default=None, optional=True,
    help="Optional. A reference group: Dunn's test then compares every other group with it only, "
         "instead of all pairs (fewer comparisons, so a milder correction). Leave empty to compare "
         "all pairs.")
_ADJUST_HELP = (
    "How P values are corrected for making several comparisons. Bonferroni: multiply each P by "
    "the number of comparisons (simple, strict). Sidak: almost the same, slightly less strict. "
    "Holm and Holm-Sidak: step-wise versions that are never less powerful (good defaults). "
    "fdr_bh: controls the expected share of false discoveries (Benjamini-Hochberg), for "
    "exploratory work. none: no correction.")
_POSTHOC_EQUAL_HELP = (
    "Which follow-up comparisons to run after the overall test. tukey: all pairs of groups "
    "(the usual choice). dunnett: each group against one control group. bonferroni, sidak, "
    "holm, holm-sidak: t tests with a correction for multiple comparisons (Holm is a good "
    "general choice). fdr_bh: controls the false discovery rate instead (exploratory work). "
    "lsd: t tests with no correction (only for a few planned comparisons). games-howell: all "
    "pairs without assuming equal variances. none: skip follow-up comparisons.")
_POSTHOC_UNEQUAL_HELP = (
    "Which follow-up comparisons to run after the overall test. games-howell: all pairs of "
    "groups without assuming equal variances (the usual choice here). bonferroni, sidak, holm, "
    "holm-sidak: Welch t tests with a correction for multiple comparisons. fdr_bh: controls "
    "the false discovery rate instead (exploratory work). lsd: Welch t tests with no "
    "correction (only for a few planned comparisons). none: skip follow-up comparisons.")


def _selected_names(table: Any, groups: Any) -> list[str]:
    available = list(table.group_names)
    if groups is None or (isinstance(groups, list | tuple) and len(groups) == 0) or groups == "":
        return available
    chosen = [groups] if isinstance(groups, str) else [str(g) for g in groups]
    missing = [g for g in chosen if g not in available]
    if missing:
        raise StatsError(
            f"Unknown group(s): {', '.join(missing)}. The table has: {', '.join(available)}.")
    if len(set(chosen)) != len(chosen):
        raise StatsError("Each group can be selected only once.")
    return chosen


def _select_groups(table: Any, groups: Any) -> dict[str, np.ndarray]:
    chosen = _selected_names(table, groups)
    data = table.groups()
    return {name: data[name] for name in chosen}


def _check_control(control: Any, names: Sequence[str]) -> str | None:
    if control is None or control == "":
        return None
    if str(control) not in names:
        raise StatsError(
            f"The control group '{control}' is not among the groups being compared "
            f"({', '.join(names)}).")
    return str(control)


@register(
    key="anova_oneway", label="One-way ANOVA", category="Compare three or more groups",
    tables=("column",),
    params=(
        _GROUPS_PARAM,
        ParamSpec("posthoc", "Post-hoc comparisons", kind="choice", default="tukey",
                  choices=_EQUAL_VAR_POSTHOCS, help=_POSTHOC_EQUAL_HELP),
        _CONTROL_PARAM,
        _ALPHA_PARAM,
    ),
    description="Tests whether the means of three or more independent groups differ, assuming "
                "each group is roughly normal with similar spread, then shows which groups differ.",
    citation=CITE_FISHER,
)
def _run_anova_oneway(table: Any, groups: Any = None, posthoc: str = "tukey",
                      control: Any = None, alpha: float = 0.05, **_: Any) -> StatResult:
    data = _select_groups(table, groups)
    return anova_oneway(data, posthoc=posthoc, control=_check_control(control, list(data)),
                        alpha=alpha)


@register(
    key="welch_anova", label="Welch's ANOVA", category="Compare three or more groups",
    tables=("column",),
    params=(
        _WELCH_GROUPS_PARAM,
        ParamSpec("posthoc", "Post-hoc comparisons", kind="choice", default="games-howell",
                  choices=_UNEQUAL_VAR_POSTHOCS, help=_POSTHOC_UNEQUAL_HELP),
        _UNEQUAL_CONTROL_PARAM,
        _ALPHA_PARAM,
    ),
    description="Tests whether the means of three or more independent groups differ without "
                "assuming the groups have the same spread. Safer than ordinary ANOVA when "
                "variances or sample sizes differ.",
    citation=CITE_WELCH,
)
def _run_welch_anova(table: Any, groups: Any = None, posthoc: str = "games-howell",
                     control: Any = None, alpha: float = 0.05, **_: Any) -> StatResult:
    data = _select_groups(table, groups)
    return welch_anova(data, posthoc=posthoc, control=_check_control(control, list(data)),
                       alpha=alpha)


@register(
    key="brown_forsythe_anova", label="Brown-Forsythe ANOVA",
    category="Compare three or more groups", tables=("column",),
    params=(
        _BF_GROUPS_PARAM,
        ParamSpec("posthoc", "Post-hoc comparisons", kind="choice", default="games-howell",
                  choices=_UNEQUAL_VAR_POSTHOCS, help=_POSTHOC_UNEQUAL_HELP),
        _UNEQUAL_CONTROL_PARAM,
        _ALPHA_PARAM,
    ),
    description="Another test of whether three or more group means differ that does not assume "
                "equal spread. An alternative to Welch's ANOVA.",
    citation=CITE_BF,
)
def _run_brown_forsythe_anova(table: Any, groups: Any = None, posthoc: str = "games-howell",
                              control: Any = None, alpha: float = 0.05, **_: Any) -> StatResult:
    data = _select_groups(table, groups)
    return brown_forsythe_anova(data, posthoc=posthoc,
                                control=_check_control(control, list(data)), alpha=alpha)


@register(
    key="kruskal_wallis", label="Kruskal-Wallis test", category="Compare three or more groups",
    tables=("column",),
    params=(
        _KW_GROUPS_PARAM,
        ParamSpec("posthoc", "Post-hoc comparisons", kind="choice", default="dunn",
                  choices=("dunn", "none"),
                  help="Which follow-up comparisons to run. dunn: Dunn's test compares the "
                       "groups pair by pair using the same ranks as the overall test. none: "
                       "skip follow-up comparisons."),
        ParamSpec("adjust", "Multiple-comparison correction", kind="choice",
                  default="bonferroni", choices=_ADJUST_CHOICES, help=_ADJUST_HELP),
        _KW_CONTROL_PARAM,
        _ALPHA_PARAM,
    ),
    description="Compares three or more independent groups using ranks instead of the raw "
                "values, so it works for skewed data, outliers and scores that are not "
                "normally distributed.",
    citation=CITE_KW,
)
def _run_kruskal_wallis(table: Any, groups: Any = None, posthoc: str = "dunn",
                        adjust: str = "bonferroni", control: Any = None, alpha: float = 0.05,
                        **_: Any) -> StatResult:
    data = _select_groups(table, groups)
    return kruskal_wallis(data, posthoc=posthoc, adjust=adjust,
                          control=_check_control(control, list(data)), alpha=alpha)


@register(
    key="friedman", label="Friedman test", category="Repeated measures",
    tables=("column",),
    params=(
        ParamSpec("groups", "Conditions to compare", kind="groups", default=None, optional=True,
                  help="Which columns (conditions) to include; each row must be one subject "
                       "measured under every condition. Leave empty to use all columns. At "
                       "least three are needed."),
        ParamSpec("posthoc", "Post-hoc comparisons", kind="choice", default="dunn",
                  choices=("dunn", "wilcoxon", "none"),
                  help="Which follow-up comparisons to run. dunn: compares the conditions pair "
                       "by pair using the same within-subject ranks as the overall test. "
                       "wilcoxon: a Wilcoxon matched-pairs signed-rank test for each pair. "
                       "none: skip follow-up comparisons."),
        ParamSpec("adjust", "Multiple-comparison correction", kind="choice",
                  default="bonferroni", choices=_ADJUST_CHOICES, help=_ADJUST_HELP),
        _ALPHA_PARAM,
    ),
    description="Compares three or more conditions measured on the same subjects using ranks, "
                "so it does not need normally distributed data. Each row of the table is one "
                "subject.",
    citation=CITE_FRIEDMAN,
)
def _run_friedman(table: Any, groups: Any = None, posthoc: str = "dunn",
                  adjust: str = "bonferroni", alpha: float = 0.05, **_: Any) -> StatResult:
    chosen = _selected_names(table, groups)
    frame = table.analysis_frame()[chosen]
    partial = int((frame.notna().any(axis=1) & frame.isna().any(axis=1)).sum())
    complete = table.paired_frame(chosen)
    result = friedman_test(
        {name: complete[name].to_numpy(dtype=float) for name in chosen},
        posthoc=posthoc, adjust=adjust, alpha=alpha)
    if partial:
        result.data_notes.append(
            f"{partial} subject(s) (rows) with a missing value in at least one condition were "
            "left out: the Friedman test needs a value in every condition for each subject.")
        for check in result.assumptions:
            if check.name == "Complete blocks":
                check.detail = check.detail.rstrip(".") + f"; {partial} incomplete row(s) were left out."
    return result

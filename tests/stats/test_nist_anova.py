"""One-way ANOVA against the NIST Statistical Reference Datasets (StRD).

NIST certifies the between / within sums of squares, mean squares, F, R-squared and the
residual standard deviation of eleven one-way ANOVA problems to 15 significant digits
(https://www.itl.nist.gov/div898/strd/anova/anova.html). The files in
``tests/fixtures/nist`` are NIST's own, unmodified.

What limits the agreement
-------------------------
The analysis receives ordinary double-precision numbers. For the "average" and "higher"
difficulty problems the data have 7 or 13 constant leading digits (for example
1000000000000.4), and such a value simply cannot be stored exactly: near 1e12 the
spacing of doubles is 1.2e-4, while the scatter being analysed is 0.1. The certified
values refer to the exact decimal data, so *no* double-precision routine fed these
numbers can reproduce more than about 4 digits on SmLs07-09 (10 digits on SmLs04-06 and
AtmWtAg). Two extra tests show that this, and not the algorithm, is the limit:

* with the constant offset removed in exact decimal arithmetic before conversion to
  double, the same code reproduces every certified value to >= 14 digits;
* for the double-precision numbers actually supplied, the code agrees with an exact
  rational-arithmetic ANOVA of those numbers to ~1e-13.

The tolerances below are the honest, measured ones (worst of SS / MS / F per dataset;
"digits" = -log10 of the relative error):

    dataset   difficulty  digits reached   tolerance used
    SiRstv    lower       13.1             1e-12
    SmLs01    lower       15               1e-13
    SmLs02    lower       15               1e-13
    SmLs03    lower       15               1e-13
    AtmWtAg   average     10.2             1e-9
    SmLs04    average     10.1             1e-9
    SmLs05    average      9.9             1e-9
    SmLs06    average      9.9             1e-9
    SmLs07    higher       4.0             5e-4   (input representation limit)
    SmLs08    higher       3.9             5e-4   (input representation limit)
    SmLs09    higher       3.9             5e-4   (input representation limit)
"""

import math
from fractions import Fraction

import pytest
from scipy import stats

from tests.stats._nist import load_nist_anova
from torusplot.stats.oneway import anova_oneway

# dataset -> relative tolerance for plain double-precision input (see module docstring)
TOLERANCES = {
    "SiRstv": 1e-12,
    "SmLs01": 1e-13,
    "SmLs02": 1e-13,
    "SmLs03": 1e-13,
    "AtmWtAg": 1e-9,
    "SmLs04": 1e-9,
    "SmLs05": 1e-9,
    "SmLs06": 1e-9,
    "SmLs07": 5e-4,
    "SmLs08": 5e-4,
    "SmLs09": 5e-4,
}
# minimum number of correct significant digits that must be reached (guards against a
# silent loss of accuracy that would still fit inside a loose tolerance)
MIN_DIGITS = {
    "SiRstv": 12.5, "SmLs01": 14, "SmLs02": 14, "SmLs03": 14, "AtmWtAg": 9.5, "SmLs04": 9.5,
    "SmLs05": 9.5, "SmLs06": 9.5, "SmLs07": 3.5, "SmLs08": 3.5, "SmLs09": 3.5,
}


def _digits(value: float, certified: float) -> float:
    if value == certified:
        return 15.0
    return min(15.0, -math.log10(abs(value - certified) / abs(certified)))


def _table(result):
    between, within, total = result.anova_table
    assert between["source"] == "Between groups" and total["source"] == "Total"
    return between, within, total


def _check(result, ref, rel):
    between, within, total = _table(result)
    assert between["df"] == ref.df_between and within["df"] == ref.df_within
    assert result.df == ref.df_between and result.df2 == ref.df_within
    assert between["ss"] == pytest.approx(ref.ss_between, rel=rel)
    assert within["ss"] == pytest.approx(ref.ss_within, rel=rel)
    assert between["ms"] == pytest.approx(ref.ms_between, rel=rel)
    assert within["ms"] == pytest.approx(ref.ms_within, rel=rel)
    assert result.statistic == pytest.approx(ref.f, rel=rel)
    assert between["f"] == result.statistic
    assert result.extra["r_squared"] == pytest.approx(ref.r_squared, rel=rel)
    assert result.extra["root_mse"] == pytest.approx(ref.residual_sd, rel=rel)
    assert total["ss"] == pytest.approx(ref.ss_between + ref.ss_within, rel=rel)
    # the P value follows from the certified F and its degrees of freedom
    assert result.p_value == pytest.approx(
        stats.f.sf(ref.f, ref.df_between, ref.df_within), rel=max(100 * rel, 1e-9), abs=1e-300)


@pytest.mark.parametrize("name", list(TOLERANCES))
def test_nist_certified_values_from_double_precision_input(name):
    ref = load_nist_anova(name)
    result = anova_oneway(ref.groups(), posthoc="none")
    assert result.n_total == ref.df_between + ref.df_within + 1
    _check(result, ref, TOLERANCES[name])
    between, within, _ = _table(result)
    worst = min(
        _digits(between["ss"], ref.ss_between), _digits(within["ss"], ref.ss_within),
        _digits(between["ms"], ref.ms_between), _digits(within["ms"], ref.ms_within),
        _digits(result.statistic, ref.f),
    )
    assert worst >= MIN_DIGITS[name], f"{name}: only {worst:.1f} digits"


@pytest.mark.parametrize("name", list(TOLERANCES))
def test_nist_certified_values_when_data_are_exactly_representable(name):
    """Remove the constant leading digits exactly (decimal arithmetic): every dataset,
    including the higher-difficulty ones, is then reproduced to >= 14 digits."""
    ref = load_nist_anova(name)
    result = anova_oneway(ref.groups_offset_removed(), posthoc="none")
    _check(result, ref, 1e-14 if name != "AtmWtAg" else 5e-14)


def _exact_anova(groups):
    """One-way ANOVA of double-precision data in exact rational arithmetic."""
    exact = [[Fraction(float(v)) for v in values] for values in groups.values()]
    n_total = sum(len(g) for g in exact)
    grand = sum(sum(g) for g in exact) / n_total
    means = [sum(g) / len(g) for g in exact]
    ssb = sum(len(g) * (m - grand) ** 2 for g, m in zip(exact, means))
    ssw = sum((v - m) ** 2 for g, m in zip(exact, means) for v in g)
    k = len(exact)
    return float(ssb), float(ssw), float((ssb / (k - 1)) / (ssw / (n_total - k)))


@pytest.mark.parametrize("name", ["SiRstv", "AtmWtAg", "SmLs04", "SmLs07", "SmLs08"])
def test_algorithm_is_exact_for_the_numbers_it_is_given(name):
    """The few digits reached on SmLs07/08 are a property of the *input* (decimal data
    rounded to doubles), not of the computation: against an exact rational-arithmetic
    ANOVA of the same doubles the code is accurate to ~13 digits."""
    ref = load_nist_anova(name)
    groups = ref.groups()
    ssb, ssw, f = _exact_anova(groups)
    result = anova_oneway(groups, posthoc="none")
    between, within, _ = _table(result)
    assert between["ss"] == pytest.approx(ssb, rel=1e-13)
    assert within["ss"] == pytest.approx(ssw, rel=1e-13)
    assert result.statistic == pytest.approx(f, rel=1e-13)


@pytest.mark.parametrize("name", ["SmLs07", "SmLs08", "SmLs09"])
def test_higher_difficulty_groups_are_not_mistaken_for_constant(name):
    """Values such as 1e12 + 0.4 differ only in the 13th digit; they must still be
    analysed (not reported as 'no variability') and the result must be finite."""
    ref = load_nist_anova(name)
    result = anova_oneway(ref.groups(), posthoc="none")
    assert math.isfinite(result.statistic) and result.statistic > 0
    assert not any("identical" in w for w in result.warnings)
    assert result.effect("eta squared").value == pytest.approx(ref.r_squared, rel=5e-4)


def test_nist_full_result_with_default_posthoc():
    """SiRstv end to end with the default Tukey comparisons and all assumption checks."""
    ref = load_nist_anova("SiRstv")
    result = anova_oneway(ref.groups())
    assert result.statistic == pytest.approx(ref.f, rel=1e-12)
    assert len(result.comparisons) == 10 and result.correction == "Tukey HSD"
    assert result.significant is False  # certified F = 1.18 with 4 and 20 df
    assert all(c.p_adjusted > 0.05 for c in result.comparisons)
    assert result.model_dump_json()

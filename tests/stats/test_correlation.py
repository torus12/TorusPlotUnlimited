"""Correlation: Pearson, Spearman, correlation matrix.

Reference values: Anscombe's quartet (published), scipy cross-checks, brute-force
enumeration of the Spearman permutation distribution, and first-principles formulas.
"""

import itertools
import json
import math
import warnings

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import stats

from torusplot.data.tables import ColumnTable, MultipleVariablesTable, XYTable
from torusplot.stats import run_analysis
from torusplot.stats._utils import StatsError
from torusplot.stats.correlation import correlation_matrix, pearson, spearman
from torusplot.stats.multiplicity import adjust_pvalues


def assert_strict_json(res):
    """The result must serialise to strict JSON: no NaN / Infinity anywhere."""
    json.dumps(json.loads(res.finalize().model_dump_json()), allow_nan=False)


RNG = np.random.default_rng(20240607)
X = RNG.normal(10, 2, 25)
Y = 0.6 * X + RNG.normal(0, 1.5, 25)

# Anscombe FJ (1973) Graphs in statistical analysis. Am Stat 27:17-21 (Table p. 19).
ANSCOMBE_X123 = [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5]
ANSCOMBE = {
    "I": (ANSCOMBE_X123, [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68]),
    "II": (ANSCOMBE_X123, [9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74]),
    "III": (ANSCOMBE_X123, [7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73]),
    "IV": ([8, 8, 8, 8, 8, 8, 8, 19, 8, 8, 8],
           [6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.50, 5.56, 7.91, 6.89]),
}


# ------------------------------------------------------------------------ Pearson
@pytest.mark.parametrize("name", list(ANSCOMBE))
def test_pearson_anscombe_quartet_published_r(name):
    x, y = ANSCOMBE[name]
    res = pearson(x, y)
    # Anscombe gives "multiple R^2 = 0.667" for all four data sets, i.e. r = 0.816; the four
    # sets were constructed to agree to about three decimals, hence the 0.001 tolerance.
    assert res.effect("Pearson r").value == pytest.approx(0.816, abs=1e-3)
    assert res.effect("r squared").value == pytest.approx(0.667, abs=1e-3)


def test_pearson_anscombe_iv_flags_the_single_influential_point():
    res = pearson(*ANSCOMBE["IV"])
    check = next(a for a in res.assumptions if "influential" in a.name)
    assert check.status == "warn" and "single point" in check.detail
    res3 = pearson(*ANSCOMBE["III"])
    check3 = next(a for a in res3.assumptions if "influential" in a.name)
    assert check3.status == "warn"  # removing the outlier moves r from 0.816 to ~1.0
    res1 = pearson(*ANSCOMBE["I"])
    assert next(a for a in res1.assumptions if "influential" in a.name).status == "pass"


@pytest.mark.parametrize("alternative", ["two-sided", "greater", "less"])
def test_pearson_matches_scipy(alternative):
    res = pearson(X, Y, alternative=alternative)
    ref = stats.pearsonr(X, Y, alternative=alternative)
    assert res.effect("Pearson r").value == pytest.approx(ref.statistic, rel=1e-12)
    assert res.p_value == pytest.approx(ref.pvalue, rel=1e-9)
    ci = ref.confidence_interval(0.95)
    es = res.effect("Pearson r")
    assert es.ci_low == pytest.approx(ci.low, rel=1e-10)
    assert es.ci_high == pytest.approx(ci.high, rel=1e-10)
    assert res.df == X.size - 2
    assert res.alternative == alternative


def test_pearson_first_principles_and_ci_level():
    res = pearson(X, Y, ci_level=0.9)
    n = X.size
    r = np.sum((X - X.mean()) * (Y - Y.mean())) / math.sqrt(
        np.sum((X - X.mean()) ** 2) * np.sum((Y - Y.mean()) ** 2))
    t = r * math.sqrt((n - 2) / (1 - r * r))
    assert res.statistic == pytest.approx(t, rel=1e-12)
    assert res.p_value == pytest.approx(2 * stats.t.sf(abs(t), n - 2), rel=1e-12)
    z = math.atanh(r)
    half = stats.norm.ppf(0.95) / math.sqrt(n - 3)
    es = res.effect("Pearson r")
    assert (es.ci_low, es.ci_high) == pytest.approx((math.tanh(z - half), math.tanh(z + half)))
    assert es.ci_level == 0.9
    r2 = res.effect("r squared")
    assert r2.value == pytest.approx(r * r)
    assert r2.ci_low == pytest.approx(es.ci_low**2) and r2.ci_high == pytest.approx(es.ci_high**2)
    assert res.parameter("r").value == pytest.approx(r)
    assert "Fisher" in res.methods_text and res.interpretation
    assert res.n == {"X / Y pairs": n}
    assert [d["group"] for d in res.descriptives] == ["X", "Y"]


def test_pearson_ci_contains_zero_iff_not_significant_and_r2_interval_through_zero():
    x = np.arange(12.0)
    y = np.array([3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 9.0])
    res = pearson(x, y)
    es = res.effect("Pearson r")
    assert (es.ci_low <= 0 <= es.ci_high) == (res.p_value >= 0.05) or abs(res.p_value - 0.05) < 0.02
    weak = pearson(x, np.array([1, -1, 2, -2, 1, -1, 2, -2, 1, -1, 2, -2.5]))
    assert weak.effect("r squared").ci_low == 0.0


def test_pearson_invariances():
    base = pearson(X, Y)
    swapped = pearson(Y, X)
    scaled = pearson(3.0 * X - 7.0, 0.01 * Y + 100.0)
    flipped = pearson(-X, Y)
    r = base.effect("Pearson r").value
    assert swapped.effect("Pearson r").value == pytest.approx(r, rel=1e-12)
    assert scaled.effect("Pearson r").value == pytest.approx(r, rel=1e-10)
    assert scaled.p_value == pytest.approx(base.p_value, rel=1e-8)
    assert flipped.effect("Pearson r").value == pytest.approx(-r, rel=1e-12)


def test_pearson_extreme_scales_do_not_overflow():
    big = pearson(X * 1e200, Y * 1e-200)
    assert big.effect("Pearson r").value == pytest.approx(pearson(X, Y).effect("Pearson r").value)


def test_pearson_degenerate_inputs():
    two = pearson([1, 2], [3, 5])
    assert two.p_value is None and two.effect("Pearson r").value == 1.0
    assert any("only 2 pairs" in w for w in two.warnings)
    assert "cannot be tested" in two.interpretation
    three = pearson([1, 2, 3], [1, 3, 2])
    assert three.p_value is not None and three.effect("Pearson r").ci_low is None
    assert any("at least 4 pairs" in w for w in three.warnings)
    perfect = pearson([1, 2, 3, 4, 5], [2, 4, 6, 8, 10])
    assert perfect.effect("Pearson r").value == 1.0 and perfect.p_value == 0.0
    assert perfect.statistic is None
    assert any("exactly on a straight line" in w for w in perfect.warnings)
    assert perfect.effect("Pearson r").ci_low == 1.0
    with_nan = pearson([1, 2, np.nan, 4, 5, 7], [2, 1, 5, np.nan, 6, 9])
    assert with_nan.n == {"X / Y pairs": 4}
    assert any("missing" in note for note in with_nan.data_notes)
    for bad in (([1.0], [2.0]), ([], []), ([np.nan, 1], [1, np.nan])):
        with pytest.raises(StatsError, match="at least 2"):
            pearson(*bad)
    with pytest.raises(StatsError, match="identical"):
        pearson([1, 1, 1, 1], [1, 2, 3, 4])
    with pytest.raises(StatsError, match="identical"):
        pearson([1, 2, 3, 4], [5, 5, 5, 5])
    with pytest.raises(StatsError, match="same number"):
        pearson([1, 2, 3], [1, 2])
    with pytest.raises(StatsError, match="numbers"):
        pearson(["a", "b"], [1, 2])
    with pytest.raises(StatsError):
        pearson(X, Y, alpha=1.5)
    with pytest.raises(StatsError):
        pearson(X, Y, alternative="bigger")
    with pytest.raises(StatsError):
        pearson(X, Y, ci_level=0)


def test_pearson_normality_assumption_flags_skewed_data():
    skew = np.exp(np.linspace(0, 6, 30))
    res = pearson(skew, np.linspace(0, 1, 30) + 0.01 * np.sin(np.arange(30)))
    normal = [a for a in res.assumptions if a.name.startswith("Normality")]
    assert normal[0].status == "fail" and "Spearman" in normal[0].detail
    small = pearson([1, 2], [1, 3])
    assert [a.status for a in small.assumptions if a.name.startswith("Normality")] == ["warn", "warn"]
    big = pearson(np.arange(5001.0), np.arange(5001.0) ** 1.01)
    assert [a.status for a in big.assumptions if a.name.startswith("Normality")] == ["info", "info"]
    assert next(a for a in big.assumptions if "influential" in a.name).status == "info"


# ----------------------------------------------------------------------- Spearman
def test_spearman_matches_scipy_with_and_without_ties():
    res = spearman(X, Y)
    ref = stats.spearmanr(X, Y)
    assert res.statistic == pytest.approx(ref.statistic, rel=1e-12)
    assert res.p_value == pytest.approx(ref.pvalue, rel=1e-8)
    assert "t approximation" in res.extra["p_method"]
    xt = np.round(X)  # many ties
    yt = np.round(Y)
    res_t = spearman(xt, yt)
    ref_t = stats.spearmanr(xt, yt)
    assert res_t.statistic == pytest.approx(ref_t.statistic, rel=1e-12)
    assert res_t.p_value == pytest.approx(ref_t.pvalue, rel=1e-8)
    assert any("average ranks" in note for note in res_t.data_notes)
    for alternative in ("greater", "less"):
        one = spearman(X, Y, alternative=alternative)
        assert one.p_value == pytest.approx(stats.spearmanr(X, Y, alternative=alternative).pvalue,
                                            rel=1e-8)


def test_spearman_no_ties_equals_classic_formula():
    x = np.array([3.1, 1.2, 5.6, 4.4, 2.0, 9.9, 7.5, 6.1, 8.8, 0.3, 11.0, 10.2])
    y = np.array([2.0, 1.1, 4.1, 6.5, 3.3, 8.0, 9.9, 5.2, 7.7, 0.1, 12.5, 10.9])
    d = stats.rankdata(x) - stats.rankdata(y)
    n = x.size
    rho = 1 - 6 * np.sum(d**2) / (n * (n * n - 1))
    assert spearman(x, y).statistic == pytest.approx(rho, rel=1e-12)


def _brute_force_spearman_p(x, y, alternative):
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    n = len(x)
    obs = float(np.corrcoef(rx, ry)[0, 1])
    count = total = 0
    for perm in itertools.permutations(range(n)):
        r = float(np.corrcoef(rx, ry[list(perm)])[0, 1])
        total += 1
        if alternative == "two-sided":
            count += abs(r) >= abs(obs) - 1e-12
        elif alternative == "greater":
            count += r >= obs - 1e-12
        else:
            count += r <= obs + 1e-12
    return count / total


@pytest.mark.parametrize("alternative", ["two-sided", "greater", "less"])
def test_spearman_exact_p_matches_brute_force_enumeration(alternative):
    x = [1.2, 3.4, 2.2, 5.9, 4.1, 7.7, 6.0]
    y = [2.0, 2.9, 1.1, 6.2, 7.0, 5.5, 8.1]
    res = spearman(x, y, alternative=alternative)
    assert res.extra["p_method"] == "exact permutation distribution"
    assert res.p_value == pytest.approx(_brute_force_spearman_p(x, y, alternative), abs=1e-12)
    assert res.df is None


def test_spearman_exact_known_small_cases():
    # n = 5 perfectly monotone: two-sided exact P = 2/5! = 1/60
    res = spearman([1, 2, 3, 4, 5], [10, 20, 30, 40, 55])
    assert res.statistic == 1.0 and res.p_value == pytest.approx(2 / 120)
    assert spearman([1, 2, 3, 4, 5], [9, 7, 6, 3, 1], alternative="less").p_value == pytest.approx(1 / 120)
    # exact limit n = 9 is used; n = 10 falls back to the t approximation
    nine = spearman(np.arange(9.0), np.array([2, 1, 4, 3, 6, 5, 8, 7, 9.0]))
    assert nine.extra["p_method"].startswith("exact")
    ten = spearman(np.arange(10.0), np.array([2, 1, 4, 3, 6, 5, 8, 7, 10, 9.0]))
    assert ten.extra["p_method"].startswith("t approximation")


def test_spearman_bonett_wright_interval():
    res = spearman(X, Y, ci_level=0.95)
    rho, n = res.statistic, X.size
    se = math.sqrt((1 + rho**2 / 2) / (n - 3))
    z = math.atanh(rho)
    es = res.effect("Spearman rho")
    assert es.ci_low == pytest.approx(math.tanh(z - stats.norm.ppf(0.975) * se), rel=1e-12)
    assert es.ci_high == pytest.approx(math.tanh(z + stats.norm.ppf(0.975) * se), rel=1e-12)
    assert "Bonett" in es.ci_method and any("Bonett" in c for c in res.citations)
    # wider than the Pearson-style 1/sqrt(n-3) interval
    naive = math.tanh(z + stats.norm.ppf(0.975) / math.sqrt(n - 3))
    assert es.ci_high > naive
    one = spearman(X, Y, alternative="greater").effect("Spearman rho")
    assert one.ci_high == 1.0 and one.ci_low > es.ci_low


def test_spearman_monotone_transform_invariance_and_outlier_robustness():
    base = spearman(X, Y).statistic
    assert spearman(np.exp(X / 5), Y**3).statistic == pytest.approx(base, rel=1e-12)
    x = np.arange(1.0, 11)
    y = x.copy()
    y[-1] = 1e6  # an extreme outlier does not change ranks
    assert spearman(x, y).statistic == 1.0


def test_spearman_degenerate_inputs():
    two = spearman([1, 2], [5, 3])
    assert two.statistic == -1.0 and two.p_value == 1.0
    assert any("only 2 pairs" in w for w in two.warnings)
    three = spearman([1, 2, 3], [1, 3, 2])
    assert any("at least 4 pairs" in w for w in three.warnings)
    ties_small = spearman([1, 1, 2, 3, 4], [2, 3, 3, 5, 6])
    assert any("exact P value is unavailable" in w for w in ties_small.warnings)
    with pytest.raises(StatsError, match="identical"):
        spearman([2, 2, 2], [1, 2, 3])
    with pytest.raises(StatsError):
        spearman([1], [1])
    assert spearman([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).p_value == 0.0


# ------------------------------------------------------------------------- matrix
def _frame():
    rng = np.random.default_rng(5)
    a = rng.normal(size=30)
    df = pd.DataFrame({
        "a": a,
        "b": 0.8 * a + rng.normal(scale=0.5, size=30),
        "c": rng.normal(size=30),
        "d": -0.5 * a + rng.normal(scale=1.0, size=30),
    })
    df.loc[[3, 7], "c"] = np.nan
    df.loc[[11], "d"] = np.nan
    return df


@pytest.mark.parametrize("method", ["pearson", "spearman"])
def test_matrix_pairwise_complete_matches_scipy(method):
    df = _frame()
    res = correlation_matrix(df, method=method)
    m = res.extra["matrix"]
    assert m["labels"] == ["a", "b", "c", "d"]
    func = stats.pearsonr if method == "pearson" else stats.spearmanr
    for i, a in enumerate(m["labels"]):
        assert m["r"][i][i] == 1.0 and m["p"][i][i] is None
        assert m["n"][i][i] == int(df[a].notna().sum())
        for j, b in enumerate(m["labels"]):
            if i == j:
                continue
            pair = df[[a, b]].dropna()
            ref = func(pair[a], pair[b])
            assert m["r"][i][j] == pytest.approx(ref.statistic, rel=1e-10)
            assert m["p"][i][j] == pytest.approx(ref.pvalue, rel=1e-7)
            assert m["n"][i][j] == len(pair)
            assert m["r"][i][j] == m["r"][j][i]
    assert m["n"][0][2] == 28 and m["n"][2][3] == 27
    assert any("pairwise-complete" in note for note in res.data_notes)
    assert len(res.extra["pairs"]) == 6 and len(res.effect_sizes) == 6
    assert res.comparison_family == "all 6 pairwise correlations among 4 variables"
    assert any("without adjusting" in w for w in res.warnings)
    # pandas agrees as well (independent implementation of pairwise deletion)
    np.testing.assert_allclose(np.array(m["r"], dtype=float), df.corr(method=method).to_numpy(),
                               rtol=1e-10)


@pytest.mark.parametrize("adjust", ["bonferroni", "holm", "sidak", "holm-sidak", "fdr_bh"])
def test_matrix_adjustment_uses_shared_multiplicity_module(adjust):
    df = _frame()
    raw = correlation_matrix(df)
    adj = correlation_matrix(df, adjust=adjust)
    p_raw = [pair["p"] for pair in raw.extra["pairs"]]
    expected = adjust_pvalues(p_raw, adjust)
    got = [pair["p_adjusted"] for pair in adj.extra["pairs"]]
    np.testing.assert_allclose(got, expected, rtol=1e-12)
    m = adj.extra["matrix"]
    assert m["p_adjusted"][0][1] == pytest.approx(expected[0])
    assert m["p_adjusted"][1][0] == m["p_adjusted"][0][1] and m["p_adjusted"][0][0] is None
    assert all(a >= r - 1e-15 for a, r in zip(got, p_raw))
    assert not any("without adjusting" in w for w in adj.warnings)
    assert adj.correction and adj.correction != "none (unadjusted)"


def test_matrix_degenerate_columns_and_errors():
    df = pd.DataFrame({"a": [1.0, 2, 3, 4, 5], "const": [2.0] * 5, "txt": list("abcde"),
                       "few": [1.0, np.nan, np.nan, np.nan, 2.0], "b": [2.0, 1, 4, 3, 6],
                       "flag": [True, False, True, False, True]})
    res = correlation_matrix(df)
    m = res.extra["matrix"]
    assert m["labels"] == ["a", "const", "few", "b"]
    i, j = m["labels"].index("a"), m["labels"].index("const")
    assert m["r"][i][j] is None and m["p"][i][j] is None and m["r"][j][j] is None
    assert any("constant" in w for w in res.warnings)
    assert any("complete pair" in w for w in res.warnings)
    assert any("Non-numeric" in note and "txt" in note for note in res.data_notes)
    assert_strict_json(res)
    with pytest.raises(StatsError, match="two numeric"):
        correlation_matrix(pd.DataFrame({"a": [1, 2, 3], "t": ["x", "y", "z"]}))
    with pytest.raises(StatsError, match="method"):
        correlation_matrix(df, method="kendall")
    with pytest.raises(StatsError, match="adjustment"):
        correlation_matrix(df, adjust="magic")
    nothing = correlation_matrix(pd.DataFrame({"a": [1.0, 1, 1], "b": [1.0, 2, 3]}))
    assert "No correlation could be computed" in nothing.interpretation
    from_dict = correlation_matrix({"a": [1, 2, 3, 4], "b": [2, 1, 4, 3]})
    assert from_dict.extra["matrix"]["r"][0][1] == pytest.approx(0.6)
    inf_frame = pd.DataFrame({"a": [1.0, 2, 3, np.inf, 5], "b": [2.0, 1, 4, 3, 6]})
    assert correlation_matrix(inf_frame).extra["matrix"]["n"][0][1] == 4


# ------------------------------------------------------------------- property tests
finite = st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)


@settings(max_examples=60, deadline=None)
@given(st.lists(st.tuples(st.one_of(finite, st.just(float("nan"))), finite), min_size=0, max_size=14),
       st.sampled_from(["pearson", "spearman"]),
       st.sampled_from(["two-sided", "greater", "less"]))
def test_property_never_crashes_and_stays_in_range(pairs, method, alternative):
    x = [p[0] for p in pairs]
    y = [p[1] for p in pairs]
    func = pearson if method == "pearson" else spearman
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        try:
            res = func(x, y, alternative=alternative)
        except StatsError:
            return
    r = res.effect_sizes[0].value
    assert -1.0 <= r <= 1.0
    assert res.p_value is None or 0.0 <= res.p_value <= 1.0
    es = res.effect_sizes[0]
    if es.ci_low is not None:
        assert -1.0 <= es.ci_low <= r + 1e-12 and r - 1e-12 <= es.ci_high <= 1.0
    assert_strict_json(res)


@settings(max_examples=40, deadline=None)
@given(st.lists(st.integers(-3, 3), min_size=4, max_size=12), st.data())
def test_property_ties_zeros_and_order_invariance(values, data):
    y = data.draw(st.lists(st.integers(-3, 3), min_size=len(values), max_size=len(values)))
    x = np.array(values, dtype=float)
    yv = np.array(y, dtype=float)
    try:
        a = spearman(x, yv)
    except StatsError:
        return
    perm = np.random.default_rng(len(values)).permutation(x.size)
    b = spearman(x[perm], yv[perm])
    assert a.statistic == pytest.approx(b.statistic, abs=1e-12)
    assert a.p_value == pytest.approx(b.p_value, abs=1e-12)
    ref = stats.spearmanr(x, yv)
    assert a.statistic == pytest.approx(ref.statistic, abs=1e-10)


# ----------------------------------------------------------------------- registry
def test_registry_correlation_on_xy_table_with_replicates():
    x = np.arange(1.0, 9.0)
    reps = np.column_stack([2 * x + 1, 2 * x + 2, 2 * x - 1 + 0.3 * np.sin(x)])
    table = XYTable.from_arrays("xy", x, {"Signal": reps, "Other": x**2}, x_name="Dose")
    res = run_analysis(table, "correlation")
    assert res.analysis == "correlation" and res.table == "xy"
    assert res.n == {"Dose / Signal pairs": 24}
    assert any("pseudo-replication" in w for w in res.warnings)
    assert res.interpretation and res.why_chosen and res.assumptions and res.significant is True
    json.loads(res.model_dump_json())
    means = run_analysis(table, "correlation", use_means=True, method="spearman", series="Other")
    assert means.n == {"Dose / Other pairs": 8} and means.test_name.startswith("Spearman")
    assert not any("pseudo-replication" in w for w in means.warnings)
    avg = run_analysis(table, "correlation", use_means=True)
    assert any("averaged" in note for note in avg.data_notes)
    with pytest.raises(StatsError, match="Unknown series"):
        run_analysis(table, "correlation", series="Nope")
    with pytest.raises(StatsError, match="method"):
        run_analysis(table, "correlation", method="kendall")


def test_registry_correlation_on_column_and_multiple_variables_tables():
    col = ColumnTable.from_dict("col", {"Before": [1.0, 2, 3, 4, 5, np.nan, 7],
                                        "After": [1.5, 1.9, 3.5, 3.9, 5.6, 6.0, 6.8],
                                        "Third": [3.0, 1, 2, 5, 4, 7, 6]})
    res = run_analysis(col, "correlation")
    ref = stats.pearsonr([1.0, 2, 3, 4, 5, 7], [1.5, 1.9, 3.5, 3.9, 5.6, 6.8])
    assert res.effect("Pearson r").value == pytest.approx(ref.statistic, rel=1e-12)
    assert any("pairs" in note for note in res.data_notes)
    other = run_analysis(col, "correlation", group_a="Third", group_b="After", alternative="greater",
                         ci_level=0.9)
    assert other.alternative == "greater" and "Third" in other.interpretation
    assert other.effect("Pearson r").ci_level == 0.9
    with pytest.raises(StatsError, match="different columns"):
        run_analysis(col, "correlation", group_a="After", group_b="After")
    with pytest.raises(StatsError, match="Unknown column"):
        run_analysis(col, "correlation", group_a="Zzz")
    one = ColumnTable.from_dict("one", {"Only": [1.0, 2, 3]})
    with pytest.raises(StatsError, match="enough"):
        run_analysis(one, "correlation")

    mv = MultipleVariablesTable("mv", pd.DataFrame({
        "group": list("aabbccdd"), "w": [1.0, 2, 3, 4, 5, 6, 7, 8],
        "h": [2.1, 3.9, 6.2, 8.1, 9.8, 12.2, 13.8, 16.5], "z": [5.0, 3, 8, 1, 9, 2, 7, 4]}))
    res_mv = run_analysis(mv, "correlation")
    assert res_mv.n == {"w / h pairs": 8}
    res_mv2 = run_analysis(mv, "correlation", x="z", y="w", method="spearman")
    assert res_mv2.statistic == pytest.approx(stats.spearmanr(mv.data["z"], mv.data["w"]).statistic)
    json.loads(res_mv2.model_dump_json())
    with pytest.raises(StatsError, match="different variables"):
        run_analysis(mv, "correlation", x="w", y="w")
    with pytest.raises(StatsError, match="Unknown numeric column"):
        run_analysis(mv, "correlation", x="group")


def test_registry_correlation_matrix():
    mv = MultipleVariablesTable("mv", pd.DataFrame({
        "group": list("aabbccdd"), "w": [1.0, 2, 3, 4, 5, 6, 7, 8],
        "h": [2.1, 3.9, 6.2, 8.1, 9.8, 12.2, 13.8, 16.5], "z": [5.0, 3, 8, 1, 9, 2, 7, np.nan]}))
    res = run_analysis(mv, "correlation_matrix")
    assert res.extra["matrix"]["labels"] == ["w", "h", "z"]
    assert res.interpretation and res.why_chosen and res.assumptions
    assert isinstance(res.extra["matrix"]["n"][0][0], int)
    json.loads(res.model_dump_json())
    sub = run_analysis(mv, "correlation_matrix", columns=["w", "z"], method="spearman", adjust="holm",
                       ci_level=0.99)
    assert sub.effect_sizes[0].ci_level == 0.99
    assert sub.extra["matrix"]["labels"] == ["w", "z"] and sub.test_name.startswith("Spearman")
    as_text = run_analysis(mv, "correlation_matrix", columns="w, h")
    assert as_text.extra["matrix"]["labels"] == ["w", "h"]
    with pytest.raises(StatsError, match="Unknown or non-numeric"):
        run_analysis(mv, "correlation_matrix", columns=["w", "group"])
    with pytest.raises(StatsError, match="at least two"):
        run_analysis(mv, "correlation_matrix", columns=["w"])
    col = ColumnTable.from_dict("col", {"A": [1.0, 2, 3, 4, 5], "B": [2.0, 1, 4, 3, 6],
                                        "C": [5.0, 4, 3, 2, 0]})
    res_col = run_analysis(col, "correlation_matrix", adjust="fdr_bh")
    assert res_col.extra["matrix"]["r"][0][2] == pytest.approx(stats.pearsonr([1, 2, 3, 4, 5],
                                                                              [5, 4, 3, 2, 0]).statistic)
    assert any("subjects" in note for note in res_col.data_notes)
    # user exclusions are respected
    col.exclude([(4, "C")])
    assert run_analysis(col, "correlation_matrix").extra["matrix"]["n"][0][2] == 4


# =============================================================================
# Audit part 2 (C1): unit and offset invariance, shared ulp-based "constant" test
# =============================================================================
def _corr_summary(res):
    return [res.statistic, res.p_value, res.effect_sizes[0].value, res.effect_sizes[0].ci_low,
            res.effect_sizes[0].ci_high, *[a.p_value for a in res.assumptions if a.p_value is not None]]


@pytest.mark.parametrize("func", [pearson, spearman])
@pytest.mark.parametrize("sx, sy", [(1e-13, 1.0), (1.0, 1e-13), (1e9, 1e-9), (1e150, 1e150), (1e-150, 1e150)])
def test_correlation_does_not_depend_on_the_units(func, sx, sy):
    base = func(X, Y)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = func(X * sx, Y * sy)
    assert _corr_summary(res) == pytest.approx(_corr_summary(base), rel=1e-9)
    assert_strict_json(res)


@pytest.mark.parametrize("func", [pearson, spearman])
@pytest.mark.parametrize("offset", [1e9, 1e12, 1e15])
def test_correlation_does_not_depend_on_a_large_common_offset(func, offset):
    """X = offset + small integers is exactly representable and clearly not constant; a plain
    x - mean(x) loses the centring once the running sum passes 2**53 (twelve values at 1e15)."""
    x = np.arange(1.0, 13.0)
    y = np.array([2.3, 3.1, 7.2, 7.9, 9.4, 13.8, 13.1, 17.6, 17.2, 21.5, 21.9, 25.0])
    assert np.array_equal((x + offset) - offset, x)
    base, res = func(x, y), func(x + offset, y)
    assert _corr_summary(res) == pytest.approx(_corr_summary(base), rel=1e-9)
    if func is pearson:
        assert res.effect_sizes[0].value == pytest.approx(stats.pearsonr(x, y)[0], rel=1e-12)


def test_constant_means_identical_up_to_rounding_only():
    from torusplot.stats.correlation import is_constant

    assert is_constant(np.array([5.0, 5.0])) and is_constant(np.array([0.1 + 0.2, 0.3]))
    assert not is_constant(np.array([1e15 + 1, 1e15 + 2]))          # distinct doubles on a huge offset
    assert not is_constant(np.array([1e-15, 3e-15]))                # small units
    assert not is_constant(np.array([1e-200, 3e-200]))              # no absolute floor at all
    with pytest.raises(StatsError, match="identical"):
        pearson([3.0, 3.0, 3.0, 3.0], [1.0, 2.0, 3.0, 4.0])
    with pytest.raises(StatsError, match="identical"):
        spearman([1.0, 2.0, 3.0], [0.1 + 0.2, 0.3, 0.3])


def test_correlation_matrix_is_offset_and_unit_invariant():
    rng = np.random.default_rng(5)
    df = pd.DataFrame(rng.normal(size=(15, 3)), columns=list("abc"))
    df["b"] += 0.5 * df["a"]
    moved = pd.DataFrame({"a": np.round(df["a"] * 8) / 8 * 1e-13, "b": df["b"] * 1e150, "c": df["c"]})
    plain = pd.DataFrame({"a": np.round(df["a"] * 8) / 8, "b": df["b"], "c": df["c"]})
    base = correlation_matrix(plain)
    res = correlation_matrix(moved)
    for pb, pr in zip(base.extra["pairs"], res.extra["pairs"]):
        assert pr["r"] == pytest.approx(pb["r"], rel=1e-9) and pr["p"] == pytest.approx(pb["p"], rel=1e-8)
    shifted = plain.assign(a=plain["a"] + 1e15)
    assert np.array_equal(shifted["a"] - 1e15, plain["a"])
    for pb, pr in zip(base.extra["pairs"], correlation_matrix(shifted).extra["pairs"]):
        assert pr["r"] == pytest.approx(pb["r"], rel=1e-9)


def test_one_tailed_correlation_says_so_in_the_interpretation():
    two = pearson(X, Y)
    one = pearson(X, Y, alternative="greater")
    assert one.p_value == pytest.approx(two.p_value / 2, rel=1e-9)
    assert "one-tailed" in one.interpretation and "one-tailed" not in two.interpretation
    assert "one-tailed" in spearman(X, Y, alternative="less").interpretation

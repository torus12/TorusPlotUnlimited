"""Effect sizes and their confidence intervals.

Reference values are derived independently inside the tests from the defining
equations (noncentral distributions via scipy), plus textbook identities.
"""

import math

import numpy as np
import pytest
from scipy import stats

from torusplot.stats import effect_sizes as es

X = np.array([5.1, 4.9, 6.2, 5.8, 6.0, 5.5, 5.3, 6.1])
Y = np.array([4.1, 4.6, 5.0, 4.4, 4.9, 4.7, 5.2])


def test_cohen_d_matches_definition_and_t():
    d = es.cohen_d_independent(X, Y)
    sp = math.sqrt(((X.size - 1) * X.var(ddof=1) + (Y.size - 1) * Y.var(ddof=1)) / (X.size + Y.size - 2))
    assert d.value == pytest.approx((X.mean() - Y.mean()) / sp, rel=1e-12)
    t = stats.ttest_ind(X, Y).statistic
    assert d.value == pytest.approx(t * math.sqrt(1 / X.size + 1 / Y.size), rel=1e-12)
    assert d.magnitude == "large"


def test_cohen_d_ci_inverts_noncentral_t():
    d = es.cohen_d_independent(X, Y)
    df = X.size + Y.size - 2
    scale = math.sqrt(1 / X.size + 1 / Y.size)
    t = d.value / scale
    # at the CI limits the observed t sits exactly at the 97.5th / 2.5th percentile
    assert stats.nct.cdf(t, df, d.ci_low / scale) == pytest.approx(0.975, abs=1e-6)
    assert stats.nct.cdf(t, df, d.ci_high / scale) == pytest.approx(0.025, abs=1e-6)
    assert d.ci_low < d.value < d.ci_high


def test_ci_contains_zero_iff_not_significant():
    rng = np.random.default_rng(1)
    for _ in range(30):
        a, b = rng.normal(0, 1, 9), rng.normal(0.6, 1, 11)
        p = stats.ttest_ind(a, b).pvalue
        d = es.cohen_d_independent(a, b)
        assert (d.ci_low <= 0 <= d.ci_high) == (p >= 0.05)


def test_hedges_g_correction():
    d = es.cohen_d_independent(X, Y)
    g = es.cohen_d_independent(X, Y, hedges=True)
    df = X.size + Y.size - 2
    j = es.hedges_correction(df)
    assert j == pytest.approx(math.gamma(df / 2) / (math.sqrt(df / 2) * math.gamma((df - 1) / 2)), rel=1e-12)
    assert j == pytest.approx(1 - 3 / (4 * df - 1), abs=2e-3)  # Hedges' well-known approximation
    assert g.value == pytest.approx(d.value * j, rel=1e-12)
    assert g.name == "Hedges' g" and abs(g.value) < abs(d.value)


def test_paired_and_one_sample_d():
    a, b = X[:7], Y
    dz = es.cohen_d_paired(a, b)
    diff = a - b
    assert dz.value == pytest.approx(diff.mean() / diff.std(ddof=1), rel=1e-12)
    t = stats.ttest_rel(a, b).statistic
    assert dz.value == pytest.approx(t / math.sqrt(diff.size), rel=1e-12)
    one = es.cohen_d_one_sample(X, 5.0)
    assert one.value == pytest.approx((X.mean() - 5.0) / X.std(ddof=1), rel=1e-12)
    assert one.ci_low < one.value < one.ci_high


def test_cohen_d_from_t_agrees():
    t = stats.ttest_ind(X, Y).statistic
    a = es.cohen_d_from_t(t, X.size, Y.size)
    b = es.cohen_d_independent(X, Y)
    assert a.value == pytest.approx(b.value, rel=1e-12)
    assert a.ci_low == pytest.approx(b.ci_low, rel=1e-8)


def test_degenerate_inputs_do_not_crash():
    assert es.cohen_d_independent(np.array([1.0]), Y).value is None
    assert es.cohen_d_independent(np.array([2.0, 2.0]), np.array([2.0, 2.0])).value is None
    assert es.cohen_d_one_sample(np.array([3.0, 3.0, 3.0])).value is None
    assert es.rank_biserial_paired(np.zeros(4)).value is None
    assert math.isnan(es.hedges_correction(1))


def test_eta_omega_squared_one_way():
    groups = [np.array([1.0, 2, 3, 4]), np.array([2.0, 3, 4, 5]), np.array([5.0, 6, 7, 9])]
    allv = np.concatenate(groups)
    ss_total = ((allv - allv.mean()) ** 2).sum()
    ss_between = sum(g.size * (g.mean() - allv.mean()) ** 2 for g in groups)
    ss_within = ss_total - ss_between
    df1, df2 = 2, allv.size - 3
    f = (ss_between / df1) / (ss_within / df2)
    assert f == pytest.approx(stats.f_oneway(*groups).statistic, rel=1e-12)
    eta = es.eta_squared(ss_between, ss_total, f, df1, df2)
    assert eta.value == pytest.approx(ss_between / ss_total, rel=1e-12)
    peta = es.partial_eta_squared(ss_between, ss_within, df1, df2)
    assert peta.value == pytest.approx(eta.value, rel=1e-12)  # identical in a one-way design
    assert es.partial_eta_squared_from_f(f, df1, df2).value == pytest.approx(eta.value, rel=1e-12)
    om = es.omega_squared(ss_between, df1, ss_within / df2, ss_total, f, df2)
    assert om.value == pytest.approx((ss_between - df1 * ss_within / df2) / (ss_total + ss_within / df2), rel=1e-12)
    assert om.value < eta.value
    # CI limits invert the noncentral F
    n = df1 + df2 + 1
    lam_lo = eta.ci_low * n / (1 - eta.ci_low)
    lam_hi = eta.ci_high * n / (1 - eta.ci_high)
    assert stats.ncf.cdf(f, df1, df2, lam_lo) == pytest.approx(0.975, abs=1e-6)
    assert stats.ncf.cdf(f, df1, df2, lam_hi) == pytest.approx(0.025, abs=1e-6)


def test_eta_ci_lower_bound_zero_when_not_significant():
    lo, hi = es.eta_squared_ci(0.5, 2, 20)
    assert lo == 0.0 and 0 < hi < 1
    assert es.omega_squared(1.0, 2, 5.0, 100.0).value == 0.0  # negative estimate floored at 0


def test_partial_omega_and_cohen_f():
    po = es.partial_omega_squared(6.0, 2, 27, 30)
    assert po.value == pytest.approx(2 * 5 / (2 * 5 + 30), rel=1e-12)
    assert es.cohen_f_from_eta2(0.2) == pytest.approx(0.5, rel=1e-12)
    assert math.isnan(es.cohen_f_from_eta2(1.0))


def test_rank_biserial_independent():
    r = es.rank_biserial_independent(X, Y)
    u = stats.mannwhitneyu(X, Y).statistic  # U for X
    assert r.value == pytest.approx(2 * u / (X.size * Y.size) - 1, rel=1e-12)
    # complete separation gives +/-1
    assert es.rank_biserial_independent(np.array([5.0, 6, 7]), np.array([1.0, 2, 3])).value == 1.0
    assert es.rank_biserial_independent(np.array([1.0, 2, 3]), np.array([5.0, 6, 7])).value == -1.0
    assert r.ci_low < r.value <= r.ci_high <= 1.0
    # antisymmetric
    assert es.rank_biserial_independent(Y, X).value == pytest.approx(-r.value, rel=1e-12)


def test_rank_biserial_paired_kerby_formula():
    diff = np.array([1.5, -0.5, 2.0, 3.0, -1.0, 2.5, 0.0])
    r = es.rank_biserial_paired(diff)
    nz = diff[diff != 0]
    ranks = stats.rankdata(np.abs(nz))
    wp, wm = ranks[nz > 0].sum(), ranks[nz < 0].sum()
    assert r.value == pytest.approx((wp - wm) / (wp + wm), rel=1e-12)
    assert -1 <= r.ci_low < r.value < r.ci_high <= 1


def test_magnitude_labels():
    assert es.magnitude_d(0.1) == "negligible" and es.magnitude_d(-0.9) == "large"
    assert es.magnitude_eta2(0.07) == "medium" and es.magnitude_r(0.35) == "medium"
    assert es.magnitude_d(float("nan")) is None


def test_kw_and_friedman_effect_sizes():
    assert es.epsilon_squared_kw(6.0, 13, 3).value == pytest.approx(0.5)
    assert es.kendall_w(8.0, 10, 3).value == pytest.approx(0.4)


def test_hedges_g_undefined_for_one_degree_of_freedom():
    g = es.cohen_d_one_sample(np.array([1.0, 3.0]), hedges=True)
    assert g.value is None and g.ci_low is None
    assert es.cohen_d_one_sample(np.array([1.0, 3.0])).value is not None


def test_is_constant_uses_floating_point_spacing():
    from torusplot.stats._utils import is_constant, p_phrase

    assert is_constant(np.array([5.0, 5.0, 5.0])) and is_constant(np.array([1e12, 1e12]))
    assert not is_constant(np.array([1e12 + 0.2, 1e12 + 0.4]))  # tiny but real spread on a huge offset
    assert not is_constant(np.array([1.0, 1.0 + 1e-9]))
    assert is_constant(np.array([0.1 + 0.2, 0.3]))  # rounding noise only
    assert p_phrase(0.00001) == "P < 0.0001" and p_phrase(0.0312) == "P = 0.0312" and p_phrase(None) == "P not available"


def test_is_constant_treats_underflowing_spread_as_constant():
    from torusplot.stats._utils import is_constant

    assert is_constant(np.array([0.0, 0.0, 1.1e-153]))  # variance and higher moments would underflow
    assert not is_constant(np.array([0.0, 0.0, 1e-60]))

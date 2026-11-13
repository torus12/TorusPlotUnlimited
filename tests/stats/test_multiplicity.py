"""p-value adjustments: hand-worked values and a cross-check against statsmodels."""

import numpy as np
import pytest
from statsmodels.stats.multitest import multipletests

from torusplot.stats.multiplicity import adjust_pvalues

P = np.array([0.01, 0.04, 0.03, 0.005, 0.2])


def test_hand_worked_values():
    np.testing.assert_allclose(adjust_pvalues(P, "bonferroni"), [0.05, 0.2, 0.15, 0.025, 1.0])
    # Holm: sorted 0.005,0.01,0.03,0.04,0.2 -> x5,x4,x3,x2,x1 then running max
    np.testing.assert_allclose(adjust_pvalues(P, "holm"), [0.04, 0.09, 0.09, 0.025, 0.2])
    # BH: sorted p*m/rank = .025,.025,.05,.05,.2 (already monotone)
    np.testing.assert_allclose(adjust_pvalues(P, "fdr_bh"), [0.025, 0.05, 0.05, 0.025, 0.2])
    np.testing.assert_allclose(adjust_pvalues(P, "sidak"), 1 - (1 - P) ** 5)
    np.testing.assert_allclose(adjust_pvalues(P, "none"), P)


@pytest.mark.parametrize(
    "ours,theirs",
    [("bonferroni", "bonferroni"), ("sidak", "sidak"), ("holm", "holm"), ("holm-sidak", "holm-sidak"), ("fdr_bh", "fdr_bh")],
)
def test_matches_statsmodels(ours, theirs):
    rng = np.random.default_rng(7)
    for m in (1, 2, 7, 40):
        p = rng.uniform(0, 1, m) ** 2
        np.testing.assert_allclose(adjust_pvalues(p, ours), multipletests(p, method=theirs)[1], rtol=1e-10, atol=1e-14)


def test_nan_handling_and_aliases():
    out = adjust_pvalues([0.01, np.nan, 0.04], "bonferroni")
    assert np.isnan(out[1]) and out[0] == pytest.approx(0.02) and out[2] == pytest.approx(0.08)
    assert np.isnan(adjust_pvalues([np.nan], "holm")).all()
    assert adjust_pvalues([], "holm").size == 0
    np.testing.assert_allclose(adjust_pvalues(P, "BH"), adjust_pvalues(P, "fdr_bh"))
    np.testing.assert_allclose(adjust_pvalues(P, "Šidák"), adjust_pvalues(P, "sidak"))
    with pytest.raises(ValueError):
        adjust_pvalues(P, "magic")


def test_adjusted_never_below_raw_and_capped():
    rng = np.random.default_rng(3)
    p = rng.uniform(0, 1, 25)
    for method in ("bonferroni", "sidak", "holm", "holm-sidak", "fdr_bh"):
        adj = adjust_pvalues(p, method)
        assert np.all(adj >= p - 1e-15) and np.all(adj <= 1.0)


def test_p_of_exactly_one_gives_no_warning():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for method in ("sidak", "holm-sidak", "holm", "bonferroni", "fdr_bh"):
            out = adjust_pvalues([1.0, 0.5, 0.01], method)
            assert out[0] == 1.0 and np.all(out <= 1.0)

"""NIST StRD nonlinear least-squares benchmarks, fitted through the public API.

Every dataset is fitted from BOTH official NIST starting points with
``fit_curve(..., "user_defined", equation=...)``, which exercises the safe equation parser,
its automatic-differentiation Jacobian and the fitter. Parameters, standard errors, the
residual sum of squares, the residual standard deviation and the degrees of freedom are
compared with the certified values in the header of each NIST file.

Tolerances (documented in tests/fixtures/sources/regression.md):

* parameters: relative 1e-6 from both starts (observed agreement: 8 to 11 digits; NIST
  certifies 11 digits);
* standard errors: relative 1e-5; residual SS: relative 1e-8;
* Lanczos1 is the one exception for SE and SS: its data were generated to 14 digits from the
  model, so the certified residual SS is 1.4e-25 and the residuals (about 1e-13) are at the
  rounding level of double precision. Its *parameters* still agree to 1e-6; SE and SS are
  only required to agree within 5% (observed: about 3 digits).
"""

import math

import numpy as np
import pytest

from tests.fixtures.nist.strd_regression import NLS_MODELS, load_nls
from torusplot.stats.nonlinear import fit_curve

PARAM_RTOL = 1e-6
SE_RTOL = 1e-5
SS_RTOL = 1e-8
ROUNDING_LIMITED = {"Lanczos1"}

CASES = [(name, start) for name in NLS_MODELS for start in (1, 2)]


def test_all_26_single_predictor_datasets_are_present():
    assert len(NLS_MODELS) == 26
    for name in NLS_MODELS:
        d = load_nls(name)
        assert d.x.size == d.n == d.y.size
        if name != "Rat43":
            assert d.df == d.n - len(d.parameters)
        assert set(d.start1) == set(d.start2) == set(d.certified) == set(d.parameters)
    # spot-check the parser against values read directly from the Misra1a header
    m = load_nls("Misra1a")
    assert m.certified == {"b1": 2.3894212918e02, "b2": 5.5015643181e-04}
    assert m.start1 == {"b1": 500.0, "b2": 1e-4} and m.start2 == {"b1": 250.0, "b2": 5e-4}
    assert m.residual_ss == 1.2455138894e-01 and (m.y[0], m.x[0]) == (10.07, 77.6)
    t = load_nls("Thurber")
    assert t.n == 37 and len(t.parameters) == 7 and t.certified["b7"] == 4.9727297349e-02


@pytest.mark.parametrize(("name", "start"), CASES, ids=[f"{n}-start{s}" for n, s in CASES])
def test_nist_certified_values(name, start):
    d = load_nls(name)
    res = fit_curve(d.x, d.y, "user_defined", equation=d.equation, initial=d.start(start))
    conv = res.extra["convergence"]
    assert conv["converged"], f"{name} start {start}: {conv['message']}"
    assert not any("did NOT converge" in w for w in res.warnings)
    gof = res.extra["goodness_of_fit"]
    df = d.n - len(d.parameters)
    assert gof["df"] == df and gof["n"] == d.n and res.df == df
    if name != "Rat43":
        # The Rat43 header prints "Degrees of Freedom: 9" for 15 observations and 4
        # parameters; its certified residual SD equals sqrt(RSS / 11), so the header line
        # is a misprint and 11 is used.
        assert d.df == df
    limited = name in ROUNDING_LIMITED
    for key in d.parameters:
        par = res.parameter(key)
        assert par.value == pytest.approx(d.certified[key], rel=PARAM_RTOL), key
        assert par.se == pytest.approx(d.certified_sd[key], rel=0.05 if limited else SE_RTOL), key
        # asymptotic CI = estimate +/- t * SE
        assert par.ci_low < par.value < par.ci_high
    assert gof["ss"] == pytest.approx(d.residual_ss, rel=0.05 if limited else SS_RTOL)
    assert gof["sy_x"] == pytest.approx(d.residual_sd, rel=0.05 if limited else 1e-8)
    assert res.extra["model"] == "user_defined"
    assert len(res.extra["curve"]["x"]) == 200


def _digits(value, certified):
    if value == certified:
        return 11.0
    return min(11.0, -math.log10(abs(value - certified) / abs(certified)))


def test_report_digits_of_agreement():
    """Summary table (shown with ``pytest -s``): worst-case digits per dataset and start."""
    rows = []
    for name in NLS_MODELS:
        d = load_nls(name)
        row = [f"{name:9s} {d.difficulty:7s}"]
        for start in (1, 2):
            res = fit_curve(d.x, d.y, "user_defined", equation=d.equation, initial=d.start(start))
            par = min(_digits(res.parameter(k).value, d.certified[k]) for k in d.parameters)
            se = min(_digits(res.parameter(k).se, d.certified_sd[k]) for k in d.parameters)
            row.append(f"start {start}: parameters {par:4.1f}  SE {se:4.1f}")
            assert par >= 6.0
        rows.append("   ".join(row))
    print("\n".join(rows))


# ----------------------------------------- built-in models on NIST data (no starting values)
def test_misra1a_with_builtin_association_model_and_automatic_start():
    d = load_nls("Misra1a")       # y = b1 * (1 - exp(-b2 x)) = association with Y0 = 0
    res = fit_curve(d.x, d.y, "one_phase_association", fixed={"Y0": 0})
    assert res.parameter("Plateau").value == pytest.approx(d.certified["b1"], rel=1e-8)
    assert res.parameter("K").value == pytest.approx(d.certified["b2"], rel=1e-8)
    assert res.parameter("Plateau").se == pytest.approx(d.certified_sd["b1"], rel=1e-6)
    assert res.parameter("K").se == pytest.approx(d.certified_sd["b2"], rel=1e-6)
    assert res.extra["goodness_of_fit"]["ss"] == pytest.approx(d.residual_ss, rel=1e-8)
    assert res.parameter("HalfLife").value == pytest.approx(math.log(2) / d.certified["b2"], rel=1e-8)


def test_boxbod_higher_difficulty_with_builtin_model_and_automatic_start():
    d = load_nls("BoxBOD")
    res = fit_curve(d.x, d.y, "one_phase_association", fixed="Y0=0")
    assert res.parameter("Plateau").value == pytest.approx(d.certified["b1"], rel=1e-7)
    assert res.parameter("K").value == pytest.approx(d.certified["b2"], rel=1e-7)
    assert res.extra["goodness_of_fit"]["ss"] == pytest.approx(d.residual_ss, rel=1e-8)


def test_eckerle4_with_builtin_gaussian_and_automatic_start():
    d = load_nls("Eckerle4")      # y = (b1 / b2) exp(-0.5 ((x - b3) / b2)^2)
    res = fit_curve(d.x, d.y, "gaussian")
    assert res.parameter("SD").value == pytest.approx(d.certified["b2"], rel=1e-7)
    assert res.parameter("Mean").value == pytest.approx(d.certified["b3"], rel=1e-9)
    assert res.parameter("Amplitude").value == pytest.approx(
        d.certified["b1"] / d.certified["b2"], rel=1e-7)
    assert res.parameter("Mean").se == pytest.approx(d.certified_sd["b3"], rel=1e-5)
    assert res.extra["goodness_of_fit"]["ss"] == pytest.approx(d.residual_ss, rel=1e-8)


def test_misra1d_is_michaelis_menten_reparameterised():
    d = load_nls("Misra1d")       # y = b1 b2 x / (1 + b2 x) = Vmax x / (Km + x), Km = 1 / b2
    res = fit_curve(d.x, d.y, "michaelis_menten")
    assert res.parameter("Vmax").value == pytest.approx(d.certified["b1"], rel=1e-8)
    assert res.parameter("Km").value == pytest.approx(1.0 / d.certified["b2"], rel=1e-8)
    assert res.parameter("Vmax").se == pytest.approx(d.certified_sd["b1"], rel=1e-6)
    assert res.extra["goodness_of_fit"]["ss"] == pytest.approx(d.residual_ss, rel=1e-8)


def test_danwood_profile_interval_brackets_the_asymptotic_one_sensibly():
    d = load_nls("DanWood")
    res = fit_curve(d.x, d.y, "user_defined", equation=d.equation, initial=d.start(2),
                    ci_method="profile")
    asym = fit_curve(d.x, d.y, "user_defined", equation=d.equation, initial=d.start(2))
    for key in d.parameters:
        p, a = res.parameter(key), asym.parameter(key)
        assert p.value == pytest.approx(d.certified[key], rel=1e-8)
        assert p.ci_low < p.value < p.ci_high
        # well-determined, nearly linear problem: the two interval types are close
        assert p.ci_low == pytest.approx(a.ci_low, rel=0.02)
        assert p.ci_high == pytest.approx(a.ci_high, rel=0.02)
    assert np.isfinite(res.parameter("b1").se)

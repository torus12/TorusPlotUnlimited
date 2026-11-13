"""Safe user-defined equation parser.

Checks: correct evaluation against plain numpy, exact derivatives against central
differences, parameter discovery, and - most importantly - that hostile or malformed text
is rejected with a StatsError and never executed.
"""

import builtins
import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from torusplot.stats._utils import StatsError
from torusplot.stats.equation import CONSTANTS, FUNCTIONS, Equation, parse_equation

X = np.array([0.25, 0.5, 1.0, 2.0, 4.0, 8.0])


def numeric_jacobian(eq: Equation, x, params, h=1e-6):
    cols = []
    for i in range(len(params)):
        up = list(params)
        dn = list(params)
        step = h * max(1.0, abs(params[i]))
        up[i] += step
        dn[i] -= step
        cols.append((eq.func(x, *up) - eq.func(x, *dn)) / (2 * step))
    if not cols:
        return np.zeros((np.size(x), 0))
    return np.column_stack(cols)


# ------------------------------------------------------------------- evaluation
def test_four_parameter_logistic_from_the_brief():
    eq = parse_equation("Y = Bottom + (Top - Bottom) / (1 + (EC50 / X)^HillSlope)")
    assert eq.parameters == ["Bottom", "Top", "EC50", "HillSlope"]
    assert eq.text == "Y = Bottom + (Top - Bottom) / (1 + (EC50 / X)^HillSlope)"
    assert eq.n_parameters == 4 and eq.uses_x
    got = eq.func(X, 2.0, 98.0, 1.5, 1.3)
    want = 2.0 + (98.0 - 2.0) / (1 + (1.5 / X) ** 1.3)
    np.testing.assert_allclose(got, want, rtol=1e-14)
    # at X = EC50 the response is exactly half way
    assert eq.func([1.5], 2.0, 98.0, 1.5, 1.3)[0] == pytest.approx(50.0, rel=1e-14)
    assert "Equation(" in repr(eq)


@pytest.mark.parametrize(
    ("text", "func"),
    [
        ("A*exp(-K*x)", lambda x, a, k: a * np.exp(-k * x)),
        ("y = A * EXP(-K * X) + C", lambda x, a, k, c: a * np.exp(-k * x) + c),
        ("f(x) = A / (1 + (x/K)**n)", lambda x, a, k, n: a / (1 + (x / k) ** n)),
        ("-x^2 + B", lambda x, b: -(x**2) + b),
        ("2^3^2 * A + x", lambda x, a: 2 ** (3**2) * a + x),
        ("A*log(x) + B*ln(x) + C*log10(x) + D*log2(x)",
         lambda x, a, b, c, d: a * np.log(x) + b * np.log(x) + c * np.log10(x) + d * np.log2(x)),
        ("sqrt(A*x) + abs(B - x)", lambda x, a, b: np.sqrt(a * x) + np.abs(b - x)),
        ("A*sin(B*x) + cos(x/C) + tan(D*x/10) + tanh(x - A)",
         lambda x, a, b, c, d: a * np.sin(b * x) + np.cos(x / c) + np.tan(d * x / 10) + np.tanh(x - a)),
        ("sinh(A*x/10) + cosh(B*x/10) + atan(x - C) + arctan(C)",
         lambda x, a, b, c: np.sinh(a * x / 10) + np.cosh(b * x / 10) + np.arctan(x - c) + np.arctan(c)),
        ("min(A*x, B) + max(x, C, 2)",
         lambda x, a, b, c: np.minimum(a * x, b) + np.maximum(np.maximum(x, c), 2)),
        # Euler's number is written exp(1): a bare 'e' is a parameter (audit defect B7; this
        # case used to rely on 'e' being silently replaced by 2.718...)
        ("pow(x, P) * pi + exp(1)*Q", lambda x, p, q: x**p * math.pi + math.e * q),
        ("a + b*x + c*x^2 + e*x^3", lambda x, a, b, c, e: a + b * x + c * x**2 + e * x**3),
        ("+A - -B*x", lambda x, a, b: a + b * x),
        ("A*x + 1e-3*B + .5 + 5.", lambda x, a, b: a * x + 1e-3 * b + 5.5),
        ("A − B×x", lambda x, a, b: a - b * x),  # unicode minus and multiplication sign
        ("A +\n B*x", lambda x, a, b: a + b * x),
    ],
)
def test_matches_numpy_and_derivatives_match_finite_differences(text, func):
    eq = parse_equation(text)
    params = [1.7, 0.9, 2.3, 1.4][: eq.n_parameters]
    np.testing.assert_allclose(eq.func(X, *params), func(X, *params), rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(eq(X, *params), eq.func(X, *params))
    jac = eq.jacobian(X, *params)
    assert jac.shape == (X.size, eq.n_parameters)
    np.testing.assert_allclose(jac, numeric_jacobian(eq, X, params), rtol=2e-6, atol=2e-7)


def test_power_rule_branches():
    # base depends on parameters, exponent constant; exponent depends, base constant; both
    for text in ("(A*x)^3", "2^(A*x)", "(A*x)^(B*x)", "x^B", "(A + x)^(1/B)"):
        eq = parse_equation(text)
        params = [1.3, 0.7][: eq.n_parameters]
        np.testing.assert_allclose(eq.jacobian(X, *params), numeric_jacobian(eq, X, params),
                                   rtol=2e-6, atol=1e-7)
    # d/dB of 0^B is 0 for B > 0 (not NaN)
    eq = parse_equation("A*x^B")
    jac = eq.jacobian([0.0, 1.0, 2.0], 3.0, 1.5)
    assert np.all(np.isfinite(jac)) and jac[0, 1] == 0.0


def test_parameters_ordered_by_first_appearance_and_case_rules():
    eq = parse_equation("b1 + b3*x + b2*X + b3")
    assert eq.parameters == ["b1", "b3", "b2"]
    # E and e are both ordinary parameters (a bare 'e' is NOT Euler's number: audit defect
    # B7, a quartic 'a + ... + e*x^4' was silently fitted with e frozen at 2.718; the old
    # assertion here, parameters == ["E"], encoded that defect). Pi/PI are the constant.
    eq2 = parse_equation("E*x + e + PI + Pi")
    assert eq2.parameters == ["E", "e"]
    assert eq2.func([0.0], 5.0, 7.0)[0] == pytest.approx(7.0 + 2 * math.pi)
    assert CONSTANTS == {"pi": math.pi} and "exp" in FUNCTIONS
    # the user is told how the ambiguous names were read
    assert any("'e'" in n and "Euler" in n and "exp(1)" in n for n in eq2.notes)
    assert any("3.14159" in n for n in eq2.notes)
    assert parse_equation("A*exp(-K*x)").notes == []
    # Euler's number remains available, and scientific notation is still a number
    eq3 = parse_equation("A*exp(1) + 1e3 + 2.5E-1")
    assert eq3.parameters == ["A"] and eq3.notes == []
    assert eq3.func([0.0], 2.0)[0] == pytest.approx(2 * math.e + 1000.25)


def test_no_x_and_no_parameters():
    eq = parse_equation("A + B")
    assert not eq.uses_x
    np.testing.assert_allclose(eq.func(X, 1.0, 2.0), np.full(X.size, 3.0))
    np.testing.assert_allclose(eq.jacobian(X, 1.0, 2.0), np.ones((X.size, 2)))
    const = parse_equation("2*x + 1")
    assert const.parameters == []
    np.testing.assert_allclose(const.func(X), 2 * X + 1)
    assert const.jacobian(X).shape == (X.size, 0)


def test_parameter_values_as_dict_and_wrong_counts():
    eq = parse_equation("A*x + B")
    np.testing.assert_allclose(eq.func(X, {"B": 1.0, "A": 2.0}), 2 * X + 1)
    with pytest.raises(StatsError, match="No value given"):
        eq.func(X, {"A": 2.0})
    with pytest.raises(StatsError, match="2 parameter"):
        eq.func(X, 1.0)
    with pytest.raises(StatsError, match="numbers"):
        eq.func(X, "a", "b")


def test_undefined_results_are_nan_or_inf_not_exceptions():
    eq = parse_equation("log(x - A) + 1/(x - B) + 0^(-1) + 9^9^9^x + sqrt(-x)")
    out = eq.func(X, 100.0, 1.0)
    assert out.shape == X.shape
    assert not np.any(np.isfinite(out))
    assert eq.jacobian(X, 100.0, 1.0).shape == (X.size, 2)
    # scalar input keeps its shape
    assert parse_equation("A*x").func(2.0, 3.0).shape == ()


# ---------------------------------------------------------------------- security
INJECTIONS = [
    "__import__('os').system('x')",
    "x.__class__",
    "(lambda: 1)()",
    "open('f')",
    "x.__class__.__mro__[1].__subclasses__()",
    "().__class__.__bases__[0]",
    "eval('1')",
    "exec('import os')",
    "compile('1', 'a', 'eval')",
    "getattr(x, 'real')",
    "globals()['x']",
    "[a for a in (1, 2)]",
    "{a: 1 for a in (1, 2)}",
    "{1, 2}",
    "x[0]",
    "x if x else 1",
    "x and A",
    "not x",
    "~x",
    "x < A",
    "x == A",
    "(x := 3)",
    "'abc'",
    "b'abc'",
    'f"{x}"',
    "x; A",
    "import os",
    "A = 3",
    "lambda: 0",
    "print(x)",
    "exp(x, *A)",
    "exp(**A)",
    "exp(x=1)",
    "max(x)",
    "exp()",
    "exp(1, 2)",
    "pow(x)",
    "exp.real(x)",
    "A.B",
    "__x__ + 1",
    "_private * x",
    "a__b + x",
    "＿＿import＿＿('os')",  # full-width underscores (NFKC normalises to __import__)
    "x @ A",
    "x // A",
    "x % A",
    "x << 2",
    "x | A",
    "x & A",
    "1j * x",
    "True + x",
    "None",
    "...",
    "exp",
    "log + 1",
    "Y + 1",
    "y*x",
    "2x",
    "x +",
    "((x)",
    "A x",
    "x, A",
    "",
    "   ",
    "Y = ",
    "x\x00",
    "await x",
    "yield x",
    "*x",
    "9" * 400 + "*x",
]


@pytest.mark.parametrize("text", INJECTIONS)
def test_injection_attempts_and_bad_syntax_are_rejected(text, monkeypatch):
    called = []

    def boom(*args, **kwargs):  # pragma: no cover - must never run
        called.append(args)
        raise AssertionError("user text was executed")

    # If the parser ever executed user text these would fire.
    monkeypatch.setattr(builtins, "open", boom)
    monkeypatch.setattr("os.system", boom)
    with pytest.raises(StatsError):
        parse_equation(text)
    assert called == []


def test_error_messages_are_helpful():
    with pytest.raises(StatsError, match="attribute access"):
        parse_equation("x.real")
    with pytest.raises(StatsError, match="square brackets"):
        parse_equation("exp[x]")
    with pytest.raises(StatsError, match="Unknown function 'foo'"):
        parse_equation("foo(x)")
    with pytest.raises(StatsError, match="is a function"):
        parse_equation("sqrt * x")
    with pytest.raises(StatsError, match="underscore"):
        parse_equation("__import__ + 1")
    with pytest.raises(StatsError, match="2\\*X, not 2X"):
        parse_equation("2x")
    with pytest.raises(StatsError, match="too long"):
        parse_equation("x+" * 1500 + "x")
    with pytest.raises(StatsError, match="nested too deeply"):
        parse_equation("+".join(["x"] * 400))
    with pytest.raises(StatsError, match="could not be understood"):
        parse_equation("(" * 300 + "x" + ")" * 300)
    with pytest.raises(StatsError, match="text"):
        parse_equation(42)
    with pytest.raises(StatsError, match="too large"):
        parse_equation("1" + "0" * 400 + " * x")
    with pytest.raises(StatsError, match="at most"):
        parse_equation(" + ".join(f"p{i}*x" for i in range(40)))
    with pytest.raises(StatsError, match="left of the '='"):
        parse_equation("Y = Y + x")


def test_module_never_calls_eval_or_exec():
    import inspect

    import torusplot.stats.equation as module

    source = inspect.getsource(module)
    code_lines = [line.split("#")[0] for line in source.splitlines()]
    in_doc = False
    for line in code_lines:
        if line.count('"""') == 1:
            in_doc = not in_doc
            continue
        if in_doc or '"""' in line:
            continue
        stripped = line.replace('"eval"', "").replace("mode=", "")
        assert "eval(" not in stripped
        assert "exec(" not in stripped
        assert "compile(" not in stripped.replace("re.compile(", "")
        assert "__import__" not in stripped
        assert "getattr(" not in stripped


# ----------------------------------------------------------------------- fuzzing
_ALPHABET = st.sampled_from(list("xXyAbK0123456789.+-*/^()[]{}_,=:;'\"\\ \n\teEpiloagsnqrt<>!@#$%&|~`"))


@settings(max_examples=300, deadline=None)
@given(st.text(alphabet=_ALPHABET, max_size=40))
def test_fuzz_random_text_only_raises_stats_error(text):
    try:
        eq = parse_equation(text)
    except StatsError:
        return
    # anything accepted must evaluate without raising, whatever the parameter values
    params = [1.5] * eq.n_parameters
    out = eq.func(X, *params)
    assert out.shape == X.shape
    assert eq.jacobian(X, *params).shape == (X.size, eq.n_parameters)


@settings(max_examples=60, deadline=None)
@given(st.text(max_size=30))
def test_fuzz_arbitrary_unicode_only_raises_stats_error(text):
    try:
        eq = parse_equation(text)
    except StatsError:
        return
    assert all(not name.startswith("_") for name in eq.parameters)
    eq.func(X, *([0.5] * eq.n_parameters))


# "exp(1)" replaces the former leaf "e": a bare 'e' is now a parameter (audit defect B7)
_LEAF = st.sampled_from(["x", "A", "B", "K", "2", "0.5", "pi", "exp(1)", "1e3"])


def _combine(children):
    binary = st.tuples(children, st.sampled_from(["+", "-", "*", "/", "^"]), children).map(
        lambda t: f"({t[0]} {t[1]} {t[2]})")
    unary = st.tuples(st.sampled_from(sorted(n for n, k in FUNCTIONS.items() if k == 1)),
                      children).map(lambda t: f"{t[0]}({t[1]})")
    nary = st.tuples(st.sampled_from(["min", "max", "pow"]), children, children).map(
        lambda t: f"{t[0]}({t[1]}, {t[2]})")
    neg = children.map(lambda c: f"-{c}")
    return st.one_of(binary, unary, nary, neg)


@settings(max_examples=150, deadline=None)
@given(st.recursive(_LEAF, _combine, max_leaves=12),
       st.lists(st.floats(-5, 5, allow_nan=False), min_size=3, max_size=3))
def test_fuzz_valid_expressions_always_parse_and_evaluate(text, values):
    eq = parse_equation(text)
    assert set(eq.parameters) <= {"A", "B", "K"}
    params = values[: eq.n_parameters]
    out = eq.func(X, *params)
    jac = eq.jacobian(X, *params)
    assert out.shape == X.shape and jac.shape == (X.size, eq.n_parameters)
    # where everything is finite and well away from kinks the AD derivative is right
    if eq.n_parameters and np.all(np.isfinite(out)) and np.all(np.isfinite(jac)) \
            and np.max(np.abs(out)) < 1e6 \
            and not any(tok in text for tok in ("abs", "min", "max", "tan(", "sqrt", "log", "ln", "^", "pow", "/")):
        num = numeric_jacobian(eq, X, params)
        if np.all(np.isfinite(num)):
            np.testing.assert_allclose(jac, num, rtol=1e-4, atol=1e-4 * (1 + np.max(np.abs(jac))))

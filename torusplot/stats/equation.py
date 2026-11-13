"""Safe parser for user-defined curve equations.

``parse_equation("Y = Bottom + (Top - Bottom) / (1 + (EC50 / X)^HillSlope)")`` returns an
:class:`Equation` whose ``.func(x, *params)`` evaluates the right-hand side with numpy and
whose ``.jacobian(x, *params)`` returns the exact partial derivatives with respect to every
parameter (forward-mode automatic differentiation over the same expression tree; Wengert
1964, Commun. ACM 7:463-464; Griewank & Walther 2008, *Evaluating Derivatives*, SIAM).

Security model
--------------
User text is **never executed**. It is turned into a syntax tree with :func:`ast.parse`
(parsing only, no code object is ever run) and that tree is translated, node by node, into
a tiny private expression format that can hold nothing except

* numeric literals and the constant ``pi``,
* the independent variable ``X``,
* parameter names,
* ``+ - * / **`` and unary ``+ -``,
* calls to a fixed whitelist of mathematical functions.

Everything else (attribute access, subscripts, lambdas, comprehensions, strings, keywords,
comparisons, names starting with an underscore, ...) is rejected with a
:class:`~torusplot.stats._utils.StatsError`. Evaluation walks the private format with numpy;
there is no ``eval``, ``exec`` or ``compile`` of user text anywhere.

Names
-----
``X`` / ``x`` is the independent variable and ``pi`` (any capitalisation) is the constant
3.14159... Every other name is a **parameter to fit**. That deliberately includes a lone
``e`` (and ``E``): people who want the exponential function write ``exp(X)``, whereas a bare
``e`` almost always is the fifth coefficient of ``a + b*x + c*x^2 + d*x^3 + e*x^4``. Reading
it as Euler's number would silently freeze that coefficient at 2.718 and fit the wrong
model. Euler's number is available as ``exp(1)``. Numbers in scientific notation such as
``1e3`` or ``2.5E-4`` are ordinary numeric literals and are not affected. Whenever ``e`` is
fitted as a parameter, or ``pi`` is read as the constant, :attr:`Equation.notes` says so and
the fitting functions pass that note on to the user.
"""

from __future__ import annotations

import ast
import math
import re
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from ._utils import StatsError

__all__ = ["CONSTANTS", "FUNCTIONS", "Equation", "parse_equation"]

#: Longest equation text accepted (characters).
MAX_LENGTH = 2000
#: Deepest nesting accepted (protects the recursive evaluator).
MAX_DEPTH = 150
#: Most parameters accepted in one equation.
MAX_PARAMETERS = 30

_LN10 = math.log(10.0)
_LN2 = math.log(2.0)

# name -> (value function, derivative function f'(a, f(a)))
_UNARY: dict[str, tuple[Callable[[Any], Any], Callable[[Any, Any], Any]]] = {
    "exp": (np.exp, lambda a, v: v),
    "log": (np.log, lambda a, v: 1.0 / a),
    "ln": (np.log, lambda a, v: 1.0 / a),
    "log10": (np.log10, lambda a, v: 1.0 / (a * _LN10)),
    "log2": (np.log2, lambda a, v: 1.0 / (a * _LN2)),
    "sqrt": (np.sqrt, lambda a, v: 0.5 / v),
    "abs": (np.abs, lambda a, v: np.sign(a)),
    "sin": (np.sin, lambda a, v: np.cos(a)),
    "cos": (np.cos, lambda a, v: -np.sin(a)),
    "tan": (np.tan, lambda a, v: 1.0 + v * v),
    "tanh": (np.tanh, lambda a, v: 1.0 - v * v),
    "sinh": (np.sinh, lambda a, v: np.cosh(a)),
    "cosh": (np.cosh, lambda a, v: np.sinh(a)),
    "atan": (np.arctan, lambda a, v: 1.0 / (1.0 + a * a)),
    "arctan": (np.arctan, lambda a, v: 1.0 / (1.0 + a * a)),
}

#: Whitelisted function names (lower case) and the number of arguments they take
#: (``None`` = two or more).
FUNCTIONS: dict[str, int | None] = {
    **dict.fromkeys(_UNARY, 1),
    "pow": 2,
    "min": None,
    "max": None,
}

#: Named constants (matched without regard to case). A bare ``e`` is NOT a constant: it is a
#: parameter name like any other (write ``exp(1)`` for Euler's number); see the module docstring.
CONSTANTS: dict[str, float] = {"pi": math.pi}

_NOTE_E = (
    "'{name}' in the equation was fitted as a parameter, not read as Euler's number "
    "(2.718...). If you meant the constant, write exp(1), or exp(X) for an exponential."
)
_NOTE_PI = "'{name}' in the equation was read as the constant 3.14159..., not as a parameter to fit."

_LHS = re.compile(r"^\s*(?:y|f\s*\(\s*x\s*\))\s*=(?!=)", re.IGNORECASE)
_REPLACEMENTS = {
    "−": "-",  # minus sign
    "–": "-",  # en dash
    "×": "*",  # multiplication sign
    "·": "*",  # middle dot
    "∗": "*",  # asterisk operator
    "÷": "/",  # division sign
}

_BINOPS = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.Pow: "^"}

_REJECT_HINTS: dict[type, str] = {
    ast.Attribute: "attribute access with '.' (use plain names such as Vmax)",
    ast.Subscript: "square brackets (use round brackets for functions and grouping)",
    ast.Lambda: "lambda expressions",
    ast.ListComp: "comprehensions",
    ast.SetComp: "comprehensions",
    ast.DictComp: "comprehensions",
    ast.GeneratorExp: "comprehensions",
    ast.IfExp: "'if ... else' conditions",
    ast.Compare: "comparisons (<, >, ==)",
    ast.BoolOp: "'and' / 'or'",
    ast.List: "lists",
    ast.Tuple: "commas outside a function call",
    ast.Dict: "curly brackets",
    ast.Set: "curly brackets",
    ast.JoinedStr: "text strings",
    ast.NamedExpr: "assignments",
    ast.Starred: "the '*' unpacking operator",
    ast.Await: "'await'",
    ast.Yield: "'yield'",
    ast.YieldFrom: "'yield'",
}


class Equation:
    """A parsed, safe-to-evaluate equation ``Y = f(X; parameters)``.

    Attributes
    ----------
    text:
        The equation as displayed, always starting with ``Y =``.
    expression:
        The right-hand side as typed by the user.
    parameters:
        Parameter names in order of first appearance (left to right).
    uses_x:
        ``False`` when ``X`` does not appear on the right-hand side.
    notes:
        Plain-English remarks about how names were read that a user could otherwise get
        wrong: a lone ``e`` is a parameter (not Euler's number), ``pi`` is the constant.
    """

    __slots__ = ("_tree", "expression", "notes", "parameters", "text", "uses_x")

    def __init__(self, expression: str, tree: tuple, parameters: list[str], uses_x: bool,
                 notes: Sequence[str] = ()) -> None:
        self.expression = expression
        self.text = f"Y = {expression}"
        self.parameters = list(parameters)
        self.uses_x = uses_x
        self.notes = list(notes)
        self._tree = tree

    def __repr__(self) -> str:
        return f"Equation({self.text!r}, parameters={self.parameters})"

    @property
    def n_parameters(self) -> int:
        return len(self.parameters)

    def _values(self, params: Sequence[float]) -> list[np.float64]:
        if len(params) == 1 and isinstance(params[0], dict):
            mapping = params[0]
            missing = [name for name in self.parameters if name not in mapping]
            if missing:
                raise StatsError(f"No value given for parameter(s): {', '.join(missing)}.")
            params = [mapping[name] for name in self.parameters]
        if len(params) != len(self.parameters):
            raise StatsError(
                f"The equation has {len(self.parameters)} parameter(s) "
                f"({', '.join(self.parameters) or 'none'}) but {len(params)} value(s) were given."
            )
        try:
            return [np.float64(v) for v in params]
        except (TypeError, ValueError) as exc:
            raise StatsError("Parameter values must be numbers.") from exc

    def func(self, x: Any, *params: Any) -> np.ndarray:
        """Evaluate the equation at ``x`` (numpy-vectorised).

        Parameter values are given in the order of :attr:`parameters`, or as one dict.
        Undefined results (log of a negative number, division by zero...) come back as
        NaN / inf rather than raising.
        """
        xv = np.asarray(x, dtype=float)
        values = self._values(params)
        with np.errstate(all="ignore"):
            out = _evaluate(self._tree, xv, values)
        return np.array(np.broadcast_to(np.asarray(out, dtype=float), xv.shape), dtype=float)

    __call__ = func

    def jacobian(self, x: Any, *params: Any) -> np.ndarray:
        """Exact partial derivatives dY/d(parameter): array of shape ``(len(x), n_parameters)``."""
        xv = np.asarray(x, dtype=float).ravel()
        values = self._values(params)
        k = len(values)
        with np.errstate(all="ignore"):
            _, der = _evaluate_dual(self._tree, xv, values, k)
        if der is None:
            return np.zeros((xv.size, k))
        return np.array(np.broadcast_to(np.asarray(der, dtype=float), (k, xv.size)).T, dtype=float)


# ------------------------------------------------------------------------- parsing
def parse_equation(text: str) -> Equation:
    """Parse a user-typed equation into a safe :class:`Equation`.

    Accepted syntax: numbers, ``X`` (or ``x``) as the independent variable, parameter
    names, ``+ - * /``, ``^`` or ``**`` for powers, round brackets, the functions
    ``exp, log (natural), ln, log10, log2, sqrt, abs, sin, cos, tan, tanh, sinh, cosh,
    atan, min, max, pow`` and the constant ``pi``. An optional leading ``Y =`` is ignored.

    Every other name is a parameter to fit, **including a lone** ``e``: Euler's number is
    written ``exp(1)`` (and the exponential function ``exp(X)``). The returned equation's
    ``notes`` say so whenever ``e`` is used as a parameter or ``pi`` as the constant.

    Raises
    ------
    StatsError
        For anything else, with a message that says what is not allowed.
    """
    if not isinstance(text, str):
        raise StatsError("The equation must be given as text, for example 'Y = A * exp(-K * X)'.")
    if len(text) > MAX_LENGTH:
        raise StatsError(f"The equation is too long (more than {MAX_LENGTH} characters).")
    expression = _LHS.sub("", text, count=1).strip()
    if not expression:
        raise StatsError("The equation is empty. Type something like 'Y = A * exp(-K * X)'.")
    if "\x00" in expression:
        raise StatsError("The equation contains an invalid character.")
    source = expression
    for old, new in _REPLACEMENTS.items():
        source = source.replace(old, new)
    source = source.replace("^", "**")
    # Join lines: an equation may be wrapped over several lines.
    source = " ".join(source.split())
    try:
        parsed = ast.parse(source, mode="eval")
    except (SyntaxError, ValueError, RecursionError, MemoryError, OverflowError) as exc:
        raise StatsError(
            f"The equation could not be understood ({_syntax_hint(exc)}). Use numbers, X, "
            "parameter names, + - * / ^, round brackets and functions such as exp(), log(), "
            "sqrt(). Write multiplication explicitly (2*X, not 2X)."
        ) from None
    names: list[str] = []
    state: dict[str, Any] = {"uses_x": False, "notes": []}
    try:
        tree = _convert(parsed.body, names, state, 0)
    except RecursionError:  # pragma: no cover - depth is limited inside _convert
        raise StatsError("The equation is nested too deeply to evaluate.") from None
    if len(names) > MAX_PARAMETERS:
        raise StatsError(
            f"The equation has {len(names)} parameters; at most {MAX_PARAMETERS} are supported."
        )
    return Equation(expression, tree, names, state["uses_x"], state["notes"])


def _syntax_hint(exc: BaseException) -> str:
    if isinstance(exc, SyntaxError) and exc.msg:
        where = f" near position {exc.offset}" if exc.offset else ""
        return f"{exc.msg}{where}"
    return type(exc).__name__


def _convert(node: ast.AST, names: list[str], state: dict[str, Any], depth: int) -> tuple:
    """Translate one ``ast`` node into the private whitelisted tuple format."""
    if depth > MAX_DEPTH:
        raise StatsError("The equation is nested too deeply; please simplify it.")
    if isinstance(node, ast.Constant):
        value = node.value
        if isinstance(value, bool) or not isinstance(value, int | float):
            what = "text strings" if isinstance(value, str | bytes) else f"values such as {value!r}"
            raise StatsError(f"Equations may only contain numbers; {what} are not allowed.")
        try:
            number = float(value)
        except OverflowError:
            raise StatsError("A number in the equation is too large.") from None
        return ("num", np.float64(number))
    if isinstance(node, ast.Name):
        return _convert_name(node.id, names, state)
    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.USub):
            return ("neg", _convert(node.operand, names, state, depth + 1))
        if isinstance(node.op, ast.UAdd):
            return _convert(node.operand, names, state, depth + 1)
        raise StatsError("Only + and - may be used as signs in an equation ('not' and '~' are not allowed).")
    if isinstance(node, ast.BinOp):
        op = _BINOPS.get(type(node.op))
        if op is None:
            raise StatsError(
                "Only + - * / and ^ (power) are allowed as operators in an equation "
                f"(found '{type(node.op).__name__}')."
            )
        left = _convert(node.left, names, state, depth + 1)
        right = _convert(node.right, names, state, depth + 1)
        return ("bin", op, left, right)
    if isinstance(node, ast.Call):
        return _convert_call(node, names, state, depth)
    hint = _REJECT_HINTS.get(type(node), f"'{type(node).__name__}' expressions")
    raise StatsError(f"Equations cannot contain {hint}.")


def _remark(state: dict[str, Any], text: str) -> None:
    notes = state.setdefault("notes", [])
    if text not in notes:
        notes.append(text)


def _convert_name(name: str, names: list[str], state: dict[str, Any]) -> tuple:
    if name.startswith("_") or "__" in name:
        raise StatsError(
            f"'{name}' is not a valid parameter name: names may not start with an underscore "
            "or contain a double underscore."
        )
    if name in ("x", "X"):
        state["uses_x"] = True
        return ("x",)
    if name in ("y", "Y"):
        raise StatsError("Y may only appear on the left of the '=' sign, not inside the equation.")
    if name.lower() in CONSTANTS:
        _remark(state, _NOTE_PI.format(name=name))
        return ("num", np.float64(CONSTANTS[name.lower()]))
    if name == "e":  # a parameter like any other, but say so: some users expect 2.718...
        _remark(state, _NOTE_E.format(name=name))
    if name.lower() in FUNCTIONS:
        raise StatsError(
            f"'{name}' is a function: write it with brackets, e.g. {name.lower()}(X). "
            "It cannot be used as a parameter name."
        )
    if name not in names:
        names.append(name)
    return ("par", names.index(name))


def _convert_call(node: ast.Call, names: list[str], state: dict[str, Any], depth: int) -> tuple:
    if not isinstance(node.func, ast.Name):
        raise StatsError(
            "Only the built-in mathematical functions can be called in an equation "
            f"({', '.join(sorted(FUNCTIONS))})."
        )
    fname = node.func.id.lower()
    if fname not in FUNCTIONS or node.func.id.startswith("_"):
        raise StatsError(
            f"Unknown function '{node.func.id}'. Available functions: {', '.join(sorted(FUNCTIONS))}."
        )
    if node.keywords:
        raise StatsError(f"{fname}() does not accept named arguments.")
    arity = FUNCTIONS[fname]
    n_args = len(node.args)
    if arity is None:
        if n_args < 2:
            raise StatsError(f"{fname}() needs at least two arguments, e.g. {fname}(A, B).")
    elif n_args != arity:
        raise StatsError(f"{fname}() takes exactly {arity} argument(s); {n_args} were given.")
    args = tuple(_convert(arg, names, state, depth + 1) for arg in node.args)
    if fname == "pow":
        return ("bin", "^", args[0], args[1])
    return ("call", fname, args)


# ---------------------------------------------------------------------- evaluation
def _evaluate(node: tuple, x: np.ndarray, p: Sequence[np.float64]) -> Any:
    kind = node[0]
    if kind == "num":
        return node[1]
    if kind == "x":
        return x
    if kind == "par":
        return p[node[1]]
    if kind == "neg":
        return np.negative(_evaluate(node[1], x, p))
    if kind == "bin":
        a = _evaluate(node[2], x, p)
        b = _evaluate(node[3], x, p)
        op = node[1]
        if op == "+":
            return np.add(a, b)
        if op == "-":
            return np.subtract(a, b)
        if op == "*":
            return np.multiply(a, b)
        if op == "/":
            return np.divide(a, b)
        return np.power(a, b)
    # kind == "call"
    fname, args = node[1], node[2]
    if fname in _UNARY:
        return _UNARY[fname][0](_evaluate(args[0], x, p))
    values = [_evaluate(arg, x, p) for arg in args]
    out = values[0]
    pick = np.minimum if fname == "min" else np.maximum
    for other in values[1:]:
        out = pick(out, other)
    return out


def _scale(factor: Any, der: Any) -> Any:
    """factor * der with the parameter axis of ``der`` in front."""
    return np.multiply(factor, der)


def _add(da: Any, db: Any, sign: float = 1.0) -> Any:
    if da is None and db is None:
        return None
    if db is None:
        return da
    if da is None:
        return db if sign > 0 else np.negative(db)
    return np.add(da, db) if sign > 0 else np.subtract(da, db)


def _evaluate_dual(node: tuple, x: np.ndarray, p: Sequence[np.float64], k: int) -> tuple[Any, Any]:
    """Return (value, derivative). The derivative is ``None`` (no dependence on any
    parameter) or an array broadcastable to shape ``(k, len(x))``."""
    kind = node[0]
    if kind == "num":
        return node[1], None
    if kind == "x":
        return x, None
    if kind == "par":
        unit = np.zeros((k, 1))
        unit[node[1], 0] = 1.0
        return p[node[1]], unit
    if kind == "neg":
        a, da = _evaluate_dual(node[1], x, p, k)
        return np.negative(a), None if da is None else np.negative(da)
    if kind == "bin":
        op = node[1]
        a, da = _evaluate_dual(node[2], x, p, k)
        b, db = _evaluate_dual(node[3], x, p, k)
        if op == "+":
            return np.add(a, b), _add(da, db)
        if op == "-":
            return np.subtract(a, b), _add(da, db, -1.0)
        if op == "*":
            left = None if da is None else _scale(b, da)
            right = None if db is None else _scale(a, db)
            return np.multiply(a, b), _add(left, right)
        if op == "/":
            val = np.divide(a, b)
            left = None if da is None else _scale(np.divide(1.0, b), da)
            right = None if db is None else _scale(np.divide(val, b), db)
            return val, _add(left, right, -1.0)
        val = np.power(a, b)
        left = None
        right = None
        if da is not None:
            left = _scale(np.multiply(b, np.power(a, np.subtract(b, 1.0))), da)
        if db is not None:
            # d/db a^b = a^b ln(a); the limit is 0 where a == 0 and b > 0.
            factor = np.multiply(val, np.log(a))
            factor = np.where(np.logical_and(np.equal(a, 0.0), np.greater(b, 0.0)), 0.0, factor)
            right = _scale(factor, db)
        return val, _add(left, right)
    fname, args = node[1], node[2]
    if fname in _UNARY:
        value_fn, deriv_fn = _UNARY[fname]
        a, da = _evaluate_dual(args[0], x, p, k)
        val = value_fn(a)
        return val, None if da is None else _scale(deriv_fn(a, val), da)
    pairs = [_evaluate_dual(arg, x, p, k) for arg in args]
    out, dout = pairs[0]
    for other, dother in pairs[1:]:
        take_other = np.less(other, out) if fname == "min" else np.greater(other, out)
        new = np.where(take_other, other, out)
        if dout is None and dother is None:
            dnew = None
        else:
            d1 = 0.0 if dout is None else dout
            d2 = 0.0 if dother is None else dother
            dnew = np.where(take_other, d2, d1)
        out, dout = new, dnew
    return out, dout

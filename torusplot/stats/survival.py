"""Survival analysis: Kaplan-Meier curves, log-rank and Gehan-Breslow-Wilcoxon tests,
hazard ratios (all estimators implemented in TorusPlot from the primary formulas).

Coding of the data
------------------
``time``   time from the start of follow-up to the event or to the end of follow-up
           (any unit; must be >= 0).
``event``  1 / True  = the event (e.g. death) was observed at that time;
           0 / False = censored: the subject was still event-free when last seen.

References
----------
* Kaplan EL, Meier P (1958) Nonparametric estimation from incomplete observations.
  *J Am Stat Assoc* 53:457-481 (product-limit estimator).
* Greenwood M (1926) The natural duration of cancer. *Reports on Public Health and
  Medical Subjects* 33:1-26 (variance of the survival estimate).
* Kalbfleisch JD, Prentice RL (1980) *The Statistical Analysis of Failure Time Data*
  (log-log transformed, "exponential Greenwood", confidence limits).
* Brookmeyer R, Crowley J (1982) A confidence interval for the median survival time.
  *Biometrics* 38:29-41.
* Mantel N (1966) Evaluation of survival data and two new rank order statistics
  arising in its consideration. *Cancer Chemother Rep* 50:163-170 (log-rank test).
* Peto R, Peto J (1972) Asymptotically efficient rank invariant test procedures.
  *J R Stat Soc A* 135:185-207.
* Gehan EA (1965) A generalized Wilcoxon test for comparing arbitrarily
  singly-censored samples. *Biometrika* 52:203-223.
* Breslow N (1970) A generalized Kruskal-Wallis test for comparing K samples subject
  to unequal patterns of censorship. *Biometrika* 57:579-594.
* Tarone RE (1975) Tests for trend in life table analysis. *Biometrika* 62:679-682.
* Mantel N, Haenszel W (1959) Statistical aspects of the analysis of data from
  retrospective studies of disease. *J Natl Cancer Inst* 22:719-748; Peto R et al.
  (1977) *Br J Cancer* 35:1-39 (one-step hazard-ratio estimate exp((O - E) / V)).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from . import multiplicity
from ._utils import StatsError, check_alpha, ci_phrase, fmt, p_phrase, sig_phrase
from .registry import ParamSpec, register
from .result import AssumptionCheck, Comparison, EffectSize, Parameter, StatResult

CI_METHODS = ("log-log", "linear")
TESTS = ("logrank", "gehan", "both")
_WEIGHT_LABELS = {
    "logrank": "log-rank (Mantel-Cox) test",
    "gehan": "Gehan-Breslow-Wilcoxon test",
}
_TOL = 1e-12

CITATIONS = {
    "km": "Kaplan EL, Meier P (1958) J Am Stat Assoc 53:457-481",
    "greenwood": "Greenwood M (1926) Rep Public Health Med Subj 33:1-26",
    "bc": "Brookmeyer R, Crowley J (1982) Biometrics 38:29-41",
    "mantel": "Mantel N (1966) Cancer Chemother Rep 50:163-170",
    "peto": "Peto R, Peto J (1972) J R Stat Soc A 135:185-207",
    "gehan": "Gehan EA (1965) Biometrika 52:203-223",
    "breslow": "Breslow N (1970) Biometrika 57:579-594",
    "tarone": "Tarone RE (1975) Biometrika 62:679-682",
}


# ============================================================================ input
def _none_if_nan(value: float | None) -> float | None:
    if value is None:
        return None
    v = float(value)
    return v if math.isfinite(v) else None


def _as_list(arr: np.ndarray) -> list[float | None]:
    return [_none_if_nan(v) for v in np.asarray(arr, dtype=float).tolist()]


def _label(value: Any) -> str | None:
    """Group label as text (``1.0`` -> ``'1'``); ``None`` for a missing label."""
    if value is None:
        return None
    if isinstance(value, float | np.floating):
        if math.isnan(value):
            return None
        return str(int(value)) if float(value).is_integer() else f"{float(value):g}"
    if isinstance(value, bool | np.bool_):
        return str(bool(value))
    if isinstance(value, int | np.integer):
        return str(int(value))
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return None
    return text


def _to_float_array(values: Iterable[Any] | None, what: str) -> np.ndarray:
    if values is None:
        raise StatsError(f"No {what} values were given.")
    try:
        raw = np.asarray(values if isinstance(values, np.ndarray) else list(values))
        if raw.dtype == bool:
            return raw.astype(float).ravel()
        if raw.dtype == object:
            raw = np.array(
                [np.nan if (v is None or (isinstance(v, str) and not v.strip())) else v for v in raw.ravel()],
                dtype=object,
            )
        return np.asarray(raw, dtype=float).ravel()
    except (TypeError, ValueError):
        if what == "event":
            raise StatsError(
                "The event column must be coded with numbers: 1 = the event (e.g. death) "
                "happened, 0 = censored (still event-free when last seen). Text codes are "
                "not accepted; recode them as 1 and 0."
            ) from None
        raise StatsError(f"The {what} values must be numbers.") from None


def _clean_survival(
    time: Iterable[Any], event: Iterable[Any], group: Iterable[Any] | None = None
) -> tuple[np.ndarray, np.ndarray, list[str] | None, list[str]]:
    """Validate and return (time, event, group labels or None, notes)."""
    t = _to_float_array(time, "time")
    e = _to_float_array(event, "event")
    if t.size != e.size:
        raise StatsError(
            f"Time and event must have the same number of values (time has {t.size}, "
            f"event has {e.size})."
        )
    labels: list[str | None] | None = None
    if group is not None:
        labels = [_label(v) for v in (group.tolist() if isinstance(group, np.ndarray) else list(group))]
        if len(labels) != t.size:
            raise StatsError(
                f"The group column has {len(labels)} values but there are {t.size} survival times."
            )
    if np.isinf(t).any():
        raise StatsError("Survival times must be finite numbers (an infinite time was found).")
    keep = ~np.isnan(t) & ~np.isnan(e)
    if labels is not None:
        keep &= np.array([lab is not None for lab in labels], dtype=bool)
    notes: list[str] = []
    dropped = int((~keep).sum())
    if dropped:
        notes.append(
            f"{dropped} row(s) with a missing time, event code or group were ignored."
        )
    t, e = t[keep], e[keep]
    if labels is not None:
        labels = [lab for lab, k in zip(labels, keep) if k]
    if t.size == 0:
        raise StatsError("No complete rows: every subject needs a time and an event code (1 or 0).")
    bad = sorted({float(v) for v in e if v not in (0.0, 1.0)})
    if bad:
        shown = ", ".join(fmt(v) for v in bad[:5])
        raise StatsError(
            "The event column must contain only 1 (event occurred, e.g. death) or 0 (censored). "
            f"Other code(s) found: {shown}. Recode the column and run again."
        )
    if (t < 0).any():
        raise StatsError(
            f"Survival times cannot be negative ({int((t < 0).sum())} negative value(s) found). "
            "Time is measured from the start of follow-up."
        )
    return t, e.astype(int), labels, notes  # type: ignore[return-value]


def _check_ci(ci_level: float, ci_method: str) -> tuple[float, str]:
    if not (0 < float(ci_level) < 1):
        raise StatsError("ci_level must be between 0 and 1 (typically 0.95).")
    key = str(ci_method).lower().replace("_", "-").replace("loglog", "log-log")
    key = {"plain": "linear", "greenwood": "linear", "exponential": "log-log"}.get(key, key)
    if key not in CI_METHODS:
        raise StatsError("ci_method must be 'log-log' (recommended) or 'linear'.")
    return float(ci_level), key


# ===================================================================== Kaplan-Meier
def _km(t: np.ndarray, e: np.ndarray, ci_level: float, ci_method: str) -> dict[str, Any]:
    """Product-limit estimate from validated arrays (see :func:`kaplan_meier`)."""
    n = int(t.size)
    times = np.unique(t)
    t_sorted = np.sort(t)
    ev_sorted = np.sort(t[e == 1])
    at_risk = n - np.searchsorted(t_sorted, times, side="left")
    total_at = np.searchsorted(t_sorted, times, side="right") - np.searchsorted(t_sorted, times, side="left")
    d = np.searchsorted(ev_sorted, times, side="right") - np.searchsorted(ev_sorted, times, side="left")
    c = total_at - d

    surv = np.cumprod(1.0 - d / at_risk)
    # floating-point dust: a curve that reaches zero must be exactly zero
    surv = np.where(np.cumsum(d == at_risk) > 0, 0.0, surv)
    with np.errstate(divide="ignore", invalid="ignore"):
        gw = np.cumsum(np.where(at_risk > d, d / (at_risk * (at_risk - d).astype(float)), np.inf))
        se = np.where(surv > 0, surv * np.sqrt(gw), np.nan)
        z = float(stats.norm.ppf(0.5 + ci_level / 2.0))
        if ci_method == "log-log":
            # CI on log(-log S): S ** exp(+/- z * sqrt(gw) / |log S|)
            log_s = np.log(surv)
            theta = z * np.sqrt(gw) / np.abs(log_s)
            lo = np.power(surv, np.exp(theta))
            hi = np.power(surv, np.exp(-theta))
        else:
            lo = np.clip(surv - z * se, 0.0, 1.0)
            hi = np.clip(surv + z * se, 0.0, 1.0)
    no_event_yet = surv >= 1.0
    lo = np.where(no_event_yet, 1.0, lo)
    hi = np.where(no_event_yet, 1.0, hi)
    se = np.where(no_event_yet, 0.0, se)
    at_zero = surv <= 0.0
    lo = np.where(at_zero, np.nan, lo)
    hi = np.where(at_zero, np.nan, hi)

    table = [
        {
            "time": float(times[i]),
            "at_risk": int(at_risk[i]),
            "events": int(d[i]),
            "censored": int(c[i]),
            "survival": float(surv[i]),
            "se": _none_if_nan(se[i]),
            "ci_low": _none_if_nan(lo[i]),
            "ci_high": _none_if_nan(hi[i]),
        }
        for i in range(times.size)
    ]

    # ---- median (first time S(t) <= 0.5) with the Brookmeyer-Crowley interval
    notes: list[str] = []
    half = 0.5 + _TOL
    # Once the curve has reached 0 the lower band is 0 (it cannot lie above the estimate), but
    # the upper band is *undefined* there (Greenwood's variance is infinite), not 0: the upper
    # limit of the median is the first time the upper band itself drops to 50% (Brookmeyer &
    # Crowley 1982) and is "not reached" when that never happens while the band exists.
    band_lo = np.where(at_zero, 0.0, lo)
    band_hi = np.where(at_zero, np.inf, hi)

    def first_at_or_below(curve: np.ndarray) -> float | None:
        idx = np.flatnonzero(curve <= half)
        return float(times[idx[0]]) if idx.size else None

    median = first_at_or_below(surv)
    median_lo = first_at_or_below(band_lo)
    median_hi = first_at_or_below(band_hi)
    median_note = None
    if median is None:
        median_note = (
            f"Median survival was not reached: the curve never drops to 50% "
            f"(lowest estimate {fmt(float(surv[-1]) * 100.0, 3)}%)."
        )
    else:
        i = int(np.flatnonzero(surv <= half)[0])
        later = np.flatnonzero((d > 0) & (np.arange(times.size) > i))
        if abs(float(surv[i]) - 0.5) <= 1e-9 and later.size:
            nxt = float(times[later[0]])
            notes.append(
                f"The survival curve is exactly 50% from time {fmt(median)} to {fmt(nxt)}. The "
                f"median is reported as {fmt(median)} (the first time survival reaches 50%); some "
                f"programs report the midpoint, {fmt((median + nxt) / 2.0)}."
            )
        if median_hi is None:
            median_note = (
                f"The upper {ci_level * 100:g}% confidence limit of the median was not reached "
                "(the upper confidence band never drops to 50%"
                + (" before the curve reaches 0, where the band is undefined; there are too few "
                   "subjects to bound the median from above)." if bool(at_zero.any()) else ").")
            )

    curve = {
        "time": [0.0, *times.tolist()],
        "survival": [1.0, *surv.tolist()],
        "ci_low": [1.0, *_as_list(lo)],
        "ci_high": [1.0, *_as_list(hi)],
        "at_risk": [n, *[int(v) for v in at_risk]],
        "censor_time": times[c > 0].tolist(),
        "censor_survival": surv[c > 0].tolist(),
    }
    return {
        "n": n,
        "events": int(e.sum()),
        "censored": int(n - e.sum()),
        "table": table,
        "curve": curve,
        "median": median,
        "median_ci_low": median_lo,
        "median_ci_high": median_hi,
        "median_note": median_note,
        "ci_level": ci_level,
        "ci_method": ci_method,
        "notes": notes,
    }


def kaplan_meier(
    time: Iterable[Any], event: Iterable[Any], *, ci_level: float = 0.95, ci_method: str = "log-log"
) -> dict[str, Any]:
    """Kaplan-Meier (product-limit) survival estimate (Kaplan & Meier 1958).

    At each distinct observed time ``t_j`` with ``n_j`` subjects at risk (time >= t_j,
    so subjects censored at ``t_j`` still count as at risk there) and ``d_j`` events::

        S(t_j) = prod_{i <= j} (1 - d_i / n_i)
        Var S(t_j) = S(t_j)^2 * sum_{i <= j} d_i / (n_i (n_i - d_i))     (Greenwood 1926)

    Confidence limits: ``ci_method="log-log"`` (default, "exponential Greenwood":
    symmetric on log(-log S), always inside [0, 1]) or ``"linear"`` (S +/- z SE, clipped).
    Limits are ``None`` once the curve has reached 0 (Greenwood's formula is undefined).

    Median survival is the first time the curve is at or below 50%; its confidence
    interval (Brookmeyer & Crowley 1982) runs from the first time the lower band is at
    or below 50% to the first time the upper band is. ``None`` means "not reached". Where
    the curve has reached 0 the upper band is undefined (not 0), so with very few subjects
    (for example one subject, or three uncensored subjects at 95% confidence) the upper
    limit is "not reached": the data cannot bound the median from above.

    Returns a dict with ``n, events, censored, table`` (one row per distinct time:
    ``time, at_risk, events, censored, survival, se, ci_low, ci_high``), ``curve``
    (plot coordinates, one point per distinct time, starting at time 0 / survival 1,
    to be drawn as a right-continuous step function, i.e. "steps-post"),
    ``median, median_ci_low, median_ci_high, median_note, ci_level, ci_method, notes``.
    """
    ci_level, ci_method = _check_ci(ci_level, ci_method)
    t, e, _, notes = _clean_survival(time, event)
    out = _km(t, e, ci_level, ci_method)
    out["notes"] = notes + out["notes"]
    return out


def _step_value(table_times: np.ndarray, table_surv: np.ndarray, at: np.ndarray) -> np.ndarray:
    """Value of the right-continuous KM step function at the times ``at``."""
    idx = np.searchsorted(table_times, at, side="right") - 1
    return np.where(idx >= 0, table_surv[np.clip(idx, 0, None)], 1.0)


# =================================================================== log-rank family
def _risk_sets(groups: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(distinct event times, at-risk counts [k x m], event counts [k x m])."""
    all_t = np.concatenate([t for t, _ in groups])
    all_e = np.concatenate([e for _, e in groups])
    tj = np.unique(all_t[all_e == 1])
    n_ij = np.empty((len(groups), tj.size))
    d_ij = np.empty((len(groups), tj.size))
    for i, (t, e) in enumerate(groups):
        ts = np.sort(t)
        ev = np.sort(t[e == 1])
        n_ij[i] = t.size - np.searchsorted(ts, tj, side="left")
        d_ij[i] = np.searchsorted(ev, tj, side="right") - np.searchsorted(ev, tj, side="left")
    return tj, n_ij, d_ij


def _weighted_logrank(
    n_ij: np.ndarray, d_ij: np.ndarray, weighting: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Score vector U, covariance V, observed and expected event counts per group.

    At each event time the events are distributed over the groups by a (multivariate)
    hypergeometric distribution under the null hypothesis; U sums weighted
    observed-minus-expected counts and V the hypergeometric covariances.
    """
    n_j = n_ij.sum(axis=0)
    d_j = d_ij.sum(axis=0)
    w = np.ones_like(n_j) if weighting == "logrank" else n_j
    share = n_ij / n_j
    expected_ij = d_j * share
    u = (w * (d_ij - expected_ij)).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        factor = np.where(n_j > 1, d_j * (n_j - d_j) / (n_j - 1.0), 0.0) * w * w
    v = np.diag((factor * share).sum(axis=1)) - (share * factor) @ share.T
    return u, v, d_ij.sum(axis=1), expected_ij.sum(axis=1)


def _clean_groups_survival(
    groups: Mapping[str, tuple[Iterable[Any], Iterable[Any]]],
) -> tuple[list[str], list[tuple[np.ndarray, np.ndarray]], list[str]]:
    if not isinstance(groups, Mapping):
        raise TypeError("groups must be a mapping {name: (time, event)}.")
    names: list[str] = []
    data: list[tuple[np.ndarray, np.ndarray]] = []
    notes: list[str] = []
    for name, pair in groups.items():
        try:
            time, event = pair
        except (TypeError, ValueError):
            raise TypeError("Each group must be a (time, event) pair.") from None
        t, e, _, n = _clean_survival(time, event)
        names.append(str(name))
        data.append((t, e))
        notes.extend(f"{name}: {text}" for text in n)
    if len(names) < 2:
        raise StatsError("Comparing survival curves needs at least two groups.")
    return names, data, notes


def _logrank(
    names: list[str],
    data: list[tuple[np.ndarray, np.ndarray]],
    weighting: str,
    trend: bool = False,
    scores: Iterable[float] | None = None,
) -> dict[str, Any]:
    if weighting not in _WEIGHT_LABELS:
        raise StatsError("weighting must be 'logrank' or 'gehan'.")
    k = len(names)
    total_events = int(sum(int(e.sum()) for _, e in data))
    if total_events == 0:
        raise StatsError(
            "No events were observed in any group (every subject is censored), so the "
            "survival curves cannot be compared."
        )
    _, n_ij, d_ij = _risk_sets(data)
    u, v, observed, expected = _weighted_logrank(n_ij, d_ij, weighting)
    notes: list[str] = []
    scale = float(np.max(np.abs(np.diag(v)))) if v.size else 0.0
    rank = int(np.linalg.matrix_rank(v, hermitian=True)) if scale > 0 else 0
    statistic: float | None
    p_value: float | None
    if rank == 0:
        statistic, p_value, df = None, None, k - 1
        notes.append(
            "The test statistic is undefined: at every event time either all subjects at "
            "risk had the event or only one group was still being followed, so the groups "
            "cannot be compared."
        )
    else:
        statistic = float(u @ np.linalg.pinv(v, hermitian=True) @ u)
        statistic = max(statistic, 0.0)
        df = rank
        p_value = float(stats.chi2.sf(statistic, df))
        if rank < k - 1:
            notes.append(
                f"Only {rank} degree(s) of freedom could be used instead of {k - 1}: at least "
                "one group was never at risk at any event time and carries no information."
            )
    out: dict[str, Any] = {
        "test": weighting,
        "test_name": _WEIGHT_LABELS[weighting],
        "statistic": statistic,
        "df": df,
        "p_value": p_value,
        "groups": names,
        "observed": [float(x) for x in observed],
        "expected": [float(x) for x in expected],
        "score": [float(x) for x in u],
        "covariance": [[float(x) for x in row] for row in v],
        "notes": notes,
    }
    if trend:
        s = np.arange(1, k + 1, dtype=float) if scores is None else np.asarray(list(scores), dtype=float)
        if s.size != k or not np.all(np.isfinite(s)):
            raise StatsError(f"The trend test needs one numeric score per group ({k} groups).")
        var = float(s @ v @ s)
        if var > _TOL * max(scale, 1.0):
            z = float(s @ u) / math.sqrt(var)
            out["trend"] = {
                "statistic": z * z,
                "z": z,
                "df": 1,
                "p_value": float(stats.chi2.sf(z * z, 1)),
                "scores": s.tolist(),
            }
        else:
            out["trend"] = {"statistic": None, "z": None, "df": 1, "p_value": None, "scores": s.tolist()}
    return out


def logrank_test(
    groups: Mapping[str, tuple[Iterable[Any], Iterable[Any]]],
    *,
    weighting: str = "logrank",
    trend: bool = False,
    scores: Iterable[float] | None = None,
) -> dict[str, Any]:
    """Log-rank (Mantel-Cox) or Gehan-Breslow-Wilcoxon test for two or more groups.

    ``groups`` maps a group name to ``(time, event)``. At each distinct event time
    ``t_j`` with ``n_j`` subjects at risk and ``d_j`` events in total, group ``i`` (with
    ``n_ij`` at risk) is expected to have ``E_ij = d_j n_ij / n_j`` events. With weights
    ``w_j`` (1 for the log-rank test; ``n_j`` for Gehan-Breslow-Wilcoxon, which gives
    more weight to early times)::

        U_i  = sum_j w_j (d_ij - E_ij)
        V_il = sum_j w_j^2 d_j (n_j - d_j) / (n_j - 1) * (n_ij / n_j) (delta_il - n_lj / n_j)
        chi2 = U' V^- U        with k - 1 degrees of freedom for k groups

    (Mantel 1966; Peto & Peto 1972; Gehan 1965; Breslow 1970). ``trend=True`` adds the
    1-df test for trend across ordered groups (Tarone 1975):
    ``(s'U)^2 / (s'Vs)`` with group scores ``s`` (default 1, 2, ..., k).

    Returns a dict: ``test, test_name, statistic, df, p_value, groups, observed,
    expected`` (unweighted event counts), ``score, covariance, notes`` (+ ``trend``).
    """
    names, data, notes = _clean_groups_survival(groups)
    out = _logrank(names, data, str(weighting).lower(), trend=trend, scores=scores)
    out["notes"] = notes + out["notes"]
    return out


def _hazard_ratio(
    a: tuple[np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray], ci_level: float
) -> dict[str, Any]:
    _, n_ij, d_ij = _risk_sets([a, b])
    z = float(stats.norm.ppf(0.5 + ci_level / 2.0))
    out: dict[str, Any] = {"ci_level": ci_level, "notes": []}
    empty = {"hr": None, "ci_low": None, "ci_high": None, "log_hr": None, "se_log_hr": None}
    if n_ij.shape[1] == 0:
        out.update({"observed": [0.0, 0.0], "expected": [0.0, 0.0], "variance": 0.0,
                    "mantel_haenszel": dict(empty), "logrank": dict(empty)})
        out["notes"].append("No events were observed, so the hazard ratio cannot be estimated.")
        return out
    u, v, observed, expected = _weighted_logrank(n_ij, d_ij, "logrank")
    var = float(v[0, 0])
    o_a, o_b = float(observed[0]), float(observed[1])
    e_a, e_b = float(expected[0]), float(expected[1])
    out.update({"observed": [o_a, o_b], "expected": [e_a, e_b], "variance": var})

    mh = dict(empty)
    if var > _TOL:
        log_hr = float(u[0]) / var
        se = 1.0 / math.sqrt(var)
        mh = {
            "hr": _none_if_nan(_safe_exp(log_hr)),
            "ci_low": _none_if_nan(_safe_exp(log_hr - z * se)),
            "ci_high": _none_if_nan(_safe_exp(log_hr + z * se)),
            "log_hr": log_hr,
            "se_log_hr": se,
        }
    else:
        out["notes"].append(
            "The Mantel-Haenszel hazard ratio is undefined (zero variance: the groups never "
            "share an informative event time)."
        )
    out["mantel_haenszel"] = mh

    lr = dict(empty)
    if o_a > 0 and o_b > 0 and e_a > 0 and e_b > 0:
        log_hr = math.log((o_a / e_a) / (o_b / e_b))
        se = math.sqrt(1.0 / e_a + 1.0 / e_b)
        lr = {
            "hr": math.exp(log_hr),
            "ci_low": math.exp(log_hr - z * se),
            "ci_high": math.exp(log_hr + z * se),
            "log_hr": log_hr,
            "se_log_hr": se,
        }
    else:
        out["notes"].append(
            "The log-rank hazard ratio (O/E ratio) is undefined because a group has no "
            "observed or no expected events."
        )
    out["logrank"] = lr
    return out


def _safe_exp(x: float) -> float:
    try:
        return math.exp(x)
    except OverflowError:
        return math.inf


def hazard_ratio(
    time_a: Iterable[Any],
    event_a: Iterable[Any],
    time_b: Iterable[Any],
    event_b: Iterable[Any],
    *,
    ci_level: float = 0.95,
) -> dict[str, Any]:
    """Hazard ratio of group A **relative to group B** from log-rank quantities.

    With O = observed events, E = events expected under the null hypothesis and V the
    hypergeometric variance of (O_A - E_A) (see :func:`logrank_test`), two estimators
    are returned, as other statistics packages report them:

    * ``mantel_haenszel``: ``HR = exp((O_A - E_A) / V)``, CI
      ``exp((O_A - E_A) / V +/- z / sqrt(V))`` (Mantel & Haenszel 1959; Peto et al. 1977).
    * ``logrank``: ``HR = (O_A / E_A) / (O_B / E_B)``, CI
      ``exp(ln HR +/- z sqrt(1 / E_A + 1 / E_B))``.

    HR > 1 means the event happens at a higher rate (sooner) in group A than in B.
    Both assume proportional hazards. Undefined quantities are ``None``.
    """
    if not (0 < float(ci_level) < 1):
        raise StatsError("ci_level must be between 0 and 1 (typically 0.95).")
    ta, ea, _, _ = _clean_survival(time_a, event_a)
    tb, eb, _, _ = _clean_survival(time_b, event_b)
    return _hazard_ratio((ta, ea), (tb, eb), float(ci_level))


def _curves_cross(km_a: dict[str, Any], km_b: dict[str, Any]) -> tuple[bool, float, float]:
    """Do two KM step functions cross? Returns (crossed, max(A - B), max(B - A))."""
    ta = np.array([r["time"] for r in km_a["table"]])
    sa = np.array([r["survival"] for r in km_a["table"]])
    tb = np.array([r["time"] for r in km_b["table"]])
    sb = np.array([r["survival"] for r in km_b["table"]])
    end = min(ta[-1], tb[-1])
    grid = np.unique(np.concatenate([ta, tb]))
    grid = grid[grid <= end]
    diff = _step_value(ta, sa, grid) - _step_value(tb, sb, grid)
    up = float(max(diff.max(initial=0.0), 0.0))
    down = float(max((-diff).max(initial=0.0), 0.0))
    return bool(up > 1e-9 and down > 1e-9), up, down


def _trend_order(names: list[str], explicit: bool) -> tuple[list[str], str]:
    """Order of the groups for the trend test and how it was decided."""
    if explicit:
        return list(names), "the order you gave"
    try:
        values = [float(g) for g in names]
    except ValueError:
        return list(names), "the order in which the groups first appear in the data"
    if all(math.isfinite(v) for v in values) and len(set(values)) == len(values):
        ranked = [g for _, g in sorted(zip(values, names), key=lambda pair: pair[0])]
        return ranked, "ascending numeric order of the group labels"
    return list(names), "the order in which the groups first appear in the data"


def _adjust_label(adjust: str) -> str:
    """Validate a multiplicity method name and return its display label."""
    try:
        multiplicity.adjust_pvalues([0.5], str(adjust))
    except ValueError:
        raise StatsError(
            f"Unknown P value adjustment '{adjust}'. Choose from {multiplicity.METHODS}."
        ) from None
    key = str(adjust).lower()
    labels = multiplicity.METHOD_LABELS
    return labels.get(key, labels.get(key.replace("-", "_"), str(adjust)))


# ============================================================== bundled analysis
def survival_analysis(
    time: Iterable[Any],
    event: Iterable[Any],
    group: Iterable[Any] | None = None,
    *,
    alpha: float = 0.05,
    test: str = "logrank",
    ci_method: str = "log-log",
    ci_level: float = 0.95,
    adjust: str = "bonferroni",
    trend: bool = False,
    group_order: Iterable[str] | None = None,
) -> StatResult:
    """Kaplan-Meier curves per group plus a comparison of the curves.

    ``event``: 1 / True = event occurred, 0 / False = censored. ``group``: one label per
    subject (omit for a single curve). ``test``: ``"logrank"`` (Mantel-Cox; most powerful
    under proportional hazards), ``"gehan"`` (Gehan-Breslow-Wilcoxon; weights early
    events more) or ``"both"`` (log-rank is the headline result). With exactly two
    groups the hazard ratio (first group relative to the second) is reported; with three
    or more, every pair is compared and the P values adjusted with ``adjust``.
    ``trend=True`` adds the log-rank test for trend (Tarone 1975; three or more *ordered*
    groups such as increasing doses; always computed with log-rank weights and equally
    spaced scores 1, 2, ..., k) in ``extra["trend"]``, and reports it in the interpretation
    and methods text. The groups are taken in ``group_order`` when given; otherwise in
    ascending numeric order when every group label is a number (doses), else in the order
    in which they first appear in the data. With two groups the trend test is identical to
    the log-rank test and is not reported separately.
    """
    alpha = check_alpha(alpha)
    ci_level, ci_method = _check_ci(ci_level, ci_method)
    test = str(test).lower().replace("-", "").replace("_", "")
    test = {"logrank": "logrank", "mantelcox": "logrank", "gehan": "gehan", "wilcoxon": "gehan",
            "gehanbreslowwilcoxon": "gehan", "both": "both"}.get(test, test)
    if test not in TESTS:
        raise StatsError("test must be 'logrank', 'gehan' or 'both'.")
    adjust_label = _adjust_label(adjust)

    t, e, labels, notes = _clean_survival(time, event, group)
    if labels is None:
        names = ["All subjects"]
        index = {"All subjects": np.ones(t.size, dtype=bool)}
    else:
        names = list(dict.fromkeys(labels))
        if group_order is not None:
            wanted = [str(g) for g in group_order]
            unknown = [g for g in wanted if g not in names]
            if unknown or len(set(wanted)) != len(names):
                raise StatsError(
                    f"group_order must list every group exactly once. Groups in the data: {names}."
                )
            names = wanted
        lab = np.array(labels, dtype=object)
        index = {g: lab == g for g in names}
    data = {g: (t[index[g]], e[index[g]]) for g in names}
    k = len(names)
    pct = f"{ci_level * 100:g}%"

    warnings: list[str] = []
    kms = {g: _km(data[g][0], data[g][1], ci_level, ci_method) for g in names}
    descriptives: list[dict[str, Any]] = []
    parameters: list[Parameter] = []
    for g in names:
        km = kms[g]
        notes.extend(f"{g}: {text}" for text in km["notes"])
        descriptives.append({
            "group": g,
            "n": km["n"],
            "events": km["events"],
            "censored": km["censored"],
            "median": km["median"],
            "median_ci_low": km["median_ci_low"],
            "median_ci_high": km["median_ci_high"],
            "median_note": km["median_note"],
        })
        parameters.append(Parameter(
            name=f"Median survival: {g}", value=km["median"], ci_low=km["median_ci_low"],
            ci_high=km["median_ci_high"], ci_level=ci_level, unit_note=km["median_note"],
        ))
        if km["median_note"]:
            notes.append(f"{g}: {km['median_note']}")
        if km["events"] == 0:
            warnings.append(
                f"{g}: no events were observed (all {km['n']} subject(s) are censored); the curve "
                "stays at 100% and no median can be estimated."
            )
        if km["n"] < 2:
            warnings.append(f"{g} has a single subject; its curve is one step and carries almost no information.")

    extra: dict[str, Any] = {
        "curves": {g: kms[g]["curve"] for g in names},
        "tables": {g: kms[g]["table"] for g in names},
        "curve_style": "steps-post",
        "ci_method": ci_method,
        "ci_level": ci_level,
        "event_coding": "1 = event occurred, 0 = censored",
    }
    assumptions = [
        AssumptionCheck(
            name="Non-informative censoring", status="info",
            detail="Censored subjects are assumed to have the same outlook as those still followed "
                   "(they did not leave the study because they were doing better or worse). This "
                   "cannot be tested from the data.",
        ),
        AssumptionCheck(
            name="Independent subjects", status="info",
            detail="Each row must be a different subject, entered once.",
        ),
    ]
    ci_words = ("log-log transformation (exponential Greenwood)" if ci_method == "log-log"
                else "Greenwood's formula (linear)")
    citations = [CITATIONS["km"], CITATIONS["greenwood"], CITATIONS["bc"]]
    median_text = "; ".join(
        f"{g}: {fmt(kms[g]['median'])}" if kms[g]["median"] is not None else f"{g}: not reached"
        for g in names
    )

    # ------------------------------------------------------------------ one curve
    if k == 1:
        g = names[0]
        km = kms[g]
        if km["median"] is not None:
            hi_text = fmt(km["median_ci_high"]) if km["median_ci_high"] is not None else "not reached"
            med = (f"Median survival was {fmt(km['median'])} ({pct} CI {fmt(km['median_ci_low'])} "
                   f"to {hi_text}).")
        else:
            med = "Median survival was not reached (the curve stays above 50%)."
        return StatResult(
            analysis="survival",
            test_name="Kaplan-Meier survival estimate",
            why_chosen="The Kaplan-Meier method estimates the fraction of subjects still event-free "
                       "over time and uses censored subjects correctly for as long as they were followed.",
            assumptions=assumptions,
            alpha=alpha,
            parameters=parameters,
            n={g: km["n"]},
            descriptives=descriptives,
            warnings=warnings,
            data_notes=notes,
            interpretation=(
                f"A Kaplan-Meier survival curve was estimated for {km['n']} subject(s) "
                f"({km['events']} event(s), {km['censored']} censored). {med} "
                "There is only one group, so no comparison was made."
            ),
            methods_text=(
                f"Survival was estimated with the Kaplan-Meier method; {pct} confidence intervals "
                f"were computed with the {ci_words}."
            ),
            citations=citations,
            extra=extra,
        )

    # ------------------------------------------------------------- compare curves
    pairs = [data[g] for g in names]
    total_events = int(e.sum())
    primary_key = "gehan" if test == "gehan" else "logrank"
    wanted_tests = ["logrank", "gehan"] if test == "both" else [primary_key]
    tests: dict[str, dict[str, Any]] = {}
    if total_events == 0:
        warnings.append(
            "No events were observed in any group, so the survival curves cannot be compared "
            "(no test statistic or P value)."
        )
    else:
        for key in wanted_tests:
            tests[key] = _logrank(names, pairs, key)
            warnings.extend(tests[key]["notes"])
        if trend and k >= 3:
            order, how = _trend_order(names, explicit=group_order is not None)
            tr = _logrank(order, [data[g] for g in order], "logrank", trend=True)["trend"]
            extra["trend"] = {"test_name": "Log-rank test for trend", **tr, "group_order": order,
                              "order_basis": how}
            assumptions.append(AssumptionCheck(
                name="Ordered groups (trend test)", status="info",
                detail=f"The trend test treats the groups as ordered: {' < '.join(order)} ({how}), with "
                       "equally spaced scores 1, 2, 3, ... It asks whether survival gets steadily better or "
                       "worse along that order; it is meaningless if the order is arbitrary.",
            ))
            if tr["statistic"] is None:
                warnings.append("The log-rank test for trend could not be computed (no usable variation "
                                "in the event pattern across the ordered groups).")
    if trend and k < 3:
        notes.append(
            "The test for trend needs three or more ordered groups; with two groups it is identical to "
            "the log-rank test, so it was not reported separately."
        )
    primary = tests.get(primary_key)
    extra["tests"] = {
        key: {f: res[f] for f in ("test_name", "statistic", "df", "p_value", "observed", "expected")}
        for key, res in tests.items()
    }
    if primary is not None:
        for row, o, ex in zip(descriptives, primary["observed"], primary["expected"]):
            row["observed"] = o
            row["expected"] = ex

    # proportional hazards / crossing curves
    crossing = []
    for ga, gb in combinations(names, 2):
        crossed, up, down = _curves_cross(kms[ga], kms[gb])
        if crossed:
            crossing.append((ga, gb, up, down))
    if crossing:
        ga, gb, up, down = crossing[0]
        more = f" (and {len(crossing) - 1} other pair(s))" if len(crossing) > 1 else ""
        assumptions.append(AssumptionCheck(
            name="Proportional hazards", status="warn",
            detail=f"The curves of {ga} and {gb} cross{more}: {ga} is up to {fmt(up * 100, 3)} "
                   f"percentage points above and up to {fmt(down * 100, 3)} below {gb}. When curves "
                   "cross, the hazard ratio is not constant, a single hazard ratio can mislead and "
                   "the log-rank test loses power. Small crossings can be chance; judge from the graph.",
        ))
    else:
        assumptions.append(AssumptionCheck(
            name="Proportional hazards", status="info",
            detail="The log-rank test is most powerful, and the hazard ratio is meaningful, when the "
                   "ratio of event rates between groups is constant over time. The curves do not "
                   "cross, which is consistent with this; it cannot be proven from the data.",
        ))
    few = [g for g in names if kms[g]["events"] < 5]
    assumptions.append(AssumptionCheck(
        name="Enough events for the chi-square approximation",
        status="warn" if few else "pass",
        detail=(f"Fewer than 5 events in: {', '.join(few)}. The P value is a large-sample "
                "approximation and is rough with so few events."
                if few else "Every group has at least 5 events."),
    ))

    # hazard ratio (two groups) / pairwise comparisons (three or more)
    effect_sizes: list[EffectSize] = []
    comparisons: list[Comparison] = []
    family = None
    correction = None
    hr_text = ""
    if k == 2 and total_events > 0:
        a, b = names
        hr = _hazard_ratio(data[a], data[b], ci_level)
        warnings.extend(hr["notes"])
        extra["hazard_ratio"] = {"group_a": a, "group_b": b, "direction": f"{a} relative to {b}",
                                 **{key: val for key, val in hr.items() if key != "notes"}}
        for key, label in (("mantel_haenszel", "Mantel-Haenszel"), ("logrank", "log-rank")):
            est = hr[key]
            parameters.append(Parameter(
                name=f"Hazard ratio ({label}), {a} vs {b}", value=est["hr"], ci_low=est["ci_low"],
                ci_high=est["ci_high"], ci_level=ci_level,
                unit_note=f"Event rate in {a} relative to {b}; above 1 means events happen sooner in {a}.",
            ))
            parameters.append(Parameter(
                name=f"ln hazard ratio ({label}), {a} vs {b}", value=est["log_hr"],
                se=est["se_log_hr"], ci_level=ci_level,
            ))
        mh = hr["mantel_haenszel"]
        effect_sizes.append(EffectSize(
            name="Hazard ratio (Mantel-Haenszel)", value=mh["hr"], ci_low=mh["ci_low"],
            ci_high=mh["ci_high"], ci_level=ci_level, ci_method="exp((O - E) / V +/- z / sqrt(V))",
            target=f"{a} vs {b}",
        ))
        if mh["hr"] is not None:
            if mh["hr"] > 1:
                direction = f"events happened at a higher rate (sooner) in {a} than in {b}"
            elif mh["hr"] < 1:
                direction = f"events happened at a lower rate (later) in {a} than in {b}"
            else:
                direction = "the event rates were the same"
            hr_text = (
                f" The hazard ratio of {a} relative to {b} was {fmt(mh['hr'])} "
                f"({ci_phrase(mh['ci_low'], mh['ci_high'], ci_level)}; Mantel-Haenszel): {direction}, "
                "assuming the ratio is constant over time."
            )
    elif k >= 3 and total_events > 0:
        raw_p: list[float] = []
        rows: list[dict[str, Any]] = []
        for ga, gb in combinations(names, 2):
            row: dict[str, Any] = {"group_a": ga, "group_b": gb, "statistic": None, "p_value": None,
                                   "hr": None, "hr_ci_low": None, "hr_ci_high": None}
            if int(data[ga][1].sum() + data[gb][1].sum()) > 0:
                res = _logrank([ga, gb], [data[ga], data[gb]], primary_key)
                hr = _hazard_ratio(data[ga], data[gb], ci_level)["mantel_haenszel"]
                row.update(statistic=res["statistic"], p_value=res["p_value"], hr=hr["hr"],
                           hr_ci_low=hr["ci_low"], hr_ci_high=hr["ci_high"])
            rows.append(row)
            raw_p.append(math.nan if row["p_value"] is None else row["p_value"])
        adjusted = multiplicity.adjust_pvalues(raw_p, adjust)
        for row, p_adj in zip(rows, adjusted):
            row["p_adjusted"] = _none_if_nan(p_adj)
            es = EffectSize(
                name="Hazard ratio (Mantel-Haenszel)", value=row["hr"], ci_low=row["hr_ci_low"],
                ci_high=row["hr_ci_high"], ci_level=ci_level,
                ci_method="exp((O - E) / V +/- z / sqrt(V))", target=f"{row['group_a']} vs {row['group_b']}",
            )
            effect_sizes.append(es)
            comparisons.append(Comparison(
                id="", group_a=row["group_a"], group_b=row["group_b"], statistic_name="chi2",
                statistic=row["statistic"], df=1 if row["statistic"] is not None else None,
                p_value=row["p_value"], p_adjusted=row["p_adjusted"], effect_size=es,
            ))
        n_pairs = len(rows)
        family = (f"all {n_pairs} pairwise comparisons among {k} groups "
                  f"({_WEIGHT_LABELS[primary_key]} on each pair of groups)")
        correction = adjust_label
        extra["pairwise"] = rows
        if any(r["p_value"] is None for r in rows):
            warnings.append("Some pairs of groups could not be compared (no usable events in the pair).")

    # ---------------------------------------------------------------------- text
    group_parts = [f"{g} (n = {kms[g]['n']}, {kms[g]['events']} event(s))" for g in names]
    group_text = " and ".join(group_parts) if k == 2 else ", ".join(group_parts)
    test_label = _WEIGHT_LABELS[primary_key]
    statistic = primary["statistic"] if primary else None
    p_value = primary["p_value"] if primary else None
    df = primary["df"] if primary else None
    if statistic is not None:
        main = (f"The difference between the survival curves was {sig_phrase(p_value, alpha)} "
                f"(chi-square = {fmt(statistic)}, df = {df:g}).")
    else:
        main = "The curves could not be compared statistically."
    other = ""
    if test == "both" and "gehan" in tests and tests["gehan"]["statistic"] is not None:
        gz = tests["gehan"]
        other = (f" The Gehan-Breslow-Wilcoxon test, which gives more weight to early events, gave "
                 f"chi-square = {fmt(gz['statistic'])}, {p_phrase(gz['p_value'])}.")
    trend_text = ""
    if extra.get("trend", {}).get("statistic") is not None:
        tr = extra["trend"]
        trend_text = (f" The log-rank test for trend across the ordered groups ({' < '.join(tr['group_order'])}) "
                      f"was {sig_phrase(tr['p_value'], alpha)} (chi-square = {fmt(tr['statistic'])}, df = 1).")
        citations_trend = [CITATIONS["tarone"]]
    else:
        citations_trend = []
    pair_text = ""
    if comparisons:
        sig = [c for c in comparisons if c.p_adjusted is not None and c.p_adjusted < alpha]
        if sig:
            listed = "; ".join(f"{c.group_a} vs {c.group_b} ({p_phrase(c.p_adjusted)})" for c in sig[:6])
            pair_text = f" After {adjust_label} adjustment, these pairs differed: {listed}."
        else:
            pair_text = f" After {adjust_label} adjustment, no pair of groups differed significantly."
    interpretation = (
        f"Survival was compared between {group_text} with the {test_label}. {main}{other}{trend_text} "
        f"Median survival: {median_text}.{hr_text}{pair_text}"
    )

    if primary_key == "logrank":
        why = ("The log-rank test compares entire survival curves between independent groups and uses "
               "censored subjects correctly. It weights all time points equally and is the standard "
               "choice when the groups' event rates stay roughly proportional over time.")
        citations += [CITATIONS["mantel"], CITATIONS["peto"]]
        if test == "both":
            citations += [CITATIONS["gehan"], CITATIONS["breslow"]]
    else:
        why = ("The Gehan-Breslow-Wilcoxon test compares entire survival curves but gives more weight "
               "to early time points (when more subjects are at risk). Choose it when early "
               "differences matter most; it can mislead if many subjects are censored early.")
        citations += [CITATIONS["gehan"], CITATIONS["breslow"]]
    citations += citations_trend
    methods = (
        f"Survival curves were estimated with the Kaplan-Meier method ({pct} confidence intervals by "
        f"the {ci_words}) and compared with the {test_label}"
        + (" and the Gehan-Breslow-Wilcoxon test" if test == "both" else "")
        + f" (alpha = {alpha:g})."
    )
    if extra.get("trend", {}).get("statistic") is not None:
        methods += (" A log-rank test for trend (Tarone 1975) was computed across the ordered groups "
                    f"({' < '.join(extra['trend']['group_order'])}; equally spaced scores).")
    if k == 2:
        methods += (" The hazard ratio and its confidence interval were computed by the "
                    "Mantel-Haenszel and log-rank (O/E) methods.")
        citations.append("Mantel N, Haenszel W (1959) J Natl Cancer Inst 22:719-748")
    elif comparisons:
        methods += f" Pairwise comparisons used the same test with {adjust_label} adjustment."

    return StatResult(
        analysis="survival",
        test_name=test_label[0].upper() + test_label[1:],
        why_chosen=why,
        assumptions=assumptions,
        statistic_name="chi2",
        statistic=statistic,
        df=df if statistic is not None else None,
        p_value=p_value,
        alpha=alpha,
        effect_sizes=effect_sizes,
        parameters=parameters,
        n={g: kms[g]["n"] for g in names},
        descriptives=descriptives,
        comparisons=comparisons,
        comparison_family=family,
        correction=correction,
        warnings=list(dict.fromkeys(warnings)),
        data_notes=notes,
        interpretation=interpretation,
        methods_text=methods,
        citations=list(dict.fromkeys(citations)),
        extra=extra,
    )


# ===================================================================== table adapter
_TIME_WORDS = ("time", "duration", "days", "weeks", "months", "years", "follow")
_EVENT_WORDS = ("event", "status", "death", "dead", "died")
_GROUP_WORDS = ("group", "arm", "treatment", "condition", "genotype", "cohort", "strain")


def _match_column(columns: list[str], words: tuple[str, ...], taken: set[str]) -> str | None:
    for word in words:
        for col in columns:
            if col not in taken and word in col.lower():
                return col
    return None


def _pick_survival_columns(
    table: Any, time: str | None, event: str | None, group: str | None
) -> tuple[str, str, str | None, list[str]]:
    columns = list(table.data.columns)
    numeric = list(table.numeric_columns)
    notes: list[str] = []
    for role, name in (("time", time), ("event", event), ("group", group)):
        if name is not None and name not in columns:
            raise StatsError(f"The {role} column '{name}' is not in the table. Available columns: {columns}.")
    taken = {c for c in (time, event, group) if c is not None}
    if time is None:
        time = _match_column(numeric, _TIME_WORDS, taken)
        if time is None:
            time = next((c for c in numeric if c not in taken), None)
        if time is None:
            raise StatsError("No numeric column is available for the survival time. Choose the time column.")
        taken.add(time)
        notes.append(f"Time column chosen automatically: '{time}'.")
    if event is None:
        event = _match_column(numeric, _EVENT_WORDS, taken)
        if event is None:
            event = next((c for c in numeric if c not in taken), None)
        if event is None:
            raise StatsError(
                "No numeric column is available for the event code (1 = event, 0 = censored). "
                "Choose the event column."
            )
        taken.add(event)
        notes.append(f"Event column chosen automatically: '{event}' (1 = event occurred, 0 = censored).")
    if group is None:
        group = _match_column(columns, _GROUP_WORDS, taken)
        if group is None:
            for col in table.categorical_columns:
                if col in taken:
                    continue
                values = table.data[col].dropna()
                if 2 <= values.nunique() < len(values):
                    group = col
                    break
        notes.append(
            f"Group column chosen automatically: '{group}'." if group is not None
            else "No group column was found or chosen: one curve was fitted to all subjects."
        )
    if "censor" in event.lower():
        notes.append(
            f"Check the coding of '{event}': this analysis reads 1 as 'event occurred' and 0 as "
            "'censored'. If your column means the opposite (1 = censored), recode it."
        )
    return time, event, group, notes


@register(
    key="survival",
    label="Survival analysis (Kaplan-Meier, log-rank)",
    category="Survival",
    tables=("multiple_variables",),
    params=(
        ParamSpec("time", "Time column", kind="column", default=None, optional=True,
                  help="The column with each subject's follow-up time (to the event or to when they "
                       "were last seen). Leave empty to use a column named like 'time', or the first "
                       "numeric column."),
        ParamSpec("event", "Event column", kind="column", default=None, optional=True,
                  help="The column that says what happened at that time: 1 = the event (e.g. death) "
                       "occurred, 0 = censored (the subject was still event-free when last seen). "
                       "Leave empty to use a column named like 'event' or 'status'."),
        ParamSpec("group", "Group column", kind="column", default=None, optional=True,
                  help="The column that says which group each subject belongs to. Leave empty to use "
                       "a column named like 'group', or to fit one curve when there is none."),
        ParamSpec("test", "Comparison test", kind="choice", default="logrank",
                  choices=("logrank", "gehan", "both"),
                  help="Log-rank weights all time points equally (the standard choice). "
                       "Gehan-Breslow-Wilcoxon gives more weight to early events. 'both' reports the two."),
        ParamSpec("alpha", "Significance level (alpha)", kind="float", default=0.05,
                  help="The P value threshold for calling a result significant. 0.05 is the usual choice."),
        ParamSpec("ci_method", "Confidence band method", kind="choice", default="log-log",
                  choices=CI_METHODS,
                  help="How the confidence band around each curve is computed. 'log-log' always stays "
                       "between 0% and 100% and is recommended; 'linear' is the plain plus/minus version."),
        ParamSpec("adjust", "Correction for multiple comparisons", kind="choice", default="bonferroni",
                  choices=multiplicity.METHODS,
                  help="With three or more groups every pair is compared; this controls how the P "
                       "values are corrected for making several comparisons. Bonferroni is the "
                       "simple, conservative choice."),
        ParamSpec("trend", "Also test for a trend across ordered groups", kind="bool", default=False,
                  help="Tick this only when the groups have a natural ORDER, such as increasing doses "
                       "(0, 10, 100 mg) or disease stages, and there are three or more of them. It adds "
                       "the log-rank test for trend, which asks whether survival gets steadily better or "
                       "worse from the first group to the last. Numeric group labels are ordered by their "
                       "value; other labels are taken in the order they first appear in the table, so "
                       "sort the table first if that is not the order you mean."),
    ),
    description="Draws Kaplan-Meier survival curves and tests whether survival differs between groups, "
                "correctly handling subjects who were still event-free when last seen (censored).",
    citation="Kaplan & Meier (1958) J Am Stat Assoc 53:457-481; Mantel (1966) Cancer Chemother Rep 50:163-170",
)
def _run_survival(
    table: Any,
    time: str | None = None,
    event: str | None = None,
    group: str | None = None,
    test: str = "logrank",
    alpha: float = 0.05,
    ci_method: str = "log-log",
    adjust: str = "bonferroni",
    trend: bool = False,
    **_: Any,
) -> StatResult:
    time, event, group, notes = _pick_survival_columns(table, time, event, group)
    df: pd.DataFrame = table.frame(None, dropna=False)
    result = survival_analysis(
        df[time].tolist(),
        df[event].tolist(),
        None if group is None else df[group].tolist(),
        alpha=alpha, test=test, ci_method=ci_method, adjust=adjust, trend=bool(trend),
    )
    result.data_notes = notes + result.data_notes
    result.extra["columns"] = {"time": time, "event": event, "group": group}
    return result

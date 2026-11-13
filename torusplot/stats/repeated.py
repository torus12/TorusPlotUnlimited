"""Repeated-measures designs: one-way RM ANOVA, mixed (split-plot) ANOVA and two-way
RM ANOVA, with sphericity tests and corrections.

How it is computed (implemented here from the sums-of-squares / contrast formulation)
---------------------------------------------------------------------------------------
The data are arranged as a matrix ``Y`` with one row per subject and one column per
within-subject condition. For a within-subject effect with ``q`` degrees of freedom we
take an orthonormal contrast matrix ``C`` (columns orthogonal to each other and to the
constant, unit length) and transform the data, ``Z = Y C`` (subjects x q).

* One group of subjects: the effect is the test that the mean of ``Z`` is zero.
  Hypothesis SSCP ``H = N * zbar zbar'``, error SSCP ``E = (Z - zbar)'(Z - zbar)``.
* Several between-subject groups (mixed design): a one-way model is fitted to ``Z``.
  ``E`` is the pooled within-group SSCP (df_e = N - g). The within main effect tests
  the *intercept*: Type III uses the unweighted mean of the group means,
  ``H = u u' / sum_j(1 / (g^2 n_j))``; Type II uses the weighted grand mean,
  ``H = N zbar zbar'``. The between x within interaction is the between-group SSCP of
  ``Z``. The between main effect is a one-way ANOVA of the subject totals / sqrt(p),
  tested against subjects-within-groups.
* Univariate (classical) F tests: SS_effect = trace(H), SS_error = trace(E), with
  df = q * df_h and q * df_e.

Sphericity (all from ``S = E / df_e``, the covariance matrix of the orthonormal
contrasts; ``lambda`` = eigenvalues of ``S``)

* Mauchly (1940): ``W = det(S) / (trace(S) / q)^q``; chi-square approximation
  ``X2 = -rho * df_e * ln W`` with ``rho = 1 - (2 q^2 + q + 2) / (6 q df_e)`` and
  ``f = q (q + 1) / 2 - 1`` df. The P value includes the second-order term of Box's
  (1949) expansion, as given by Anderson (2003, section 10.7):
  ``P = P(chi2_f > X2) + w2 * (P(chi2_(f+4) > X2) - P(chi2_f > X2))`` with
  ``w2 = (q + 2)(q - 1)(q - 2)(2 q^3 + 6 q^2 + 3 q + 2) / (288 q^2 df_e^2 rho^2)``.
  This is what other major packages report (the first-order P value is kept in
  ``extra["sphericity"]["p_first_order"]``); the two agree for q = 2 and for large samples.
* Greenhouse-Geisser epsilon (Box 1954; Greenhouse & Geisser 1959):
  ``eps = trace(S)^2 / (q * trace(S^2))``; lower bound ``1 / q``.
* Huynh-Feldt epsilon (Huynh & Feldt 1976), reported value:
  ``(N q eps - 2) / (q (N - g - q eps))`` capped at 1, with N subjects in g groups.
  The corrected version of Lecoutre (1991), ``((N - g + 1) q eps - 2) / (q (N - g - q eps))``,
  is also computed (identical when g = 1).
* Corrected P values multiply both degrees of freedom of the F test by epsilon.

References
----------
* Mauchly JW (1940) Significance test for sphericity of a normal n-variate
  distribution. *Ann Math Stat* 11:204-209.
* Box GEP (1949) A general distribution theory for a class of likelihood criteria.
  *Biometrika* 36:317-346; Anderson TW (2003) *An Introduction to Multivariate
  Statistical Analysis*, 3rd ed., section 10.7. Wiley (distribution of the sphericity
  criterion).
* Box GEP (1954) Some theorems on quadratic forms applied in the study of analysis of
  variance problems, II. *Ann Math Stat* 25:484-498.
* Greenhouse SW, Geisser S (1959) On methods in the analysis of profile data.
  *Psychometrika* 24:95-112.
* Huynh H, Feldt LS (1976) Estimation of the Box correction for degrees of freedom
  from sample data in randomized block and split-plot designs. *J Educ Stat* 1:69-82.
* Lecoutre B (1991) A correction for the epsilon-tilde approximate test in repeated
  measures designs with two or more independent groups. *J Educ Stat* 16:371-372.
* Olejnik S, Algina J (2003) Generalized eta and omega squared statistics.
  *Psychol Methods* 8:434-447; Bakeman R (2005) Recommended effect size statistics
  for repeated measures designs. *Behav Res Methods* 37:379-384.
* Winer BJ, Brown DR, Michels KM (1991) *Statistical Principles in Experimental
  Design*, 3rd ed. McGraw-Hill (split-plot sums of squares).
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from . import effect_sizes as es
from ._utils import StatsError, check_alpha, fmt, p_phrase, sig_phrase
from .factorial import (
    _brown_forsythe,
    _check_posthoc,
    _check_ss_type,
    _ci_multiplier,
    _ci_note,
    _describe,
    _encode_factor,
    _f_text,
    _level_label,
    _none_if_nan,
    _normality_check,
    _quiet,
    _smd_pooled,
)
from .multiplicity import METHOD_LABELS, adjust_pvalues
from .registry import ParamSpec, register
from .result import AssumptionCheck, Comparison, EffectSize, StatResult

RM_POSTHOC_METHODS = ("none", "bonferroni", "sidak", "holm", "holm-sidak", "fdr_bh")
CORRECTIONS = ("auto", "none", "gg", "hf")
_CORRECTION_LABELS = {"none": "none (sphericity assumed)", "gg": "Greenhouse-Geisser", "hf": "Huynh-Feldt"}

_CITE_MAUCHLY = (
    "Mauchly JW (1940) Significance test for sphericity of a normal n-variate distribution. "
    "Ann Math Stat 11:204-209; P value with the second-order term of Box GEP (1949) Biometrika 36:317-346"
)
_CITE_BOX = "Box GEP (1954) Ann Math Stat 25:484-498"
_CITE_GG = "Greenhouse SW, Geisser S (1959) On methods in the analysis of profile data. Psychometrika 24:95-112"
_CITE_HF = "Huynh H, Feldt LS (1976) J Educ Stat 1:69-82"
_CITE_LECOUTRE = "Lecoutre B (1991) J Educ Stat 16:371-372"
_CITE_GES = "Olejnik S, Algina J (2003) Psychol Methods 8:434-447; Bakeman R (2005) Behav Res Methods 37:379-384"
_CITE_SPLIT = "Winer BJ, Brown DR, Michels KM (1991) Statistical Principles in Experimental Design, 3rd ed. McGraw-Hill"
_CITE_ADJUST = {
    "bonferroni": "Dunn OJ (1961) J Am Stat Assoc 56:52-64",
    "sidak": "Sidak Z (1967) J Am Stat Assoc 62:626-633",
    "holm": "Holm S (1979) Scand J Stat 6:65-70",
    "holm-sidak": "Holm S (1979) Scand J Stat 6:65-70; Sidak Z (1967) J Am Stat Assoc 62:626-633",
    "fdr_bh": "Benjamini Y, Hochberg Y (1995) J R Stat Soc B 57:289-300",
}

_HELP_CORRECTION = (
    "Repeated-measures ANOVA assumes 'sphericity' (the differences between every pair of conditions "
    "vary equally). 'auto' applies the Greenhouse-Geisser correction whenever there are more than two "
    "conditions: this is always safe, costs very little when sphericity holds, and avoids relying on "
    "Mauchly's test, which often misses violations in small samples. 'hf' (Huynh-Feldt) is slightly "
    "less conservative; 'none' assumes sphericity."
)


# =============================================================================
# Engine
# =============================================================================
def _orthonormal_contrasts(k: int) -> np.ndarray:
    """k x (k - 1) matrix of normalized Helmert contrasts (orthonormal, orthogonal to 1)."""
    c = np.zeros((k, k - 1))
    for j in range(k - 1):
        c[: j + 1, j] = 1.0
        c[j + 1, j] = -(j + 1.0)
        c[:, j] /= math.sqrt((j + 1.0) * (j + 2.0))
    return c


@dataclass
class _Scores:
    """Sums of squares and sphericity statistics of one within-subject effect."""

    q: int
    n_subjects: int
    n_groups: int
    df_error: int  # N - g (multiply by q for the univariate error df)
    ss_intercept: float  # within main effect: trace(H_intercept)
    ss_group: float  # between x within interaction: trace(H_group)
    ss_error: float  # trace(E)
    mauchly_w: float | None = None
    mauchly_chi2: float | None = None
    mauchly_df: int | None = None
    mauchly_p: float | None = None  # with Box's second-order term
    mauchly_p_first_order: float | None = None
    eps_gg: float | None = None
    eps_hf: float | None = None  # reported value (chosen variant, capped at 1)
    eps_hf_original: float | None = None  # uncapped Huynh-Feldt (1976)
    eps_hf_lecoutre: float | None = None  # uncapped Lecoutre (1991)
    eps_lb: float = 1.0
    hf_variant: str = "original"
    note: str = ""


def _analyse_scores(z: np.ndarray, codes: np.ndarray, n_groups: int, ss_type: int = 3,
                    hf_variant: str = "original", tiny: float = 0.0) -> _Scores:
    """Hypothesis / error sums of squares and sphericity statistics for ``Z = Y C``.

    ``codes`` assigns each subject (row) to one of ``n_groups`` between-subject groups
    (all zeros for a pure within-subject design). ``tiny`` is the absolute tolerance
    below which the error variation is treated as zero.
    """
    z = np.asarray(z, dtype=float)
    n, q = z.shape
    counts = np.bincount(codes, minlength=n_groups).astype(float)
    means = np.vstack([z[codes == j].mean(axis=0) for j in range(n_groups)])
    centred = z - means[codes]
    e = centred.T @ centred
    df_error = n - n_groups
    grand = z.mean(axis=0)
    if ss_type == 3:
        unweighted = means.mean(axis=0)
        ss_intercept = float(unweighted @ unweighted) / float(np.sum(1.0 / (n_groups**2 * counts)))
    else:
        ss_intercept = float(n * (grand @ grand))
    ss_group = float((counts[:, None] * (means - grand) ** 2).sum())
    out = _Scores(q=q, n_subjects=n, n_groups=n_groups, df_error=df_error, ss_intercept=ss_intercept,
                  ss_group=ss_group, ss_error=float(np.trace(e)), eps_lb=1.0 / q, hf_variant=hf_variant)
    if df_error < 1:
        out.note = "Too few subjects to estimate the error covariance."
        return out
    lam = np.clip(np.linalg.eigvalsh(e / df_error), 0.0, None)
    trace = float(lam.sum())
    if not trace > max(tiny, 0.0) / max(df_error, 1):
        out.note = "The within-subject error variation is zero, so sphericity statistics are undefined."
        return out
    if q == 1:
        out.eps_gg = out.eps_hf = out.eps_hf_original = out.eps_hf_lecoutre = 1.0
        out.note = "With only two levels there is a single difference score, so sphericity holds by definition."
        return out
    eps = trace**2 / (q * float((lam**2).sum()))
    eps = min(max(eps, 1.0 / q), 1.0)
    out.eps_gg = eps
    denom = q * (n - n_groups - q * eps)
    if denom > 1e-12:
        out.eps_hf_original = (n * q * eps - 2.0) / denom
        out.eps_hf_lecoutre = ((n - n_groups + 1.0) * q * eps - 2.0) / denom
        chosen = out.eps_hf_lecoutre if hf_variant == "lecoutre" else out.eps_hf_original
        out.eps_hf = min(max(chosen, eps), 1.0)
    else:
        out.note = ("The Huynh-Feldt epsilon is not defined with this few subjects; "
                    "use the Greenhouse-Geisser correction.")
    # Mauchly's W needs a non-singular S (at least q error df).
    multiplier = df_error - (2.0 * q * q + q + 2.0) / (6.0 * q)
    if df_error >= q and lam.min() > 1e-12 * lam.max() and multiplier > 0:
        log_w = float(np.log(lam).sum() - q * math.log(trace / q))
        log_w = min(log_w, 0.0)
        out.mauchly_w = math.exp(log_w)
        out.mauchly_chi2 = -multiplier * log_w
        out.mauchly_df = q * (q + 1) // 2 - 1
        p1 = float(stats.chi2.sf(out.mauchly_chi2, out.mauchly_df))
        p2 = float(stats.chi2.sf(out.mauchly_chi2, out.mauchly_df + 4))
        rho = multiplier / df_error
        w2 = ((q + 2.0) * (q - 1.0) * (q - 2.0) * (2.0 * q**3 + 6.0 * q**2 + 3.0 * q + 2.0)
              / (288.0 * (q * df_error * rho) ** 2))
        out.mauchly_p_first_order = p1
        out.mauchly_p = min(max(p1 + w2 * (p2 - p1), 0.0), 1.0)
    else:
        extra = ("Mauchly's test needs more subjects than conditions (the covariance matrix of the "
                 "differences is singular).")
        out.note = (out.note + " " + extra).strip()
    return out


def _check_correction(correction: str) -> str:
    key = str(correction).strip().lower().replace("_", "-")
    key = {"greenhouse-geisser": "gg", "huynh-feldt": "hf", "geisser-greenhouse": "gg", "": "auto"}.get(key, key)
    if key not in CORRECTIONS:
        raise StatsError("correction must be one of 'auto', 'none', 'gg' (Greenhouse-Geisser) or 'hf' (Huynh-Feldt).")
    return key


def _check_hf_variant(hf_variant: str) -> str:
    key = str(hf_variant).strip().lower()
    if key not in {"original", "lecoutre"}:
        raise StatsError("hf_variant must be 'original' (Huynh & Feldt 1976) or 'lecoutre' (Lecoutre 1991).")
    return key


def _resolve_correction(correction: str, scores: _Scores) -> tuple[str, str | None]:
    """Which correction the headline P value uses: ('none' | 'gg' | 'hf', warning or None).

    Rule: 'auto' = Greenhouse-Geisser whenever the effect has more than one degree of
    freedom (more than two levels), otherwise no correction (none is possible).
    """
    if scores.q == 1 or scores.eps_gg is None:
        return "none", None
    if correction == "auto":
        return "gg", None
    if correction == "hf" and scores.eps_hf is None:
        return "gg", ("The Huynh-Feldt correction is not defined for this few subjects; the "
                      "Greenhouse-Geisser correction was used instead.")
    return correction, None


def _within_row(source: str, kind: str, ss: float, df1: int, scores: _Scores, correction: str,
                ss_total: float, error_name: str) -> dict[str, Any]:
    """ANOVA-table row of a within-subject effect with corrected P values."""
    df2 = scores.q * scores.df_error
    ms = ss / df1
    ms_error = scores.ss_error / df2 if df2 > 0 else math.nan
    row: dict[str, Any] = {
        "source": source, "kind": kind, "ss": ss, "df": df1, "ms": ms, "f": None, "p": None,
        "df_error": df2, "error_term": error_name,
        "eps_gg": scores.eps_gg, "eps_hf": scores.eps_hf, "eps_lb": scores.eps_lb,
        "p_gg": None, "p_hf": None, "p_lb": None,
        "p_reported": None, "df_reported": df1, "df_error_reported": df2,
        "correction": _CORRECTION_LABELS["none"],
        "partial_eta_sq": ss / (ss + scores.ss_error) if (ss + scores.ss_error) > 0 else None,
        "percent_of_variation": 100.0 * ss / ss_total if ss_total > 0 else None,
    }
    if not (df2 > 0 and ms_error > 0):
        return row
    f = ms / ms_error
    row["f"] = f
    row["p"] = float(stats.f.sf(f, df1, df2))
    if scores.eps_gg is not None:
        row["p_gg"] = float(stats.f.sf(f, scores.eps_gg * df1, scores.eps_gg * df2))
    if scores.eps_hf is not None:
        row["p_hf"] = float(stats.f.sf(f, scores.eps_hf * df1, scores.eps_hf * df2))
    row["p_lb"] = float(stats.f.sf(f, scores.eps_lb * df1, scores.eps_lb * df2))
    used, _ = _resolve_correction(correction, scores)
    eps = {"none": 1.0, "gg": scores.eps_gg, "hf": scores.eps_hf}[used]
    row["p_reported"] = {"none": row["p"], "gg": row["p_gg"], "hf": row["p_hf"]}[used]
    row["df_reported"], row["df_error_reported"] = eps * df1, eps * df2
    row["correction"] = _CORRECTION_LABELS[used]
    return row


def _plain_row(source: str, kind: str, ss: float, df: int, ss_total: float, ms_error: float | None = None,
               df_error: int | None = None, error_name: str | None = None) -> dict[str, Any]:
    ms = ss / df if df > 0 else None
    row: dict[str, Any] = {
        "source": source, "kind": kind, "ss": ss, "df": df, "ms": ms, "f": None, "p": None,
        "df_error": df_error, "error_term": error_name, "p_reported": None,
        "partial_eta_sq": None,
        "percent_of_variation": 100.0 * ss / ss_total if ss_total > 0 else None,
    }
    if ms is not None and ms_error is not None and df_error and ms_error > 0:
        row["f"] = ms / ms_error
        row["p"] = row["p_reported"] = float(stats.f.sf(row["f"], df, df_error))
    return row


def _sphericity_extra(effect: str, scores: _Scores) -> dict[str, Any]:
    return {
        "effect": effect, "levels_df": scores.q,
        "mauchly_w": scores.mauchly_w, "chi2": scores.mauchly_chi2, "df": scores.mauchly_df,
        "p": scores.mauchly_p, "p_first_order": scores.mauchly_p_first_order,
        "eps_gg": scores.eps_gg, "eps_hf": scores.eps_hf,
        "eps_hf_original_uncapped": scores.eps_hf_original,
        "eps_hf_lecoutre_uncapped": scores.eps_hf_lecoutre,
        "eps_lower_bound": scores.eps_lb, "hf_variant": scores.hf_variant,
        "hf_note": "eps_hf is the Huynh & Feldt (1976) estimate"
                   + (" as corrected by Lecoutre (1991)" if scores.hf_variant == "lecoutre" else "")
                   + ", capped at 1. Lecoutre B (1991) J Educ Stat 16:371-372 gives the corrected formula "
                     "for designs with two or more groups (identical for a single group).",
        "note": scores.note,
    }


def _sphericity_check(effect: str, scores: _Scores) -> AssumptionCheck:
    name = "Sphericity (Mauchly's test)"
    if scores.q == 1:
        return AssumptionCheck(name=name, target=effect, status="pass",
                               detail="Only two repeated levels: sphericity holds automatically and no correction is needed.")
    if scores.mauchly_p is None:
        return AssumptionCheck(
            name=name, target=effect, status="warn",
            detail=(scores.note or "Mauchly's test could not be computed.")
            + " The Greenhouse-Geisser corrected P value is the safe choice.",
        )
    eps_text = f"Greenhouse-Geisser epsilon = {fmt(scores.eps_gg, 3)}"
    if scores.eps_hf is not None:
        eps_text += f", Huynh-Feldt epsilon = {fmt(scores.eps_hf, 3)}"
    stat_text = f"W = {fmt(scores.mauchly_w, 3)}, chi-square({scores.mauchly_df}) = {fmt(scores.mauchly_chi2)}, {p_phrase(scores.mauchly_p)}"
    if scores.mauchly_p < 0.05:
        return AssumptionCheck(
            name=name, target=effect, statistic=scores.mauchly_w, p_value=scores.mauchly_p, status="fail",
            detail=f"Sphericity is violated ({stat_text}). Use a corrected P value "
                   f"(Greenhouse-Geisser or Huynh-Feldt); {eps_text}.",
        )
    return AssumptionCheck(
        name=name, target=effect, statistic=scores.mauchly_w, p_value=scores.mauchly_p, status="pass",
        detail=f"No evidence against sphericity ({stat_text}); {eps_text}. The test has low power with "
               "few subjects, so a corrected P value is still a sensible default.",
    )


# =============================================================================
# Paired / unpaired comparison helpers
# =============================================================================
@dataclass
class _Raw:
    group_a: str
    group_b: str
    label: str | None
    cell_a: dict[str, str] | None
    cell_b: dict[str, str] | None
    estimate: float
    se: float
    df: float
    effect: EffectSize | None = None


def _paired_raw(a: np.ndarray, b: np.ndarray, name_a: str, name_b: str, label: str | None,
                cell_a: dict[str, str] | None, cell_b: dict[str, str] | None) -> _Raw:
    """Paired t test ingredients for two columns measured in the same subjects."""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    n = diff.size
    if n < 2:
        return _Raw(name_a, name_b, label, cell_a, cell_b, float(diff.mean()) if n else math.nan, math.nan, 0.0)
    sd = float(diff.std(ddof=1))
    scale = max(1.0, float(np.max(np.abs(a))), float(np.max(np.abs(b))))
    if sd <= 1e-12 * scale:
        sd = 0.0
    effect = es.cohen_d_paired(a, b, target=f"{name_a} vs {name_b}") if sd > 0 else None
    return _Raw(name_a, name_b, label, cell_a, cell_b, float(diff.mean()), sd / math.sqrt(n), float(n - 1), effect)


def _assemble(raws: list[_Raw], method: str, alpha: float) -> tuple[list[Comparison], int]:
    """Adjust a family of t-type comparisons; returns (comparisons, number not testable)."""
    t_val = np.full(len(raws), np.nan)
    p_raw = np.full(len(raws), np.nan)
    for i, r in enumerate(raws):
        if r.df >= 1 and math.isfinite(r.se) and r.se > 0:
            t_val[i] = r.estimate / r.se
            p_raw[i] = float(2.0 * stats.t.sf(abs(t_val[i]), r.df))
    p_adj = adjust_pvalues(p_raw, method)
    m = int(np.isfinite(p_raw).sum())
    crit_cache: dict[float, float] = {}
    out = []
    for i, r in enumerate(raws):
        ok = bool(np.isfinite(p_raw[i]))
        crit = None
        if ok:
            if r.df not in crit_cache:
                crit_cache[r.df] = _ci_multiplier(method, m, r.df, alpha)
            crit = crit_cache[r.df]
        out.append(Comparison(
            id="", group_a=r.group_a, group_b=r.group_b, label=r.label, cell_a=r.cell_a, cell_b=r.cell_b,
            estimate=_none_if_nan(r.estimate), se=_none_if_nan(r.se),
            ci_low=r.estimate - crit * r.se if ok else None,
            ci_high=r.estimate + crit * r.se if ok else None,
            statistic_name="t", statistic=float(t_val[i]) if ok else None,
            df=float(r.df) if r.df >= 1 else None,
            p_value=float(p_raw[i]) if ok else None,
            p_adjusted=float(p_adj[i]) if ok else None,
            effect_size=r.effect if ok else None,
        ))
    return out, len(raws) - m


def _differences_normality(columns: dict[str, np.ndarray], pairs: list[tuple[str, str]]) -> AssumptionCheck:
    """Shapiro-Wilk on the paired differences behind each comparison (summarised)."""
    name = "Normality of paired differences (Shapiro-Wilk)"
    worst: tuple[float, float, str] | None = None
    flagged = []
    tested = 0
    for a, b in pairs:
        diff = columns[a] - columns[b]
        if diff.size < 3 or float(np.ptp(diff)) <= 1e-10 * max(1.0, float(np.max(np.abs(diff)))):
            continue
        with _quiet():
            w, p = stats.shapiro(diff)
        if not math.isfinite(float(p)):
            continue
        tested += 1
        if worst is None or p < worst[1]:
            worst = (float(w), float(p), f"{a} - {b}")
        if p < 0.05:
            flagged.append(f"{a} - {b}")
    if worst is None:
        return AssumptionCheck(name=name, status="warn",
                               detail="Too few subjects (or no variation) to test the normality of the differences.")
    if flagged:
        shown = ", ".join(flagged[:4]) + (", ..." if len(flagged) > 4 else "")
        return AssumptionCheck(
            name=name, target=worst[2], statistic=worst[0], p_value=worst[1], status="warn",
            detail=f"{len(flagged)} of {tested} sets of differences deviate from normality ({shown}; smallest "
                   f"{p_phrase(worst[1])}, not adjusted for testing {tested} sets). Paired t tests are fairly "
                   "robust, but consider the Wilcoxon signed-rank test for those comparisons.",
        )
    return AssumptionCheck(
        name=name, target=worst[2], statistic=worst[0], p_value=worst[1], status="pass",
        detail=f"No evidence against normality in any of the {tested} sets of paired differences "
               f"(smallest {p_phrase(worst[1])}).",
    )


def _effect(e: EffectSize) -> EffectSize:
    e.value = _none_if_nan(e.value)
    e.ci_low, e.ci_high = _none_if_nan(e.ci_low), _none_if_nan(e.ci_high)
    return e


def _partial_eta(ss: float, ss_error: float, df1: float, df2: float, target: str, q: int) -> EffectSize:
    e = _effect(es.partial_eta_squared(ss, ss_error, df1, df2, target=target))
    if e.ci_method and q > 1:
        e.ci_method += "; uses the uncorrected df (assumes sphericity)"
    return e


def _generalized_eta(ss: float, denominator_extra: float, target: str) -> EffectSize:
    """Generalized eta squared: SS_effect / (SS_effect + all subject and error SS)."""
    denom = ss + denominator_extra
    value = ss / denom if denom > 0 else None
    return EffectSize(name="generalized eta squared", value=value, target=target,
                      magnitude=es.magnitude_eta2(value) if value is not None else None)


def _complete_matrix(matrix: Any, names: Any) -> tuple[np.ndarray, list[str], list[str]]:
    """Validate a subjects x conditions matrix and keep the complete rows."""
    try:
        arr = np.asarray(matrix, dtype=float)
    except (TypeError, ValueError) as exc:
        raise StatsError("The data must be numeric: one row per subject, one column per condition.") from exc
    if arr.ndim != 2:
        raise StatsError("The data must be a table with one row per subject and one column per condition.")
    names = [str(n) for n in names]
    if len(names) != arr.shape[1]:
        raise StatsError(f"{len(names)} condition name(s) were given for {arr.shape[1]} column(s) of data.")
    if len(set(names)) != len(names):
        raise StatsError("Condition names must be unique.")
    if arr.shape[1] < 2:
        raise StatsError("Repeated-measures ANOVA needs at least two conditions (columns).")
    finite = np.isfinite(arr)
    empty = ~finite.any(axis=1)
    complete = finite.all(axis=1)
    partial = int((~complete & ~empty).sum())
    notes = []
    if partial:
        notes.append(
            f"{partial} subject(s) (rows) with a missing value in at least one condition were left out: "
            "repeated-measures ANOVA needs complete data for every subject."
        )
    return arr[complete], names, notes


# =============================================================================
# One-way repeated-measures ANOVA
# =============================================================================
def anova_rm_oneway(
    matrix: Any,
    names: list[str] | tuple[str, ...],
    *,
    correction: str = "auto",
    posthoc: str | None = "holm",
    control: str | None = None,
    alpha: float = 0.05,
    hf_variant: str = "original",
    factor_name: str = "Condition",
) -> StatResult:
    """One-way repeated-measures ANOVA.

    Parameters
    ----------
    matrix : 2-D array, rows = subjects, columns = conditions / time points. Rows with a
        missing value are left out (and reported).
    names : condition names, one per column.
    correction : which P value is the headline result. ``"auto"`` (default) uses the
        Greenhouse-Geisser correction whenever there are more than two conditions (a
        safe default: Mauchly's test has little power in small samples, and the
        correction costs almost nothing when sphericity holds); ``"none"`` assumes
        sphericity; ``"gg"`` / ``"hf"`` force Greenhouse-Geisser / Huynh-Feldt. The
        table always lists the uncorrected, GG, HF and lower-bound P values.
    posthoc : "holm" (default), "bonferroni", "sidak", "holm-sidak", "fdr_bh" or
        "none". Comparisons are paired t tests (each with its own error term, so they
        do not depend on sphericity), all pairs or each condition against ``control``.
    hf_variant : "original" (Huynh & Feldt 1976, what other major packages print) or
        "lecoutre" (Lecoutre 1991; identical for a single group of subjects).
    """
    alpha = check_alpha(alpha)
    correction = _check_correction(correction)
    hf_variant = _check_hf_variant(hf_variant)
    method = _check_posthoc(posthoc, RM_POSTHOC_METHODS)
    y, names, notes = _complete_matrix(matrix, names)
    n, k = y.shape
    if control is not None and str(control) not in names:
        raise StatsError(f"Control condition '{control}' is not one of the conditions: {names}.")
    if n < 2:
        raise StatsError(
            f"Only {n} subject(s) have a value in every condition; repeated-measures ANOVA needs at least 2 "
            "(and realistically many more)."
        )
    q = k - 1
    grand = float(y.mean())
    ss_total = float(((y - grand) ** 2).sum())
    scale = max(1.0, float(np.max(np.abs(y))))
    if float(np.ptp(y)) <= 1e-12 * scale:
        ss_total = 0.0
    tiny = 1e-12 * ss_total if ss_total > 0 else math.inf
    scores = _analyse_scores(y @ _orthonormal_contrasts(k), np.zeros(n, dtype=int), 1, 3, hf_variant, tiny)
    ss_effect = scores.ss_intercept if ss_total > 0 else 0.0
    if scores.ss_error <= tiny:
        scores.ss_error = 0.0
    subject_means = y.mean(axis=1)
    ss_subjects = float(k * ((subject_means - grand) ** 2).sum()) if ss_total > 0 else 0.0
    df_error = q * (n - 1)
    ms_error = scores.ss_error / df_error

    warnings_out: list[str] = []
    row = _within_row(factor_name, "within", ss_effect, q, scores, correction, ss_total, "Residual")
    used, fallback = _resolve_correction(correction, scores)
    if fallback:
        warnings_out.append(fallback)
    rows = [
        row,
        _plain_row("Subjects", "subjects", ss_subjects, n - 1, ss_total, ms_error, df_error, "Residual"),
        {"source": "Residual", "kind": "residual", "ss": scores.ss_error, "df": df_error, "ms": ms_error,
         "f": None, "p": None, "df_error": None, "error_term": None, "p_reported": None,
         "partial_eta_sq": None,
         "percent_of_variation": 100.0 * scores.ss_error / ss_total if ss_total > 0 else None},
        {"source": "Total", "kind": "total", "ss": ss_total, "df": n * k - 1, "ms": None, "f": None, "p": None,
         "df_error": None, "error_term": None, "p_reported": None, "partial_eta_sq": None,
         "percent_of_variation": 100.0 if ss_total > 0 else None},
    ]
    if ss_total == 0:
        warnings_out.append("All values are identical: there is no variation to analyse, so no F test is possible.")
    elif row["f"] is None:
        warnings_out.append(
            "The residual variation is zero (every subject changes by exactly the same amount), so the "
            "F ratio and P values cannot be computed."
        )
    if n <= k and k > 2:
        warnings_out.append(
            f"There are only {n} subjects for {k} conditions: sphericity cannot be tested and the "
            "results rest heavily on assumptions."
        )

    # ---------------------------------------------------------------- comparisons
    columns = {name: y[:, j] for j, name in enumerate(names)}
    if control is not None:
        pairs = [(name, str(control)) for name in names if name != str(control)]
    else:
        pairs = list(itertools.combinations(names, 2))
    comparisons: list[Comparison] = []
    family_text = correction_label = None
    if method != "none":
        raws = [_paired_raw(columns[a], columns[b], a, b, None, {factor_name: a}, {factor_name: b})
                for a, b in pairs]
        comparisons, untestable = _assemble(raws, method, alpha)
        if untestable:
            warnings_out.append(
                f"{untestable} comparison(s) could not be tested because the paired differences do not vary."
            )
        correction_label = METHOD_LABELS[method]
        family_text = (
            (f"{len(pairs)} comparisons of each condition with the control '{control}'" if control is not None
             else f"all {len(pairs)} pairwise comparisons among the {k} conditions")
            + "; paired t tests (each comparison uses its own differences, so sphericity is not required), "
              "treated as one family."
        )

    # ---------------------------------------------------------------- assumptions
    resid = y - subject_means[:, None] - y.mean(axis=0)[None, :] + grand
    assumptions = [
        AssumptionCheck(
            name="Repeated measures on the same subjects", status="info",
            detail=f"Each row is one subject measured under all {k} conditions; {n} subject(s) with complete "
                   "data were analysed. Subjects must be independent of each other.",
        ),
        _sphericity_check(factor_name, scores),
        _normality_check(resid, target="residuals (subject and condition effects removed)"),
    ]
    if method != "none":
        assumptions.append(_differences_normality(columns, pairs))

    # ---------------------------------------------------------------- effect sizes, text
    effects = [
        _partial_eta(ss_effect, scores.ss_error, q, df_error, factor_name, q),
        _generalized_eta(ss_effect, ss_subjects + scores.ss_error, factor_name),
    ]
    descriptives = [{"group": name, **_describe(columns[name])} for name in names]
    cell_means = [{factor_name: d["group"], "mean": d["mean"], "sd": d["sd"], "sem": d["sem"], "n": d["n"]}
                  for d in descriptives]
    sentences = [f"Repeated-measures one-way ANOVA comparing {k} conditions measured in the same {n} subjects."]
    if row["f"] is not None:
        corr_text = ""
        if used != "none":
            eps = scores.eps_gg if used == "gg" else scores.eps_hf
            corr_text = f" with the {_CORRECTION_LABELS[used]} correction (epsilon = {fmt(eps, 3)})"
        sentences.append(
            f"The difference between conditions was {sig_phrase(row['p_reported'], alpha)}, "
            f"{_f_text(row['f'], row['df_reported'], row['df_error_reported'])}{corr_text}; partial eta squared = "
            f"{fmt(effects[0].value, 3)}, generalized eta squared = {fmt(effects[1].value, 3)}."
        )
        if scores.mauchly_p is not None:
            sentences.append(
                "Mauchly's test " + ("indicated a violation of sphericity" if scores.mauchly_p < 0.05
                                     else "gave no evidence against sphericity")
                + f" (W = {fmt(scores.mauchly_w, 3)}, {p_phrase(scores.mauchly_p)})."
            )
    else:
        sentences.append("The difference between conditions could not be tested.")
    if comparisons:
        n_sig = sum(1 for c in comparisons if c.p_adjusted is not None and c.p_adjusted < alpha)
        sentences.append(
            f"Paired comparisons ({correction_label}): {n_sig} of {len(comparisons)} were significant at "
            f"alpha = {alpha:g}."
        )
    methods = (
        "Conditions were compared by one-way repeated-measures ANOVA"
        + (f" with the {_CORRECTION_LABELS[used]} correction for non-sphericity" if used != "none"
           else (" (sphericity assumed)" if q > 1 else ""))
        + f" (alpha = {alpha:g})"
        + (f", followed by paired t tests with {correction_label} adjustment" if comparisons else "") + "."
    )
    citations = [_CITE_MAUCHLY, _CITE_BOX, _CITE_GG, _CITE_HF, _CITE_LECOUTRE, _CITE_GES]
    if method != "none":
        citations.append(_CITE_ADJUST[method])
    extra: dict[str, Any] = {
        "sphericity": _sphericity_extra(factor_name, scores),
        "correction_requested": correction, "correction_applied": used,
        "p_uncorrected": row["p"], "p_gg": row["p_gg"], "p_hf": row["p_hf"], "p_lower_bound": row["p_lb"],
        "cell_means": cell_means, "factors": [factor_name],
        "n_subjects": n, "n_observations": n * k,
        "subject_means": [float(v) for v in subject_means],
    }
    if comparisons:
        extra["comparisons_note"] = _ci_note(method)
    return StatResult(
        analysis="anova_rm_oneway",
        test_name="Repeated-measures one-way ANOVA",
        why_chosen=(
            "The same subjects were measured under three or more conditions (or at several time points). "
            "Comparing each subject with itself removes differences between subjects and makes the test "
            "more sensitive than an ordinary one-way ANOVA."
        ),
        assumptions=assumptions,
        statistic_name="F", statistic=row["f"], df=row["df_reported"], df2=row["df_error_reported"],
        p_value=row["p_reported"], alpha=alpha,
        effect_sizes=effects, n=dict.fromkeys(names, n), n_total=n,
        descriptives=descriptives, anova_table=rows,
        comparisons=comparisons, comparison_family=family_text, correction=correction_label,
        warnings=warnings_out, data_notes=notes,
        interpretation=" ".join(sentences), methods_text=methods, citations=citations, extra=extra,
    )


# =============================================================================
# Mixed (split-plot) ANOVA: one between-subjects x one within-subjects factor
# =============================================================================
@dataclass
class _Wide:
    y: np.ndarray  # subjects x within levels (complete subjects only)
    codes: np.ndarray  # between-group code of each subject
    subjects: list[str]
    between_levels: list[str]
    within_levels: list[str]
    notes: list[str]


def _pivot_mixed(df: pd.DataFrame, value: str, subject: str, between: str | None, within: str,
                 within_label: str) -> _Wide:
    """Long data -> subjects x within-level matrix; checks the repeated-measures layout."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame in long (tidy) format.")
    columns = [c for c in (value, subject, between, within) if c is not None]
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise StatsError(f"Column(s) not found in the data: {missing}. Available: {list(df.columns)}.")
    if len(set(columns)) != len(columns):
        raise StatsError("Each column can play only one role (outcome, subject, between or within factor).")
    yv = pd.to_numeric(df[value], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(yv)
    for c in columns[1:]:
        ok &= df[c].notna().to_numpy()
    notes: list[str] = []
    if int((~ok).sum()):
        notes.append(f"{int((~ok).sum())} row(s) with a missing or non-numeric value were ignored.")
    if not ok.any():
        raise StatsError("There are no complete rows of data to analyse.")
    sub = df.loc[ok]
    yv = yv[ok]
    if between is not None:
        b_codes, b_levels = _encode_factor(sub[between])
    else:
        b_codes, b_levels = np.zeros(len(sub), dtype=int), ["all"]
    w_codes, w_levels = _encode_factor(sub[within])
    s_labels = [_level_label(v) for v in sub[subject].tolist()]
    units: dict[tuple[int, str], int] = {}
    for key in zip(b_codes.tolist(), s_labels):
        units.setdefault(key, len(units))
    p = len(w_levels)
    y = np.full((len(units), p), np.nan)
    seen = np.zeros((len(units), p), dtype=int)
    for key, w, v in zip(zip(b_codes.tolist(), s_labels), w_codes.tolist(), yv.tolist()):
        r = units[key]
        y[r, w] = v
        seen[r, w] += 1
    if (seen > 1).any():
        r, w = np.argwhere(seen > 1)[0]
        who = list(units)[int(r)][1]
        raise StatsError(
            f"Subject '{who}' has more than one value at {within_label} = '{w_levels[int(w)]}'. Each subject "
            "must have exactly one value per level; average the replicates first or check the subject column."
        )
    if between is not None:
        labels_only = [k[1] for k in units]
        if len(set(labels_only)) < len(labels_only):
            notes.append(
                "The same subject label appears in more than one group; subjects were treated as different "
                "individuals in each group (as required for a between-subjects factor)."
            )
    complete = np.isfinite(y).all(axis=1)
    dropped = int((~complete).sum())
    if dropped:
        notes.append(
            f"{dropped} subject(s) without a value at every level of {within_label} were left out: "
            "repeated-measures ANOVA needs complete data for every subject."
        )
    keys = [k for k, keep in zip(units, complete) if keep]
    return _Wide(y=y[complete], codes=np.array([k[0] for k in keys], dtype=int),
                 subjects=[k[1] for k in keys], between_levels=b_levels, within_levels=w_levels, notes=notes)


MIXED_COMPARE = ("between_per_level", "within_per_group", "between_main", "within_main")


def anova_mixed(
    df: pd.DataFrame,
    value: str,
    subject: str,
    between: str,
    within: str,
    *,
    correction: str = "auto",
    ss_type: int = 3,
    posthoc: str | None = None,
    posthoc_compare: str = "between_per_level",
    alpha: float = 0.05,
    hf_variant: str = "original",
    factor_labels: dict[str, str] | None = None,
) -> StatResult:
    """Mixed-design (split-plot) ANOVA: one between-subjects x one within-subjects factor.

    ``df`` is long: one row per measurement with the outcome, a subject identifier, the
    between-subjects factor (each subject is in exactly one group) and the
    within-subjects factor (each subject is measured at every level).

    What is computed (see the module docstring): the between factor is tested against
    subjects-within-groups; the within factor and the interaction against the within
    residual, with the same sphericity estimates (pooled within-group covariance).
    With unequal group sizes, ``ss_type=3`` (default) tests the within factor on the
    *unweighted* mean of the group means (every group counts equally); ``ss_type=2``
    weights groups by their size. The between-factor and interaction sums of squares
    do not depend on the type in this design.

    ``posthoc_compare``:

    * ``"between_per_level"``: groups compared at each within level, unpaired t tests
      that use only that level's data with the error pooled over all groups at that
      level (df = subjects - groups; the ordinary two-sample t test when there are two
      groups). This does not rely on sphericity or on equal variances across levels.
    * ``"within_per_group"``: within levels compared inside each group, paired t tests
      on that group's subjects (df = n_group - 1).
    * ``"between_main"``: groups compared on each subject's average over the levels.
    * ``"within_main"``: within levels compared across all subjects (differences
      averaged over groups as for the main effect; df = subjects - groups).

    All comparisons of the chosen set form one family for the P-value adjustment.
    """
    alpha = check_alpha(alpha)
    correction = _check_correction(correction)
    hf_variant = _check_hf_variant(hf_variant)
    ss_type = _check_ss_type(ss_type)
    method = _check_posthoc(posthoc, RM_POSTHOC_METHODS)
    if method != "none" and posthoc_compare not in MIXED_COMPARE:
        raise StatsError(f"posthoc_compare must be one of: {', '.join(MIXED_COMPARE)}.")
    labels = factor_labels or {}
    b_label, w_label = str(labels.get(between, between)), str(labels.get(within, within))
    if b_label == w_label:
        b_label, w_label = f"{b_label} (between)", f"{w_label} (within)"
    wide = _pivot_mixed(df, value, subject, between, within, w_label)
    y, codes = wide.y, wide.codes
    b_levels, w_levels = wide.between_levels, wide.within_levels
    g_all, p = len(b_levels), len(w_levels)
    if p < 2:
        raise StatsError(f"The within-subjects factor '{w_label}' needs at least two levels.")
    counts = np.bincount(codes, minlength=g_all)
    if g_all < 2:
        raise StatsError(
            f"The between-subjects factor '{b_label}' has only one group; use the one-way repeated-measures "
            "ANOVA instead."
        )
    if (counts == 0).any():
        lost = [b_levels[j] for j in np.flatnonzero(counts == 0)]
        raise StatsError(
            f"No subject with complete data is left in group(s) {lost}. Every group needs at least one "
            f"subject measured at every level of {w_label}."
        )
    n, g = y.shape[0], g_all
    if n - g < 1:
        raise StatsError(
            "There is only one subject per group, so the variation between subjects cannot be estimated. "
            "Each group needs at least two subjects."
        )
    q = p - 1
    warnings_out: list[str] = []
    if (counts < 2).any():
        warnings_out.append("At least one group has a single subject; that group contributes nothing to the error estimates.")
    balanced = bool(counts.min() == counts.max())

    grand = float(y.mean())
    ss_total = float(((y - grand) ** 2).sum())
    if float(np.ptp(y)) <= 1e-12 * max(1.0, float(np.max(np.abs(y)))):
        ss_total = 0.0
    tiny = 1e-12 * ss_total if ss_total > 0 else math.inf
    scores = _analyse_scores(y @ _orthonormal_contrasts(p), codes, g, ss_type, hf_variant, tiny)
    if scores.ss_error <= tiny:
        scores.ss_error = 0.0
    z0 = y.sum(axis=1) / math.sqrt(p)
    group_z0 = np.array([z0[codes == j].mean() for j in range(g)])
    ss_between = float((counts * (group_z0 - z0.mean()) ** 2).sum()) if ss_total > 0 else 0.0
    ss_subjects = float(((z0 - group_z0[codes]) ** 2).sum())
    if ss_subjects <= tiny:
        ss_subjects = 0.0
    ss_within = scores.ss_intercept if ss_total > 0 else 0.0
    ss_inter = scores.ss_group if ss_total > 0 else 0.0
    df_subjects, df_resid = n - g, q * (n - g)
    ms_subjects, ms_resid = ss_subjects / df_subjects, scores.ss_error / df_resid

    inter_name = f"{b_label} x {w_label}"
    subj_name = f"Subjects within {b_label}"
    row_b = _plain_row(b_label, "between", ss_between, g - 1, ss_total, ms_subjects, df_subjects, subj_name)
    row_b["partial_eta_sq"] = ss_between / (ss_between + ss_subjects) if (ss_between + ss_subjects) > 0 else None
    row_w = _within_row(w_label, "within", ss_within, q, scores, correction, ss_total, "Residual")
    row_i = _within_row(inter_name, "interaction", ss_inter, q * (g - 1), scores, correction, ss_total, "Residual")
    rows = [
        row_b,
        _plain_row(subj_name, "subjects", ss_subjects, df_subjects, ss_total, ms_resid, df_resid, "Residual"),
        row_w,
        row_i,
        {"source": "Residual", "kind": "residual", "ss": scores.ss_error, "df": df_resid, "ms": ms_resid,
         "f": None, "p": None, "df_error": None, "error_term": None, "p_reported": None, "partial_eta_sq": None,
         "percent_of_variation": 100.0 * scores.ss_error / ss_total if ss_total > 0 else None},
        {"source": "Total", "kind": "total", "ss": ss_total, "df": n * p - 1, "ms": None, "f": None, "p": None,
         "df_error": None, "error_term": None, "p_reported": None, "partial_eta_sq": None,
         "percent_of_variation": 100.0 if ss_total > 0 else None},
    ]
    used, fallback = _resolve_correction(correction, scores)
    if fallback:
        warnings_out.append(fallback)
    if ss_total == 0:
        warnings_out.append("All values are identical: there is no variation to analyse, so no F tests are possible.")
    else:
        if row_w["f"] is None:
            warnings_out.append(
                f"The within-subject residual variation is zero, so {w_label} and the interaction cannot be tested."
            )
        if row_b["f"] is None:
            warnings_out.append(
                f"Subjects within a group do not differ in their averages, so {b_label} cannot be tested."
            )
    if not balanced:
        warnings_out.append(
            f"Group sizes are unequal ({', '.join(f'{lv}: {int(c)}' for lv, c in zip(b_levels, counts))}). "
            + ("Type III sums of squares were used: the test of " + w_label + " gives every group equal weight."
               if ss_type == 3 else
               "Type II sums of squares were used: the test of " + w_label + " weights groups by their size.")
            + " Mixed ANOVA is more sensitive to unequal covariance matrices when group sizes differ."
        )

    # ---------------------------------------------------------------- comparisons
    comparisons: list[Comparison] = []
    family_text = correction_label = None
    if method != "none":
        raws: list[_Raw] = []
        if posthoc_compare == "between_per_level":
            for k_lv, lv in enumerate(w_levels):
                col = y[:, k_lv]
                gm = np.array([col[codes == j].mean() for j in range(g)])
                mse = float(((col - gm[codes]) ** 2).sum()) / (n - g)
                for a, b in itertools.combinations(range(g), 2):
                    est = float(gm[a] - gm[b])
                    se = math.sqrt(mse * (1.0 / counts[a] + 1.0 / counts[b])) if mse > 0 else 0.0
                    raws.append(_Raw(
                        f"{lv}: {b_levels[a]}", f"{lv}: {b_levels[b]}", f"{w_label} = {lv}",
                        {within: lv, between: b_levels[a]}, {within: lv, between: b_levels[b]},
                        est, se, float(n - g),
                        _smd_pooled(est, se, math.sqrt(mse), n - g, target=f"{lv}: {b_levels[a]} vs {b_levels[b]}",
                                    ci_method="noncentral t; standardized by the pooled within-group SD at this level")
                        if mse > 0 else None,
                    ))
            family_text = (
                f"{len(raws)} comparisons: the groups of {b_label} compared at each level of {w_label} "
                f"(unpaired t tests on that level's data, error pooled over the {g} groups, df = {n - g})"
            )
        elif posthoc_compare == "within_per_group":
            for j, grp in enumerate(b_levels):
                block = y[codes == j]
                for a, b in itertools.combinations(range(p), 2):
                    raws.append(_paired_raw(
                        block[:, a], block[:, b], f"{grp}: {w_levels[a]}", f"{grp}: {w_levels[b]}",
                        f"{b_label} = {grp}", {between: grp, within: w_levels[a]}, {between: grp, within: w_levels[b]},
                    ))
            family_text = (
                f"{len(raws)} comparisons: the levels of {w_label} compared within each group of {b_label} "
                "(paired t tests on that group's subjects)"
            )
        elif posthoc_compare == "between_main":
            subj_mean = y.mean(axis=1)
            gm = np.array([subj_mean[codes == j].mean() for j in range(g)])
            mse = float(((subj_mean - gm[codes]) ** 2).sum()) / (n - g)
            for a, b in itertools.combinations(range(g), 2):
                est = float(gm[a] - gm[b])
                se = math.sqrt(mse * (1.0 / counts[a] + 1.0 / counts[b])) if mse > 0 else 0.0
                raws.append(_Raw(
                    b_levels[a], b_levels[b], f"Main effect of {b_label}",
                    {between: b_levels[a]}, {between: b_levels[b]}, est, se, float(n - g),
                    _smd_pooled(est, se, math.sqrt(mse), n - g, target=f"{b_levels[a]} vs {b_levels[b]}",
                                ci_method="noncentral t; standardized by the pooled SD of the subject averages")
                    if mse > 0 else None,
                ))
            family_text = (
                f"all {len(raws)} pairwise comparisons among the {g} groups of {b_label}, using each subject's "
                f"average over {w_label} (error pooled over groups, df = {n - g})"
            )
        else:  # within_main
            for a, b in itertools.combinations(range(p), 2):
                diff = y[:, a] - y[:, b]
                gm = np.array([diff[codes == j].mean() for j in range(g)])
                var = float(((diff - gm[codes]) ** 2).sum()) / (n - g)
                if var <= 1e-24 * max(1.0, float(np.max(np.abs(y))) ** 2):
                    var = 0.0
                if ss_type == 3:
                    est, se = float(gm.mean()), math.sqrt(var * float(np.sum(1.0 / (g**2 * counts))))
                else:
                    est, se = float(diff.mean()), math.sqrt(var / n)
                raws.append(_Raw(
                    w_levels[a], w_levels[b], f"Main effect of {w_label}",
                    {within: w_levels[a]}, {within: w_levels[b]}, est, se, float(n - g),
                    _smd_pooled(est, se, math.sqrt(var), n - g, target=f"{w_levels[a]} vs {w_levels[b]}",
                                name="Cohen's d_z (paired)", correct=False,
                                ci_method="noncentral t; standardized by the pooled within-group SD of the differences")
                    if var > 0 else None,
                ))
            family_text = (
                f"all {len(raws)} pairwise comparisons among the {p} levels of {w_label}, on within-subject "
                f"differences averaged over groups ({'unweighted' if ss_type == 3 else 'weighted by group size'}; "
                f"df = {n - g})"
            )
        comparisons, untestable = _assemble(raws, method, alpha)
        if untestable:
            warnings_out.append(
                f"{untestable} comparison(s) could not be tested (no variation, or too few subjects in a group)."
            )
        correction_label = METHOD_LABELS[method]
        family_text += "; treated as one family."

    # ---------------------------------------------------------------- assumptions
    subj_mean = y.mean(axis=1)
    group_mean = np.array([subj_mean[codes == j].mean() for j in range(g)])
    cell_mean = np.vstack([y[codes == j].mean(axis=0) for j in range(g)])
    resid_within = y - subj_mean[:, None] - cell_mean[codes] + group_mean[codes][:, None]
    assumptions = [
        AssumptionCheck(
            name="Design", status="info",
            detail=f"Each subject belongs to one group of {b_label} and was measured at every level of "
                   f"{w_label}; {n} subjects with complete data were analysed. Subjects must be independent.",
        ),
        _sphericity_check(f"{w_label} and {inter_name}", scores),
        _normality_check(resid_within, target="within-subject residuals"),
        _normality_check(subj_mean - group_mean[codes], target="subject averages (between-subjects part)"),
    ]
    level_checks = [
        _brown_forsythe({b_levels[j]: y[codes == j, k_lv] for j in range(g)}, target=f"{w_label} = {lv}",
                        name="Equal variances between groups (Brown-Forsythe)")
        for k_lv, lv in enumerate(w_levels)
    ]
    if len(level_checks) <= 6:
        assumptions.extend(level_checks)
    else:
        tested = [c for c in level_checks if c.p_value is not None]
        failing = [c.target for c in tested if c.p_value < 0.05]
        if not tested:
            assumptions.append(AssumptionCheck(
                name="Equal variances between groups (Brown-Forsythe)", status="warn",
                detail="Equal variances between groups could not be checked at any level.",
            ))
        else:
            worst = min(tested, key=lambda c: c.p_value)
            assumptions.append(AssumptionCheck(
                name="Equal variances between groups (Brown-Forsythe)", target=worst.target,
                statistic=worst.statistic, p_value=worst.p_value, status="warn" if failing else "pass",
                detail=(f"Tested separately at each of the {len(tested)} levels of {w_label}: "
                        + (f"{len(failing)} level(s) show unequal spread between groups "
                           f"({', '.join(str(t) for t in failing[:4])}{', ...' if len(failing) > 4 else ''}); "
                           if failing else "no level shows evidence of unequal spread; ")
                        + f"smallest {p_phrase(worst.p_value)} (not adjusted for the number of levels)."),
            ))
    assumptions.append(AssumptionCheck(
        name="Equal covariance matrices across groups", status="info",
        detail="Mixed ANOVA also assumes that the pattern of variances and correlations across the repeated "
               "levels is the same in every group. This is not formally tested here; it matters most when "
               "group sizes are unequal.",
    ))
    assumptions.append(AssumptionCheck(
        name="Balanced groups", status="pass" if balanced else "warn",
        detail=(f"Every group has {int(counts[0])} subjects." if balanced
                else "Group sizes differ: " + ", ".join(f"{lv}: {int(c)}" for lv, c in zip(b_levels, counts)) + "."),
    ))

    # ---------------------------------------------------------------- effect sizes
    error_all = ss_subjects + scores.ss_error
    effects = [
        _partial_eta(ss_inter, scores.ss_error, q * (g - 1), df_resid, inter_name, q),
        _partial_eta(ss_within, scores.ss_error, q, df_resid, w_label, q),
        _effect(es.partial_eta_squared(ss_between, ss_subjects, g - 1, df_subjects, target=b_label)),
        _generalized_eta(ss_inter, error_all, inter_name),
        _generalized_eta(ss_within, error_all, w_label),
        _generalized_eta(ss_between, error_all, b_label),
    ]

    descriptives, cell_means, n_map = [], [], {}
    for j, grp in enumerate(b_levels):
        for k_lv, lv in enumerate(w_levels):
            desc = _describe(y[codes == j, k_lv])
            name = f"{grp}: {lv}"
            n_map[name] = desc["n"]
            descriptives.append({between: grp, within: lv, "group": name, **desc})
            cell_means.append({between: grp, within: lv, "mean": desc["mean"], "sd": desc["sd"],
                               "sem": desc["sem"], "n": desc["n"]})

    def corr_text(row: dict[str, Any]) -> str:
        if used == "none" or row["f"] is None:
            return ""
        eps = scores.eps_gg if used == "gg" else scores.eps_hf
        return f", {_CORRECTION_LABELS[used]} corrected (epsilon = {fmt(eps, 3)})"

    sentences = [
        f"Mixed-design ANOVA of {value} with {b_label} as between-subjects factor ({g} groups) and {w_label} "
        f"as within-subjects factor ({p} levels), {n} subjects."
    ]
    for what, row in (("Interaction", row_i), ("Main effect of", row_w)):
        if row["f"] is None:
            sentences.append(f"{what} {row['source']}: could not be tested.")
        else:
            sentences.append(
                f"{what} {row['source']}: {sig_phrase(row['p_reported'], alpha)}, "
                f"{_f_text(row['f'], row['df_reported'], row['df_error_reported'])}{corr_text(row)}, "
                f"partial eta squared = {fmt(row['partial_eta_sq'], 3)}."
            )
    if row_b["f"] is None:
        sentences.append(f"Main effect of {b_label}: could not be tested.")
    else:
        sentences.append(
            f"Main effect of {b_label}: {sig_phrase(row_b['p'], alpha)}, "
            f"{_f_text(row_b['f'], g - 1, df_subjects)}, partial eta squared = {fmt(row_b['partial_eta_sq'], 3)}."
        )
    if row_i["p_reported"] is not None and row_i["p_reported"] < alpha:
        sentences.append(
            f"The significant interaction is evidence that the change across {w_label} differs between the "
            f"groups of {b_label}; interpret the main effects with caution."
        )
    if comparisons:
        n_sig = sum(1 for c in comparisons if c.p_adjusted is not None and c.p_adjusted < alpha)
        sentences.append(
            f"Multiple comparisons ({correction_label}): {n_sig} of {len(comparisons)} were significant at "
            f"alpha = {alpha:g}."
        )
    details = [f"Type {'III' if ss_type == 3 else 'II'} sums of squares"] if not balanced else []
    details.append(f"alpha = {alpha:g}")
    methods = (
        f"Data were analysed by mixed-design (split-plot) ANOVA with {b_label} as between-subjects factor and "
        f"{w_label} as within-subjects factor"
        + (f", with the {_CORRECTION_LABELS[used]} correction for non-sphericity" if used != "none" else "")
        + f" ({'; '.join(details)})"
        + (f", followed by {correction_label}-adjusted t tests" if comparisons else "") + "."
    )
    citations = [_CITE_SPLIT, _CITE_MAUCHLY, _CITE_BOX, _CITE_GG, _CITE_HF, _CITE_LECOUTRE, _CITE_GES]
    if method != "none":
        citations.append(_CITE_ADJUST[method])
    extra: dict[str, Any] = {
        "sphericity": _sphericity_extra(f"{w_label}; {inter_name}", scores),
        "correction_requested": correction, "correction_applied": used,
        "cell_means": cell_means, "factors": [between, within],
        "factor_labels": {between: b_label, within: w_label},
        "levels": {between: list(b_levels), within: list(w_levels)},
        "between": between, "within": within,
        "n_subjects": n, "n_per_group": {lv: int(c) for lv, c in zip(b_levels, counts)},
        "n_observations": n * p, "ss_type": ss_type, "balanced": balanced,
        "headline_term": inter_name,
    }
    if comparisons:
        extra["comparisons_note"] = _ci_note(method)
    return StatResult(
        analysis="anova_mixed",
        test_name="Mixed-design (split-plot) two-way ANOVA",
        why_chosen=(
            "Separate groups of subjects were each measured repeatedly (for example treatment groups followed "
            "over time). A mixed ANOVA tests the group difference between subjects, the repeated factor within "
            "subjects, and whether the groups change differently (the interaction)."
        ),
        assumptions=assumptions,
        statistic_name="F", statistic=row_i["f"], df=row_i["df_reported"], df2=row_i["df_error_reported"],
        p_value=row_i["p_reported"], alpha=alpha,
        effect_sizes=effects, n=n_map, n_total=n,
        descriptives=descriptives, anova_table=rows,
        comparisons=comparisons, comparison_family=family_text, correction=correction_label,
        warnings=warnings_out, data_notes=list(wide.notes),
        interpretation=" ".join(sentences), methods_text=methods, citations=citations, extra=extra,
    )


# =============================================================================
# Two-way repeated-measures ANOVA (both factors within subjects)
# =============================================================================
def anova_rm_twoway(
    df: pd.DataFrame,
    value: str,
    subject: str,
    within_a: str,
    within_b: str,
    *,
    correction: str = "auto",
    alpha: float = 0.05,
    hf_variant: str = "original",
) -> StatResult:
    """Two-way repeated-measures ANOVA: every subject is measured at every combination
    of two within-subjects factors (one value per combination).

    Each effect (A, B, A x B) is tested against its own interaction with subjects, using
    Kronecker products of orthonormal contrasts, and gets its own sphericity estimates
    and corrected P values. No multiple comparisons are produced by this analysis.
    """
    alpha = check_alpha(alpha)
    correction = _check_correction(correction)
    hf_variant = _check_hf_variant(hf_variant)
    for c in (value, subject, within_a, within_b):
        if not isinstance(df, pd.DataFrame) or c not in df.columns:
            raise StatsError(f"Column '{c}' was not found in the data.")
    if len({value, subject, within_a, within_b}) != 4:
        raise StatsError("Each column can play only one role (outcome, subject, factor A or factor B).")
    work = df[[value, subject, within_a, within_b]]
    ok = (np.isfinite(pd.to_numeric(work[value], errors="coerce").to_numpy(dtype=float))
          & work[[subject, within_a, within_b]].notna().all(axis=1).to_numpy())
    notes: list[str] = []
    if int((~ok).sum()):
        notes.append(f"{int((~ok).sum())} row(s) with a missing or non-numeric value were ignored.")
    work = work.loc[ok]
    if work.empty:
        raise StatsError("There are no complete rows of data to analyse.")
    a_codes, a_levels = _encode_factor(work[within_a])
    b_codes, b_levels = _encode_factor(work[within_b])
    na, nb = len(a_levels), len(b_levels)
    if na < 2 or nb < 2:
        raise StatsError("Both within-subjects factors need at least two levels.")
    combined = pd.DataFrame({
        "y": pd.to_numeric(work[value], errors="coerce").to_numpy(dtype=float),
        "s": work[subject].to_numpy(),
        "cell": [f"{i}|{j}" for i, j in zip(a_codes, b_codes)],
    })
    cell_levels = [f"{i}|{j}" for i in range(na) for j in range(nb)]
    combined["cell"] = pd.Categorical(combined["cell"], categories=cell_levels)
    wide = _pivot_mixed(combined, "y", "s", None, "cell", f"{within_a} x {within_b}")
    notes += wide.notes
    order = [wide.within_levels.index(c) if c in wide.within_levels else -1 for c in cell_levels]
    if -1 in order:
        raise StatsError(
            "Some combinations of the two factors have no data at all. Every subject must be measured at "
            "every combination of the two within-subjects factors."
        )
    y = wide.y[:, order]
    n = y.shape[0]
    if n < 2:
        raise StatsError(
            f"Only {n} subject(s) have a value at every combination of {within_a} and {within_b}; at least 2 "
            "are needed."
        )
    grand = float(y.mean())
    ss_total = float(((y - grand) ** 2).sum())
    if float(np.ptp(y)) <= 1e-12 * max(1.0, float(np.max(np.abs(y)))):
        ss_total = 0.0
    tiny = 1e-12 * ss_total if ss_total > 0 else math.inf
    ca, cb = _orthonormal_contrasts(na), _orthonormal_contrasts(nb)
    one_a, one_b = np.full((na, 1), 1.0 / math.sqrt(na)), np.full((nb, 1), 1.0 / math.sqrt(nb))
    zeros = np.zeros(n, dtype=int)
    inter_name = f"{within_a} x {within_b}"
    specs = [(within_a, np.kron(ca, one_b)), (within_b, np.kron(one_a, cb)), (inter_name, np.kron(ca, cb))]
    subject_means = y.mean(axis=1)
    ss_subjects = float(na * nb * ((subject_means - grand) ** 2).sum()) if ss_total > 0 else 0.0
    rows: list[dict[str, Any]] = []
    effects: list[EffectSize] = []
    assumptions = [AssumptionCheck(
        name="Repeated measures on the same subjects", status="info",
        detail=f"Each of the {n} subjects was measured once at every combination of {within_a} ({na} levels) "
               f"and {within_b} ({nb} levels). Subjects must be independent of each other.",
    )]
    warnings_out: list[str] = []
    sphericity = []
    all_scores = []
    for _, contrast in specs:
        sc = _analyse_scores(y @ contrast, zeros, 1, 3, hf_variant, tiny)
        if sc.ss_error <= tiny:
            sc.ss_error = 0.0
        all_scores.append(sc)
    error_total = ss_subjects + sum(sc.ss_error for sc in all_scores)
    for (name, _), sc in zip(specs, all_scores):
        ss = sc.ss_intercept if ss_total > 0 else 0.0
        err_name = f"{name} x Subjects"
        row = _within_row(name, "within" if name != inter_name else "interaction", ss, sc.q, sc, correction,
                          ss_total, err_name)
        rows.append(row)
        rows.append({"source": err_name, "kind": "error", "ss": sc.ss_error, "df": sc.q * (n - 1),
                     "ms": sc.ss_error / (sc.q * (n - 1)), "f": None, "p": None, "df_error": None,
                     "error_term": None, "p_reported": None, "partial_eta_sq": None,
                     "percent_of_variation": 100.0 * sc.ss_error / ss_total if ss_total > 0 else None})
        effects.append(_partial_eta(ss, sc.ss_error, sc.q, sc.q * (n - 1), name, sc.q))
        effects.append(_generalized_eta(ss, error_total, name))
        assumptions.append(_sphericity_check(name, sc))
        sphericity.append(_sphericity_extra(name, sc))
        _, fallback = _resolve_correction(correction, sc)
        if fallback and fallback not in warnings_out:
            warnings_out.append(fallback)
        if ss_total > 0 and row["f"] is None:
            warnings_out.append(f"The error variation for {name} is zero, so it cannot be tested.")
    rows.append(_plain_row("Subjects", "subjects", ss_subjects, n - 1, ss_total))
    rows.append({"source": "Total", "kind": "total", "ss": ss_total, "df": n * na * nb - 1, "ms": None,
                 "f": None, "p": None, "df_error": None, "error_term": None, "p_reported": None,
                 "partial_eta_sq": None, "percent_of_variation": 100.0 if ss_total > 0 else None})
    if ss_total == 0:
        warnings_out.append("All values are identical: there is no variation to analyse, so no F tests are possible.")
    cell_mean = y.mean(axis=0)
    a_idx = np.repeat(np.arange(na), nb)
    b_idx = np.tile(np.arange(nb), na)
    a_eff = np.array([cell_mean[a_idx == i].mean() for i in range(na)])
    b_eff = np.array([cell_mean[b_idx == j].mean() for j in range(nb)])
    subj_a = np.column_stack([y[:, a_idx == i].mean(axis=1) for i in range(na)])
    subj_b = np.column_stack([y[:, b_idx == j].mean(axis=1) for j in range(nb)])
    resid = (y - subj_a[:, a_idx] - subj_b[:, b_idx] + subject_means[:, None]
             - cell_mean[None, :] + a_eff[a_idx][None, :] + b_eff[b_idx][None, :] - grand)
    assumptions.append(_normality_check(resid, target="residuals of the highest-order error term"))

    descriptives, cell_means, n_map = [], [], {}
    for col, (i, j) in enumerate(itertools.product(range(na), range(nb))):
        desc = _describe(y[:, col])
        name = f"{a_levels[i]}: {b_levels[j]}"
        n_map[name] = desc["n"]
        descriptives.append({within_a: a_levels[i], within_b: b_levels[j], "group": name, **desc})
        cell_means.append({within_a: a_levels[i], within_b: b_levels[j], "mean": desc["mean"], "sd": desc["sd"],
                           "sem": desc["sem"], "n": desc["n"]})
    headline = rows[4]
    sentences = [
        f"Two-way repeated-measures ANOVA of {value} with {within_a} ({na} levels) and {within_b} ({nb} levels) "
        f"measured in the same {n} subjects."
    ]
    for row, sc in zip((rows[4], rows[0], rows[2]), (all_scores[2], all_scores[0], all_scores[1])):
        what = "Interaction" if row["kind"] == "interaction" else "Main effect of"
        if row["f"] is None:
            sentences.append(f"{what} {row['source']}: could not be tested.")
            continue
        used, _ = _resolve_correction(correction, sc)
        corr = ""
        if used != "none":
            corr = f", {_CORRECTION_LABELS[used]} corrected (epsilon = {fmt(sc.eps_gg if used == 'gg' else sc.eps_hf, 3)})"
        sentences.append(
            f"{what} {row['source']}: {sig_phrase(row['p_reported'], alpha)}, "
            f"{_f_text(row['f'], row['df_reported'], row['df_error_reported'])}{corr}, "
            f"partial eta squared = {fmt(row['partial_eta_sq'], 3)}."
        )
    return StatResult(
        analysis="anova_rm_twoway",
        test_name="Two-way repeated-measures ANOVA",
        why_chosen=(
            "Every subject was measured at every combination of two factors (for example each treatment at "
            "each time point), so both factors are compared within subjects."
        ),
        assumptions=assumptions,
        statistic_name="F", statistic=headline["f"], df=headline["df_reported"],
        df2=headline["df_error_reported"], p_value=headline["p_reported"], alpha=alpha,
        effect_sizes=effects, n=n_map, n_total=n, descriptives=descriptives, anova_table=rows,
        warnings=warnings_out, data_notes=notes,
        interpretation=" ".join(sentences),
        methods_text=(
            f"Data were analysed by two-way repeated-measures ANOVA ({within_a} x {within_b}, both within "
            "subjects), each effect tested against its interaction with subjects"
            + (" with Greenhouse-Geisser correction for non-sphericity where an effect has more than one "
               "degree of freedom" if correction in {"auto", "gg"} else
               (" with Huynh-Feldt correction" if correction == "hf" else ""))
            + f" (alpha = {alpha:g})."
        ),
        citations=[_CITE_SPLIT, _CITE_MAUCHLY, _CITE_BOX, _CITE_GG, _CITE_HF, _CITE_LECOUTRE, _CITE_GES],
        extra={
            "sphericity": sphericity, "correction_requested": correction,
            "cell_means": cell_means, "factors": [within_a, within_b],
            "levels": {within_a: list(a_levels), within_b: list(b_levels)},
            "n_subjects": n, "n_observations": n * na * nb, "headline_term": inter_name,
        },
    )


# =============================================================================
# Registered (table-level) analyses
# =============================================================================
_HELP_ALPHA = "The P value threshold for calling a result significant. 0.05 is the usual choice."
_HELP_RM_POSTHOC = (
    "Follow-up tests that say which conditions differ, with the P values adjusted for the number of "
    "comparisons. Holm is a good default (it controls false positives like Bonferroni but is more "
    "powerful); fdr_bh controls the false discovery rate; 'none' skips the comparisons."
)


@register(
    key="anova_rm_oneway",
    label="Repeated-measures one-way ANOVA",
    category="Repeated measures",
    tables=("column",),
    params=(
        ParamSpec("groups", "Conditions to compare", kind="groups", default=None, optional=True,
                  help="Which columns (conditions or time points) to include. Leave empty to use all columns. "
                       "Each row must be one subject measured in every chosen column."),
        ParamSpec("correction", "Sphericity correction", kind="choice", default="auto",
                  choices=CORRECTIONS, help=_HELP_CORRECTION),
        ParamSpec("posthoc", "Multiple comparisons", kind="choice", default="holm",
                  choices=RM_POSTHOC_METHODS, help=_HELP_RM_POSTHOC),
        ParamSpec("control", "Control condition", kind="group", default=None, optional=True,
                  help="Leave empty to compare every pair of conditions. Choose a column to compare every "
                       "other condition only with that control (fewer comparisons, more power)."),
        ParamSpec("alpha", "Significance level (alpha)", kind="float", default=0.05, help=_HELP_ALPHA),
    ),
    description=(
        "Compares three or more conditions or time points measured in the same subjects (each row is one "
        "subject, each column one condition). Rows with a missing value are left out."
    ),
    citation=_CITE_GG,
)
def _run_anova_rm_oneway(table: Any, groups: Any = None, correction: str = "auto", posthoc: str | None = "holm",
                         control: str | None = None, alpha: float = 0.05) -> StatResult:
    names = list(table.group_names)
    if groups:
        chosen = [groups] if isinstance(groups, str) else [str(g) for g in groups]
        unknown = [g for g in chosen if g not in names]
        if unknown:
            raise StatsError(f"Unknown column(s): {unknown}. Available: {names}.")
        if len(set(chosen)) != len(chosen):
            raise StatsError("Each column can be chosen only once.")
        names = chosen
    if len(names) < 2:
        raise StatsError("Repeated-measures ANOVA needs at least two columns (conditions).")
    frame = table.analysis_frame()[names]
    return anova_rm_oneway(frame.to_numpy(dtype=float), names, correction=correction, posthoc=posthoc,
                           control=control, alpha=alpha)


def _guess_mixed_columns(table: Any, value: str | None, subject: str | None, between: str | None,
                         within: str | None) -> tuple[str, str, str, str, list[str]]:
    """Fill in missing column roles of a Multiple variables table for a mixed design."""
    columns = list(table.data.columns)
    for name in (value, subject, between, within):
        if name is not None and name not in columns:
            raise StatsError(f"Column '{name}' is not in the table. Available columns: {columns}.")
    notes: list[str] = []
    given = {c for c in (value, subject, between, within) if c is not None}
    if value is None:
        candidates = [c for c in table.numeric_columns if c not in given]
        if not candidates:
            raise StatsError("The table has no numeric column to analyse. Choose the outcome column.")
        frame = table.frame(dropna=False)
        value = max(candidates, key=lambda c: frame[c].nunique())
        notes.append(f"Outcome column chosen automatically: '{value}' (the numeric column with the most "
                     "distinct values).")
        given.add(value)
    frame = table.frame(dropna=False)
    pool = [c for c in table.categorical_columns if c not in given]
    pool += [c for c in table.numeric_columns if c not in given]
    if subject is None:
        if not pool:
            raise StatsError("Could not find a subject column. Choose the column that identifies each subject.")
        subject = max(pool, key=lambda c: (frame[c].nunique(), c in table.categorical_columns))
        pool.remove(subject)
        notes.append(f"Subject column chosen automatically: '{subject}' (the column with the most distinct values).")
    varies = {c: bool((frame.groupby(subject, observed=True)[c].nunique() > 1).any()) for c in pool}
    if between is None:
        options = [c for c in pool if not varies[c] and c != within]
        if not options:
            raise StatsError(
                "Could not find a between-subjects column (one that is constant for each subject). "
                "Choose the columns yourself."
            )
        between = options[0]
        pool.remove(between)
        notes.append(f"Between-subjects factor chosen automatically: '{between}' (constant within each subject).")
    if within is None:
        options = [c for c in pool if varies.get(c) and c != between]
        if not options:
            raise StatsError(
                "Could not find a within-subjects column (one that changes within each subject, such as "
                "time). Choose the columns yourself."
            )
        within = options[0]
        notes.append(f"Within-subjects factor chosen automatically: '{within}' (varies within each subject).")
    return value, subject, between, within, notes


@register(
    key="anova_mixed",
    label="Mixed-design ANOVA (repeated measures by group)",
    category="Repeated measures",
    tables=("grouped", "multiple_variables"),
    params=(
        ParamSpec("value", "Outcome column", kind="column", default=None, optional=True,
                  help="Multiple variables tables only: the numeric column that was measured. Leave empty to "
                       "pick it automatically. Ignored for Grouped tables."),
        ParamSpec("subject", "Subject column", kind="column", default=None, optional=True,
                  help="Multiple variables tables only: the column that identifies each subject (animal, "
                       "patient, well). Leave empty to pick it automatically."),
        ParamSpec("between", "Between-subjects factor", kind="column", default=None, optional=True,
                  help="Multiple variables tables only: the grouping column in which each subject belongs to "
                       "just one group (for example treatment). Leave empty to pick it automatically."),
        ParamSpec("within", "Within-subjects factor", kind="column", default=None, optional=True,
                  help="Multiple variables tables only: the column for the repeated factor that every subject "
                       "goes through (for example time point). Leave empty to pick it automatically."),
        ParamSpec("correction", "Sphericity correction", kind="choice", default="auto",
                  choices=CORRECTIONS, help=_HELP_CORRECTION),
        ParamSpec("ss_type", "Sums of squares type", kind="choice", default=3, choices=(3, 2),
                  help="Only matters when groups have different numbers of subjects. Type III (default, as in "
                       "most other statistics packages) gives every group equal weight when testing the "
                       "repeated factor; Type II weights groups by their size."),
        ParamSpec("posthoc", "Multiple comparisons", kind="choice", default="none",
                  choices=RM_POSTHOC_METHODS,
                  help="Follow-up t tests with P values adjusted for the number of comparisons. Holm is a good "
                       "default; 'none' skips them."),
        ParamSpec("posthoc_compare", "What to compare", kind="choice", default="between_per_level",
                  choices=MIXED_COMPARE,
                  help="between_per_level: compare the groups at each repeated level (e.g. treatments at each "
                       "time point). within_per_group: compare the repeated levels inside each group. "
                       "between_main / within_main: compare the overall means of one factor."),
        ParamSpec("alpha", "Significance level (alpha)", kind="float", default=0.05, help=_HELP_ALPHA),
    ),
    description=(
        "For separate groups of subjects that are each measured repeatedly (for example treatment groups "
        "followed over time). In a Grouped table, put the repeated levels (time points) in rows and the "
        "groups in column groups; each replicate sub-column is one subject, so the first sub-column of a "
        "group must hold the same subject in every row."
    ),
    citation=_CITE_SPLIT,
)
def _run_anova_mixed(table: Any, value: str | None = None, subject: str | None = None,
                     between: str | None = None, within: str | None = None, correction: str = "auto",
                     ss_type: Any = 3, posthoc: str | None = "none",
                     posthoc_compare: str = "between_per_level", alpha: float = 0.05) -> StatResult:
    if table.kind == "grouped":
        long = table.to_long()
        result = anova_mixed(long, "value", "replicate", "column", "row", correction=correction,
                             ss_type=ss_type, posthoc=posthoc, posthoc_compare=posthoc_compare, alpha=alpha,
                             factor_labels={"row": table.row_label, "column": table.column_label})
        result.data_notes = [n for n in result.data_notes if not n.startswith("The same subject label")]
        result.data_notes.insert(
            0, f"Layout: rows = repeated levels ({table.row_label}); column groups = groups of subjects "
               f"({table.column_label}); each replicate sub-column within a group is one subject.")
        return result
    value, subject, between, within, notes = _guess_mixed_columns(table, value, subject, between, within)
    result = anova_mixed(table.frame(dropna=False), value, subject, between, within, correction=correction,
                         ss_type=ss_type, posthoc=posthoc, posthoc_compare=posthoc_compare, alpha=alpha)
    result.data_notes[:0] = notes
    return result


@register(
    key="anova_rm_twoway",
    label="Two-way repeated-measures ANOVA",
    category="Repeated measures",
    tables=("multiple_variables",),
    params=(
        ParamSpec("value", "Outcome column", kind="column", default=None, optional=True,
                  help="The numeric column that was measured. Leave empty to pick it automatically (the numeric "
                       "column with the most distinct values)."),
        ParamSpec("subject", "Subject column", kind="column", default=None, optional=True,
                  help="The column that identifies each subject. Leave empty to use the remaining column with "
                       "the most distinct values."),
        ParamSpec("within_a", "First repeated factor", kind="column", default=None, optional=True,
                  help="A factor that every subject goes through (for example treatment). Leave empty to pick "
                       "it automatically."),
        ParamSpec("within_b", "Second repeated factor", kind="column", default=None, optional=True,
                  help="The other factor that every subject goes through (for example time point). Leave "
                       "empty to pick it automatically."),
        ParamSpec("correction", "Sphericity correction", kind="choice", default="auto",
                  choices=CORRECTIONS, help=_HELP_CORRECTION),
        ParamSpec("alpha", "Significance level (alpha)", kind="float", default=0.05, help=_HELP_ALPHA),
    ),
    description=(
        "For designs in which every subject is measured at every combination of two factors (for example "
        "each subject receives every treatment and is measured at every time point)."
    ),
    citation=_CITE_GG,
)
def _run_anova_rm_twoway(table: Any, value: str | None = None, subject: str | None = None,
                         within_a: str | None = None, within_b: str | None = None, correction: str = "auto",
                         alpha: float = 0.05) -> StatResult:
    columns = list(table.data.columns)
    for name in (value, subject, within_a, within_b):
        if name is not None and name not in columns:
            raise StatsError(f"Column '{name}' is not in the table. Available columns: {columns}.")
    notes: list[str] = []
    given = {c for c in (value, subject, within_a, within_b) if c is not None}
    if value is None:
        candidates = [c for c in table.numeric_columns if c not in given]
        if not candidates:
            raise StatsError("The table has no numeric column to analyse. Choose the outcome column.")
        frame = table.frame(dropna=False)
        value = max(candidates, key=lambda c: frame[c].nunique())
        given.add(value)
        notes.append(f"Outcome column chosen automatically: '{value}' (the numeric column with the most "
                     "distinct values).")
    frame = table.frame(dropna=False)
    pool = [c for c in table.categorical_columns if c not in given]
    pool += [c for c in table.numeric_columns if c not in given]
    if subject is None:
        if not pool:
            raise StatsError("Could not find a subject column. Choose the column that identifies each subject.")
        subject = max(pool, key=lambda c: (frame[c].nunique(), c in table.categorical_columns))
        pool.remove(subject)
        notes.append(f"Subject column chosen automatically: '{subject}' (the column with the most distinct values).")
    picked = []
    for role, name in (("First repeated factor", within_a), ("Second repeated factor", within_b)):
        if name is None:
            if not pool:
                raise StatsError(f"Could not choose a column for the {role.lower()}. Choose the columns yourself.")
            name = pool.pop(0)
            notes.append(f"{role} chosen automatically: '{name}'.")
        elif name in pool:
            pool.remove(name)
        picked.append(name)
    result = anova_rm_twoway(frame, value, subject, picked[0], picked[1], correction=correction, alpha=alpha)
    result.data_notes[:0] = notes
    return result

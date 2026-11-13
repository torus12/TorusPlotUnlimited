"""Multiple-comparison p-value adjustments (implemented here, cross-checked in tests).

Methods
-------
``bonferroni``  Dunn OJ (1961) J Am Stat Assoc 56:52-64.
``sidak``       Sidak Z (1967) J Am Stat Assoc 62:626-633.
``holm``        Holm S (1979) Scand J Stat 6:65-70 (step-down Bonferroni).
``holm-sidak``  Holm's step-down procedure using Sidak's per-step adjustment.
``fdr_bh``      Benjamini Y, Hochberg Y (1995) J R Stat Soc B 57:289-300.
``none``        No adjustment.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

METHODS = ("none", "bonferroni", "sidak", "holm", "holm-sidak", "fdr_bh")

METHOD_LABELS = {
    "none": "none (unadjusted)",
    "bonferroni": "Bonferroni",
    "sidak": "Šidák",
    "holm": "Holm",
    "holm-sidak": "Holm-Šidák",
    "fdr_bh": "Benjamini-Hochberg FDR",
}


def adjust_pvalues(pvalues: Iterable[float], method: str = "holm") -> np.ndarray:
    """Return multiplicity-adjusted p values (same order as the input).

    NaN entries stay NaN and are not counted in the family size.
    """
    p = np.asarray(list(pvalues), dtype=float)
    key = method.lower().replace("_", "-").replace("š", "s").replace("á", "a")
    key = {"fdr-bh": "fdr_bh", "bh": "fdr_bh", "benjamini-hochberg": "fdr_bh", "fdr": "fdr_bh",
           "holm-bonferroni": "holm", "holmsidak": "holm-sidak"}.get(key, key)
    if key not in METHODS:
        raise ValueError(f"Unknown adjustment method '{method}'. Choose from {METHODS}.")
    out = np.full(p.shape, np.nan)
    valid = ~np.isnan(p)
    m = int(valid.sum())
    if m == 0:
        return out
    pv = np.clip(p[valid], 0.0, 1.0)
    if key == "none":
        adj = pv.copy()
    elif key == "bonferroni":
        adj = np.minimum(pv * m, 1.0)
    elif key == "sidak":
        adj = -np.expm1(m * np.log1p(-pv))
    elif key in {"holm", "holm-sidak"}:
        order = np.argsort(pv, kind="stable")
        ranked = pv[order]
        k = m - np.arange(m)  # m, m-1, ..., 1
        step = np.minimum(ranked * k, 1.0) if key == "holm" else -np.expm1(k * np.log1p(-ranked))
        step = np.maximum.accumulate(step)
        adj = np.empty(m)
        adj[order] = step
    else:  # fdr_bh
        order = np.argsort(pv, kind="stable")
        ranked = pv[order]
        step = ranked * m / (np.arange(m) + 1)
        step = np.minimum.accumulate(step[::-1])[::-1]
        adj = np.empty(m)
        adj[order] = np.minimum(step, 1.0)
    out[valid] = np.clip(adj, 0.0, 1.0)
    return out

"""Deterministic figures used for golden-image tests (one per plot type and style preset).

Data come from the committed synthetic CSVs in ``examples/``; statistics results are
small hand-built ``StatResult`` objects so the images depend only on the plot layer.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from matplotlib.figure import Figure

from torusplot.data.samples import load_sample
from torusplot.plot import preset_names, render_figure
from torusplot.plot.compose import render_composition
from torusplot.spec import CompositionSpec, FigureSpec
from torusplot.stats.result import Comparison, Parameter, StatResult

DPI = 80


def tables() -> dict:
    out = {}
    for name in ("nanoparticle_size", "dose_response", "cell_uptake", "cell_uptake_cells", "survival", "paired",
                 "contingency"):
        t = load_sample(name)
        out[t.name] = t
    return out


T_SIZE = "Nanoparticle diameter (nm)"
T_DOSE = "Viability (%)"
T_UPTAKE = "Uptake (a.u.)"
T_CELLS = "Per-cell uptake"
T_SURV = "Survival study"
T_PAIRED = "Tumour volume (mm3)"
T_CONT = "Response by treatment"


def _hill(x: np.ndarray, ic50: float, slope: float) -> np.ndarray:
    return 4.0 + 94.0 / (1.0 + (x / ic50) ** slope)


def _km(time: np.ndarray, event: np.ndarray) -> dict:
    order = np.argsort(time, kind="stable")
    time, event = time[order], event[order]
    t_out, s_out, s, at_risk = [0.0], [1.0], 1.0, len(time)
    c_t, c_s = [], []
    for t in np.unique(time):
        d = int(((time == t) & (event == 1)).sum())
        c = int(((time == t) & (event == 0)).sum())
        if d:
            s *= 1.0 - d / at_risk
            t_out.append(float(t))
            s_out.append(s)
        if c:
            c_t += [float(t)] * c
            c_s += [s] * c
        at_risk -= d + c
    return {"time": t_out, "survival": s_out, "censor_time": c_t, "censor_survival": c_s,
            "ci_low": [max(v - 0.12, 0.0) for v in s_out], "ci_high": [min(v + 0.12, 1.0) for v in s_out]}


def results() -> dict[str, StatResult]:
    names = ["Bulk mixing", "Sonication", "Extrusion", "Microfluidic"]
    pvals = {(0, 1): 0.0021, (0, 2): 0.00001, (0, 3): 0.000001, (1, 2): 0.0113, (1, 3): 0.00003, (2, 3): 0.0584}
    anova = StatResult(
        id="sr_anova", analysis="anova_oneway", test_name="One-way ANOVA", statistic_name="F", statistic=52.3,
        df=3, df2=28, p_value=1.2e-11, correction="Tukey HSD",
        comparisons=[
            Comparison(id="", group_a=names[i], group_b=names[j], estimate=10.0 * (j - i), ci_low=10.0 * (j - i) - 8,
                       ci_high=10.0 * (j - i) + 8, p_adjusted=p)
            for (i, j), p in pvals.items()
        ],
    ).finalize()
    cx = np.logspace(-2, 2, 120)
    fits = {}
    for rid, series, ic50, slope in (("sr_fit_free", "Free drug", 4.2, 1.1), ("sr_fit_nano", "Nanoformulation", 0.85, 1.3)):
        y = _hill(cx, ic50, slope)
        fits[rid] = StatResult(
            id=rid, analysis="nonlinear_regression", test_name="Nonlinear regression (4PL)",
            extra={"curve": {"x": cx.tolist(), "y": y.tolist(), "ci_low": (y - 3).tolist(), "ci_high": (y + 3).tolist(),
                             "pred_low": (y - 8).tolist(), "pred_high": (y + 8).tolist()}, "series": series},
        )
    two_way = StatResult(
        id="sr_twoway", analysis="anova_twoway", test_name="Two-way ANOVA", correction="Tukey HSD",
        comparisons=[
            Comparison(id="", group_a="24 h: Free dye", group_b="24 h: Targeted NP", p_adjusted=0.00001,
                       cell_a={"row": "24 h", "column": "Free dye"}, cell_b={"row": "24 h", "column": "Targeted NP"}),
            Comparison(id="", group_a="24 h: Nanoparticle", group_b="24 h: Targeted NP", p_adjusted=0.0008,
                       cell_a={"row": "24 h", "column": "Nanoparticle"}, cell_b={"row": "24 h", "column": "Targeted NP"}),
            Comparison(id="", group_a="4 h: Free dye", group_b="4 h: Targeted NP", p_adjusted=0.003,
                       cell_a={"row": "4 h", "column": "Free dye"}, cell_b={"row": "4 h", "column": "Targeted NP"}),
            Comparison(id="", group_a="1 h: Free dye", group_b="1 h: Nanoparticle", p_adjusted=0.41,
                       cell_a={"row": "1 h", "column": "Free dye"}, cell_b={"row": "1 h", "column": "Nanoparticle"}),
        ],
    ).finalize()
    surv_table = load_sample("survival").data
    curves = {}
    for g in ["Vehicle", "Free drug", "Nano drug"]:
        sub = surv_table[surv_table["Group"] == g]
        curves[g] = _km(sub["Time (days)"].to_numpy(float), sub["Event"].to_numpy(int))
    surv = StatResult(id="sr_surv", analysis="survival", test_name="Log-rank (Mantel-Cox) test", statistic_name="chi2",
                      statistic=14.2, df=2, p_value=0.0008, extra={"curves": curves})
    paired = StatResult(
        id="sr_ttest", analysis="ttest_paired", test_name="Paired t test", statistic_name="t", statistic=6.1, df=9,
        p_value=0.0002, n={"After": 10, "Before": 10},
        parameters=[Parameter(name="Mean difference (After - Before)", value=-92.0, ci_low=-126.0, ci_high=-58.0)],
        comparisons=[Comparison(id="", group_a="Before", group_b="After", p_value=0.0002, estimate=92.0, ci_low=58.0, ci_high=126.0)],
    ).finalize()
    xs = np.linspace(0, 10, 24)
    resid = StatResult(
        id="sr_resid", analysis="linear_regression", test_name="Simple linear regression",
        extra={"residuals": {"x": xs.tolist(), "fitted": (2 * xs + 1).tolist(), "residual": (np.sin(xs * 2.1) * 1.5).tolist()}},
    )
    corr = StatResult(
        id="sr_corr", analysis="correlation_matrix", test_name="Correlation matrix (Pearson)",
        extra={"matrix": {"labels": ["Size", "PDI", "Zeta", "Uptake"],
                          "r": [[1, 0.62, -0.31, 0.74], [0.62, 1, -0.12, 0.38], [-0.31, -0.12, 1, -0.55], [0.74, 0.38, -0.55, 1]]}},
    )
    return {r.id: r for r in (anova, two_way, surv, paired, resid, corr, *fits.values())}


def specs() -> dict[str, FigureSpec]:
    brackets = [{"stat_result_id": "sr_anova", "comparisons": "significant"}]
    out = {
        "bar": FigureSpec(plot_type="bar", data={"table": T_SIZE}, annotations=[{"stat_result_id": "sr_anova"}],
                          title="Size by synthesis method"),
        "scatter": FigureSpec(plot_type="scatter", data={"table": T_SIZE}, annotations=brackets),
        "box": FigureSpec(plot_type="box", data={"table": T_SIZE},
                          annotations=[{"stat_result_id": "sr_anova", "kind": "summary_text"}]),
        "violin": FigureSpec(plot_type="violin", data={"table": T_SIZE}),
        "box_violin": FigureSpec(plot_type="box_violin", data={"table": T_SIZE}),
        "beeswarm": FigureSpec(plot_type="beeswarm", data={"table": T_CELLS, "x": "Treatment", "y": "Uptake"}),
        "superplot": FigureSpec(plot_type="superplot",
                                data={"table": T_CELLS, "x": "Treatment", "y": "Uptake", "replicate": "Experiment"}),
        "estimation": FigureSpec(
            plot_type="estimation", data={"table": T_PAIRED},
            layers=[{"type": "data_points_beeswarm"}, {"type": "mean_dash"}, {"type": "ci95_errorbar"},
                    {"type": "effect_axis", "stat_result_id": "sr_ttest"}],
        ),
        "paired": FigureSpec(plot_type="paired", data={"table": T_PAIRED},
                             annotations=[{"stat_result_id": "sr_ttest", "format": "p"}]),
        "xy": FigureSpec(plot_type="xy", data={"table": T_DOSE}, axes={"x": {"scale": "log10"}}),
        "dose_response": FigureSpec(
            plot_type="dose_response", data={"table": T_DOSE},
            layers=[{"type": "confidence_band", "stat_result_id": "sr_fit_free"}, {"type": "sem_errorbar"},
                    {"type": "mean_dot"}, {"type": "curve_fit", "stat_result_id": "sr_fit_free"},
                    {"type": "curve_fit", "stat_result_id": "sr_fit_nano"}],
        ),
        "survival": FigureSpec(plot_type="survival", data={"table": T_SURV, "stat_result_id": "sr_surv"},
                               annotations=[{"stat_result_id": "sr_surv", "kind": "summary_text", "position": "bottom left"}]),
        "grouped_bar": FigureSpec(plot_type="grouped_bar", data={"table": T_UPTAKE},
                                  annotations=[{"stat_result_id": "sr_twoway"}], layout={"width_mm": 120, "height_mm": 80}),
        "stacked_bar": FigureSpec(plot_type="stacked_bar", data={"table": T_CONT}),
        "histogram": FigureSpec(plot_type="histogram", data={"table": T_CELLS, "y": "Uptake", "hue": "Treatment"}),
        "heatmap": FigureSpec(plot_type="heatmap", data={"table": T_UPTAKE}),
        "heatmap_correlation": FigureSpec(plot_type="heatmap", data={"table": T_UPTAKE, "stat_result_id": "sr_corr"}),
        "qq": FigureSpec(plot_type="qq", data={"table": T_SIZE}),
        "residuals": FigureSpec(plot_type="residuals", data={"table": T_DOSE, "stat_result_id": "sr_resid"}),
    }
    for preset in preset_names():
        key = "style_" + "".join(ch if ch.isalnum() else "_" for ch in preset.lower()).strip("_")
        while "__" in key:
            key = key.replace("__", "_")
        out[key] = FigureSpec(plot_type="bar", data={"table": T_SIZE}, annotations=brackets, style={"preset": preset})
    return out


def composition() -> CompositionSpec:
    return CompositionSpec(
        rows=2, cols=2, width_mm=183, height_mm=140,
        panels=[{"figure": "bar", "row": 0, "col": 0}, {"figure": "dose_response", "row": 0, "col": 1},
                {"figure": "grouped_bar", "row": 1, "col": 0}, {"figure": "superplot", "row": 1, "col": 1}],
    )


def build_cases() -> dict[str, Callable[[], Figure]]:
    """name -> zero-argument function returning the rendered Figure."""
    tbl, res, sp = tables(), results(), specs()
    cases: dict[str, Callable[[], Figure]] = {
        name: (lambda s=spec: render_figure(s, tbl, res)) for name, spec in sp.items()
    }
    cases["composition"] = lambda: render_composition(composition(), sp, tbl, res)
    return cases

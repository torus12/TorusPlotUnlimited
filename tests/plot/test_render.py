import matplotlib as mpl
import numpy as np
import pandas as pd
import pytest

from tests.golden import cases
from torusplot.data import ColumnTable, MultipleVariablesTable, Project
from torusplot.plot import PlotError, figure_warnings, render_figure, summarize
from torusplot.plot.brackets import (
    Bracket,
    comparison_label,
    draw_brackets,
    select_comparisons,
    summary_text,
)
from torusplot.plot.render import beeswarm_offsets
from torusplot.spec import FigureSpec, StatAnnotation
from torusplot.stats.result import Comparison, StatResult

TABLES = cases.tables()
RESULTS = cases.results()
SPECS = cases.specs()


def ctx(fig):
    return fig._torusplot_context


@pytest.mark.parametrize("name", sorted(SPECS))
def test_every_case_renders_at_exact_size_without_warnings(name):
    spec = SPECS[name]
    fig = render_figure(spec, TABLES, RESULTS)
    assert figure_warnings(fig) == []
    w, h = fig.get_size_inches()
    from torusplot.plot import resolve_style

    style = resolve_style(spec.style)
    assert w * 25.4 == pytest.approx(spec.layout.width_mm or style["width_mm"])
    assert h * 25.4 == pytest.approx(spec.layout.height_mm or style["height_mm"])


def test_rendering_is_deterministic():
    from torusplot.export import figure_bytes

    spec = SPECS["bar"]
    a = figure_bytes(render_figure(spec, TABLES, RESULTS), "png", dpi=80)
    b = figure_bytes(render_figure(spec, TABLES, RESULTS), "png", dpi=80)
    assert a == b


def test_bars_show_means_and_all_points():
    fig = render_figure(FigureSpec(plot_type="bar", data={"table": cases.T_SIZE}), TABLES)
    ax = ctx(fig).ax
    table = TABLES[cases.T_SIZE]
    heights = [p.get_height() for p in ax.patches]
    means = [g.mean() for g in table.groups().values()]
    assert heights == pytest.approx(means)
    n_points = sum(len(c.get_offsets()) for c in ax.collections if c.get_gid() and "points" in c.get_gid())
    assert n_points == sum(len(g) for g in table.groups().values())
    assert [t.get_text() for t in ax.get_xticklabels()] == table.group_names
    assert ax.get_ylim()[0] == 0  # bars are anchored at zero


def test_summary_matches_stats_engine_formulas():
    v = np.array([4.0, 5.5, 6.1, 7.2, 9.9])
    s = summarize(v)
    from scipy import stats

    assert s["mean"] == pytest.approx(v.mean())
    assert s["sd"] == pytest.approx(v.std(ddof=1))
    assert s["sem"] == pytest.approx(stats.sem(v))
    lo, hi = stats.t.interval(0.95, v.size - 1, loc=v.mean(), scale=stats.sem(v))
    assert (s["ci_low"], s["ci_high"]) == pytest.approx((lo, hi))
    assert summarize(np.array([]))["n"] == 0
    assert np.isnan(summarize(np.array([3.0]))["sd"])


def test_group_selection_order_and_colour_override():
    spec = FigureSpec(plot_type="bar", data={"table": cases.T_SIZE, "groups": ["Microfluidic", "Bulk mixing"]},
                      style={"colors": {"Microfluidic": "#FF0000"}})
    fig = render_figure(spec, TABLES)
    ax = ctx(fig).ax
    assert [t.get_text() for t in ax.get_xticklabels()] == ["Microfluidic", "Bulk mixing"]
    assert mpl.colors.to_hex(ax.patches[0].get_facecolor()[:3]) == "#ff0000"
    with pytest.raises(PlotError, match="Unknown group"):
        render_figure(FigureSpec(plot_type="bar", data={"table": cases.T_SIZE, "groups": ["Nope"]}), TABLES)


def test_brackets_come_only_from_the_stat_result():
    spec = SPECS["bar"]
    fig = render_figure(spec, TABLES, RESULTS)
    c = ctx(fig)
    res = RESULTS["sr_anova"]
    assert len(c.brackets_drawn) == len(res.comparisons)
    by_id = {b.comparison_id: b for b in c.brackets_drawn}
    for comp in res.comparisons:
        assert by_id[comp.id].label == comparison_label(comp, "stars")
    # change the computed p value -> the figure label follows; nothing else can change it
    changed = res.model_copy(deep=True)
    changed.comparisons[0].p_adjusted = 0.5
    fig2 = render_figure(spec, TABLES, {**RESULTS, "sr_anova": changed})
    labels2 = {b.comparison_id: b.label for b in ctx(fig2).brackets_drawn}
    assert labels2[changed.comparisons[0].id] == "ns"
    # brackets never overlap the data: every bracket sits above the tallest point it spans
    top_of_data = max(g.max() for g in TABLES[cases.T_SIZE].groups().values())
    assert c.ax.get_ylim()[1] > top_of_data


def test_missing_result_is_a_warning_not_a_made_up_number():
    spec = FigureSpec(plot_type="bar", data={"table": cases.T_SIZE}, annotations=[{"stat_result_id": "sr_missing"}])
    fig = render_figure(spec, TABLES, RESULTS)
    assert ctx(fig).brackets_drawn == []
    assert any("sr_missing" in w for w in figure_warnings(fig))


def test_bracket_selection_and_formats():
    res = RESULTS["sr_anova"]
    sig = select_comparisons(res, StatAnnotation(stat_result_id=res.id, comparisons="significant"))
    assert all(c.significant for c in sig) and len(sig) == 5
    ids = [res.comparisons[0].id, res.comparisons[2].id]
    assert [c.id for c in select_comparisons(res, StatAnnotation(stat_result_id=res.id, comparisons=ids))] == ids
    pairs = select_comparisons(res, StatAnnotation(stat_result_id=res.id, pairs=[("Sonication", "Bulk mixing")]))
    assert len(pairs) == 1
    no_ns = select_comparisons(res, StatAnnotation(stat_result_id=res.id, show_ns=False))
    assert len(no_ns) == 5
    comp = Comparison(id="x", group_a="a", group_b="b", p_value=0.03, p_adjusted=0.00004)
    assert comparison_label(comp, "stars") == "****"
    assert comparison_label(comp, "stars", "apa") == "***"
    assert comparison_label(comp, "p") == "P < 0.0001"
    assert comparison_label(Comparison(id="x", group_a="a", group_b="b", p_value=0.0312), "p") == "P = 0.0312"
    assert comparison_label(Comparison(id="x", group_a="a", group_b="b", p_value=0.0312), "p_short") == "0.0312"
    assert comparison_label(Comparison(id="x", group_a="a", group_b="b", p_value=0.2), "stars_p") == "ns (P = 0.2000)"
    assert comparison_label(Comparison(id="x", group_a="a", group_b="b"), "stars") == "n/a"
    assert summary_text(res) == "One-way ANOVA: F(3, 28) = 52.3, P < 0.0001"
    assert summary_text(StatResult(analysis="x", test_name="Mann-Whitney test", statistic_name="U", statistic=12, p_value=0.04)) == \
        "Mann-Whitney test: U = 12, P = 0.0400"


def test_bracket_stacking_never_overlaps():
    from matplotlib.figure import Figure

    fig = Figure()
    ax = fig.add_subplot()
    ax.set_ylim(0, 10)
    tops = {0.0: 5.0, 1.0: 8.0, 2.0: 3.0}
    top = draw_brackets(ax, [Bracket(0, 2, "*"), Bracket(0, 1, "**"), Bracket(1, 2, "ns")], tops)
    ys = sorted(line.get_ydata()[1] for line in ax.lines)
    assert ys[0] > 8.0  # above the tallest bar they span
    assert len(set(np.round(ys, 6))) == 3 or ys[0] != ys[1]
    assert ys[-1] > max(ys[:-1])  # the widest bracket is on top
    assert top > ys[-1] and ax.get_ylim()[1] >= top
    assert draw_brackets(ax, [], tops) is None


def test_brackets_on_log_axis():
    t = ColumnTable.from_dict("t", {"A": [1, 2, 3], "B": [100, 200, 300]})
    res = StatResult(id="sr_l", analysis="x", test_name="t", comparisons=[Comparison(id="", group_a="A", group_b="B", p_value=0.01)]).finalize()
    fig = render_figure(FigureSpec(plot_type="scatter", data={"table": "t"}, axes={"y": {"scale": "log10"}},
                                   annotations=[{"stat_result_id": "sr_l"}]), t, {"sr_l": res})
    c = ctx(fig)
    assert c.ax.get_yscale() == "log" and len(c.brackets_drawn) == 1
    assert c.ax.get_ylim()[1] > 300


def test_statannotations_engine_draws_same_labels():
    spec = FigureSpec(plot_type="bar", data={"table": cases.T_SIZE},
                      annotations=[{"stat_result_id": "sr_anova", "comparisons": "significant", "engine": "statannotations"}])
    fig = render_figure(spec, TABLES, RESULTS)
    assert not any("statannotations could not" in w for w in figure_warnings(fig))
    c = ctx(fig)
    texts = sorted(t.get_text() for t in c.ax.texts)
    expected = sorted(comparison_label(comp, "stars") for comp in RESULTS["sr_anova"].comparisons if comp.significant)
    assert texts == expected
    assert [b.label for b in c.brackets_drawn] and len(c.brackets_drawn) == len(expected)


def test_grouped_brackets_use_cells_and_skip_unmatched():
    fig = render_figure(SPECS["grouped_bar"], TABLES, RESULTS)
    c = ctx(fig)
    assert len(c.brackets_drawn) == 4
    for b in c.brackets_drawn:
        assert abs(b.x1 - b.x2) < 1.0  # within one time point
    bad = StatResult(id="sr_bad", analysis="x", test_name="t",
                     comparisons=[Comparison(id="", group_a="X", group_b="Y", p_value=0.01)]).finalize()
    fig = render_figure(FigureSpec(plot_type="grouped_bar", data={"table": cases.T_UPTAKE}, annotations=[{"stat_result_id": "sr_bad"}]),
                        TABLES, {"sr_bad": bad})
    assert any("do not match" in w for w in figure_warnings(fig))


def test_superplot_summaries_use_replicate_means():
    fig = render_figure(SPECS["superplot"], TABLES, RESULTS)
    c = ctx(fig)
    frame = TABLES[cases.T_CELLS].data
    rep_means = frame.groupby(["Treatment", "Experiment"])["Uptake"].mean()
    expected = rep_means.groupby("Treatment").mean()
    dashes = {line.get_gid(): line.get_ydata()[0] for line in c.ax.lines if (line.get_gid() or "").startswith("mean-")}
    assert dashes["mean-Control"] == pytest.approx(expected["Control"])
    assert dashes["mean-Targeted_NP"] == pytest.approx(expected["Targeted NP"])
    assert len([ln for ln in c.ax.lines if "replicate-mean" in (ln.get_gid() or "")]) == 6
    assert [lab for _, lab in c.legend] == ["Exp 1", "Exp 2", "Exp 3"]


def test_dose_response_uses_log_axis_and_attached_curves():
    fig = render_figure(SPECS["dose_response"], TABLES, RESULTS)
    c = ctx(fig)
    assert c.ax.get_xscale() == "log"
    curves = [ln for ln in c.ax.lines if (ln.get_gid() or "").startswith("curve-")]
    assert len(curves) == 2
    np.testing.assert_allclose(curves[0].get_ydata(), RESULTS["sr_fit_free"].extra["curve"]["y"])
    fig = render_figure(FigureSpec(plot_type="dose_response", data={"table": cases.T_DOSE}), TABLES, RESULTS)
    assert any("No fit is attached" in w for w in figure_warnings(fig))
    lin = render_figure(FigureSpec(plot_type="dose_response", data={"table": cases.T_DOSE}, axes={"x": {"scale": "linear"}},
                                   layers=[{"type": "mean_dot"}]), TABLES)
    assert ctx(lin).ax.get_xscale() == "linear"


def test_survival_plot_from_result_and_percent_axis():
    fig = render_figure(SPECS["survival"], TABLES, RESULTS)
    c = ctx(fig)
    steps = [ln for ln in c.ax.lines if (ln.get_gid() or "").startswith("km-curve")]
    assert len(steps) == 3 and steps[0].get_ydata()[0] == 100.0
    assert c.ax.get_ylabel() == "Percent survival"


def test_estimation_plot_reads_difference_from_result():
    fig = render_figure(SPECS["estimation"], TABLES, RESULTS)
    c = ctx(fig)
    assert len(c.axes) == 2
    effect = [ln for ln in c.axes[1].lines if (ln.get_gid() or "").startswith("effect")][0]
    assert effect.get_ydata()[0] == pytest.approx(RESULTS["sr_ttest"].comparisons[0].estimate)


def test_axis_options_title_text_legend():
    spec = FigureSpec(
        plot_type="bar", data={"table": cases.T_SIZE}, title="My title",
        axes={"y": {"label": "Size (nm)", "min": 0, "max": 300, "ticks": [0, 150, 300], "grid": True}, "x": {"label": "Method", "tick_rotation": 45},
              "zero_line": True},
        texts=[{"text": "pilot data", "x": 0.02, "y": 0.98}],
        layers=[{"type": "mean_bar", "fill": False}, {"type": "sd_errorbar", "direction": "up"}, {"type": "data_points_beeswarm"},
                {"type": "reference_line", "value": 100, "label": "target"}, {"type": "n_labels"}, {"type": "value_labels"}],
        legend={"show": True, "position": "outside right", "title": "Key"},
    )
    fig = render_figure(spec, TABLES)
    ax = ctx(fig).ax
    assert ax.get_ylabel() == "Size (nm)" and ax.get_xlabel() == "Method" and ax.get_title() == "My title"
    assert ax.get_ylim() == (0.0, 300.0) and list(ax.get_yticks()) == [0, 150, 300]
    assert any(t.get_text() == "pilot data" for t in ax.texts)
    assert ax.get_legend() is not None and ax.get_legend().get_title().get_text() == "Key"
    assert figure_warnings(fig) == []


def test_multiple_variables_table_roles_and_errors():
    df = pd.DataFrame({"g": list("aabbcc"), "h": list("xyxyxy"), "v": [1.0, 2, 3, 4, 5, 6], "s": [1, 2, 1, 2, 1, 2]})
    t = MultipleVariablesTable("mv", df)
    fig = render_figure(FigureSpec(plot_type="bar", data={"table": "mv", "x": "g", "y": "v", "hue": "h"}), t)
    assert len(ctx(fig).ax.patches) == 6
    fig = render_figure(FigureSpec(plot_type="paired", data={"table": "mv", "x": "h", "y": "v", "subject": "g"}), t)
    assert len([ln for ln in ctx(fig).ax.lines if "pair-line" in (ln.get_gid() or "")]) == 3
    with pytest.raises(PlotError, match="not in table"):
        render_figure(FigureSpec(plot_type="bar", data={"table": "mv", "x": "zzz", "y": "v"}), t)
    with pytest.raises(PlotError, match="numbers"):
        render_figure(FigureSpec(plot_type="bar", data={"table": "mv", "x": "g", "y": "h"}), t)
    with pytest.raises(PlotError, match="not found"):
        render_figure(FigureSpec(plot_type="bar", data={"table": "other"}), {"mv": t})
    with pytest.raises(PlotError):
        render_figure(FigureSpec(plot_type="xy", data={"table": cases.T_SIZE}), TABLES)
    with pytest.raises(PlotError, match="fit result"):
        render_figure(FigureSpec(plot_type="residuals", data={"table": cases.T_DOSE}), TABLES)
    fig = render_figure(FigureSpec(plot_type="superplot", data={"table": "mv", "x": "g", "y": "v"}), t)
    assert any("replicate column" in w for w in figure_warnings(fig))


def test_project_can_be_passed_directly():
    p = Project("demo")
    for t in TABLES.values():
        p.add_table(t)
    for r in RESULTS.values():
        p.add_result(r)
    fig = render_figure(SPECS["bar"], p)
    assert len(ctx(fig).brackets_drawn) == 6


def test_beeswarm_offsets_do_not_overlap_and_are_deterministic():
    y = np.array([1.0, 1.0, 1.0, 1.01, 1.02, 2.0, 2.0, 5.0])
    off = beeswarm_offsets(y, dy=0.1, dx=0.1, max_half_width=10)
    np.testing.assert_array_equal(off, beeswarm_offsets(y, 0.1, 0.1, 10))
    for i in range(len(y)):
        for j in range(i + 1, len(y)):
            dist = np.hypot((off[i] - off[j]) / 0.1, (y[i] - y[j]) / 0.1)
            assert dist >= 1 - 1e-6
    assert off[-1] == 0.0  # an isolated point stays on the centre line
    squeezed = beeswarm_offsets(np.zeros(50), 0.1, 0.1, 0.4)
    assert np.max(np.abs(squeezed)) <= 0.4 + 1e-12
    assert beeswarm_offsets(np.array([3.0]), 0.1, 0.1, 1).tolist() == [0.0]


def test_missing_font_falls_back_with_warning():
    fig = render_figure(FigureSpec(plot_type="bar", data={"table": cases.T_SIZE}, style={"font_family": "No Such Font 123"}), TABLES)
    assert any("not installed" in w for w in figure_warnings(fig))


def test_rendering_ignores_machine_matplotlib_settings():
    """A system matplotlibrc / user style must not change TorusPlot figures."""
    from torusplot.export import figure_bytes

    spec = SPECS["bar"]
    reference = figure_bytes(render_figure(spec, TABLES, RESULTS), "png", dpi=80)
    hostile = {"text.hinting": "no_hinting", "font.size": 23.0, "font.family": ["serif"], "axes.linewidth": 4.0,
               "lines.linewidth": 5.0, "axes.facecolor": "#FFEEEE", "xtick.direction": "in", "savefig.dpi": 37}
    with mpl.rc_context(hostile):
        assert figure_bytes(render_figure(spec, TABLES, RESULTS), "png", dpi=80) == reference
        fig = render_figure(spec, TABLES, RESULTS)
    assert figure_bytes(fig, "png", dpi=80) == reference
    before = dict(mpl.rcParams)
    render_figure(spec, TABLES, RESULTS)
    assert dict(mpl.rcParams) == before


def test_bundled_fonts_win_over_system_copies():
    from matplotlib import font_manager

    from torusplot.plot.fonts import BUNDLED_FAMILIES, FONT_DIR, register_bundled_fonts

    register_bundled_fonts()
    for family in BUNDLED_FAMILIES:
        found = font_manager.findfont(family, fallback_to_default=False)
        assert str(FONT_DIR.resolve()) in str(__import__("pathlib").Path(found).resolve()), (family, found)

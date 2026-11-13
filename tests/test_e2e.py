"""End-to-end test of the core promise:

load a sample Grouped table -> recommend_test -> two-way ANOVA + Tukey -> bar+points figure
with brackets -> export SVG/PDF/TIFF -> save .tplot -> reload -> re-render identically.
"""

import pytest
from PIL import Image

from torusplot.data import Project
from torusplot.data.samples import load_sample
from torusplot.export import (
    export_all,
    figure_bytes,
    methods_paragraph,
    validate_svg,
    write_stats_report,
)
from torusplot.plot import figure_warnings, render_figure
from torusplot.plot.brackets import comparison_label
from torusplot.spec import FigureSpec
from torusplot.stats import run_analysis
from torusplot.stats.recommend import recommend_test


@pytest.mark.slow
def test_grouped_table_to_published_figure_and_back(tmp_path):
    # 1. data
    project = Project("E2E")
    table = project.add_table(load_sample("cell_uptake"))
    assert table.kind == "grouped"

    # 2. recommendation
    rec = recommend_test(table)
    assert rec.analysis == "anova_twoway" and rec.params.get("posthoc") == "tukey"
    assert rec.reason and rec.steps

    # 3. statistics: two-way ANOVA + Tukey
    result = project.add_result(run_analysis(table, rec.analysis, **rec.params))
    assert result.test_name.lower().startswith("two-way anova")
    sources = {row["source"] for row in result.anova_table}
    assert {"Time", "Formulation"} <= sources and any("x" in s or ":" in s or "×" in s for s in sources)
    assert result.correction and "tukey" in result.correction.lower()
    assert len(result.comparisons) == 9  # 3 formulations compared within each of 3 time points
    for comp in result.comparisons:
        assert comp.id.startswith(result.id + ":c")
        assert comp.p_adjusted is not None and 0 <= comp.p_adjusted <= 1
        assert comp.cell_a and comp.cell_b and comp.cell_a["row"] == comp.cell_b["row"]
    assert result.why_chosen and result.interpretation and result.assumptions and result.effect_sizes

    # 4. figure: bars + individual points + brackets taken from the result by ID
    spec = FigureSpec(
        name="Uptake", plot_type="bar", data={"table": table.name},
        annotations=[{"stat_result_id": result.id}], layout={"width_mm": 120, "height_mm": 80},
    )
    project.figures[spec.name] = spec
    fig = render_figure(spec, project)
    assert figure_warnings(fig) == []
    ctx = fig._torusplot_context
    assert len(ctx.ax.patches) == 9  # 3 x 3 bars
    n_points = sum(len(c.get_offsets()) for c in ctx.ax.collections if "points" in (c.get_gid() or ""))
    assert n_points == 36  # every data point is shown
    assert len(ctx.brackets_drawn) == 9
    labels = {b.comparison_id: b.label for b in ctx.brackets_drawn}
    for comp in result.comparisons:  # figure text == stats table, always
        assert labels[comp.id] == comparison_label(comp, "stars")

    # 5. exports
    paths = export_all(fig, tmp_path / "uptake", ["svg", "pdf", "tiff"], dpi=300)
    svg = validate_svg(paths["svg"])
    assert svg["ok"] and svg["text_elements"] > 10 and "bracket-1" in svg["group_ids"]
    assert paths["pdf"].read_bytes().startswith(b"%PDF")
    tif = Image.open(paths["tiff"])
    assert tif.info["compression"] == "tiff_lzw" and tif.size[0] == round(120 / 25.4 * 300)
    report = write_stats_report([result], tmp_path / "uptake", spec=spec)
    text = report["markdown"].read_text(encoding="utf-8")
    assert result.id in text and "Tukey" in text
    assert "two-way ANOVA" in methods_paragraph([result], spec)

    # 6. save, reload, re-render: identical
    before_png = figure_bytes(fig, "png", dpi=100)
    before_svg = figure_bytes(fig, "svg")
    saved = project.save(tmp_path / "e2e.tplot")
    reloaded = Project.load(saved)
    assert reloaded.figures["Uptake"] == spec
    assert reloaded.results[result.id].model_dump() == result.model_dump()
    assert reloaded.tables[table.name].equals(table)
    fig2 = render_figure(reloaded.figures["Uptake"], reloaded)
    assert figure_bytes(fig2, "png", dpi=100) == before_png
    assert figure_bytes(fig2, "svg") == before_svg
    # and saving the reloaded project gives the same bytes
    assert reloaded.save(tmp_path / "again.tplot").read_bytes() == saved.read_bytes()


@pytest.mark.slow
def test_dose_response_fit_to_figure(tmp_path):
    project = Project("DR")
    table = project.add_table(load_sample("dose_response"))
    rec = recommend_test(table)
    assert rec.analysis == "nonlinear_regression"
    fits = []
    for series in table.series_names:
        fits.append(project.add_result(run_analysis(table, "nonlinear_regression", **{**rec.params, "series": series})))
    for fit in fits:
        assert fit.extra["curve"]["x"] and fit.parameters
    names = [p.name.lower() for p in fits[0].parameters]
    assert any("ec50" in n or "ic50" in n for n in names)
    spec = FigureSpec(
        name="DR", plot_type="dose_response", data={"table": table.name},
        layers=[{"type": "sem_errorbar"}, {"type": "mean_dot"}, *[{"type": "curve_fit", "stat_result_id": f.id} for f in fits]],
    )
    fig = render_figure(spec, project)
    assert figure_warnings(fig) == []
    assert len([ln for ln in fig._torusplot_context.ax.lines if (ln.get_gid() or "").startswith("curve-")]) == 2
    project.figures["DR"] = spec
    reloaded = Project.load(project.save(tmp_path / "dr.tplot"))
    assert figure_bytes(render_figure(reloaded.figures["DR"], reloaded), "png", dpi=80) == figure_bytes(fig, "png", dpi=80)


@pytest.mark.slow
def test_survival_and_estimation_plots_compute_through_the_stats_engine():
    surv = load_sample("survival")
    fig = render_figure(FigureSpec(plot_type="survival", data={"table": surv.name, "time": "Time (days)", "event": "Event", "hue": "Group"}), surv)
    assert figure_warnings(fig) == []
    assert len([ln for ln in fig._torusplot_context.ax.lines if (ln.get_gid() or "").startswith("km-curve")]) == 3
    paired = load_sample("paired")
    fig = render_figure(FigureSpec(plot_type="estimation", data={"table": paired.name}), paired)
    assert figure_warnings(fig) == []
    assert len(fig._torusplot_context.axes) == 2

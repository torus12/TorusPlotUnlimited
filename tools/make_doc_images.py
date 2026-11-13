#!/usr/bin/env python
"""Render the documentation gallery images with TorusPlot itself.

    python tools/make_doc_images.py

Uses the synthetic sample datasets and the real statistics engine, so every image in the
docs is something the app can produce. (App screenshots are captured separately by
tools/make_screenshots.js, which drives the running app in a headless browser.)
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from torusplot.data import Project  # noqa: E402
from torusplot.data.samples import load_sample  # noqa: E402
from torusplot.export import figure_bytes  # noqa: E402
from torusplot.plot import preset_names, render_figure  # noqa: E402
from torusplot.plot.compose import render_composition  # noqa: E402
from torusplot.spec import CompositionSpec, FigureSpec  # noqa: E402
from torusplot.stats import run_analysis  # noqa: E402

OUT = ROOT / "docs" / "images"


def build_project() -> Project:
    p = Project("Documentation gallery (synthetic data)")
    size = p.add_table(load_sample("nanoparticle_size"))
    dose = p.add_table(load_sample("dose_response"))
    uptake = p.add_table(load_sample("cell_uptake"))
    cells = p.add_table(load_sample("cell_uptake_cells"))
    surv = p.add_table(load_sample("survival"))
    paired = p.add_table(load_sample("paired"))

    anova = p.add_result(run_analysis(size, "anova_oneway", posthoc="tukey"))
    twoway = p.add_result(run_analysis(uptake, "anova_twoway", posthoc="tukey"))
    fits = [p.add_result(run_analysis(dose, "nonlinear_regression", series=s, model="dose_response_4pl"))
            for s in dose.series_names]
    km = p.add_result(run_analysis(surv, "survival"))
    ttest = p.add_result(run_analysis(paired, "ttest_paired"))

    figs = {
        "bar": FigureSpec(name="bar", plot_type="bar", data={"table": size.name},
                          annotations=[{"stat_result_id": anova.id, "comparisons": "significant"}]),
        "dose_response": FigureSpec(
            name="dose_response", plot_type="dose_response", data={"table": dose.name},
            layers=[{"type": "confidence_band", "stat_result_id": fits[0].id},
                    {"type": "confidence_band", "stat_result_id": fits[1].id},
                    {"type": "sem_errorbar"}, {"type": "mean_dot"},
                    *[{"type": "curve_fit", "stat_result_id": f.id} for f in fits]],
            axes={"y": {"label": "Viability (%)"}}),
        "grouped_bar": FigureSpec(name="grouped_bar", plot_type="grouped_bar", data={"table": uptake.name},
                                  annotations=[{"stat_result_id": twoway.id, "comparisons": "significant"}]),
        "superplot": FigureSpec(name="superplot", plot_type="superplot",
                                data={"table": cells.name, "x": "Treatment", "y": "Uptake", "replicate": "Experiment"}),
        "survival": FigureSpec(name="survival", plot_type="survival",
                               data={"table": surv.name, "stat_result_id": km.id},
                               axes={"x": {"label": "Time (days)"}}),
        "estimation": FigureSpec(name="estimation", plot_type="estimation", data={"table": paired.name},
                                 layers=[{"type": "data_points_beeswarm"}, {"type": "mean_dash"}, {"type": "ci95_errorbar"},
                                         {"type": "effect_axis", "stat_result_id": ttest.id}]),
        "violin": FigureSpec(name="violin", plot_type="box_violin", data={"table": size.name}),
        "paired": FigureSpec(name="paired", plot_type="paired", data={"table": paired.name},
                             annotations=[{"stat_result_id": ttest.id, "format": "p"}]),
    }
    p.figures.update(figs)
    return p


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    project = build_project()
    comp = CompositionSpec(
        name="gallery", rows=2, cols=3, width_mm=240, height_mm=140,
        panels=[{"figure": n, "row": i // 3, "col": i % 3} for i, n in
                enumerate(["bar", "dose_response", "grouped_bar", "superplot", "survival", "estimation"])],
    )
    (OUT / "gallery_composition.png").write_bytes(figure_bytes(render_composition(comp, project), "png", dpi=150))
    print("wrote gallery_composition.png")
    for name in ("bar", "dose_response", "grouped_bar", "superplot", "survival", "estimation", "violin", "paired"):
        (OUT / f"plot_{name}.png").write_bytes(figure_bytes(render_figure(project.figures[name], project), "png", dpi=130))
        print(f"wrote plot_{name}.png")
    presets = CompositionSpec(
        name="presets", rows=2, cols=4, width_mm=260, height_mm=120, panel_labels="none",
        panels=[{"figure": f"preset_{i}", "row": i // 4, "col": i % 4} for i in range(len(preset_names()))],
    )
    for i, preset in enumerate(preset_names()):
        project.figures[f"preset_{i}"] = project.figures["bar"].model_copy(
            update={"name": f"preset_{i}", "title": preset, "style": project.figures["bar"].style.model_copy(update={"preset": preset})})
    (OUT / "style_presets.png").write_bytes(figure_bytes(render_composition(presets, project), "png", dpi=130))
    print("wrote style_presets.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

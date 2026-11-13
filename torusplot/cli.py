"""TorusPlot command line.

Examples::

    python cli.py samples
    python cli.py recommend --sample nanoparticle_size
    python cli.py analyze --sample nanoparticle_size --analysis anova_oneway --param posthoc=tukey
    python cli.py demo --out output/demo
    python cli.py render --project my.tplot --out output --formats svg pdf tiff
    python cli.py schema > figure_spec.schema.json
    python cli.py app          # start the desktop app
    python cli.py mcp          # start the MCP server (stdio)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent


def _load_table(args: argparse.Namespace):
    from .data import import_table
    from .data.samples import load_sample

    if args.sample:
        return load_sample(args.sample)
    if not args.file:
        raise SystemExit("Give --sample NAME or --file PATH (with --kind).")
    if not args.kind:
        raise SystemExit("--kind is required with --file (column, xy, grouped, multiple_variables, contingency).")
    return import_table(args.file, args.kind, replicates=args.replicates) if args.replicates else import_table(args.file, args.kind)


def _parse_value(raw: str) -> Any:
    low = raw.strip().lower()
    if low in {"true", "yes"}:
        return True
    if low in {"false", "no"}:
        return False
    if low in {"none", "null", ""}:
        return None
    try:
        return json.loads(raw)
    except ValueError:
        return [v.strip() for v in raw.split(",")] if "," in raw else raw


def _params(pairs: list[str] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise SystemExit(f"--param expects name=value, got '{pair}'.")
        key, value = pair.split("=", 1)
        out[key.strip()] = _parse_value(value)
    return out


def cmd_samples(_args: argparse.Namespace) -> int:
    from .data.samples import SAMPLES

    for name, info in SAMPLES.items():
        print(f"{name:20s} {info['kind']:20s} {info['description']}")
    return 0


def cmd_analyses(args: argparse.Namespace) -> int:
    from .stats import analyses_for

    for kind in ([args.kind] if args.kind else ["column", "xy", "grouped", "multiple_variables", "contingency"]):
        print(f"\n[{kind}]")
        for info in analyses_for(kind):
            print(f"  {info.key:26s} {info.label}  ({info.category})")
    return 0


def cmd_recommend(args: argparse.Namespace) -> int:
    from .stats.recommend import recommend_test

    table = _load_table(args)
    rec = recommend_test(table, paired=args.paired or None, control=args.control)
    print(rec.explain())
    print(f"\nRun it with: --analysis {rec.analysis} " + " ".join(f"--param {k}={json.dumps(v)}" for k, v in rec.params.items()))
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    from .export import stats_report_markdown, write_stats_report
    from .stats import run_analysis

    table = _load_table(args)
    result = run_analysis(table, args.analysis, allow_approximate=args.allow_approximate, **_params(args.param))
    print(stats_report_markdown([result], f"{result.test_name}: {table.name}"))
    if args.out:
        paths = write_stats_report([result], Path(args.out) / args.analysis)
        print("Wrote: " + ", ".join(str(p) for p in paths.values()))
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    from .data import Project
    from .export import export_all, write_stats_report
    from .plot import render_figure
    from .plot.compose import render_composition

    project = Project.load(args.project)
    out = Path(args.out)
    names = args.figure or [*project.figures, *project.compositions]
    if not names:
        raise SystemExit("The project contains no figures.")
    for name in names:
        if name in project.figures:
            spec = project.figures[name]
            fig = render_figure(spec, project)
            ids = spec.stat_result_ids()
        elif name in project.compositions:
            fig = render_composition(project.compositions[name], project)
            ids = [i for p in project.compositions[name].panels for i in project.figures[p.figure].stat_result_ids()]
            spec = None
        else:
            raise SystemExit(f"'{name}' is not a figure or composition in this project.")
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)
        paths = export_all(fig, out / safe, args.formats, dpi=args.dpi)
        results = [project.results[i] for i in dict.fromkeys(ids) if i in project.results]
        if results:
            paths.update(write_stats_report(results, out / safe, spec=spec))
        print(f"{name}: " + ", ".join(str(p) for p in paths.values()))
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    """End-to-end demonstration on synthetic data: analysis -> figure -> exports -> project file."""
    from .data import Project
    from .data.samples import load_sample
    from .export import export_all, write_stats_report
    from .plot import render_figure
    from .spec import FigureSpec
    from .stats import run_analysis
    from .stats.recommend import recommend_test

    out = Path(args.out)
    project = Project("TorusPlot demo (synthetic data)")
    table = project.add_table(load_sample("cell_uptake"))
    rec = recommend_test(table)
    print(rec.explain(), "\n")
    result = project.add_result(run_analysis(table, rec.analysis, **rec.params))
    spec = FigureSpec(name="Uptake over time", plot_type="grouped_bar", data={"table": table.name},
                      annotations=[{"stat_result_id": result.id, "comparisons": "significant"}],
                      axes={"y": {"label": "Uptake (a.u.)"}}, layout={"width_mm": 120, "height_mm": 80})
    project.figures[spec.name] = spec
    fig = render_figure(spec, project)
    paths = export_all(fig, out / "uptake_over_time", ["svg", "pdf", "tiff", "png"], dpi=300)
    paths.update(write_stats_report([result], out / "uptake_over_time", spec=spec))
    paths["project"] = project.save(out / "demo.tplot")
    for kind, path in paths.items():
        print(f"{kind:9s} {path}")
    return 0


def cmd_schema(_args: argparse.Namespace) -> int:
    from .spec import figure_spec_schema_json

    print(figure_spec_schema_json())
    return 0


def cmd_app(args: argparse.Namespace) -> int:
    app = ROOT / "app" / "streamlit_app.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(app), "--browser.gatherUsageStats", "false"]
    if args.port:
        cmd += ["--server.port", str(args.port)]
    return subprocess.call(cmd)


def cmd_mcp(_args: argparse.Namespace) -> int:
    return subprocess.call([sys.executable, "-m", "torusplot.mcp_server"])


def build_parser() -> argparse.ArgumentParser:
    from . import __version__

    parser = argparse.ArgumentParser(prog="torusplot", description="TorusPlot: scientific graphing and statistics.")
    parser.add_argument("--version", action="version", version=f"TorusPlot {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def data_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--sample", help="Name of a bundled synthetic sample (see: samples)")
        p.add_argument("--file", help="CSV/TSV/XLSX file to load")
        p.add_argument("--kind", choices=["column", "xy", "grouped", "multiple_variables", "contingency"])
        p.add_argument("--replicates", type=int, help="Replicate columns side by side (XY / Grouped)")

    sub.add_parser("samples", help="List the synthetic sample datasets").set_defaults(func=cmd_samples)
    p = sub.add_parser("analyses", help="List analyses available for each table type")
    p.add_argument("--kind", choices=["column", "xy", "grouped", "multiple_variables", "contingency"])
    p.set_defaults(func=cmd_analyses)
    p = sub.add_parser("recommend", help="Recommend a statistical test and explain why")
    data_args(p)
    p.add_argument("--paired", action="store_true", help="Rows/replicates are the same subjects measured repeatedly")
    p.add_argument("--control", help="Control group name (Dunnett-style comparisons)")
    p.set_defaults(func=cmd_recommend)
    p = sub.add_parser("analyze", help="Run an analysis and print the statistics report")
    data_args(p)
    p.add_argument("--analysis", required=True, help="Analysis key (see: analyses)")
    p.add_argument("--param", action="append", help="Option as name=value (repeatable)")
    p.add_argument("--allow-approximate", action="store_true", help="Allow statistics on digitized (approximate) data")
    p.add_argument("--out", help="Folder for the report files")
    p.set_defaults(func=cmd_analyze)
    p = sub.add_parser("render", help="Re-render figures from a .tplot project and export them")
    p.add_argument("--project", required=True)
    p.add_argument("--figure", action="append", help="Figure or composition name (default: all)")
    p.add_argument("--formats", nargs="+", default=["svg", "pdf", "png"], choices=["svg", "pdf", "tiff", "png"])
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--out", default="output")
    p.set_defaults(func=cmd_render)
    p = sub.add_parser("demo", help="Run an end-to-end demo on synthetic data")
    p.add_argument("--out", default="output/demo")
    p.set_defaults(func=cmd_demo)
    sub.add_parser("schema", help="Print the FigureSpec JSON schema").set_defaults(func=cmd_schema)
    p = sub.add_parser("app", help="Start the TorusPlot app in your browser")
    p.add_argument("--port", type=int)
    p.set_defaults(func=cmd_app)
    sub.add_parser("mcp", help="Start the TorusPlot MCP server (stdio)").set_defaults(func=cmd_mcp)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except (ValueError, KeyError, FileNotFoundError) as exc:  # user-facing errors (StatsError, PlotError, ...)
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

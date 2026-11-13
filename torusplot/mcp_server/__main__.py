"""Run the TorusPlot MCP server over stdio: ``python -m torusplot.mcp_server``.

Standard output carries only MCP protocol messages; everything else (the start-up
line, logs, warnings) goes to standard error.
"""

from __future__ import annotations

import argparse
import sys

from .session import DEFAULT_OUTPUT_DIR, OUTPUT_DIR_ENV

LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m torusplot.mcp_server",
        description="TorusPlot MCP server (stdio). Connect it to an MCP client such as Claude Desktop "
        "or Claude Code; see docs/MCP.md.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        metavar="DIR",
        help=f"Folder that receives exported figures, reports and project files "
        f"(default: ${OUTPUT_DIR_ENV}, else ./{DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=LOG_LEVELS,
        help="How much the server logs to standard error (default: WARNING).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    import matplotlib

    matplotlib.use("Agg")  # never open a window; figures are rendered off-screen

    try:
        from .server import build_server
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.split(".")[0] == "mcp":
            print(
                "The TorusPlot MCP server needs one optional package that is not installed by default.\n"
                "Install it with:  python -m pip install -r requirements-mcp.txt\n"
                "(see docs/MCP.md)",
                file=sys.stderr,
            )
            return 1
        raise

    server = build_server(output_dir=args.output_dir, log_level=args.log_level)
    print(
        f"TorusPlot MCP server running on stdio. Output directory: {server.torus_session.output_dir}",
        file=sys.stderr,
        flush=True,
    )
    try:
        server.run("stdio")
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""TorusPlot MCP server.

Lets AI agents (Claude Desktop, Claude Code and other MCP clients) load data, run
statistics and build figures through TorusPlot's own engine.

* :mod:`torusplot.mcp_server.session`: all tool logic as a plain class
  (:class:`TorusSession`); importable and testable without the ``mcp`` package.
* :mod:`torusplot.mcp_server.server`: registers those methods as MCP tools.
* ``python -m torusplot.mcp_server``: runs the server over stdio.

Importing this package starts nothing. ``build_server`` and ``main`` are loaded lazily
so that ``TorusSession`` stays usable where the MCP SDK is not installed.
"""

from __future__ import annotations

from typing import Any

from .session import SessionError, TorusSession

__all__ = ["SessionError", "TorusSession", "build_server", "main"]


def __getattr__(name: str) -> Any:
    if name == "build_server":
        from .server import build_server

        return build_server
    if name == "main":
        from .__main__ import main

        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

"""Protocol-level tests of the MCP server.

Approach: the installed MCP SDK (2.x) ships an in-memory client, ``mcp.Client(server)``,
which is used here in both negotiation modes ("auto" = direct dispatch, "legacy" = the
JSON-RPC initialize handshake current desktop clients speak). One additional smoke test
spawns ``python -m torusplot.mcp_server`` and talks to it over real stdio.
"""

from __future__ import annotations

import base64
import inspect
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import anyio
import pytest
from mcp import Client, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.server.mcpserver import MCPServer
from mcp.types import ImageContent, TextContent

import torusplot
from tests.mcp import PNG_MAGIC
from torusplot.mcp_server import __main__ as cli
from torusplot.mcp_server.server import INSTRUCTIONS, REQUIRED_TOOLS, SERVER_NAME, build_server
from torusplot.mcp_server.session import OUTPUT_DIR_ENV, SessionError, TorusSession

REPO_ROOT = Path(torusplot.__file__).resolve().parents[1]
EXTRA_TOOLS = (
    "list_analyses", "list_state", "get_result", "save_project", "load_project", "list_styles", "get_figure_schema",
)
TIMEOUT = 120


@pytest.fixture
def server(tmp_path: Path) -> MCPServer:
    return build_server(output_dir=tmp_path / "out")


def with_client(server: Any, body: Any, mode: str = "auto") -> Any:
    """Run ``await body(client)`` against a connected client, with a timeout."""

    async def main() -> Any:
        with anyio.fail_after(TIMEOUT):
            async with Client(server, mode=mode) as client:
                return await body(client)

    return anyio.run(main)


def text_of(result: Any) -> str:
    return "\n".join(block.text for block in result.content if isinstance(block, TextContent))


# ====================================================================== registration
def test_all_required_tools_are_registered(server: MCPServer):
    tools = anyio.run(server.list_tools)
    names = [tool.name for tool in tools]
    assert len(REQUIRED_TOOLS) == 8 and set(REQUIRED_TOOLS) <= set(names)
    assert set(names) == set(REQUIRED_TOOLS) | set(EXTRA_TOOLS) and len(names) == len(set(names))
    assert server.name == SERVER_NAME == "torusplot" and server.version == torusplot.__version__
    assert "THE HARD RULE" in server.instructions and server.instructions == INSTRUCTIONS


def test_every_tool_is_described_for_the_calling_model(server: MCPServer):
    for tool in anyio.run(server.list_tools):
        assert tool.title and len(tool.description) > 60, tool.name
        assert tool.annotations is not None and tool.annotations.open_world_hint is False  # local, no network
        schema = tool.input_schema
        assert schema["type"] == "object"
        for name, prop in schema.get("properties", {}).items():
            assert prop.get("description"), f"{tool.name}.{name} has no description"
    tools = {tool.name: tool for tool in anyio.run(server.list_tools)}
    assert tools["load_table"].input_schema["required"] == ["name"]
    assert tools["run_analysis"].input_schema["required"] == ["table", "analysis"]
    assert "nanoparticle_size" in json.dumps(tools["load_table"].input_schema)
    assert "dose_response" in tools["make_figure"].description
    assert tools["list_state"].annotations.read_only_hint is True
    assert tools["run_analysis"].annotations.read_only_hint is False
    # no tool takes a statistic as input
    forbidden = {"p", "p_value", "pvalue", "stars", "significance", "statistic", "label", "text_annotation"}
    for tool in tools.values():
        assert not forbidden & set(tool.input_schema.get("properties", {})), tool.name


def test_tools_mirror_the_session_methods(server: MCPServer):
    """The server is a thin layer: every tool has exactly the parameters of the session method."""
    for tool in anyio.run(server.list_tools):
        method = getattr(TorusSession, tool.name)
        expected = [p for p in inspect.signature(method).parameters if p != "self"]
        assert list(tool.input_schema.get("properties", {})) == expected, tool.name


def test_the_documentation_lists_every_tool_and_parameter(server: MCPServer):
    doc = (REPO_ROOT / "docs" / "MCP.md").read_text(encoding="utf-8")
    for tool in anyio.run(server.list_tools):
        assert f"| `{tool.name}` |" in doc, f"docs/MCP.md has no table row for {tool.name}"
        row = next(line for line in doc.splitlines() if line.startswith(f"| `{tool.name}` |"))
        for name in tool.input_schema.get("properties", {}):
            assert f"`{name}" in row, f"docs/MCP.md does not document {tool.name}.{name}"
    for word in ("claude_desktop_config.json", "claude mcp add torusplot", ".mcp.json", "TORUSPLOT_OUTPUT_DIR",
                 "no network calls", "Scripts\\\\python.exe", ".venv/bin/python"):
        assert word in doc, word


def test_build_server_uses_the_given_session_or_output_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    session = TorusSession(tmp_path / "mine")
    assert build_server(session).torus_session is session
    assert build_server(output_dir=tmp_path / "given").torus_session.output_dir == (tmp_path / "given").resolve()
    monkeypatch.setenv(OUTPUT_DIR_ENV, str(tmp_path / "env"))
    assert build_server().torus_session.output_dir == (tmp_path / "env").resolve()


# ============================================================== in-memory client calls
@pytest.mark.parametrize("mode", ["auto", "legacy"])
def test_client_lists_tools_and_calls_load_table_and_list_state(server: MCPServer, mode: str):
    async def body(client: Client) -> None:
        assert client.server_info.name == "torusplot"
        listed = await client.list_tools()
        assert set(REQUIRED_TOOLS) <= {tool.name for tool in listed.tools}

        loaded = await client.call_tool("load_table", {"name": "np", "sample": "nanoparticle_size"})
        assert loaded.is_error is False
        assert loaded.structured_content["kind"] == "column"
        assert loaded.structured_content["n_per_group"]["Extrusion"] == 8
        assert json.loads(text_of(loaded)) == loaded.structured_content  # text and structured forms agree

        pasted = await client.call_tool(
            "load_table", {"name": "assay", "kind": "column", "text": "Control\tDrug\n1\t2\n2\t4\n3\t5"}
        )
        assert pasted.structured_content["groups"] == ["Control", "Drug"]

        state = await client.call_tool("list_state", {})
        assert state.is_error is False
        assert [t["name"] for t in state.structured_content["tables"]] == ["np", "assay"]
        assert state.structured_content["figures"] == []

    with_client(server, body, mode)


@pytest.mark.parametrize("mode", ["auto", "legacy"])
def test_make_figure_returns_structured_text_and_a_png_image(server: MCPServer, mode: str):
    async def body(client: Client) -> None:
        await client.call_tool("load_table", {"name": "np", "sample": "nanoparticle_size"})
        made = await client.call_tool(
            "make_figure", {"name": "fig", "table": "np", "plot_type": "bar", "title": "Particle size"}
        )
        assert made.is_error is False
        assert [type(block) for block in made.content] == [TextContent, ImageContent]
        image = made.content[1]
        png = base64.b64decode(image.data)
        assert image.mime_type == "image/png" and png.startswith(PNG_MAGIC)
        payload = json.loads(made.content[0].text)
        assert payload == made.structured_content and payload["spec"]["title"] == "Particle size"
        assert Path(payload["preview_path"]).read_bytes() == png

        # arguments sent as JSON text (a habit of some clients) are parsed by the SDK
        restyled = await client.call_tool(
            "restyle_figure", {"name": "fig", "colors": '{"Extrusion": "#FF0000"}', "error_bars": "sd"}
        )
        assert restyled.structured_content["spec"]["style"]["colors"] == {"Extrusion": "#FF0000"}
        assert isinstance(restyled.content[1], ImageContent)

        panel = await client.call_tool("compose_panels", {"name": "panel", "figures": ["fig", "fig"]})
        assert panel.structured_content["grid"] == {"rows": 1, "cols": 2} and isinstance(panel.content[1], ImageContent)

        exported = await client.call_tool("export_figure", {"name": "panel", "formats": ["svg", "png"], "dpi": 100})
        assert [f["format"] for f in exported.structured_content["files"]] == ["svg", "png"]
        assert all(Path(f["path"]).is_file() for f in exported.structured_content["files"])

        for name in ("list_styles", "get_figure_schema"):
            assert (await client.call_tool(name, {})).is_error is False
        report = await client.call_tool("get_stats_report", {"figure": "fig"})
        assert report.structured_content["result_ids"] == []
        saved = await client.call_tool("save_project", {"filename": "session.tplot"})
        loaded = await client.call_tool("load_project", {"path": saved.structured_content["path"]})
        assert [f["name"] for f in loaded.structured_content["figures"]] == ["fig"]

    with_client(server, body, mode)


def test_tool_errors_follow_the_sdk_convention(server: MCPServer):
    """Anticipated failures come back as is_error results carrying the user-facing message."""

    async def body(client: Client) -> None:
        await client.call_tool("load_table", {"name": "np", "sample": "nanoparticle_size"})
        cases = [
            ("make_figure", {"name": "f", "table": "ghost", "plot_type": "bar"}, "Table 'ghost' not found"),
            ("make_figure", {"name": "f", "table": "np", "plot_type": "bar", "title": "p < 0.05"},
             "looks like a statistical result"),
            ("make_figure", {"name": "f", "table": "np", "plot_type": "bar", "annotate_result_id": "sr_fake"},
             "No statistics result with ID 'sr_fake'"),
            ("make_figure", {"name": "f", "table": "np", "plot_type": "bar",
                             "spec": {"texts": [{"text": "***", "x": 0.5, "y": 0.5}]}}, "statistical result"),
            ("load_table", {"name": "t", "kind": "column", "text": "a\tb\n1\t2", "sample": "paired"},
             "exactly one of text, path or sample"),
            ("load_table", {"name": "t", "kind": "pie", "text": "a\tb\n1\t2"}, "kind"),
            ("export_figure", {"name": "ghost"}, "No figure or composition named 'ghost'"),
            ("export_figure", {"name": "f", "formats": ["bmp"]}, "formats"),
            ("save_project", {"filename": "../outside.tplot"}, "outside the output directory"),
            ("restyle_figure", {"name": "ghost"}, "Figure 'ghost' not found"),
            ("compose_panels", {"name": "p", "figures": ["ghost"]}, "not found"),
            ("get_result", {"result_id": "sr_fake"}, "No statistics result"),
            ("run_analysis", {"table": "np"}, "analysis"),
        ]
        for tool, arguments, message in cases:
            result = await client.call_tool(tool, arguments)
            text = text_of(result)
            assert result.is_error is True, (tool, arguments)
            assert message in text and text.startswith(f"Error executing tool {tool}"), text
            assert "Traceback" not in text and 'File "' not in text
        state = await client.call_tool("list_state", {})
        assert state.structured_content["figures"] == [] and state.structured_content["compositions"] == []

    with_client(server, body)


def test_unexpected_exceptions_do_not_leak_details(server: MCPServer, monkeypatch: pytest.MonkeyPatch):
    def boom() -> dict[str, Any]:
        raise RuntimeError("internal detail /home/secret/path")

    monkeypatch.setattr(server.torus_session, "list_styles", boom)

    async def body(client: Client) -> None:
        result = await client.call_tool("list_styles", {})
        assert result.is_error is True
        assert text_of(result) == "Error executing tool list_styles"

    with_client(server, body)


def test_session_errors_become_tool_errors_in_process(server: MCPServer):
    from mcp.server.mcpserver.exceptions import ToolError

    async def call() -> None:
        await server.call_tool("restyle_figure", {"name": "ghost"})

    with pytest.raises(ToolError, match="Figure 'ghost' not found") as err:
        anyio.run(call)
    assert isinstance(err.value.__cause__, ToolError) and isinstance(err.value.__cause__.__cause__, SessionError)


def test_statistics_flow_over_the_protocol(server: MCPServer):
    try:
        from torusplot.stats.registry import load_all

        load_all()
    except Exception:  # noqa: BLE001
        pytest.skip("statistics analysis modules are not ready")

    async def body(client: Client) -> None:
        await client.call_tool("load_table", {"name": "np", "sample": "nanoparticle_size"})
        rec = (await client.call_tool("recommend_test", {"table": "np"})).structured_content
        listed = (await client.call_tool("list_analyses", {"table": "np", "analysis": rec["analysis"]})).structured_content
        assert [a["key"] for a in listed["analyses"]] == [rec["analysis"]]
        ran = await client.call_tool("run_analysis", {"table": "np", "analysis": rec["analysis"], "params": rec["params"]})
        result = ran.structured_content
        assert ran.is_error is False and result["id"].startswith("sr_") and result["comparisons"]
        again = (await client.call_tool("get_result", {"result_id": result["id"]})).structured_content
        assert again["p_value"] == result["p_value"] and again["statistic"] == result["statistic"]
        made = await client.call_tool(
            "make_figure", {"name": "fig", "table": "np", "plot_type": "bar", "annotate_result_id": result["id"]}
        )
        assert made.structured_content["brackets_drawn"] == len(result["comparisons"])
        exported = (await client.call_tool("export_figure", {"name": "fig", "formats": ["svg"]})).structured_content
        assert [Path(f["path"]).name for f in exported["stats_report"]] == ["fig_stats.md", "fig_stats.csv"]
        report = (await client.call_tool("get_stats_report", {"figure": "fig", "format": "csv"})).structured_content
        assert result["id"] in report["report"]
        unknown = await client.call_tool("run_analysis", {"table": "np", "analysis": "magic"})
        assert unknown.is_error and "anova_oneway" in text_of(unknown)

    with_client(server, body, "legacy")


# ==================================================================== stdio subprocess
@pytest.mark.slow
def test_stdio_subprocess_speaks_clean_mcp(tmp_path: Path):
    """Spawn ``python -m torusplot.mcp_server`` and drive it with the SDK's stdio client."""
    out_dir = tmp_path / "output dir with spaces"
    log_path = tmp_path / "server_stderr.log"
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "torusplot.mcp_server", "--output-dir", str(out_dir)],
        cwd=str(REPO_ROOT),
        env={"MPLBACKEND": "Agg", "PYTHONUTF8": "1"},
    )

    async def main(errlog: Any) -> None:
        with anyio.fail_after(TIMEOUT):
            async with Client(stdio_client(params, errlog=errlog), mode="legacy") as client:
                assert client.server_info.name == "torusplot"
                listed = await client.list_tools()
                assert set(REQUIRED_TOOLS) <= {tool.name for tool in listed.tools}
                loaded = await client.call_tool("load_table", {"name": "np", "sample": "nanoparticle_size"})
                assert loaded.is_error is False and loaded.structured_content["shape"] == {"rows": 8, "columns": 4}
                made = await client.call_tool("make_figure", {"name": "fig", "table": "np", "plot_type": "box"})
                assert made.is_error is False and isinstance(made.content[1], ImageContent)
                state = await client.call_tool("list_state", {})
                assert [t["name"] for t in state.structured_content["tables"]] == ["np"]
                assert state.structured_content["output_dir"] == str(out_dir.resolve())
                failed = await client.call_tool("export_figure", {"name": "fig", "filename": "../escape"})
                assert failed.is_error is True and "outside the output directory" in text_of(failed)

    with log_path.open("w", encoding="utf-8") as errlog:
        anyio.run(main, errlog)
    assert (out_dir / "fig_preview.png").read_bytes().startswith(PNG_MAGIC)
    assert not list(tmp_path.glob("escape*"))
    # everything that is not protocol goes to stderr (the client above parsed stdout without a hiccup)
    assert "TorusPlot MCP server running on stdio" in log_path.read_text(encoding="utf-8")


# ============================================================================ CLI
def test_main_runs_the_stdio_server_and_keeps_stdout_clean(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys):
    started: dict[str, Any] = {}

    def fake_run(self: MCPServer, transport: str = "stdio", **kwargs: Any) -> None:
        started["transport"] = transport
        started["output_dir"] = self.torus_session.output_dir
        started["tools"] = len(anyio.run(self.list_tools))

    monkeypatch.setattr(MCPServer, "run", fake_run)
    assert cli.main(["--output-dir", str(tmp_path / "cli out"), "--log-level", "ERROR"]) == 0
    assert started == {"transport": "stdio", "output_dir": (tmp_path / "cli out").resolve(), "tools": 15}
    captured = capsys.readouterr()
    assert captured.out == ""  # stdout is reserved for protocol messages
    assert "TorusPlot MCP server running on stdio" in captured.err and "cli out" in captured.err

    def interrupted(self: MCPServer, transport: str = "stdio", **kwargs: Any) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(MCPServer, "run", interrupted)
    monkeypatch.setenv(OUTPUT_DIR_ENV, str(tmp_path / "from env"))
    assert cli.main([]) == 0  # Ctrl+C is a normal way to stop
    assert "from env" in capsys.readouterr().err


def test_cli_arguments(capsys):
    args = cli.build_parser().parse_args([])
    assert args.output_dir is None and args.log_level == "WARNING"
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--help"])
    assert exit_info.value.code == 0 and "--output-dir" in capsys.readouterr().out
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--log-level", "LOUD"])
    assert exit_info.value.code == 2


def test_importing_the_package_starts_nothing_and_does_not_need_the_sdk():
    code = (
        "import sys, threading\n"
        "import torusplot.mcp_server as pkg\n"
        "from torusplot.mcp_server.session import TorusSession\n"
        "assert 'mcp' not in sys.modules and 'mcp_types' not in sys.modules, 'the SDK was imported eagerly'\n"
        "assert threading.active_count() == 1\n"
        "assert pkg.TorusSession is TorusSession\n"
        "print('clean')\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", code], cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=TIMEOUT, check=False
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "clean"


def test_lazy_package_attributes():
    import torusplot.mcp_server as pkg

    assert pkg.build_server is build_server and pkg.main is cli.main
    assert pkg.SessionError is SessionError
    with pytest.raises(AttributeError, match="no attribute 'nope'"):
        _ = pkg.nope

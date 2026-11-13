# TorusPlot MCP server

Use TorusPlot from an AI assistant: load data, run statistics, build figures and export
them, all through TorusPlot's own engine.

## What is MCP?

The Model Context Protocol (MCP) is an open standard that lets an AI assistant (such as
Claude Desktop or Claude Code) call tools offered by another program. The TorusPlot MCP
server is that other program: it offers tools like "load a table", "run an analysis"
and "make a figure", and the assistant calls them on your behalf.

## What stays on your computer

- The server runs **locally**, as a normal program on your computer. It is started by
  your AI client and talks to it through standard input/output only.
- The server makes **no network calls**. It has no telemetry, no update check and no
  cloud storage.
- It reads only the data files you point it to (`.csv`, `.tsv`, `.txt`, `.xlsx`) and
  `.tplot` project files, and it writes only inside one output folder (see
  [Output folder](#output-folder)).
- **Your AI client is a separate product.** Whatever the assistant sends to its model
  (your prompts, and the tool results it receives from TorusPlot, which include table
  summaries, the first rows of a table, statistics and figure previews) is handled under
  the terms and privacy policy of that AI client's provider. Check those terms before
  working with confidential or regulated data.

## Before you start

You need TorusPlot installed in its virtual environment (the `.venv` folder inside the
TorusPlot folder; start the app once with `run_torusplot.bat` or `run_torusplot.sh` to
create it).

**One extra install step.** The MCP server needs the MCP SDK (`mcp`, MIT license), which is
an optional extra and is *not* installed by the launcher. Open a terminal in the TorusPlot
folder and run:

| System | Command |
|---|---|
| Windows | `.venv\Scripts\python.exe -m pip install --require-hashes -r requirements-optional-lock.txt` |
| macOS / Linux | `.venv/bin/python -m pip install --require-hashes -r requirements-optional-lock.txt` |

(That installs the exact, hash-checked versions of the optional extras. If it does not work
on your Python version, use `-r requirements-mcp.txt` instead.)

Why it is optional: on Windows the MCP SDK brings in the `pywin32` package, whose wheel also
contains an unrelated module (`adodbapi`) under the LGPL-2.1 license. TorusPlot never uses
that module, but it is kept out of the default install so that the standard setup contains
only permissively licensed components.

You will need the full path of the Python program inside that virtual environment:

| System | Path of Python |
|---|---|
| Windows | `C:\path\to\torusplot\.venv\Scripts\python.exe` |
| macOS / Linux | `/path/to/torusplot/.venv/bin/python` |

Replace `C:\path\to\torusplot` or `/path/to/torusplot` with the real location of your
TorusPlot folder everywhere below.

## Start the server by hand (optional check)

Your AI client normally starts the server for you. To check that it works, run:

```
# macOS / Linux
/path/to/torusplot/.venv/bin/python -m torusplot.mcp_server

# Windows (PowerShell)
& "C:\path\to\torusplot\.venv\Scripts\python.exe" -m torusplot.mcp_server
```

You should see one line such as
`TorusPlot MCP server running on stdio. Output directory: ...` and then nothing: the
server is waiting for a client. Press `Ctrl+C` to stop it.

Options:

| Option | Meaning |
|---|---|
| `--output-dir DIR` | Folder that receives exported figures, reports, previews and project files. |
| `--log-level LEVEL` | `DEBUG`, `INFO`, `WARNING` (default), `ERROR` or `CRITICAL`. Logs go to standard error. |
| `--help` | Show the options and exit. |

### Output folder

The output folder is chosen in this order:

1. `--output-dir DIR` on the command line;
2. the environment variable `TORUSPLOT_OUTPUT_DIR`;
3. otherwise a folder named `torusplot_output` inside the folder the server was started
   from.

AI clients start the server from a folder of their own choosing (sometimes one you
cannot write to), so **always set the output folder explicitly**, as in the examples
below. The folder is created the first time something is written.

## Connect to Claude Desktop

1. Open Claude Desktop, then **Settings > Developer > Edit Config**. This opens (or
   creates) the file `claude_desktop_config.json`:
   - Windows: `%APPDATA%\Claude\claude_desktop_config.json`
   - macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
2. Add a `torusplot` entry under `mcpServers`.

   Windows (note the doubled backslashes, which JSON requires):

   ```json
   {
     "mcpServers": {
       "torusplot": {
         "command": "C:\\path\\to\\torusplot\\.venv\\Scripts\\python.exe",
         "args": ["-m", "torusplot.mcp_server"],
         "env": {
           "TORUSPLOT_OUTPUT_DIR": "C:\\Users\\you\\Documents\\TorusPlot figures"
         }
       }
     }
   }
   ```

   macOS / Linux:

   ```json
   {
     "mcpServers": {
       "torusplot": {
         "command": "/path/to/torusplot/.venv/bin/python",
         "args": ["-m", "torusplot.mcp_server"],
         "env": {
           "TORUSPLOT_OUTPUT_DIR": "/Users/you/Documents/TorusPlot figures"
         }
       }
     }
   }
   ```

   If the file already has other servers, add only the `"torusplot": { ... }` part
   inside the existing `mcpServers` block, separated by a comma.
3. Quit Claude Desktop completely and start it again. TorusPlot's tools then appear in
   the tools list of a new conversation.

Notes on `cwd` and `env`:

- `env` sets environment variables for the server. `TORUSPLOT_OUTPUT_DIR` is the one
  that matters. Instead of `env` you can put the folder in `args`:
  `["-m", "torusplot.mcp_server", "--output-dir", "C:\\Users\\you\\Documents\\TorusPlot figures"]`.
- Some MCP clients also accept a `cwd` entry (the folder the server is started in). Do
  not rely on it: not every client honours it, which is why the examples above give the
  output folder as a full path.
- If TorusPlot was copied to the computer without being installed into the virtual
  environment and the client reports `No module named torusplot`, add
  `"PYTHONPATH": "C:\\path\\to\\torusplot"` (or `/path/to/torusplot`) to `env`.

## Connect to Claude Code

Run this once in a terminal (one line):

```
# macOS / Linux
claude mcp add torusplot --env TORUSPLOT_OUTPUT_DIR=/path/to/figures -- /path/to/torusplot/.venv/bin/python -m torusplot.mcp_server

# Windows (PowerShell)
claude mcp add torusplot --env TORUSPLOT_OUTPUT_DIR=C:\path\to\figures -- "C:\path\to\torusplot\.venv\Scripts\python.exe" -m torusplot.mcp_server
```

The shortest form is `claude mcp add torusplot -- <python> -m torusplot.mcp_server`,
where `<python>` is the path from the table above; exports then go to
`torusplot_output` inside the folder Claude Code was started in.

Check with `claude mcp list`, or type `/mcp` inside Claude Code.

To share the setup with everyone who works in a project folder, put the equivalent
`.mcp.json` file in that folder instead:

```json
{
  "mcpServers": {
    "torusplot": {
      "command": "/path/to/torusplot/.venv/bin/python",
      "args": ["-m", "torusplot.mcp_server"],
      "env": {
        "TORUSPLOT_OUTPUT_DIR": "/path/to/figures"
      }
    }
  }
}
```

On Windows use `"command": "C:\\path\\to\\torusplot\\.venv\\Scripts\\python.exe"`.

## Other MCP clients

Any client that can start a local ("stdio") MCP server works the same way: the command
is the virtual environment's Python and the arguments are `-m torusplot.mcp_server`.

## The hard rule about statistics

The assistant chooses **what** to analyse and **how the figure looks**. Every number
comes from TorusPlot's statistics engine.

- No tool accepts a p value, significance stars or free statistical text.
- A figure shows statistics only by pointing at the ID of a computed result (for example
  `sr_3f9a1c2b`). The bracket labels and summary lines are derived from that result when
  the figure is drawn.
- Text that looks like a typed statistic (`p < 0.05`, `***`, `ns`) is rejected in
  titles, axis labels, legend titles, tick labels, text labels, panel labels and in the
  column or group names of a table.
- A result ID that does not exist is an error. Nothing is drawn in its place.
- Tool answers repeat the stored results exactly. Numbers that cannot be written in
  standard JSON are sent as `null` (not-a-number) or as the text `"Infinity"` /
  `"-Infinity"`.

The server tells the assistant to quote results exactly as returned. An assistant can
still make a mistake when it writes prose, so treat the statistics report (see
`get_stats_report` and the `_stats.md` file written by `export_figure`) as the reference.

## Tools

Parameters in **bold** are required. Everything else is optional; the default is shown
after `=`.

### Data

| Tool | Parameters | What it does |
|---|---|---|
| `load_table` | **`name`**, `kind` (`column`, `xy`, `grouped`, `multiple_variables`, `contingency`; required with `text` or `path`), `text`, `path`, `sample`, `replicates`, `row_label`, `column_label`, `sheet` | Creates a table from exactly one of: pasted delimited text (`text`), a `.csv` / `.tsv` / `.txt` / `.xlsx` file (`path`), or a bundled synthetic sample (`sample`). Returns the table summary (shape, group / series / column names, row levels, n per group, first rows), the analyses valid for it and the valid plot types. A table with the same name is replaced. |
| `list_state` | none | Lists tables, statistics results (with IDs), figures, compositions, the output folder and the bundled samples. |

Bundled synthetic samples: `nanoparticle_size` (column), `dose_response` (xy),
`cell_uptake` (grouped), `cell_uptake_cells` (multiple_variables), `survival`
(multiple_variables), `paired` (column), `contingency` (contingency).

### Statistics

| Tool | Parameters | What it does |
|---|---|---|
| `list_analyses` | **`table`**, `analysis` | Lists every analysis valid for the table, with each option's name, kind, default, choices and help text. With `analysis`, only that one. |
| `recommend_test` | **`table`**, `paired`, `control`, `groups`, `goal` | Inspects the design and the data and recommends an analysis: key, ready-to-use `params`, reason, decision steps, assumption checks, alternatives, cautions. It does not run the test. |
| `run_analysis` | **`table`**, **`analysis`**, `params`, `allow_approximate = false` | Runs a registered analysis and stores the result. Returns its `id`, test name, why it was chosen, statistic (with name and degrees of freedom), `p_value`, `p_adjusted`, `significant`, effect sizes, parameters, ANOVA table, comparisons (each with an ID), assumption checks, warnings, interpretation and methods text. An unknown analysis or option is answered with the list of valid ones. |
| `get_result` | **`result_id`** | Returns a stored result again (same content as `run_analysis`). |

### Figures

| Tool | Parameters | What it does |
|---|---|---|
| `make_figure` | **`name`**, `table`, `plot_type` (both required unless given inside `spec`), `spec`, `annotate_result_id`, `comparisons = "all"` (`all`, `significant`, or a list of comparison IDs), `annotation_format = "stars"` (`stars`, `p`, `p_short`, `stars_p`), `annotation_kind = "auto"` (`auto`, `brackets`, `summary_text`), `fit_result_ids`, `data_result_id`, `title`, `x_label`, `y_label`, `style_preset`, `palette`, `width_mm`, `height_mm`, `x`, `y`, `hue`, `replicate`, `subject`, `groups` | Builds a figure, stores it, renders it and returns its spec, rendering warnings, the brackets that were drawn and a PNG preview (as an image, and saved as `<name>_preview.png` in the output folder). `annotate_result_id` adds significance brackets or a one-line test summary from a result; `fit_result_ids` draws fitted curves on `xy` / `dose_response` plots; `data_result_id` draws `survival`, `residuals`, `qq` or correlation `heatmap` plots from a result. `x`, `y`, `hue`, `replicate` and `subject` choose the columns of a `multiple_variables` table. `spec` is a full or partial figure spec for anything the simple parameters do not cover; the simple parameters win. |
| `restyle_figure` | **`name`**, `style_preset`, `palette`, `colors`, `font_family`, `font_size`, `width_mm`, `height_mm`, `title`, `x_label`, `y_label`, `y_min`, `y_max`, `y_scale`, `x_scale`, `error_bars` (`sem`, `sd`, `ci95`, `range`, `iqr`, `none`), `points` (`jitter`, `beeswarm`, `plain`, `none`), `legend_position`, `style_overrides` | Changes only the styling and layout of an existing figure (never its data or statistics), renders it again and returns the same content as `make_figure`. |
| `compose_panels` | **`name`**, `figures`, `rows`, `cols`, `width_mm = 183`, `height_mm`, `panel_labels = "A"` (`A`, `a`, `(A)`, `(a)`, `1`, `none`), `share_x = "none"`, `share_y = "none"` (`none`, `row`, `col`, `all`), `shared_legend = false`, `style_preset`, `layout` | Arranges existing figures as lettered panels at an exact size in millimetres. `figures` are placed row by row on an automatic grid; `layout` is an explicit list of `{figure, row, col, rowspan, colspan}` (rows and columns start at 0). |
| `list_styles` | none | Lists style presets, palettes and plot types with descriptions, plus the valid values of the styling options. |
| `get_figure_schema` | none | Returns the JSON Schema of the figure spec and a small example, for assistants that send a `spec`. |

Plot types: `bar`, `scatter`, `box`, `violin`, `box_violin`, `beeswarm`, `superplot`,
`estimation`, `paired`, `xy`, `dose_response`, `survival`, `grouped_bar`, `stacked_bar`,
`histogram`, `heatmap`, `qq`, `residuals`. Each table type supports a subset;
`load_table` returns the list for the table it created.

### Export and projects

| Tool | Parameters | What it does |
|---|---|---|
| `export_figure` | **`name`** (figure or composition), `formats = ["svg", "pdf", "png"]` (also `tiff`), `dpi = 300`, `transparent = false`, `with_stats_report = true`, `filename` | Writes the files into the output folder. SVG keeps text as editable text; PDF embeds the fonts; TIFF uses LZW compression. With `with_stats_report` it also writes `<name>_stats.md` and `<name>_stats.csv` for the results the figure(s) use. Returns the paths and sizes. |
| `get_stats_report` | `figure`, `result_ids`, `format = "markdown"` (`markdown`, `csv`) | Returns the statistics report and a methods paragraph for a figure's results, for the given result IDs, or for every result when neither is given. |
| `save_project` | `filename = "project.tplot"` | Saves tables, results, figures and compositions as a `.tplot` project file in the output folder. |
| `load_project` | **`path`** | Loads a `.tplot` file and replaces the current session with it. |

### File safety

- `load_table(path=...)` opens only `.csv`, `.tsv`, `.txt` and `.xlsx` files (at most
  50 MB), and `load_project` only `.tplot` files.
- Everything is written inside the output folder. A file name that points elsewhere
  (`../x`, an absolute path outside the folder, a drive letter) is refused, and
  characters that are unsafe in file names are replaced by `_`. For example the figure
  `Figure 1` is exported as `Figure_1.svg`.
- Exporting again under the same name overwrites the earlier files.

### One session per server

The server keeps one project in memory for as long as it runs. Restarting the client
(or the server) starts empty: use `save_project` before and `load_project` after to
carry work over.

## Worked example

You type the request in plain language; the assistant makes the tool calls. The values
below are from the bundled synthetic sample, abbreviated.

> **You:** Load the nanoparticle sample, pick the right test, and make a bar chart with
> significance brackets. Export it for the paper.

**1. Load the data**

```
load_table(name="np", sample="nanoparticle_size")
```

```json
{
  "name": "np", "kind": "column", "shape": {"rows": 8, "columns": 4},
  "groups": ["Bulk mixing", "Sonication", "Extrusion", "Microfluidic"],
  "n_per_group": {"Bulk mixing": 8, "Sonication": 8, "Extrusion": 8, "Microfluidic": 8},
  "analyses": [{"key": "anova_oneway", "label": "One-way ANOVA", "category": "Compare three or more groups"}, "..."],
  "plot_types": ["bar", "scatter", "box", "violin", "..."]
}
```

**2. Ask which test fits**

```
recommend_test(table="np")
```

```json
{
  "analysis": "anova_oneway",
  "label": "One-way ANOVA with Tukey's test for all pairs",
  "params": {},
  "reason": "There are three or more independent groups, the residuals look normal and the spreads are similar, ...",
  "steps": ["4 independent groups defined by one factor.", "..."],
  "alternatives": [{"analysis": "welch_anova", "...": "..."}, {"analysis": "kruskal_wallis", "...": "..."}]
}
```

**3. Run it**

```
run_analysis(table="np", analysis="anova_oneway", params={})
```

```json
{
  "id": "sr_3f9a1c2b",
  "test_name": "One-way ANOVA",
  "statistic_name": "F", "statistic": 80.142, "df": 3.0, "df2": 28.0,
  "p_value": 7.4e-14, "significant": true,
  "comparisons": [
    {"id": "sr_3f9a1c2b:c1", "group_a": "Bulk mixing", "group_b": "Sonication", "p_adjusted": 5.3e-05, "significant": true},
    "... 5 more"
  ],
  "interpretation": "A one-way ANOVA compared the means of 4 groups ...",
  "methods_text": "Group means were compared with a one-way ANOVA (alpha = 0.05), followed by Tukey's HSD test ..."
}
```

The result ID is different every time; the assistant uses the one it received.

**4. Make the figure with brackets**

```
make_figure(name="Figure 1", table="np", plot_type="bar",
            annotate_result_id="sr_3f9a1c2b", y_label="Hydrodynamic diameter (nm)")
```

The answer contains the figure spec, `"brackets_drawn": 6` (one per comparison of the
result), any rendering warnings, and a PNG preview that the assistant can look at. To
show fewer brackets it can pass `comparisons="significant"` or a list of comparison IDs.
It cannot type a label: the stars come from the result.

**5. Adjust the look (optional)**

```
restyle_figure(name="Figure 1", style_preset="Journal Single-Column (89 mm)", error_bars="sd")
```

**6. Export**

```
export_figure(name="Figure 1", formats=["svg", "pdf", "tiff"], dpi=600)
```

```json
{
  "files": [
    {"format": "svg", "path": ".../Figure_1.svg", "size_bytes": 19260},
    {"format": "pdf", "path": ".../Figure_1.pdf", "size_bytes": 13743},
    {"format": "tiff", "path": ".../Figure_1.tif", "size_bytes": 222100}
  ],
  "stats_report": [
    {"format": "markdown", "path": ".../Figure_1_stats.md"},
    {"format": "csv", "path": ".../Figure_1_stats.csv"}
  ]
}
```

The paths start with your output folder, and file sizes differ a little between
computers. `Figure_1_stats.md` holds the full results and a methods paragraph to paste
into the paper.

## Troubleshooting

**The TorusPlot tools do not show up.**
- Quit the client completely and start it again after editing its configuration.
- Check that the configuration file is valid JSON (commas between entries, every brace
  closed). On Windows every backslash in a path must be doubled: `C:\\path\\to`.
- Check that `command` is the Python inside TorusPlot's `.venv`, not another Python.
- Run the "start by hand" command above. If it prints an error, that error is the cause.
- Look at the client's own record of the server: Claude Desktop lists its MCP servers
  and their logs under **Settings > Developer**; Claude Code shows the status under
  `/mcp`.

**`No module named torusplot` or `No module named mcp`.**
The command is not the virtual environment's Python, or TorusPlot is not installed into
that environment. Use the path from the table at the top; if needed add `PYTHONPATH` as
described under "Notes on `cwd` and `env`".

**Paths with spaces.**
In JSON configuration files a path with spaces needs no extra quoting: each entry of
`args` is already one argument. On a command line (the `claude mcp add` command, or
starting the server by hand) put double quotes around such a path.

**"The output directory ... cannot be created" or files appear in an unexpected place.**
The client started the server in a folder of its own. Set `TORUSPLOT_OUTPUT_DIR` (or
`--output-dir`) to a full path of a folder you can write to, then restart the client.
`list_state` reports the folder in use as `output_dir`.

**"points outside the output directory".**
An export or project file name tried to leave the output folder. Give a plain file name
(sub-folders inside the output folder are fine).

**"looks like a statistical result".**
A title, label or column name contained something like `p < 0.05` or `***`. Remove it and
attach the computed result with `annotate_result_id` instead.

**"Only .csv, .tsv, .txt, .xlsx files can be loaded".**
Save the data in one of these formats, or paste the cells as text.

**The assistant's reply got cut off or the client reports a very large tool answer.**
`list_analyses` for a whole table and `get_figure_schema` are long. The assistant can ask
for one analysis at a time (`list_analyses(table, analysis=...)`).

## For developers

All tool logic lives in `torusplot/mcp_server/session.py` as the plain class
`TorusSession` (one method per tool, JSON-safe inputs and outputs, no dependency on the
MCP SDK). `torusplot/mcp_server/server.py` registers those methods as tools on the SDK's
`MCPServer` class.

```python
from torusplot.mcp_server import TorusSession

session = TorusSession(output_dir="figures")
session.load_table(name="np", sample="nanoparticle_size")
result = session.run_analysis("np", "anova_oneway")
session.make_figure("Figure 1", "np", "bar", annotate_result_id=result["id"])
session.export_figure("Figure 1", formats=["svg", "png"])
```

Anticipated problems are raised as `SessionError` with a message meant for the user; the
server reports them to the client as tool errors (`isError: true`) without a traceback.

Tests: `python -m pytest tests/mcp`.

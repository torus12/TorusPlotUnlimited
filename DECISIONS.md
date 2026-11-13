# DECISIONS

Every decision made autonomously during the build is logged here:
**decision**, **alternatives considered**, **reason**. Newest entries are at the bottom.

---

## D-001 Git remote: `torus12/TorusPlotUnlimited` (provided by the owner mid-build)
- **Decision:** The session started with no repository attached, so work began as a local
  git repository in `/home/user/torusplot`. The owner then supplied
  https://github.com/torus12/TorusPlotUnlimited (empty), which was attached as `origin`;
  every milestone is committed and pushed to its `main` branch.
- **Alternatives:** (a) push into one of the owner's unrelated existing repositories
  (rejected: would pollute another project); (b) stop and wait (rejected: autonomy rule 1).
- **Reason:** Keeps `main` always runnable and lets GitHub Actions run the Windows + Linux
  CI matrix on every push.

## D-002 Python 3.11 is the development baseline
- **Decision:** Develop and test in a Python 3.11 virtual environment; CI matrix covers
  3.11 and 3.12 on Windows and Linux; the final audit re-runs the suite in a clean venv.
- **Alternatives:** develop on the newest interpreter (3.13).
- **Reason:** The brief requires "Python 3.11+"; testing on the oldest supported version
  catches accidental use of newer syntax.

## D-003 Extra dev-only tool: pytest-cov
- **Decision:** Add `pytest-cov` (MIT; uses `coverage`, Apache-2.0) to `requirements-dev.txt`.
- **Alternatives:** no coverage measurement.
- **Reason:** The brief sets a >=85% coverage target for `torusplot/stats` and
  `torusplot/spec`, which needs a coverage tool. It is a developer tool only: never
  imported by the app and not in `requirements.txt`.

## D-004 How the license gate works
- **Decision:** `tools/check_licenses.py` computes the *runtime closure* (everything
  reachable from `requirements.txt` for the current platform) with `importlib.metadata`,
  reads license metadata via `pip-licenses`, and evaluates SPDX-style expressions:
  `A OR B` passes if one side is allowed, `A AND B` requires every part to be allowed, and
  any GPL/LGPL/AGPL/SSPL/BUSL/Commons-Clause/non-commercial marker fails. Blank or unknown
  metadata fails until verified in `tools/license_overrides.yaml` with a source URL.
- **Alternatives:** check every installed package including dev tools (rejected: dev tools
  are not shipped; the policy is about runtime dependencies).
- **Reason:** Matches the LICENSE POLICY exactly and keeps the allowlist literal.

## D-005 Three manually verified license overrides (numpy, cffi, pillow)
- **Decision:** Accept `numpy` (reports `BSD-3-Clause AND 0BSD AND MIT AND Zlib AND
  CC0-1.0`), `cffi` (`MIT-0`) and `pillow` (`MIT-CMU`) through recorded overrides rather
  than adding new identifiers to the allowlist.
- **Alternatives:** widen the allowlist to include 0BSD, Zlib, MIT-0, MIT-CMU.
- **Reason:** These identifiers are all permissive (no copyleft, no field-of-use limits)
  but are not literally named in the policy, so each is accepted by explicit review with
  the upstream license URL (read on 2026-11-13). An override stops applying if the
  package's reported license ever changes.

## D-006 statannotations installs cleanly and stays a runtime dependency
- **Decision:** Keep `statannotations` 0.7.x (MIT) as a runtime dependency. It resolved
  without downgrading seaborn (0.13.2) or pandas.
- **Alternatives:** drop it and use only TorusPlot's own bracket module.
- **Reason:** The brief lists it as a runtime dependency. See D-012 for how it is wired
  next to TorusPlot's own bracket engine.

## D-007 GCC runtime libraries inside the NumPy/SciPy binary wheels
- **Decision:** Keep NumPy and SciPy (both BSD-3-Clause, both explicitly allowed by the
  brief) and document one nuance: their PyPI *binary wheels* contain two GCC runtime
  shared libraries: `libgfortran` (GPL-3.0-or-later WITH GCC-exception-3.1) and
  `libquadmath` (LGPL-2.1-or-later).
- **Alternatives:** none practical (every scientific Python stack ships these).
- **Reason / what it means for selling TorusPlot:**
  1. TorusPlot's repository and source distribution contain none of these files. The
     launcher downloads the wheels from PyPI onto the user's machine, so TorusPlot is not
     the distributor of them.
  2. `libgfortran`'s GCC Runtime Library Exception expressly permits combining it with
     proprietary programs.
  3. `libquadmath` is LGPL: it is loaded dynamically as a separate, replaceable file, which
     is the usage LGPL permits for closed-source programs. If TorusPlot is ever shipped as
     a *frozen installer* that bundles the wheels (PyInstaller etc.), keep that library as
     a separate file and include its license notice (the wheel already carries it).
  This is an engineering note, not legal advice; have counsel confirm before a paid
  commercial release of a bundled installer.

## D-008 Reference repositories: verified licenses and what changed versus the brief
- **Decision:** All reference repos were shallow-cloned into the gitignored `_reference/`
  folder, read for ideas only, and summarised in `docs/REFERENCE_NOTES.md` (commit hashes
  and the license file read are recorded there). Nothing from `_reference/` is imported,
  installed, executed or copied.
- **Findings that differ from the brief:**
  - Helix (`mickaphd/Helix`) was found; it is GPL-3.0-or-later -> ideas only.
  - `OpenBMB/MatPlotAgent` was not reachable (auth prompt); the public repo for the same
    paper, `thunlp/MatPlotAgent`, has **no license file** -> ideas only.
  - `SvenPfiffner/PlotLLM` is MIT; still used for ideas only (no code needed from it).
  - Graphite is dual-licensed MIT OR Apache-2.0 (export target only, not a dependency).
  - `paper-plot-skills` has no license file -> ideas only.
- **Alternatives:** none (policy is fixed by the brief).
- **Reason:** Clean-room policy; see the notes file for per-repo details.

## D-009 Default local (Ollama) model recommendations
- **Decision:** Recommend `qwen2.5:7b` (text) and `granite3.2-vision` (vision) for the
  optional Ollama backend. Both are Apache-2.0 according to the vendors' own repositories
  (https://github.com/QwenLM/Qwen2.5 : all sizes except 3B and 72B are Apache-2.0;
  https://github.com/ibm-granite/granite-vision-models : Apache-2.0). No weights are
  bundled; the user pulls a model themselves.
- **Caveat:** `ollama.com` and `huggingface.co` were blocked from the build sandbox, so the
  license was read from the vendor GitHub repositories rather than the Ollama library
  pages. `docs/MODEL_LICENSE_RESEARCH.md` lists what was verified vs. not, and how to
  confirm locally (`ollama show <tag> --license`).
- **Alternatives:** `llava` (rejected: checkpoints inherit Llama community license terms),
  `deepseek-r1` distills based on Llama (rejected for the same reason), `qwen2.5:3b` /
  `:72b` (rejected: non-Apache license).
- **Reason:** The brief allows only Apache-2.0 / MIT models as defaults.

## D-010 FigureSpec is a layered, declarative schema (pydantic) with strict validation
- **Decision:** A figure = `plot_type` + data binding + an ordered list of layers using a
  small verb vocabulary (`mean_bar`, `sem_errorbar`, `data_points_beeswarm`, `boxplot`,
  `curve_fit` ...), plus axes, style (preset + overrides), layout in millimetres, legend,
  export options. Empty `layers` means "the plot type's defaults". Unknown fields are
  rejected (`extra="forbid"`). Multi-panel figures are a separate `CompositionSpec` that
  refers to figures by name.
- **Alternatives:** free-form matplotlib code per figure (rejected: not reproducible, not
  safe for AI); one flat options dict (rejected: cannot express layer order/add/remove).
- **Reason:** The add/adjust/remove verb idea (studied in the MIT-licensed tidyplots R
  package; no code used) makes specs short, diffable and easy for a model to emit, and
  strict validation gives precise error messages to feed back to a model.

## D-011 How "AI never touches statistics" is enforced in code
- **Decision:** (1) `StatAnnotation` has **no field** that can hold a number, star string or
  label; it only holds a `stat_result_id` (+ optional comparison IDs / group pairs). The
  renderer derives every bracket label from `Comparison.p_adjusted`/`p_value` of the
  referenced `StatResult`. (2) Titles, free-text annotations and layer labels are rejected
  at validation if they look like a typed statistic ("p < 0.05", "***", "ns"). (3) A missing
  result ID draws nothing and raises a visible warning; it never falls back to text.
- **Alternatives:** trust the prompt to keep the model honest (rejected: not enforceable).
- **Reason:** Makes the hard rule a property of the schema rather than of model behaviour;
  covered by tests in `tests/spec/test_figure_spec.py` and `tests/plot/test_render.py`.

## D-012 Two bracket engines: TorusPlot's own (default) and statannotations (optional)
- **Decision:** Significance brackets are drawn by TorusPlot's own module by default;
  `StatAnnotation.engine = "statannotations"` routes simple one-factor categorical plots
  through the statannotations library in its custom-annotation mode. Both receive label
  strings computed from the StatResult, so they always show the same text (tested).
- **Alternatives:** statannotations everywhere (rejected: it only supports seaborn-style
  categorical plots, so grouped/factorial cell comparisons, log axes handled our way, and
  multi-panel id naming would be inconsistent between plot types).
- **Reason:** One consistent look across every plot type, while still honouring the brief's
  request to use statannotations with precomputed results where it applies.

## D-013 Bundled fonts: Inter 4.1, Source Sans 3 (3.052), Liberation Sans 2.1.5 (all OFL-1.1)
- **Decision:** Bundle four styles of each family, unmodified, with their OFL texts
  (`torusplot/plot/fonts/`). Liberation Sans is the Arial-metric substitute. Inter and
  Source Sans 3 came from the projects' GitHub releases; Liberation Sans 2.1.5 files were
  taken from the Debian `fonts-liberation` package (the upstream release download was
  blocked by the sandbox proxy); the font files' own name tables state OFL-1.1.
- **Alternatives:** Roboto, Lato (not needed: three families cover UI, journal, humanist).
- **Reason:** OFL-1.1 permits bundling with commercial closed-source software. No
  Arial/Helvetica files are shipped; system fonts can be selected by name at runtime.

## D-014 Golden images: generated only by an explicit, logged script
- **Decision:** `tools/regenerate_goldens.py --reason "..."` is the only way golden images
  are (re)written; each run appends to `tests/golden/REGENERATION_LOG.md`. Tests compare at
  80 dpi with an RMS tolerance of 12 (override with `TORUSPLOT_GOLDEN_TOL`). Point jitter
  uses a deterministic golden-ratio sequence (no random generator), and `matplotlib` is
  pinned to the 3.11 series, so images are stable across platforms.
- **First generation:** 2026-11-13, "Initial golden images (M3/M4)".
- **Alternatives:** exact pixel equality (rejected: brittle across OS font rasterisation).
- **Reason:** Catches unintended visual changes without flaky failures.

## D-015 SVG validity is checked with an XML parser and xmllint; Inkscape was not available
- **Decision:** Exported SVGs are verified in tests to be well-formed XML with real
  `<text>` elements, unique ids, named groups and no dangling references
  (`torusplot.export.validate_svg`, plus `xmllint --noout` when installed).
- **Alternatives:** open each file in Inkscape/Graphite in CI (not possible: neither is
  installed in the build sandbox, and the proxy blocks installing them).
- **Reason:** Those checks are the properties vector editors need. Manual spot-check in
  Inkscape/Graphite is listed as a follow-up in FINAL_REPORT.md.

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
- **Reason:** Ideas-only study policy; see the notes file for per-repo details.

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

## D-016 Dunnett's test uses TorusPlot's own deterministic integration (scipy is the cross-check)
- **Decision:** `posthoc.dunnett` evaluates Dunnett's (1955) double integral with fixed
  Gauss-Legendre quadrature; with a single treatment group it uses the exact t distribution.
  `scipy.stats.dunnett` is used only as a cross-check in tests.
- **Alternatives:** call `scipy.stats.dunnett` directly (the brief allows it).
- **Reason:** scipy integrates by randomised quasi-Monte-Carlo, so its P values change in
  the 3rd-4th significant digit from run to run (e.g. 0.00595 / 0.00580 / 0.00579 on the same
  data). A statistics program must give the same answer every time. Our value agrees with an
  independent adaptive quadrature to about 2e-9 and with scipy within its Monte-Carlo noise.

## D-017 NIST reference files: download was blocked, so verified copies were used
- **Decision:** The NIST StRD data files under `tests/fixtures/nist/` were not downloaded
  from itl.nist.gov (the sandbox proxy refused the host). The ANOVA files are byte-for-byte
  copies of NIST's files as redistributed inside the SciPy wheel's test data; see
  `tests/fixtures/sources/` for per-file provenance, including how the nonlinear-regression
  files and certified values were obtained.
- **Alternatives:** hand-typing every dataset (error-prone for the 18,000-value sets).
- **Reason:** NIST StRD data are works of the U.S. Government (not subject to copyright in
  the United States) and are redistributed with attribution. Certified values are asserted
  to documented tolerances; on the three "higher difficulty" ANOVA sets double precision
  itself limits agreement to about 4 digits (same as scipy), and a second test proves the
  algorithm is exact to 1e-13 once the constant offset is removed.

## D-018 Mauchly's sphericity test reports the second-order chi-square approximation
- **Decision:** Mauchly's P value uses the two-term expansion (Box 1949; Anderson 2003,
  sec. 10.7); the first-order value is kept in `extra["sphericity"]["p_first_order"]`.
- **Alternatives:** first-order chi-square approximation only.
- **Reason:** It reproduces the published textbook value (P = .047 for the repeated-measures
  worked example; the first-order formula gives .044) and is better calibrated in small
  samples (5.6% vs 7.1% rejections at n = 5, k = 4 under sphericity in simulation).

## D-019 Huynh-Feldt epsilon: original 1976 formula by default, Lecoutre's correction available
- **Decision:** `eps_hf` is the original Huynh & Feldt (1976) estimator (what other major
  packages print); the Lecoutre (1991) corrected value is reported in `extra` and selectable
  with `hf_variant="lecoutre"`. The default headline P for 3+ levels is Greenhouse-Geisser.
- **Alternatives:** Lecoutre by default.
- **Reason:** Users compare output with other software; always-correct-with-GG is the safe
  default because Mauchly's test has little power in small samples.

## D-020 Own implementations where a library option was unsuitable (each cross-checked)
- **Kaplan-Meier:** implemented in TorusPlot (lifelines lacks the linear CI option); lifelines
  is the cross-check for estimates, log-log CIs, medians, log-rank and weighted tests.
- **Fisher's exact test for r x c tables:** exact enumeration (up to 1,000,000 tables), because
  scipy 1.17's r x c method is Monte Carlo and gives a different P on every call; beyond that
  size a fixed-seed Monte Carlo is used with a warning.
- **D'Agostino-Pearson K2:** computed in TorusPlot because `scipy.stats.skewtest` returns a
  wrong z for exactly symmetric samples; agrees with scipy to 1e-10 on generic data.
- **Reason:** determinism and correctness; all are tested against the libraries they replace.

## D-021 ROUT outlier method was implemented from the published description, not the paper text
- **Decision:** ROUT (Motulsky & Brown 2006) follows the published algorithm as summarised in
  the brief (Lorentzian robust fit; RSDR = 68.27th percentile of |residuals| x N/(N-K);
  FDR-style test on the largest 30% of residuals with alpha_i = Q(N-(i-1))/N, t distribution
  with N-K df). The PMC page could not be opened from the sandbox (host blocked).
- **Verification done:** simulations reproduce the behaviour the paper reports (with Q = 1%,
  about 1-2% of clean Gaussian datasets have any flagged point; planted gross outliers are
  found in > 95% of datasets; the robust fit is barely moved by the outlier).
- **Open item:** an equation-by-equation read against the paper (merit-function scale and
  percentile interpolation) is listed in FINAL_REPORT.md as a follow-up.

## D-022 Classic published datasets are typed in and cited to their primary source
- **Decision:** Worked-example tests use small classic datasets cited to the original paper
  or book (e.g. Student 1908; Dobson 1983; Hollander & Wolfe 1973; Montgomery; Freireich
  1963; Treloar 1974; Crampton 1947). Each was typed in and kept only if TorusPlot reproduces
  the published results to printed precision. Sources are in `tests/fixtures/SOURCES.md`.
- **Alternatives:** only synthetic data (weaker validation).
- **Reason:** the brief asks for published worked examples. The numbers are facts from the
  primary literature; no file was copied from a copyleft or unlicensed repository. Several
  of these datasets are also redistributed with other statistics software; that is noted in
  the sources file so any of them can be dropped if the owner prefers.

## D-023 Anthropic provider uses the official `anthropic` SDK as an optional extra
- **Decision:** The optional Anthropic backend imports the official `anthropic` Python SDK
  lazily (MIT license; `pip install anthropic`). It is **not** in `requirements.txt`; without
  it the provider reports itself unavailable. Ollama, OpenAI and Gemini backends use only the
  Python standard library (`urllib`).
- **Alternatives:** raw HTTP for Anthropic too.
- **Reason:** Anthropic's guidance is to use the official SDK from Python; keeping it optional
  respects the brief's fixed runtime dependency list. Requests enable server-side refusal
  fallbacks by default (falling back to a plain request on SDKs that lack the parameter).

## D-024 Figures ignore machine-level matplotlib settings; goldens regenerated
- **Decision:** All rendering and exporting happens inside `style_context`, which starts
  from matplotlib's built-in defaults (`rcdefaults`) and applies only the TorusPlot style.
  Bundled fonts take priority over system fonts of the same family.
- **What happened:** The first CI run failed 21 golden-image tests on *both* Linux and
  Windows with identical numbers. Cause: the build sandbox has a system-wide
  `/etc/matplotlibrc` that disables text hinting and an older system copy of Inter, so the
  first goldens captured that machine's look. After the fix, local renders reproduce the CI
  differences digit-for-digit.
- **Golden regeneration:** 2026-11-13, logged in `tests/golden/REGENERATION_LOG.md`.
- **Reason:** A FigureSpec must render identically on every computer.

## D-025 pandas pinned below 3; lifelines pinned to >= 0.30.3
- **Decision:** `pandas>=2.2,<3`, `lifelines>=0.30.3,<0.31`.
- **Alternatives:** allow pandas 3.
- **Reason:** The newest lifelines requires `pandas<3`, so `pip` installs pandas 2.3 for
  users. Pinning makes the developer environment, CI and users agree on one combination.

## D-026 Test-recommendation policy
- **Decision:** `recommend_test` follows the conventional decision path (design -> normality
  -> equal variance) and always lists the assumption-free alternative. Normality for 3+
  groups is judged on residuals standardised within each group, so unequal spreads are not
  mistaken for non-normality. Small-sample caveats are stated explicitly (e.g. a two-sided
  Mann-Whitney test cannot reach P < 0.05 with n = 3 vs 3).
- **Alternatives:** always recommend Welch-type tests without pre-tests.
- **Reason:** Matches what journals and reviewers expect while being honest about the limits
  of pre-testing with small n.

## D-027 AI layer design (how each feature stays inside the hard rule)
- **Decision:**
  - *Prompts* give a model only table/column/group names, counts, and for statistics only
    result IDs, test names and comparison group names/IDs. No data values and no P values are
    sent (tested by planting distinctive P values and asserting they never reach a prompt).
  - *Outputs* are parsed as JSON, validated with `FigureSpec.model_validate`, then sanitised:
    unknown result IDs / tables / groups are removed with warnings. Validation errors are fed
    back to the model, at most 3 retries (4 calls).
  - *Match this figure* measures colours and proportions deterministically with Pillow
    (k-means palette) and cross-checks any colours a vision model reports against them. The
    feedback loop is capped at 3 rounds, scores renders with SSIM, and applies only
    whitelisted style/layout patches (never `data`, `annotations`, `title`, layer types).
  - *Digitize* is fully deterministic (no model), outputs an XY table flagged
    `approximate=True` and named "... (APPROXIMATE)"; the statistics registry refuses such
    tables unless `allow_approximate=True` is passed explicitly.
  - *Outside AI instructions* import JSON offline with lenient clean-up; objects that try to
    carry P values or stars are dropped with a warning.
- **Alternatives:** trusting model output; letting the model see results to "choose wisely".
- **Reason:** the model can pick *which* computed comparisons to show (by ID or "significant")
  without ever seeing or producing a number.
- **Not exercised:** no real provider (Ollama, Gemini, OpenAI, Anthropic) could be called from
  the sandbox; providers are tested against fakes that assert the request shape.

## D-028 MCP server design
- **Decision:** All logic lives in a plain `TorusSession` class (no MCP import); `server.py`
  registers its methods with the SDK's `MCPServer` (mcp 2.x; `FastMCP` was renamed). 15 tools:
  the 8 required ones plus list/save/load/schema helpers and `get_result`. Exports are confined
  to an output directory (path traversal rejected). Tool errors are user-facing messages.
- **Alternatives:** tools written directly against the SDK (harder to unit-test).
- **Reason:** testable without a running server; protocol tested in-memory and over stdio.
- **Not exercised:** not run against a real Claude Desktop / Claude Code client from the sandbox.

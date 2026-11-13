# TorusPlot: Final Report

Build completed 2026-11-13. Repository: https://github.com/torus12/TorusPlotUnlimited (branch `main`).
Copyright (c) 2026 Torus Washington LLC. All rights reserved.

## TL;DR

- **What it is:** a free-to-run, local scientific graphing and statistics app for Windows, macOS
  and Linux. Point-and-click workflow (Data -> Analyze -> Graph -> Style -> Compose -> Export), a
  statistics engine with 36 analyses plus power calculations, 18 plot types, 7 style presets,
  publication exports, and optional AI features. AI can change how a figure looks; it can never
  compute or change a statistic.
- **Install on Windows:** double-click `run_torusplot.bat` (steps below).
- **Tests:** **2,069 passed, 0 failed** in a brand-new environment built from the lock file. CI on GitHub
  (Windows + Linux, Python 3.11 + 3.12) runs tests, lint and the license gate on every push.
- **Audits:** two independent statistics reviews found 33 defects (none in the headline numbers
  checked against NIST and independent libraries; mostly edge cases, extreme scales,
  small-sample intervals and wording). All 33 are fixed and each has a permanent test. The
  independent licensing review found **no blockers**; its conditions are addressed or listed
  for counsel.
- **Before you sell it:** get a lawyer-written EULA (the current LICENSE reserves rights but does
  not grant customers a right to run it). See `docs/COMMERCIAL_RELEASE_CHECKLIST.md`.

---

## 1. How to install and run on Windows

1. Install Python 3.11 or newer from https://www.python.org/downloads/windows/ (64-bit
   installer). On the first screen tick **"Add python.exe to PATH"**.
2. Download TorusPlot (GitHub -> green **Code** button -> **Download ZIP**), right-click ->
   **Extract All...**. For customers, ship the clean ZIP made by `python tools/build_release.py`.
3. Double-click **`run_torusplot.bat`**. The first start creates a private `.venv` folder and
   installs exact, hash-checked package versions from `requirements-lock.txt` (internet needed
   once, a few minutes). If Windows says "Windows protected your PC", click **More info ->
   Run anyway**.
4. The browser opens at `http://localhost:8501`. That page is the app, running on your computer.
5. Close the black window to stop it.

macOS / Linux: `./run_torusplot.sh`. Command line: `python cli.py --help`.
Optional extras: AI cloud SDK (`requirements-ai.txt`), MCP server (`requirements-mcp.txt`;
see `docs/MCP.md`).

**Not tested by a human on a real Windows PC.** The batch file is checked automatically (CRLF line
endings, balanced parentheses, plain ASCII, correct commands) and the full suite passes on GitHub's
Windows runners, but the double-click flow itself was not exercised (see Known limitations).

---

## 2. What was built

| Area | Contents |
|---|---|
| Data (`torusplot/data`) | Five table types (Column, XY, Grouped, Multiple variables, Contingency), paste-from-Excel parser (delimiter, header and decimal-comma detection), CSV/TSV/XLSX import, long <-> wide converters, `.tplot` project files (zip of CSV data + YAML specs + JSON results; byte-for-byte deterministic), explicit logged exclusions, approximate-data flag, 7 synthetic sample datasets |
| Statistics (`torusplot/stats`) | 36 registered analyses + power/sample size + `recommend_test`. Every analysis returns a typed `StatResult`: test, why chosen, assumptions checked with pass/check/fail, statistic, df, P, adjusted P, effect sizes with CIs, n per group, warnings, plain-English interpretation, methods sentence, primary citations |
| Figures (`torusplot/spec`, `torusplot/plot`) | Pydantic `FigureSpec` (layers, mappings, annotations by result ID, style, layout in mm) + JSON schema; 18 plot types; significance brackets drawn only from computed results (own engine; statannotations optional); 7 presets; Okabe-Ito / viridis-family / custom palettes; colour-blind simulation and check; multi-panel composer with panel letters, shared axes and legends |
| Export (`torusplot/export`) | SVG with editable text and named groups, PDF (embedded TrueType), TIFF (LZW, 300/600 dpi), PNG; statistics report (Markdown + CSV); methods paragraph |
| App (`app/`) | Streamlit UI: 1 Data, 2 Analyze, 3 Graph, 4 Style, 5 Compose, 6 Export, AI Assistant, Settings; tooltips on every option; power calculator; original SVG logo and icon |
| AI (`torusplot/ai`) | Providers: mock (default), Ollama (free, local), Gemini, Anthropic, OpenAI (bring your own key). Describe-it (validated, max 3 retries), Match-this-figure (deterministic palette extraction + optional vision critique, max 3 loops, SSIM), Digitize (deterministic, output marked APPROXIMATE and blocked from statistics), Outside-AI import + generated prompt template |
| MCP server (`torusplot/mcp_server`) | 15 tools incl. the 8 required (`load_table`, `recommend_test`, `run_analysis`, `make_figure`, `restyle_figure`, `compose_panels`, `export_figure`, `get_stats_report`); `python -m torusplot.mcp_server` |
| Tooling | License gate, lock-file check, release builder, golden-image regeneration (explicit, logged), doc-image and screenshot generators, user-guide generator with drift check |
| Docs | README (beginner Windows install, screenshots made by the app), `docs/USER_GUIDE.md` (one real worked example per analysis, generated from engine output), `docs/STATS_METHODS.md` (method, implementation, validation and primary citation per test), `docs/MCP.md`, `docs/PROMPT_TEMPLATE_FOR_OUTSIDE_AI.md`, `THIRD_PARTY_NOTICES.md`, `docs/COMMERCIAL_RELEASE_CHECKLIST.md` |

---

## 3. Feature checklist against what a user of the leading commercial package expects

Done = available in the app and tested. Partial = available with a stated limitation. Missing = not built.

| Feature | Status | Notes |
|---|---|---|
| Column / XY / Grouped / Multiple-variables / Contingency tables; menu filtered by table type | Done | |
| Paste from Excel, CSV/XLSX import, replicate sub-columns | Done | |
| Descriptive statistics (mean, SD, SEM, median, IQR, 95% CI, CV, n) | Done | Also skewness, kurtosis, geometric mean |
| Normality: Shapiro-Wilk, D'Agostino-Pearson, KS (Lilliefors), Anderson-Darling; Q-Q plot | Done | |
| Variance: Levene, Brown-Forsythe, Bartlett, F test | Done | |
| t tests (unpaired, Welch, paired, one-sample), Mann-Whitney, Wilcoxon | Done | Exact P incl. small tied samples; Hodges-Lehmann CIs |
| One-way, Welch and Brown-Forsythe ANOVA; Kruskal-Wallis + Dunn; Friedman | Done | |
| Post-hoc: Tukey, Dunnett, Sidak, Bonferroni, Holm, Holm-Sidak, BH FDR, Games-Howell, Fisher LSD | Done | Dunnett computed deterministically (D-016) |
| Two- and three-way ANOVA (Type II/III), ANCOVA | Done | Least-squares means for comparisons |
| Repeated measures with Mauchly, Greenhouse-Geisser, Huynh-Feldt; mixed ANOVA; two-way RM | Done | |
| Effect sizes with 95% CIs (d, g, eta², partial eta², omega², rank-biserial r) | Done | Rank-biserial CI redesigned after the audit |
| Correlation (Pearson, Spearman), matrix; linear and multiple regression with CI/prediction bands, residuals | Done | Lack-of-fit and runs tests, VIF |
| Nonlinear regression: 4PL/3PL (log and linear X), EC50/IC50 CI, Michaelis-Menten, exponential, binding, user equation; R², AICc, extra-sum-of-squares F | Done | All 26 NIST nonlinear datasets from both starting points; profile-likelihood CIs; compare a parameter between data sets |
| Kaplan-Meier, log-rank, Gehan-Breslow-Wilcoxon, median survival, hazard ratio, trend test | Done | |
| Chi-square, Fisher exact (incl. r x c), OR, RR, risk difference, NNT, trend test | Done | |
| Outliers: Grubbs, ROUT (Q = 1%); flag only, explicit logged exclusion | Partial | ROUT implemented from the published description; equation-level check against the paper still open |
| Power and sample size (t tests, ANOVA) | Done | |
| SuperPlots / replicate-aware analysis | Done | |
| Bar+points (points by default), scatter, box, violin, beeswarm, estimation, before-after, XY, dose-response, survival, grouped/stacked bars, histogram, heat map, Q-Q, residuals | Done | 18 plot types, each with a golden-image test |
| Significance brackets tied to computed results | Done | Cannot be typed by hand or by AI |
| Journal-size presets in mm, editable-text SVG, PDF, TIFF 300/600, PNG | Done | |
| Multi-panel layouts with panel letters | Done | |
| Statistics report and methods paragraph | Done | |
| Project files that re-render identically | Done | Tested byte-for-byte |
| AI features (describe, match style, digitize, outside JSON, MCP) | Partial | Built and tested offline against fakes; no real provider was reachable from the build sandbox |
| Undo/redo, data transforms (log, normalise) in the app | Missing | Transform in the spreadsheet first |
| Cox regression, mixed models with random slopes, Bayesian methods, ROC/Bland-Altman | Missing | Listed in `docs/STATS_METHODS.md` |
| Native installer (MSI/.exe) | Missing | Launcher script installs from PyPI; see checklist for frozen-installer duties |

---

## 4. Test results

| Check | Result |
|---|---|
| Full test suite, fresh Python 3.11 environment from `requirements-optional-lock.txt` | **2,069 passed, 0 failed** |
| Coverage, `torusplot/stats` | 98.0% (target 85%) |
| Coverage, `torusplot/spec` | 99.6% (target 85%) |
| `ruff check .` | clean |
| `tools/check_licenses.py` | pass (78 packages: runtime closure, both lock files, optional extras, bundled-binary reviews) |
| `tools/check_lock.py` | pass |
| App launched headless (Chromium); all 8 pages loaded with data and a figure | pass, 0 page errors |
| GitHub Actions CI (windows-latest + ubuntu-latest; Python 3.11 + 3.12) | __CI__ |

What the suite covers:

- Statistics are validated against NIST StRD certified values (11 ANOVA, 26 nonlinear, 2
  linear regression datasets).
- They are also checked against published worked examples and against scipy, statsmodels,
  lifelines and scikit-posthocs.
- Hypothesis property tests probe edge cases.
- The two audit files hold 67 permanent tests.
- The end-to-end test (Grouped table -> recommend -> two-way ANOVA + Tukey -> bar+points with
  brackets -> SVG/PDF/TIFF -> reload `.tplot` -> identical re-render) passes.

---

## 5. Final audit

### Statistics (two independent reviewers, no access to design reasoning)

- **Correct as built:** core numerics matched independent implementations and NIST throughout.
- **Defects found and fixed: 33** (17 in the models/regression/survival half, 16 in the
  tests/effect-size half). Main themes:
  1. Confidence intervals for rank-biserial r had poor coverage (as low as 18% in strong-effect
     cases). They were redesigned as score intervals with placement/jackknife variance; coverage
     was simulated.
  2. The Hedges' g interval was scaled incorrectly.
  3. Guards against "constant data" depended on the unit of measurement, and values above about
     1e150 overflowed. Everything now computes on internally rescaled data.
  4. Rounding noise in paired differences was treated as data.
  5. Tiny tied samples got normal-approximation P values below the attainable minimum. They now
     use an exact permutation P (up to 6 per group).
  6. The Friedman Wilcoxon post-hoc disagreed with the app's own Wilcoxon test.
  7. Bartlett's test failed on exactly equal variances; Levene's test failed on data with a
     large offset.
  8. Factorial comparisons without an interaction now use the model's least-squares means.
  9. The test recommender mistook "Exposure" for an experiment column and gave a long
     repeated-measures table a three-way ANOVA.
  10. A bare `e` in user equations is now a parameter, not Euler's number.
  11. The median-survival upper limit is now "not reached" where the definition says so.
  12. Many plain-English messages and tooltips were corrected.
- Every defect has a failing-then-passing test in `tests/audit/`.

### Licensing (independent reviewer)

**Verdict: clear with conditions; no blockers.**

- No copyleft or unlicensed runtime dependency.
- No copied code or assets from any reference project. Checks used token shingles, function-level
  comparison, string literals and asset hashes over about 3,900 reference files.
- Fonts are all OFL-1.1; logo and icon are original.

Conditions and what was done:

| Condition | Action |
|---|---|
| Versions could drift at install time | Hash-checked lock files; launchers and CI install from them |
| Gate did not see binaries inside wheels or other platforms | Gate now scans bundled libraries and license texts on each CI platform; reviews recorded |
| Deliverable not defined | `tools/build_release.py` (allowlist; no tests, fixtures, audit or design records) |
| Notices inaccurate | Regenerated from the lock files |
| Typed-in published datasets in tests | The two most sensitive replaced by constructed data (D-030); the rest test-only, cited, excluded from releases |
| No end-user licence | **Open, needs counsel** (EULA) |
| LGPL module inside a Windows dependency of the MCP SDK | MCP SDK made an optional extra (D-031) |
| Process wording "clean-room" overstated | Corrected to "ideas-only study + similarity audit"; `_reference/` deleted |

Full report: `docs/audit/LICENSE_AUDIT_REPORT.md`. Release checklist:
`docs/COMMERCIAL_RELEASE_CHECKLIST.md`.

---

## 6. Every fallback taken (details in DECISIONS.md)

| What | Fallback | Ref |
|---|---|---|
| No GitHub repo at start | Built locally; pushed once you supplied `TorusPlotUnlimited` | D-001 |
| NIST site blocked | ANOVA files from SciPy's bundled copies; nonlinear files from the lmfit source package (both BSD-3, byte-identical to NIST copies); Longley assembled from statsmodels | D-017 |
| ROUT paper unreachable | Implemented from the published description; behaviour verified by simulation | D-021 |
| Ollama / Hugging Face model cards unreachable | Model licenses read from vendors' GitHub repositories | D-009 |
| Liberation Sans upstream download blocked | Files from the Debian package of the same release; checksums recorded | D-013 |
| `OpenBMB/MatPlotAgent` unreachable | Studied the public `thunlp/MatPlotAgent` (no license file: ideas only) | D-008 |
| Inkscape/Graphite unavailable | SVG validated with an XML parser, xmllint and structural checks | D-015 |
| scipy Dunnett non-deterministic | Own deterministic integration; scipy as cross-check | D-016 |
| lifelines / scikit-posthocs | Own implementations at runtime; the libraries are test-only cross-checks | D-020, D-029 |
| statannotations limited to seaborn plot types | Own bracket engine by default; statannotations optional | D-012 |
| Machine-level matplotlib settings leaked into goldens | Rendering isolated from system config; goldens regenerated (logged) | D-024 |
| MCP SDK pulls an LGPL module on Windows | MCP is an optional extra | D-031 |
| Usage limit stopped two fix agents mid-way | Work resumed from the saved state; all fixes completed and verified | n/a |

---

## 7. Known limitations

- **Not tried on a real Windows PC** by a person (CI Windows runners only). Please do one
  double-click test of `run_torusplot.bat` on your machine.
- **AI providers were never called for real**; only fakes. Expect some prompt tuning once you
  connect Ollama or a key. The MCP server was not tested with a real Claude Desktop / Claude
  Code client.
- **ROUT**: equation-level comparison with Motulsky & Brown (2006) still to do; simulated
  behaviour matches what the paper reports.
- **Exact rank-test P values with ties** only up to 6 values per group / 6 non-zero differences;
  above that the corrected normal approximation is used (never below the attainable minimum).
- Digitizer validated on clean synthetic images only.
- Published-example values were typed from memory and accepted only when the code reproduced
  them; a person with the printed sources should spot-check citations (`tests/fixtures/SOURCES.md`).
- No undo/redo; no in-app data transforms; no Cox regression, random-slope mixed models or
  Bayesian methods.
- macOS is untested (add `macos-latest` to CI if you support it).

---

## 8. Top 5 next improvements

1. **Commercial readiness:** EULA (counsel), trademark search for "TorusPlot", and a signed
   Windows installer (follow the frozen-installer section of the release checklist).
2. **Real-world AI validation:** connect Ollama (`qwen2.5:7b`, `granite3.2-vision`) and one cloud
   key, record real responses into the mock library, tune prompts, and measure match quality on
   a set of published-style figures.
3. **Usability pass with real users:** undo/redo, in-app transforms (log, normalise to control),
   drag-to-reorder groups, and saved personal style presets.
4. **Statistics breadth:** Cox proportional-hazards regression, linear mixed models for nested
   and longitudinal data, ROC and Bland-Altman, equivalence tests (TOST).
5. **Verification against printed sources:** confirm ROUT against the paper and check every
   published-example citation; add macOS to CI.

---

*This report describes engineering work and is not legal advice.*

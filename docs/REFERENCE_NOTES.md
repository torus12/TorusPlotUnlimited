# Reference notes (clean-room study log)

**Clean-room policy.** TorusPlot is proprietary software owned by Torus Washington LLC. The
projects below were shallow-cloned into `_reference/` (git-ignored, never committed, never
packaged, never imported) on 2026-11-13 and **read only**: nothing in them was executed,
installed, or built. Their files were treated as untrusted data. We studied them for
*ideas* (what features exist, how workflows are organised, what a scientist expects to see)
and for the *public API surface* of the two permissively licensed projects we either depend
on (statannotations) or take naming inspiration from (tidyplots). **No source code,
docstrings, test data, sample data, icons, screenshots or other assets were copied** into
TorusPlot from any of them. Everything in TorusPlot that resembles an idea listed here is
written fresh from this prose description, from primary statistical literature, and from
the documentation of our own permissively licensed dependencies. Each licence below was
read from the licence file inside the clone, not recalled from memory. Where a product name
would be a third-party trademark we say "the commercial incumbent" instead.

Roles used below:

- **IDEAS ONLY** - read for feature and workflow ideas; no code, no API shape taken.
- **API DESIGN REFERENCE** - permissive licence; we looked at public function *names* to
  inform our own naming. Still no code copied.
- **RUNTIME DEPENDENCY** - permissive licence; installed from PyPI and called through its
  public API.
- **EXPORT TARGET** - a third-party program whose import behaviour we target with our
  exported files. No code relationship at all.

---

## 1. LabPlot

- **URL:** https://github.com/KDE/labplot (GitHub mirror; upstream is invent.kde.org/education/labplot)
- **Commit studied:** `4c336decf9d688f8b796504ba8aab323813395b0` (sparse, blobless shallow
  clone of `src/backend`, `src/frontend`, `doc`, `LICENSES`, plus top-level files)
- **Licence (verified):** **GPL-2.0-or-later** for the application. Files read: the
  `LICENSES/` directory (REUSE layout; contains `GPL-2.0-or-later.txt`, `GPL-2.0-only.txt`,
  `GPL-3.0-only.txt`, `GPL-3.0-or-later.txt`, `LGPL-2.0-or-later.txt`,
  `LGPL-3.0-or-later.txt`, `BSD-2-Clause.txt`, `BSD-3-Clause.txt`, `MIT.txt`,
  `Apache-2.0.txt`, `CC0-1.0.txt`, `CC-BY-3.0.txt`), the per-file `SPDX-License-Identifier`
  headers (797 of the 818 tagged files under `src/` say `GPL-2.0-or-later`), and
  `org.kde.labplot.appdata.xml` (`project_license` = `GPL-2.0+`).
- **Role:** IDEAS ONLY (copyleft).

**What we learned**

- *Data model.* A project is a tree shown in a project explorer: folders, workbooks
  (tabbed containers), spreadsheets, matrices, worksheets (the page plots live on), notes,
  notebooks and "datapickers". Selecting any node shows its settings in a dockable
  properties panel; edits are undoable and several objects of the same kind can be edited
  at once.
- *Spreadsheet.* The column is the unit of data. Each column has a name, a mode (numeric,
  text, month name, day name, date-time) and a display format, plus a "plot designation"
  (X, Y, Z, X-error, X-error plus/minus, Y-error, Y-error plus/minus) that lets the app
  guess sensible plots. Cells can be *masked* (kept but ignored by plots and analyses) and
  values can be masked or dropped by rule (equal to / greater than / less than ...).
  Columns can be filled by generators: row numbers, constant, equidistant values (by count
  or by step), random values from a chosen distribution, or a formula over other columns.
  A separate "matrix" container holds Z(x, y) grids with a table view and an image view.
- *Column statistics panel.* A child "statistics spreadsheet" reports, per column, a
  user-selected set of metrics: count, minimum, maximum, arithmetic / geometric / harmonic
  / contraharmonic mean, mode, first quartile, median, third quartile, IQR, percentiles
  (1, 5, 10, 90, 95, 99), trimean, variance, standard deviation, mean deviation, mean
  deviation around the median, median deviation, skewness, kurtosis, entropy, range, sum.
- *Hypothesis-test object.* A dedicated project node offers one-sample, two-sample, paired
  and Welch t tests, one-way and repeated-measures one-way ANOVA, Mann-Whitney U,
  Kruskal-Wallis, Wilcoxon, Friedman, chi-square independence and goodness of fit,
  log-rank, Mann-Kendall trend, Wald-Wolfowitz runs and Ramirez-Runger tests, with its own
  results view.
- *Analysis menu.* Every analysis produces a new curve that stays linked to its source
  X/Y columns and is restyled like any other curve. The handbook lists: data reduction
  (line simplification: Douglas-Peucker, Visvalingam-Whyatt, Reumann-Witkam, perpendicular
  distance, n-th point, radial distance, interpolation, Opheim, Lang), differentiation
  (1st to 6th order, selectable accuracy order), integration (rectangle, trapezoid, Simpson
  1/3 and 3/8), interpolation (linear, polynomial, cubic / Akima / Steffen splines incl.
  periodic variants, cosine, exponential, piecewise cubic Hermite, rational; evaluated as
  function, first or second derivative, or integral), smoothing (central or lagged moving
  average, percentile filter, Savitzky-Golay; choice of edge padding mode and window
  weights), curve fitting, Fourier filter (low / high / band pass / band reject, in ideal,
  Butterworth, Chebyshev I/II, Legendre or Bessel form; cutoff in frequency, fraction or
  index units) and Fourier transform (window function, output as magnitude / amplitude /
  phase / dB etc., one- or two-sided, x axis as frequency / index / period). The source
  tree shows newer additions: baseline correction, convolution / deconvolution, auto- and
  cross-correlation, Hilbert transform, piecewise-linear fit, peak finding, change-point
  detection, kernel density estimation and seasonal decomposition of time series.
- *Fit models.* Grouped into categories: basic (polynomial, power, exponential, inverse
  exponential, Fourier series), peak (Gaussian, Lorentz, sech, logistic, Voigt,
  pseudo-Voigt), growth / sigmoidal (arctangent, tanh, algebraic sigmoid, sigmoid, error
  function, Hill, Gompertz, Gudermann), statistical distributions, and a free-form custom
  expression with any number of parameters. A parameter table gives each parameter a start
  value, a "fixed" switch and lower / upper bounds. Options: optional weight column,
  maximum iterations, tolerance, number of evaluated points, evaluate over the full range
  or only the fitted x-range, and "use results as new start values". Results appear as a
  text report with goodness-of-fit statistics.
- *Statistical plot types* beyond the XY curve: histogram, box plot, bar plot, lollipop
  plot, KDE plot, Q-Q plot, Pareto chart, run chart, process-behaviour (control) chart;
  plus helper objects: reference line, reference range, custom point, cursor, info element.
- *Plot property editor.* One dock per object type, each split into tabs. Observed tab
  sets: plot area - General, Title, Layout, Range Breaks, Area, Stack, Cursor; axis -
  General, Title, Line, Ticks, Labels, Grid; XY curve - General, Line, Symbol, Values,
  Filling, Error Bars, Margins; legend - General, Title, Background, Layout; histogram -
  General, Shape, Symbol, Values, Filling, Error Bars, Margins; box plot - General, Box,
  Markers, Whiskers, Margins; bar plot - General, Bars, Values, Error Bars; worksheet
  (page) - General, Background, Layout. Reusable sub-panels (line, background, error bar,
  value label) recur across docks. The front end also has template and theme handlers so
  a configured look can be saved and re-applied.
- *Image digitiser ("datapicker") workflow.* (1) Load an image of a published plot;
  zoom and rotate it if skewed. (2) Choose the coordinate system of the plot in the image:
  linear Cartesian, log-x, log-y, log-log (natural or base 10), polar (degrees or radians)
  or ternary. (3) Click three reference ("axis") points on the image and type their true
  coordinates; this calibrates pixel-to-data mapping. (4) Create one or more named curves;
  each curve declares its error-bar type for X and Y (none, symmetric, asymmetric).
  (5) In "set curve points" mode, click data points; each click is converted to data
  coordinates and appended to that curve's own spreadsheet, which then behaves like any
  other data table. Helpers: a magnifier with several zoom levels, nudging the last point
  with the arrow keys, select-and-move for multiple points, and draggable handles for
  error-bar ends. (6) Semi-automatic mode for solid lines: the image is thresholded into a
  black-and-white mask by user-set ranges of hue, saturation, value, intensity and
  foreground (distance from the background colour), with a histogram to guide the range
  choice; connected runs of mask pixels become selectable "segments" (minimum segment
  length adjustable); clicking a segment places points along it at an adjustable spacing.
  The user can toggle between the original and the processed image.

**How TorusPlot uses the idea (written fresh)**

- TorusPlot's tables are typed (see Helix below) rather than free-form spreadsheets, but
  we adopt the ideas of an *exclude-without-deleting* cell state, per-column data kinds,
  column fill helpers, and a per-column descriptive-statistics panel with a selectable
  metric list. All are implemented on pandas / NumPy / SciPy.
- Our XY analysis menu (smoothing, differentiation, integration / area under the curve,
  interpolation, FFT, histogram / KDE, curve fitting with a model catalogue, per-parameter
  start / fixed / bounds, optional weights) is specified from the feature list above and
  implemented with SciPy and our own code. Each analysis is a derived, re-computable node
  attached to its source table.
- The figure inspector in the Streamlit UI groups controls in the same *spirit* (figure,
  axes, series, error bars, legend, annotations), each group mapping to a section of
  TorusPlot's own FigureSpec. The grouping and wording are ours.
- "Digitise a plot image" is planned as: calibrate axes by clicking reference points and
  typing values (linear / log scales), pick points per named series, optional
  colour-range masking to trace a line, output to an ordinary XY table. Implemented with
  Pillow and NumPy code of our own; optionally assisted by a local vision model that
  proposes axis ranges for the user to confirm.

**Code copied: none.**

---

## 2. Helix

- **URL:** https://github.com/mickaphd/Helix (found at the first URL tried; project site
  helix-desktop.com)
- **Commit studied:** `aa63df1e511d104fe48c33f87ae4fb219343454a`
- **Licence (verified):** **GPL-3.0-or-later**. File read: `LICENSE` (GNU GPL version 3
  text); the README's licence section says "version 3 or later". Third-party components
  are listed in `THIRD-PARTY-NOTICES.md`.
- **Role:** IDEAS ONLY (copyleft; not MIT/BSD/Apache, so nothing beyond ideas).

**What we learned**

- *Project shape.* A project is a list of **tables**; each table owns its **analyses**
  and **graphs** as children. Analyses store *what to compute* (type + parameters + which
  data sets), never results; results are recomputed from the table whenever it changes,
  so results and figures can never go stale. The project file is plain JSON with
  forward/backward-compatibility promises (unknown fields preserved, bad values fall back
  to defaults) and holds no personal information.
- *Table types.* Six types; the type decides how data are entered and which analyses and
  graphs are offered:
  - **Column** - independent groups side by side (control, dose 1, dose 2 ...).
  - **XY** - first data column is X; one or more Y data sets, optionally with replicate
    sub-columns per Y.
  - **Grouped** - row categories (time points, genes) by column groups, with replicate
    sub-columns per group.
  - **Multiple Variables** - one row per subject, one column per variable (tidy/wide).
  - **Contingency** - counts of outcomes; whole numbers only.
  - **Survival** - one subject per row: time, then event (1) or censored (0) under its
    group's column.
  Every table has a leading row-title column whose labels follow into results and graphs.
- *Table type -> analyses offered.*
  - Column: descriptive statistics; normality / lognormality test; compare two groups
    (unpaired, Welch, paired, ratio-paired, lognormal t tests; Mann-Whitney; Wilcoxon);
    compare many groups (one-way ANOVA, Welch ANOVA, repeated-measures ANOVA with
    Greenhouse-Geisser correction, Kruskal-Wallis, Friedman, with Tukey / Bonferroni /
    Dunnett / Games-Howell / Dunnett T3 / Dunn follow-ups); compare one group to a value;
    identify outliers (ROUT or Grubbs).
  - XY: correlation (Pearson / Spearman); linear regression; area under the curve;
    nonlinear regression (dose-response 3-, 4-, 5-parameter; exponential growth; one-phase
    decay; Michaelis-Menten; hyperbola; polynomials; semilog line), with outlier removal
    and comparison of two curves by extra-sum-of-squares F test; interpolation of unknowns
    from a standard curve; simple logistic regression.
  - Grouped: two-way ANOVA (ordinary, repeated measures, rank-based) with within-family
    multiple comparisons; multiple t tests (one per row) with Holm or FDR correction.
  - Multiple Variables: descriptive statistics; normality; correlation matrix; linear
    regression between two chosen variables; multiple regression; outliers.
  - Contingency: Fisher's exact (2 x 2) or chi-square, with odds ratio, relative risk,
    phi / Cramer's V, and a warning for small expected counts.
  - Survival: Kaplan-Meier with median survival and log-rank / trend / Gehan-Breslow-
    Wilcoxon tests and hazard ratios; Cox proportional-hazards regression.
- *Table type -> graphs offered.*
  - Column: individual points, box-and-violin, mean-with-error, histogram, pie, donut.
  - XY: scatter / line, XY bars, XY area (also survival curve and volcano as special cases).
  - Grouped: interleaved scatter, box / violin, interleaved bars, lines, stacked bars,
    horizontal bars, heatmap, volcano.
  - Multiple Variables: scatter of two chosen variables, histogram, heatmap, volcano.
  - Contingency: grouped bars, stacked bars, horizontal bars, heatmap.
  - Survival: survival curve.
- *Guided analysis.* A short wizard asks which data sets, whether paired, and whether to
  assume a normal distribution (or tests it), then names and runs the exact test. Results
  state the exact test that ran, a plain-language conclusion, every statistic, the
  multiple-comparison table and descriptive statistics. A test that cannot run says why
  (too few values, no complete row, non-integer count) instead of showing an empty result.
- *Methods transparency.* A methods document states, per analysis, the estimator and every
  choice made (n-1 SD, quantile definition, which pairs enter a paired test, type III
  sums of squares, how P values are formatted: 4 significant figures, "<0.0001" below).
- *Graph ideas.* Statistics drawn on the graph from the linked analysis (stars or P
  values, regression line with confidence band and equation, fitted curve); colour-blind-
  safe palettes first; per-series and per-point styling; "match another graph's look";
  export at true printed size at 150 / 300 / 600 dpi or as SVG.

**How TorusPlot uses the idea (written fresh)**

- TorusPlot's data layer defines its own typed tables - Column, XY, Grouped, Multiple
  Variables, Contingency (and later Survival) - as Python dataclasses over pandas, with a
  registry that maps each table type to the analyses and figure kinds it can feed. The
  registry content is derived from standard biostatistics practice and the lists above;
  the code and schema (`.tplot`) are our own design.
- Analyses are stored as declarative requests and recomputed from the table, so results
  and brackets on figures always match the data.
- Our results objects always carry: exact test name, assumptions checked, statistic,
  degrees of freedom, P value, effect size with CI, and a "why this could not run" message.
- A METHODS document for TorusPlot will describe our own estimators (our own
  implementations on NumPy / SciPy, validated against published reference values).

**Code copied: none.**

---

## 3. pingouin

- **URL:** https://github.com/raphaelvallat/pingouin
- **Commit studied:** `1ffb923219c636a396c3f749eac7092b00e80e8d`
- **Licence (verified):** **GPL-3.0-only**. Files read: `LICENSE` (GNU GPL version 3 text)
  and `pyproject.toml` (`license = "GPL-3.0-only"`).
- **Role:** IDEAS ONLY. Copyleft: **never installed, never imported, never a dependency**
  of TorusPlot (the licence gate in `tools/check_licenses.py` would reject it).

**What we learned** (feature names and result-table column names only)

- *Scope of a "complete" stats toolbox.* ANOVA family: one-way and N-way between
  (`anova`), ANCOVA, repeated measures (`rm_anova`), mixed between-within
  (`mixed_anova`), Welch ANOVA, epsilon estimators. t tests incl. one-sample, paired,
  Welch, and equivalence (TOST). Non-parametric: Mann-Whitney U, Wilcoxon signed-rank,
  Kruskal-Wallis, Friedman, Cochran Q, median-absolute-deviation outlier rule. Post hoc:
  pairwise tests (parametric or not, between / within / mixed), Tukey HSD, Games-Howell,
  generic multiple-comparison correction. Correlation: Pearson, Spearman, Kendall and
  robust variants (biweight midcorrelation, percentage bend, Shepherd, skipped), partial
  and semi-partial, repeated-measures correlation, distance correlation, pairwise
  correlation tables and correlation matrices with significance. Regression: linear,
  logistic, mediation. Assumption checks: normality (Shapiro-Wilk and alternatives),
  homoscedasticity (Levene / Bartlett), sphericity (Mauchly), Anderson-Darling,
  multivariate normality, Box's M. Effect sizes and their conversion and confidence
  intervals (analytic and bootstrap). Power: t test, ANOVA, repeated-measures ANOVA,
  correlation, chi-square. Contingency: chi-square independence (several power-divergence
  variants), McNemar. Reliability: Cronbach's alpha, intraclass correlation. Bayes factors
  for t tests, correlations and binomial tests. Plots: Bland-Altman, paired plot, Q-Q
  plot, repeated-measures correlation plot.
- *Options a scientist expects.* `alternative` = two-sided / greater / less everywhere;
  `paired`; automatic Welch correction when variances or sample sizes differ
  (`correction="auto"`); confidence level; sums-of-squares type (1, 2, 3) for unbalanced
  designs; choice of effect size (partial eta-squared, eta-squared, generalised
  eta-squared; Cohen d, Hedges g, r, AUC, common-language effect size, odds ratio);
  sphericity correction on / off / auto; `detailed` output adding sums of squares and mean
  squares; P-adjustment method (none, Bonferroni, Sidak, Holm, FDR Benjamini-Hochberg,
  FDR Benjamini-Yekutieli); parametric vs non-parametric pairwise tests; marginal vs
  simple-effects comparisons in mixed designs; wide- or long-format input.
- *Output columns per result table* (current naming uses underscores; releases before the
  rename used dashes / percent signs, e.g. "p-val", "CI95%"):
  - t test: `T`, `dof`, `alternative`, `p_val`, `CI95`, `cohen_d`, `power`, `BF10`.
  - One-way / N-way ANOVA: `Source`, `SS`, `DF`, `MS`, `F`, `p_unc`, `np2` (the effect-size
    column is named after the chosen effect size).
  - Welch ANOVA: `Source`, `ddof1`, `ddof2`, `F`, `p_unc`, `np2`.
  - ANCOVA: `Source`, `SS`, `DF`, `F`, `p_unc`, `np2`.
  - Repeated-measures ANOVA: `Source`, `ddof1`, `ddof2`, `F`, `p_unc`, `ng2`, `eps`, and
    when relevant `p_GG_corr`, `W_spher`, `p_spher`, `sphericity` (boolean).
  - Mixed ANOVA: same family, one row each for between factor, within factor and
    interaction, with `np2`, `eps`, `p_GG_corr`, `W_spher`, `p_spher`, `sphericity`.
  - Pairwise tests: `Contrast`, `A`, `B`, `Paired`, `Parametric`, `T` (or `U_val` /
    `W_val` for the non-parametric versions), `dof`, `alternative`, `p_unc`, `p_corr`,
    `p_adjust`, `BF10`, and one effect-size column named after the chosen effect size
    (default `hedges`).
  - Tukey: `A`, `B`, `mean_A`, `mean_B`, `diff`, `se`, `T`, `p_tukey`, effect size.
  - Games-Howell: `A`, `B`, `mean_A`, `mean_B`, `diff`, `se`, `T`, `df`, `pval`, effect size.
  - Correlation: `n`, `r`, `CI95`, `p_val`, `BF10`, `power` (plus `outliers` for robust
    methods). Pairwise correlation table adds `X`, `Y`, `method`, `covar`, `alternative`,
    `p_unc`, `p_corr`, `p_adjust`. Partial correlation: `n`, `r`, `CI95`, `p_val`.
    Repeated-measures correlation: `r`, `dof`, `pval`, `CI95`, `power`.
  - Mann-Whitney: `U_val`, `alternative`, `p_val`, `RBC` (rank-biserial), `CLES`.
  - Wilcoxon: `W_val`, `alternative`, `p_val`, `RBC`, `CLES`.
  - Kruskal-Wallis: `H`, `dof`, `p_unc`. Friedman: `W` (Kendall's W), `Q`, `dof`, `p_unc`
    (or, for the F-test variant, `W`, `F`, `dof1`, `dof2`, `p_unc`). Cochran: `Q`, `dof`,
    `p_unc`.
  - Normality: `W`, `pval`, `normal` (boolean). Homoscedasticity: test statistic (W for
    Levene, T for Bartlett), `pval`, `equal_var` (boolean). Sphericity (Mauchly) returns a
    boolean, W, chi-square, degrees of freedom and P. Epsilon: Greenhouse-Geisser,
    Huynh-Feldt or lower-bound.
  - Linear regression: `names`, `coef`, `se`, `T`, `pval`, `r2`, `adj_r2`, `CI2.5`,
    `CI97.5` (optional relative-importance columns). Logistic regression: `names`, `coef`,
    `se`, `z`, `pval`, `CI2.5`, `CI97.5`.
  - Chi-square independence: `test`, `lambda`, `chi2`, `dof`, `pval`, `cramer`, `power`.
  - Intraclass correlation: `Type`, `ICC`, `F`, `df1`, `df2`, `pval`, `CI95`.
  - TOST: `bound`, `dof`, `pval`.
- *Design lesson.* Every test returns one tidy table with an uncorrected and a corrected
  P column side by side, the effect size in the same row, and booleans for assumption
  checks, so a result can be pasted straight into a report.

**How TorusPlot uses the idea (written fresh)**

- TorusPlot's statistics engine (`torusplot/stats`) is implemented independently on NumPy
  and SciPy from textbook formulas and primary papers, and validated against published
  reference values and R output - not against, and not by calling, pingouin.
- We adopt the *convention* that every analysis returns one tidy pandas table with:
  uncorrected and adjusted P values in separate columns, the adjustment method named,
  effect size and its CI in the same row, degrees of freedom, and explicit sphericity /
  normality / equal-variance flags. Our column names are our own scheme (documented in
  the TorusPlot stats reference); where a generic statistical abbreviation coincides
  (e.g. "F", "T", "dof") that is ordinary notation, not copied expression.
- Our option vocabulary (two-sided / greater / less, paired, Welch auto-correction, SS
  type, P-adjust method, effect-size choice) follows common statistical usage.

**Code copied: none.**

---

## 4. figrecipe

- **URL:** https://github.com/ywatanabe1989/figrecipe
- **Commit studied:** `162f6237790420462432f2c5ae8ad9e98b5bf259`
- **Licence (verified):** **AGPL-3.0-only**. Files read: `LICENSE` (GNU Affero GPL
  version 3 text) and `pyproject.toml` (`license = "AGPL-3.0-only"`). `CLA.md` describes a
  dual-licensing model (commercial licence on request); we have none.
- **Role:** IDEAS ONLY (network copyleft; never installed or imported).

**What we learned**

- *Figure = recipe + data + image.* Saving a figure writes three things together: the
  rendered image, a human-readable YAML recipe that records each plotting call and its
  arguments, and CSV files holding the arrays that were plotted (one per identified
  trace). The figure can later be reproduced from the recipe alone, without the original
  analysis script.
- *Restyle without recomputing.* Because data and style are separated, the same recipe can
  be re-rendered under a different named style preset (journal style, dark style, plain
  matplotlib) with the data untouched.
- *Bundle.* A single zip can carry a semantic spec ("what is plotted"), a style file ("how
  it looks"), the immutable data CSV, the recipe, and regenerated exports. Spec and style
  are editable; data and exports are not.
- *Validation.* A "validate" step re-renders from the recipe and compares against the
  original image (mean-squared-error threshold) to prove the recipe is faithful.
- *Millimetre layout.* Figure and axes sizes and margins are specified in millimetres to
  hit journal column widths exactly; whitespace auto-cropping with a mm margin.
- *Composition.* Panels (recipes or images) are composed into a multi-panel figure by
  horizontal / vertical / grid layout or free-form mm coordinates, with automatic panel
  letters; alignment and distribution helpers.
- *Other.* Data extraction back out of a recipe; a "hit map" image so a GUI can map clicks
  to figure elements; statistical bracket annotation; a browser GUI editor with an editable
  data worksheet; a linter that flags non-reproducible plotting habits.
- *MCP server tools* (names as documented in the README and the bundled skill pages; the
  skill pages show them with a `plt_` prefix): core - `plot` (create a figure from a
  declarative spec dict, optionally saving the recipe), `reproduce` (re-render a recipe to
  PNG / PDF / SVG), `compose` (combine panels by layout keyword or free-form mm
  coordinates, with panel letters), `crop` (trim whitespace with a mm margin), `info`
  (recipe metadata), `validate` (re-render and compare against the original with an error
  threshold), `extract_data` (get plotted arrays back out of a recipe); discovery - list
  plot types, list style presets; a per-plot-type tool for each supported plot (line,
  scatter, bar, horizontal bar, histogram, box, violin, error bar, filled band, stacked
  area, contour, image / heatmap, pie, stem, spectral plots and more); a set of
  pre-styled "scientific" plot tools (mean with SD / CI, median with IQR, shaded line,
  violin with points, scatter with marginal histograms, annotated heatmap, confusion
  matrix, raster, ECDF); diagram tools (create from spec, render, compile to Mermaid or
  Graphviz, presets); and MCP *resources* that serve the declarative spec schema and a
  cheat sheet so an agent can learn the format on demand. The README table additionally
  lists a text-to-speech relay tool.

**How TorusPlot uses the idea (written fresh)**

- TorusPlot already centres on its own declarative **FigureSpec** (our schema, our
  renderer). We take from this study only the confirmation of the *concept*: a `.tplot`
  project stores spec + data + style separately so any figure is re-renderable and
  restylable, and exports are always regenerated, never edited.
- Planned from this idea, in our own design: named style presets applied at render time;
  sizes in mm; a "verify round-trip" check that re-renders a saved figure and compares
  hashes; a composer for multi-panel figures.
- TorusPlot's MCP server (milestone M7) exposes its own tool set designed around our
  objects (create table, run analysis, create / update figure from FigureSpec, render,
  export, list styles, describe schema). Tool names, arguments and schemas are ours.

**Code copied: none.**

---

## 5. paper-plot-skills

- **URL:** https://github.com/Trae1ounG/paper-plot-skills
- **Commit studied:** `cde5e8486b3511d7819d54cc569df5d5636aa91a`
- **Licence (verified):** **No license file (all rights reserved).** Checked: repository
  root listing (only `README.md`, `originals/`, `plot-from-data/`, `plot-from-image/`,
  `repro/`), a recursive search for LICENSE / COPYING files (none), and the README (no
  licence statement).
- **Role:** IDEAS ONLY (no licence grants any reuse; the example figures also derive from
  third-party papers).

**What we learned**

- Two agent "skills" packaged as instruction files plus reference notes and example
  scripts: **plot-from-image** and **plot-from-data**.
- *Plot-from-image approach.* Given a screenshot of a published figure the agent:
  (1) measures the image's pixel size and aspect ratio and sets the figure size to match;
  (2) tries to match the figure to a known style in its library (fast path);
  (3) otherwise analyses from scratch along a checklist - font family (serif vs
  sans-serif, LaTeX-rendered or not) and relative font sizes; axis frame style (L-shaped
  vs four-sided box, arrowheads on axis ends); tick direction (in / out); colour
  identification (default colour cycle vs custom, sampled from the image); grid style
  (dashed / dotted / none); legend placement; special elements (insets, broken axes,
  radar grids, annotation boxes, hatching, value labels);
  (4) writes a script, renders it, visually compares with the original and iterates
  against a checklist (aspect ratio, font family, colours within a small RGB tolerance,
  frame style, tick direction, grid, legend position, annotation positions).
  It also keeps a growing list of "lessons learned" (e.g. smooth noisy curves before
  plotting, typical band transparency, how far radar labels sit from the web).
- *Plot-from-data approach.* A small catalogue of curated, named styles distilled from
  real published figures (paired bars with gain arrows, grouped bars with hatching and
  value labels, line with confidence band, training curve with reference lines, loss curve
  with zoom inset, annotated cluster scatter, broken-axis scatter, dual-series radar).
  Each style has a parameter sheet (fonts and weights, sizes, colours, frame and tick
  settings, when to use it, variants). The user picks a style and supplies data; output is
  fixed at 300 dpi.

**How TorusPlot uses the idea (written fresh)**

- TorusPlot's AI layer (milestone M6) gets a "match this figure's look" feature of our
  own design: a local vision model is asked for a *structured* description (aspect ratio,
  font class, frame style, tick direction, grid, palette as hex values, legend position,
  plot kind), aspect ratio and dominant colours are additionally measured deterministically
  with Pillow / NumPy, and the result is mapped onto FigureSpec style fields - never onto
  free-form generated code.
- Our style presets are original parameter sets expressed in FigureSpec terms, built from
  journal author guidelines and our own design choices; no parameter sheets, scripts or
  images from this repository are used.

**Code copied: none.**

---

## 6. MatPlotAgent

- **URL requested:** https://github.com/OpenBMB/MatPlotAgent - **not reachable** (three
  `git ls-remote` / clone attempts all ended in an authentication prompt, which is what
  GitHub returns for a non-existent or private repository; other OpenBMB repositories were
  reachable in the same session).
- **URL actually studied:** https://github.com/thunlp/MatPlotAgent (the public repository
  for the same paper, arXiv 2402.11453).
- **Commit studied:** `9cafa262aae7bdf85fccf6d02b2153fb772bc376`
- **Licence (verified):** **No license file (all rights reserved).** Checked: repository
  root (README.md, agents/, assets/, benchmark_data/, evaluation/, models/, three .py
  files, requirements.txt), recursive search for LICENSE / COPYING (none), README (no
  licence statement; only a citation request).
- **Role:** IDEAS ONLY.

**What we learned**

- A model-agnostic three-stage agent for scientific plotting:
  1. **Query expansion** - an LLM rewrites the user's short request into explicit,
     step-by-step plotting instructions (what to load, which plot type, every label and
     styling requirement).
  2. **Code generation with iterative self-debugging** - a code LLM writes plotting code;
     it is executed; if it errors (or runs but produces no image file) the error message
     is fed back and the model retries, up to a small fixed number of attempts.
  3. **Visual feedback** - the rendered image is given to a multimodal model together
     with the original request; the model lists concrete discrepancies and improvements in
     words; the code model then revises the code using that critique and the figure is
     rendered again.
- The code model and the vision model can be different models; the vision model only
  critiques, it does not write the code.
- The accompanying benchmark (100 human-verified cases) scores outputs by asking a
  vision model to compare the generated figure with a ground-truth figure and emit a
  numeric final score - i.e. vision-model-as-judge. A non-executable result scores zero.

**How TorusPlot uses the idea (written fresh)**

- TorusPlot's AI layer implements its own bounded loop: **propose FigureSpec -> validate
  against schema -> render -> local vision model critiques the PNG against the request ->
  revise spec**, with a hard cap on rounds and a "keep the best so far" rule.
- Key difference by design: our models emit a *declarative FigureSpec patch*, not Python
  code, so nothing model-written is ever executed. Schema-validation errors play the role
  of the traceback in the self-debugging step.
- The critique prompt, the scoring rubric and all orchestration code are written by us.

**Code copied: none.**

---

## 7. PlotLLM

- **URL:** https://github.com/SvenPfiffner/PlotLLM
- **Commit studied:** `deccf3eb15285b225ee9fad91ce1b01f872f7a8e`
- **Licence (verified):** **MIT**. File read: `LICENSE` ("MIT License", Copyright (c) 2025
  Sven Pfiffner); README licence section agrees.
- **Role:** IDEAS ONLY (permissive, but we take no code; it is a small proof of concept
  whose central mechanism we deliberately do not use).

**What we learned**

- *Architecture.* Two small files: a Streamlit page and a chat helper. The helper talks to
  a locally running Ollama service through the `ollama` Python client's chat call,
  sending a system prompt plus the conversation history, and a small service check lets
  the page tell the user when the Ollama service is not running.
- *UI pattern.* A first-run disclaimer gate stored in session state; a sidebar settings
  expander with a model picker (several sizes of one reasoning-model family) that also
  shows the download size of the selected model; a chat input; conversation history kept
  in session state; the generated code shown in a code block with a copy button; the
  resulting plot shown beside it.
- *Response handling.* The reasoning model's "thinking" section is separated from the
  final answer, and the fenced code block is extracted from the answer.
- *Weakness (stated by its own README).* The generated Python is executed directly in the
  app process, so a crafted prompt can run arbitrary code; the author warns against
  deploying it. Small models (below roughly 14B parameters) often produce broken code.
  There is no data upload: plots are made from data the model invents or the prompt states.

**How TorusPlot uses the idea (written fresh)**

- We keep the *deployment pattern*: Streamlit front end, local Ollama service, model
  picker with size hints, clear "service not running / model not pulled" messages, chat
  history in session state.
- We reject the execution model: TorusPlot never executes model output. The model returns
  JSON constrained to the FigureSpec / analysis-request schema (Ollama structured output),
  which is validated and then rendered by our own renderer. Reasoning text is stripped
  before parsing.
- The Ollama call goes through a thin backend interface of our own (HTTP to the local
  Ollama REST API) so other local or hosted back ends can be swapped in.

**Code copied: none.**

---

## 8. tidyplots

- **URL:** https://github.com/jbengler/tidyplots
- **Commit studied:** `78974535e3739d76cf4ebf158d8688423da7d8eb`
- **Licence (verified):** **MIT**. Files read: `LICENSE.md` ("MIT License", Copyright (c)
  2024 Jan Broder Engler), `LICENSE` (CRAN stub: year 2024, holder Jan Broder Engler) and
  `DESCRIPTION` (`License: MIT + file LICENSE`).
- **Role:** API DESIGN REFERENCE (an R package; TorusPlot is Python, so no code could be
  shared even in principle - we looked only at the exported function names in `NAMESPACE`).

**What we learned**

- *Grammar.* Start a plot from a data frame with x / y / colour mappings, then pipe it
  through small single-purpose verbs. Verb families and names:
  - **Entry / output:** `tidyplot`, `save_plot`, `view_plot`, `split_plot`, `flip_plot`.
  - **add_ (data layers by statistic x geometry):**
    `add_data_points`, `add_data_points_jitter`, `add_data_points_beeswarm`;
    `add_mean_bar`, `add_mean_dash`, `add_mean_dot`, `add_mean_line`, `add_mean_area`,
    `add_mean_value`; the same six shapes for `add_median_*`, `add_sum_*` and
    `add_count_*`; `add_sem_errorbar`, `add_sd_errorbar`, `add_ci95_errorbar`,
    `add_range_errorbar`; `add_sem_ribbon`, `add_sd_ribbon`, `add_ci95_ribbon`,
    `add_range_ribbon`; `add_boxplot`, `add_violin`, `add_histogram`, `add_heatmap`,
    `add_line`, `add_area`, `add_curve_fit`, `add_ellipse`; `add_barstack_absolute`,
    `add_barstack_relative`, `add_areastack_absolute`, `add_areastack_relative`;
    `add_pie`, `add_donut`.
  - **add_ (statistics):** `add_test_pvalue`, `add_test_asterisks`,
    `add_test_pvalue_manual`, `add_test_asterisks_manual`.
  - **add_ (annotation):** `add_title`, `add_caption`, `add_reference_lines`,
    `add_data_labels`, `add_data_labels_repel`, `add_annotation_text`,
    `add_annotation_rectangle`, `add_annotation_line`, and the generic `add`.
  - **remove_:** `remove_legend`, `remove_legend_title`, `remove_title`, `remove_caption`,
    `remove_padding`, `remove_clipping`, `remove_x_axis`, `remove_x_axis_title`,
    `remove_x_axis_labels`, `remove_x_axis_ticks`, `remove_x_axis_line`, and the same five
    for the y axis.
  - **adjust_:** `adjust_colors`, `adjust_font`, `adjust_size`, `adjust_padding`,
    `adjust_title`, `adjust_caption`, `adjust_legend_title`, `adjust_legend_position`,
    `adjust_x_axis`, `adjust_x_axis_title`, `adjust_y_axis`, `adjust_y_axis_title`,
    `adjust_theme_details`.
  - **Label / level helpers:** `rename_`, `reorder_`, `sort_`, `reverse_` each for
    `x_axis_labels`, `y_axis_labels`, `color_labels` (and the matching `_levels` forms).
  - **Themes:** `theme_tidyplot`, `theme_ggplot2`, `theme_minimal_x`, `theme_minimal_y`,
    `theme_minimal_xy`.
  - **Data subsetting inside a layer:** `all_rows`, `filter_rows`, `max_rows`, `min_rows`,
    `first_rows`, `last_rows`, `sample_rows`.
  - **Misc:** `normalize_x`, `normalize_y`, `normalize_color`, `format_p_value`,
    `new_color_scheme`, `tidyplots_options`, plus families of named discrete, continuous
    and diverging colour schemes.
- *Design lessons.* (a) Layer names are composed as statistic + geometry, so the name says
  exactly what is drawn and what is computed. (b) "Manual" test verbs accept externally
  computed P values, keeping the statistics engine and the drawing separate. (c) Absolute
  vs relative stacking is a verb-level choice. (d) Plot size is set in physical units
  (width, height, unit) rather than pixels. (e) remove_ and adjust_ are symmetrical and
  axis-specific, so discoverability by auto-complete is excellent.

**How TorusPlot uses the idea (written fresh)**

- TorusPlot's optional Python convenience API and the *intent vocabulary* understood by
  the AI layer follow the same add / adjust / remove mental model (for example: add a
  mean bar, add SEM error bars, add individual points, add a test bracket, remove the
  legend, adjust the y-axis). Each verb is a pure function that returns a modified
  FigureSpec. The implementation is original Python operating on our own schema.
- We use the statistic + geometry naming idea for FigureSpec layer kinds, and we mirror
  the "manual / precomputed P value" split: TorusPlot's statistics engine computes, the
  bracket layer only draws.

**Code copied: none.** (Only public function names are listed above, as an API reference
under the MIT licence.)

---

## 9. statannotations

- **URL:** https://github.com/trevismd/statannotations
- **Commit studied:** `3f020ae631ca88a091b6ee3e9a9fd32158920879` (repository version file
  says 0.7.2). The copy TorusPlot actually runs is the PyPI wheel **0.7.2** installed at
  `.venv/lib/python3.11/site-packages/statannotations` (metadata: `License: MIT License`,
  requires numpy, seaborn >= 0.9.0, matplotlib, pandas, scipy); its public API was checked
  there against seaborn 0.13.2.
- **Licence (verified):** **MIT**. File read: `LICENSE` - two MIT notices, Copyright (c)
  2019 Marc Weber (original statannot code) and Copyright (c) 2020 Florian Charlier;
  `setup.py` says `license="MIT License"`.
- **Role:** RUNTIME DEPENDENCY (installed from PyPI, called through its public API; its
  MIT notice is carried in `THIRD_PARTY_NOTICES.md`).

**What we learned** (API call names only)

- *Three modes of one class, `Annotator`:* (1) custom text per pair
  (`set_custom_annotations`), (2) supply your own P values and let it format them
  (`set_pvalues`), (3) let it run a test (`apply_test`). TorusPlot uses modes 1 and 2
  only, because our own engine computes the statistics.
- *Call sequence for precomputed results:*
  1. Draw a seaborn categorical plot on a Matplotlib Axes.
  2. Construct `Annotator(ax, pairs, plot=..., data=..., x=..., y=..., hue=..., order=...,
     hue_order=...)`, passing **exactly the same data / x / y / hue / order arguments**
     that were given to the seaborn call. `engine` only accepts "seaborn".
  3. Optionally `configure(...)`. Accepted keys in 0.7.2: `alpha`, `color`,
     `comparisons_correction`, `line_height`, `line_offset`, `line_offset_to_group`,
     `line_width`, `loc` ("inside" or "outside"), `pvalue_format`, `show_test_name`,
     `test`, `test_short_name`, `text_offset`, `use_fixed_offset`, `verbose`,
     `hide_non_significant`; plus P-value format keys `text_format` ("star", "simple",
     "full"), `pvalue_thresholds`, `pvalue_format_string`, `simple_format_string`,
     `fontsize`, `correction_format`, `p_capitalized`, `p_separators`.
  4. Either `set_pvalues(pvalues)` (one number per pair, same order as `pairs`; optional
     `num_comparisons`) or `set_custom_annotations(list_of_strings)` (one string per pair).
  5. `annotate()` draws the brackets and returns the Axes and the annotation list.
  Shortcuts: `set_pvalues_and_annotate`, `annotate_custom_annotations`,
  `apply_and_annotate`. Helpers: `get_annotations_text`, `get_configuration`,
  `reset_configuration`, `new_plot` (re-target an annotator, e.g. per facet),
  `Annotator.get_empty_annotator`, `plot_and_annotate`, `plot_and_annotate_facets`.
- *Pair format.* Without hue: a list of `(category_a, category_b)` tuples. With hue: a
  list of `((category, hue_level), (category, hue_level))` tuples.
- *Constraints.*
  - Needs a **seaborn-style categorical plot**; supported `plot` kinds in 0.7.2 are
    exactly `barplot`, `boxplot`, `stripplot`, `swarmplot`, `violinplot`. No line,
    scatter, point-plot, histogram or plain-Matplotlib bar support.
  - The number of P values / custom strings must equal the number of pairs, else it raises.
  - In `set_pvalues` mode no `test` may be configured (it raises if one is set); a
    `test_short_name` may be given purely for labelling. In custom-text mode the test name
    is switched off automatically.
  - If `comparisons_correction` is configured together with `set_pvalues`, the library
    will *adjust the supplied P values again* - so pass already-adjusted values and leave
    correction unset (None).
  - `loc="outside"` draws above the axes area; "inside" expands the value-axis limit to
    make room. Both vertical and horizontal orientations are handled. `verbose` prints a
    legend of star thresholds to stdout unless disabled.
  - 0.7.x supports pandas 2 and seaborn 0.12+ (per its README / changelog); it resolved
    with seaborn 0.13.2 in our environment.
  - Built-in test names (unused by us): t-test_ind, t-test_welch, t-test_paired,
    Mann-Whitney (and one-sided variants), Levene, Wilcoxon, Kruskal, Brunner-Munzel.

**How TorusPlot uses the idea (written fresh)**

- TorusPlot computes all statistics and multiple-comparison adjustments in its own engine,
  then - for figure kinds that are seaborn categorical plots - hands the already-adjusted
  P values (or fully formatted strings such as an exact "P = 0.012") to statannotations
  via `Annotator(...)` then `configure(...)` then `set_pvalues(...)` /
  `set_custom_annotations(...)` then `annotate()`, with `comparisons_correction` unset
  and `verbose` off.
- For every figure kind statannotations cannot handle (XY, grouped lines, plain Matplotlib
  bars, survival curves) TorusPlot's own bracket module draws the annotations; the two
  paths share one FigureSpec "comparison annotation" description.
- The adapter is our code; the library is used unmodified as an installed dependency.

**Code copied: none.** (It is imported as a dependency; no source was vendored or adapted.)

---

## 10. Graphite

- **URL:** https://github.com/GraphiteEditor/Graphite
- **Commit studied:** `1652698c5b50a7cc6a98fb2c4ad4b19765b36edd` (`master` HEAD from
  `git ls-remote`; the repository was *not* cloned - only `LICENSE-APACHE`, `LICENSE-MIT`,
  `README.md` and one source file describing SVG import were fetched as raw files for
  reading).
- **Licence (verified):** **MIT OR Apache-2.0** (dual, at the user's option). Files read:
  `LICENSE-APACHE` (Apache License 2.0 text), `LICENSE-MIT` (MIT text, "Copyright (c)
  Graphite contributors"), and the README licence section, which states the dual licence
  "unless otherwise noted within the repository". (The brief expected Apache-2.0; that is
  one of the two options.)
- **Role:** EXPORT TARGET ONLY. TorusPlot writes SVG files that a user may open in
  Graphite; there is no code, linking or bundling relationship.

**What we learned** (how its SVG import behaves at this commit; the editor is alpha
software and this may change)

- SVG is parsed with a standard Rust SVG library using default options, then walked as a
  tree of four node kinds: group, path, image, text.
- **Groups** (`<g>`) become layer folders, preserving nesting and order - so a
  well-grouped SVG arrives as a well-organised layer tree.
- **Paths** (and basic shapes, which the parser normalises to paths) become editable
  vector layers with their fill (solid colour, linear or radial gradient), their stroke
  (solid colour only; width, line cap, line join, miter limit, dash pattern and dash
  offset are read) and their transforms.
- **Text** (`<text>`) becomes a live, editable text layer containing the original string -
  but the importer assigns the editor's default font family and style, default
  typesetting, and black fill. The text *content* stays editable; font, size and colour
  are not carried over.
- **Embedded raster images** (`<image>`) are skipped on import.
- The import code we read does not transfer group-level opacity, clip paths or masks.
- A file-open import keeps the SVG's viewBox coordinates; a paste import centres on the
  content's bounding box.

**How TorusPlot uses the idea (written fresh)**

- TorusPlot's "editable SVG" export profile is tuned for vector editors in general and
  checked against the behaviour above:
  - keep text as real `<text>` elements (Matplotlib `svg.fonttype = "none"`) so labels
    stay editable, and offer an alternative "text as outlines" profile for exact
    appearance;
  - give every logical element its own named `<g>` (axes, each series, error bars,
    legend, annotations, brackets) via Matplotlib group ids, so the layer tree is
    meaningful;
  - avoid embedded rasters in this profile (no rasterised artists; heatmaps drawn as
    vector cells or the user is warned);
  - avoid relying on clip paths, masks or group opacity for correctness (pre-clip data,
    use per-element opacity);
  - plain solid fills and strokes; document that font and size may need re-applying after
    import into this particular editor.
- These are export settings of our own renderer; nothing from Graphite is used.

**Code copied: none.**

---

## Summary

| # | Repository | Verified licence (file read) | Role | Code copied |
|---|---|---|---|---|
| 1 | KDE/labplot @ `4c336dec` | GPL-2.0-or-later (`LICENSES/`, SPDX headers, appdata) | IDEAS ONLY | none |
| 2 | mickaphd/Helix @ `aa63df1e` | GPL-3.0-or-later (`LICENSE` + README) | IDEAS ONLY | none |
| 3 | raphaelvallat/pingouin @ `1ffb9232` | GPL-3.0-only (`LICENSE`, `pyproject.toml`) | IDEAS ONLY (never installed) | none |
| 4 | ywatanabe1989/figrecipe @ `162f6237` | AGPL-3.0-only (`LICENSE`, `pyproject.toml`) | IDEAS ONLY | none |
| 5 | Trae1ounG/paper-plot-skills @ `cde5e848` | No license file (all rights reserved) | IDEAS ONLY | none |
| 6 | thunlp/MatPlotAgent @ `9cafa262` (OpenBMB/MatPlotAgent unreachable) | No license file (all rights reserved) | IDEAS ONLY | none |
| 7 | SvenPfiffner/PlotLLM @ `deccf3eb` | MIT (`LICENSE`) | IDEAS ONLY | none |
| 8 | jbengler/tidyplots @ `78974535` | MIT (`LICENSE.md`, `DESCRIPTION`) | API DESIGN REFERENCE | none |
| 9 | trevismd/statannotations @ `3f020ae6` (PyPI 0.7.2 installed) | MIT (`LICENSE`) | RUNTIME DEPENDENCY | none |
| 10 | GraphiteEditor/Graphite @ `1652698c` (not cloned) | MIT OR Apache-2.0 (`LICENSE-MIT`, `LICENSE-APACHE`) | EXPORT TARGET | none |

Candidate local-model licences for the Ollama backend are recorded separately in
`docs/_model_license_research.md`.

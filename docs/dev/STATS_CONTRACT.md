# Statistics engine: developer contract

This is the shared contract for everyone writing code in `torusplot/stats/`.
Read it fully before writing anything.

## 0. Ground rules (legal + hygiene)

* TorusPlot is **proprietary** (Torus Washington LLC). All code must be written fresh.
* **Never** open, read, import or copy from `_reference/` and never install or import
  `pingouin` (GPL) or any package not already in `.venv`. Do not `pip install` anything.
  Implement from formulas in the primary literature and standard textbooks; cite the
  primary paper in the module/function docstring.
* Allowed libraries: `numpy`, `pandas`, `scipy`, `statsmodels`, `scikit_posthocs`
  (import name), `lifelines`. Tests may also use `pytest`, `hypothesis`.
* Never write the names of other commercial statistics products, or journal/publisher
  names as style names, anywhere: code, comments, tests, docs (the banned list is enforced by
  `tests/test_docs.py`). Say "other statistics packages".
* Do **not** run any `git` command. Do not edit files you do not own (ownership is in
  your task brief). Other people are editing other files in this repo right now.
* Python 3.11 syntax. Cross-platform (Windows + Linux): no shell-outs, use `pathlib`,
  always `encoding="utf-8"` when opening text files.
* Run things with the project venv: `.venv/bin/python -m pytest tests/stats/test_x.py -q`
  and `.venv/bin/ruff check torusplot/stats tests/stats` (must pass; line length 100,
  rule E501 is ignored, imports sorted).

## 1. What already exists (read these files first)

| File | What it gives you |
|---|---|
| `torusplot/stats/result.py` | `StatResult`, `AssumptionCheck`, `EffectSize`, `Parameter`, `Comparison`, `format_p`, `p_to_stars` |
| `torusplot/stats/_utils.py` | `StatsError`, `clean`, `clean_groups`, `clean_paired`, `require_n`, `is_constant`, `fmt`, `p_phrase`, `sig_phrase`, `ci_phrase`, `check_alpha`, `check_alternative`, `alternative_label`, `jsonable` |
| `torusplot/stats/effect_sizes.py` | Cohen's d / Hedges' g (independent, paired, one-sample, from t), eta², partial eta², omega², partial omega², rank-biserial r (independent + matched pairs), epsilon² (KW), Kendall's W, all with CIs; magnitude labels |
| `torusplot/stats/multiplicity.py` | `adjust_pvalues(p, method)` with `none, bonferroni, sidak, holm, holm-sidak, fdr_bh`; `METHOD_LABELS` |
| `torusplot/stats/registry.py` | `register(...)` decorator, `ParamSpec`, `run_analysis(table, key, **params)`, `CATEGORIES` |
| `torusplot/data/tables.py` | The five table types (API summarised in section 4) |

Use the shared helpers; do not re-implement effect sizes or p-adjustments. If a shared
helper is missing something you need, add a *private* helper in your own module and
mention it in your final report (do not edit the shared files).

## 2. Two layers per analysis

1. **Array-level function** (pure numbers in, `StatResult` out), e.g.

   ```python
   def ttest_unpaired(a, b, *, names=("A", "B"), alpha=0.05, alternative="two-sided",
                      ci_level=0.95) -> StatResult: ...
   ```
   Keyword-only options after the data. No table objects here.

2. **Table-level adapter** registered for the UI/CLI/MCP:

   ```python
   @register(
       key="ttest_unpaired", label="Unpaired t test", category="Compare two groups",
       tables=("column",),
       params=(
           ParamSpec("group_a", "First group", kind="group", default=None, optional=True,
                     help="Which column is group A. Leave empty to use the first column."),
           ParamSpec("alpha", "Significance level (alpha)", kind="float", default=0.05,
                     help="The P value threshold for calling a result significant. 0.05 is the usual choice."),
           ...
       ),
       description="Compares the means of two independent groups, assuming both are roughly normal with similar spread.",
       citation="Student (1908) Biometrika 6:1-25",
   )
   def _run_ttest_unpaired(table, **params) -> StatResult: ...
   ```
   * `run_analysis` fills `result.analysis`, `result.table`, `result.params`, adds the
     table's exclusion notes, converts numpy values in `extra`/`descriptives`/
     `anova_table` to plain Python, and calls `result.finalize()` (which assigns
     `Comparison.id`s and `significant` flags). Adapters must accept every declared
     param as a keyword (use `**params` or explicit keywords) and must work when called
     with all defaults.
   * Every `ParamSpec.help` is a **beginner-friendly tooltip** (one or two plain
     sentences, no jargon without explanation). `description` is one or two sentences a
     first-year grad student understands.
   * `category` must be one of `registry.CATEGORIES`.

Importing your module must register its analyses (decorators at module level).

## 3. How to fill a `StatResult`

| Field | Requirement |
|---|---|
| `test_name` | Full conventional name ("Welch's t test", "Kruskal-Wallis test"). |
| `why_chosen` | 1-2 plain-English sentences: when this test is the right one for the design. |
| `assumptions` | Every assumption the test relies on, **actually checked** where checkable (normality per group/residuals via Shapiro-Wilk, equal variances via Brown-Forsythe/Levene, sphericity, expected counts...). `status`: `pass` (no evidence against), `warn` (borderline, or cannot be checked e.g. n < 3), `fail` (violated: say which alternative test to use), `info` (design assumptions that cannot be tested from data, e.g. independence). Keep the `detail` text short and practical. For normality/variance checks use alpha 0.05 and compute them inline with scipy (do not depend on another person's in-progress module). |
| `statistic_name`, `statistic`, `df`, `df2`, `p_value` | The main test. F tests: `df` = numerator, `df2` = denominator. |
| `alpha`, `alternative` | As run. |
| `effect_sizes` | At least one, primary first, each with a 95% CI wherever a sound method exists. |
| `parameters` | Estimates with SE/CI: mean difference, regression coefficients, EC50, medians, odds ratio... |
| `n`, `descriptives` | `n` = `{group: count}`. `descriptives` = list of dict rows (`group, n, mean, sd, sem, median, ...`). |
| `anova_table` | ANOVA-type analyses: list of dict rows with keys `source, ss, df, ms, f, p` (+ `partial_eta_sq` etc.). |
| `comparisons`, `comparison_family`, `correction` | Post-hoc/multiple comparisons. `estimate` = mean(A) - mean(B) (or mean-rank difference), `p_value` = unadjusted where defined, `p_adjusted` = adjusted (for Tukey/Dunnett/Games-Howell the method's own p goes in `p_adjusted` and the adjusted CI in `ci_low/ci_high`). `comparison_family` describes the family in words ("all 6 pairwise comparisons among 4 groups"). Give any `id` placeholder such as `""`; `finalize()` overwrites it. For factorial designs also fill `cell_a` / `cell_b` (`{factor: level}`) and `label`. |
| `warnings` | Anything the user must know: tiny n, ties with exact test unavailable, a constant group, unequal n with Type-specific SS, convergence trouble, extrapolation. |
| `data_notes` | Notes from `clean()` about dropped missing values, replicate handling. |
| `interpretation` | 2-4 plain-English sentences a non-statistician can paste into a lab notebook: what was compared, the result with numbers, what it means, and the effect size. Never claim "proves"; say "evidence that". Use `sig_phrase`, `p_phrase`, `fmt`. |
| `methods_text` | One sentence for a methods section, e.g. "Groups were compared with an unpaired two-tailed Welch's t test (alpha = 0.05)." |
| `citations` | Primary method references (author, year, journal). |
| `extra` | JSON-able payload for plots (see section 5). Lists of floats, no numpy arrays needed (run_analysis converts). |

## 4. Table API (from `torusplot/data/tables.py`)

* `ColumnTable`: `.group_names`, `.groups(dropna=True) -> {name: 1-D array}` (user
  exclusions already removed), `.paired_frame(columns=None) -> DataFrame` (rows =
  subjects with a value in every selected column), `.to_long()` -> columns
  `group, subject, value`. Build one in tests with
  `ColumnTable.from_dict("name", {"A": [...], "B": [...]})`.
* `XYTable`: `.x`, `.x_name`, `.series_names`, `.series(name) -> DataFrame` of replicate
  columns, `.xy_points(name) -> (x, y)` flattened non-missing points,
  `.xy_means(name) -> DataFrame[x, mean, sd, sem, n]`, `.to_long()` -> `x, series,
  replicate, y`. Build with `XYTable.from_arrays("name", x, {"Series": 1-D or 2-D array})`.
* `GroupedTable`: `.row_levels`, `.column_levels`, `.row_label`, `.column_label`,
  `.cell(row, column) -> array`, `.to_long()` -> `row, column, replicate, value`
  (`replicate` = sub-column position, 1-based; in repeated-measures layouts it
  identifies the subject within a column group). Build with
  `GroupedTable.from_cells("name", {row: {column: [replicates]}}, row_label=, column_label=)`.
* `MultipleVariablesTable`: `.data`, `.numeric_columns`, `.categorical_columns`,
  `.frame(columns=None, dropna=True) -> DataFrame` (complete cases). Roles are chosen
  via `ParamSpec(kind="column")` params.
* `ContingencyTable`: `.counts -> DataFrame` (index = row labels), `.row_levels`,
  `.column_levels`, `.row_label`, `.column_label`. Build with
  `ContingencyTable.from_counts("name", [[a, b], [c, d]], row_names, column_names)`.

`ParamSpec(kind="group")` values are column/series names of the table; `None` means
"use the default" (document what the default is in `help`). Validate names and raise
`StatsError` with a helpful message when a name does not exist.

## 5. `extra` conventions (the plot layer reads these; keep the exact keys)

* Curve fits and regressions:
  `extra["curve"] = {"x": [...], "y": [...], "ci_low": [...], "ci_high": [...],
  "pred_low": [...], "pred_high": [...]}` (about 200 points across the data's x range;
  bands optional but include them when you can; `x` is in the *original data units*),
  `extra["residuals"] = {"x": [...], "fitted": [...], "residual": [...]}`,
  `extra["equation"] = "Y = Bottom + (Top - Bottom) / (1 + 10^((LogEC50 - X) * HillSlope))"`,
  `extra["model"] = "<model key>"`, `extra["series"] = "<series name>"`,
  `extra["x_is_log"] = bool` (True when the model expects log10(x) as X).
* Normality: `extra["qq"] = {group: {"theoretical": [...], "ordered": [...],
  "slope": float, "intercept": float}}`.
* Survival: `extra["curves"] = {group: {"time": [...], "survival": [...],
  "ci_low": [...], "ci_high": [...], "at_risk": [...], "censor_time": [...],
  "censor_survival": [...]}}` (step coordinates starting at time 0, survival 1).
* Correlation matrix: `extra["matrix"] = {"labels": [...], "r": [[...]], "p": [[...]], "n": [[...]]}`.
* Outlier detection: `extra["flagged"] = [{"group": name, "row": zero_based_row_in_table,
  "value": float}, ...]` so the app can offer explicit exclusion; never remove anything yourself.
* Factorial ANOVA: `extra["cell_means"] = [{"<factor>": level, ..., "mean":, "sd":, "sem":, "n":}, ...]`,
  `extra["factors"] = [names]`.
* Anything else useful for tables in the UI may go in `extra` under clear keys.

## 6. Robustness policy (tested with hypothesis)

* Impossible input (too few values, a missing group, wrong shapes, all values missing)
  -> raise `StatsError` with a message that says what to fix. No other exception types
  may escape for bad *data* (bad programmer usage may raise `TypeError`/`ValueError`).
* Degenerate-but-computable input (n = 2, ties, zeros, a constant group, all values
  identical, unequal n, NaN among values) -> return a result with explicit `warnings`
  and `None`/NaN only where the quantity is genuinely undefined; never a silently wrong
  number, never a crash. NaN are dropped by `clean()` and reported in `data_notes`.
* Do not let numpy/scipy `RuntimeWarning`s leak for expected degenerate cases: guard
  them and turn them into `result.warnings` text.
* Nothing is ever removed or transformed automatically (outliers are only flagged).

## 7. Validation requirements for your tests (non-negotiable)

For **every** analysis you own, `tests/stats/` must contain:

1. **Reference-value tests**
   * NIST Statistical Reference Datasets where they exist (ANOVA, nonlinear and linear
     regression): https://www.itl.nist.gov/div898/strd/ (US government work; cite NIST).
     Save data files under `tests/fixtures/nist/` with an attribution header. If the
     download is blocked, hand-enter the data and certified values.
   * **Published worked examples** with reported results (textbooks, papers, official
     method papers). Type the numbers in yourself. Only use an example if you are
     confident of both the data and the published results; your code must reproduce the
     published values to the published precision. If it does not, investigate; never
     tweak an expected value just to make a test pass, and never invent a "published"
     example. If you cannot verify a source, drop it and rely on (2).
     Never copy datasets from copyleft or unlicensed repositories.
   * Record every source in **your own** file `tests/fixtures/sources/<your_area>.md`
     (dataset, full citation, which values are checked, how you obtained them).
2. **Cross-implementation checks** inside the allowed stack: your implementation vs
   scipy vs statsmodels (vs scikit_posthocs / lifelines where relevant) must agree to
   rel 1e-6 for statistics and 1e-4 for p values (looser, documented tolerance for
   iterative fits). Where you wrap a library call, cross-check against an independent
   from-first-principles computation written in the test.
3. **Property-based edge cases** with `hypothesis` (keep `max_examples` modest, e.g.
   30-60, `deadline=None`): n = 2, ties, zeros, NaNs, unequal group sizes, a constant
   group, all-identical data -> `StatsError` or a result with warnings; p values in
   [0, 1] or None; no crash; invariances where they exist (group order, location/scale).
4. Registry tests: each registered analysis runs through `run_analysis` on a small
   table with default params, returns a finalized result with `interpretation`,
   `why_chosen`, at least one assumption entry, and JSON-serialises
   (`result.model_dump_json()`).

Coverage of each module you own should be >= 90%
(`.venv/bin/python -m pytest tests/stats/test_x.py --cov=torusplot.stats.<module> --cov-report=term-missing`).
Tests must not need the network at run time and should run in well under a minute per file.

## 8. Final report format

When done, report: files written; analyses registered (keys); for each validation source
whether it is a NIST / published / cross-check; any published example you tried and
dropped; any deviation from this contract; anything you believe is still weak or
unverified. Be candid: an honest "this CI method is approximate" is far more useful
than a claim of perfection.

# Validation sources: factorial designs and repeated measures

Modules: `torusplot/stats/factorial.py`, `torusplot/stats/repeated.py`
Tests: `tests/stats/test_factorial.py`, `tests/stats/test_repeated.py`

No NIST Statistical Reference Dataset exists for these designs (the StRD ANOVA
collection is one-way only), so validation rests on published worked examples,
cross-implementation checks and from-first-principles computations.

## 1. Published worked examples (typed in by hand; no data files)

All three examples below were reproduced by the code to the published precision without
any adjustment of expected values. None of the books was at hand while writing the
tests: the data and the reported results were typed from the task brief and from memory
of the sources, and are internally consistent with an independent computation (see the
notes per example). **A maintainer with the physical sources should confirm the page /
table references before these are quoted anywhere else.**

### 1.1 Battery life, 3 x 3 factorial with n = 4 (two-way ANOVA)

* Source: Montgomery DC. *Design and Analysis of Experiments*. Wiley. Chapter 5
  ("Introduction to Factorial Designs"), the battery design experiment (Example 5.1):
  life in hours of batteries for 3 plate materials x 3 temperatures (15, 70, 125 deg F).
* Data: 36 values, in `BATTERY` in `tests/stats/test_factorial.py`.
* Checked (test `test_published_montgomery_battery_life`):
  SS Material = 10,683.72; SS Temperature = 39,118.72; SS Interaction = 9,613.78;
  SS Error = 18,230.75; SS Total = 77,646.97; df = 2, 2, 4, 27, 35; MS Error = 675.21;
  F = 7.91, 28.97, 3.56. Tolerance: half a unit in the last printed digit.
* Also used (same data) for: equality of Type II / Type III / sequential SS in a
  balanced design, classical balanced formulas, Tukey comparisons against
  `statsmodels.stats.multicomp.pairwise_tukeyhsd`, and the Grouped-table adapter.

### 1.2 Tooth growth in guinea pigs, 2 x 3 factorial with n = 10 (two-way ANOVA)

* Primary source of the measurements: Crampton EW (1947) The growth of the odontoblasts
  of the incisor tooth as a criterion of the vitamin C intake of the guinea pig.
  *J Nutr* 33:491-504. Also analysed in Bliss CI (1952) *The Statistics of Bioassay*,
  Academic Press.
* Data: 60 values (length by supplement VC / OJ and dose 0.5 / 1 / 2 mg per day), in
  `TOOTH` in `tests/stats/test_factorial.py`. These are the values widely distributed
  as the `ToothGrowth` example dataset of the R statistical environment; they were
  typed from memory, not copied from any file, and the typing is verified in the test
  against the well-known cell means (7.98, 16.77, 26.14, 13.23, 22.70, 26.06) and grand
  mean (18.8133).
* Checked (test `test_published_tooth_growth`): the standard two-way table for
  `len ~ supp * dose` (dose as a factor): F(supp) = 15.572, F(dose) = 92.000,
  F(supp x dose) = 4.107 with 1, 2, 2 and 54 df; SS = 205.35, 2426.43, 108.32,
  residual 712.11; interaction P = 0.02186.
* Licence note for the maintainer: the numbers are factual measurements from the 1947
  paper, but the convenient copy most people know ships with GPL-licensed software.
  The task brief explicitly allowed this example; if the project prefers to avoid any
  doubt under the "no datasets from copyleft repositories" rule, delete
  `test_published_tooth_growth` and the `TOOTH` constant (two other tests reuse the
  constant only as convenient balanced data: `test_missing_values_level_order_and_labels`
  and the numeric-factor case in the registry test). Example 1.1 covers the same analysis.

### 1.3 "Bushtucker trial", 8 subjects x 4 conditions (one-way repeated measures)

* Source: Field A. *Discovering Statistics Using IBM SPSS Statistics*. SAGE. Chapter on
  repeated-measures designs: time to retch for 8 celebrities eating 4 foods (stick
  insect, kangaroo testicle, fish eye, witchetty grub).
* Data: 32 values, in `FIELD` in `tests/stats/test_repeated.py`.
* Checked (test `test_published_field_bushtucker_example`):
  F(3, 21) = 3.79, p = .026 (sphericity assumed); Mauchly W = .136,
  chi-square(5) = 11.41, p = .047; Greenhouse-Geisser epsilon = .533 (p = .063);
  Huynh-Feldt epsilon = .666 (p = .048). Tolerance: half a unit in the last digit.
* Investigation recorded here because it changed the implementation: the plain
  chi-square tail probability of 11.41 on 5 df is .044, not the published .047. The
  published value is reproduced exactly when the second-order term of Box's (1949)
  expansion of the distribution of the sphericity criterion is included (Anderson TW,
  *An Introduction to Multivariate Statistical Analysis*, section 10.7), which is what
  the major packages print. `repeated.py` therefore reports the second-order P value
  as the Mauchly P value and keeps the first-order value in
  `extra["sphericity"]["p_first_order"]` (asserted to be .044 in the same test). A
  simulation with the fixed seed in `test_mauchly_p_is_roughly_uniform_under_sphericity`
  checks the calibration of the reported P value.

### Published examples tried and dropped

None were dropped after being tried. No published worked example is used for ANCOVA,
the mixed (split-plot) ANOVA or the two-way repeated-measures ANOVA: I could not recall
any dataset *and* its printed results with enough certainty to type them in, so those
analyses rely on sections 2 and 3.

## 2. Cross-implementation checks (allowed stack only)

| Analysis | Checked against | Quantities | Tolerance |
|---|---|---|---|
| Two- and three-way ANOVA, unbalanced, Type II and III | `statsmodels` `ols` + `anova_lm(typ=2/3)` with a hand-written formula on the original columns | SS, df, F, P | rel 1e-8 (P: 1e-6) |
| Balanced two- and three-way ANOVA | `anova_lm(typ=1)` (sequential) | SS identical for Types I, II, III | rel 1e-9 |
| Additive model / one value per cell | `anova_lm` on `y ~ C(A) + C(B)`; `scipy.stats.f` | F, P | rel 1e-8 |
| Tukey / Tukey-Kramer, all cells | `statsmodels` `pairwise_tukeyhsd` (balanced and unbalanced) | differences, adjusted P, simultaneous CI | rel 1e-4 (P abs 1e-7) |
| Tukey P and CI, simple and main effects | `scipy.stats.studentized_range` with the documented k | adjusted P, CI half-width | rel 1e-6 |
| Bonferroni / Sidak / Holm / Holm-Sidak / BH families | `statsmodels` `multipletests` | adjusted P | rel 1e-10 |
| Main-effect comparisons (unweighted marginal means) | `statsmodels` `t_test` on coefficient contrasts of the sum-coded model | estimate, SE, P | rel 1e-9 |
| ANCOVA | `ols("y ~ C(g) + x")`, `anova_lm(typ=2)`, `t_test` | F, P, slope + SE + CI, adjusted-mean differences + SE | rel 1e-8 |
| ANCOVA equal-slopes test | `anova_lm` on `y ~ C(g) * x` | F, P | rel 1e-8 |
| Assumption checks | `scipy.stats.levene(center="median")`, `scipy.stats.shapiro` | statistic, P | rel 1e-10 |
| One-way RM ANOVA | `statsmodels.stats.anova.AnovaRM` | uncorrected F, df, P | rel 1e-6 |
| One-way RM with two levels | `scipy.stats.ttest_rel` | F = t squared, P | rel 1e-10 |
| RM post-hoc tests | `scipy.stats.ttest_rel` (statistic, P, `confidence_interval` at the adjusted level) | t, P, CI | rel 1e-9 |
| Mixed ANOVA, balanced | `statsmodels` `mixedlm` (random intercept per subject, REML), `wald_test_terms` | Wald chi-square / df = F for between, within and interaction; residual and subject variance components | rel 1e-3 (numerical optimiser; 2e-3 for the variance component) |
| Mixed ANOVA, unbalanced | `anova_lm(typ=3)` on each orthonormal contrast score regressed on the sum-coded group | within effect = intercept SS, interaction = group SS, residual | rel 1e-8 |
| Mixed ANOVA, two levels / two groups | `scipy.stats.ttest_ind`, `f_oneway` on change scores and subject sums | F, P | rel 1e-10 |
| Mixed post-hoc tests | `scipy.stats.ttest_ind`, `ttest_rel`; `ols` intercept of difference scores | t, P, estimate, SE | rel 1e-10 |
| Two-way RM ANOVA | `AnovaRM` with two within factors | F, df, P for both main effects and the interaction | rel 1e-6 |

## 3. From-first-principles computations written in the tests

* Classical balanced formulas for two-way and three-way sums of squares (marginal and
  cell means), randomized-block formulas for one value per cell.
* Type II / Type III sums of squares by comparing nested least-squares fits of a
  hand-built sum-to-zero design matrix (`scratch_ss`), also inside a hypothesis test on
  random unbalanced designs.
* Type III main effect as the quadratic form for "all unweighted marginal means are
  equal" computed from the cell means and cell sizes.
* ANCOVA: SS by nested fits; adjusted means `ybar_i - b (xbar_i - xbar)` and their
  standard errors `MSE (1/n_i + (xbar_i - xbar)^2 / Exx)`.
* One-way RM and split-plot sums of squares from the textbook definitions (Winer BJ,
  Brown DR, Michels KM (1991) *Statistical Principles in Experimental Design*, 3rd ed.).
* Unbalanced mixed design: full dummy-variable least-squares model with sum-coded
  group, time, group x time and subject-within-group columns; every Type III SS by
  nested fits (`dummy_model_ss`), also inside a hypothesis test.
* Greenhouse-Geisser epsilon from the elements of the raw k x k covariance matrix (Box's
  formula, no contrasts), Mauchly's W from determinants with an arbitrary orthonormal
  contrast basis, the first- and second-order chi-square approximations, and both
  Huynh-Feldt formulas (original 1976; Lecoutre 1991) written out in the tests.
* Simulations with fixed seeds: compound-symmetric data give epsilon close to 1 and a
  noisy condition gives a small epsilon; under sphericity the Mauchly P value is close
  to uniform (rejection rate at 5% between 3% and 7% over 1500 replicates,
  Kolmogorov-Smirnov check).

## 4. Property-based tests (hypothesis)

F invariant to row order, level relabelling, column order, change of units (shift and
scale) and to adding a constant to each subject; epsilon_GG within [1/q, 1];
Huynh-Feldt (before capping) >= Lecoutre >= Greenhouse-Geisser; all P values in [0, 1]
or `None`; adjusted P >= raw P; confidence intervals contain the estimate; integer
data with ties, zeros, constant cells, perfect fits and missing values never crash and
never leak a numerical warning; empty cells, too few subjects and other impossible
designs raise `StatsError`.

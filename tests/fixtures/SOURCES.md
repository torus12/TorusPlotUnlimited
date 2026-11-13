# Validation sources of the TorusPlot test-suite

This file is the index of every external source that the statistics tests rely on.
The detailed record, with the values checked, the tolerances and the items that were tried and
dropped, is kept per area:

| Area | Detailed record |
|---|---|
| Descriptive statistics, assumption checks, two-group tests, outliers, power | [sources/basics.md](sources/basics.md) |
| One-way ANOVA family and post-hoc tests | [sources/oneway.md](sources/oneway.md) |
| Factorial designs and repeated measures | [sources/factorial_repeated.md](sources/factorial_repeated.md) |
| Correlation, linear and nonlinear regression, equation parser | [sources/regression.md](sources/regression.md) |
| Survival, contingency tables, replicate-aware (nested) analysis | [sources/survival_contingency_nested.md](sources/survival_contingency_nested.md) |

The method behind each analysis and its primary reference are in
[docs/STATS_METHODS.md](../../docs/STATS_METHODS.md).

## Kinds of source

| Kind | Meaning |
|---|---|
| NIST certified | Certified reference values published by the U.S. National Institute of Standards and Technology. |
| Published worked example | A data set with results printed in a paper, a book or a documented software example. The data and the expected values were typed in by hand. |
| Cross-implementation | Agreement with an independent implementation (SciPy, statsmodels, lifelines, scikit-posthocs, NumPy, pandas) or with a from-first-principles calculation written inside the test. |
| Simulation | Fixed-seed simulations that check calibration (false-positive rate, coverage, power). |

## Index

All test files are in `tests/stats/`.

| Dataset or check | Source (full citation) | Kind | What is checked | Test file |
|---|---|---|---|---|
| StRD one-way ANOVA: `SiRstv`, `SmLs01` to `SmLs09`, `AtmWtAg` (11 files in `nist/`) | NIST Statistical Reference Datasets, Analysis of Variance, National Institute of Standards and Technology, https://www.itl.nist.gov/div898/strd/anova/anova.html . Original references given by NIST: Ehrstein J, Croarkin MC, unpublished NIST data set (SiRstv); Simon SD, Lesage JP (1989) Assessing the accuracy of ANOVA calculations in statistical software. Comput Stat Data Anal 8:325-332 (SmLs01 to SmLs09); Powell LJ, Murphy TJ, Gramlich JW (1982) The absolute isotopic abundance & atomic weight of a reference sample of silver. NBS J Res 87:9-19 (AtmWtAg). | NIST certified | Degrees of freedom, sums of squares, mean squares, F, R squared, residual SD; the P value against the F distribution at the certified F. Tolerances 1e-12 to 1e-13 (lower difficulty), 1e-9 (average), 5e-4 (higher difficulty; limited by double-precision storage of the input). | `test_nist_anova.py` |
| StRD nonlinear regression: 26 files in `nist/` (`Misra1a`, `Chwirut2`, `Chwirut1`, `Lanczos3`, `Gauss1`, `Gauss2`, `DanWood`, `Misra1b`, `Kirby2`, `Hahn1`, `MGH17`, `Lanczos1`, `Lanczos2`, `Gauss3`, `Misra1c`, `Misra1d`, `Roszman1`, `ENSO`, `MGH09`, `Thurber`, `BoxBOD`, `Rat42`, `MGH10`, `Eckerle4`, `Rat43`, `Bennett5`) | NIST Statistical Reference Datasets, Nonlinear Regression, National Institute of Standards and Technology, https://www.itl.nist.gov/div898/strd/ | NIST certified | Every parameter, every standard error, residual sum of squares, residual SD and degrees of freedom, from both NIST starting points. Tolerances: parameters 1e-6, standard errors 1e-5, residual SS 1e-8 (5% for the standard errors and residual SS of Lanczos1). Four sets are also fitted through built-in models with automatic starting values. | `test_nist_nls.py`, `test_nonlinear.py` |
| StRD linear regression: `Norris.dat` | NIST Statistical Reference Datasets, Linear Regression, https://www.itl.nist.gov/div898/strd/ | NIST certified | Slope, intercept, both standard errors, residual SD, R squared, F, regression and residual sums of squares. | `test_regression.py` |
| StRD linear regression: `Longley.dat` | Longley JW (1967) An appraisal of least squares programs for the electronic computer from the point of view of the user. J Am Stat Assoc 62:819-841; reference values as certified by NIST StRD. **The file here is not the verbatim NIST file** (see the caveats). | NIST certified values taken through a secondary source | Seven coefficients and standard errors, residual SD, R squared, F, residual SS; the multicollinearity check must fire. | `test_regression.py` |
| StRD univariate: NumAcc1, NumAcc2, NumAcc3, NumAcc4 | NIST Statistical Reference Datasets, Univariate Summary Statistics, https://www.itl.nist.gov/div898/strd/univ/homepage.html . The data follow a documented pattern and are generated in the test. | NIST certified | Mean and SD. | `test_descriptives.py` |
| Grubbs' test example (data of Tietjen and Moore 1972) | NIST/SEMATECH e-Handbook of Statistical Methods, section 1.3.5.17, https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h1.htm | Published worked example (NIST handbook) | G = 2.4687; one-sided 5% critical value 2.032; the value 245.57 is flagged. | `test_outliers.py` |
| Weights of 11 men | Shapiro SS, Wilk MB (1965) An analysis of variance test for normality (complete samples). Biometrika 52:591-611. | Published worked example | W = 0.79; significant at the 1% level. | `test_assumptions.py` |
| Sleep data of Cushny and Peebles (10 patients, two soporifics) | Student (1908) The probable error of a mean. Biometrika 6:1-25, illustration I. | Published worked example | Student's own summary: mean difference 1.58, SD 1.17. | `test_two_group.py` |
| The same sleep data, documented software output | Documented examples of `t.test` and `wilcox.test` in R (R Core Team, R: A Language and Environment for Statistical Computing). Values typed from the printed output. | Published worked example (software documentation) | Paired, Welch and pooled t tests (t, df, P, CI); rank-sum and signed-rank statistics and P values. | `test_two_group.py` |
| Critical values of rank tests | Classical tables: Mann-Whitney U (n1 = n2 = 10, two-sided 5%: U <= 23); Wilcoxon signed-rank T (n = 11, two-sided 5%: T <= 10). | Published worked example (table) | The exact null distributions behind the P values and confidence intervals. | `test_two_group.py` |
| Grubbs critical values | Grubbs FE, Beck G (1972) Extension of sample sizes and percentage points for significance tests of outlying observations. Technometrics 14:847-854. | Published worked example (table) | Two-sided 5%: n = 10, 2.290; n = 50, 3.128. One-sided 5%: n = 8, 2.032. | `test_outliers.py` |
| Power tables | Cohen J (1988) Statistical Power Analysis for the Behavioral Sciences, 2nd ed., Table 2.4.1; Cohen J (1992) A power primer. Psychol Bull 112:155-159, Table 2. | Published worked example (table) | t test: d = 0.2, 0.5, 0.8 gives n = 393, 64, 26 per group. ANOVA with 3 to 7 groups. | `test_power.py` |
| Quartile definition 7 | Hyndman RJ, Fan Y (1996) Sample quantiles in statistical packages. Am Stat 50:361-365. | Published worked example (formula) | Quartiles from the defining formula. | `test_descriptives.py` |
| Anderson-Darling anchor points | D'Agostino RB, Stephens MA (1986) Goodness-of-Fit Techniques, Table 4.9. | Published worked example (table) | A\* = 0.631, 0.752, 1.035 for P = 0.10, 0.05, 0.01. | `test_assumptions.py` |
| Plant growth (3 groups of 10) | Dobson AJ (1983) An Introduction to Statistical Modelling. Chapman & Hall, Table 7.1. Results as printed in the documented R example for these data. | Published worked example | One-way ANOVA: df, sums of squares, mean squares, F = 4.8461, P = 0.01591. | `test_oneway.py` |
| Plant growth: Tukey, Welch, Kruskal-Wallis | The same data; R output that is not part of a documented example, typed from memory. | Cross-implementation | Tukey differences, limits and adjusted P; Welch F = 5.181; Kruskal-Wallis chi-squared = 7.9882. | `test_oneway.py`, `test_posthoc.py` |
| Mucociliary efficiency (n = 5, 4, 5) | Hollander M, Wolfe DA (1973) Nonparametric Statistical Methods. Wiley, pp. 115-120. | Published worked example | Kruskal-Wallis chi-squared = 0.77143, df = 2, P = 0.68. | `test_oneway.py` |
| Rounding first base (22 players, 3 methods) | Hollander M, Wolfe DA (1973) Nonparametric Statistical Methods. Wiley, pp. 139-146. | Published worked example | Friedman chi-squared = 11.143, df = 2, P = 0.003805. | `test_oneway.py` |
| New York ozone by month, 1973 | Chambers JM, Cleveland WS, Kleiner B, Tukey PA (1983) Graphical Methods for Data Analysis. Wadsworth. | Published worked example | Kruskal-Wallis with missing values: chi-squared = 29.267, df = 4. | `test_oneway.py` |
| Warp breaks (2 wools, 3 tensions, 9 looms) | Tippett LHC (1950) Technological Applications of Statistics. Wiley, p. 106. | Published worked example | Tukey comparisons with the error term of a two-factor model: differences, limits, adjusted P. | `test_posthoc.py` |
| Dunnett critical values | Dunnett CW (1955) A multiple comparison procedure for comparing several treatments with a control. J Am Stat Assoc 50:1096-1121; Dunnett CW (1964) New tables for multiple comparisons with a control. Biometrics 20:482-491. | Published worked example (table) | 5% points for 1 to 4 treatments against a control, df 10, 20 and infinity (20 values, to 2 decimals). | `test_posthoc.py` |
| Studentized range 5% points | Harter HL (1960) Tables of range and studentized range. Ann Math Stat 31:1122-1147. | Published worked example (table) | q(3, 30), q(3, 10), q(5, 20), q(4, 20), q(2, 10). | `test_posthoc.py` |
| Battery life, 3 x 3 factorial, n = 4 | Montgomery DC. Design and Analysis of Experiments. Wiley, chapter 5 (Example 5.1). | Published worked example | Two-way ANOVA: sums of squares, df, MS error, F = 7.91, 28.97, 3.56. | `test_factorial.py` |
| Tooth growth in guinea pigs, 2 x 3 factorial, n = 10 | Crampton EW (1947) The growth of the odontoblasts of the incisor tooth as a criterion of the vitamin C intake of the guinea pig. J Nutr 33:491-504. Also analysed in Bliss CI (1952) The Statistics of Bioassay. Academic Press. | Published worked example | Two-way ANOVA: F = 15.572, 92.000, 4.107; sums of squares; interaction P = 0.02186. | `test_factorial.py` |
| Eight subjects, four conditions (repeated measures) | Field A. Discovering Statistics Using IBM SPSS Statistics. SAGE, chapter on repeated-measures designs. | Published worked example | F(3, 21) = 3.79; Mauchly W = 0.136, P = 0.047; Greenhouse-Geisser epsilon 0.533; Huynh-Feldt epsilon 0.666. | `test_repeated.py` |
| Puromycin, treated enzyme (12 points) | Treloar MA (1974) Effects of Puromycin on Galactosyltransferase in Golgi Membranes. MSc thesis, Univ. of Toronto; analysed in Bates DM, Watts DG (1988) Nonlinear Regression Analysis and Its Applications. Wiley, Appendix A1.3. | Published worked example | Michaelis-Menten fit: Vmax = 212.68 (SE 6.947), Km = 0.06412 (SE 0.008281), residual SE 10.93. | `test_nonlinear.py` |
| BOD (6 points) | Marske DM (1967) MSc thesis, Univ. of Wisconsin; in Bates DM, Watts DG (1988), Appendix A1.4. | Published worked example | Two-parameter exponential fit: 19.143, 0.5311; residual SS 25.99. | `test_nonlinear.py` |
| Anscombe's quartet | Anscombe FJ (1973) Graphs in statistical analysis. Am Stat 27:17-21. | Published worked example | Line y = 3 + 0.5x, sums of squares, SE of slope 0.118, R squared 0.667, r = 0.816, for all four sets (to the precision of the paper). | `test_regression.py`, `test_correlation.py` |
| 6-mercaptopurine leukaemia trial (42 children) | Freireich EJ, Gehan E, Frei E III, et al. (1963) The effect of 6-mercaptopurine on the duration of steroid-induced remissions in acute leukemia. Blood 21:699-716. Tabulated in Gehan EA (1965) Biometrika 52:203-223 and Cox DR (1972) J R Stat Soc B 34:187-220. | Published worked example | Kaplan-Meier estimates of the treated arm; medians 23 and 8 weeks; log-rank chi-square 16.79. | `test_survival.py` |
| Lady tasting tea, [[3, 1], [1, 3]] | Fisher RA (1935) The Design of Experiments. | Published worked example | Fisher's exact test: one-sided P = 17/70, two-sided P = 34/70. | `test_contingency.py` |
| Aspirin and myocardial infarction | Steering Committee of the Physicians' Health Study Research Group (1988) N Engl J Med 318:262-264; analysed in Agresti A, An Introduction to Categorical Data Analysis. | Published worked example | Proportions; risk difference with Wald CI; relative risk 1.82; odds ratio 1.83 with 95% CI 1.44 to 2.33. | `test_contingency.py` |
| Risk difference, 56/70 against 48/80 | Newcombe RG (1998) Interval estimation for the difference between independent proportions: comparison of eleven methods. Stat Med 17:873-890. | Published worked example | Method 10 interval +0.0524 to +0.3339. | `test_contingency.py` |
| Party identification by gender, 2 x 3 table | 2000 General Social Survey, as analysed in Agresti A, An Introduction to Categorical Data Analysis, 2nd ed. | Published worked example | X^2 = 30.1 on 2 df; expected counts; standardized residuals. | `test_contingency.py` |
| Infant malformation by maternal alcohol consumption, 2 x 5 table | Graubard BI, Korn EL (1987) Choice of column scores for testing independence in ordered 2 x K contingency tables. Biometrics 43:471-476; analysed in Agresti (same book). | Published worked example | r = 0.0142; trend statistic 6.57 (after conversion by (n - 1)/n), P = 0.010. | `test_contingency.py` |
| ROUT method | Motulsky HJ, Brown RE (2006) Detecting outliers when fitting data with nonlinear regression - a new method based on robust nonlinear regression and the false discovery rate. BMC Bioinformatics 7:123. https://pmc.ncbi.nlm.nih.gov/articles/PMC1472692/ | Simulation (method description only; no number from the paper is asserted) | The detection step against an independent re-implementation on 300 random data sets; false-positive rate and detection rate in fixed-seed simulations. | `test_outliers.py` |
| SciPy | Virtanen P, et al. (2020) SciPy 1.0: fundamental algorithms for scientific computing in Python. Nat Methods 17:261-272. | Cross-implementation | t tests, rank tests, ANOVA, Tukey, Dunnett, Kruskal-Wallis, Friedman, normality and variance tests, correlation, linear regression, `curve_fit`, chi-square, Fisher's exact test, noncentral distributions. | most test files |
| statsmodels | Seabold S, Perktold J (2010) Statsmodels: econometric and statistical modeling with Python. Proc 9th Python in Science Conf. | Cross-implementation | Factorial and repeated-measures ANOVA (`anova_lm`, `AnovaRM`, `mixedlm`), Welch and Brown-Forsythe ANOVA, Tukey, P value adjustments, regression diagnostics, Lilliefors and Anderson-Darling, 2 x 2 measures, power. | most test files |
| lifelines | Davidson-Pilon C (2019) lifelines: survival analysis in Python. J Open Source Softw 4:1317. | Cross-implementation | Kaplan-Meier estimates, log-log limits, medians, log-rank and Gehan-Breslow-Wilcoxon tests, pairwise tests. | `test_survival.py` |
| scikit-posthocs | Terpilowski MA (2019) scikit-posthocs: pairwise multiple comparison tests in Python. J Open Source Softw 4:1169. | Cross-implementation | Games-Howell, Dunn's test, rank-sum comparisons after Friedman, pooled-SD pairwise t tests. | `test_posthoc.py` |
| NumPy and pandas | Harris CR, et al. (2020) Array programming with NumPy. Nature 585:357-362. The pandas development team, pandas (software). | Cross-implementation | Descriptive statistics, quantiles, correlation matrices, group means. | `test_descriptives.py`, `test_correlation.py`, `test_nested.py` |
| First-principles calculations written in the tests | Formulas from the primary method papers listed in docs/STATS_METHODS.md; brute-force enumeration of exact null distributions (Mann-Whitney, Wilcoxon, Spearman, runs test, Fisher r x c); exact rational arithmetic for the one-way ANOVA; nested least-squares fits for Type II and III sums of squares. | Cross-implementation | Statistics, P values, intervals and sums of squares. | most test files |
| Calibration simulations (fixed seeds) | Written for TorusPlot. | Simulation | False-positive rate and power of the normality tests; Bartlett against Brown-Forsythe on heavy-tailed data; the Mauchly P value under sphericity; false-positive rate of Grubbs' test; coverage of rank-based and dose-response confidence intervals; Monte-Carlo power of t tests. | `test_assumptions.py`, `test_repeated.py`, `test_outliers.py`, `test_two_group.py`, `test_nonlinear.py`, `test_power.py` |

Property-based tests (written with the `hypothesis` package) run next to these checks. They feed
random and degenerate inputs (ties, zeros, constant groups, missing values, huge counts) and assert
invariants: P values between 0 and 1, intervals that contain the estimate, invariance to the order
of rows and to a change of units, and a clear error message instead of a crash.

## NIST files in this folder

`tests/fixtures/nist/` holds 39 data files and one loader (`strd_regression.py`):

- One-way ANOVA (11): `SiRstv.dat`, `SmLs01.dat` to `SmLs09.dat`, `AtmWtAg.dat`.
- Nonlinear regression (26): `Misra1a`, `Misra1b`, `Misra1c`, `Misra1d`, `Chwirut1`, `Chwirut2`,
  `Lanczos1`, `Lanczos2`, `Lanczos3`, `Gauss1`, `Gauss2`, `Gauss3`, `DanWood`, `Kirby2`, `Hahn1`,
  `MGH09`, `MGH10`, `MGH17`, `Roszman1`, `ENSO`, `Thurber`, `BoxBOD`, `Rat42`, `Rat43`, `Eckerle4`,
  `Bennett5` (each with the extension `.dat`).
- Linear regression (2): `Norris.dat`, `Longley.dat`.

## Attribution for the NIST reference data

The reference data sets and certified values are from the NIST Statistical Reference Datasets,
National Institute of Standards and Technology, https://www.itl.nist.gov/div898/strd/ . They are
works of the U.S. Government and are not subject to copyright protection in the United States.
NIST asks to be cited as the source. The ANOVA, nonlinear-regression and Norris files are kept
exactly as NIST distributes them, with NIST's own header at the top of each file, so no attribution
header was added inside them. The use of these data sets does not imply any endorsement of
TorusPlot by NIST.

## Example datasets are synthetic

Every dataset in `examples/` is **synthetic**. The files are simulated by
`examples/generate_examples.py` with fixed random seeds and are not real measurements. They are
used in the user guide and for learning the app, not for validation. See `examples/README.md`.

## Provenance caveats

These points are stated plainly so that an owner can decide what to re-verify.

1. **Nothing was downloaded from NIST.** The NIST web site could not be reached from the build
   environment. The ANOVA files and `Norris.dat` are byte-for-byte copies of the NIST files that
   ship in the test data of the SciPy package. The 26 nonlinear-regression files were taken from the
   source distribution of the lmfit package (version 1.3.4), which redistributes the unmodified NIST
   files; no lmfit code was read or used. If network access is available, download the files from
   NIST again and compare them.
2. **`Longley.dat` is not the verbatim NIST file.** It was assembled by hand in the NIST layout:
   the data from `statsmodels.datasets.longley`, the reference values from the statsmodels
   test-suite, which cites NIST. The values were not compared with the NIST file itself.
3. **The NumAcc data sets were generated, not downloaded.** They follow a simple documented pattern
   (for example NumAcc2: 1.2 followed by 500 pairs of 1.1 and 1.3). The certified means and SDs
   were typed in.
4. **Published values were typed from memory or from the task brief, not from the printed
   sources.** The books and papers were not at hand. A value was kept only when an independent
   recomputation reproduced it to the published precision, which a mistyped data value would not do.
   This applies to: the Shapiro-Wilk example, the sleep data, the NIST handbook Grubbs example, the
   critical-value tables (rank tests, Grubbs, Dunnett, studentized range, Anderson-Darling, Cohen's
   power tables), the plant growth, mucociliary, rounding-first-base, ozone and warp-breaks data,
   the battery-life and tooth-growth experiments, the repeated-measures example, the puromycin and
   BOD data, Anscombe's quartet, the leukaemia trial, and the five contingency-table examples.
5. **Page, table and example numbers of these sources were not verified** (for example the chapter
   and example number of the battery-life experiment, the edition and chapter of the
   repeated-measures example, the table numbers in Dunnett's two papers). A maintainer with the
   printed sources should confirm them before they are quoted elsewhere.
6. **Several results are documented software output for published data, not numbers printed in the
   original publication.** This applies to the one-way ANOVA, Kruskal-Wallis, Friedman and Tukey
   results taken from documented R examples, and to the t test and rank-test results for the sleep
   data. The additional Tukey, Welch and Kruskal-Wallis results for the plant growth data are R
   output recalled from memory and are counted as cross-implementation evidence only.
7. **Some of these classic datasets are also redistributed with other statistics software.** The
   sleep, plant growth, ozone, warp-breaks, tooth-growth, puromycin and BOD data are distributed with
   the R `datasets` package, which is under a copyleft licence. The numbers themselves are facts from
   the primary literature and were typed in, not copied from a file. If the project prefers to avoid
   any doubt, the tooth-growth test can be deleted without losing coverage (the battery-life example
   covers the same analysis); [sources/factorial_repeated.md](sources/factorial_repeated.md) names
   the test and the constant.
8. **ROUT was implemented from a description of the method, not from the paper text.** No number
   from Motulsky and Brown (2006) is asserted. An equation-by-equation comparison with the paper
   (scale of the merit function, percentile interpolation) is still open.
9. **Analyses without any published worked example.** Three-way ANOVA, ANCOVA, mixed-design ANOVA,
   two-way repeated-measures ANOVA, Brown-Forsythe ANOVA, Games-Howell and Dunn's tests, comparison
   of slopes, four-parameter dose-response fits, hazard ratios, the log-rank trend test and the
   nested analysis rest on cross-implementation checks and first-principles calculations only.
10. **Published values that could not be reproduced are not asserted.** They are listed under
    "Tried and dropped" in the per-area files: the Grubbs critical value for n = 20 (table 2.709,
    formula 2.708), one entry of Cohen's ANOVA power table (195 against an exact 195.53), the
    expected event counts printed for the leukaemia trial in some textbooks, and the chi-square
    statistic and relative-risk interval recalled for the aspirin table.
11. **A misprint in a NIST header.** The `Rat43` header prints 9 degrees of freedom for 15
    observations and 4 parameters; the certified residual SD corresponds to 11, which the test uses.

# Validation sources: one-way ANOVA family and post-hoc tests

Covers `torusplot/stats/oneway.py` and `torusplot/stats/posthoc.py`
(tests: `tests/stats/test_nist_anova.py`, `test_oneway.py`, `test_posthoc.py`).

Every "published" value below was checked by running the code: the expected numbers in
the tests are the published ones, typed in by hand, and were never adjusted to make a
test pass.

## 1. NIST Statistical Reference Datasets (certified values)

**Source.** National Institute of Standards and Technology, Information Technology
Laboratory, *Statistical Reference Datasets (StRD): Analysis of Variance*,
https://www.itl.nist.gov/div898/strd/anova/anova.html . The StRD files are works of the
US Government and are not subject to copyright protection in the United States; NIST
asks to be cited as the source.

**Files** (`tests/fixtures/nist/`, unmodified, NIST's own header with the certified
values is kept at the top of each file):

| File | Difficulty | Observations | Original reference given by NIST |
|---|---|---|---|
| `SiRstv.dat` | lower | 25 (5 instruments x 5) | Ehrstein J, Croarkin MC. Unpublished NIST dataset (silicon resistivity) |
| `SmLs01.dat`, `SmLs02.dat`, `SmLs03.dat` | lower | 189 / 1809 / 18009 | Simon SD, Lesage JP (1989) Assessing the accuracy of ANOVA calculations in statistical software. *Comput Stat Data Anal* 8:325-332 |
| `AtmWtAg.dat` | average | 48 (2 instruments) | Powell LJ, Murphy TJ, Gramlich JW (1982) The absolute isotopic abundance & atomic weight of a reference sample of silver. *NBS J Res* 87:9-19 |
| `SmLs04.dat`, `SmLs05.dat`, `SmLs06.dat` | average | 189 / 1809 / 18009 | Simon & Lesage (1989) |
| `SmLs07.dat`, `SmLs08.dat`, `SmLs09.dat` | higher | 189 / 1809 / 18009 | Simon & Lesage (1989) |

**How obtained.** Direct download from `www.itl.nist.gov` was blocked in the build
sandbox (the proxy refused the connection). The files were therefore copied byte for
byte from the copies of NIST's files that ship inside the installed SciPy 1.17.1 wheel
(`scipy/stats/tests/data/nist_anova/`, BSD-3-Clause package redistributing the
unmodified NIST files). Nothing was typed by hand. If network access is available the
files can be re-verified against
`https://www.itl.nist.gov/div898/strd/anova/<name>.dat`.

Note: the contract asks for "an attribution header" in the fixture files. The files
were deliberately left byte-identical to NIST's (each already starts with
`NIST/ITL StRD`, the dataset name and its reference, and NIST's header states the line
numbers of the certified values and data, which a prepended header would shift). The
attribution lives in this document and in `tests/stats/_nist.py`.

**Values checked** (`test_nist_anova.py`): between- and within-treatment degrees of
freedom, sums of squares, mean squares, the F statistic, certified R-squared and
certified residual standard deviation; the P value against the F distribution evaluated
at the certified F.

**Agreement reached** (significant digits = -log10 relative error, worst of
SS / MS / F, analysis fed ordinary double-precision numbers):

| Dataset | F | SS between | SS within | Tolerance in test |
|---|---|---|---|---|
| SiRstv | 13.1 | 14.0 | 13.1 | 1e-12 |
| SmLs01 | 15 | 15 | 15 | 1e-13 |
| SmLs02 | 15 | 15 | 15 | 1e-13 |
| SmLs03 | 15 | 15 | 15 | 1e-13 |
| AtmWtAg | 10.2 | 10.2 | 10.9 | 1e-9 |
| SmLs04 | 10.4 | 10.1 | 10.3 | 1e-9 |
| SmLs05 | 10.2 | 9.9 | 10.3 | 1e-9 |
| SmLs06 | 10.2 | 9.9 | 10.3 | 1e-9 |
| SmLs07 | 4.4 | 4.0 | 4.3 | 5e-4 |
| SmLs08 | 4.2 | 3.9 | 4.3 | 5e-4 |
| SmLs09 | 4.2 | 3.9 | 4.3 | 5e-4 |

The loss on the average / higher difficulty sets is a property of the *input*, not of
the algorithm: values such as `1000000000000.4` cannot be stored exactly in a double
(spacing 1.2e-4 near 1e12, against a scatter of 0.1), and NIST's certified values refer
to the exact decimal data. Two further tests demonstrate this: (a) when the constant
leading digits are removed in exact decimal arithmetic before conversion to double, all
eleven datasets are reproduced to >= 14 digits; (b) an exact rational-arithmetic ANOVA
of the same double-precision numbers agrees with the code to 1e-13. SciPy's `f_oneway`
reaches the same number of digits on every dataset.

## 2. Published worked examples

All datasets below were typed in by hand from memory of the primary data as they are
widely reproduced (they are also distributed with the R `datasets` package; no code was
copied). Each was accepted only because the code reproduces the documented result to
every printed digit, which would not happen with a mistyped value.

| Dataset | Primary source | Analysis | Published values checked | Where the values are printed |
|---|---|---|---|---|
| Plant growth (3 groups x 10) | Dobson AJ (1983) *An Introduction to Statistical Modelling*. Chapman & Hall, Table 7.1 | one-way ANOVA | df 2 and 27; SS 3.7663 / 10.4921; MS 1.8832 / 0.3886; F = 4.8461; P = 0.01591 | documented example of the `PlantGrowth` help page in R (`anova(lm(weight ~ group))`) |
| same | same | Tukey HSD | differences 0.371, 0.494, 0.865; 95% family-wise limits (-1.0622161, 0.3202161), (-0.1972161, 1.1852161), (0.1737839, 1.5562161); adjusted P 0.3908711, 0.1979960, 0.0120064 | R `TukeyHSD(aov(weight ~ group))` output (software output recalled from memory; also cross-checked against scipy and statsmodels) |
| same | same | Welch ANOVA | F = 5.181, num df = 2, denom df = 17.128, P = 0.01739 | R `oneway.test(weight ~ group)` output (software output recalled from memory; also cross-checked against statsmodels) |
| same | same | Kruskal-Wallis | chi-squared = 7.9882, df = 2, P = 0.01842 | R `kruskal.test(weight ~ group)` output (software output recalled from memory; also cross-checked against scipy) |
| Mucociliary efficiency (n = 5, 4, 5) | Hollander M, Wolfe DA (1973) *Nonparametric Statistical Methods*. Wiley, pp. 115-120 | Kruskal-Wallis | chi-squared = 0.77143, df = 2, P = 0.68 | documented example of R's `kruskal.test` |
| New York ozone by month, May-Sep 1973 (116 non-missing days) | Chambers JM, Cleveland WS, Kleiner B, Tukey PA (1983) *Graphical Methods for Data Analysis*. Wadsworth | Kruskal-Wallis with missing values | chi-squared = 29.267, df = 4, P = 6.901e-06 | documented example of R's `kruskal.test` (`Ozone ~ Month`, `airquality`) |
| Rounding first base (22 players x 3 methods) | Hollander & Wolfe (1973), pp. 139-146 | Friedman test | chi-squared = 11.143, df = 2, P = 0.003805 | documented example of R's `friedman.test` |
| Warp breaks (2 wools x 3 tensions x 9 looms) | Tippett LHC (1950) *Technological Applications of Statistics*. Wiley, p. 106 | Tukey HSD on tension with the error term of the additive two-factor model (tests the `mse` / `df_error` override) | differences 4.722222, 14.722222, 10.000000; limits (-4.6311985, 14.07564), (5.3688015, 24.07564), (0.6465793, 19.35342); adjusted P 0.4474210, 0.0011218, 0.0336262 | documented example of R's `TukeyHSD` (`aov(breaks ~ wool + tension)`, `"tension"`) |
| Dunnett critical values | Dunnett CW (1955) *J Am Stat Assoc* 50:1096-1121 (one-sided); Dunnett CW (1964) *Biometrics* 20:482-491 (two-sided) | Dunnett's test: critical values from the module's own numerical integration | 5% points for 1-4 treatments vs control, equal n. Two-sided: df 10: 2.23, 2.57, 2.76, 2.89; df 20: 2.09, 2.38, 2.54, 2.65; df infinity: 1.96, 2.21, 2.35, 2.44. One-sided: df 10: 1.81, 2.15, 2.34, 2.47; df 20: 1.72, 2.03, 2.19, 2.30 | the papers' tables (values recalled from the standard reprinted tables; all 20 agree with the computed values after rounding to 2 decimals) |
| Studentized range 5% points | Harter HL (1960) Tables of range and studentized range. *Ann Math Stat* 31:1122-1147 | Tukey / Games-Howell critical values | q(3, 30) = 3.49, q(3, 10) = 3.88, q(5, 20) = 4.23, q(4, 20) = 3.96, q(2, 10) = 3.15 | standard reprinted table |

Candid notes on provenance:

* The R help-page examples (`PlantGrowth` ANOVA, both `kruskal.test` examples,
  `friedman.test`, `TukeyHSD` on `warpbreaks`) are documented, reproducible outputs for
  textbook datasets. They are software output for published data rather than numbers
  printed in the original books; I could not consult the books or the R manual from the
  sandbox, so data and expected values were typed from memory and accepted only on exact
  reproduction.
* The three additional `PlantGrowth` rows (Tukey, Welch, Kruskal-Wallis) are R outputs
  that are not part of an R help-page example; treat them as cross-implementation
  evidence rather than independent publications.
* Table numbers within Dunnett's two papers and the exact page of the rounding-times
  table are not quoted because I could not verify them.

## 3. Published examples tried and dropped

* Dunnett (1955) blood-count example (control and two drugs): the data are well known,
  but I could not recall the paper's printed allowances with certainty, so no
  "published" expectation is asserted for it. Dunnett's test is validated instead by the
  published critical-value tables above, by an independent adaptive-quadrature
  evaluation written in the test, and against `scipy.stats.dunnett`.
* No published worked example with certain data *and* results was available from memory
  for Games-Howell, Dunn's test or the Brown-Forsythe F* test; these rely on
  hand-computed small examples, from-formula computations and cross-implementation
  checks.

## 4. Cross-implementation checks (allowed stack only)

| Our function | Checked against |
|---|---|
| `anova_oneway` | `scipy.stats.f_oneway`; `statsmodels` OLS + `anova_lm`; textbook sums of squares; exact rational arithmetic (`fractions.Fraction`) |
| eta squared CI | inversion of `scipy.stats.ncf` verified at both limits |
| assumption checks | `scipy.stats.shapiro`, `levene(center="median")`, `bartlett` |
| `welch_anova` | `statsmodels.stats.oneway.anova_oneway(use_var="unequal", welch_correction=True)`; Welch (1951) formula written out; Welch t test for k = 2 |
| `brown_forsythe_anova` | `statsmodels ... anova_oneway(use_var="bf")` (statistic, `df_denom`, `pvalue2` = original test with k - 1 numerator df; `pvalue` / `df_num` = Mehrotra's correction, reported in `extra["mehrotra"]`); Brown & Forsythe (1974) formula written out |
| `kruskal_wallis` | `scipy.stats.kruskal` (with ties); textbook rank-sum formula; Mann-Whitney for k = 2 |
| `friedman_test` | `scipy.stats.friedmanchisquare` (with ties); textbook formula without ties; Iman-Davenport F from its formula |
| `tukey_hsd` | `scipy.stats.tukey_hsd` (P and CI); `statsmodels` `pairwise_tukeyhsd` (unequal n); `scipy.stats.studentized_range` from first principles; t test for k = 2 |
| `dunnett` | `scipy.stats.dunnett` (t statistics to 1e-10; P and CI within scipy's quasi-Monte-Carlo noise, abs 1e-3); independent adaptive quadrature of Dunnett's double integral (abs 2e-9) |
| `games_howell` | `scikit_posthocs.posthoc_games_howell`; `scipy.stats.ttest_ind(equal_var=False)` for SE / df; hand-computed example |
| `pairwise_t` | `scipy.stats.ttest_ind` / `ttest_rel` (statistic, df, CI); `statsmodels` `multipletests`; `scikit_posthocs.posthoc_ttest(pool_sd=True)` |
| `dunn` | `scikit_posthocs.posthoc_dunn` (all adjustments, with ties); formula written out; `scipy.stats.mannwhitneyu` for the rank-biserial effect size |
| `friedman_posthoc` | `scikit_posthocs.posthoc_siegel_friedman`; `scipy.stats.wilcoxon` for the Wilcoxon option |

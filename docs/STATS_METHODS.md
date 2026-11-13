# TorusPlot statistical methods reference

This document says, for every analysis and supporting method in TorusPlot:

- **What is computed**: the formula, in plain notation.
- **Implementation**: whether the number comes from TorusPlot's own code or from a library function.
- **Assumptions**: what the method assumes and how TorusPlot checks it.
- **Limitations**: where the method is approximate or stops working.
- **Validation**: what the test-suite compares it with.
- **Primary reference**: the paper or book that introduced the method, as cited in the code.

For worked examples see [USER_GUIDE.md](USER_GUIDE.md). For every validation source see
[tests/fixtures/SOURCES.md](../tests/fixtures/SOURCES.md).

Links are given only where the address is certain. Other references are full citations without a link.

TorusPlot is a tool, not a statistician. For clinical data, regulatory submissions and other
high-stakes work, have a statistician review the plan and the results.

## Contents

1. [Summary table](#1-summary-table)
2. [How the engine is built](#2-how-the-engine-is-built)
3. [Descriptive statistics](#3-descriptive-statistics)
4. [Assumption tests](#4-assumption-tests)
5. [Comparing two groups](#5-comparing-two-groups)
6. [Effect sizes and their confidence intervals](#6-effect-sizes-and-their-confidence-intervals)
7. [Comparing three or more groups](#7-comparing-three-or-more-groups)
8. [Multiple-comparison procedures](#8-multiple-comparison-procedures)
9. [Factorial ANOVA and ANCOVA](#9-factorial-anova-and-ancova)
10. [Repeated measures](#10-repeated-measures)
11. [Correlation](#11-correlation)
12. [Linear regression](#12-linear-regression)
13. [Nonlinear regression](#13-nonlinear-regression)
14. [Survival analysis](#14-survival-analysis)
15. [Contingency tables](#15-contingency-tables)
16. [Outlier detection](#16-outlier-detection)
17. [Replicate-aware (nested) analysis](#17-replicate-aware-nested-analysis)
18. [Power and sample size](#18-power-and-sample-size)
19. [Test recommendation](#19-test-recommendation)
20. [Conventions](#20-conventions)
21. [What TorusPlot does not do (yet)](#21-what-torusplot-does-not-do-yet)

## 1. Summary table

"Validated against" names the strongest evidence for each analysis. *NIST* means certified values of
the NIST Statistical Reference Datasets (https://www.itl.nist.gov/div898/strd/). *Published* means a
worked example from the literature. *Cross-check* means agreement with an independent
implementation (SciPy, statsmodels, lifelines, scikit-posthocs) or with a from-first-principles
calculation written in the test. Where no published example is used, the table says so.

| Analysis | Registry key | Table types | Primary reference (as cited in the code) | Validated against |
|---|---|---|---|---|
| Descriptive statistics | `descriptives` | Column, Grouped, XY, Multiple variables | Hyndman & Fan (1996) Am Stat 50:361-365 (quartile definition 7) | NIST StRD univariate sets NumAcc1 to NumAcc4 (certified mean and SD); cross-check with NumPy, SciPy, pandas, statsmodels |
| Normality tests | `normality` | Column, Multiple variables | Shapiro & Wilk (1965) Biometrika 52:591-611; D'Agostino & Pearson (1973) Biometrika 60:613-622; Lilliefors (1967) J Am Stat Assoc 62:399-402 | Published: Shapiro & Wilk (1965) worked example (W = 0.79); cross-check with SciPy and statsmodels; simulation |
| Equal variance tests | `equal_variance` | Column | Brown & Forsythe (1974) J Am Stat Assoc 69:364-367; Levene (1960); Bartlett (1937) Proc R Soc A 160:268-282 | Cross-check with SciPy (`levene`, `bartlett`) and formulas; simulation. No published example |
| Unpaired t test | `ttest_unpaired` | Column | Student (1908) Biometrika 6:1-25 | Published: Student (1908) sleep data, with documented R output; cross-check with SciPy and statsmodels |
| Welch's t test | `ttest_welch` | Column | Welch (1947) Biometrika 34:28-35 | Documented R output for the sleep data; cross-check with SciPy and statsmodels |
| Paired t test | `ttest_paired` | Column | Student (1908) Biometrika 6:1-25 | Published: Student (1908) sleep data, with documented R output; cross-check with SciPy |
| One-sample t test | `ttest_one_sample` | Column | Student (1908) Biometrika 6:1-25 | Cross-check with SciPy and statsmodels. No published example |
| Mann-Whitney test | `mann_whitney` | Column | Mann & Whitney (1947) Ann Math Stat 18:50-60 | Documented R output for the sleep data; cross-check with SciPy; brute-force exact null distribution; classical critical-value table |
| Wilcoxon matched-pairs signed-rank test | `wilcoxon_signed_rank` | Column | Wilcoxon (1945) Biometrics Bull 1:80-83 | Documented R output for the sleep data; cross-check with SciPy; brute-force exact null distribution; classical critical-value table |
| Wilcoxon signed-rank test (one sample) | `wilcoxon_one_sample` | Column | Wilcoxon (1945) Biometrics Bull 1:80-83 | Cross-check with SciPy; shares the engine of the matched-pairs test |
| One-way ANOVA | `anova_oneway` | Column | Fisher RA (1925) Statistical Methods for Research Workers. Oliver & Boyd | NIST StRD one-way ANOVA, all 11 data sets; published: Dobson (1983) plant growth data; cross-check with SciPy, statsmodels, exact rational arithmetic |
| Welch's ANOVA | `welch_anova` | Column | Welch BL (1951) Biometrika 38:330-336 | Cross-check with statsmodels and Welch's formula. No independent published example |
| Brown-Forsythe ANOVA | `brown_forsythe_anova` | Column | Brown MB, Forsythe AB (1974) Technometrics 16:129-132 | Cross-check with statsmodels and the formula. No published example |
| Kruskal-Wallis test | `kruskal_wallis` | Column | Kruskal WH, Wallis WA (1952) J Am Stat Assoc 47:583-621 | Published: Hollander & Wolfe (1973) data; ozone data with missing values; cross-check with SciPy |
| Friedman test | `friedman` | Column | Friedman M (1937) J Am Stat Assoc 32:675-701 | Published: Hollander & Wolfe (1973) rounding-first-base data; cross-check with SciPy |
| Two-way ANOVA | `anova_twoway` | Grouped, Multiple variables | Fisher RA (1925) Statistical Methods for Research Workers. Oliver & Boyd, Edinburgh | Published: Montgomery battery-life experiment; constructed balanced design with exact sums of squares; cross-check with statsmodels |
| Three-way ANOVA | `anova_threeway` | Multiple variables | Fisher RA (1925) Statistical Methods for Research Workers. Oliver & Boyd, Edinburgh | Cross-check with statsmodels `anova_lm`; classical balanced formulas. No published example |
| ANCOVA (analysis of covariance) | `ancova` | Multiple variables | Cochran WG (1957) Analysis of covariance: its nature and uses. Biometrics 13:261-281 | Cross-check with statsmodels; first-principles adjusted means. No published example |
| Repeated-measures one-way ANOVA | `anova_rm_oneway` | Column | Greenhouse SW, Geisser S (1959) On methods in the analysis of profile data. Psychometrika 24:95-112 | Constructed 8 x 4 data checked against first-principles F, Mauchly and epsilons; cross-check with statsmodels `AnovaRM` |
| Mixed-design ANOVA (repeated measures by group) | `anova_mixed` | Grouped, Multiple variables | Winer BJ, Brown DR, Michels KM (1991) Statistical Principles in Experimental Design, 3rd ed. McGraw-Hill | Cross-check with statsmodels (`mixedlm`, `anova_lm` on contrast scores); first principles. No published example |
| Two-way repeated-measures ANOVA | `anova_rm_twoway` | Multiple variables | Greenhouse SW, Geisser S (1959) On methods in the analysis of profile data. Psychometrika 24:95-112 | Cross-check with statsmodels `AnovaRM`. No published example |
| Correlation (Pearson or Spearman) | `correlation` | XY, Column, Multiple variables | Pearson K (1896) Philos Trans R Soc A 187:253-318; Spearman C (1904) Am J Psychol 15:72-101 | Published: Anscombe's quartet; cross-check with SciPy; exact Spearman P by brute-force enumeration |
| Correlation matrix | `correlation_matrix` | Multiple variables, Column | Pearson K (1896) Philos Trans R Soc A 187:253-318; Spearman C (1904) Am J Psychol 15:72-101 | Cross-check with SciPy pair by pair and pandas `corr` |
| Simple linear regression | `linear_regression` | XY | Draper NR, Smith H (1998) Applied Regression Analysis, 3rd ed., Wiley | NIST StRD Norris (original file); published: Anscombe's quartet; cross-check with SciPy and statsmodels |
| Compare slopes of regression lines | `compare_slopes` | XY | Zar JH (2010) Biostatistical Analysis, 5th ed., ch. 18 | Cross-check with statsmodels nested models. No published example |
| Multiple linear regression | `multiple_regression` | Multiple variables | Draper NR, Smith H (1998) Applied Regression Analysis, 3rd ed., Wiley | Longley data (NIST values taken from the statsmodels test-suite, not from the NIST file); cross-check with statsmodels |
| Nonlinear regression (curve fit) | `nonlinear_regression` | XY | Bates DM, Watts DG (1988) Nonlinear Regression Analysis and Its Applications, Wiley | NIST StRD nonlinear regression, 26 data sets from both starting points; published: puromycin and BOD data (Bates & Watts 1988); cross-check with SciPy `curve_fit` |
| Compare two curve fits (which model is better?) | `compare_fits` | XY | Bates DM, Watts DG (1988) Nonlinear Regression Analysis and Its Applications, Wiley; Hurvich CM, Tsai CL (1989) Biometrika 76:297-307 | Cross-check with two independent `curve_fit` fits and statsmodels `anova_lm`. No published example |
| Compare a curve parameter between data sets (e.g. EC50) | `compare_curve_parameter` | XY | Bates DM, Watts DG (1988) Nonlinear Regression Analysis and Its Applications, Wiley | Cross-check with an independent global fit; reproduces the slope comparison for straight lines. No published example |
| Survival analysis (Kaplan-Meier, log-rank) | `survival` | Multiple variables | Kaplan & Meier (1958) J Am Stat Assoc 53:457-481; Mantel (1966) Cancer Chemother Rep 50:163-170 | Published: Freireich et al. (1963) 6-mercaptopurine trial; cross-check with lifelines. Hazard ratios: first principles only |
| Chi-square test | `chi_square` | Contingency | Pearson (1900) Philos Mag 50:157-175; Yates (1934) J R Stat Soc Suppl 1:217-235 | Published: Agresti, party identification 2 x 3 table; aspirin trial; cross-check with SciPy and statsmodels |
| Fisher's exact test | `fisher_exact` | Contingency | Fisher (1922) J R Stat Soc 85:87-94; Freeman & Halton (1951) Biometrika 38:141-149 | Published: lady tasting tea; cross-check with SciPy; brute-force enumeration for larger tables |
| Chi-square test for trend | `chi_square_trend` | Contingency | Cochran (1954) Biometrics 10:417-451; Armitage (1955) Biometrics 11:375-386 | Published: Graubard & Korn (1987) data as analysed by Agresti; cross-check with statsmodels |
| Identify outliers (Grubbs) | `outliers_grubbs` | Column | Grubbs (1969) Technometrics 11:1-21 | NIST/SEMATECH e-Handbook example (Tietjen & Moore data); published critical values; simulation |
| Identify outliers (ROUT) | `outliers_rout` | Column | Motulsky & Brown (2006) BMC Bioinformatics 7:123 | Independent re-implementation of the detection step; simulation. Not checked equation by equation against the paper |
| Replicate-aware comparison (SuperPlot analysis) | `nested_analysis` | Multiple variables | Lord SJ et al. (2020) J Cell Biol 219:e202001064 | Cross-check with SciPy and statsmodels on hand-aggregated means; mixed-model identity. No published example |

Two tools are not registered analyses: the power calculator (section 18) and the test
recommendation (section 19).

## 2. How the engine is built

All statistics are computed in `torusplot/stats/`. Most methods are TorusPlot's own code, written
from the primary formulas, on top of NumPy arrays and SciPy's probability distributions
(`scipy.stats.t`, `f`, `chi2`, `norm`, `nct`, `ncf`, `ncx2`, `studentized_range`).

Library routines that compute a result directly:

| Library routine | Used for |
|---|---|
| `scipy.stats.shapiro` | Shapiro-Wilk test (everywhere it appears) |
| `scipy.stats.levene(center="median")` | Brown-Forsythe check in the test recommendation, the factorial ANOVA, ANCOVA and mixed repeated-measures ANOVA results, and the nested analysis (the one-way ANOVA results and the stand-alone *Equal variance tests* analysis use TorusPlot's own formula) |
| `scipy.stats.probplot` | Q-Q plot coordinates |
| `statsmodels.stats.diagnostic.lilliefors` | Lilliefors P value of the Kolmogorov-Smirnov test |
| `statsmodels` OLS (`formula.api.ols`) and `anova_lm` | Factorial ANOVA and ANCOVA: model fit and Type II / III table |
| `statsmodels.api.OLS` | Multiple regression: the least-squares fit |
| `scipy.optimize.least_squares` | Nonlinear regression and ROUT: the minimiser |
| `scipy.stats.fisher_exact`, `scipy.stats.contingency.odds_ratio` | Fisher's exact test for 2 x 2 tables; conditional odds ratio with exact CI |
| `statsmodels.stats.power` (`TTestIndPower`, `TTestPower`, `FTestAnovaPower`) | Power calculations |
| `statsmodels` `mixedlm` | Only the mixed-model cross-check stored with a nested analysis |

lifelines and scikit-posthocs are installed with TorusPlot but the statistics engine does not call
them. The test-suite uses them as independent cross-checks (survival; Dunn, Games-Howell and
Friedman post-hoc tests).

**Numerical range.** Test statistics, P values and standardised effect sizes do not depend on the
origin or unit of the data. They are therefore computed on rescaled data: a shared location is
subtracted and the result divided by a power of two (an exact operation). Means, SDs and confidence
limits are scaled back to the units of the data. "Constant" (no variability) means a spread within
a few units in the last place of the magnitude of the data; there is no absolute floor, so 1e12 +
{0.1, 0.3} or {1e-80, 2e-80} are not constant. Paired differences are compared at the rounding
error of the original values (100.1 - 100 and 1.1 - 1 are the same difference). A quantity that
cannot be represented as a double-precision number is reported as missing (None) with a warning,
or the analysis stops with a clear error.

Every analysis returns one result object. Figures, reports and AI features refer to that result by
its ID. They never carry their own numbers.

## 3. Descriptive statistics

Registry key: `descriptives`.

- **What is computed.** Per group: n, mean, SD (n - 1 in the denominator), SEM = SD / sqrt(n),
  confidence interval of the mean = mean +/- t(1 - (1 - level)/2, n - 1) x SEM, median, quartiles,
  IQR = Q3 - Q1, minimum, maximum, sum, coefficient of variation = 100 x SD / mean, geometric mean
  = exp(mean(ln x)), skewness and excess kurtosis.
- **Quartiles.** Linear interpolation between order statistics: definition 7 of Hyndman and Fan.
  This is the default of NumPy and of R's `quantile`.
- **Skewness and kurtosis.** The adjusted Fisher-Pearson skewness G1 = g1 x sqrt(n(n - 1)) / (n - 2)
  (needs n >= 3) and the sample excess kurtosis G2 = (n - 1) / ((n - 2)(n - 3)) x ((n + 1) g2 + 6)
  (needs n >= 4; 0 for a normal distribution).
- **Implementation.** TorusPlot's own code; the t quantile from `scipy.stats.t`. All moments are
  computed on the values divided by a power of two near their largest magnitude (exact) and scaled
  back, so the summary is valid over the full double-precision range (about 1e-300 to 1e300). A
  sum beyond the largest representable number is reported as missing (None); an SD or confidence
  limit that cannot be represented stops the analysis with an error.
- **Assumptions.** None are tested. The mean, SD and CI describe roughly symmetric data well; for
  skewed data the median and quartiles are the better summary.
- **Limitations.** Other programs use other quartile definitions, so quartiles of small samples can
  differ slightly between packages. With one value the SD, SEM and CI are not defined. The geometric
  mean exists only when every value is positive. The CV is not defined when the mean is 0.
- **Validation.** NIST StRD univariate data sets NumAcc1 to NumAcc4 (certified mean and SD; the data
  follow a documented pattern and are generated in the test); the Hyndman and Fan formula written
  out; NumPy, SciPy, pandas and statsmodels for every returned quantity.
- **Primary references.** Hyndman RJ, Fan Y (1996) Sample quantiles in statistical packages.
  Am Stat 50:361-365. Joanes DN, Gill CA (1998) Comparing measures of sample skewness and kurtosis.
  The Statistician 47:183-189.

## 4. Assumption tests

### 4.1 Normality tests

Registry key: `normality`. The same Shapiro-Wilk check is run inside every parametric analysis.
On a Multiple variables table, *Groups to check* picks the numeric columns to test (each on its
own); left empty, *Variable to check* (optionally with *Split by*) is used.

| Test | What is computed | Implementation | Limits |
|---|---|---|---|
| Shapiro-Wilk | W statistic; P value by Royston's algorithm | `scipy.stats.shapiro` | Needs n >= 3. P is approximate for n > 5000. |
| D'Agostino-Pearson | K2 = Z(skewness)^2 + Z(kurtosis)^2, compared with chi-square on 2 df | TorusPlot's own code | Needs n >= 8. The kurtosis part is only roughly calibrated below n = 20. |
| Kolmogorov-Smirnov | D = largest distance between the sample distribution and a normal curve with the sample mean and SD; **Lilliefors** P value | D in TorusPlot's own code; P from `statsmodels` (`lilliefors`, table method) | Needs n >= 5. The table bounds the P value to 0.001 to 0.99; a value at a bound is shown as "P < 0.001" or "P > 0.99" and flagged (`p_limit`, `p_is_table_limit` in the rows of `extra["table"]`). |
| Anderson-Darling (optional) | A2 and the modified A\* = A2 (1 + 0.75/n + 2.25/n^2); P from the formulas of D'Agostino and Stephens | TorusPlot's own code | Needs n >= 8. |

- **Why the Lilliefors P value.** The classical Kolmogorov-Smirnov P value assumes that the mean and
  SD of the reference curve were fixed in advance. When they are estimated from the same sample the
  classical P value is far too large. TorusPlot therefore never reports it.
- **Why D'Agostino-Pearson is TorusPlot's own code.** SciPy's skewness test returns a non-zero z
  score for a perfectly symmetric sample, where the correct value is 0. The own implementation
  agrees with SciPy on generic data.
- **How a check is used.** Alpha is 0.05. P below alpha gives a FAIL badge. A test that cannot be
  run on a group (too few values, identical values) gives a CHECK badge and does not stop the analysis.
- **Limitations.** Normality tests have little power with small samples and flag trivial deviations
  in very large ones. A pass is not proof of normality. The result warns when a group has fewer
  than 8 values.
- **Q-Q plot.** Theoretical quantiles are Filliben's estimates of the normal order statistics
  (`scipy.stats.probplot`); the line is the least-squares line through the points.
- **Validation.** Published: the worked example of Shapiro and Wilk (1965), weights of 11 men,
  W = 0.79. Cross-checks: SciPy (`shapiro`, `normaltest`, `skewtest`, `kurtosistest`, `kstest`,
  `anderson`, `probplot`), statsmodels (`lilliefors`, `normal_ad`), Royston's algorithm
  re-implemented in the test, the anchor points of D'Agostino and Stephens (1986) Table 4.9.
  Simulation: type I error on normal samples and power on exponential samples, with fixed seeds.
- **Primary references.** Shapiro SS, Wilk MB (1965) An analysis of variance test for normality
  (complete samples). Biometrika 52:591-611. Royston P (1995) Remark AS R94. Appl Stat 44:547-551.
  D'Agostino RB, Pearson ES (1973) Tests for departure from normality. Biometrika 60:613-622.
  D'Agostino RB, Belanger A, D'Agostino RB Jr (1990) A suggestion for using powerful and
  informative tests of normality. Am Stat 44:316-321. Lilliefors HW (1967) On the
  Kolmogorov-Smirnov test for normality with mean and variance unknown. J Am Stat Assoc 62:399-402.
  Anderson TW, Darling DA (1952) Ann Math Stat 23:193-212. Stephens MA (1974) EDF statistics for
  goodness of fit and some comparisons. J Am Stat Assoc 69:730-737.

### 4.2 Equal-variance tests

Registry key: `equal_variance`.

| Test | What is computed | Implementation |
|---|---|---|
| Brown-Forsythe (headline) | One-way ANOVA F on z = abs(x - group median) | TorusPlot's own code |
| Levene | The same with z = abs(x - group mean) | TorusPlot's own code |
| Bartlett | T (below), referred to chi-square on k - 1 df | TorusPlot's own code |
| F test (two groups only) | F = var(a) / var(b) on (n_a - 1, n_b - 1) df, two-sided | TorusPlot's own code |

Levene and Brown-Forsythe: F = [(N - k) / (k - 1)] x sum_i n_i (zbar_i - zbar)^2 / sum_ij (z_ij - zbar_i)^2,
on (k - 1, N - k) degrees of freedom.

Bartlett: with the pooled variance s_p^2 = sum (n_i - 1) s_i^2 / (N - k),
T = [(N - k) ln s_p^2 - sum (n_i - 1) ln s_i^2] / C, C = 1 + (sum 1/(n_i - 1) - 1/(N - k)) / (3 (k - 1)).
The numerator cannot be negative; a slightly negative value from rounding is set to 0, so equal
variances give T = 0 and P = 1.

All tests here are invariant to the origin and unit of the data and are computed on rescaled data
(section 2). When every group has the same SD (to rounding) the result text says so.

- **Assumptions.** Independent observations. Bartlett's test and the F test also assume normal data
  and are very sensitive to non-normality.
- **Limitations.** Low power with small groups (the result warns below 5 values per group). When
  the deviations have no spread within the groups (for example 2 values per group) F is undefined
  and no P value is given. A non-significant result does not prove equal variances.
- **Validation.** Cross-checks with `scipy.stats.levene` (mean and median centre), `f_oneway` on
  the absolute deviations, `scipy.stats.bartlett`, Bartlett's formula written out, and a
  fixed-seed simulation of Bartlett against Brown-Forsythe on heavy-tailed data. No published
  worked example.
- **Primary references.** Levene H (1960) Robust tests for equality of variances. In: Contributions
  to Probability and Statistics, Stanford Univ Press, pp 278-292. Brown MB, Forsythe AB (1974)
  Robust tests for the equality of variances. J Am Stat Assoc 69:364-367. Bartlett MS (1937)
  Properties of sufficiency and statistical tests. Proc R Soc A 160:268-282.

## 5. Comparing two groups

All tests in this section are TorusPlot's own code. Differences are always first group minus second
group (or sample minus hypothetical value). Confidence intervals are two-sided even when the P value
is one-tailed; the result says so.

### 5.1 Unpaired t test

Registry key: `ttest_unpaired`.

- **What is computed.** t = (mean_a - mean_b) / sqrt(s_p^2 (1/n_a + 1/n_b)), with the pooled variance
  s_p^2 and df = n_a + n_b - 2. Mean difference with its CI. Cohen's d and Hedges' g (section 6).
- **Assumptions and checks.** Independent observations (stated, cannot be tested). Normality of
  each group (Shapiro-Wilk). Equal variances (Brown-Forsythe); a failure is marked FAIL and Welch's
  test is recommended.
- **Limitations.** Needs at least 2 values per group. With 2 values in a group the test has very
  little power and normality cannot be checked. If neither group has any spread, no statistic is given.
- **Validation.** Published: Student (1908), Cushny and Peebles sleep data; documented output of R's
  `t.test` for the same data. Cross-checks: SciPy `ttest_ind`, statsmodels `ttest_ind` and
  `CompareMeans.tconfint_diff`, formulas written out.
- **Primary reference.** Student (1908) The probable error of a mean. Biometrika 6:1-25. Fisher RA
  (1925) Statistical Methods for Research Workers. Oliver & Boyd.

### 5.2 Welch's t test

Registry key: `ttest_welch`.

- **What is computed.** t = (mean_a - mean_b) / sqrt(s_a^2/n_a + s_b^2/n_b), with the
  Welch-Satterthwaite degrees of freedom, which are also used for the CI of the difference.
- **Assumptions and checks.** Independent observations; normality of each group. Equal variances
  are not required (the Brown-Forsythe result is shown for information).
- **Limitations.** Cohen's d and Hedges' g are standardised by the pooled SD of the two groups, as
  for the unpaired t test, not by a Welch-specific SD. The result says so.
- **Validation.** Documented R output for the sleep data (t = -1.8608, df = 17.776); SciPy and
  statsmodels; the Welch-Satterthwaite formula written out.
- **Primary references.** Welch BL (1947) The generalization of 'Student's' problem when several
  different population variances are involved. Biometrika 34:28-35. Satterthwaite FE (1946) An
  approximate distribution of estimates of variance components. Biometrics Bull 2:110-114.

### 5.3 Paired t test

Registry key: `ttest_paired`.

- **What is computed.** A one-sample t test on the within-pair differences d = a - b:
  t = mean(d) / (SD(d) / sqrt(n)), df = n - 1. Also the Pearson correlation between the pairs (how
  effective the pairing was), Cohen's d_z and Hedges' g_z.
- **Assumptions and checks.** Independent pairs (stated). Normality of the differences
  (Shapiro-Wilk). Effective pairing: a positive correlation between the pairs. The check reports
  the one-tailed P for a positive correlation; the estimate rows show the two-tailed P of the same
  correlation. Both are labelled.
- **Limitations.** Pairs with a missing value are dropped and counted. Needs at least 2 complete pairs.
  Whether every pair has the same difference is judged at the rounding error of the input values
  (100.1 - 100 and 1.1 - 1 are the same difference); then no t, P or d_z is given.
- **Validation.** Published: Student (1908), mean difference 1.58; documented R output
  (t = -4.0621, df = 9, P = 0.002833). Cross-checks: SciPy `ttest_rel`, `pearsonr`.
- **Primary reference.** Student (1908) Biometrika 6:1-25.

### 5.4 One-sample t test

Registry key: `ttest_one_sample`.

- **What is computed.** t = (mean - hypothetical value) / (SD / sqrt(n)), df = n - 1. Mean and
  difference with CIs. Cohen's d = (mean - hypothetical value) / SD and Hedges' g.
- **Assumptions and checks.** Independent observations (stated). Normality (Shapiro-Wilk).
- **Validation.** SciPy `ttest_1samp`; statsmodels `DescrStatsW.ttest_mean`.
- **Primary reference.** Student (1908) Biometrika 6:1-25.

### 5.5 Mann-Whitney test

Registry key: `mann_whitney`.

- **What is computed.** All values are ranked together (average ranks for ties). U_A = number of
  (a, b) pairs with a > b, ties counting 1/2. The reported statistic is U = the smaller of U_A and
  U_B. Hodges-Lehmann estimate of the shift = median of all n_a x n_b pairwise differences, with a
  distribution-free CI from the order statistics of those differences. Rank-biserial r = 2 U_A /
  (n_a n_b) - 1.
- **P value.** Exact, from the null distribution of U counted by dynamic programming, when there
  are no ties and both groups have at most 50 values. With ties, exact from the permutation
  distribution of the (doubled) mid-ranks when both groups have at most 6 values
  (`EXACT_TIES_MAX_N` in `two_group.py`). Otherwise the normal approximation with tie correction
  and a continuity correction of 0.5, never reported below the smallest P value the exact test can
  attain. The result states which.
- **Assumptions and checks.** Independent observations; values that can be ranked. Similar
  distribution shape (Brown-Forsythe test of spread) is needed only to read the result as a
  difference in medians.
- **Limitations.** The test compares distributions (mean ranks). It is a test of medians only when
  both distributions have the same shape. With very small groups the smallest possible exact P
  value can exceed alpha (0.10 for 3 against 3); the result warns. The Hodges-Lehmann estimate is
  skipped above 10,000,000 pairs.
- **Validation.** Documented R output for the sleep data (W = 25.5, P = 0.06933). Cross-checks:
  SciPy `mannwhitneyu` (exact and asymptotic), 800 random data sets with ties, brute-force
  enumeration of the exact null distribution and of the tied permutation P value, brute-force
  Hodges-Lehmann estimates, coverage
  simulations. Published table: critical U for n1 = n2 = 10.
- **Primary references.** Mann HB, Whitney DR (1947) On a test of whether one of two random
  variables is stochastically larger than the other. Ann Math Stat 18:50-60. Hodges JL, Lehmann EL
  (1963) Estimates of location based on rank tests. Ann Math Stat 34:598-611. Bauer DF (1972)
  Constructing confidence sets using rank statistics. J Am Stat Assoc 67:687-690.

### 5.6 Wilcoxon signed-rank tests

Registry keys: `wilcoxon_signed_rank` (matched pairs) and `wilcoxon_one_sample`.

- **What is computed.** The differences d (a - b, or x - hypothetical value) are ranked by absolute
  size. W+ and W- are the rank sums of the positive and negative differences. The reported
  statistic is W = W+ - W-. Hodges-Lehmann pseudo-median = median of all Walsh averages
  (d_i + d_j) / 2, with a distribution-free CI. Matched-pairs rank-biserial r = (W+ - W-) / (W+ + W-).
- **Zero differences and ties.** Differences are snapped at the rounding error of the input values:
  one within its tolerance of zero counts as zero, and absolute differences that agree within their
  tolerances count as ties (so 5.4 - 5.1 and 5.2 - 4.9 are tied). `wilcox` (default) discards
  zeros. `pratt` ranks them with the others and then drops their ranks.
- **P value.** Exact when no two non-zero differences have the same size and there are at most 50
  of them. With ties, exact over all 2^n sign patterns of the (doubled) mid-ranks when there are at
  most 6 non-zero differences. With Pratt's method and zeros, the exact P value is conditional on
  the ranks of the non-zero differences. Otherwise the normal approximation with tie and continuity
  correction, never reported below the smallest attainable P value 2 / 2^n (1 / 2^n one-tailed).
- **Assumptions and checks.** Independent pairs (or observations). The differences are distributed
  symmetrically around their median (stated; the sample skewness is reported).
- **Limitations.** With n non-zero differences the smallest two-sided exact P is 2 / 2^n, so fewer
  than 6 cannot reach 0.05; the result warns when the smallest attainable P exceeds the chosen
  alpha. The CI of the rank-biserial r is a score interval (section 6); when r is exactly +1 or -1
  it is one-sided with positive width.
- **Validation.** Documented R output for the sleep data (V = 0, P = 0.009091). Cross-checks: SciPy
  `wilcoxon` (exact; asymptotic with correction; both zero methods), brute-force null distributions,
  brute-force enumeration of the tied sign-pattern P value, published table: critical T for n = 11.
- **Primary references.** Wilcoxon F (1945) Individual comparisons by ranking methods. Biometrics
  Bull 1:80-83. Pratt JW (1959) Remarks on zeros and ties in the Wilcoxon signed rank procedures.
  J Am Stat Assoc 54:655-667. Hodges and Lehmann (1963); Bauer (1972), as above.

## 6. Effect sizes and their confidence intervals

All effect sizes are TorusPlot's own code (`torusplot/stats/effect_sizes.py`).

| Effect size | Formula | Confidence interval |
|---|---|---|
| Cohen's d (two groups) | (mean_a - mean_b) / pooled SD | Noncentral t: the noncentrality parameters that put the observed t at the upper and lower tails are converted to d |
| Hedges' g | d x J(df), with the exact correction J(df) = Gamma(df/2) / (sqrt(df/2) Gamma((df - 1)/2)) | The noncentral-t interval for the population delta, the same as for d: J applies to the point estimate only |
| Cohen's d_z (paired) | mean of differences / SD of differences | Noncentral t |
| Cohen's d (one sample) | (mean - hypothetical value) / SD | Noncentral t |
| Eta squared | SS_effect / SS_total | Noncentral F (one-way designs): eta squared = ncp / (ncp + df1 + df2 + 1) |
| Partial eta squared | SS_effect / (SS_effect + SS_error) = F df1 / (F df1 + df2) | Noncentral F |
| Omega squared | (SS_effect - df_effect x MS_error) / (SS_total + MS_error); negative estimates are reported as 0 | Noncentral F interval for the population value |
| Partial omega squared | df1 (F - 1) / (df1 (F - 1) + N) | Noncentral F |
| Generalized eta squared (repeated measures) | SS_effect / (SS_effect + all subject and error SS) | None |
| Cohen's f | sqrt(eta squared / (1 - eta squared)) | None (used for power) |
| Rank-biserial r (two groups) | 2 U_A / (n_a n_b) - 1 | Score interval for theta = (r + 1) / 2 (Newcombe 2006, method 5: Hanley-McNeil variance evaluated at the limits), with the variance multiplied by max(1, V_placement / V_model), V_placement the distribution-free placement-value variance (Sen 1960; DeLong et al. 1988); mapped back to r |
| Matched-pairs rank-biserial r | (W+ - W-) / (W+ + W-) | Score interval from the exact mean and variance of W+ / (n (n + 1) / 2) under a normal-shift working model (Hollander & Wolfe 1999), evaluated at the limits, with the variance multiplied by max(1, V_jackknife / V_model) |
| Rank epsilon squared (Kruskal-Wallis) | H / (N - 1) | None |
| Kendall's W (Friedman) | chi-square / (n (k - 1)) | None |
| Pearson r, Spearman rho | Section 11 | Fisher z |
| phi, Cramér's V | Section 15 | Noncentral chi-square for V |
| Hazard ratio, odds ratio, relative risk | Sections 14 and 15 | Log-scale normal intervals |

- **Magnitude labels.** The words small, medium and large follow Cohen (1988): d 0.2 / 0.5 / 0.8;
  eta-squared-type measures 0.01 / 0.06 / 0.14; correlation-type measures 0.1 / 0.3 / 0.5. They are
  conventions from the behavioural sciences. In a well-controlled bench experiment effects are often
  far larger, and the label "large" then says little.
- **A point to know about eta-squared intervals.** The F test is one-sided, so the 95% noncentral-F
  interval corresponds to a test at alpha = 0.025. A 95% interval can therefore start at 0 although
  the F test has P slightly below 0.05.
- **After Welch-type tests** eta squared is given as a description of the sample only, without an
  interval, because the usual interval assumes equal variances.
- **Limitations.** Hedges' correction does not exist for 1 degree of freedom. Noncentral intervals
  assume normal data. d and g are not given (None) when the SD is zero to rounding error. The
  rank-biserial intervals are approximate. Both respect [-1, 1] and, with complete separation
  (r = +1 or -1), are one-sided with positive width. In simulations the 95% interval for two groups
  covered the true value in about 93 to 99% of samples (n = 3 to 100 per group), dropping to about
  90% with only 3 or 4 values per group, or with very unequal variances together with unequal group
  sizes. The matched-pairs interval covered about 93 to 99% (n = 5 to 100), dropping to about 90%
  for heavy-tailed differences with very strong effects.
- **Validation.** `tests/stats/test_effect_sizes.py`: inversion of `scipy.stats.nct` and
  `scipy.stats.ncf` verified at both limits; definitions written out; cross-checks inside the tests
  of each analysis.
- **Primary references.** Cohen J (1988) Statistical Power Analysis for the Behavioral Sciences,
  2nd ed. Hedges LV (1981) Distribution theory for Glass's estimator of effect size and related
  estimators. J Educ Stat 6:107-128. Steiger JH, Fouladi RT (1997) Noncentrality interval estimation
  and the evaluation of statistical models. In: What if there were no significance tests? Cumming G,
  Finch S (2001) A primer on the understanding, use, and calculation of confidence intervals that
  are based on central and noncentral distributions. Educ Psychol Meas 61:532-574. Steiger JH (2004)
  Beyond the F test. Psychol Methods 9:164-182. Olejnik S, Algina J (2003) Generalized eta and omega
  squared statistics. Psychol Methods 8:434-447. Hays WL (1963) Statistics for Psychologists.
  Kerby DS (2014) The simple difference formula. Compr Psychol 3:11.IT.3.1. Cureton EE (1956)
  Rank-biserial correlation. Psychometrika 21:287-290. Newcombe RG (2006) Confidence intervals for
  an effect size measure based on the Mann-Whitney statistic. Part 1: general issues and
  tail-area-based methods; Part 2: asymptotic methods and evaluation. Stat Med 25:543-557 and
  559-573. Hanley JA, McNeil BJ (1982) The meaning and use of the area under a receiver operating
  characteristic (ROC) curve. Radiology 143:29-36. DeLong ER, DeLong DM, Clarke-Pearson DL (1988)
  Comparing the areas under two or more correlated receiver operating characteristic curves: a
  nonparametric approach. Biometrics 44:837-845. Sen PK (1960) On some convergence properties of
  U-statistics. Calcutta Stat Assoc Bull 10:1-18. Hollander M, Wolfe DA (1999) Nonparametric
  Statistical Methods, 2nd ed. Wiley.

## 7. Comparing three or more groups

All tests in this section are TorusPlot's own code (`torusplot/stats/oneway.py`). The inline
assumption checks are `scipy.stats.shapiro` plus TorusPlot's own Brown-Forsythe and Bartlett tests
(section 4.2).

### 7.1 One-way ANOVA

Registry key: `anova_oneway`.

- **What is computed.** F = MS_between / MS_within on (k - 1, N - k) degrees of freedom. F, P and
  the effect sizes are computed on rescaled data (section 2); the sums of squares are computed there
  on data centred on the grand mean with exactly rounded summation, so large offsets with small
  variances do not destroy the result. Sums of squares and mean squares are scaled back to the units
  of the data and shown as missing, with a warning, when they fall outside the representable range
  (for example values near 1e200 or 1e-200). Post-hoc estimates, SEs and CIs are computed on the
  data divided by a power of two and scaled back. Eta squared and omega squared with
  noncentral-F intervals. Follow-up comparisons: Tukey (default), Dunnett, Games-Howell, or
  pooled-SD t tests with Bonferroni, Sidak, Holm, Holm-Sidak, Benjamini-Hochberg or no adjustment
  (Fisher's LSD).
- **Assumptions and checks.** Independent observations (stated). Normality: Shapiro-Wilk on the
  pooled residuals (primary) and on each group. Equal variances: Brown-Forsythe and Bartlett.
- **Limitations.** Needs replicates: if every group has one value the test is refused. A group with
  one value makes the result lean entirely on the equal-variance assumption (warning). With two
  groups the test equals the unpaired t test.
- **Validation.** NIST StRD one-way ANOVA, all 11 data sets (SiRstv, SmLs01 to SmLs09, AtmWtAg):
  degrees of freedom, sums of squares, mean squares, F, R squared, residual SD. Agreement is 13 to 15
  digits on the lower-difficulty sets, about 10 on the average ones and about 4 on the three
  higher-difficulty ones. That last limit is set by double-precision storage of the input (values
  near 1e12 with a scatter of 0.1), not by the algorithm; a second test shows 14 digits once the
  constant offset is removed in exact decimal arithmetic. Published: Dobson (1983) plant growth data
  (F = 4.8461). Cross-checks: SciPy `f_oneway`, statsmodels `anova_lm`, exact rational arithmetic.
- **Primary reference.** Fisher RA (1925) Statistical Methods for Research Workers. Oliver & Boyd.

### 7.2 Welch's ANOVA

Registry key: `welch_anova`.

- **What is computed.** With weights w_i = n_i / s_i^2, W = sum w_i and the weighted grand mean
  xw = sum w_i xbar_i / W:
  F = [sum w_i (xbar_i - xw)^2 / (k - 1)] / [1 + 2 (k - 2) / (k^2 - 1) x L],
  L = sum (1 - w_i / W)^2 / (n_i - 1), df1 = k - 1, df2 = (k^2 - 1) / (3 L).
  Follow-up: Games-Howell (default) or Welch t tests with a P value adjustment.
- **Assumptions and checks.** Independent observations; normality (as for the one-way ANOVA).
  Equal variances are not required.
- **Limitations.** Every group needs at least 2 values and a non-zero variance; with a constant
  group the statistic is undefined. Eta squared is descriptive only.
- **Validation.** statsmodels `anova_oneway(use_var="unequal", welch_correction=True)`; Welch's
  formula written out; Welch's t test for two groups. The R output for the plant growth data
  (F = 5.181) was typed from memory and counts as a cross-implementation check, not a publication.
- **Primary reference.** Welch BL (1951) On the comparison of several mean values: an alternative
  approach. Biometrika 38:330-336.

### 7.3 Brown-Forsythe ANOVA

Registry key: `brown_forsythe_anova`.

- **What is computed.** F\* = SS_between / sum (1 - n_i/N) s_i^2, df1 = k - 1, and the Satterthwaite
  df2 = 1 / sum [c_i^2 / (n_i - 1)] with c_i = (1 - n_i/N) s_i^2 / sum (1 - n_j/N) s_j^2.
  Mehrotra's (1997) corrected numerator df and its P value are stored with the result.
- **Assumptions and checks.** As for Welch's ANOVA.
- **Limitations.** This is the test of means. It is not the Brown-Forsythe test of equal variances.
  No published worked example is used.
- **Validation.** statsmodels `anova_oneway(use_var="bf")` (statistic, denominator df, both P
  values); the formula written out.
- **Primary references.** Brown MB, Forsythe AB (1974) The small sample behavior of some statistics
  which test the equality of several means. Technometrics 16:129-132. Mehrotra DV (1997) Improving
  the Brown-Forsythe solution to the generalized Behrens-Fisher problem. Commun Stat Simul Comput
  26:1139-1145.

### 7.4 Kruskal-Wallis test

Registry key: `kruskal_wallis`.

- **What is computed.** All values are ranked together.
  H = [12 / (N (N + 1))] sum n_i (Rbar_i - (N + 1)/2)^2 / C, with the tie correction
  C = 1 - sum (t^3 - t) / (N^3 - N). H is referred to chi-square on k - 1 df. Rank epsilon squared.
  Follow-up: Dunn's test.
- **Assumptions and checks.** Independent observations; rankable values. Similar shape and spread
  (Brown-Forsythe) are needed to read the result as a shift in typical values.
- **Limitations.** The P value is a chi-square approximation. It is rough when every group has 5 or
  fewer values (warning). **There is no exact Kruskal-Wallis P value.**
- **Validation.** Published: Hollander and Wolfe (1973) mucociliary efficiency data
  (chi-squared = 0.77143); New York ozone by month (chi-squared = 29.267, with missing values).
  Cross-check: SciPy `kruskal`; Mann-Whitney for two groups.
- **Primary reference.** Kruskal WH, Wallis WA (1952) Use of ranks in one-criterion variance
  analysis. J Am Stat Assoc 47:583-621.

### 7.5 Friedman test

Registry key: `friedman`.

- **What is computed.** Values are ranked within each subject.
  Q = 12 sum_j (R_j - n (k + 1)/2)^2 / [n k (k + 1) - sum_rows sum (t^3 - t) / (k - 1)],
  referred to chi-square on k - 1 df. Kendall's W = Q / (n (k - 1)). The Iman-Davenport F version
  is stored with the result. Follow-up: Dunn-type rank-sum comparisons or Wilcoxon tests.
- **Assumptions and checks.** Independent subjects; complete blocks (each subject measured under
  every condition); rankable values.
- **Limitations.** Subjects with a missing value are left out. The chi-square approximation is
  rough with few subjects (warning). There is no exact Friedman P value. Needs at least 3 conditions
  and 2 complete subjects.
- **Validation.** Published: Hollander and Wolfe (1973) rounding-first-base data
  (chi-squared = 11.143). Cross-check: SciPy `friedmanchisquare`; the Iman-Davenport formula.
- **Primary references.** Friedman M (1937) The use of ranks to avoid the assumption of normality
  implicit in the analysis of variance. J Am Stat Assoc 32:675-701. Kendall MG, Babington Smith B
  (1939) The problem of m rankings. Ann Math Stat 10:275-287. Iman RL, Davenport JM (1980)
  Approximations of the critical region of the Friedman statistic. Commun Stat Theory Methods
  9:571-595.

## 8. Multiple-comparison procedures

All procedures are TorusPlot's own code (`torusplot/stats/posthoc.py`, `multiplicity.py`), written
from the primary papers. In every comparison the estimate is mean(A) - mean(B) (a difference of
mean ranks for rank procedures).

### 8.1 Procedures with their own distribution

| Procedure | What is computed | Limits and notes |
|---|---|---|
| Tukey HSD (Tukey-Kramer for unequal n) | q = abs(mean_i - mean_j) / sqrt(MSE/2 x (1/n_i + 1/n_j)), referred to the studentized range with k groups and the error df. Simultaneous intervals: difference +/- q_crit / sqrt(2) x SE. | Assumes equal variances. The studentized range comes from `scipy.stats.studentized_range`. |
| Dunnett | t_i = (mean_i - mean_control) / sqrt(MSE (1/n_i + 1/n_control)). The adjusted P is the probability that the largest of the k - 1 statistics exceeds the observed one. | The multivariate t integral is evaluated by fixed Gauss-Legendre quadrature, so the result is the same on every run. SciPy's own Dunnett routine uses randomised integration and is used only as a cross-check. |
| Games-Howell | SE = sqrt(s_i^2/n_i + s_j^2/n_j) with Welch-Satterthwaite df; q = sqrt(2) abs(diff) / SE is referred to the studentized range with k groups and that df. | No equal-variance assumption. Every group needs at least 2 values. |
| Dunn (after Kruskal-Wallis) | z = (Rbar_i - Rbar_j) / sqrt((N(N + 1)/12 - sum(t^3 - t) / (12 (N - 1))) (1/n_i + 1/n_j)); two-sided normal P, then a P value adjustment. | No confidence interval: the standard error holds only under the null hypothesis. |
| Rank-sum comparisons after Friedman | z = (R_i - R_j) / sqrt(n k (k + 1) / 6) on within-subject rank sums. | No tie correction, which makes the comparisons slightly conservative when there are ties within subjects. No confidence interval. |
| Wilcoxon comparisons after Friedman | A Wilcoxon signed-rank test per pair, by the same engine as the two-group Wilcoxon matched-pairs test (section 5.6: zero differences discarded, ties snapped at the input rounding error, exact / exact with ties / normal approximation with tie and continuity correction), then a P value adjustment. Statistic W = W+ - W-. | The estimate is the Hodges-Lehmann median of the paired differences. |
| Pairwise t tests | Pooled error variance of all groups (after ANOVA), separate variances (Welch), or paired differences (repeated measures), then a P value adjustment. With no adjustment and pooled variance this is Fisher's LSD. | See 8.2 for the intervals. Paired: when the differences are constant to the input rounding error the SE is 0, giving t = infinity and P = 0 (no test if the difference is 0); a warning says so. |

### 8.2 P value adjustments

With m comparisons and ordered raw P values p(1) <= ... <= p(m):

| Method | Adjusted P | Controls |
|---|---|---|
| Bonferroni | min(1, m p) | Family-wise error rate |
| Sidak | 1 - (1 - p)^m | Family-wise error rate (assumes independent or positively dependent tests) |
| Holm | Step-down: max over j <= i of min(1, (m - j + 1) p(j)) | Family-wise error rate; never less powerful than Bonferroni |
| Holm-Sidak | Step-down with 1 - (1 - p(j))^(m - j + 1) | Family-wise error rate |
| Benjamini-Hochberg (`fdr_bh`) | Step-up: min over j >= i of m p(j) / j | False discovery rate |
| none | p | Nothing |

- **Confidence intervals.** Bonferroni and Sidak intervals are simultaneous (the per-comparison
  alpha is adjusted to match). Step-wise and false-discovery-rate methods have no matching interval.
  After a one-way ANOVA their intervals are unadjusted and a warning says so. In factorial and
  repeated-measures designs Holm uses the Bonferroni critical value and Holm-Sidak the Sidak
  critical value (conservative companions); `fdr_bh` and `none` give unadjusted intervals.
- **Missing P values** (comparisons that could not be tested) are not counted in m.
- **What counts as one family.** After a one-way test: all pairs, or all comparisons with the
  control. In factorial designs with Tukey: each level of the conditioning factor is its own family;
  with the other methods all requested comparisons form one family. The result states the family.
- **Validation.** Published: Tukey limits and adjusted P for the plant growth and warp-breaks data
  (documented R output); Dunnett's critical values from Dunnett (1955, 1964), all 20 tabulated
  values reproduced after rounding to 2 decimals; studentized-range 5% points from Harter (1960).
  Cross-checks: SciPy `tukey_hsd` and `dunnett`, statsmodels `pairwise_tukeyhsd` and
  `multipletests`, scikit-posthocs (`posthoc_games_howell`, `posthoc_dunn`,
  `posthoc_siegel_friedman`, `posthoc_ttest`), an independent adaptive quadrature of Dunnett's
  integral. No published worked example is used for Games-Howell or Dunn's test.
- **Primary references.** Tukey JW (1949) Comparing individual means in the analysis of variance.
  Biometrics 5:99-114. Kramer CY (1956) Extension of multiple range tests to group means with
  unequal numbers of replications. Biometrics 12:307-310. Dunnett CW (1955) A multiple comparison
  procedure for comparing several treatments with a control. J Am Stat Assoc 50:1096-1121. Dunnett
  CW (1964) New tables for multiple comparisons with a control. Biometrics 20:482-491. Games PA,
  Howell JF (1976) Pairwise multiple comparison procedures with unequal n's and/or variances: a
  Monte Carlo study. J Educ Stat 1:113-125. Fisher RA (1935) The Design of Experiments. Dunn OJ
  (1961) Multiple comparisons among means. J Am Stat Assoc 56:52-64. Dunn OJ (1964) Multiple
  comparisons using rank sums. Technometrics 6:241-252. Siegel S, Castellan NJ (1988) Nonparametric
  Statistics for the Behavioral Sciences, 2nd ed. Sidak Z (1967) J Am Stat Assoc 62:626-633. Holm S
  (1979) A simple sequentially rejective multiple test procedure. Scand J Stat 6:65-70. Benjamini Y,
  Hochberg Y (1995) J R Stat Soc B 57:289-300.

## 9. Factorial ANOVA and ANCOVA

Registry keys: `anova_twoway`, `anova_threeway`, `ancova`.

- **What is computed.** An ordinary least-squares linear model with sum-to-zero coding for every
  factor. The ANOVA table has one row per main effect and interaction, with the requested type of
  sums of squares (section 20). Partial eta squared and partial omega squared per term. Follow-up
  comparisons use the residual mean square and residual df of the whole model.
- **Implementation.** The fit is statsmodels OLS; the table is `anova_lm`. When the residual
  variation is exactly zero the same sums of squares are obtained by comparing nested least-squares
  fits in TorusPlot's own code. Contrasts, Tukey P values, intervals and effect sizes are TorusPlot's
  own code.
- **Comparisons.** `simple_b_within_a` and the like compare the levels of one factor within each
  level of the others; `main_a` and the like compare estimated marginal (least-squares) means: the
  fitted model's cell predictions averaged with equal weights, with standard errors from the model's
  coefficient covariance, so in a model without interaction the comparisons follow the additive fit;
  `all_cells` compares every cell with every other. Tukey: P = P(Q[k, df] > abs(t) sqrt(2)),
  with k the number of means in one family. The standardized effect of each comparison is
  Hedges' g = difference / sqrt(MS residual) x J(df), with a noncentral-t interval.
- **ANCOVA.** Model: value = group effect + common slope x covariate + error. Reported: the group
  effect adjusted for the covariates, the slopes, and the adjusted (least-squares) means at the mean
  of each covariate. The equal-slopes assumption is tested with a separate model that adds the
  group x covariate interaction. The Tukey-Kramer adjustment of adjusted means is exact only when the
  covariate means are equal across groups; otherwise it is an approximation, usually slightly
  conservative, and the result label says "approximate".
- **Assumptions and checks.** Independent observations (stated). Normality of the residuals
  (Shapiro-Wilk). Equal variances across cells (Brown-Forsythe). Balance and replication are
  reported. ANCOVA adds: equal slopes (tested), a linear relation with the covariate (stated), and a
  covariate that is not affected by the treatment (stated).
- **Limitations.** Every cell needs data. With one value per cell the highest-order interaction
  cannot be separated from error and is dropped, with a warning; the tests are then valid only if
  that interaction is truly absent. In unbalanced designs the sums of squares do not add up to the
  total. Type II main effects assume no interaction; a warning appears when one is significant.
  The models are fixed-effects models. There is no rank-based two-way test.
- **Validation.** Published: Montgomery, battery-life 3 x 3 experiment (sums of squares, F = 7.91,
  28.97, 3.56). Constructed balanced 2 x 3 data with exact sums of squares. Cross-checks:
  statsmodels `anova_lm` (Type I, II, III) on a hand-written formula, `pairwise_tukeyhsd`,
  `multipletests`, `t_test` on coefficient contrasts; classical balanced formulas; nested fits of a
  hand-built design matrix. **No published worked example is used for the three-way ANOVA or for
  ANCOVA.**
- **Primary references.** Fisher RA (1925) Statistical Methods for Research Workers. Oliver & Boyd.
  Yates F (1934) The analysis of multiple classifications with unequal numbers in the different
  classes. J Am Stat Assoc 29:51-66. Searle SR (1987) Linear Models for Unbalanced Data. Wiley.
  Cochran WG (1957) Analysis of covariance: its nature and uses. Biometrics 13:261-281.

## 10. Repeated measures

Registry keys: `anova_rm_oneway`, `anova_mixed`, `anova_rm_twoway`. (`friedman` is in section 7.5.)

All of this is TorusPlot's own code (`torusplot/stats/repeated.py`), written from the
sums-of-squares and contrast formulation.

- **What is computed.** The data form a matrix Y with one row per subject and one column per
  condition. For a within-subject effect with q degrees of freedom, Y is multiplied by a matrix C of
  orthonormal contrasts: Z = Y C. With hypothesis and error matrices H and E of Z, the classical F
  test uses SS_effect = trace(H) and SS_error = trace(E).
  - One-way: the effect tests that the mean of Z is zero.
  - Mixed design: the between-subjects factor is a one-way ANOVA of the subject averages, tested
    against subjects within groups. The within factor and the interaction are tested against the
    within-subject residual.
  - Two-way repeated measures: each effect (A, B, A x B) is tested against its own interaction
    with subjects.
- **Effect sizes.** Partial eta squared (with a noncentral-F interval) and generalized eta squared.
- **Follow-up tests.** One-way: paired t tests, each with its own error term, so they do not depend
  on sphericity; Holm adjustment by default. Mixed: groups at each repeated level, repeated levels
  within each group, or main-effect comparisons. The two-way repeated-measures analysis produces no
  follow-up comparisons.

### 10.1 Sphericity: Mauchly's test and the corrections

Sphericity means that the differences between every pair of conditions have the same variance.
All quantities come from S = E / df_e, the covariance matrix of the orthonormal contrasts.

| Quantity | Formula |
|---|---|
| Mauchly's W | det(S) / (trace(S) / q)^q |
| Chi-square approximation | X2 = -rho x df_e x ln W, rho = 1 - (2 q^2 + q + 2) / (6 q df_e), on q (q + 1)/2 - 1 df |
| Mauchly P value | P(chi2_f > X2) + w2 x (P(chi2_(f+4) > X2) - P(chi2_f > X2)): the second-order term of Box's expansion |
| Greenhouse-Geisser epsilon | trace(S)^2 / (q x trace(S^2)); lower bound 1/q |
| Huynh-Feldt epsilon | (N q eps - 2) / (q (N - g - q eps)), capped at 1 (N subjects in g groups) |
| Lecoutre's corrected Huynh-Feldt epsilon | ((N - g + 1) q eps - 2) / (q (N - g - q eps)); stored, not the headline |
| Corrected P value | Both degrees of freedom of the F test are multiplied by epsilon |

- **Default.** With more than two repeated levels the headline P value is Greenhouse-Geisser
  corrected (`correction = auto`). The table always lists the uncorrected, Greenhouse-Geisser,
  Huynh-Feldt and lower-bound P values. With two levels no correction is needed or possible.
- **Why not decide from Mauchly's test.** Mauchly's test has little power in small samples and often
  misses violations. Always correcting costs very little when sphericity holds.
- **Why the second-order Mauchly P value.** It reproduces the published textbook value for the
  repeated-measures example in the test-suite (0.047; the first-order formula gives 0.044) and is
  better calibrated in small samples. The first-order value is stored with the result.
- **Assumptions and checks.** Independent subjects (stated). Sphericity (Mauchly). Normality of the
  residuals (Shapiro-Wilk), and of the paired differences behind the follow-up tests. Mixed designs:
  equal variances between groups at each level (Brown-Forsythe).
- **Limitations.** Subjects with a missing value are dropped completely (complete-case analysis).
  There is no mixed-effects model that keeps incomplete subjects. With very few subjects sphericity
  cannot be tested (warning). With unequal group sizes in a mixed design, Type III gives every group
  equal weight for the within factor and Type II weights groups by size. Designs with more than one
  between factor or more than two within factors are not supported.
- **Validation.** Constructed 8 subjects x 4 conditions with violated sphericity: F, Mauchly W and
  chi-square, Greenhouse-Geisser and Huynh-Feldt epsilons and corrected P values agree with
  first-principles formulas. (An earlier textbook example was reproduced exactly, F(3, 21) = 3.79,
  W = 0.136, P = 0.047, epsilons 0.533 / 0.666, then removed for licensing reasons, D-030.)
  Cross-checks: statsmodels `AnovaRM` (one and two within factors), `mixedlm` for the balanced mixed
  design (agreement to about 1e-3, limited by the optimiser), `anova_lm` on contrast scores for the
  unbalanced mixed design, SciPy `ttest_rel` and `ttest_ind`. First principles: textbook
  sums of squares; epsilon from the raw covariance matrix; a fixed-seed simulation of the Mauchly P
  value under sphericity. **No published worked example is used for the mixed design or the two-way
  repeated-measures ANOVA.**
- **Primary references.** Mauchly JW (1940) Significance test for sphericity of a normal n-variate
  distribution. Ann Math Stat 11:204-209. Box GEP (1949) A general distribution theory for a class of
  likelihood criteria. Biometrika 36:317-346. Anderson TW (2003) An Introduction to Multivariate
  Statistical Analysis, 3rd ed., section 10.7. Wiley. Box GEP (1954) Some theorems on quadratic
  forms applied in the study of analysis of variance problems, II. Ann Math Stat 25:484-498.
  Greenhouse SW, Geisser S (1959) On methods in the analysis of profile data. Psychometrika
  24:95-112. Huynh H, Feldt LS (1976) Estimation of the Box correction for degrees of freedom from
  sample data in randomized block and split-plot designs. J Educ Stat 1:69-82. Lecoutre B (1991)
  A correction for the epsilon-tilde approximate test in repeated measures designs with two or more
  independent groups. J Educ Stat 16:371-372. Winer BJ, Brown DR, Michels KM (1991) Statistical
  Principles in Experimental Design, 3rd ed. McGraw-Hill. Bakeman R (2005) Recommended effect size
  statistics for repeated measures designs. Behav Res Methods 37:379-384.

## 11. Correlation

Registry keys: `correlation`, `correlation_matrix`. TorusPlot's own code.

- **Pearson.** r = S_xy / sqrt(S_xx S_yy). t = r sqrt((n - 2) / (1 - r^2)) on n - 2 df.
  CI: tanh(atanh(r) +/- z / sqrt(n - 3)) (Fisher z). With a one-sided alternative the interval is
  one-sided too.
- **Spearman.** Pearson's r computed on average ranks, so ties are handled. P value: the exact
  permutation distribution when n <= 9 and there are no ties; otherwise the t approximation on
  n - 2 df. CI: Fisher z with the Bonett and Wright standard error sqrt((1 + rho^2 / 2) / (n - 3)).
- **Matrix.** Every pair of numeric variables, each on the rows where both values are present
  (pairwise-complete), so n can differ between pairs. Two-sided P values (t approximation for
  Spearman), optionally adjusted across the unique pairs (section 8.2).
- **Assumptions and checks.** Independent pairs (stated). For Pearson: normality of each variable
  (Shapiro-Wilk), a straight-line relation (stated), and a leave-one-out check that no single point
  drives r.
- **Limitations.** A CI needs at least 4 pairs. With 2 pairs r is always +1 or -1 and nothing can be
  concluded. With XY tables every replicate is one point unless *Use the mean of replicates* is
  ticked; repeat readings of one sample are then pseudo-replicates (warning). Correlation does not
  show cause. An unadjusted matrix of many pairs produces chance findings (warning).
- **Validation.** Published: Anscombe's quartet (r = 0.816 for all four sets, to the precision of
  the paper). Cross-checks: SciPy `pearsonr` (with CI) and `spearmanr`; pandas `corr`; the exact
  Spearman P by brute-force enumeration of all permutations; the Bonett-Wright interval from its
  formula.
- **Primary references.** Pearson K (1896) Philos Trans R Soc A 187:253-318. Fisher RA (1915)
  Biometrika 10:507-521. Fisher RA (1921) Metron 1:3-32. Spearman C (1904) Am J Psychol 15:72-101.
  Bonett DG, Wright TA (2000) Psychometrika 65:23-28. Kendall MG, Gibbons JD (1990) Rank Correlation
  Methods, 5th ed.

## 12. Linear regression

### 12.1 Simple linear regression

Registry key: `linear_regression`. Closed-form ordinary least squares in TorusPlot's own code.

- **What is computed.** slope = S_xy / S_xx; intercept = mean(y) - slope x mean(x);
  s^2 = SS_res / (n - 2); SE(slope) = s / sqrt(S_xx); SE(intercept) = s sqrt(1/n + mean(x)^2 / S_xx).
  The slope is tested against 0 (or a value you give). R squared with a noncentral-F interval.
  Confidence band of the line at x0: yhat +/- t s sqrt(1/n + (x0 - mean(x))^2 / S_xx); the
  prediction band adds 1 under the root. X-intercept = -intercept / slope with a Fieller interval.
  With *Force the line through the origin* the model is Y = slope x X (n - 1 df, uncentred R squared).
- **Assumptions and checks.** Normality of residuals (Shapiro-Wilk). Constant scatter (Koenker's
  studentized Breusch-Pagan test). Model adequacy (lack-of-fit F test, when there are replicates).
  No systematic deviation (runs test). Stated: a linear relation, independent observations, X
  measured without error.
- **Limitations.** By default every replicate is one point; that is right only for independent
  samples. With 2 points the line fits exactly and no standard errors exist. Predictions outside the
  X range are extrapolations.
- **Validation.** NIST StRD Norris (original file): slope, intercept, both standard errors, residual
  SD, R squared, F, to about 1e-13. Published: Anscombe's quartet. Cross-checks: SciPy `linregress`;
  statsmodels OLS including `get_prediction` for both bands; first principles for the Fieller interval.
- **Primary references.** Draper NR, Smith H (1998) Applied Regression Analysis, 3rd ed., Wiley.
  Fieller EC (1954) J R Stat Soc B 16:175-185.

### 12.2 Comparing slopes

Registry key: `compare_slopes`. TorusPlot's own code.

- **What is computed.** Three nested models are fitted: (1) a separate slope and intercept per data
  set (N - 2k df); (2) one common slope, separate intercepts (N - k - 1 df); (3) one line for all
  data. Slopes: F = [(SS_common - SS_sep) / (k - 1)] / [SS_sep / (N - 2k)].
  Elevations: F = [(SS_one - SS_common) / (k - 1)] / [SS_common / (N - k - 1)].
- **Assumptions.** Each data set is a straight line; similar scatter in every data set; independent
  observations; normal residuals (Shapiro-Wilk).
- **Limitations.** The elevation test is meaningful only when the slopes do not differ.
- **Validation.** statsmodels formula models `y ~ x*g`, `y ~ x+g`, `y ~ x` with `anova_lm`, for 2
  and 3 groups with unequal n. No published worked example.
- **Primary reference.** Zar JH (2010) Biostatistical Analysis, 5th ed., ch. 18. Draper and Smith
  (1998), extra-sum-of-squares principle.

### 12.3 Multiple linear regression

Registry key: `multiple_regression`.

- **What is computed.** Y = b0 + b1 X1 + b2 X2 + ... by ordinary least squares. Coefficients with
  standard errors, t tests and CIs; overall F test; R squared, adjusted R squared and Cohen's f
  squared; optional standardized coefficients (b x SD(x) / SD(y)); variance inflation factors
  (1 / (1 - R_j^2)).
- **Implementation.** The fit is statsmodels OLS. The design matrix (dummy coding, interaction
  columns), VIFs, standardized coefficients and diagnostics are TorusPlot's own code.
- **Categorical predictors.** Text columns are dummy-coded. The reference level is the first level
  in alphabetical order. Interactions are product columns for the pairs you name.
- **Assumptions and checks.** Normal residuals (Shapiro-Wilk), constant scatter (Breusch-Pagan),
  no multicollinearity (VIF above 5 is a caution, above 10 a problem). Stated: linear and additive
  effects, independent observations.
- **Limitations.** VIFs are per design column; there is no generalised VIF for factors with several
  levels. A warning appears with fewer than about 10 rows per predictor. No variable selection, no
  regularisation, no logistic or Poisson regression.
- **Validation.** Longley data: all 7 coefficients and standard errors, residual SD, R squared, F;
  the VIF check must flag multicollinearity. **The Longley reference values were taken from the
  statsmodels test-suite, which cites NIST; they were not compared with the NIST file itself.**
  Cross-checks: statsmodels `formula.api.ols` with categorical terms and interactions,
  `variance_inflation_factor`.
- **Primary references.** Draper NR, Smith H (1998) Applied Regression Analysis, 3rd ed., Wiley.
  Belsley DA, Kuh E, Welsch RE (1980) Regression Diagnostics, Wiley.

### 12.4 Diagnostics shared by linear and nonlinear fits

| Diagnostic | What is computed | Limits |
|---|---|---|
| Lack-of-fit F test | The scatter among replicates at the same X gives the pure error, SS_pe (N - m df, m distinct X). SS_lof = SS_res - SS_pe (m - p df). F = (SS_lof / df_lof) / (SS_pe / df_pe). | Needs replicates and more distinct X values than parameters. |
| Runs test | The number of runs of same-sign residuals along X. One-sided P (this few runs or fewer), exact from the combinatorial distribution for n <= 2000, else a normal approximation with continuity correction. | Zero residuals are ignored. With replicates it is computed on the mean residual at each X. |
| Breusch-Pagan (Koenker) | Squared residuals are regressed on the predictors; LM = n R^2, chi-square with (number of predictors) df. | Linear fits only. |
| Constant scatter of a curve fit | Spearman correlation between the size of the residuals and the predicted Y (`scipy.stats.spearmanr`). | A screening check, not a formal test of a weighting scheme. |
| Durbin-Watson | sum (e_t - e_(t-1))^2 / sum e_t^2; about 2 means no serial correlation. | Reported as a number, no P value. |
| AIC, AICc, BIC | AIC = N ln(SS/N) + 2K; AICc = AIC + 2K(K + 1)/(N - K - 1); BIC = N ln(SS/N) + K ln N; K = fitted parameters + 1. | Constants of the likelihood are dropped, so only differences between models fitted to the same data mean anything. AICc is undefined when N <= K + 1. |

- **Validation.** The exact runs distribution by enumeration of all sign arrangements; statsmodels
  `runstest_1samp`, `het_breuschpagan`, `durbin_watson`; `anova_lm` for the lack-of-fit test; the
  identity between this AIC and the likelihood-based statsmodels AIC.
- **Primary references.** Wald A, Wolfowitz J (1940) Ann Math Stat 11:147-162 (runs test). Breusch
  TS, Pagan AR (1979) Econometrica 47:1287-1294. Koenker R (1981) J Econometrics 17:107-112. Durbin
  J, Watson GS (1950) Biometrika 37:409-428. Hurvich CM, Tsai CL (1989) Biometrika 76:297-307.
  Burnham KP, Anderson DR (2002) Model Selection and Multimodel Inference, 2nd ed., Springer.

## 13. Nonlinear regression

Registry keys: `nonlinear_regression`, `compare_fits`, `compare_curve_parameter`.

### 13.1 The fit

- **What is computed.** Parameters that minimise the (optionally weighted) sum of squared residuals.
- **Implementation.** `scipy.optimize.least_squares` (trust-region-reflective, plus MINPACK
  Levenberg-Marquardt when there are no bounds), with analytic Jacobians: closed form for the
  built-in models, forward-mode automatic differentiation for user-defined equations. Everything
  around the minimiser is TorusPlot's own code: starting values, restarts, a final Gauss-Newton
  polish, covariance, intervals, bands and diagnostics.
- **Starting values.** Built-in models find their own by a separable least-squares grid search.
  If the first fit fails or stops at a bound, up to 6 restarts from perturbed starting values are
  tried. A user-defined equation has no automatic starting values: parameters without one start at
  1.0, with a warning.
- **Convergence.** Judged by the relative-offset criterion of Bates and Watts; values below 0.0001
  indicate a true least-squares solution. A fit that did not converge is reported as such.
- **Built-in models.** Dose-response with four parameters (X as dose, or X as log10 of dose) and
  with three parameters (Hill slope fixed at +1 or -1); Michaelis-Menten; one-phase decay; one-phase
  association; one-site binding (specific, total); exponential growth; Gaussian; straight line;
  quadratic. Dose-response models are fitted on LogEC50.
- **User-defined equations.** The text is parsed, never executed. Only numbers, X, parameter names,
  `+ - * / ^`, brackets and a fixed list of mathematical functions are accepted. `pi` is the
  constant 3.14159...; a bare `e` is a **parameter** like any other letter (write `exp(1)` for
  Euler's number), and the result notes how each constant was read.
- **Weighting.** none; 1/Y; 1/Y^2; 1/SD^2 (the SD of the replicates at each X). **The 1/Y and 1/Y^2
  weights use the observed Y values.** Some other packages weight by the predicted Y, which gives
  slightly different estimates.

### 13.2 Standard errors, intervals and bands

| Quantity | How |
|---|---|
| Standard errors | Square roots of the diagonal of s^2 (J'WJ)^-1, with s^2 = SS / (N - P) |
| Asymptotic CI (default) | estimate +/- t(1 - alpha/2, N - P) x SE |
| Profile-likelihood CI (optional) | The values of one parameter for which the SS, minimised over all other parameters, stays below SS_best x (1 + F(1 - alpha; 1, N - P) / (N - P)). Asymmetric; exact for models linear in the parameters. |
| EC50, half-life and similar | The CI of the fitted parameter is transformed (EC50 = 10^LogEC50), so it is asymmetric |
| Span and similar | Delta method |
| Confidence band of the curve | yhat +/- t sqrt(g' Cov g), g = gradient of the model with respect to the parameters (delta method) |
| Prediction band | The same with s^2 added under the root |
| Parameter dependency | 1 - 1 / ((J'J)_ii ((J'J)^-1)_ii): how much a parameter is entangled with the others |

- **Assumptions and checks.** Convergence. Identifiability: each plateau must be defined by the
  data, and no parameter may be fully dependent on the others. Normal residuals (Shapiro-Wilk).
  Model adequacy (lack-of-fit test with replicates). No systematic deviation (runs test). Constant
  scatter. Stated: independent observations; X measured without error.
- **Limitations.** Asymptotic intervals are approximate for nonlinear models and can be misleading
  when a parameter is poorly determined; use the profile-likelihood interval then. R squared is
  descriptive and can be negative for an unsuitable model. Each replicate is one point by default;
  that is right for independent samples only. One X variable only. With as many points as
  parameters there are no standard errors. Results at the limit of double precision (a
  mathematically perfect fit) are labelled "perfect fit".
- **Validation.** NIST StRD nonlinear regression: 26 data sets, each fitted from both NIST starting
  points through the user-defined-equation path. Test tolerances: parameters 1e-6, standard errors
  1e-5, residual SS 1e-8 (relative). Observed agreement is 8 to 11 digits. Known exception:
  Lanczos1, whose certified residual SS is 1.4e-25; parameters agree to 10 digits, the residual SS
  and standard errors to about 3 (the limit of double precision). All 26 converge from both starts.
  Published: puromycin data (Vmax = 212.68, Km = 0.06412) and BOD data from Bates and Watts (1988).
  Cross-checks: `scipy.optimize.curve_fit` for every built-in model; finite-difference Jacobians;
  coverage of the 95% CI in 200 simulated dose-response experiments. **No published dose-response
  (four-parameter logistic) worked example is used.**
- **Primary references.** Bates DM, Watts DG (1988) Nonlinear Regression Analysis and Its
  Applications, Wiley. Bates DM, Watts DG (1981) Technometrics 23:179-183. Seber GAF, Wild CJ (1989)
  Nonlinear Regression, Wiley. Golub GH, Pereyra V (1973) SIAM J Numer Anal 10:413-432. Branch MA,
  Coleman TF, Li Y (1999) SIAM J Sci Comput 21:1-23. More JJ (1978) Lecture Notes in Mathematics
  630:105-116. Venzon DJ, Moolgavkar SH (1988) Appl Stat 37:87-94. Hill AV (1910) J Physiol
  40:iv-vii. DeLean A, Munson PJ, Rodbard D (1978) Am J Physiol 235:E97-E102. Michaelis L, Menten ML
  (1913) Biochem Z 49:333-369.

### 13.3 Comparing two fits

Registry key: `compare_fits`.

- **Extra-sum-of-squares F test** (nested models only):
  F = [(SS_simple - SS_complex) / (df_simple - df_complex)] / (SS_complex / df_complex).
  The simpler model is kept unless P < alpha.
- **AICc** (any two models fitted to the same points): the AICc difference, the Akaike weights
  (the probability that each model is the better of the two) and the evidence ratio.
- **Nesting** is decided from the model library: the same model with more parameters fixed, three-
  against four-parameter dose-response, line against quadratic, specific against total binding.
- **Limitations.** For user-defined equations nesting cannot be verified. Models with the same
  number of parameters can only be compared by AICc. If the complex model fits worse than the
  simple one, the complex fit is stuck and a warning says so.
- **Validation.** F and AICc from two independent `curve_fit` fits; line against quadratic against
  statsmodels `anova_lm`.
- **Primary references.** Bates and Watts (1988). Hurvich CM, Tsai CL (1989) Biometrika 76:297-307.
  Burnham and Anderson (2002).

### 13.4 Comparing one parameter between data sets

Registry key: `compare_curve_parameter`.

- **What is computed.** Two fits: separate curves with every parameter free (SS_sep, df_sep), and a
  global fit in which one parameter is shared by all data sets (df_shared = df_sep + k - 1).
  F = [(SS_shared - SS_sep) / (k - 1)] / (SS_sep / df_sep). For dose-response models the shared
  parameter is LogEC50, and the EC50 ratio between two data sets is reported with a CI.
- **Assumptions.** The same scatter in all data sets; independent data sets.
- **Limitations.** Built-in models only. One shared parameter at a time.
- **Validation.** An independent global fit with `scipy.optimize.least_squares` written in the
  test; sharing the slope of straight lines reproduces `compare_slopes` exactly.
- **Primary references.** Bates and Watts (1988). Motulsky HJ, Christopoulos A (2004) Fitting
  Models to Biological Data Using Linear and Nonlinear Regression. Oxford Univ Press, ch. 27.

## 14. Survival analysis

Registry key: `survival`. All estimators are TorusPlot's own code.

- **Kaplan-Meier estimate.** At each distinct time t_j with n_j subjects at risk and d_j events:
  S(t_j) = product over i <= j of (1 - d_i / n_i). Subjects censored at t_j still count as at risk there.
- **Standard error.** Greenwood: Var S(t_j) = S(t_j)^2 x sum over i <= j of d_i / (n_i (n_i - d_i)).
- **Confidence limits.** Default `log-log` ("exponential Greenwood"): symmetric on log(-log S), so
  always inside 0 to 1. Alternative `linear`: S +/- z SE, clipped. No limits are given once the curve
  reaches 0.
- **Median survival.** The first time the curve is at or below 50%. Its CI (Brookmeyer and Crowley)
  runs from the first time the lower band is at or below 50% to the first time the upper band is.
  "Not reached" is reported when that never happens; in particular, when the upper band never drops
  to 50% (for example a very small group, or once the curve reaches 0) the upper limit is "not
  reached" rather than the last event time, as the Brookmeyer-Crowley definition requires. When the curve is exactly 50% over an interval,
  TorusPlot reports the first time; some programs report the midpoint, and a note says so.
- **Log-rank (Mantel-Cox) test.** At each event time, group i is expected to have
  E_ij = d_j n_ij / n_j events. U_i = sum_j w_j (d_ij - E_ij), with hypergeometric covariance V.
  chi2 = U' V^- U on k - 1 df. Weights w_j = 1.
- **Gehan-Breslow-Wilcoxon test.** The same with w_j = n_j, which gives more weight to early events.
- **Three or more groups.** Every pair is compared with the same test and the P values are adjusted
  (Bonferroni by default).
- **Hazard ratios** (first group relative to second), from the log-rank quantities:
  Mantel-Haenszel: HR = exp((O_A - E_A) / V), CI exp((O_A - E_A) / V +/- z / sqrt(V)).
  Log-rank (O/E): HR = (O_A / E_A) / (O_B / E_B), CI exp(ln HR +/- z sqrt(1/E_A + 1/E_B)).
- **Assumptions and checks.** Non-informative censoring, independent subjects and proportional
  hazards are stated; they cannot be tested from the table. The result checks that every group has
  at least 5 events for the chi-square approximation.
- **Limitations.** The event column must hold only 1 (event) and 0 (censored). The hazard ratio and
  the power of the log-rank test rest on proportional hazards; crossing curves make a single hazard
  ratio misleading. No Cox regression, no covariates, no stratified test. The log-rank test for
  trend (Tarone 1975) is available as the `trend` option for ordered groups (for example doses).
- **Validation.** Published: the 6-mercaptopurine leukaemia trial of Freireich et al. (1963):
  Kaplan-Meier estimates of the treated arm, medians 23 and 8 weeks, log-rank chi-square 16.79.
  Cross-checks: lifelines (`KaplanMeierFitter`, `logrank_test`, `multivariate_logrank_test`,
  `pairwise_logrank_test`, Wilcoxon weighting); an explicit loop over event times in the test.
  **Hazard ratios and the trend test are validated from first principles only**, because lifelines
  has no matching function and no published value could be quoted with certainty.
- **Primary references.** Kaplan EL, Meier P (1958) Nonparametric estimation from incomplete
  observations. J Am Stat Assoc 53:457-481. Greenwood M (1926) The natural duration of cancer.
  Reports on Public Health and Medical Subjects 33:1-26. Kalbfleisch JD, Prentice RL (1980) The
  Statistical Analysis of Failure Time Data. Brookmeyer R, Crowley J (1982) A confidence interval
  for the median survival time. Biometrics 38:29-41. Mantel N (1966) Evaluation of survival data and
  two new rank order statistics arising in its consideration. Cancer Chemother Rep 50:163-170. Peto
  R, Peto J (1972) J R Stat Soc A 135:185-207. Gehan EA (1965) Biometrika 52:203-223. Breslow N
  (1970) Biometrika 57:579-594. Tarone RE (1975) Biometrika 62:679-682. Mantel N, Haenszel W (1959)
  J Natl Cancer Inst 22:719-748. Peto R et al. (1977) Br J Cancer 35:1-39.

## 15. Contingency tables

Orientation used throughout: rows are the groups, with the first row the exposed or treated group.
Columns are the outcome, with the first column "outcome present". For a 2 x 2 table
[[a, b], [c, d]]: risk1 = a / (a + b), risk2 = c / (c + d). Cells must be counts of subjects.

### 15.1 Chi-square test

Registry key: `chi_square`. TorusPlot's own code.

- **What is computed.** chi2 = sum (O - E)^2 / E with E = row total x column total / N, on
  (r - 1)(c - 1) df. Expected counts, Pearson residuals and adjusted standardized residuals are
  stored with the result.
- **Yates' correction.** Each abs(O - E) is reduced by 0.5 (never past zero) before squaring. It
  exists only for 2 x 2 tables. `auto` applies it to 2 x 2 tables and not otherwise. It makes the
  test conservative. The uncorrected statistic is stored.
- **Effect sizes.** phi (2 x 2) and Cramér's V = sqrt(chi2 / (N x (min(r, c) - 1))), with a
  noncentral chi-square interval, plus Bergsma's bias-corrected V.
- **Assumptions and checks.** Cochran's rule: no expected count below 1 and at most 20% below 5.
  Whole-number counts (anything else is refused). Independent observations (stated).
- **Limitations.** An approximation that fails with small expected counts; the result then points
  to Fisher's exact test or to merging categories. Not for paired data (McNemar's test is not
  available).
- **Validation.** Published: party identification by gender, 2 x 3 table from Agresti (X^2 = 30.1,
  expected counts, standardized residuals). Cross-checks: SciPy `chi2_contingency` (with and
  without Yates), `contingency.association`; statsmodels `Table`.
- **Primary references.** Pearson K (1900) Philos Mag 50:157-175. Yates F (1934) Contingency tables
  involving small numbers and the chi-square test. J R Stat Soc Suppl 1:217-235. Cochran WG (1954)
  Some methods for strengthening the common chi-square tests. Biometrics 10:417-451. Cramér H
  (1946) Mathematical Methods of Statistics. Bergsma W (2013) A bias-correction for Cramér's V and
  Tschuprow's T. J Korean Stat Soc 42:323-328.

### 15.2 Fisher's exact test

Registry key: `fisher_exact`.

- **2 x 2 tables.** With all margins fixed, the top-left count follows a hypergeometric
  distribution. The two-sided P value is the total probability of all tables that are no more
  probable than the observed one. Computed by `scipy.stats.fisher_exact`. One-sided alternatives
  are available. The conditional maximum-likelihood odds ratio with its exact CI comes from
  `scipy.stats.contingency.odds_ratio`.
- **Larger tables** (Freeman-Halton). The same definition, computed in TorusPlot's own code by
  enumerating every table with the observed margins, up to 1,000,000 tables. Beyond that the P value
  is estimated by Monte Carlo with a fixed seed, and a warning gives its standard error. Only the
  two-sided test exists.
- **Assumptions.** The P value is conditional on fixed row and column totals. When the totals were
  not fixed by design the test is still valid but somewhat conservative. Counts; independent
  observations.
- **Limitations.** The exact CI of the conditional odds ratio is not computed above 5,000 subjects.
- **Validation.** Published: the lady tasting tea, [[3, 1], [1, 3]] (one-sided P = 17/70, two-sided
  34/70). Cross-checks: SciPy `fisher_exact`; hypergeometric probabilities from `math.comb`;
  brute-force enumeration of every assignment for r x c tables.
- **Primary references.** Fisher RA (1922) On the interpretation of chi-square from contingency
  tables, and the calculation of P. J R Stat Soc 85:87-94. Fisher RA (1935) The Design of
  Experiments. Freeman GH, Halton JH (1951) Note on an exact treatment of contingency, goodness of
  fit and other problems of significance. Biometrika 38:141-149.

### 15.3 Chi-square test for trend

Registry key: `chi_square_trend`. TorusPlot's own code.

- **What is computed.** For k ordered groups and a yes/no outcome, with scores s_i, group totals
  n_i, counts x_i, R = sum x_i and N = sum n_i:
  chi2_trend = N (N sum x_i s_i - R sum n_i s_i)^2 / (R (N - R) (N sum n_i s_i^2 - (sum n_i s_i)^2)),
  on 1 df. This equals N r^2 for the correlation r between outcome and score. The rest of the
  ordinary chi-square (k - 2 df) tests departure from a straight-line trend.
- **Assumptions.** Ordered categories with scores that reflect their spacing (option `scores`, for
  example "0, 1, 5"; default 1, 2, ..., k in table order); expected counts of at least 5; counts;
  independent observations.
- **Limitations.** With the default scores the test treats the steps between groups as equal.
  TorusPlot reports the Cochran-Armitage form N r^2; some packages report
  (N - 1) r^2, which is slightly smaller.
- **Validation.** Published: infant malformation by maternal alcohol consumption (Graubard and Korn
  1987, as analysed by Agresti): r = 0.0142, M^2 = 6.57 after conversion by (N - 1)/N. Cross-checks:
  statsmodels `Table.test_ordinal_association`; the per-subject correlation.
- **Primary references.** Cochran WG (1954) Biometrics 10:417-451. Armitage P (1955) Tests for
  linear trends in proportions and frequencies. Biometrics 11:375-386.

### 15.4 Measures for 2 x 2 tables

TorusPlot's own code.

| Measure | Formula | Confidence interval |
|---|---|---|
| Odds ratio | (a d) / (b c) | Woolf logit interval: exp(ln OR +/- z sqrt(1/a + 1/b + 1/c + 1/d)) |
| Relative risk | risk1 / risk2 | Katz log interval, SE(ln RR) = sqrt(1/a - 1/(a + b) + 1/c - 1/(c + d)) |
| Risk difference | risk1 - risk2 | Newcombe hybrid score interval (method 10, no continuity correction); the Wald interval is stored for comparison |
| Number needed to treat | 1 / abs(risk difference) | Reciprocals of the risk-difference limits |
| Risk in each row; sensitivity, specificity, predictive values (optional) | Proportions | Wilson score interval |

- **Limitations.** If a cell is zero, 0.5 is added to all four cells before the odds ratio is
  computed (Haldane-Anscombe) and a warning says the values are rough. When the CI of the risk
  difference includes zero, the NNT has no single interval; the warning explains the two-sided range.
- **Validation.** Published: aspirin and myocardial infarction (Physicians' Health Study): odds
  ratio 1.83 with 95% CI 1.44 to 2.33, relative risk 1.82; Newcombe (1998), 56/70 against 48/80:
  interval +0.0524 to +0.3339. Cross-checks: statsmodels `Table2x2`,
  `confint_proportions_2indep`, `proportion_confint`; SciPy `relative_risk`.
- **Primary references.** Woolf B (1955) On estimating the relation between blood group and
  disease. Ann Hum Genet 19:251-253. Haldane JBS (1956) Ann Hum Genet 20:309-311. Anscombe FJ (1956)
  Biometrika 43:461-464. Katz D, Baptista J, Azen SP, Pike MC (1978) Obtaining confidence intervals
  for the risk ratio in cohort studies. Biometrics 34:469-474. Wilson EB (1927) Probable inference,
  the law of succession, and statistical inference. J Am Stat Assoc 22:209-212. Newcombe RG (1998)
  Interval estimation for the difference between independent proportions: comparison of eleven
  methods. Stat Med 17:873-890. Altman DG (1998) Confidence intervals for the number needed to
  treat. BMJ 317:1309-1312.

## 16. Outlier detection

Outliers are only flagged. Nothing is removed or changed. Excluding a value is always the user's
explicit decision and is reported with every later result.

### 16.1 Grubbs' test

Registry key: `outliers_grubbs`. TorusPlot's own code.

- **What is computed.** G = max abs(x_i - mean) / SD. Critical value:
  G_crit = ((n - 1) / sqrt(n)) x sqrt(t^2 / (n - 2 + t^2)), with t the upper alpha / (2n) point of
  the t distribution on n - 2 df. The P value is the same relation solved for alpha:
  P = min(1, 2 n x P(T > t)).
- **Assumptions and checks.** The values other than the outlier are normal (Shapiro-Wilk on the
  values that were not flagged).
- **Limitations.** The P value is a Bonferroni-type bound: accurate for small P values,
  conservative otherwise. The test looks for a single outlier; two or more can mask each other. The
  iterative option repeats the test at the same alpha each round, so the overall false-positive risk
  is higher than alpha. Needs at least 3 values, and has little power with very few. When n - 1
  values are identical, G = (n - 1) / sqrt(n), its largest possible value, whatever the size of the
  deviation; the result then warns that the flag reflects tied values (measurement resolution),
  not an extreme value.
- **Validation.** NIST/SEMATECH e-Handbook of Statistical Methods, section 1.3.5.17
  (https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h1.htm), Tietjen and Moore data:
  G = 2.4687. Published critical values (Grubbs and Beck 1972): n = 10, 2.290; n = 50, 3.128.
  Simulation of the false-positive rate. The tabulated value for n = 20 (2.709) is not reproduced to
  three decimals by the formula (2.708) and is not asserted.
- **Primary references.** Grubbs FE (1950) Sample criteria for testing outlying observations. Ann
  Math Stat 21:27-58. Grubbs FE (1969) Procedures for detecting outlying observations in samples.
  Technometrics 11:1-21.

### 16.2 ROUT

Registry key: `outliers_rout`. TorusPlot's own code; the minimiser for curves is
`scipy.optimize.least_squares`.

- **What is computed.**
  1. A robust fit that assumes Lorentzian scatter: the parameters minimise
     sum ln(1 + (D_i / RSDR)^2), where D_i are the residuals. For a column of values the "model" is
     a single constant (the robust centre).
  2. RSDR, the robust standard deviation of the residuals: the 68.27th percentile of the absolute
     residuals x N / (N - K), with K fitted parameters. It is re-estimated until the fit stops changing.
  3. Outlier identification adapted from the Benjamini-Hochberg procedure: the absolute residuals
     are sorted; for i from int(0.70 N) to N, alpha_i = Q (N - (i - 1)) / N, t = abs(residual_i) / RSDR,
     with a two-tailed P from the t distribution on N - K df. The first i with P < alpha_i, and every
     point with a larger residual, is flagged.
- **Q** is the maximum desired false discovery rate among the flagged points (default 1%).
- **Limitations.** Needs at least 3 values. If more than two thirds of the values are identical the
  robust SD is zero and the method cannot be used. **The implementation follows the published
  description of the method. The paper itself could not be consulted while the code was written, so
  the scale of the merit function and the percentile interpolation rule have not been checked
  equation by equation against the paper.** In the app, ROUT is offered for columns of values, not
  for fitted curves.
- **Validation.** The detection step re-implemented independently in the test and compared on 300
  random data sets; the robust fit checked to minimise the merit function; simulations that
  reproduce the behaviour the paper reports (with Q = 1%, about 1 to 2% of clean Gaussian data sets
  have a flagged value; planted gross outliers are found in more than 95% of data sets). No numbers
  from the paper are asserted.
- **Primary reference.** Motulsky HJ, Brown RE (2006) Detecting outliers when fitting data with
  nonlinear regression - a new method based on robust nonlinear regression and the false discovery
  rate. BMC Bioinformatics 7:123. https://pmc.ncbi.nlm.nih.gov/articles/PMC1472692/

## 17. Replicate-aware (nested) analysis

Registry key: `nested_analysis`. TorusPlot's own code.

- **What is computed.** (1) Technical replicates are summarised (mean or median) within each
  biological replicate of each group. (2) The test runs on those summaries, so n is the number of
  biological replicates.
  - Two groups: Welch's t test (default for unmatched replicates), Student's t test, or a paired
    t test when the replicates are matched.
  - Three or more groups: a one-way ANOVA, or a randomized-block ANOVA when the replicates are
    matched (F = MS_treatment / MS_error on (k - 1, (k - 1)(b - 1)) df), followed by pairwise
    t tests (Welch or paired) with Holm adjustment.
- **Matching.** `auto` treats replicates as matched when exactly the same replicate labels occur in
  every group, as Lord et al. recommend for experiments that each contain all conditions.
- **Extras.** The P value of the wrong analysis (every measurement as an independent sample) is
  computed only to show the size of the pseudoreplication error in a warning. A linear mixed model
  on the individual measurements (statsmodels `mixedlm`, REML) is stored as a cross-check; its Wald
  P value is too small with few replicates, and the result says to report the replicate-means test.
- **Assumptions and checks.** Independent biological replicates (stated). Normality of the
  replicate means or of their paired differences (Shapiro-Wilk), with a note that the check has
  little power with so few replicates.
- **Limitations.** A group with a single biological replicate cannot be analysed. With 2 replicates
  a test has almost no power (warning); at least 3 are recommended. Averaging gives every biological
  replicate the same weight, whatever its number of technical replicates. Mixed-effects models are
  not offered as an analysis.
- **Validation.** Cross-checks on simulated data with fixed seeds: SciPy t tests and `f_oneway` on
  hand-aggregated means; statsmodels `anova_lm` for the randomized-block ANOVA; in balanced designs
  the REML Wald chi-square of `mixedlm` equals t^2 (tolerance 2 to 5%, limited by the optimiser).
  **No published worked example with reported numbers is used.**
- **Primary references.** Lord SJ, Velle KB, Mullins RD, Fritz-Laylin LK (2020) SuperPlots:
  communicating reproducibility and variability in cell biology. J Cell Biol 219:e202001064.
  https://doi.org/10.1083/jcb.202001064 . Hurlbert SH (1984) Pseudoreplication and the design of
  ecological field experiments. Ecol Monogr 54:187-211. Lazic SE (2010) The problem of
  pseudoreplication in neuroscientific studies: is it affecting your analysis? BMC Neurosci 11:5.
  Laird NM, Ware JH (1982) Random-effects models for longitudinal data. Biometrics 38:963-974.

## 18. Power and sample size

Functions `power_ttest` and `power_anova` in `torusplot/stats/power.py`. Not registered analyses:
they need no data table. In the app: *Power and sample size calculator* on page **2 Analyze**.

- **What is computed.** Given two of effect size, sample size and power, the third is solved for.
  t tests (unpaired, paired, one-sample): the noncentral t distribution. One-way ANOVA: the
  noncentral F distribution with df = (k - 1, N - k) and noncentrality f^2 N.
- **Implementation.** The power function is statsmodels (`TTestIndPower`, `TTestPower`,
  `FTestAnovaPower`). The unknown is found by root-finding in TorusPlot's own code. Where SciPy's
  noncentral t returns no value in a far tail, the power is computed tail by tail and the negligible
  tail is taken as 0.
- **Effect sizes.** t tests: Cohen's d (for paired designs d_z: mean of the differences / SD of the
  differences). ANOVA: Cohen's f = SD of the group means / common within-group SD.
- **Limitations.** n is per group (unpaired, ANOVA), pairs (paired) or values (one sample). For
  unequal groups in the unpaired design n is the size of group 1, with group 2 = ratio x n1. The
  ANOVA assumes equal groups. A solved n is given exactly and rounded up; an entered n is reported
  as entered. For a lower-tailed test the detectable effect is described as "or more negative".
  The ANOVA power
  is for the overall F test, not for pairwise comparisons. No power calculations for rank tests,
  proportions, correlation, survival, repeated measures or factorial designs. The answer depends
  entirely on the assumed effect size.
- **Validation.** Published: Cohen (1988) Table 2.4.1 and Cohen (1992) Table 2 (alpha = 0.05,
  power = 0.80): d = 0.2 / 0.5 / 0.8 gives n = 393 / 64 / 26 per group; ANOVA tables for 3 to 7
  groups. Cohen's entries equal the exact solution rounded to the nearest integer; TorusPlot's
  "rounded up" n can therefore be one larger. One table entry (7 groups, f = 0.10: 195) is not
  reproduced (exact solution 195.53) and is not asserted. Cross-checks: formulas written out with
  `scipy.stats.nct` and `ncf`; statsmodels `solve_power`; Monte-Carlo simulation.
- **Primary references.** Cohen J (1988) Statistical Power Analysis for the Behavioral Sciences,
  2nd ed. Cohen J (1992) A power primer. Psychol Bull 112:155-159.

## 19. Test recommendation

Function `recommend_test` in `torusplot/stats/recommend.py`. Not a registered analysis.

- **What it does.** It inspects the table and returns the key of one analysis, ready-to-use options,
  the checks that led there, cautions and alternatives. It does not run the test and does not change
  the data. The decision tables are in the user guide.
- **Checks.** Shapiro-Wilk (`scipy.stats.shapiro`) and Brown-Forsythe
  (`scipy.stats.levene(center="median")`), both at alpha = 0.05. For three or more independent
  groups normality is judged on residuals standardised within each group, so that unequal spread is
  not mistaken for non-normality.
- **Limitations.** Choosing a test from a pre-test of the same data is a convention, not a
  guarantee: pre-tests have little power in small samples, and the two-stage procedure changes the
  error rates of the final test somewhat. Welch-type tests are always listed as alternatives. For
  Multiple variables tables the proposal depends on column names and types, matched as whole words
  (so a factor called "Exposure" is not mistaken for an "exp"eriment column) and confirmed by the
  structure of the data: a column is treated as a biological replicate only if each group x
  replicate combination holds several rows, and as a subject ID only if each subject appears once
  per level of a within-subject factor (then a repeated-measures or mixed ANOVA is proposed). Sparse
  contingency tables (Cochran's rule fails) are offered Fisher's exact test, which TorusPlot computes
  for r x c tables. Small-sample cautions give the smallest attainable P value of each rank test. The P values of these
  checks can differ from the checks printed with the analysis itself, which uses plain residuals.
- **Validation.** `tests/stats/test_recommend.py` checks each branch on constructed data.
- **References.** Welch BL (1947) Biometrika 34:28-35. Ruxton GD (2006) The unequal variance t-test
  is an underused alternative to Student's t-test and the Mann-Whitney U test. Behav Ecol 17:688-690.

## 20. Conventions

| Topic | Convention |
|---|---|
| Tails | Two-sided by default. One-sided tests are an option for the two-group tests, correlation and Fisher's exact test (2 x 2). Choose one only if the direction was decided before seeing the data. |
| Alpha | 0.05 by default; an option of every analysis. The assumption checks that run inside an analysis use 0.05. |
| Direction of differences | First group minus second group (A - B); sample minus hypothetical value. |
| P value format | Four decimals. Values below 0.0001 are written `<0.0001`. A P value is never written as 0. |
| Stars | ns: P >= 0.05; one star: P < 0.05; two: P < 0.01; three: P < 0.001; four: P < 0.0001. An alternative style stops at three stars. Stars on a figure use the adjusted P value when there is one. |
| Confidence level | 95% by default. The two-group tests and the descriptive statistics have a *Confidence level* option. In ANOVA follow-up tests, regression and curve fits the level is 1 - alpha. |
| Confidence intervals with one-sided tests | The t tests and rank tests keep two-sided intervals and say so. Correlation gives a one-sided interval with a one-sided test. |
| Sums of squares | Type III by default: each term is adjusted for all others, including interactions that contain it; main effects compare unweighted marginal means. Type II is an option: each term is adjusted for the terms that do not contain it. With equal cell sizes all types agree. |
| Factor coding | Sum-to-zero coding in factorial ANOVA. Dummy coding with an alphabetical reference level in multiple regression. |
| Missing values | Empty cells, text in a numeric column and non-finite numbers are ignored and counted in the data notes. Unpaired analyses drop single values. Paired and repeated-measures analyses drop the whole pair or subject. Correlation matrices use pairwise-complete rows. |
| Exclusions | Only values that the user excluded explicitly are left out. The raw table is never changed. Every result run afterwards carries a data note that lists the excluded cells, and the methods paragraph repeats it. |
| Approximate data | A table digitized from an image is marked approximate. Analyses on it are refused unless the user overrides; the result is then flagged in every report. |
| Exact and asymptotic P values for rank tests | *Exact*: the P value is computed from the complete null distribution of the statistic (every possible arrangement of the ranks). TorusPlot uses it for Mann-Whitney (no ties and both groups at most 50; with ties, both groups at most 6, from the permutation distribution of the mid-ranks), Wilcoxon signed-rank (no tied differences and at most 50 non-zero differences; with ties, at most 6 non-zero differences, over all sign patterns of the mid-ranks) and Spearman (no ties, n at most 9). *Asymptotic*: a normal, t or chi-square approximation that becomes accurate as n grows; used with ties in larger samples, with larger samples, and always for Kruskal-Wallis, Friedman and Dunn's test. The normal approximations of the Mann-Whitney and Wilcoxon tests are never reported below the smallest P value the exact test can attain. Each result states which was used. |
| Continuity correction | Applied in the normal approximations of the Mann-Whitney and Wilcoxon tests and, by default, as Yates' correction in 2 x 2 chi-square tests. |
| Number format | About four significant digits in the interpretation text. Full precision is kept in the result object and in the project file. |
| Reproducibility | The same data and options always give the same numbers. Random numbers are used in two places only, each with a fixed seed: the restarts of a curve fit whose first attempt failed, and the Monte Carlo estimate of Fisher's exact test for tables too large to enumerate. |

## 21. What TorusPlot does not do (yet)

Each item below was checked against the code of `torusplot/stats/`. If you need one of these, use
a dedicated statistics package or ask a statistician.

- **Bayesian methods.** None.
- **Mixed-effects models.** Not offered as an analysis: no random slopes, no unbalanced
  repeated-measures model. (A random-intercept model is fitted only as a cross-check inside the
  nested analysis.) Repeated-measures analyses are complete-case only.
- **Exact P values for the Kruskal-Wallis and Friedman tests.** Chi-square approximations only.
- **Permutation and bootstrap methods.** None, apart from the exact null distributions listed in
  section 20.
- **Repeated-measures designs beyond two factors**, and mixed designs with more than one between
  or one within factor.
- **A rank-based two-way analysis**, and a rank-based ANCOVA.
- **Multivariate methods**: MANOVA, Box's M test, principal components, clustering.
- **Regression for other outcome types**: logistic, Poisson, ordinal. No Cox proportional-hazards
  regression, no stratified log-rank test. Deming and Passing-Bablok regression are absent.
- **Other correlation measures**: Kendall's tau, partial correlation.
- **McNemar's test** for paired proportions, the binomial test and the chi-square goodness-of-fit
  test.
- **A two-sample Kolmogorov-Smirnov test.**
- **Other multiple-comparison procedures**: Scheffe, Newman-Keuls, Benjamini-Yekutieli, a test for
  linear trend across the groups of a one-way ANOVA.
- **Equivalence and non-inferiority tests** (TOST).
- **Method-comparison and diagnostic tools**: Bland-Altman analysis, ROC curves. (Sensitivity,
  specificity and predictive values of a 2 x 2 table are available.)
- **Power calculations** other than for t tests and the one-way ANOVA.
- **Curve fitting with more than one X variable**, global fits that share several parameters, and
  robust curve fitting or ROUT for curves in the app.
- **Data transformations in the app** (log, normalisation). Transform the values in your
  spreadsheet before you paste them.

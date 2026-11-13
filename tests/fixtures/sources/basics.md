# Validation sources: descriptives, assumption checks, two-group tests, outliers, power

Tests: `tests/stats/test_descriptives.py`, `test_assumptions.py`, `test_two_group.py`,
`test_outliers.py`, `test_power.py`.

All numbers below were typed in by hand. The network was not reachable while these tests
were written (requests to `www.itl.nist.gov` and `pmc.ncbi.nlm.nih.gov` were refused), so
nothing was downloaded; every published value was checked by recomputing it, and a value
was only kept when the recomputation reproduced it to the published precision. Values
that could not be reproduced are listed under "Tried and dropped".

Kinds of source: **NIST** = NIST reference data / handbook (US government work),
**published** = worked example or table from the literature, **cross-check** = agreement
between independent implementations (ours vs scipy vs statsmodels vs a from-first-
principles computation written inside the test).

## Descriptive statistics (`test_descriptives.py`)

| Kind | Source | What is checked |
|---|---|---|
| NIST | NIST Statistical Reference Datasets, univariate summary statistics, data sets NumAcc1, NumAcc2, NumAcc3, NumAcc4 (https://www.itl.nist.gov/div898/strd/univ/homepage.html). The data follow a simple documented pattern (NumAcc1: 10000001, 10000003, 10000002; NumAcc2: 1.2 followed by 500 pairs 1.1, 1.3; NumAcc3: 1000000.2 followed by 500 pairs 1000000.1, 1000000.3; NumAcc4: 10000000.2 followed by 500 pairs 10000000.1, 10000000.3) and were generated from that pattern in the test, not downloaded. | Certified mean and SD: NumAcc1 10000002 / 1; NumAcc2 1.2 / 0.1; NumAcc3 1000000.2 / 0.1; NumAcc4 10000000.2 / 0.1. Tolerances on the SD of NumAcc3/4 (1e-8 / 1e-7 relative) reflect that the data themselves are not exactly representable in double precision. |
| published | Hyndman RJ, Fan Y (1996) Sample quantiles in statistical packages. Am Stat 50:361-365, definition 7. | Quartiles from the defining formula h = (n - 1)p + 1 written out in the test. |
| cross-check | numpy (`mean`, `std`, `percentile`), scipy (`sem`, `iqr`, `t.interval`, `variation`, `skew(bias=False)`, `kurtosis(bias=False)`, `gmean`), pandas (`quantile`, `skew`, `kurt`), statsmodels (`DescrStatsW.tconfint_mean`). | Every key returned by `describe`. |

## Normality and equal-variance tests (`test_assumptions.py`)

| Kind | Source | What is checked |
|---|---|---|
| published | Shapiro SS, Wilk MB (1965) An analysis of variance test for normality (complete samples). Biometrika 52:591-611, worked example: weights (lb) of 11 men: 148, 154, 158, 160, 161, 162, 166, 170, 182, 195, 236. | W = 0.79 (as printed, two decimals), significant at the 1% level. |
| cross-check | scipy `shapiro`; Royston's algorithm (Royston P (1992) Stat Comput 2:117-119; (1995) Appl Stat 44:547-551) re-implemented in the test from the published coefficients; the exact n = 3 formula. | W and P. |
| cross-check | scipy `normaltest`, `skewtest`, `kurtosistest`; the D'Agostino (1970) skewness z and Anscombe & Glynn (1983) kurtosis z written out in the test. | K2, both z scores, P. |
| cross-check | statsmodels `lilliefors(pvalmethod="table")`; scipy `kstest`; the KS distance written out in the test; simulation showing the uncorrected KS P value is invalid with estimated parameters. | D, Lilliefors P. |
| cross-check | statsmodels `normal_ad`; scipy `anderson(method="interpolate")` (statistic only); D'Agostino & Stephens (1986) Table 4.9 anchor points (A* = 0.631, 0.752, 1.035 for P = 0.10, 0.05, 0.01). | A2, A*, P. |
| cross-check | scipy `probplot`; Filliben (1975) order-statistic medians written out in the test; `numpy.polyfit`. | QQ-plot coordinates and line. |
| cross-check | scipy `levene(center="mean"/"median")`; `f_oneway` on absolute deviations; scipy `bartlett`; Bartlett's formula written out; F-ratio test from the F distribution. | Levene, Brown-Forsythe, Bartlett, F test of two variances. |
| simulation | Fixed-seed simulations: type I error of all four normality tests on normal samples, power on exponential samples; Bartlett vs Brown-Forsythe on heavy-tailed data. | Calibration and the advice given in docstrings. |

## Two-group tests (`test_two_group.py`)

| Kind | Source | What is checked |
|---|---|---|
| published | Student (1908) The probable error of a mean. Biometrika 6:1-25, illustration I (Cushny & Peebles data, additional hours of sleep with two soporifics, 10 patients): 0.7, -1.6, -0.2, -1.2, -0.1, 3.4, 3.7, 0.8, 0.0, 2.0 and 1.9, 0.8, 1.1, 0.1, -0.1, 4.4, 5.5, 1.6, 4.6, 3.4. | Student's own summary: mean difference 1.58, SD 1.17 (divisor n), z = 1.35, P = 0.9985. |
| published | The same data are R's built-in `sleep` data set; results printed in the documented examples of R's `t.test` and `wilcox.test` (R Core Team, R: A Language and Environment for Statistical Computing). Values typed from the printed output, not produced by running R here. | Paired t = -4.0621, df = 9, P = 0.002833, 95% CI -2.4598858 to -0.7001142; Welch t = -1.8608, df = 17.776, P = 0.07939, CI -3.3654832 to 0.2054832; pooled t: df = 18, P = 0.07919; rank-sum W = 25.5, P = 0.06933; signed-rank V = 0, P = 0.009091. |
| cross-check | scipy `ttest_ind` (pooled and Welch), `ttest_rel`, `ttest_1samp`, each with `.confidence_interval()`; statsmodels `ttest_ind`, `CompareMeans.tconfint_diff`, `DescrStatsW.ttest_mean` / `tconfint_mean`; formulas written out in the test (pooled SE, Welch-Satterthwaite df). | t, df, P (all three alternatives), CI. |
| cross-check | scipy `mannwhitneyu` (exact and asymptotic with continuity), `wilcoxon` (exact; asymptotic with `correction=True`, `zero_method` wilcox and pratt), incl. 800 random tie-laden data sets; brute-force enumeration of the exact null distributions; brute-force Hodges-Lehmann estimates and CI ranks; test/CI duality and coverage simulations. | U, W+, W-, P, Hodges-Lehmann estimates and their CIs. |
| cross-check | scipy `pearsonr` (with CI), `spearmanr`, `shapiro`, `levene`; the shared `effect_sizes` module. | Pairing correlation, inline assumption checks, effect sizes. |
| published (table) | Classical critical-value tables for rank tests: Mann-Whitney U, n1 = n2 = 10, two-sided 5%: U <= 23; Wilcoxon signed-rank T, n = 11, two-sided 5%: T <= 10. | The exact null distributions that drive the P values and CIs. |

## Outliers (`test_outliers.py`)

| Kind | Source | What is checked |
|---|---|---|
| NIST | NIST/SEMATECH e-Handbook of Statistical Methods, section 1.3.5.17 "Grubbs' Test for Outliers" (https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h1.htm), example with the Tietjen & Moore (1972) data 199.31, 199.53, 200.19, 200.82, 201.92, 201.95, 202.18, 245.57. Typed from memory of the page (site unreachable); kept because both the statistic and the critical value reproduce. | G = 2.4687 (we get 2.46876); 5% critical value of the upper one-sided test 2.032; conclusion: 245.57 is an outlier. |
| published (table) | Grubbs FE, Beck G (1972) Technometrics 14:847-854 / ASTM E178 critical values. | Two-sided 5%: n = 10 -> 2.290, n = 50 -> 3.128; one-sided 5%, n = 8 -> 2.032. |
| cross-check | The critical-value formula written out in the test; inversion P(G_crit(alpha)) = alpha; false-positive simulation. | Critical values, P values, calibration. |
| published (method) | Motulsky HJ, Brown RE (2006) Detecting outliers when fitting data with nonlinear regression - a new method based on robust nonlinear regression and the false discovery rate. BMC Bioinformatics 7:123. The paper could not be fetched; the implementation follows the method description (RSDR = P68.27 x N/(N-K); Lorentzian merit function; FDR-style scan of the largest 30% of residuals). | The detection step is re-implemented independently in the test and compared on 300 random data sets; the robust fit is checked to minimise the Lorentzian merit function (scipy optimisers); simulations reproduce the behaviour the paper reports (Q = 1%: at least one false outlier in < 6% of 400 clean Gaussian data sets, observed about 1-2%; planted gross outliers found in > 95% of data sets; robust fit barely moved by outliers while least squares is). |

## Power (`test_power.py`)

| Kind | Source | What is checked |
|---|---|---|
| published | Cohen J (1988) Statistical Power Analysis for the Behavioral Sciences, 2nd ed., Table 2.4.1; Cohen J (1992) A power primer. Psychol Bull 112:155-159, Table 2 (alpha = .05, power = .80). | t test: d = .2/.5/.8 -> n = 393/64/26 per group. ANOVA, f = .10/.25/.40: 3 groups 322/52/21; 4 groups 274/45/18; 5 groups 240/39/16; 6 groups 215/35/14; 7 groups (.25/.40) 32/13. Cohen's entries equal our exact solution rounded to the nearest integer; our "rounded up" n can therefore be one larger (e.g. 53 instead of 52), which is stated in the test. |
| cross-check | Noncentral t (`scipy.stats.nct`) and noncentral F (`scipy.stats.ncf`) power formulas written out in the test; statsmodels `solve_power`; Monte-Carlo simulation of t tests. | Power, required n, detectable effect, curves. |

## Tried and dropped

* Grubbs two-sided 5% critical value for n = 20: tables give 2.709, the t-based formula
  gives 2.7082 (rounds to 2.708). Not asserted, because the tabulated value is not
  reproduced to three decimals by the formula (the formula is a Bonferroni-type
  approximation to the exact percentage point).
* Cohen (1992) Table 2, ANOVA with 7 groups and a small effect (f = .10): the table gives
  195, our exact solution is 195.53 (rounds to 196). Not asserted.
* NIST e-Handbook examples for Levene's and Bartlett's tests (GEAR.DAT) and the NIST StRD
  univariate data sets with long listings (Michelson, Lew, Lottery, Mavro, PiDigits): the
  data could not be downloaded and are too long to type from memory reliably.
* Reading the ROUT paper itself (PMC1472692): host unreachable.

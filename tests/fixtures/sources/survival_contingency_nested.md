# Validation sources: survival, contingency tables, replicate-aware (nested) analysis

Tests: `tests/stats/test_survival.py`, `tests/stats/test_contingency.py`,
`tests/stats/test_nested.py`.

All data below were typed in by hand from memory of the cited primary sources and
standard textbooks; nothing was downloaded or copied from a code repository. Every
"published" value was first recomputed with the allowed stack (scipy / statsmodels /
lifelines) and with an independent from-first-principles calculation, and is only used
where the recomputed number equals the published one at the published precision.
Values that could not be confirmed that way were dropped (listed at the end).

Kinds of check: **published** = worked example with reported results;
**cross-check** = agreement between independent implementations inside the allowed
stack; **first principles** = formula written out in the test file itself.
There is no NIST Statistical Reference Dataset for these methods.

## Survival (`test_survival.py`)

### Published: 6-mercaptopurine leukaemia trial

* Data: remission times (weeks) of 42 children with acute leukaemia, 21 on
  6-mercaptopurine (9 relapses, 12 censored) and 21 on placebo (21 relapses).
* Primary source: Freireich EJ, Gehan E, Frei E III, et al. (1963) The effect of
  6-mercaptopurine on the duration of steroid-induced remissions in acute leukemia.
  *Blood* 21:699-716. The data are tabulated in Gehan EA (1965) *Biometrika*
  52:203-223 and Cox DR (1972) *J R Stat Soc B* 34:187-220, and worked through in
  Kleinbaum DG, Klein M, *Survival Analysis: A Self-Learning Text*, and many other
  textbooks.
* Values checked (published precision):
  * Kaplan-Meier estimates for the 6-MP arm: S(6) = 0.857, S(7) = 0.807,
    S(10) = 0.753, S(13) = 0.690, S(16) = 0.627, S(22) = 0.538, S(23) = 0.448;
  * median remission 23 weeks (6-MP) and 8 weeks (placebo);
  * log-rank chi-square = 16.79 on 1 df, P < 0.0001;
  * first life-table row of the 6-MP arm: 21 at risk at week 6, 3 relapses, 1 censored.
* The placebo arm has no censoring, so its Kaplan-Meier curve is additionally checked
  against the plain fraction still in remission (first principles).

### Cross-checks

| What | Against |
|---|---|
| KM survival, log-log ("exponential Greenwood") confidence limits | `lifelines.KaplanMeierFitter` (`survival_function_`, `confidence_interval_`) |
| Median and Brookmeyer-Crowley interval | `KaplanMeierFitter.median_survival_time_`, `lifelines.utils.median_survival_times` |
| Greenwood SE, log-log and linear limits, at-risk / event / censored counts | explicit loop in the test file (first principles) |
| Log-rank test, 2 to 4 groups | `lifelines.statistics.logrank_test`, `multivariate_logrank_test`, plus an explicit loop over event times building O, E and the hypergeometric covariance matrix |
| Gehan-Breslow-Wilcoxon test | lifelines with `weightings="wilcoxon"`, plus the explicit loop with weights n_j |
| Pairwise comparisons | `lifelines.statistics.pairwise_logrank_test`; Bonferroni / Holm via the shared `multiplicity` module |
| Hazard ratios (Mantel-Haenszel and O/E) and their CIs | formulas written out in the test from the explicit-loop O, E, V; reciprocal under group swap; identity (ln HR / SE)^2 = log-rank chi-square |
| Log-rank test for trend | formula (s'U)^2 / (s'Vs) written out in the test; equals the log-rank test for two groups; invariant to linear rescaling of the scores |

lifelines has no hazard-ratio-from-log-rank or trend-test function, so those two rest on
the first-principles checks only.

### Property-based

Random small data sets with heavy ties and censoring: survival within [0, 1] and
non-increasing, limits bracket the estimate, agreement with lifelines at every time,
no infinities / NaN in JSON, P in [0, 1] or `None`, all-censored data handled, row
shuffling (which permutes group order) leaves the omnibus statistic unchanged,
agreement with `multivariate_logrank_test`, hazard-ratio reciprocity.

## Contingency tables (`test_contingency.py`)

### Published

1. **Lady tasting tea**, table [[3, 1], [1, 3]]. Fisher RA (1935) *The Design of
   Experiments* (the experiment); the 3-of-4 outcome is the usual textbook
   illustration (e.g. Agresti A, *Categorical Data Analysis*). Checked: one-sided
   P = 0.2429 (= 17/70), two-sided P = 0.4857 (= 34/70); the fractions are also
   derived analytically in the test.
2. **Aspirin and myocardial infarction**, Physicians' Health Study: MI in 189 of
   11,034 on placebo and 104 of 11,037 on aspirin. Steering Committee of the
   Physicians' Health Study Research Group (1988) *N Engl J Med* 318:262-264; analysed
   in Agresti A, *An Introduction to Categorical Data Analysis* and *Categorical Data
   Analysis*. Checked: proportions 0.0171 and 0.0094; difference 0.0077 with Wald 95%
   CI (0.005, 0.011); relative risk 1.82 (aspirin vs placebo 0.55); odds ratio 1.83
   with 95% CI (1.44, 2.33).
3. **Newcombe's risk-difference example**, 56/70 vs 48/80. Newcombe RG (1998) Interval
   estimation for the difference between independent proportions: comparison of eleven
   methods. *Stat Med* 17:873-890. Checked: method 10 (hybrid score, no continuity
   correction) interval +0.0524 to +0.3339.
4. **Party identification by gender** (2000 General Social Survey), 2 x 3 table
   [[762, 327, 468], [484, 239, 477]], from Agresti A, *An Introduction to Categorical
   Data Analysis* (2nd ed.). Checked: X^2 = 30.1 on 2 df; expected counts 703.7, 319.6,
   533.7 / 542.3, 246.4, 411.3; standardized (adjusted) residuals 4.50, 0.70, -5.32
   for the first row (sign-reversed for the second).
5. **Infant malformation by maternal alcohol consumption**, 2 x 5 table
   [[48, 38, 5, 1, 1], [17066, 14464, 788, 126, 37]] with scores 0, 0.5, 1.5, 4.0, 7.0.
   Data: Graubard BI, Korn EL (1987) *Biometrics* 43:471-476; analysed in Agresti (same
   book). Checked: correlation r = 0.0142; linear trend statistic
   M^2 = (n - 1) r^2 = 6.57, P = 0.010; Pearson X^2 = 12.1. (TorusPlot reports the
   Cochran-Armitage form n r^2; the test converts with the factor (n - 1) / n.)

### Cross-checks

| What | Against |
|---|---|
| Chi-square statistic, P, df, expected counts (with and without Yates) | `scipy.stats.chi2_contingency`; explicit sum (O - E)^2 / E; textbook Yates formula n(\|ad - bc\| - n/2)^2 / (r1 r2 c1 c2) |
| Pearson and adjusted residuals, cell contributions | `statsmodels.stats.contingency_tables.Table` (`resid_pearson`, `standardized_resids`, `chi2_contribs`) |
| Cramér's V | `scipy.stats.contingency.association(method="cramer")` |
| phi | Pearson correlation of the two 0/1 variables (`numpy.corrcoef`) |
| Cramér's V CI | limits put the observed chi-square at the 97.5th / 2.5th percentile of `scipy.stats.ncx2` |
| Bias-corrected V | Bergsma (2013) formula written out in the test |
| Fisher 2x2 P (two-sided and both one-sided) | `scipy.stats.fisher_exact`; hypergeometric probabilities from `math.comb` in the test |
| Conditional MLE odds ratio and exact CI | `scipy.stats.contingency.odds_ratio(kind="conditional")` |
| Fisher r x c (Freeman-Halton) exact P | brute-force enumeration of every assignment of subjects to columns (test file); the 2x2 special case vs `scipy.stats.fisher_exact`; scipy's Monte Carlo `fisher_exact(method=MonteCarloMethod)` within 5 standard errors |
| Odds ratio + Woolf CI, relative risk + Katz CI | `statsmodels` `Table2x2` (`oddsratio`, `oddsratio_confint`, `log_oddsratio_se`, `riskratio`, `riskratio_confint`); `scipy.stats.contingency.relative_risk` |
| Risk difference: Newcombe hybrid score and Wald | `statsmodels.stats.proportion.confint_proportions_2indep(method="newcomb" / "wald", compare="diff")` |
| Wilson intervals (risks, sensitivity, specificity, PPV, NPV) | `statsmodels.stats.proportion.proportion_confint(method="wilson")`; formula in the test |
| Cochran-Armitage trend | n r^2 from per-subject correlation; `Table.test_ordinal_association` (which uses (n - 1) r^2); least-squares slope via `numpy.polyfit` |

The scipy documentation example [[8, 2], [1, 5]] (P = 0.0350) is used as a readable
case; it is a cross-check, not an independent published result.

### Property-based

Random 2x2 tables (including zero cells and counts up to 10^9) and random r x c tables:
`StatsError` exactly when a row or column total is zero; P in [0, 1]; CI brackets the
estimate; no infinities / NaN in JSON; Haldane-Anscombe flag whenever a cell is zero;
chi-square invariant under row / column swaps and transposition; Fisher P invariant
under row swap; risk difference antisymmetric under row swap.

## Replicate-aware / nested (`test_nested.py`)

No published worked example with reported numerical results is used: Lord et al. (2020)
describe the method (average technical replicates within each biological replicate,
test the replicate means, pair by experiment) but I do not know a data set with
published results that I can reproduce with certainty. Method references: Lord SJ,
Velle KB, Mullins RD, Fritz-Laylin LK (2020) *J Cell Biol* 219:e202001064; Hurlbert SH
(1984) *Ecol Monogr* 54:187-211; Lazic SE (2010) *BMC Neurosci* 11:5.

### Cross-checks (simulated data, fixed seeds)

| What | Against |
|---|---|
| Replicate means / medians, technical n and SD | `pandas.groupby` |
| Welch / Student / paired t on replicate means: t, df, P, CI of the difference | `scipy.stats.ttest_ind` / `ttest_rel` on hand-aggregated means, including `confidence_interval()` |
| One-way ANOVA on replicate means | `scipy.stats.f_oneway`; sums of squares from definitions |
| Randomized-block (repeated-measures) ANOVA: SS, F, P for treatment and blocks | `statsmodels` OLS `anova_lm` on `value ~ C(group) + C(replicate)`; with two conditions F = paired t^2 |
| Pairwise comparisons | scipy t tests per pair; Holm via the shared `multiplicity` module |
| Effect sizes (Hedges' g, d_z, eta squared, partial eta squared) | shared `effect_sizes` module (itself tested in `test_effect_sizes.py`) and definitions |
| Assumption checks | `scipy.stats.shapiro`, `scipy.stats.levene(center="median")` |
| Naive pooled P shown in the warning | `scipy.stats.ttest_ind(equal_var=False)` / `f_oneway` on all measurements |
| Mixed-model cross-check | in balanced designs the REML Wald chi-square from `statsmodels` `mixedlm` equals t^2 (or (k - 1) F) of the replicate-means test; tolerance 2-5% because the optimiser's stopping rule limits precision (documented looser tolerance for an iterative fit) |

### Property-based

Random nested frames (2-3 groups, 1-4 biological replicates, 1-4 technical values,
constants and NaN mixed in): `StatsError` exactly when a group has a single biological
replicate; otherwise P in [0, 1] or `None` with a "no variability" warning; n never
exceeds the number of biological replicates; statistic invariant under linear
rescaling of the values; the mixed-model cross-check never raises.

## Tried and dropped

* **Expected event counts for the 6-MP trial** as printed in some textbooks
  (19.26 / 10.74): the exact computation gives 19.2505 / 10.7495; the printed values
  appear to be sums of rounded per-time terms, so they are not asserted. The
  chi-square (16.79) is unaffected.
* **Chi-square statistic for the aspirin table** (I recall 25.0 from a textbook; the
  computed value is 25.01) and the **relative-risk confidence interval** for the same
  table (I recall 1.43 to 2.30; the Katz interval computes to 1.433 to 2.306): I am
  not certain of the printed values, so both are checked against scipy / statsmodels
  only, not asserted as published.
* **Mantel-Haenszel hazard ratio for the 6-MP trial**: I know of no published value I
  can quote with certainty (published hazard ratios for these data are usually Cox
  model estimates, a different estimator), so it is validated from first principles only.

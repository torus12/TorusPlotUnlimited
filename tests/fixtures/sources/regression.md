# Validation sources: correlation, linear regression, nonlinear regression, equation parser

Covers `torusplot/stats/correlation.py`, `regression.py`, `nonlinear.py`, `equation.py`
and the tests `tests/stats/test_correlation.py`, `test_regression.py`, `test_nonlinear.py`,
`test_nist_nls.py`, `test_equation.py`.

Legend: **NIST** = certified reference values; **Published** = worked example from the
literature, numbers typed in by hand; **Cross-check** = independent implementation inside
the allowed stack or a first-principles computation written in the test.

## 1. NIST Statistical Reference Datasets (StRD)

Source: National Institute of Standards and Technology, Information Technology
Laboratory, *Statistical Reference Datasets*, https://www.itl.nist.gov/div898/strd/
(work of the US government, public domain). Loader: `tests/fixtures/nist/strd_regression.py`.

### How the files were obtained (please read)

The NIST web site (`www.itl.nist.gov`) was **blocked by the network policy** of the build
environment, so nothing was downloaded from NIST directly.

* **Nonlinear regression, 26 files** (`Misra1a.dat` ... `Bennett5.dat`): the NIST files
  with their NIST headers intact (certified values, standard deviations, residual SS, both
  starting points), copied unchanged from the `NIST_STRD/` directory of the source
  distribution of the Python package *lmfit* 1.3.4 on PyPI (BSD-3-Clause; the `.dat` files
  themselves are NIST's). Only the `.dat` files were extracted; no lmfit code was read or
  used. They could not be diffed against NIST's own copies. The model equations in the
  loader were typed from the "Model:" section of each header. `Nelson.dat` (two
  predictors) is not used because the fitter takes one X.
* **Norris.dat** (linear, 36 observations): the NIST file as shipped in the test data of the
  installed SciPy wheel (`scipy/stats/tests/data/nist_linregress/Norris.dat`), copied
  unchanged.
* **Longley.dat** (multiple regression): **not the verbatim NIST file.** It was assembled by
  hand in the NIST layout: data from `statsmodels.datasets.longley` (Longley 1967, public
  domain data), reference values as recorded in the statsmodels test suite
  (`statsmodels/regression/tests/results/results_regression.py`, class `Longley`, which
  cites NIST StRD). The values were not compared against the NIST file itself; the file
  header says so. Treat this one as "NIST values via a secondary source".
* Not available offline, therefore not tested: NIST *Pontius* (and the other linear
  least-squares files).

If the environment later allows it, re-download the files from NIST and diff them against
these copies.

### Nonlinear least squares: what is checked and how well it agrees

Every dataset is fitted from **both** NIST starting points with
`fit_curve(x, y, "user_defined", equation=..., initial=<start>)` (safe equation parser +
automatic-differentiation Jacobian + fitter). Checked against the certified values:
every parameter, every standard error, residual sum of squares, residual standard
deviation, degrees of freedom.

Test tolerances: parameters rel 1e-6, standard errors rel 1e-5, residual SS rel 1e-8
(from both starts, for all datasets, with the single exception noted below).

Observed agreement (digits = -log10 of the relative error of the *worst* parameter /
standard error; capped at 11 because NIST certifies 11 digits). All 26 datasets converge
from both starting points, including the "higher difficulty" ones; there are no known
failures to list.

| Dataset | NIST difficulty | n / parameters | Start 1: parameters, SE, RSS | Start 2: parameters, SE, RSS |
|---|---|---|---|---|
| Misra1a | lower | 14 / 2 | 11.0, 10.8, 10.5 | 11.0, 10.8, 10.5 |
| Chwirut2 | lower | 54 / 3 | 10.9, 11.0, 11.0 | 10.9, 11.0, 11.0 |
| Chwirut1 | lower | 214 / 3 | 10.7, 10.9, 11.0 | 10.7, 10.9, 11.0 |
| Lanczos3 | lower | 24 / 6 | 10.5, 11.0, 10.6 | 10.5, 11.0, 10.6 |
| Gauss1 | lower | 250 / 8 | 10.6, 10.6, 11.0 | 10.6, 10.6, 11.0 |
| Gauss2 | lower | 250 / 8 | 10.3, 10.4, 10.6 | 10.3, 10.4, 10.6 |
| DanWood | lower | 6 / 2 | 11.0, 11.0, 11.0 | 11.0, 11.0, 11.0 |
| Misra1b | lower | 14 / 2 | 11.0, 10.8, 11.0 | 11.0, 10.8, 11.0 |
| Kirby2 | average | 151 / 5 | 10.5, 10.7, 11.0 | 10.5, 10.7, 11.0 |
| Hahn1 | average | 236 / 7 | 10.8, 10.5, 10.6 | 10.8, 10.5, 10.6 |
| MGH17 | average | 33 / 5 | 10.8, 10.6, 11.0 | 10.8, 10.6, 11.0 |
| Lanczos1 | average | 24 / 6 | 10.6, 3.0, 2.7 | 10.6, 3.1, 2.8 |
| Lanczos2 | average | 24 / 6 | 10.4, 10.4, 10.3 | 10.4, 10.3, 10.1 |
| Gauss3 | average | 250 / 8 | 10.5, 10.5, 11.0 | 10.5, 10.5, 11.0 |
| Misra1c | average | 14 / 2 | 10.8, 10.6, 11.0 | 10.8, 10.6, 11.0 |
| Misra1d | average | 14 / 2 | 11.0, 11.0, 11.0 | 11.0, 11.0, 11.0 |
| Roszman1 | average | 25 / 4 | 10.9, 11.0, 11.0 | 10.9, 11.0, 11.0 |
| ENSO | average | 168 / 9 | 8.2, 9.4, 11.0 | 8.2, 9.4, 11.0 |
| MGH09 | higher | 11 / 4 | 9.2, 9.3, 11.0 | 9.7, 9.8, 11.0 |
| Thurber | higher | 37 / 7 | 10.4, 9.7, 11.0 | 10.1, 9.4, 11.0 |
| BoxBOD | higher | 6 / 2 | 11.0, 10.4, 10.4 | 11.0, 10.4, 10.4 |
| Rat42 | higher | 9 / 3 | 11.0, 10.6, 11.0 | 11.0, 10.6, 11.0 |
| MGH10 | higher | 16 / 3 | 10.9, 11.0, 11.0 | 10.9, 11.0, 11.0 |
| Eckerle4 | higher | 35 / 3 | 10.5, 10.5, 10.7 | 10.5, 10.5, 10.7 |
| Rat43 | higher | 15 / 4 | 11.0, 10.7, 11.0 | 11.0, 10.7, 11.0 |
| Bennett5 | higher | 154 / 3 | 11.0, 10.4, 11.0 | 11.0, 10.4, 11.0 |

(Numbers measured on Linux x86-64, Python 3.11, NumPy 2.4, SciPy 1.17; the last digit or two
can move on another platform, which is why the test tolerances are much looser.)

Known limitations and oddities:

* **Lanczos1**: the data were generated from the model and rounded to 14 digits, so the
  certified residual SS is 1.4e-25 and the residuals (about 1e-13) are at the rounding
  level of IEEE double precision. The parameters agree to 10 digits, but the residual SS
  and therefore the standard errors only to about 3 digits. No double-precision program
  can do much better; the test requires 5% for SE/RSS on this dataset only. (The fitter
  also labels this fit "perfect fit", which is the honest description at double precision.)
* **Rat43**: the NIST header prints "Degrees of Freedom: 9" for 15 observations and 4
  parameters. The certified residual standard deviation equals sqrt(RSS / 11), so the
  header line is a misprint; the test uses 11.
* **ENSO**: agreement "only" 8 digits for the parameters; the final Gauss-Newton polishing
  stalls slightly earlier on this 9-parameter trigonometric model.
* Convergence from both starts is obtained by running trust-region-reflective **and**
  MINPACK Levenberg-Marquardt from the given start and keeping the better converged fit.
  No dataset needed the random restarts.

Additional NIST checks through the **built-in models with automatic starting values** (no
start supplied): Misra1a and BoxBOD via `one_phase_association` with `Y0` fixed at 0,
Eckerle4 via `gaussian`, Misra1d via `michaelis_menten` (Km = 1 / b2). Parameters rel 1e-7
or better, RSS rel 1e-8.

### Linear least squares

| Dataset | Kind | Checked | Agreement |
|---|---|---|---|
| Norris | NIST (original file) | slope, intercept, both SE, residual SD, R squared, F, regression/residual SS and MS via `linear_regression` | rel 1e-10 or better (observed about 1e-13) |
| Longley | NIST values via statsmodels (see above) | 7 coefficients and SE, residual SD, R squared, F, residual SS via `multiple_regression`; VIF check must flag multicollinearity | rel 1e-8 required (observed 1e-11 coefficients, 1e-13 SE) |

## 2. Published worked examples (typed in by hand)

| Example | Citation | Values reproduced | Used in |
|---|---|---|---|
| Puromycin, treated enzyme (12 points) | Treloar MA (1974) *Effects of Puromycin on Galactosyltransferase in Golgi Membranes*, MSc thesis, Univ. of Toronto; analysed in Bates DM, Watts DG (1988) *Nonlinear Regression Analysis and Its Applications*, Wiley (Appendix A1.3 and ch. 2-3) | Michaelis-Menten fit: Vmax = 212.68 (SE 6.947), Km = 0.06412 (SE 0.008281), residual SE 10.93 on 10 df (reproduced to every printed digit) | `test_nonlinear.py::test_puromycin_michaelis_menten_bates_watts`, profile-likelihood tests |
| BOD (6 points) | Marske DM (1967) MSc thesis, Univ. of Wisconsin; in Bates & Watts (1988) Appendix A1.4 | y = t1 (1 - exp(-t2 x)): t1 = 19.143, t2 = 0.5311, residual SS 25.99, s^2 = 6.498 on 4 df | `test_nonlinear.py::test_bod_bates_watts` |
| Anscombe's quartet (4 x 11 points) | Anscombe FJ (1973) Graphs in statistical analysis. *Am Stat* 27:17-21 | For all four sets: line y = 3 + 0.5x, regression SS 27.50, residual SS 13.75 (9 df), SE of slope 0.118, R squared 0.667 (r = 0.816) | `test_regression.py`, `test_correlation.py` |

Note on Anscombe: the paper gives one set of summary numbers for all four data sets, but
the sets match them only approximately (for example the residual SS is 13.74, 13.78, 13.76
and 13.74). The test tolerances (0.001 on slope, r and R squared; 0.005 on the intercept;
0.04 on the sums of squares) reflect that, not any adjustment of the published values.

Examples tried and dropped: none had to be dropped for failing to reproduce. I did not
include a published dose-response (four-parameter logistic) example because I am not
certain enough of any published data set *and* its fitted values; the four-parameter
logistic is validated by NIST-style cross-checks instead (section 3).

## 3. Cross-implementation checks

| Analysis | Checked against | Tolerance |
|---|---|---|
| `pearson` | `scipy.stats.pearsonr` (r, P for all three alternatives, `confidence_interval`), defining formulas | 1e-10 / 1e-9 |
| `spearman` | `scipy.stats.spearmanr` (with and without ties, three alternatives); classic 1 - 6 sum(d^2) / (n(n^2 - 1)); **exact P by brute-force enumeration of all permutations written in the test**; Bonett-Wright interval from its formula | 1e-12 / 1e-8 |
| `correlation_matrix` | `scipy` pair by pair on pairwise-complete rows, `pandas.DataFrame.corr`, shared `multiplicity.adjust_pvalues` | 1e-10 |
| `linear_regression` | `scipy.stats.linregress`; `statsmodels` OLS (coefficients, SE, CI, P, adjusted R squared, residuals); `get_prediction().summary_frame()` for the confidence and prediction bands at 95% and 99%; through-origin vs OLS without constant; `t_test` for a non-zero null slope | 1e-10 / 1e-8 |
| X-intercept (Fieller) | first principles: at each limit the confidence band of the line touches Y = 0 | 1e-9 |
| Breusch-Pagan (Koenker), Durbin-Watson | `statsmodels.stats.diagnostic.het_breuschpagan`, `statsmodels.stats.stattools.durbin_watson` | 1e-9 |
| Runs test | exact distribution by enumeration of all sign arrangements (in the test); z statistic vs `statsmodels.sandbox.stats.runs.runstest_1samp`; exact vs normal approximation | 1e-12 / 1e-10 |
| Lack-of-fit F test | `statsmodels` `anova_lm` of the model vs one mean per X | 1e-9 |
| AIC / AICc | formula; identity with the likelihood-based statsmodels AIC (constant offset) | 1e-10 |
| `compare_slopes` | `statsmodels` formula models `y ~ x*g`, `y ~ x+g`, `y ~ x` with `anova_lm` (2 and 3 groups, unequal n) | 1e-9 |
| `multiple_regression` | `statsmodels.formula.api.ols` with categorical terms and interactions; `variance_inflation_factor`; standardized betas vs a fit to z-scored variables | 1e-9 / 1e-6 (P) |
| `fit_curve`, every built-in model | `scipy.optimize.curve_fit` with tight tolerances: parameters (1e-6), SE from `pcov` (1e-4: curve_fit differentiates numerically), SS, R squared, Sy.x, AICc, BIC, parameter correlation matrix | as stated |
| built-in Jacobians, AD Jacobian of user equations | central finite differences | 1e-6 |
| weighting 1/Y, 1/Y^2, 1/SD^2 | `curve_fit(sigma=...)` | 1e-7 / 1e-4 |
| confidence / prediction band of a curve | delta method computed in the test from `curve_fit`'s covariance; for the `linear` model also statsmodels' exact bands | 2e-4 / 1e-8 |
| profile-likelihood CI | equals the t interval for a model linear in its parameters (1e-6); for Michaelis-Menten the profiled SS at each limit equals SS_best (1 + F / df), computed independently | 1e-6 |
| `compare_fits` | F and AICc from two independent `curve_fit` fits; line vs quadratic vs `anova_lm` | 1e-7 |
| `compare_datasets` | independent global fit with `scipy.optimize.least_squares` written in the test; sharing the slope of straight lines reproduces `compare_slopes` exactly | 1e-6 |
| synthetic recovery | noise-free data for all 13 built-in models: true parameters recovered to 1e-6 from the automatic starting values | 1e-6 |
| CI coverage | 200 simulated dose-response experiments: coverage of the 95% CI of LogEC50, Hill slope and Top within 0.89 to 0.99 | loose |
| `equation.py` | plain numpy evaluation; finite-difference derivatives; 60+ injection / malformed inputs rejected with `StatsError`; hypothesis fuzzing (random text, arbitrary unicode, random valid expression trees) | - |

## 4. Conventions worth knowing when comparing with other statistics packages

* AICc uses the least-squares form N ln(SS/N) + 2K + 2K(K+1)/(N-K-1) with K = number of
  fitted parameters + 1. Only differences between models fitted to the same data matter.
* The runs test P value is one-sided (too few runs), exact for n <= 2000.
* `1/y` and `1/y^2` weighting use the **observed** Y values as weights (same as
  `curve_fit(sigma=...)`); some packages weight by the predicted Y instead, which gives
  slightly different estimates.
* Spearman: exact permutation P for n <= 9 without ties, otherwise the t approximation.
* Categorical predictors in multiple regression: reference level = first level in
  alphabetical order; VIFs are per design column (no generalised VIF for multi-level factors).

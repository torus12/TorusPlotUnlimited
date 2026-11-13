# TorusPlot licence and IP audit

- **Subject:** `/home/user/torusplot` at commit `71d17d8` (311 tracked files), owner Torus Washington LLC.
- **Audit date:** 2026-11-13.
- **Question asked:** is there anything that would restrict the owner from selling, close-sourcing or re-licensing the product?
- **Nature of this document:** an engineering audit. It is **not legal advice**. Section 5 lists the points that need a lawyer rather than an engineer.
- **Caveat on scope:** the repository changed while the audit ran (two commits landed: `a262d7c` -> `7457954` -> `71d17d8`, "M8: documentation"). All similarity scans and searches were re-run at `71d17d8`; anything committed later is not covered.

---

## 1. Executive verdict: CLEAR WITH CONDITIONS

**No blocker was found.**

- No GPL, LGPL, AGPL, SSPL, BUSL, Commons-Clause, non-commercial or unlicensed package is imported by the product or sits in its declared dependency closure. `pingouin` and `figrecipe` are neither installed nor imported.
- No copied or closely paraphrased code, docstring, comment, prompt, UI string, test or asset from the ten `_reference/` projects was found by systematic comparison (methods and numbers in section 4).
- All bundled fonts carry SIL OFL 1.1 in their own name tables and ship with their licence texts. The logo and icon are original SVG. All 45 PNG files are the product's own output.
- No third-party trademark is used as a product, preset, palette, style or file-format name. All mentions found are nominative.

**Conditions before a paid release** (details in section 2):

| # | Condition | Applies to |
|---|---|---|
| F-1 | Pin and re-check dependency versions per platform; extend the licence gate to the optional AI package and to binaries inside wheels | every release |
| F-2 | Write end-user licence terms; the current `LICENSE` is a notice, not a grant | every release |
| F-3 | Define the customer deliverable; today the whole repository would ship, including tests and internal provenance notes | every release |
| F-4 | Get a legal opinion on, or replace, the textbook and R-distributed datasets typed into the tests | if tests leave the company |
| F-5 | Correct three inaccuracies in `THIRD_PARTY_NOTICES.md` | every release |
| F-6 | Meet LGPL, notice and source-pointer obligations | only if a frozen/bundled installer is built |

---

## 2. Findings

Severity: **BLOCKER** = stops a sale or closed-source release. **SHOULD-FIX** = fix or decide before a paid release. **NOTE** = low risk, recorded for completeness.

### BLOCKER

None.

### SHOULD-FIX

#### F-1 Dependency licences are checked on the developer machine, not on what the customer installs

- **Where:** `requirements.txt:3-18` (version ranges, no lock file); `run_torusplot.sh:27` and `run_torusplot.bat:53` (`pip install -r requirements.txt` on the customer's machine); `tools/check_licenses.py:38` (reads `requirements.txt` only) and `:153-166` (reads package metadata through `pip-licenses`); `requirements-ai.txt:4`.
- **Evidence:**
  - The two environments already resolve five packages differently: contourpy 1.3.3 / 1.4.0, networkx 3.6.1 / 3.7, numpy 2.4.6 / 2.5.4, scipy 1.17.1 / 1.18.1, tifffile 2026.3.3 / 2026.11.12. All five are still permissive, verified from their licence files.
  - `THIRD_PARTY_NOTICES.md` records the versions of one Linux / Python 3.11 environment. It is not what a customer gets.
  - On Windows the closure gains `pywin32` (required by `mcp`). It is not installed here, so its licence could not be read, and it is not in the notices file. On macOS the closure loses `watchdog`.
  - `requirements-ai.txt` (`anthropic>=1,<2`, lazily imported at `torusplot/ai/providers/anthropic.py:80`) is not covered by the gate and is not installed in either environment, so its closure was **not verified** here.
  - The gate trusts metadata. It passes SciPy on its "BSD License" classifier although the wheel contains an LGPL library (see F-6). That fact is documented by hand in the notices, not detected.
- **Why it matters:** a future release of any of 76 packages can change licence or add a dependency, and the customer's install would pick it up with no check. The owner would be recommending an install set it never reviewed.
- **Remediation:**
  1. Ship a per-platform constraints/lock file with hashes (Windows, macOS, Linux) and have the launchers install with it.
  2. Generate `THIRD_PARTY_NOTICES.md` from those lock files, one section per platform.
  3. Run the gate over `requirements-ai.txt` as well.
  4. Add a scan of `*.libs/`, `*.dylibs/` and wheel SBOMs (`*.dist-info/sboms/`) to the gate.

#### F-2 `LICENSE` is an all-rights-reserved notice, not a licence a customer can use

- **Where:** `LICENSE:1-9`; `pyproject.toml:11`.
- **Evidence:** the text says no part may be "copied, modified, distributed, sublicensed, or sold without the prior written permission". It grants no right to install or run. It also makes no carve-out for the third-party parts that are shipped in the same folder under their own terms (the twelve OFL font files).
- **Why it matters:** this does not restrict the owner, but selling needs terms. A flat "no modification" clause also conflicts with LGPL section 6 if LGPL libraries are ever bundled (F-6).
- **Remediation:** have counsel draft a EULA. Add one sentence to `LICENSE` stating that third-party components (fonts, and any bundled libraries) remain under their own licences, which prevail for those components.

#### F-3 The customer deliverable is undefined, so the whole repository would ship

- **Where:** `.gitattributes:1-6` (no `export-ignore` rules); `pyproject.toml:22-23` (the wheel excludes `tests*` and `_reference*`, but the documented distribution is the repository plus launcher, see `README.md:37-45`).
- **Evidence:** a repository zip contains `tests/` (39 NIST data files and the datasets of F-4), `docs/REFERENCE_NOTES.md` (761 lines of study notes on GPL, AGPL and unlicensed projects), `DECISIONS.md`, `docs/MODEL_LICENSE_RESEARCH.md` and `tools/`.
- **Why it matters:** none of these is needed to run the app. Shipping them widens what is redistributed (F-4, F-14) and hands internal provenance records to customers and competitors.
- **Remediation:** build releases with `git archive` and add `export-ignore` for `tests/`, `tools/`, `docs/REFERENCE_NOTES.md`, `docs/MODEL_LICENSE_RESEARCH.md`, `docs/dev/`, `docs/audit/`, `DECISIONS.md`, `PROGRESS.md`, `FINAL_REPORT.md`, `.github/`. Never zip the working folder: it contains `_reference/` (213 MB of GPL/AGPL/unlicensed code, see F-9).

#### F-4 Tests embed datasets from copyrighted textbooks and from compilations distributed under the GPL

- **Where (data typed into test files):**

  | Dataset | File:line | Stated primary source | Also distributed as |
  |---|---|---|---|
  | "Bushtucker trial", 32 values | `tests/stats/test_repeated.py:40-42` | Field, *Discovering Statistics Using IBM SPSS Statistics* (SAGE) | textbook example only |
  | Battery life, 36 values | `tests/stats/test_factorial.py:45` | Montgomery, *Design and Analysis of Experiments* (Wiley), Example 5.1 | textbook |
  | Tooth growth, 60 values | `tests/stats/test_factorial.py:53` | Crampton 1947, *J Nutr* | R `datasets::ToothGrowth` (GPL) |
  | Plant growth, 30 values | `tests/stats/test_oneway.py:32`, `tests/stats/test_posthoc.py:25` | Dobson 1983 | R `PlantGrowth` |
  | Mucociliary, rounding times | `tests/stats/test_oneway.py:39,61`, `tests/stats/test_posthoc.py:567` | Hollander & Wolfe 1973 | R help examples |
  | New York ozone 1973 | `tests/stats/test_oneway.py:47` | Chambers et al. 1983 | R `airquality` |
  | Warp breaks, 54 values | `tests/stats/test_posthoc.py:32` | Tippett 1950 | R `warpbreaks` |
  | Sleep, 20 values | `tests/stats/test_two_group.py:31-33` | Student 1908 | R `sleep` |
  | Puromycin, BOD | `tests/stats/test_nonlinear.py:97-98,117-118` | Treloar 1974; Marske 1967 (via Bates & Watts 1988) | R `Puromycin`, `BOD` |
  | Anscombe's quartet | `tests/stats/test_correlation.py:35-37` | Anscombe 1973 | R `anscombe` |
  | 6-MP leukaemia trial, 42 values | `tests/stats/test_survival.py:32-36` | Freireich et al. 1963, *Blood* | many textbooks; also in `_reference/Helix` (F-15) |
  | Five contingency tables | `tests/stats/test_contingency.py:30-35` | Fisher 1935; NEJM 1988; GSS 2000; Graubard & Korn 1987 | Agresti textbooks |

- **Evidence:** the project's own records say these were "typed from memory", not copied from a file (`tests/fixtures/SOURCES.md:124-146`, `tests/fixtures/sources/factorial_repeated.md:36-56`, `DECISIONS.md:233` D-022). The comparison in section 4 found none of these arrays in `_reference/` except the 6-MP data. R is not installed, so a comparison against R's files was not possible. Expected results for several tests are R output recalled from memory (`tests/fixtures/SOURCES.md:136-140`).
- **Why it matters:**
  - Measured values from primary literature are facts and are generally not protected by copyright in the United States. That argument is weaker for (a) the Field example, whose numbers appear to be invented by the textbook author rather than measured, and (b) jurisdictions with database rights.
  - The tests are not imported by the product and are excluded from the wheel, so this does not touch what runs on the customer's machine. It matters if the repository itself is sold, licensed or published (F-3, F-17).
- **Remediation, in order of cost:**
  1. Keep `tests/` out of the customer deliverable (F-3).
  2. Replace the Field example with synthetic data checked against `statsmodels.AnovaRM` plus a first-principles Mauchly computation; the Mauchly second-order P value it validates can be asserted that way.
  3. Delete `TOOTH` as the project already suggests; the battery-life example covers the same analysis.
  4. Ask counsel whether the rest need any action (question Q-2).

#### F-5 `THIRD_PARTY_NOTICES.md` is complete for one environment but has three inaccuracies

- **Where:** `THIRD_PARTY_NOTICES.md:4`, `:89-95`, `:100`; generator text at `tools/check_licenses.py:242-277` (the font sentence is `:275`).
- **Evidence:**
  1. Line 100 says the bundled fonts are under "SIL Open Font License 1.1 / Apache-2.0". All twelve font files state OFL 1.1 only; no Apache-licensed font is bundled.
  2. The 76 rows match the 76 packages I enumerated in `.venv` exactly (names and versions), but see F-1: versions differ in a fresh install and `pywin32` is missing for Windows.
  3. The "compiled wheels" note covers NumPy and SciPy only. Pillow (18 bundled libraries), PyArrow (bundled Arrow C++ and dependencies), cryptography (statically linked OpenSSL) and matplotlib (FreeType, Qhull, fonts) are not mentioned.
- **Why it matters:** for the current model (customer's pip downloads the wheels) TorusPlot does not redistribute these packages, so no notice duty arises from them and the file is informational. An inaccurate notices file is still a poor basis for a sale or due-diligence review.
- **Remediation:** fix line 100 in the generator; regenerate per platform from lock files (F-1); extend the note to name every wheel that bundles third-party binaries.

#### F-6 Conditions that apply only if a frozen or bundled installer is ever built

Today TorusPlot ships no dependency binaries: nothing below is tracked in git. If the product is packaged with PyInstaller, Briefcase, an MSI or similar, the owner becomes the distributor of everything in the closure and these apply.

- **LGPL-2.1-or-later `libquadmath`** in `numpy.libs/` and `scipy.libs/` (numpy 2.4.6 and 2.5.4, scipy 1.17.1 and 1.18.1; stated at `numpy-*.dist-info/licenses/LICENSE.txt:915-919` and `scipy-*.dist-info/LICENSE.txt:914-918`).
  - Keep it a separate, replaceable shared library. Ship the LGPL text.
  - Provide the library's source or a written offer for it (LGPL-2.1 section 4/6). `DECISIONS.md:60` (D-007) mentions the notice but not the source obligation.
  - The EULA must not forbid modification of, or reverse engineering for debugging modifications to, that library (F-2).
- **GPL-3.0-or-later WITH GCC-exception-3.1 `libgfortran`** in the same folders (`LICENSE.txt:131-135`). The exception permits combination with proprietary code. Ship the licence and exception texts.
- **Windows wheels were not inspected** (F-18). Pillow's SBOM lists FriBiDi (LGPL-2.1+) as an optional runtime-loaded library and libimagequant (GPL-3.0+) as optional. In the Linux wheels in both environments libimagequant is **not** compiled in (`PIL.features.check_feature("libimagequant")` is False) and no FriBiDi file is bundled. Check the Windows and macOS wheels for `fribidi*.dll`/`.dylib` before freezing.
- **Licence texts and NOTICE files** must travel with the binaries: 119 licence files across the 76 packages, including the Apache `NOTICE` files of `pyarrow` and `requests`. `streamlit` 1.67.0 ships **no licence file at all** in its wheel; add the Apache-2.0 text and the upstream third-party notices for its 26 MB JavaScript and font bundle.
- **MPL-2.0 `certifi`**: state where the source of the CA bundle is available.
- **Bundled sample data and fonts inside wheels**: exclude `skimage/data/`, `statsmodels/datasets/` (34 folders), `lifelines/datasets/` (32 files) and `matplotlib/mpl-data/sample_data/`; the product uses none of them and their individual provenance was not audited. Matplotlib's fonts (DejaVu, STIX, BaKoMa Computer Modern) are redistributable with their notices.
- **The interpreter itself** (PSF licence, bundled OpenSSL/Tcl/zlib notices) and the Microsoft C runtime redistribution terms on Windows.

### NOTE

#### F-7 The enum value `"gp"` alludes to the market leader's four-star convention

- **Where:** `torusplot/spec/figure.py:293`; `torusplot/stats/result.py:230,233,238`; `torusplot/plot/brackets.py:37`; `torusplot/export/methods.py:129`; published in the schema at `docs/PROMPT_TEMPLATE_FOR_OUTSIDE_AI.md:995-998`.
- **Evidence:** `star_style: Literal["gp", "apa"]`. No trademark is spelled out anywhere: the words "GraphPad" and "Prism" occur only in tests that ban them (`tests/plot/test_styles_palettes.py:62`, `tests/stats/test_registry_integration.py:129,153`, `tests/test_docs.py:23`, `tools/build_user_guide.py:2605`) and nowhere in product code or user documentation.
- **Why it matters:** negligible trademark risk for a two-letter value. It is part of a public JSON schema, so it is cheap to rename now and a breaking change later.
- **Remediation:** rename to a descriptive value such as `"four_star"` / `"three_star"`.

#### F-8 Liberation Sans binaries came from a Debian package, not the upstream release

- **Where:** `torusplot/plot/fonts/LiberationSans/*.ttf`; `DECISIONS.md:144` (D-013); `torusplot/plot/fonts/README.md:6,12`.
- **Evidence:** the name tables state "Version 2.1.5" and "Licensed under the SIL Open Font License, Version 1.1"; `OFL.txt` declares the Reserved Font Name "Liberation". The README calls the files unmodified and cites the upstream GitHub release, while D-013 records they were taken from Debian's `fonts-liberation` because the release download was blocked.
- **Why it matters:** OFL permits bundling and embedding either way. A distribution rebuild is not guaranteed byte-identical to the upstream release, which is what "unmodified" and the Reserved Font Name clause refer to.
- **Remediation:** download the upstream 2.1.5 release, compare hashes, replace the four files if they differ, and correct the README's source line.

#### F-9 `_reference/` lives inside the working tree, and "clean-room" overstates the process

- **Where:** `.gitignore:1-2`; `docs/REFERENCE_NOTES.md:1-15`; `DECISIONS.md:92`.
- **Evidence:**
  - `_reference/` is ignored (`git check-ignore` confirms), has zero tracked files and never appears in history. Licences read from the clones: pingouin GPL-3.0-only, figrecipe AGPL-3.0-only, Helix GPL-3.0-or-later, LabPlot GPL-2.0-or-later, MatPlotAgent and paper-plot-skills no licence, PlotLLM, tidyplots and statannotations MIT, Graphite MIT OR Apache-2.0.
  - The notes file describes those projects' features in detail and repeats a few short phrases from their documentation (for example `docs/REFERENCE_NOTES.md:171` against `_reference/Helix/README.md:40`; `:207-208` against `_reference/Helix/METHODS.md:20`; `:419-420` against `_reference/paper-plot-skills/plot-from-image/SKILL.md:49`). Each is a feature list of under 12 words.
  - A clean room in the legal sense separates the people who read the original from the people who write the new code. Here the same author studied the GPL/AGPL sources and wrote the product.
- **Why it matters:** ideas and feature lists are not protected, and my comparison found no copied expression, which supports independent creation. The label "clean-room" claims more than the record shows and could be challenged in a dispute.
- **Remediation:** call the document a "study log / independent implementation record"; keep it internal (F-3); move `_reference/` outside the repository folder so no archive of the folder can include it.

#### F-10 All commits are authored by an AI system

- **Where:** `git log`: 13 commits, author and committer `Claude <noreply@anthropic.com>`, each with a `Co-Authored-By: Claude ...` trailer.
- **Evidence:** no file carries a third-party copyright header; every notice names Torus Washington LLC (`LICENSE:1`, `README.md:9`, `app/assets/*.svg`).
- **Why it matters:** this does not suggest third-party ownership. It does raise two lawyer questions: whether AI-generated code and artwork attract copyright the owner can enforce, and which Anthropic terms governed the session (Q-1).
- **Remediation:** none technical. Keep the commit history as provenance evidence.

#### F-11 Terminology follows the market leader; the product's own name is not cleared

- **Where:** `torusplot/data/tables.py:36-44` (table kinds Column, XY, Grouped, Multiple variables, Contingency); analysis names such as "Ordinary two-way ANOVA" (`torusplot/stats/recommend.py:379`), "Log-rank (Mantel-Cox) test", "ROUT" (`torusplot/stats/outliers.py`), "SuperPlot" (`torusplot/spec/figure.py`).
- **Evidence:** these are generic statistical terms or terms from the academic literature (ROUT: Motulsky & Brown 2006; SuperPlots: Lord et al. 2020). The same terms appear in Helix. No UI string longer than 31 characters matches any reference project.
- **Why it matters:** functional vocabulary is not protected by copyright; I found no logo, layout or wording copied. Whether "TorusPlot" / "Torus" is free to use as a mark was not checked (Q-4).

#### F-12 AI features: no weights bundled, defaults not verified against the model cards

- **Where:** `torusplot/ai/settings.py:36-38,47-84`; `docs/MODEL_LICENSE_RESEARCH.md`; `.env.example`; `app/views/settings.py:18-21`.
- **Evidence:** no model weights, API keys or vendor SDK are tracked. Defaults name `qwen2.5:7b` and `granite3.2-vision` (Apache-2.0 according to the vendors' GitHub READMEs). The research file states the Ollama and Hugging Face pages could not be read. Cloud providers are bring-your-own-key; Ollama, Gemini and OpenAI are reached with the standard library, Anthropic through the optional MIT SDK. `torusplot/ai/recorded/*.json` are hand-written canned answers for the offline mock, not model output.
- **Why it matters:** the weights' licences bind the user who downloads them, not the product. The product's recommendations should still be right.
- **Remediation:** run `ollama show <tag> --license` for the two defaults and record the result. Mention in the EULA that provider terms apply to the user's own account.

#### F-13 Two runtime dependencies are used only by tests

- **Where:** `requirements.txt:7-8` (`scikit-posthocs`, `lifelines`).
- **Evidence:** AST scan of `torusplot/`, `app/` and `cli.py` finds no import of either; they are imported in `tests/stats/test_posthoc.py:12` and `tests/stats/test_survival.py:19-21`. Removing both drops four packages from the closure (autograd, autograd-gamma, lifelines, scikit-posthocs: 76 -> 72).
- **Remediation:** move them to `requirements-dev.txt`.

#### F-14 NIST reference files were copied from other packages' distributions

- **Where:** `tests/fixtures/nist/*.dat` (39 files); `tests/fixtures/sources/regression.md:17-40`; `tests/fixtures/sources/oneway.md:29-34`.
- **Evidence (hash comparison):** 11 ANOVA files and `Norris.dat` are byte-identical to the copies inside the installed SciPy wheel (BSD-3-Clause). 26 nonlinear-regression files are byte-identical to `NIST_STRD/` in the lmfit 1.3.4 source distribution (BSD-3-Clause; its `LICENSE` was read). `Longley.dat` is hand-assembled from statsmodels data and test values and says so in its header (`tests/fixtures/nist/Longley.dat:1-11`).
- **Why it matters:** the data are works of the US government. They are in tests only. Status outside the United States and NIST's citation request are for counsel (Q-3).

#### F-15 One classic dataset is also present in a GPL reference project

- **Where:** `tests/stats/test_survival.py:32-36,187` against `_reference/Helix/tests/reference.ts:270-293` and `_reference/Helix/tests/reference.json:13166`.
- **Evidence:** both contain the 42 remission times of the 6-mercaptopurine trial (Freireich et al. 1963). TorusPlot stores two time lists and two event lists; Helix stores `[time, event]` pairs. No surrounding code or expected values match.
- **Judgement:** independent use of a dataset printed in most survival textbooks. Covered by F-4.

#### F-16 Generated methods text names SciPy and statsmodels

- **Where:** `torusplot/export/methods.py:112-113`: "produced with TorusPlot ... whose statistics engine is built on SciPy and statsmodels."
- **Why it matters:** factual attribution is fine. BSD-3 clause 3 forbids using the projects' names to endorse or promote a derived product without permission, so keep this wording out of marketing copy.

#### F-17 The repository has a GitHub remote whose visibility was not checked

- **Where:** `git remote -v` -> `github.com/torus12/torusplotunlimited`; `DECISIONS.md:8` (D-001).
- **Why it matters:** if that repository is public, the source is already published and GitHub's terms let other users view and fork it. That does not transfer ownership, but it weakens any trade-secret position and makes F-4 a live distribution question. I did not query GitHub.
- **Remediation:** confirm it is private.

#### F-18 Platform coverage of this audit

- Both environments are Linux x86-64 (Python 3.11.18 and 3.12.3). Windows and macOS wheels bundle different binaries and were not inspected. The Windows-only dependency `pywin32` was not available to read.

#### F-19 Smaller observations

- `torusplot/plot/palettes.py:31`: two of the six "Blue-Orange" values (`#F28E2B`, `#FFBE7D`) coincide with a well-known 20-colour categorical palette. Individual colour values are not protectable; the module docstring's "designed for TorusPlot" is slightly generous. The Okabe-Ito palette (`:23`) and matplotlib default colours used in tests are free to use.
- `torusplot/plot/colorblind.py:26-38` and `torusplot/stats/assumptions.py:245-256` contain published numeric constants (Machado et al. 2009; D'Agostino & Stephens 1986). The same constants appear in statsmodels and seaborn; the surrounding code differs.
- The licence overrides in `tools/license_overrides.yaml` are correct: numpy's expression is all-permissive (the GCC runtime libraries are a separate matter, F-6); cffi's file is the MIT No Attribution text; Pillow's file is the historical PIL notice that SPDX names MIT-CMU.
- `MPL-2.0` is on the gate's allowlist (`tools/check_licenses.py:55`). It is a file-level copyleft, acceptable here because `certifi` is used unmodified, but it is not "permissive" in the strict sense the README claims (`README.md:176`).
- `.git/info/exclude` hides `docs/audit/` and `tests/audit/` from `git status`. This report will not show as untracked.

---

## 3. Dependency enumeration

Method: an independent script walked `importlib.metadata` from the 16 names in `requirements.txt`, honouring extras and environment markers, in both environments. For each distribution it read `License`, `License-Expression` and classifiers, located every licence/notice/copying file in `*.dist-info` and inside the package, classified each text by signature phrases, listed `*.libs` contents, and grepped all texts for copyleft and non-commercial terms. Every hit was read in context. The project's own gate was also run in both environments (both pass, 76 packages).

- **Runtime closure:** 76 distributions in each environment; 5 version differences; 0 missing. Windows adds `pywin32`; macOS drops `watchdog`.
- **Licence files read:** 119 in `.venv`. One package has none (`streamlit`).
- **Wheels with vendored shared libraries:** numpy (3 files), scipy (5), pillow (18). 21 packages contain compiled extensions.
- **Wheel SBOMs read:** cryptography (39 Rust crates + OpenSSL), pydantic-core (101), rpds-py (17), pillow (24 components).
- **Copyleft terms found and resolved:** all in section F-6, or benign (PSF licence history text in pandas, typing_extensions, et_xmlfile; FreeType's FTL-or-GPL choice in matplotlib and Pillow; xz build-script note in Pillow; LLVM exception and OpenSSL wording in PyArrow).
- **Third-party imports in product code (AST):** PIL, matplotlib, mcp, numpy, openpyxl, pandas, pydantic, scipy, skimage, statannotations, statsmodels, streamlit, yaml. All declared. Lazy import: `anthropic` only. No `pingouin`, `figrecipe`, `seaborn`, `lifelines` or `scikit_posthocs` import in product code.
- **Optional:** `anthropic` (not installed, not verified; upstream states MIT).
- **Developer tools (not shipped, not imported by the app):** coverage 7.17.3 (Apache-2.0), hypothesis 6.170.15 (MPL-2.0), iniconfig 2.3.1 (MIT), pip-licenses 5.5.5 (MIT), pluggy 1.7.0 (MIT), prettytable 3.18.0 (BSD-3), pygments 2.21.0 (BSD-2), pytest 9.2.0 (MIT), pytest-cov 7.1.0 (MIT), ruff 0.17.5 (MIT; its SBOM lists three MPL-2.0 crates), sortedcontainers 2.4.0 (Apache-2.0), wcwidth 0.9.4 (MIT). `tools/make_screenshots.js` needs Playwright (Apache-2.0), also developer-only and not vendored.


---

## 4. What was checked and found clean

### 4.1 Copied code, text and data: method

**Corpora**

- *Product:* every tracked text file at `71d17d8` (243 text files; 141 Python files in `torusplot/`, `app/`, `tools/`, `tests/`, `examples/`, `cli.py`).
- *Reference:* every text file under `_reference/` except `.git` (3,864 files: Python, R, C/C++ headers and sources, TypeScript, Rust, Markdown, YAML, JSON, CSV, HTML, Qt `.ui`, notebooks).
- *Dependencies (second pass, for attribution duties):* 3,665 Python files of the installed scipy, statsmodels, scikit-posthocs, lifelines, seaborn, statannotations, matplotlib, scikit-image, mcp, streamlit, autograd, formulaic, patsy, pandas core/io, pydantic, openpyxl, Pillow, hypothesis, anyio, PyYAML.

**Techniques.** Each file was tokenised, every run of *k* consecutive tokens was hashed (64-bit), and product hashes were looked up in the corpus index. A "run" is a maximal stretch of consecutive matching *k*-grams. Every run at or above the threshold was read on both sides.

| # | Technique | k | Report threshold | Reference index | Product shingles | Matching shingles | Runs over threshold |
|---|---|---|---|---|---|---|---|
| 1 | Language-agnostic tokens, lower-cased, comments included, all file types | 12 | 20 tokens and 6 distinct words | 13,617,338 | 1,050,663 | 30,014 (2.86%) | 17 |
| 2 | Python tokens, exact (comments dropped) | 8 | 14 tokens and 6 distinct names | 759,594 | 486,282 | 8,344 (1.72%) | 33 |
| 3 | Python tokens with identifiers, numbers and strings normalised (detects renaming) | 30 | 50 tokens | 729,163 | 483,210 | 4,393 (0.91%) | 53 |
| 4 | Words only (prose, docstrings, comments, UI text) | 7 | 9 words | 4,384,121 | 405,360 | 1,437 (0.35%) | 171 |
| 5 | Words only, import lines removed | 5 | 7 words, 5 content words | 3,852,650 | 392,726 | 3,991 | 237 |
| 6 | Numbers with 3 or more significant digits | 4 | 5 numbers | 396,508 | 50,678 | 72 | 2 |
| 7 | Any numbers, sliding window | 8 | non-trivial run | 403,027 | 30,090 | 1,817 | 14 |
| 8 | Numeric groups (list literals, CSV columns, table rows of 6 or more numbers) compared as sorted multisets, so reordering does not hide a match | - | identical multiset | 2,808 groups | 752 groups | 9 | 9 |
| 9 | Function level: 6,947 reference functions against 2,382 product functions; shared 5-grams (exact) and 12-grams (normalised), then `difflib` ratio | - | containment 0.30 or more, 40 or more tokens | - | - | 121 candidate pairs | all read: none copied |
| 10 | Same-named functions, `difflib` ratio on exact tokens | - | all pairs listed | - | - | 209 pairs | highest 0.635 |
| 11 | String literals of 12 to 300 characters, exact match after case and whitespace folding | - | any | 31,139 distinct | 7,436 | 190 | longest 31 characters |
| 12 | SHA-256 of binary assets (images, fonts, PDFs) | - | identical file | 642 reference assets | 45 PNG, 12 TTF, 2 SVG | 0 | 0 |

**Every non-trivial match, with judgement**

| Product | Reference | What matches | Judgement |
|---|---|---|---|
| `torusplot/mcp_server/session.py:133-135` | `_reference/figrecipe/tests/develop/test__gallery_templates_packaged.py:67-72` | Set of Windows reserved device names built with two generator expressions (56 tokens after lower-casing) | Common idiom. Different case, container type, variable name and purpose. Independent. |
| `torusplot/ai/__init__.py:83-84`, `torusplot/mcp_server/__init__.py:33` | `_reference/figrecipe/src/figrecipe/__init__.py:273-274` | `raise AttributeError(f"module {__name__!r} has no attribute {name!r}")` | The standard PEP 562 line; also in anyio and many libraries. Independent. |
| `torusplot/plot/render.py:557-560` | `_reference/figrecipe/src/figrecipe/_scitex_compat/_scientific.py:283-286` | `try: kde = gaussian_kde(vals) except (np.linalg.LinAlgError, ValueError): return` | Four-line guard around a SciPy call; surrounding functions differ. Independent. |
| `tests/ai/test_package.py:35,47` | 26 figrecipe test files | `subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=...)` | Standard-library idiom. |
| `torusplot/stats/repeated.py:215-227` | `_reference/pingouin/src/pingouin/distribution.py:515-530` | Mauchly's W and Box's second-order correction | Published formula. Structure and variable names differ; TorusPlot uses `3q` where pingouin deliberately uses `3k`. Independent. |
| `torusplot/stats/survival.py:361-362` | `_reference/pingouin/src/pingouin/correlation.py:148` | `.sum(axis=1)` followed by `with np.errstate(divide="ignore", invalid="ignore"):` | NumPy idiom in unrelated functions. |
| `torusplot/stats/result.py:230-243` (`p_to_stars`) | `_reference/figrecipe/src/figrecipe/_wrappers/_stat_annotation.py:51-70` | Same function name; threshold ladder | Ratio 0.479. Different signature, return values and cut-offs. Independent. |
| `torusplot/data/tables.py:148-157` | `_reference/figrecipe/src/figrecipe/_editor/_overrides.py:193-202` | A method returning a dict literal (normalised containment 0.957) | Structure only; every key and attribute differs. |
| `torusplot/stats/multiplicity.py:5-9`, `torusplot/stats/two_group.py:15`, `torusplot/stats/posthoc.py:19-28`, `torusplot/stats/outliers.py:24`, `docs/STATS_METHODS.md` (many) | pingouin docstrings, `_reference/Helix/METHODS.md:144-159` | Titles and page numbers of cited papers | Bibliographic citations. Not copied prose. |
| `torusplot/stats/oneway.py:530`, `torusplot/stats/nested.py:714` | `_reference/Helix/src/stats/column/anova.ts:1` | "one-way ANOVA compares the means of three or more" (10 words) | Textbook definition; the sentences continue differently. |
| `tests/stats/test_survival.py:32-36,187` | `_reference/Helix/tests/reference.ts:270-293` | 6-MP leukaemia trial data | Same public dataset, different layout. See F-15. |
| `tests/stats/test_factorial.py:664` | `_reference/Helix/tests/checks.ts:814` | Nine small integers with the same multiset | Coincidence: different order, different meaning (a covariate against table cells). |
| `torusplot/plot/palettes.py:128`, `torusplot/plot/render.py:939` | figrecipe, seaborn | 0.2126, 0.7152, 0.0722 | WCAG luminance coefficients. |
| `docs/REFERENCE_NOTES.md:70-71,171,207-208,419-420,637-638` | LabPlot, Helix, paper-plot-skills, statannotations documentation | Short feature lists and option names | Study notes describing those projects. See F-9. |
| `.github/workflows/ci.yml:20-30`, `pyproject.toml` | workflow and config files in statannotations, tidyplots, pingouin, figrecipe | GitHub Actions and setuptools/ruff boilerplate | Tool-defined syntax. |
| Three bundled `OFL` texts | `_reference/statannotations/LICENSE` and others | "THE SOFTWARE IS PROVIDED AS IS..." | Licence boilerplate. |

All remaining runs above the thresholds are import blocks, type-annotated signatures, `mkdir(parents=True, exist_ok=True)`, `warnings.catch_warnings()` blocks, or literal lists whose *shape* (not content) matches after normalisation. None of the 3,864 reference files shares a function body, docstring sentence, prompt or constants table with the product.

**Against the installed dependencies:** exact-token pass 8,286,446 reference shingles, 19,254 product matches (3.96%), 81 runs of 14 or more tokens, all signatures and import blocks; normalised pass 111 runs, all literal-list shapes; prose pass 268 runs, citations and imports; numbers 12 runs: the Longley data (F-14), the NIST Grubbs example, published Anderson-Darling and luminance constants, and page numbers. No dependency code was copied into the product, so no extra attribution duty arises.

**Limits of the method.** Shingling finds verbatim and lightly edited copying and renamed-identifier copying. It cannot rule out a re-implementation written while looking at another program's logic, and it compares only against the ten reference projects and the listed packages. R, GraphPad documentation and other outside sources were not available for comparison.

### 4.2 Datasets and fixtures

- `examples/*.csv` (7 files): synthetic. Regenerating them with `torusplot.data.samples.write_examples` into a scratch folder reproduced all seven byte for byte.
- `tests/fixtures/nist/*.dat` (39 files): provenance verified by hash, see F-14.
- `torusplot/ai/recorded/*.json` (10 files): hand-written canned responses for the mock provider; no third-party model output.
- Typed-in published datasets: F-4.

### 4.3 Bundled assets (all 311 tracked files, by type)

| Files | Count | Result |
|---|---|---|
| Python | 151 | Product code and tests. No third-party copyright header or SPDX tag anywhere. |
| PNG | 45 | 18 in `docs/images/`, 27 golden images. 37 carry the PNG tag `Software: TorusPlot 0.1.0`; the 8 `app_*.png` are 2800x1800 screenshots of the app showing only TorusPlot's own logo and synthetic data. None is identical to any of 642 reference assets. |
| NIST `.dat` | 39 | F-14. |
| Markdown | 22 | Project documentation. `docs/REFERENCE_NOTES.md`, `DECISIONS.md`, `docs/MODEL_LICENSE_RESEARCH.md`, `tests/fixtures/**/*.md` are provenance and policy records, not user-facing. |
| TTF | 12 | Inter 4.001 (4 files, "Copyright 2016 The Inter Project Authors"), Source Sans 3 3.052 (4 files, Adobe, Reserved Font Name "Source"), Liberation Sans 2.1.5 (4 files, Google / Red Hat, Reserved Font Name "Liberation"). Read with fontTools: every file's name ID 13 states SIL OFL 1.1, name ID 14 is the OFL URL, and `fsType` is 0 (no embedding restriction). Matching licence texts are shipped: `Inter/OFL.txt`, `SourceSans3/OFL.md`, `LiberationSans/OFL.txt`, and are included in package data (`pyproject.toml:26`). |
| JSON | 10 | Recorded mock responses. |
| YAML | 8 | 7 style presets written for the product (names: Torus Signature, Classic Lab, Journal Single-Column (89 mm), Journal 1.5-Column (120 mm), Journal Double-Column (183 mm), Minimal, Presentation) and the licence overrides. |
| CSV | 7 | Synthetic examples. |
| SVG | 2 | `app/assets/icon.svg`, `app/assets/logo.svg`: 12 and 13 lines of hand-written ellipses and circles, marked as original work of Torus Washington LLC. No resemblance to any reference SVG. |
| Other | 15 | Launchers, CI workflow, configuration, licence texts. |

### 4.4 Trademarks and naming

Searched all tracked text files, and separately the UI strings in `app/` and `torusplot/`.

| Term | Result |
|---|---|
| GraphPad, Prism | Zero occurrences outside the four tests/tools that ban them. See F-7 for the `"gp"` value. |
| Nature, Science, Cell, ACS, Elsevier, Lancet, NEJM, JAMA, PNAS | Only in bibliographic citations (for example "J Cell Biol", "N Engl J Med"). Style presets use generic names and millimetre widths. A test bans journal names in preset names (`tests/plot/test_styles_palettes.py:62`). |
| Excel, Microsoft, Google Sheets, Windows, macOS | Nominative: "Paste from Excel" (`app/views/data.py:85-86`), file-type labels, launcher instructions. |
| Illustrator, Adobe, Inkscape, Graphite | Nominative: "Best for touching up in Inkscape, Graphite or Illustrator" (`app/views/export.py:27`). Adobe appears only in the font licence and the trademark acknowledgement in `torusplot/plot/fonts/README.md:19`. |
| Arial, Helvetica | Comparative and nominative: "Liberation Sans has the same letter widths as Arial" (`app/views/style.py:126-127`). Neither font is bundled. `logo.svg:12` lists Arial as a CSS fallback only. |
| Claude, Anthropic, OpenAI, Gemini, Ollama | Names of services the user chooses to connect to (`app/views/settings.py:18-21`, provider modules, `docs/MCP.md`). No logos. ChatGPT: zero occurrences. |
| Qwen, Granite and other model names | Model tags in settings and in the internal research file. |
| Streamlit | Code comments and docstrings only; the UI shows no "Made with Streamlit" mark (`.streamlit/config.toml`: `toolbarMode = "minimal"`). |
| SPSS | Title of a cited textbook, in test provenance files only. |
| Inter, Source Sans 3, Liberation Sans | Font family names shown in the font picker, which is what the OFL expects for unmodified fonts. |

### 4.5 Repository hygiene

- `.gitignore:2` excludes `_reference/`; `.gitignore:4` excludes `.env`. Neither has ever been committed (history scanned: 0 paths under `_reference/`).
- Secret scan of tracked files (API-key patterns, private keys, tokens): two obviously fake test constants only (`tests/ai/test_providers.py:30`, `tests/ai/test_settings.py:17`). `.env.example` has empty values.
- `torusplot.egg-info/` is untracked build output and lists no `_reference` path.

---

## 5. Open questions for a lawyer

- **Q-1 AI authorship.** The code, documentation and logo were generated by an AI system. Is there enforceable copyright in them in the jurisdictions that matter, and should the owner rely on contract (EULA) and trade-secret protection instead? Which Anthropic terms applied to the session, and do they assign output to the owner without restriction?
- **Q-2 Typed-in datasets (F-4).** Are small datasets from primary literature free to reuse in the relevant markets (US facts doctrine; EU database right)? Does it matter that the convenient copies ship with GPL software, or that the values were reproduced from memory? Is the Field textbook example, whose values appear to be invented, treated differently?
- **Q-3 NIST data (F-14).** Confirm that redistributing NIST StRD files, obtained through SciPy's and lmfit's BSD-licensed distributions, is acceptable worldwide, and what attribution NIST expects.
- **Q-4 Trademark clearance.** Is "TorusPlot" (and "Torus") available as a mark for scientific software? Are the nominative uses listed in 4.4 acceptable in marketing material as well as in the UI? Is the term "SuperPlot" free to use as a plot-type name?
- **Q-5 Exposure to copyleft sources (F-9).** The author read GPL and AGPL source code before writing an independent implementation. No copied expression was found. Is the record sufficient, and should the "clean-room" wording be dropped?
- **Q-6 Look and feel.** The table types, analysis names and four-star convention follow the dominant commercial product. Is there any trade-dress or unfair-competition exposure?
- **Q-7 Frozen installer (F-6).** LGPL-2.1 section 6 terms for `libquadmath`, the form of the source offer, the scope of the GCC runtime exception for a separately shipped `libgfortran`, and the notice format for roughly 80 packages.
- **Q-8 Patents.** No patent search was done for any implemented method (for example ROUT). Statistical methods from decades-old literature are unlikely to be encumbered, but this was not checked.
- **Q-9 Outbound terms.** EULA content: licence grant, third-party carve-outs, the user's responsibility for images uploaded to "Match this figure" and "Digitize", and for the terms of any AI provider or model they connect.

---

## Appendix A. Runtime dependency closure (76 packages)

"Licence text found" is what the licence files in the wheel actually contain, classified by signature phrases, not what the metadata claims. `=` means the clean Python 3.12 environment resolved the same version.

| # | Package | .venv (Py 3.11) | clean (Py 3.12) | Licence per metadata | Licence text found in dist-info | Notes |
|---|---|---|---|---|---|---|
| 1 | altair | 6.3.0 | = | BSD License | BSD |  |
| 2 | annotated-types | 0.8.0 | = | MIT | MIT |  |
| 3 | anyio | 4.16.0 | = | MIT | MIT |  |
| 4 | attrs | 26.1.0 | = | MIT | MIT |  |
| 5 | autograd | 1.9.1 | = | MIT | MIT |  |
| 6 | autograd-gamma | 0.5.0 | = | MIT License | MIT |  |
| 7 | certifi | 2026.10.23 | = | MPL-2.0 / Mozilla Public License 2.0 (MPL 2.0) | MPL-2.0 | MPL-2.0 (file-level copyleft on the CA bundle; unmodified use is fine, notice needed if redistributed) |
| 8 | cffi | 2.1.1 | = | MIT-0 | MIT-0 | MIT-0 (MIT No Attribution) - override correct |
| 9 | charset-normalizer | 3.5.2 | = | MIT | MIT |  |
| 10 | click | 8.5.1 | = | BSD-3-Clause | BSD |  |
| 11 | contourpy | 1.3.3 | 1.4.0 | BSD License | BSD |  |
| 12 | cryptography | 50.0.2 | = | Apache-2.0 OR BSD-3-Clause | Apache-2.0, BSD | dual licence, Apache-2.0 or BSD-3 chosen; statically links OpenSSL 4.0.3 (Apache-2.0) and 39 Rust crates (SBOM: one `Apache-2.0 OR GPL-2.0-only` crate, self_cell - Apache-2.0 chosen) |
| 13 | cycler | 0.12.1 | = | BSD License | BSD |  |
| 14 | et_xmlfile | 2.0.0 | = | MIT / MIT License | PSF, HPND/MIT-CMU, ISC, 0BSD, MIT | MIT; `LICENCE.python` (PSF) for code derived from CPython |
| 15 | fonttools | 4.68.0 | = | MIT | MIT, BSD, Apache-2.0, OFL | MIT; `LICENSE.external` lists OFL/Apache/BSD test fonts and code |
| 16 | formulaic | 1.2.2 | = | MIT | MIT |  |
| 17 | h11 | 0.16.0 | = | MIT / MIT License | MIT |  |
| 18 | httpcore2 | 2.16.0 | = | BSD-3-Clause | BSD |  |
| 19 | httpx2 | 2.16.0 | = | BSD-3-Clause | BSD |  |
| 20 | idna | 3.20 | = | BSD-3-Clause | BSD |  |
| 21 | ImageIO | 2.38.1 | = | BSD-2-Clause | BSD |  |
| 22 | interface_meta | 2.0.1 | = | MIT | MIT |  |
| 23 | itsdangerous | 2.2.0 | = | BSD License | BSD |  |
| 24 | Jinja2 | 3.1.6 | = | BSD License | BSD |  |
| 25 | jsonschema | 4.26.0 | = | MIT | MIT |  |
| 26 | jsonschema-specifications | 2025.9.1 | = | MIT | MIT |  |
| 27 | kiwisolver | 1.5.1 | = | BSD License | BSD | BSD-3 ("Kiwi licensing terms") |
| 28 | lazy-loader | 0.6 | = | BSD-3-Clause | BSD |  |
| 29 | lifelines | 0.30.3 | = | MIT / MIT License | MIT | imported only by tests (F-13); bundles 32 sample dataset files |
| 30 | MarkupSafe | 3.0.4 | = | BSD-3-Clause | BSD |  |
| 31 | matplotlib | 3.11.2 | = | Python Software Foundation License | MIT, Apache-2.0, CC0, OFL, FTL | PSF-style Matplotlib licence; bundles DejaVu (Bitstream Vera licence), STIX/Last Resort (OFL), BaKoMa fonts licence, FreeType (FTL chosen over GPL-2), Qhull, ColorBrewer (Apache-2.0) |
| 32 | mcp | 2.3.0 | = | MIT / MIT License | MIT |  |
| 33 | mcp-types | 2.3.0 | = | MIT / MIT License | MIT |  |
| 34 | narwhals | 2.28.0 | = | MIT | MIT |  |
| 35 | networkx | 3.6.1 | 3.7 | BSD-3-Clause | BSD |  |
| 36 | numpy | 2.4.6 | 2.5.4 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | BSD, GPL-text, GCC-exception, Zlib, ISC, 0BSD, Apache-2.0, CC0, MIT | wheel bundles `numpy.libs/`: OpenBLAS (BSD-3), libgfortran (GPL-3.0+ WITH GCC-exception-3.1), libquadmath (LGPL-2.1+) - see F-6 |
| 37 | openpyxl | 3.1.5 | = | MIT / MIT License | MIT |  |
| 38 | opentelemetry-api | 1.45.2 | = | Apache-2.0 | Apache-2.0 |  |
| 39 | packaging | 26.3 | = | Apache-2.0 OR BSD-2-Clause | Apache-2.0, BSD | dual licence, either may be chosen |
| 40 | pandas | 2.3.3 | = | BSD License | MIT, BSD, Apache-2.0, PSF, HPND/MIT-CMU, ISC, 0BSD | BSD-3 with bundled permissive components (MIT/Apache/PSF/HPND/ISC) |
| 41 | patsy | 1.0.3 | = | 2-clause BSD / BSD License | BSD, PSF | BSD-2 with PSF-licensed portions |
| 42 | pillow | 13.0.0 | = | MIT-CMU | MIT, BSD, Apache-2.0, HPND/MIT-CMU, Zlib, FTL | wheel bundles 18 libs in `pillow.libs/` (libjpeg-turbo, libtiff, libpng, freetype [FTL], harfbuzz, lcms2, openjp2, webp, avif, xz/liblzma [0BSD], zstd, brotli, xcb, Xau); libimagequant (GPL) NOT compiled in (`features.check_feature`=False); FriBiDi (LGPL) not bundled on Linux (runtime-loaded shim) |
| 43 | protobuf | 7.36.2 | = | 3-Clause BSD License | BSD |  |
| 44 | pyarrow | 26.0.1 | = | Apache-2.0 | MIT, BSD, Apache-2.0, HPND/MIT-CMU, Zlib, CC0, Boost | Apache-2.0 + NOTICE.txt; long list of bundled permissive components (BSD/MIT/Boost/Zlib/ZPL-2.1/OpenSSL) |
| 45 | pycparser | 3.12 | = | BSD-3-Clause | BSD |  |
| 46 | pydantic | 2.14.3 | = | MIT | MIT |  |
| 47 | pydantic_core | 2.50.4 | = | MIT | MIT | SBOM: 101 Rust crates, all permissive (one `MIT OR Apache-2.0 OR LGPL-2.1-or-later`, r-efi - MIT chosen) |
| 48 | pydeck | 0.9.3 | = | Apache License 2.0 | Apache-2.0 (notice) | Apache-2.0 (short notice file; full text not in wheel) |
| 49 | PyJWT | 2.15.1 | = | MIT | MIT |  |
| 50 | pyparsing | 3.3.3 | = | MIT | MIT |  |
| 51 | python-dateutil | 4.0.0 | = | BSD-3-Clause | BSD, Apache-2.0 (notice) | licence file describes a BSD-3 / Apache-2.0 split by contribution date; both permissive |
| 52 | python-multipart | 0.0.33 | = | Apache-2.0 | Apache-2.0 |  |
| 53 | pytz | 2026.6 | = | MIT / MIT License | MIT |  |
| 54 | PyYAML | 6.0.3 | = | MIT / MIT License | MIT |  |
| 55 | referencing | 0.37.0 | = | MIT | MIT |  |
| 56 | requests | 2.34.2 | = | Apache-2.0 / Apache Software License | Apache-2.0 |  |
| 57 | rpds-py | 2026.9.1 | = | MIT | MIT | SBOM: 17 Rust crates, all MIT/Apache-2.0 |
| 58 | scikit-image | 0.26.0 | = | BSD License | MIT, BSD | BSD-3 with BSD-2/MIT portions; `skimage/data` sample images have individual licences (not used by the product) |
| 59 | scikit-posthocs | 0.18.0 | = | MIT License | MIT | imported only by tests (F-13) |
| 60 | scipy | 1.17.1 | 1.18.1 | BSD License | BSD, GPL-text, GCC-exception | wheel bundles `scipy.libs/`: OpenBLAS (BSD-3), libgfortran (GPL-3.0+ WITH GCC-exception-3.1), libquadmath (LGPL-2.1+); Qhull licence - see F-6 |
| 61 | seaborn | 0.13.2 | = | BSD License | BSD | BSD-3; pulled in only by statannotations |
| 62 | sse-starlette | 3.7.0 | = | BSD-3-Clause | BSD |  |
| 63 | starlette | 1.10.0 | = | BSD-3-Clause | BSD |  |
| 64 | statannotations | 0.7.2 | = | MIT License | MIT | MIT (two copyright holders: M. Weber 2019, F. Charlier 2020); lazily imported at `torusplot/plot/brackets.py:179` |
| 65 | statsmodels | 0.15.0 | = | BSD-3-Clause | BSD | BSD-3; bundles 34 `datasets/` folders with their own provenance (not used by the product) |
| 66 | streamlit | 1.67.0 | = | Apache-2.0 | (none in wheel) | NO licence file inside the wheel (metadata `License-Expression: Apache-2.0` only); 26 MB JS bundle incl. Source Sans/Serif/Code, KaTeX, Material Symbols fonts; no copyleft marker found by grep |
| 67 | tifffile | 2026.3.3 | 2026.11.12 | BSD-3-Clause | BSD |  |
| 68 | truststore | 0.10.5 | = | MIT | MIT |  |
| 69 | typing_extensions | 4.16.0 | = | PSF-2.0 | PSF, HPND/MIT-CMU, ISC, 0BSD | PSF-2.0 |
| 70 | typing-inspection | 0.4.5 | = | MIT | MIT |  |
| 71 | tzdata | 2026.6 | = | Apache-2.0 | Apache-2.0 (notice), Apache-2.0 | Apache-2.0 wrapper; IANA tz data is public domain |
| 72 | urllib3 | 2.8.0 | = | MIT | MIT |  |
| 73 | uvicorn | 0.54.0 | = | BSD-3-Clause | BSD |  |
| 74 | watchdog | 7.1.0 | = | Apache-2.0 | Apache-2.0 (notice), Apache-2.0 |  |
| 75 | websockets | 17.4 | = | BSD-3-Clause | BSD |  |
| 76 | wrapt | 2.5.2 | = | BSD-2-Clause | BSD |  |

Platform differences: Windows adds `pywin32` (not installed here, not verified); macOS omits `watchdog`.

## Appendix B. Reproducing the checks

The scripts used are in the session scratch folder `/tmp/claude-0/-home-user/1ea00402-b8d4-5283-8558-77b3104c95d1/scratchpad/audit/` (`closure.py`, `lictext.py`, `simlib.py`, `simrun.py`, `words5.py`, `numsets.py`, `funcsim.py`, `strlits.py`; results as JSON in `run2/`). They are not part of the repository. No tracked file was modified and nothing was installed during this audit; the only file created in the repository is this report.

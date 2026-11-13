# Commercial release checklist

What must be true before TorusPlot is sold or handed to customers. This is an engineering
checklist produced from the independent licensing audit (2026-11-13). **It is not legal
advice.** Items marked "counsel" need a lawyer, not an engineer.

## Already in place (enforced automatically)

| Guard | Where |
|---|---|
| Only permissive runtime dependencies (MIT, BSD, Apache-2.0, ISC, PSF, HPND, Matplotlib, Unlicense, CC0, MPL-2.0 unmodified) | `tools/check_licenses.py`, runs on every push on Windows and Linux |
| Exact, hash-checked dependency versions | `requirements-lock.txt` (the launchers install from it); `tools/check_lock.py` |
| Every wheel that bundles native libraries or mentions a GPL-family license has a recorded human review | `tools/license_overrides.yaml` -> `bundled_review`; the gate fails on a new, unreviewed one |
| Optional extras are checked too | `requirements-ai.txt` (Anthropic SDK, MIT) |
| No copyleft package imported; no reference-repo files tracked; no secrets tracked | `tests/test_docs.py` |
| No third-party product names in UI, presets, palettes or docs | `tests/test_docs.py`, `tests/plot/test_styles_palettes.py` |
| Only OFL-1.1 fonts are bundled, with their license texts | `tests/plot/test_styles_palettes.py`, `torusplot/plot/fonts/README.md` |
| A defined customer deliverable without tests, fixtures, audit records or design notes | `python tools/build_release.py` -> `dist/TorusPlot-<version>.zip` |

Independent audit result: no copied code was found against the ten reference projects
(token-shingle, function-level and string comparisons) or against installed dependencies.

## Before the first paid release

1. **End-user license agreement (counsel).** `LICENSE` is an all-rights-reserved notice; it
   does not grant customers the right to run the software. Have counsel write a EULA
   covering: license grant and restrictions, the bundled OFL fonts (they stay under the OFL
   and may not be sold on their own), third-party notices, warranty disclaimer for
   statistical results, user-uploaded images in the AI features, and the fact that cloud AI
   providers are used under the customer's own account and terms.
2. **Ship the release ZIP, not the repository.** Build it with `python tools/build_release.py`.
   The repository also contains tests, validation data and design records that are not part
   of the product (see item 5).
3. **Keep the lock file current.** After changing `requirements.txt`, regenerate
   `requirements-lock.txt` (command in `tools/check_lock.py`), then run
   `python tools/check_licenses.py --notices` and commit both. Dependency licenses can change
   between versions; the lock file plus the gate is what keeps the audited set fixed.
4. **Windows and macOS wheels.** The gate runs on GitHub's Windows and Linux runners. If you
   support macOS, add `macos-latest` to `.github/workflows/ci.yml` so its wheels are scanned.
5. **Typed-in published datasets in the tests (counsel, low risk).** The validation suite
   reproduces published worked examples (for example Student 1908, Freireich 1963, textbook
   ANOVA tables). The numbers were typed in and cited to primary sources; measured values are
   facts, but compilations and invented teaching examples can carry rights in some
   jurisdictions. They live only under `tests/` and are excluded from the release ZIP. Two
   examples flagged as the most sensitive were replaced by constructed data (see DECISIONS.md
   D-030). `tests/fixtures/SOURCES.md` lists every source so any other can be dropped.
6. **Name clearance (counsel).** "TorusPlot" has not been searched or registered as a
   trademark. Table-type and analysis names are generic statistical terms.
7. **AI-assisted authorship (counsel).** The code was written with an AI coding assistant
   under the owner's direction. Ask counsel how that affects copyright registration in your
   jurisdiction and confirm the assistant provider's terms assign output to you.
8. **Reference material.** The study-only clones (`_reference/`, never tracked) were deleted
   from the build machine after the audit. `docs/REFERENCE_NOTES.md` records what was read
   and that nothing was copied; the author had read access to those sources, so this is a
   documented ideas-only process with a similarity audit, not a formal clean-room.
9. **Patents.** No patent search was done (for example for the ROUT outlier method).

## Additional duties if you ship a frozen installer (PyInstaller, MSIX, .app ...)

Today the launcher downloads dependencies from PyPI onto the customer's computer, so
TorusPlot does not distribute them. A frozen installer *does* distribute them. Then:

- Include every dependency's license text (about 80 packages; each wheel's `*.dist-info`
  has them) and the NOTICE files of Apache-2.0 packages.
- NumPy and SciPy wheels contain `libgfortran` (GPL-3.0-or-later WITH GCC-exception-3.1:
  allowed in proprietary programs) and `libquadmath` (LGPL-2.1-or-later). For the LGPL
  library: keep it as a separate, replaceable file, include the LGPL text, and state in the
  EULA that reverse engineering to debug modifications of that library is permitted.
- `certifi` is MPL-2.0: ship it unmodified and say where its source is available.
- Re-run the license gate inside the build environment for each target platform.
- Do not bundle model weights.

# PROGRESS

Last updated: 2026-11-13. Remote: https://github.com/torus12/TorusPlotUnlimited (branch `main`).

| Milestone | Scope | Status | Notes |
|---|---|---|---|
| M0 | Scaffold, proprietary LICENSE, license gate, CI, reference notes | done, pushed | License gate: 76 runtime packages, all permissive |
| M1 | Data layer: 5 table types, paste/import, reshape, `.tplot` | done, pushed | |
| M2 | Statistics engine + validation suite | code complete, integrating | 36 registered analyses + power; NIST + published + cross-checks |
| M3 | FigureSpec + renderer + brackets | done, pushed | 18 plot types |
| M4 | Styles, palettes, fonts, composer, exports, reports | done, pushed | 7 presets; SVG/PDF/TIFF/PNG; report + methods |
| M5 | Streamlit UI + launchers | built, tests pass locally | waits on M2 commit |
| M6 | AI layer | in progress | |
| M7 | MCP server | in progress | |
| M8 | Documentation | in progress | |
| M9 | Final audit + FINAL_REPORT.md | pending | |

CI (GitHub Actions, windows-latest + ubuntu-latest, Python 3.11 + 3.12): green on the latest
pushed commit (tests + ruff + license check).

# PROGRESS

Last updated: 2026-11-13. Remote: https://github.com/torus12/TorusPlotUnlimited (branch `main`).

| Milestone | Scope | Status |
|---|---|---|
| M0 | Scaffold, proprietary LICENSE, license gate, CI, reference notes | done |
| M1 | Data layer: 5 table types, paste/import, reshape, `.tplot` | done |
| M2 | Statistics engine (36 analyses + power) + validation suite | done |
| M3 | FigureSpec + renderer (18 plot types) + brackets | done |
| M4 | Styles, palettes, fonts, composer, exports, reports | done |
| M5 | Streamlit UI + launchers + CLI | done |
| M6 | AI layer | done (tested offline; no real provider reachable from the build sandbox) |
| M7 | MCP server | done (optional extra) |
| M8 | Documentation | done |
| M9 | Final audit + FINAL_REPORT.md | done |

Final verification: 2,069 tests pass in a fresh environment built from the lock file;
ruff clean; license gate and lock check pass; all app pages load headless. See FINAL_REPORT.md.

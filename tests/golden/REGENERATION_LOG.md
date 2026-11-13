# Golden image regeneration log

| Date | matplotlib | Cases | Reason |
|---|---|---|---|
| 2026-11-13 | 3.11.2 | all (27) | Initial golden images (M3/M4) |
| 2026-11-13 | 3.11.2 | all (27) | Rendering now ignores machine-level matplotlibrc (build machine had text.hinting off) and always uses bundled fonts; goldens re-rendered to the machine-independent look, matching GitHub runners |

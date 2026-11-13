# Bundled fonts

TorusPlot bundles only fonts under the SIL Open Font License 1.1 (OFL-1.1). The OFL
allows the fonts to be bundled, embedded and redistributed with software, including
commercial and closed-source software, as long as the fonts are not sold by themselves
and the license text travels with them. The font files are unmodified.

| Family | Files | Version | License | Source |
|---|---|---|---|---|
| Inter | `Inter/Inter-{Regular,Bold,Italic,BoldItalic}.ttf` | 4.1 | OFL-1.1 (`Inter/OFL.txt`) | https://github.com/rsms/inter (release v4.1) |
| Source Sans 3 | `SourceSans3/SourceSans3-{Regular,Bold,It,BoldIt}.ttf` | 3.052 | OFL-1.1 (`SourceSans3/OFL.md`) | https://github.com/adobe-fonts/source-sans (release 3.052R) |
| Liberation Sans | `LiberationSans/LiberationSans-{Regular,Bold,Italic,BoldItalic}.ttf` | 2.1.5 | OFL-1.1 (`LiberationSans/OFL.txt`) | Upstream project https://github.com/liberationfonts/liberation-fonts ; files taken from the Debian package `fonts-liberation` 1:2.1.5-3 (see provenance below) |

Liberation Sans is metric-compatible with Arial, so figures laid out with it keep
their proportions if a journal later substitutes Arial. Arial and Helvetica themselves
are **not** bundled (they are proprietary); if they are installed on your computer you
can select them, or any other system font, by name at runtime.

"Source" is a trademark of Adobe; "Liberation" is a trademark of Red Hat, Inc. The
names are used here only to identify the unmodified fonts.

## Provenance and checksums

- Inter: `extras/ttf/` in the release archive `Inter-4.1.zip` from the project's GitHub releases.
- Source Sans 3: `TTF/` in `TTF-source-sans-3.052R.zip` from the project's GitHub releases.
- Liberation Sans: the upstream release download was not reachable from the build machine,
  so the four files were copied from the Debian package `fonts-liberation` (version
  1:2.1.5-3), which packages upstream release 2.1.5. Each file's own name table states
  "Version 2.1.5" and "Licensed under the SIL Open Font License, Version 1.1". The license
  text is upstream's `LICENSE` file.

Every font file's embedded license string is OFL-1.1 and its embedding flag (`fsType`) is 0
(installable embedding). SHA-256 of the bundled files:

| File | SHA-256 |
|---|---|
| `Inter/Inter-Bold.ttf` | `288316099b1e0a47a4716d159098005eef7c0066921f34e3200393dbdb01947f` |
| `Inter/Inter-BoldItalic.ttf` | `948405a16cdc62701da5f4005ed068ca5f4d27061d98f7974ccfc37831d9581d` |
| `Inter/Inter-Italic.ttf` | `bbc051dd204b5019a1aa0bc0ae2aa8a05ab13e7a3f979fa357631dc7feb6833a` |
| `Inter/Inter-Regular.ttf` | `40d692fce188e4471e2b3cba937be967878f631ad3ebbbdcd587687c7ebe0c82` |
| `LiberationSans/LiberationSans-Bold.ttf` | `3973aa5054fb467dd5627245d3dc82e37bf16fe075756156a570455871351582` |
| `LiberationSans/LiberationSans-BoldItalic.ttf` | `c80fa7f2ffa0e01d4d8dcd6a6d1e43eda665222d1b4db597dde1174c456006cf` |
| `LiberationSans/LiberationSans-Italic.ttf` | `830c5fa600505fb4c1a271b4271c53c44bae43f492b2a240d0e98a3a7a380121` |
| `LiberationSans/LiberationSans-Regular.ttf` | `4659bc0c58c5028dd488ec928d41d9265db43d9b669fc14ca8b0832daca7b144` |
| `SourceSans3/SourceSans3-Bold.ttf` | `9214b9d95e4231c609802815c2646c98174e2102d0d37f88978a7f8e71006e6a` |
| `SourceSans3/SourceSans3-BoldIt.ttf` | `7978291fc1bf314db887e0366853b33c5cf2e964c7b95cfb9ce403a6ec46a842` |
| `SourceSans3/SourceSans3-It.ttf` | `192afd78f0f54a3c69eaf02d43f4d9a821e9d6110e41d3d25d61a7385cd580e4` |
| `SourceSans3/SourceSans3-Regular.ttf` | `4644c81b86ec9caaa76b634889968ed3c4f4f52f054855933acc7c2b21e53b0f` |

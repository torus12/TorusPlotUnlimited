# Example datasets (SYNTHETIC)

Every file in this folder is **synthetic**: simulated by `generate_examples.py` with fixed
random seeds. None of it is real experimental data. Use it to learn the app.

| File | Table type | What it imitates | Try this |
|---|---|---|---|
| `nanoparticle_size.csv` | Column | Hydrodynamic diameter (nm) of nanoparticles from four synthesis methods, 8 batches each | One-way ANOVA + Tukey, bar + points |
| `dose_response_cytotoxicity.csv` | XY (3 replicates) | Cell viability (%) vs dose (uM) for a free drug and a nanoformulation | Nonlinear regression (4PL), IC50, dose-response curve |
| `cell_uptake_timecourse.csv` | Grouped (4 replicates) | Uptake (a.u.) at 1, 4, 24 h for three formulations | Two-way ANOVA + Tukey, grouped bars |
| `cell_uptake_superplot.csv` | Multiple variables | Per-cell uptake, 20 cells x 3 independent experiments x 2 treatments | Nested analysis on replicate means, SuperPlot |
| `survival.csv` | Multiple variables | Time to event (days), event flag, three groups of 12 | Kaplan-Meier, log-rank, survival curve |
| `paired_before_after.csv` | Column (rows = animals) | Tumour volume before/after treatment in 10 animals | Paired t test, before-after plot |
| `contingency_response.csv` | Contingency | Responders vs non-responders, treated vs control | Fisher's exact test, odds ratio |

In the app: **1 Data -> Load a sample dataset**.

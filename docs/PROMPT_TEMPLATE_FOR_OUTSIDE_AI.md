# Prompt template for an outside AI chatbot

<!-- GENERATED FILE: do not edit by hand. Regenerate with `python tools/build_prompt_template.py`. -->

TorusPlot can build a figure from instructions written by any chatbot, with no API key and no AI provider configured:

1. Copy everything below the line **PROMPT STARTS HERE** into your chatbot.
2. Replace the parts in [SQUARE BRACKETS] with your table, your result ID (if any) and your wish.
3. Paste the chatbot's JSON answer into TorusPlot's "Outside AI instructions" box.

TorusPlot checks the answer before using it: settings it does not know are ignored, a wrong table name falls back to your current table, and statistics can never be pasted in. P values and significance stars always come from an analysis that TorusPlot itself has run.

---

**PROMPT STARTS HERE**

You are helping me make a scientific figure in TorusPlot, a graphing and statistics program. TorusPlot draws figures from a JSON description called a FigureSpec. Write the FigureSpec for the figure I describe. TorusPlot computes every number itself.

## Rules

1. Answer with ONE JSON object that is a valid FigureSpec. JSON only: no prose, no markdown, no comments.
2. You only describe how the figure LOOKS. You never compute, estimate, write or change statistics. Never put P values, significance stars, "ns", test statistics, effect sizes, confidence limits or claims such as "significant" anywhere (title, labels, legend, tick labels, texts).
3. Significance brackets are drawn by TorusPlot from a result it has already computed. To show them, add an item to "annotations" whose "stat_result_id" is one of the listed result IDs (optionally choose comparisons by ID or by group pairs, or use "comparisons": "significant"). If no result is listed, leave "annotations" empty. Never invent an ID.
4. Use only the table, column, series, group and row names that are listed, spelled exactly as given.
5. "data.table" must be the name of a listed table; unless the request says otherwise use the current table.
6. Choose a "plot_type" that is valid for the kind of that table (see PLOT TYPES).
7. Leave out every field you do not need: unset fields use the style preset. "layers": [] draws the default layers of the plot type.
8. Colours are hex strings such as "#1F77B4". Sizes of the figure are in millimetres.

## My data table

[DESCRIBE YOUR TABLE HERE: its name in TorusPlot, its kind (column, xy, grouped, multiple_variables or contingency) and the names of its columns / groups / series. Names only: do not paste confidential data.]

## Statistics results already computed in TorusPlot (optional)

[IF YOU RAN AN ANALYSIS IN TORUSPLOT AND WANT ITS BRACKETS ON THE FIGURE, PASTE THE RESULT ID HERE, e.g. sr_3f9a1c2b. OTHERWISE WRITE: none.]

To show a computed result, add for example `{"stat_result_id": "<the ID above>", "comparisons": "significant"}` to "annotations". Without an ID, leave "annotations" empty.

## The figure I want

[DESCRIBE THE FIGURE HERE: for example "a box plot of all groups with a log Y axis, the Okabe-Ito palette and a single-column journal size".]

## Plot types

`plot_type`: what it is. "Tables" lists the kinds of table it can be drawn from.

- bar: Bar + individual points. Bars at the mean with error bars; every data point is always shown. Tables: column, grouped, multiple_variables. Default layers: mean_bar, sem_errorbar, data_points_jitter.
- scatter: Scatter dot plot (mean ± error). Every value as a dot with a line at the mean and an error bar. Tables: column, grouped, multiple_variables. Default layers: data_points_jitter, mean_dash, sd_errorbar.
- box: Box plot + points. Median, quartiles and whiskers, with the raw points on top. Tables: column, grouped, multiple_variables. Default layers: boxplot, data_points_jitter.
- violin: Violin plot + points. Smoothed distribution shape for each group with the raw points. Tables: column, grouped, multiple_variables. Default layers: violin, data_points_jitter, median_dash.
- box_violin: Violin + box + points. Violin outline, a slim box plot inside, and the raw points. Tables: column, grouped, multiple_variables. Default layers: violin, boxplot, data_points_jitter.
- beeswarm: Beeswarm. Points arranged so none overlap, with a line at the mean. Tables: column, grouped, multiple_variables. Default layers: data_points_beeswarm, mean_dash, sd_errorbar.
- superplot: SuperPlot (replicate-coloured). Small dots = technical replicates, large dots = mean of each biological replicate, coloured by replicate; statistics use the replicate means. Tables: multiple_variables. Default layers: data_points_beeswarm, replicate_means, mean_dash, sem_errorbar.
- estimation: Estimation plot. Raw data on the left, the mean difference and its 95% CI on the right. Tables: column, multiple_variables. Default layers: data_points_beeswarm, mean_dash, ci95_errorbar, effect_axis.
- paired: Before-after (paired lines). One line per subject connecting its values across conditions. Tables: column, multiple_variables. Default layers: paired_lines, data_points.
- xy: XY (points, error bars, lines). Y versus X with error bars from replicates; optional fitted line or curve. Tables: xy, multiple_variables. Default layers: mean_line, sem_errorbar, mean_dot.
- dose_response: Dose-response curve. Response versus dose on a log axis with the fitted sigmoidal curve. Tables: xy. Default layers: sem_errorbar, mean_dot, curve_fit.
- survival: Survival curve (Kaplan-Meier). Staircase survival curves with tick marks for censored subjects. Tables: multiple_variables. Default layers: km_curve, censor_marks.
- grouped_bar: Grouped bars. Side-by-side bars for two grouping variables, with points and error bars. Tables: grouped, multiple_variables. Default layers: mean_bar, sem_errorbar, data_points_jitter.
- stacked_bar: Stacked bars. Bars stacked on top of each other (means or counts). Tables: grouped, multiple_variables, contingency. Default layers: stack_bar.
- histogram: Histogram. Frequency distribution of values. Tables: column, multiple_variables, xy. Default layers: histogram.
- heatmap: Heat map. A colour-coded grid of values (means, counts or correlations). Tables: grouped, multiple_variables, contingency, column. Default layers: heatmap.
- qq: Q-Q plot (normality). Data quantiles against normal quantiles; points on the line = normal. Tables: column, multiple_variables. Default layers: qq_line, qq_points.
- residuals: Residual plot. Residuals of a fitted model versus X or fitted value (needs a fit result). Tables: xy, multiple_variables, column, grouped. Default layers: reference_line, residual_points.

## Style presets and palettes

STYLE PRESETS (style.preset):
- Torus Signature: Modern, high-contrast and clean: bold palette, raw points on tinted bars, open top/right. (89 x 70 mm)
- Classic Lab: The traditional lab-meeting look: black axes, black/grey bars with black outlines, capped error bars. (100 x 80 mm)
- Journal Single-Column (89 mm): Compact single-column figure: 89 mm wide, 7 pt sans-serif text, thin lines, colourblind-safe palette. (89 x 65 mm)
- Journal 1.5-Column (120 mm): One-and-a-half-column figure: 120 mm wide, 7 pt text, colourblind-safe palette. (120 x 80 mm)
- Journal Double-Column (183 mm): Full-width double-column figure: 183 mm wide, 7 pt text, colourblind-safe palette. (183 x 90 mm)
- Minimal: Quiet and airy: light axes, no bar outlines, soft colours, subtle horizontal grid. (89 x 70 mm)
- Presentation (large type): For slides and posters: large type, thick lines and big markers that read from the back of the room. (220 x 150 mm)

PALETTES (style.palette; or give your own list of hex colours):
- Okabe-Ito: #0072B2, #E69F00, #009E73, #CC79A7, #56B4E9, #D55E00, #F0E442, #000000
- Torus: #27318B, #E4572E, #17A398, #F2A541, #7B4BB7, #4A4A4A, #8FB339, #C2185B
- Torus Soft: #5B6BD6, #F28C6A, #4FC3B5, #F6C177, #A98BD9, #8C8C8C, #B5CF6B, #E07AA5
- Classic Lab: #1A1A1A, #8C8C8C, #D0D0D0, #FFFFFF, #4D4D4D, #B3B3B3
- Grayscale: #222222, #555555, #888888, #AAAAAA, #CCCCCC, #E5E5E5
- Blue-Orange: #1F5FA8, #F28E2B, #7FB3E6, #FFBE7D, #0B3C6F, #B85C00
- Sequential colormaps sampled as palettes: viridis, magma, plasma, inferno, cividis
- Diverging colormaps (style.colormap, for heat maps): RdBu_r, coolwarm, PuOr_r, BrBG

## Example of a valid answer

```json
{
 "plot_type": "bar",
 "data": {
  "table": "My table"
 },
 "layers": [
  {
   "type": "mean_bar"
  },
  {
   "type": "sem_errorbar"
  },
  {
   "type": "data_points_jitter"
  }
 ],
 "title": "Particle size by condition",
 "axes": {
  "y": {
   "label": "Hydrodynamic diameter (nm)"
  }
 },
 "style": {
  "preset": "Torus Signature",
  "palette": "Okabe-Ito"
 },
 "layout": {
  "width_mm": 89,
  "height_mm": 70
 },
 "legend": {
  "show": false
 }
}
```

## Full JSON schema of a FigureSpec

Fields that are not in this schema are ignored. Leave out everything you do not need.

```json
{
 "$defs": {
  "AxesSpec": {
   "additionalProperties": false,
   "properties": {
    "x": {
     "$ref": "#/$defs/AxisSpec"
    },
    "y": {
     "$ref": "#/$defs/AxisSpec"
    },
    "zero_line": {
     "default": false,
     "description": "Draw a thin line at y = 0",
     "title": "Zero Line",
     "type": "boolean"
    }
   },
   "title": "AxesSpec",
   "type": "object"
  },
  "AxisSpec": {
   "additionalProperties": false,
   "properties": {
    "label": {
     "anyOf": [
      {
       "type": "string"
      },
      {
       "type": "null"
      }
     ],
     "default": null,
     "description": "Axis title; None = automatic from the data",
     "title": "Label"
    },
    "scale": {
     "default": "auto",
     "description": "'auto' = linear, except the X axis of dose-response plots, which becomes log10 when the doses are on a linear scale",
     "enum": [
      "auto",
      "linear",
      "log10",
      "log2",
      "symlog"
     ],
     "title": "Scale",
     "type": "string"
    },
    "min": {
     "anyOf": [
      {
       "type": "number"
      },
      {
       "type": "null"
      }
     ],
     "default": null,
     "title": "Min"
    },
    "max": {
     "anyOf": [
      {
       "type": "number"
      },
      {
       "type": "null"
      }
     ],
     "default": null,
     "title": "Max"
    },
    "ticks": {
     "anyOf": [
      {
       "items": {
        "type": "number"
       },
       "type": "array"
      },
      {
       "type": "null"
      }
     ],
     "default": null,
     "description": "Explicit tick positions",
     "title": "Ticks"
    },
    "tick_labels": {
     "anyOf": [
      {
       "items": {
        "type": "string"
       },
       "type": "array"
      },
      {
       "type": "null"
      }
     ],
     "default": null,
     "description": "Labels for the explicit ticks",
     "title": "Tick Labels"
    },
    "tick_rotation": {
     "anyOf": [
      {
       "maximum": 90,
       "minimum": -90,
       "type": "number"
      },
      {
       "type": "null"
      }
     ],
     "default": null,
     "title": "Tick Rotation"
    },
    "tick_format": {
     "anyOf": [
      {
       "type": "string"
      },
      {
       "type": "null"
      }
     ],
     "default": null,
     "description": "'%.1f'-style format, or 'sci', 'percent', 'log_exp' (10^n labels)",
     "title": "Tick Format"
    },
    "grid": {
     "anyOf": [
      {
       "type": "boolean"
      },
      {
       "type": "null"
      }
     ],
     "default": null,
     "title": "Grid"
    },
    "invert": {
     "default": false,
     "title": "Invert",
     "type": "boolean"
    },
    "show": {
     "default": true,
     "description": "Draw this axis (ticks, labels and spine)",
     "title": "Show",
     "type": "boolean"
    }
   },
   "title": "AxisSpec",
   "type": "object"
  },
  "DataBinding": {
   "additionalProperties": false,
   "description": "Which table to draw and the role of its columns.",
   "properties": {
    "table": {
     "description": "Name of the data table in the project",
     "title": "Table",
     "type": "string"
    },
    "groups": {
     "anyOf": [
      {
       "items": {
        "type": "string"
       },
       "type": "array"
      },
      {
       "type": "null"
      }
     ],
     "default": null,
     "description": "Subset and order of groups (Column table columns, XY series, or the column-factor levels of a Grouped table). None = all, in table order.",
     "title": "Groups"
    },
    "rows": {
     "anyOf": [
      {
       "items": {
        "type": "string"
       },
       "type": "array"
      },
      {
       "type": "null"
      }
     ],
     "default": null,
     "description": "Subset and order of row levels of a Grouped table. None = all.",
     "title": "Rows"
    },
    "x": {
     "anyOf": [
      {
       "type": "string"
      },
      {
       "type": "null"
      }
     ],
     "default": null,
     "description": "Multiple-variables table: column for the X axis / categories",
     "title": "X"
    },
    "y": {
     "anyOf": [
      {
       "type": "string"
      },
      {
       "type": "null"
      }
     ],
     "default": null,
     "description": "Multiple-variables table: numeric column to plot",
     "title": "Y"
    },
    "hue": {
     "anyOf": [
      {
       "type": "string"
      },
      {
       "type": "null"
      }
     ],
     "default": null,
     "description": "Multiple-variables table: column for the colour grouping",
     "title": "Hue"
    },
    "replicate": {
     "anyOf": [
      {
       "type": "string"
      },
      {
       "type": "null"
      }
     ],
     "default": null,
     "description": "Multiple-variables table: biological-replicate column (SuperPlots)",
     "title": "Replicate"
    },
    "subject": {
     "anyOf": [
      {
       "type": "string"
      },
      {
       "type": "null"
      }
     ],
     "default": null,
     "description": "Multiple-variables table: subject ID (paired plots)",
     "title": "Subject"
    },
    "time": {
     "anyOf": [
      {
       "type": "string"
      },
      {
       "type": "null"
      }
     ],
     "default": null,
     "description": "Survival: time column",
     "title": "Time"
    },
    "event": {
     "anyOf": [
      {
       "type": "string"
      },
      {
       "type": "null"
      }
     ],
     "default": null,
     "description": "Survival: event column (1 = event, 0 = censored)",
     "title": "Event"
    },
    "columns": {
     "anyOf": [
      {
       "items": {
        "type": "string"
       },
       "type": "array"
      },
      {
       "type": "null"
      }
     ],
     "default": null,
     "description": "Multiple-variables table: columns for heat maps / histograms / Q-Q plots",
     "title": "Columns"
    },
    "stat_result_id": {
     "anyOf": [
      {
       "type": "string"
      },
      {
       "type": "null"
      }
     ],
     "default": null,
     "description": "ID of the StatResult a statistics-driven plot is drawn from (survival curves, residuals, correlation heat map).",
     "title": "Stat Result Id"
    },
    "transpose": {
     "default": false,
     "description": "Grouped table: swap the row and column factors",
     "title": "Transpose",
     "type": "boolean"
    }
   },
   "required": [
    "table"
   ],
   "title": "DataBinding",
   "type": "object"
  },
  "ExportSpec": {
   "additionalProperties": false,
   "properties": {
    "formats": {
     "items": {
      "enum": [
       "svg",
       "pdf",
       "tiff",
       "png"
      ],
      "type": "string"
     },
     "title": "Formats",
     "type": "array"
    },
    "dpi": {
     "default": 300,
     "maximum": 1200,
     "minimum": 72,
     "title": "Dpi",
     "type": "integer"
    },
    "transparent": {
     "default": false,
     "title": "Transparent",
     "type": "boolean"
    }
   },
   "title": "ExportSpec",
   "type": "object"
  },
  "Layer": {
   "additionalProperties": false,
   "description": "One drawn element. Unset options fall back to the style preset.",
   "properties": {
    "type": {
     "enum": [
      "mean_bar",
      "median_bar",
      "mean_dash",
      "median_dash",
      "mean_dot",
      "mean_line",
      "stack_bar",
      "sem_errorbar",
      "sd_errorbar",
      "ci95_errorbar",
      "range_errorbar",
      "iqr_errorbar",
      "data_points",
      "data_points_jitter",
      "data_points_beeswarm",
      "paired_lines",
      "replicate_means",
      "boxplot",
      "violin",
      "histogram",
      "density",
      "curve_fit",
      "confidence_band",
      "prediction_band",
      "km_curve",
      "km_ci",
      "censor_marks",
      "effect_axis",
      "qq_points",
      "qq_line",
      "residual_points",
      "heatmap",
      "reference_line",
      "value_labels",
      "n_labels"
     ],
     "title": "Type",
     "type": "string"
    },
    "color": {
     "anyOf": [
      {
       "type": "string"
      },
      {
       "type": "null"
      }
     ],
     "default": null,
     "description": "Hex colour overriding the palette for this layer",
     "title": "Color"
    },
    "edge_color": {
     "anyOf": [
      {
       "type": "string"
      },
      {
       "type": "null"
      }
     ],
     "default": null,
     "title": "Edge Color"
    },
    "alpha": {
     "anyOf": [
      {
       "maximum": 1,
       "minimum": 0,
       "type": "number"
      },
      {
       "type": "null"
      }
     ],
     "default": null,
     "title": "Alpha"
    },
    "size": {
     "anyOf": [
      {
       "minimum": 0,
       "type": "number"
      },
      {
       "type": "null"
      }
     ],
     "default": null,
     "description": "Marker size in points",
     "title": "Size"
    },
    "width": {
     "anyOf": [
      {
       "exclusiveMinimum": 0,
       "maximum": 1,
       "type": "number"
      },
      {
       "type": "null"
      }
     ],
     "default": null,
     "description": "Bar/box/violin width (fraction of slot)",
     "title": "Width"
    },
    "linewidth": {
     "anyOf": [
      {
       "minimum": 0,
       "type": "number"
      },
      {
       "type": "null"
      }
     ],
     "default": null,
     "title": "Linewidth"
    },
    "linestyle": {
     "anyOf": [
      {
       "enum": [
        "solid",
        "dashed",
        "dotted",
        "dashdot"
       ],
       "type": "string"
      },
      {
       "type": "null"
      }
     ],
     "default": null,
     "title": "Linestyle"
    },
    "marker": {
     "anyOf": [
      {
       "type": "string"
      },
      {
       "type": "null"
      }
     ],
     "default": null,
     "description": "Matplotlib marker: o s ^ v D ...",
     "title": "Marker"
    },
    "fill": {
     "anyOf": [
      {
       "type": "boolean"
      },
      {
       "type": "null"
      }
     ],
     "default": null,
     "title": "Fill"
    },
    "cap_size": {
     "anyOf": [
      {
       "minimum": 0,
       "type": "number"
      },
      {
       "type": "null"
      }
     ],
     "default": null,
     "description": "Error-bar cap size in points",
     "title": "Cap Size"
    },
    "jitter": {
     "anyOf": [
      {
       "maximum": 0.5,
       "minimum": 0,
       "type": "number"
      },
      {
       "type": "null"
      }
     ],
     "default": null,
     "description": "Horizontal spread of points",
     "title": "Jitter"
    },
    "direction": {
     "anyOf": [
      {
       "enum": [
        "both",
        "up",
        "down"
       ],
       "type": "string"
      },
      {
       "type": "null"
      }
     ],
     "default": null,
     "description": "Error-bar direction",
     "title": "Direction"
    },
    "bins": {
     "anyOf": [
      {
       "maximum": 500,
       "minimum": 1,
       "type": "integer"
      },
      {
       "type": "null"
      }
     ],
     "default": null,
     "description": "Histogram bins (None = automatic)",
     "title": "Bins"
    },
    "value": {
     "anyOf": [
      {
       "type": "number"
      },
      {
       "type": "null"
      }
     ],
     "default": null,
     "description": "Position of a reference line",
     "title": "Value"
    },
    "orientation": {
     "anyOf": [
      {
       "enum": [
        "horizontal",
        "vertical"
       ],
       "type": "string"
      },
      {
       "type": "null"
      }
     ],
     "default": null,
     "title": "Orientation"
    },
    "label": {
     "anyOf": [
      {
       "type": "string"
      },
      {
       "type": "null"
      }
     ],
     "default": null,
     "description": "Legend label / text for reference lines",
     "title": "Label"
    },
    "series": {
     "anyOf": [
      {
       "type": "string"
      },
      {
       "type": "null"
      }
     ],
     "default": null,
     "description": "Restrict this layer to one series/group",
     "title": "Series"
    },
    "stat_result_id": {
     "anyOf": [
      {
       "type": "string"
      },
      {
       "type": "null"
      }
     ],
     "default": null,
     "description": "StatResult this layer is drawn from (curve fits, survival, effect axis)",
     "title": "Stat Result Id"
    }
   },
   "required": [
    "type"
   ],
   "title": "Layer",
   "type": "object"
  },
  "LayoutSpec": {
   "additionalProperties": false,
   "properties": {
    "width_mm": {
     "anyOf": [
      {
       "exclusiveMinimum": 10,
       "maximum": 1000,
       "type": "number"
      },
      {
       "type": "null"
      }
     ],
     "default": null,
     "description": "Figure width in mm",
     "title": "Width Mm"
    },
    "height_mm": {
     "anyOf": [
      {
       "exclusiveMinimum": 10,
       "maximum": 1000,
       "type": "number"
      },
      {
       "type": "null"
      }
     ],
     "default": null,
     "description": "Figure height in mm",
     "title": "Height Mm"
    },
    "orientation": {
     "default": "vertical",
     "description": "horizontal draws categories on the Y axis",
     "enum": [
      "vertical",
      "horizontal"
     ],
     "title": "Orientation",
     "type": "string"
    }
   },
   "title": "LayoutSpec",
   "type": "object"
  },
  "LegendSpec": {
   "additionalProperties": false,
   "properties": {
    "show": {
     "anyOf": [
      {
       "type": "boolean"
      },
      {
       "type": "null"
      }
     ],
     "default": null,
     "description": "None = automatic (shown when there is >1 series)",
     "title": "Show"
    },
    "position": {
     "default": "best",
     "enum": [
      "best",
      "upper right",
      "upper left",
      "lower left",
      "lower right",
      "right",
      "outside right",
      "top",
      "bottom"
     ],
     "title": "Position",
     "type": "string"
    },
    "title": {
     "anyOf": [
      {
       "type": "string"
      },
      {
       "type": "null"
      }
     ],
     "default": null,
     "title": "Title"
    },
    "frame": {
     "default": false,
     "title": "Frame",
     "type": "boolean"
    },
    "ncol": {
     "default": 1,
     "maximum": 10,
     "minimum": 1,
     "title": "Ncol",
     "type": "integer"
    }
   },
   "title": "LegendSpec",
   "type": "object"
  },
  "StatAnnotation": {
   "additionalProperties": false,
   "description": "Significance brackets / result text driven by a computed StatResult.\n\nThere is deliberately no field for a p value or star string: the renderer reads\nthem from the referenced result.",
   "properties": {
    "stat_result_id": {
     "description": "ID of the StatResult (e.g. 'sr_3f9a1c2b')",
     "title": "Stat Result Id",
     "type": "string"
    },
    "kind": {
     "default": "brackets",
     "description": "'brackets' draws pairwise comparison brackets; 'summary_text' writes the overall test and its P value in a corner.",
     "enum": [
      "brackets",
      "summary_text"
     ],
     "title": "Kind",
     "type": "string"
    },
    "comparisons": {
     "anyOf": [
      {
       "enum": [
        "all",
        "significant"
       ],
       "type": "string"
      },
      {
       "items": {
        "type": "string"
       },
       "type": "array"
      }
     ],
     "default": "all",
     "description": "Which comparisons of the result to draw: 'all', 'significant', or a list of comparison IDs ('sr_xxx:c1').",
     "title": "Comparisons"
    },
    "pairs": {
     "anyOf": [
      {
       "items": {
        "maxItems": 2,
        "minItems": 2,
        "prefixItems": [
         {
          "type": "string"
         },
         {
          "type": "string"
         }
        ],
        "type": "array"
       },
       "type": "array"
      },
      {
       "type": "null"
      }
     ],
     "default": null,
     "description": "Alternative selection by group names, e.g. [['Control', 'Drug A']]",
     "title": "Pairs"
    },
    "format": {
     "default": "stars",
     "description": "stars: ns/*/**/***/****; p: 'P = 0.0123'; p_short: '0.012'; stars_p: both",
     "enum": [
      "stars",
      "p",
      "p_short",
      "stars_p"
     ],
     "title": "Format",
     "type": "string"
    },
    "show_ns": {
     "default": true,
     "description": "Draw brackets for non-significant comparisons",
     "title": "Show Ns",
     "type": "boolean"
    },
    "star_style": {
     "default": "gp",
     "description": "gp: up to 4 stars; apa: up to 3",
     "enum": [
      "gp",
      "apa"
     ],
     "title": "Star Style",
     "type": "string"
    },
    "position": {
     "default": "top left",
     "description": "Corner for summary_text",
     "enum": [
      "top left",
      "top right",
      "bottom left",
      "bottom right"
     ],
     "title": "Position",
     "type": "string"
    },
    "engine": {
     "default": "torus",
     "description": "Bracket drawing engine. 'torus' works for every plot type; 'statannotations' uses that library for simple categorical plots.",
     "enum": [
      "torus",
      "statannotations"
     ],
     "title": "Engine",
     "type": "string"
    },
    "line_width": {
     "anyOf": [
      {
       "minimum": 0,
       "type": "number"
      },
      {
       "type": "null"
      }
     ],
     "default": null,
     "title": "Line Width"
    },
    "text_size": {
     "anyOf": [
      {
       "exclusiveMinimum": 0,
       "type": "number"
      },
      {
       "type": "null"
      }
     ],
     "default": null,
     "title": "Text Size"
    },
    "tip_length": {
     "anyOf": [
      {
       "maximum": 0.2,
       "minimum": 0,
       "type": "number"
      },
      {
       "type": "null"
      }
     ],
     "default": null,
     "description": "Bracket tip, fraction of axis height",
     "title": "Tip Length"
    },
    "step": {
     "anyOf": [
      {
       "exclusiveMinimum": 0,
       "maximum": 0.5,
       "type": "number"
      },
      {
       "type": "null"
      }
     ],
     "default": null,
     "description": "Vertical spacing between stacked brackets",
     "title": "Step"
    }
   },
   "required": [
    "stat_result_id"
   ],
   "title": "StatAnnotation",
   "type": "object"
  },
  "StyleSpec": {
   "additionalProperties": false,
   "description": "Style = a named preset plus optional overrides (None = use the preset's value).",
   "properties": {
    "preset": {
     "default": "Torus Signature",
     "description": "Style preset name",
     "title": "Preset",
     "type": "string"
    },
    "palette": {
     "anyOf": [
      {
       "type": "string"
      },
      {
       "items": {
        "type": "string"
       },
       "type": "array"
      },
      {
       "type": "null"
      }
     ],
     "default": null,
     "description": "Palette name (e.g. 'Okabe-Ito', 'viridis') or a list of hex colours",
     "title": "Palette"
    },
    "colors": {
     "additionalProperties": {
      "type": "string"
     },
     "description": "Per-group colour overrides: {group name: hex colour}",
     "title": "Colors",
     "type": "object"
    },
    "font_family": {
     "anyOf": [
      {
       "type": "string"
      },
      {
       "type": "null"
      }
     ],
     "default": null,
     "title": "Font Family"
    },
    "font_size": {
     "anyOf": [
      {
       "exclusiveMinimum": 0,
       "type": "number"
      },
      {
       "type": "null"
      }
     ],
     "default": null,
     "description": "Base font size in points",
     "title": "Font Size"
    },
    "title_size": {
     "anyOf": [
      {
       "exclusiveMinimum": 0,
       "type": "number"
      },
      {
       "type": "null"
      }
     ],
     "default": null,
     "title": "Title Size"
    },
    "label_size": {
     "anyOf": [
      {
       "exclusiveMinimum": 0,
       "type": "number"
      },
      {
       "type": "null"
      }
     ],
     "default": null,
     "title": "Label Size"
    },
    "tick_size": {
     "anyOf": [
      {
       "exclusiveMinimum": 0,
       "type": "number"
      },
      {
       "type": "null"
      }
     ],
     "default": null,
     "title": "Tick Size"
    },
    "legend_size": {
     "anyOf": [
      {
       "exclusiveMinimum": 0,
       "type": "number"
      },
      {
       "type": "null"
      }
     ],
     "default": null,
     "title": "Legend Size"
    },
    "axis_linewidth": {
     "anyOf": [
      {
       "minimum": 0,
       "type": "number"
      },
      {
       "type": "null"
      }
     ],
     "default": null,
     "title": "Axis Linewidth"
    },
    "line_width": {
     "anyOf": [
      {
       "minimum": 0,
       "type": "number"
      },
      {
       "type": "null"
      }
     ],
     "default": null,
     "description": "Width of data lines and curves",
     "title": "Line Width"
    },
    "errorbar_linewidth": {
     "anyOf": [
      {
       "minimum": 0,
       "type": "number"
      },
      {
       "type": "null"
      }
     ],
     "default": null,
     "title": "Errorbar Linewidth"
    },
    "cap_size": {
     "anyOf": [
      {
       "minimum": 0,
       "type": "number"
      },
      {
       "type": "null"
      }
     ],
     "default": null,
     "title": "Cap Size"
    },
    "point_size": {
     "anyOf": [
      {
       "minimum": 0,
       "type": "number"
      },
      {
       "type": "null"
      }
     ],
     "default": null,
     "title": "Point Size"
    },
    "point_alpha": {
     "anyOf": [
      {
       "maximum": 1,
       "minimum": 0,
       "type": "number"
      },
      {
       "type": "null"
      }
     ],
     "default": null,
     "title": "Point Alpha"
    },
    "point_edge_color": {
     "anyOf": [
      {
       "type": "string"
      },
      {
       "type": "null"
      }
     ],
     "default": null,
     "title": "Point Edge Color"
    },
    "point_edge_width": {
     "anyOf": [
      {
       "minimum": 0,
       "type": "number"
      },
      {
       "type": "null"
      }
     ],
     "default": null,
     "title": "Point Edge Width"
    },
    "point_color_mode": {
     "anyOf": [
      {
       "enum": [
        "group",
        "dark",
        "black",
        "white"
       ],
       "type": "string"
      },
      {
       "type": "null"
      }
     ],
     "default": null,
     "description": "Colour of raw points drawn over bars/boxes",
     "title": "Point Color Mode"
    },
    "bar_width": {
     "anyOf": [
      {
       "exclusiveMinimum": 0,
       "maximum": 1,
       "type": "number"
      },
      {
       "type": "null"
      }
     ],
     "default": null,
     "title": "Bar Width"
    },
    "bar_alpha": {
     "anyOf": [
      {
       "maximum": 1,
       "minimum": 0,
       "type": "number"
      },
      {
       "type": "null"
      }
     ],
     "default": null,
     "title": "Bar Alpha"
    },
    "bar_edge_color": {
     "anyOf": [
      {
       "type": "string"
      },
      {
       "type": "null"
      }
     ],
     "default": null,
     "description": "Hex colour, or 'group' to match the fill",
     "title": "Bar Edge Color"
    },
    "bar_edge_width": {
     "anyOf": [
      {
       "minimum": 0,
       "type": "number"
      },
      {
       "type": "null"
      }
     ],
     "default": null,
     "title": "Bar Edge Width"
    },
    "bar_fill": {
     "anyOf": [
      {
       "type": "boolean"
      },
      {
       "type": "null"
      }
     ],
     "default": null,
     "description": "False = outlined (hollow) bars",
     "title": "Bar Fill"
    },
    "spines": {
     "anyOf": [
      {
       "items": {
        "enum": [
         "left",
         "bottom",
         "top",
         "right"
        ],
        "type": "string"
       },
       "type": "array"
      },
      {
       "type": "null"
      }
     ],
     "default": null,
     "title": "Spines"
    },
    "tick_direction": {
     "anyOf": [
      {
       "enum": [
        "out",
        "in",
        "inout"
       ],
       "type": "string"
      },
      {
       "type": "null"
      }
     ],
     "default": null,
     "title": "Tick Direction"
    },
    "tick_length": {
     "anyOf": [
      {
       "minimum": 0,
       "type": "number"
      },
      {
       "type": "null"
      }
     ],
     "default": null,
     "title": "Tick Length"
    },
    "grid": {
     "anyOf": [
      {
       "type": "boolean"
      },
      {
       "type": "null"
      }
     ],
     "default": null,
     "title": "Grid"
    },
    "grid_color": {
     "anyOf": [
      {
       "type": "string"
      },
      {
       "type": "null"
      }
     ],
     "default": null,
     "title": "Grid Color"
    },
    "background": {
     "anyOf": [
      {
       "type": "string"
      },
      {
       "type": "null"
      }
     ],
     "default": null,
     "title": "Background"
    },
    "text_color": {
     "anyOf": [
      {
       "type": "string"
      },
      {
       "type": "null"
      }
     ],
     "default": null,
     "title": "Text Color"
    },
    "axis_color": {
     "anyOf": [
      {
       "type": "string"
      },
      {
       "type": "null"
      }
     ],
     "default": null,
     "title": "Axis Color"
    },
    "colormap": {
     "anyOf": [
      {
       "type": "string"
      },
      {
       "type": "null"
      }
     ],
     "default": null,
     "description": "Colormap for heat maps / gradients",
     "title": "Colormap"
    },
    "gradient": {
     "anyOf": [
      {
       "type": "string"
      },
      {
       "type": "null"
      }
     ],
     "default": null,
     "description": "Colour groups along a gradient: a colormap name (e.g. 'viridis')",
     "title": "Gradient"
    }
   },
   "title": "StyleSpec",
   "type": "object"
  },
  "TextAnnotation": {
   "additionalProperties": false,
   "description": "Free text placed on the figure (labels, notes). Not for statistics.",
   "properties": {
    "text": {
     "title": "Text",
     "type": "string"
    },
    "x": {
     "description": "X position (axes fraction 0-1 by default)",
     "title": "X",
     "type": "number"
    },
    "y": {
     "description": "Y position (axes fraction 0-1 by default)",
     "title": "Y",
     "type": "number"
    },
    "coords": {
     "default": "axes",
     "enum": [
      "axes",
      "data"
     ],
     "title": "Coords",
     "type": "string"
    },
    "size": {
     "anyOf": [
      {
       "exclusiveMinimum": 0,
       "type": "number"
      },
      {
       "type": "null"
      }
     ],
     "default": null,
     "title": "Size"
    },
    "color": {
     "anyOf": [
      {
       "type": "string"
      },
      {
       "type": "null"
      }
     ],
     "default": null,
     "title": "Color"
    },
    "ha": {
     "default": "left",
     "enum": [
      "left",
      "center",
      "right"
     ],
     "title": "Ha",
     "type": "string"
    },
    "va": {
     "default": "top",
     "enum": [
      "top",
      "center",
      "bottom"
     ],
     "title": "Va",
     "type": "string"
    },
    "weight": {
     "default": "normal",
     "enum": [
      "normal",
      "bold"
     ],
     "title": "Weight",
     "type": "string"
    },
    "style": {
     "default": "normal",
     "enum": [
      "normal",
      "italic"
     ],
     "title": "Style",
     "type": "string"
    }
   },
   "required": [
    "text",
    "x",
    "y"
   ],
   "title": "TextAnnotation",
   "type": "object"
  }
 },
 "additionalProperties": false,
 "description": "Complete, re-renderable description of one figure.",
 "properties": {
  "version": {
   "default": 1,
   "description": "Spec format version",
   "title": "Version",
   "type": "integer"
  },
  "name": {
   "default": "Figure 1",
   "description": "Figure name inside the project",
   "title": "Name",
   "type": "string"
  },
  "plot_type": {
   "description": "Kind of graph",
   "enum": [
    "bar",
    "scatter",
    "box",
    "violin",
    "box_violin",
    "beeswarm",
    "superplot",
    "estimation",
    "paired",
    "xy",
    "dose_response",
    "survival",
    "grouped_bar",
    "stacked_bar",
    "histogram",
    "heatmap",
    "qq",
    "residuals"
   ],
   "title": "Plot Type",
   "type": "string"
  },
  "data": {
   "$ref": "#/$defs/DataBinding"
  },
  "layers": {
   "description": "Elements to draw, bottom to top. Empty = the default layers of the plot type.",
   "items": {
    "$ref": "#/$defs/Layer"
   },
   "title": "Layers",
   "type": "array"
  },
  "annotations": {
   "items": {
    "$ref": "#/$defs/StatAnnotation"
   },
   "title": "Annotations",
   "type": "array"
  },
  "texts": {
   "items": {
    "$ref": "#/$defs/TextAnnotation"
   },
   "title": "Texts",
   "type": "array"
  },
  "title": {
   "anyOf": [
    {
     "type": "string"
    },
    {
     "type": "null"
    }
   ],
   "default": null,
   "title": "Title"
  },
  "axes": {
   "$ref": "#/$defs/AxesSpec"
  },
  "style": {
   "$ref": "#/$defs/StyleSpec"
  },
  "layout": {
   "$ref": "#/$defs/LayoutSpec"
  },
  "legend": {
   "$ref": "#/$defs/LegendSpec"
  },
  "export": {
   "$ref": "#/$defs/ExportSpec"
  }
 },
 "required": [
  "plot_type",
  "data"
 ],
 "title": "FigureSpec",
 "type": "object"
}
```

Answer with the JSON object only.

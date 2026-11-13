"""Turn (FigureSpec, table) into plain arrays the drawing code can use."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats as _st

from ..data.tables import (
    ColumnTable,
    ContingencyTable,
    DataTable,
    GroupedTable,
    MultipleVariablesTable,
    XYTable,
)
from ..spec.figure import FigureSpec

Key = tuple[str, str | None]


class PlotError(ValueError):
    """Raised when a figure cannot be drawn from the given spec + data (message is user-facing)."""


def summarize(values: np.ndarray, ci_level: float = 0.95) -> dict[str, float]:
    """Descriptive summary used for bars / error bars (same formulas as torusplot.stats)."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    n = v.size
    nan = float("nan")
    if n == 0:
        return dict.fromkeys(("mean", "median", "sd", "sem", "ci_low", "ci_high", "min", "max", "q1", "q3"), nan) | {"n": 0}
    mean = float(v.mean())
    sd = float(v.std(ddof=1)) if n > 1 else nan
    sem = sd / np.sqrt(n) if n > 1 else nan
    half = float(_st.t.ppf(0.5 + ci_level / 2.0, n - 1)) * sem if n > 1 else nan
    q1, med, q3 = (float(x) for x in np.percentile(v, [25, 50, 75]))
    return {
        "n": n, "mean": mean, "median": med, "sd": sd, "sem": sem,
        "ci_low": mean - half, "ci_high": mean + half,
        "min": float(v.min()), "max": float(v.max()), "q1": q1, "q3": q3,
    }


@dataclass
class CatData:
    """Data for categorical plots: values per (category, hue) cell."""

    categories: list[str]
    hues: list[str] | None
    values: dict[Key, np.ndarray]
    cat_key: str = "group"
    hue_key: str | None = None
    x_label: str | None = None
    y_label: str | None = None
    hue_label: str | None = None
    #: replicate label of every value (SuperPlots)
    replicates: dict[Key, np.ndarray] = field(default_factory=dict)
    #: subject id of every value (paired plots)
    subjects: dict[Key, np.ndarray] = field(default_factory=dict)

    def keys(self) -> list[Key]:
        if self.hues is None:
            return [(c, None) for c in self.categories]
        return [(c, h) for c in self.categories for h in self.hues]


@dataclass
class XYSeries:
    name: str
    x: np.ndarray  # unique x values
    mean: np.ndarray
    sd: np.ndarray
    sem: np.ndarray
    n: np.ndarray
    ci_half: np.ndarray
    x_points: np.ndarray  # every individual point
    y_points: np.ndarray


@dataclass
class XYData:
    series: list[XYSeries]
    x_label: str | None = None
    y_label: str | None = None


def _select(available: list[str], wanted: list[str] | None, what: str) -> list[str]:
    if not wanted:
        return list(available)
    missing = [w for w in wanted if w not in available]
    if missing:
        raise PlotError(f"Unknown {what}: {missing}. Available: {available}")
    return list(wanted)


def _mv_column(table: MultipleVariablesTable, name: str | None, role: str, numeric: bool | None = None) -> str:
    if name is None:
        raise PlotError(f"Choose the {role} column for this plot.")
    if name not in table.data.columns:
        raise PlotError(f"Column '{name}' ({role}) is not in table '{table.name}'.")
    if numeric is True and name not in table.numeric_columns:
        raise PlotError(f"Column '{name}' ({role}) must contain numbers.")
    return name


def prepare_categorical(spec: FigureSpec, table: DataTable) -> CatData:
    d = spec.data
    if isinstance(table, ColumnTable):
        cats = _select(table.group_names, d.groups, "group(s)")
        frame = table.analysis_frame()
        values, subjects = {}, {}
        for c in cats:
            col = frame[c].to_numpy(dtype=float)
            ok = ~np.isnan(col)
            values[(c, None)] = col[ok]
            subjects[(c, None)] = np.arange(len(col))[ok]
        return CatData(cats, None, values, cat_key="group", y_label=table.name, subjects=subjects)

    if isinstance(table, GroupedTable):
        rows = _select(table.row_levels, d.rows, "row level(s)")
        cols = _select(table.column_levels, d.groups, "group(s)")
        values = {}
        subjects = {}
        if d.transpose:
            for c in cols:
                for r in rows:
                    v = table.cell(r, c, dropna=False)
                    ok = ~np.isnan(v)
                    values[(c, r)] = v[ok]
                    subjects[(c, r)] = np.arange(len(v))[ok]
            return CatData(cols, rows, values, cat_key="column", hue_key="row", x_label=table.column_label,
                           hue_label=table.row_label, y_label=table.name, subjects=subjects)
        for r in rows:
            for c in cols:
                v = table.cell(r, c, dropna=False)
                ok = ~np.isnan(v)
                values[(r, c)] = v[ok]
                subjects[(r, c)] = np.arange(len(v))[ok]
        hues = cols if len(cols) > 1 or len(rows) > 1 else cols
        return CatData(rows, hues, values, cat_key="row", hue_key="column", x_label=table.row_label,
                       hue_label=table.column_label, y_label=table.name, subjects=subjects)

    if isinstance(table, MultipleVariablesTable):
        y = d.y or (table.numeric_columns[0] if table.numeric_columns else None)
        y = _mv_column(table, y, "Y (values)", numeric=True)
        cols = [c for c in (d.x, y, d.hue, d.replicate, d.subject) if c]
        for c in cols:
            _mv_column(table, c, "plot")
        frame = table.frame(list(dict.fromkeys(cols)))
        if frame.empty:
            raise PlotError("No complete rows to plot (check for missing values in the chosen columns).")
        xs = frame[d.x].astype(str) if d.x else pd.Series(["All"] * len(frame))
        cats = _select(list(dict.fromkeys(xs)), d.groups, "categories")
        hues = list(dict.fromkeys(frame[d.hue].astype(str))) if d.hue else None
        values, reps, subs = {}, {}, {}
        for c in cats:
            for h in hues or [None]:
                mask = np.array(xs == c, dtype=bool)
                if h is not None:
                    mask = mask & np.array(frame[d.hue].astype(str) == h, dtype=bool)
                values[(c, h)] = frame.loc[mask, y].to_numpy(dtype=float)
                if d.replicate:
                    reps[(c, h)] = frame.loc[mask, d.replicate].astype(str).to_numpy()
                if d.subject:
                    subs[(c, h)] = frame.loc[mask, d.subject].astype(str).to_numpy()
        return CatData(cats, hues, values, cat_key=d.x or "group", hue_key=d.hue, x_label=d.x, y_label=y,
                       hue_label=d.hue, replicates=reps, subjects=subs)

    if isinstance(table, ContingencyTable):
        counts = table.counts
        rows = _select(table.row_levels, d.rows, "row(s)")
        cols = _select(table.column_levels, d.groups, "column(s)")
        values = {(r, c): np.array([float(counts.loc[r, c])]) for r in rows for c in cols}
        return CatData(rows, cols, values, cat_key="row", hue_key="column", x_label=table.row_label,
                       hue_label=table.column_label, y_label="Count")

    raise PlotError(f"A {table.kind_label} table cannot be drawn as a '{spec.plot_type}' plot.")


def _series_from_points(name: str, x: np.ndarray, y: np.ndarray) -> XYSeries:
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    ux = np.unique(x)
    mean, sd, sem, n, half = (np.full(ux.size, np.nan) for _ in range(5))
    for i, xv in enumerate(ux):
        s = summarize(y[x == xv])
        mean[i], sd[i], sem[i], n[i] = s["mean"], s["sd"], s["sem"], s["n"]
        half[i] = s["ci_high"] - s["mean"]
    return XYSeries(name, ux, mean, sd, sem, n, half, x, y)


def prepare_xy(spec: FigureSpec, table: DataTable) -> XYData:
    d = spec.data
    if isinstance(table, XYTable):
        names = _select(table.series_names, d.groups, "series")
        series = []
        for name in names:
            x, y = table.xy_points(name)
            series.append(_series_from_points(name, x, y))
        y_label = names[0] if len(names) == 1 else table.name
        return XYData(series, x_label=table.x_name, y_label=y_label)
    if isinstance(table, MultipleVariablesTable):
        x = _mv_column(table, d.x, "X", numeric=True)
        y = _mv_column(table, d.y, "Y", numeric=True)
        cols = [x, y] + ([d.hue] if d.hue else [])
        if d.hue:
            _mv_column(table, d.hue, "series")
        frame = table.frame(cols)
        if d.hue:
            names = _select(list(dict.fromkeys(frame[d.hue].astype(str))), d.groups, "series")
            series = [
                _series_from_points(
                    nm,
                    frame.loc[frame[d.hue].astype(str) == nm, x].to_numpy(float),
                    frame.loc[frame[d.hue].astype(str) == nm, y].to_numpy(float),
                )
                for nm in names
            ]
        else:
            series = [_series_from_points(y, frame[x].to_numpy(float), frame[y].to_numpy(float))]
        return XYData(series, x_label=x, y_label=y)
    raise PlotError(f"A {table.kind_label} table cannot be drawn as an XY plot. Use an XY table.")


def prepare_value_sets(spec: FigureSpec, table: DataTable) -> dict[str, np.ndarray]:
    """Named sets of raw values (histograms, Q-Q plots)."""
    d = spec.data
    if isinstance(table, ColumnTable):
        groups = table.groups()
        return {g: groups[g] for g in _select(table.group_names, d.groups, "group(s)")}
    if isinstance(table, XYTable):
        return {s: table.xy_points(s)[1] for s in _select(table.series_names, d.groups, "series")}
    if isinstance(table, MultipleVariablesTable):
        cols = d.columns or ([d.y] if d.y else table.numeric_columns[:1])
        out: dict[str, np.ndarray] = {}
        for c in cols:
            _mv_column(table, c, "values", numeric=True)
            if d.hue or d.x:
                by = d.hue or d.x
                frame = table.frame([c, by])
                for level in dict.fromkeys(frame[by].astype(str)):
                    label = level if len(cols) == 1 else f"{c}: {level}"
                    out[label] = frame.loc[frame[by].astype(str) == level, c].to_numpy(float)
            else:
                out[c] = table.frame([c])[c].to_numpy(float)
        return out
    if isinstance(table, GroupedTable):
        return {c: np.concatenate([table.cell(r, c) for r in table.row_levels])
                for c in _select(table.column_levels, d.groups, "group(s)")}
    raise PlotError(f"A {table.kind_label} table has no value sets for this plot.")


def prepare_matrix(spec: FigureSpec, table: DataTable) -> tuple[np.ndarray, list[str], list[str], str]:
    """(matrix, row labels, column labels, colourbar label) for heat maps."""
    d = spec.data
    if isinstance(table, GroupedTable):
        rows = _select(table.row_levels, d.rows, "row level(s)")
        cols = _select(table.column_levels, d.groups, "group(s)")
        mat = np.array([[summarize(table.cell(r, c))["mean"] for c in cols] for r in rows])
        return mat, rows, cols, "Mean"
    if isinstance(table, ContingencyTable):
        rows = _select(table.row_levels, d.rows, "row(s)")
        cols = _select(table.column_levels, d.groups, "column(s)")
        return table.counts.loc[rows, cols].to_numpy(float), rows, cols, "Count"
    if isinstance(table, ColumnTable):
        cols = _select(table.group_names, d.groups, "group(s)")
        mat = table.analysis_frame()[cols].to_numpy(float)
        return mat, [str(i + 1) for i in range(mat.shape[0])], cols, "Value"
    if isinstance(table, MultipleVariablesTable):
        cols = d.columns or table.numeric_columns
        for c in cols:
            _mv_column(table, c, "heat map", numeric=True)
        frame = table.analysis_frame()
        mat = frame[cols].to_numpy(float)
        if d.x and d.x in frame.columns:
            rows = frame[d.x].astype(str).tolist()
        else:
            rows = [str(i + 1) for i in range(mat.shape[0])]
        return mat, rows, list(cols), "Value"
    raise PlotError(f"A {table.kind_label} table cannot be drawn as a heat map.")

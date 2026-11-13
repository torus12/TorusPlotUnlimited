"""Converters between tidy *long* data and TorusPlot's wide table types."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .tables import (
    ColumnTable,
    ContingencyTable,
    DataTable,
    GroupedTable,
    MultipleVariablesTable,
    TableError,
    XYTable,
)


def _levels(series: pd.Series) -> list[str]:
    if isinstance(series.dtype, pd.CategoricalDtype):
        present = set(series.dropna().astype(str))
        return [str(c) for c in series.cat.categories if str(c) in present]
    return list(dict.fromkeys(series.dropna().astype(str)))


def _need(df: pd.DataFrame, *cols: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise TableError(f"Column(s) not found: {missing}. Available: {list(df.columns)}")


def to_long(table: DataTable) -> pd.DataFrame:
    """Tidy long-format view of any table."""
    return table.to_long()


def long_to_column(df: pd.DataFrame, group: str, value: str, name: str = "Table",
                   subject: str | None = None) -> ColumnTable:
    """Long -> Column table. With ``subject`` rows are aligned per subject (paired data)."""
    _need(df, group, value, *([subject] if subject else []))
    levels = _levels(df[group])
    g = df[group].astype(str)
    if subject:
        wide = df.assign(**{group: g}).pivot_table(
            index=subject, columns=group, values=value, aggfunc="mean", sort=False
        )
        wide = wide.reindex(columns=levels)
        return ColumnTable(name, wide.reset_index(drop=True))
    cols = {lvl: pd.to_numeric(df.loc[g == lvl, value], errors="coerce").dropna().tolist() for lvl in levels}
    return ColumnTable.from_dict(name, cols)


def long_to_grouped(df: pd.DataFrame, row: str, column: str, value: str, name: str = "Table",
                    replicate: str | None = None) -> GroupedTable:
    """Long -> Grouped table (rows = ``row`` levels, column groups = ``column`` levels)."""
    _need(df, row, column, value, *([replicate] if replicate else []))
    rows, cols = _levels(df[row]), _levels(df[column])
    r, c = df[row].astype(str), df[column].astype(str)
    cells: dict[str, dict[str, list[float]]] = {}
    for rl in rows:
        cells[rl] = {}
        for cl in cols:
            sub = df[(r == rl) & (c == cl)]
            if replicate:
                sub = sub.sort_values(replicate, kind="stable")
            cells[rl][cl] = pd.to_numeric(sub[value], errors="coerce").tolist()
    if replicate:
        # keep replicate positions aligned across rows (subject matching)
        reps = list(dict.fromkeys(df[replicate].dropna().tolist()))
        for rl in rows:
            for cl in cols:
                sub = df[(r == rl) & (c == cl)]
                by_rep = dict(zip(sub[replicate], pd.to_numeric(sub[value], errors="coerce")))
                present = [k for k in reps if k in set(df.loc[c == cl, replicate])]
                cells[rl][cl] = [float(by_rep.get(k, np.nan)) for k in present]
    return GroupedTable.from_cells(name, cells, row_label=row, column_label=column)


def long_to_xy(df: pd.DataFrame, x: str, y: str, series: str | None = None, name: str = "Table") -> XYTable:
    """Long -> XY table; repeated (x, series) rows become replicate sub-columns."""
    _need(df, x, y, *([series] if series else []))
    work = df.copy()
    work[x] = pd.to_numeric(work[x], errors="coerce")
    work[y] = pd.to_numeric(work[y], errors="coerce")
    work = work.dropna(subset=[x])
    xs = sorted(work[x].unique())
    names = _levels(work[series]) if series else [y]
    data = {}
    for sname in names:
        sub = work[work[series].astype(str) == sname] if series else work
        grouped = {xv: grp[y].tolist() for xv, grp in sub.groupby(x, sort=True)}
        n_rep = max((len(v) for v in grouped.values()), default=1) or 1
        arr = np.full((len(xs), n_rep), np.nan)
        for i, xv in enumerate(xs):
            vals = grouped.get(xv, [])
            arr[i, : len(vals)] = vals
        data[sname] = arr
    return XYTable.from_arrays(name, xs, data, x_name=x)


def long_to_contingency(df: pd.DataFrame, row: str, column: str, name: str = "Table",
                        count: str | None = None) -> ContingencyTable:
    """Long -> Contingency table by cross-tabulating two categorical columns."""
    _need(df, row, column, *([count] if count else []))
    rows, cols = _levels(df[row]), _levels(df[column])
    r, c = df[row].astype(str), df[column].astype(str)
    if count:
        tab = pd.crosstab(r, c, values=pd.to_numeric(df[count], errors="coerce"), aggfunc="sum").fillna(0)
    else:
        tab = pd.crosstab(r, c)
    tab = tab.reindex(index=rows, columns=cols, fill_value=0)
    return ContingencyTable.from_counts(name, tab.to_numpy(), rows, cols, row_label=row, column_label=column)


def to_multiple_variables(table: DataTable, name: str | None = None) -> MultipleVariablesTable:
    """Any table -> tidy Multiple-variables table (its long format)."""
    long = table.to_long().copy()
    for col in long.columns:
        if isinstance(long[col].dtype, pd.CategoricalDtype):
            long[col] = long[col].astype(str)
    return MultipleVariablesTable(name or f"{table.name} (long)", long, approximate=table.approximate)

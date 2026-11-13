"""Data table types.

TorusPlot organises data the way bench scientists already think about it. The *type*
of a table decides which analyses and graphs make sense:

===================  ==========================================================
Table type           Typical use
===================  ==========================================================
Column               One grouping variable; each column is a group
                     (t tests, one-way ANOVA, column scatter / bar graphs).
XY                   Every point has an X and one or more Y replicates
                     (regression, dose-response, time courses).
Grouped              Two grouping variables: rows x columns with replicate
                     sub-columns (two-way ANOVA, grouped bars).
Multiple variables   One row per subject/observation, one column per variable
                     (multiple regression, ANCOVA, correlation matrix,
                     survival, nested "SuperPlot" data).
Contingency          Counts of subjects in row x column categories
                     (chi-square, Fisher's exact test).
===================  ==========================================================

Every table wraps a *wide* :class:`pandas.DataFrame` (what the user sees in the grid)
plus a little metadata, and can always produce a tidy *long* frame via ``to_long()``
with fixed, documented column names.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any, ClassVar

import numpy as np
import pandas as pd

TABLE_KINDS = ("column", "xy", "grouped", "multiple_variables", "contingency")

KIND_LABELS = {
    "column": "Column",
    "xy": "XY",
    "grouped": "Grouped",
    "multiple_variables": "Multiple variables",
    "contingency": "Contingency",
}

KIND_DESCRIPTIONS = {
    "column": "Each column is one group (e.g. Control, Drug A, Drug B). Rows are replicates.",
    "xy": "First column is X (dose, time, concentration). Other columns are Y values, "
    "optionally with replicates side by side.",
    "grouped": "Rows are one factor (e.g. time point), column groups are a second factor "
    "(e.g. treatment), with replicate sub-columns.",
    "multiple_variables": "One row per observation, one column per variable (like a tidy "
    "spreadsheet). Pick the role of each column when you analyze.",
    "contingency": "Counts of subjects: rows are exposures/groups, columns are outcomes.",
}


class TableError(ValueError):
    """Raised when a table is malformed or used in an invalid way."""


def _to_numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for col in df.columns:
        out[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    return out


def _unique_names(names: Iterable[Any]) -> list[str]:
    """Make column names unique, non-empty strings (``A``, ``A (2)``, ...)."""
    seen: dict[str, int] = {}
    out = []
    for i, raw in enumerate(names):
        name = str(raw).strip() if raw is not None else ""
        if not name or name.lower() in {"nan", "none"} or re.fullmatch(r"Unnamed: \d+", name):
            name = f"Column {i + 1}"
        if name in seen:
            seen[name] += 1
            name = f"{name} ({seen[name]})"
        seen.setdefault(name, 1)
        out.append(name)
    return out


class DataTable:
    """Base class for all table types."""

    kind: ClassVar[str] = ""

    def __init__(
        self,
        name: str,
        data: pd.DataFrame,
        *,
        approximate: bool = False,
        notes: str = "",
        excluded: Iterable[tuple[int, str]] | None = None,
    ) -> None:
        self.name = str(name).strip() or "Table"
        self.data = data.reset_index(drop=True)
        self.data.columns = [str(c) for c in self.data.columns]
        #: True for digitized ("approximate, for reference only") data.
        self.approximate = bool(approximate)
        self.notes = notes
        #: Cells the user explicitly excluded from analysis: (row index, column name).
        self.excluded: list[tuple[int, str]] = [(int(r), str(c)) for r, c in (excluded or [])]

    # ------------------------------------------------------------------ helpers
    @property
    def kind_label(self) -> str:
        return KIND_LABELS[self.kind]

    def analysis_frame(self) -> pd.DataFrame:
        """The wide data with user-excluded cells blanked (set to NaN / None)."""
        df = self.data.copy()
        for row, col in self.excluded:
            if col == "*":
                if 0 <= row < len(df):
                    df = df.drop(index=row)
            elif col in df.columns and 0 <= row < len(self.data) and row in df.index:
                df.loc[row, col] = np.nan
        return df

    def exclude(self, cells: Iterable[tuple[int, str]]) -> None:
        """Explicitly exclude cells from analysis (never automatic; always logged)."""
        for row, col in cells:
            cell = (int(row), str(col))
            if cell not in self.excluded:
                self.excluded.append(cell)

    def clear_exclusions(self) -> None:
        self.excluded = []

    def exclusion_notes(self) -> list[str]:
        if not self.excluded:
            return []
        cells = ", ".join(
            f"row {r + 1}" if c == "*" else f"{c} row {r + 1}" for r, c in sorted(self.excluded)
        )
        return [
            f"{len(self.excluded)} value(s) were explicitly excluded by the user before "
            f"analysis: {cells}."
        ]

    def to_long(self) -> pd.DataFrame:  # pragma: no cover - overridden
        raise NotImplementedError

    def meta(self) -> dict[str, Any]:
        """Metadata needed (with the CSV) to rebuild this table."""
        return {
            "name": self.name,
            "kind": self.kind,
            "approximate": self.approximate,
            "notes": self.notes,
            "excluded": [[r, c] for r, c in self.excluded],
            **self._extra_meta(),
        }

    def _extra_meta(self) -> dict[str, Any]:
        return {}

    def summary(self) -> str:
        rows, cols = self.data.shape
        flag = " [APPROXIMATE: for reference only]" if self.approximate else ""
        return f"{self.kind_label} table '{self.name}': {rows} rows x {cols} columns{flag}"

    def copy(self) -> DataTable:
        return table_from_meta(self.data.copy(), self.meta())

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.summary()}>"

    def equals(self, other: object) -> bool:
        if not isinstance(other, DataTable) or self.meta() != other.meta():
            return False
        if list(self.data.columns) != list(other.data.columns) or len(self.data) != len(other.data):
            return False
        for col in self.data.columns:
            a, b = self.data[col], other.data[col]
            if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
                if not np.array_equal(a.to_numpy(float), b.to_numpy(float), equal_nan=True):
                    return False
            elif [None if pd.isna(v) else str(v) for v in a] != [
                None if pd.isna(v) else str(v) for v in b
            ]:
                return False
        return True


# ============================================================================ Column
class ColumnTable(DataTable):
    """One grouping variable. Each column is a group; rows are replicates/subjects.

    For paired / repeated-measures designs each *row* is one subject.
    Long format columns: ``group``, ``subject``, ``value``.
    """

    kind = "column"

    def __init__(self, name: str, data: pd.DataFrame, **kwargs: Any) -> None:
        data = data.copy()
        data.columns = _unique_names(data.columns)
        super().__init__(name, _to_numeric_frame(data), **kwargs)
        if self.data.shape[1] == 0:
            raise TableError("A Column table needs at least one column of numbers.")

    @classmethod
    def from_dict(cls, name: str, groups: dict[str, Iterable[float]], **kwargs: Any) -> ColumnTable:
        """Build from ``{"Control": [..], "Treated": [..]}`` (lengths may differ)."""
        series = {str(k): pd.Series(list(v), dtype=float) for k, v in groups.items()}
        return cls(name, pd.DataFrame(series), **kwargs)

    @property
    def group_names(self) -> list[str]:
        return list(self.data.columns)

    def groups(self, dropna: bool = True) -> dict[str, np.ndarray]:
        """``{group name: 1-D float array}`` with exclusions (and NaN) removed."""
        df = self.analysis_frame()
        out = {}
        for col in df.columns:
            arr = df[col].to_numpy(dtype=float)
            out[col] = arr[~np.isnan(arr)] if dropna else arr
        return out

    def paired_frame(self, columns: list[str] | None = None) -> pd.DataFrame:
        """Rows (subjects) that have a value in *every* selected column."""
        df = self.analysis_frame()
        if columns is not None:
            missing = [c for c in columns if c not in df.columns]
            if missing:
                raise TableError(f"Unknown column(s): {missing}")
            df = df[columns]
        return df.dropna(axis=0, how="any")

    def to_long(self) -> pd.DataFrame:
        df = self.analysis_frame()
        long = df.reset_index(names="subject").melt(
            id_vars="subject", var_name="group", value_name="value"
        )
        long = long.dropna(subset=["value"]).reset_index(drop=True)
        long["group"] = pd.Categorical(long["group"], categories=self.group_names, ordered=True)
        return long[["group", "subject", "value"]]


# ================================================================================ XY
class XYTable(DataTable):
    """X column plus one or more Y series, each with one or more replicate sub-columns.

    Long format columns: ``x``, ``series``, ``replicate``, ``y``.
    """

    kind = "xy"

    def __init__(
        self,
        name: str,
        data: pd.DataFrame,
        *,
        x: str | None = None,
        series: dict[str, list[str]] | None = None,
        **kwargs: Any,
    ) -> None:
        data = data.copy()
        data.columns = _unique_names(data.columns)
        super().__init__(name, _to_numeric_frame(data), **kwargs)
        if self.data.shape[1] < 2:
            raise TableError("An XY table needs an X column and at least one Y column.")
        self.x_name = str(x) if x is not None else self.data.columns[0]
        if self.x_name not in self.data.columns:
            raise TableError(f"X column '{self.x_name}' not found.")
        y_cols = [c for c in self.data.columns if c != self.x_name]
        if series is None:
            series = {c: [c] for c in y_cols}
        self.series_map: dict[str, list[str]] = {str(k): [str(c) for c in v] for k, v in series.items()}
        used = [c for cols in self.series_map.values() for c in cols]
        unknown = [c for c in used if c not in y_cols]
        if unknown:
            raise TableError(f"Series refer to unknown column(s): {unknown}")
        if len(set(used)) != len(used):
            raise TableError("A column can belong to only one Y series.")

    @classmethod
    def from_arrays(
        cls,
        name: str,
        x: Iterable[float],
        series: dict[str, Any],
        x_name: str = "X",
        **kwargs: Any,
    ) -> XYTable:
        """``series`` maps a name to a 1-D array or a 2-D array (rows = x, cols = replicates)."""
        frame = pd.DataFrame({x_name: list(x)})
        mapping: dict[str, list[str]] = {}
        for sname, values in series.items():
            arr = np.asarray(values, dtype=float)
            if arr.ndim == 1:
                arr = arr[:, None]
            if arr.shape[0] != len(frame):
                raise TableError(f"Series '{sname}' has {arr.shape[0]} rows; X has {len(frame)}.")
            cols = [sname] if arr.shape[1] == 1 else [f"{sname}:{i + 1}" for i in range(arr.shape[1])]
            for j, col in enumerate(cols):
                frame[col] = arr[:, j]
            mapping[sname] = cols
        return cls(name, frame, x=x_name, series=mapping, **kwargs)

    @property
    def series_names(self) -> list[str]:
        return list(self.series_map)

    @property
    def x(self) -> np.ndarray:
        return self.data[self.x_name].to_numpy(dtype=float)

    def series(self, name: str) -> pd.DataFrame:
        """Replicate columns of one series (rows follow the X column; exclusions blanked)."""
        if name not in self.series_map:
            raise TableError(f"Unknown series '{name}'. Available: {self.series_names}")
        return self.analysis_frame()[self.series_map[name]]

    def xy_points(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """All individual (x, y) points of a series, replicates flattened, NaN removed."""
        ys = self.series(name).to_numpy(dtype=float)
        xs = np.repeat(self.x[:, None], ys.shape[1], axis=1)
        mask = ~np.isnan(ys) & ~np.isnan(xs)
        return xs[mask], ys[mask]

    def xy_means(self, name: str) -> pd.DataFrame:
        """Per-X summary of a series: columns ``x, mean, sd, sem, n``."""
        ys = self.series(name).to_numpy(dtype=float)
        n = (~np.isnan(ys)).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                mean = np.nanmean(ys, axis=1)
                sd = np.where(n > 1, np.nanstd(ys, axis=1, ddof=1), np.nan)
            sem = sd / np.sqrt(n)
        out = pd.DataFrame({"x": self.x, "mean": mean, "sd": sd, "sem": sem, "n": n})
        return out[(n > 0) & ~np.isnan(out["x"])].reset_index(drop=True)

    def to_long(self) -> pd.DataFrame:
        df = self.analysis_frame()
        frames = []
        for sname, cols in self.series_map.items():
            for i, col in enumerate(cols):
                frames.append(
                    pd.DataFrame(
                        {"x": df[self.x_name], "series": sname, "replicate": i + 1, "y": df[col]}
                    )
                )
        long = pd.concat(frames, ignore_index=True).dropna(subset=["x", "y"]).reset_index(drop=True)
        long["series"] = pd.Categorical(long["series"], categories=self.series_names, ordered=True)
        return long

    def _extra_meta(self) -> dict[str, Any]:
        return {"x": self.x_name, "series": {k: list(v) for k, v in self.series_map.items()}}


# =========================================================================== Grouped
class GroupedTable(DataTable):
    """Two grouping variables: rows x column groups, with replicate sub-columns.

    The first column holds the row-factor level names. ``groups`` maps each
    column-factor level to its replicate sub-columns. For repeated-measures designs a
    replicate position (sub-column index) identifies the same subject across rows.

    Long format columns: ``row``, ``column``, ``replicate``, ``value``.
    """

    kind = "grouped"

    def __init__(
        self,
        name: str,
        data: pd.DataFrame,
        *,
        groups: dict[str, list[str]] | None = None,
        row_label: str | None = None,
        column_label: str = "Group",
        **kwargs: Any,
    ) -> None:
        data = data.copy()
        data.columns = _unique_names(data.columns)
        if data.shape[1] < 2:
            raise TableError("A Grouped table needs a row-label column and at least one data column.")
        first = data.columns[0]
        body = _to_numeric_frame(data.iloc[:, 1:])
        labels = _unique_names(
            [v if not (isinstance(v, float) and np.isnan(v)) else "" for v in data[first].tolist()]
        )
        frame = pd.concat([pd.Series(labels, name=first), body.reset_index(drop=True)], axis=1)
        super().__init__(name, frame, **kwargs)
        self.row_label = str(row_label) if row_label else first
        self.column_label = str(column_label)
        self._row_col = first
        value_cols = list(self.data.columns[1:])
        if groups is None:
            groups = {c: [c] for c in value_cols}
        self.group_map: dict[str, list[str]] = {str(k): [str(c) for c in v] for k, v in groups.items()}
        used = [c for cols in self.group_map.values() for c in cols]
        unknown = [c for c in used if c not in value_cols]
        if unknown:
            raise TableError(f"Groups refer to unknown column(s): {unknown}")
        if len(set(used)) != len(used):
            raise TableError("A column can belong to only one group.")

    @classmethod
    def from_cells(
        cls,
        name: str,
        cells: dict[str, dict[str, Iterable[float]]],
        row_label: str = "Row",
        column_label: str = "Group",
        **kwargs: Any,
    ) -> GroupedTable:
        """Build from ``{row level: {column level: [replicates...]}}``."""
        row_levels = list(cells)
        col_levels: list[str] = []
        for row in cells.values():
            for c in row:
                if c not in col_levels:
                    col_levels.append(c)
        n_rep = {
            c: max((len(list(cells[r].get(c, []))) for r in row_levels), default=0) or 1
            for c in col_levels
        }
        frame = pd.DataFrame({row_label: row_levels})
        mapping: dict[str, list[str]] = {}
        for c in col_levels:
            cols = [c] if n_rep[c] == 1 else [f"{c}:{i + 1}" for i in range(n_rep[c])]
            mapping[c] = cols
            for i, col in enumerate(cols):
                vals = []
                for r in row_levels:
                    reps = list(cells[r].get(c, []))
                    vals.append(float(reps[i]) if i < len(reps) else np.nan)
                frame[col] = vals
        return cls(name, frame, groups=mapping, row_label=row_label, column_label=column_label, **kwargs)

    @property
    def row_levels(self) -> list[str]:
        return [str(v) for v in self.data[self._row_col].tolist()]

    @property
    def column_levels(self) -> list[str]:
        return list(self.group_map)

    def cell(self, row: str, column: str, dropna: bool = True) -> np.ndarray:
        if column not in self.group_map:
            raise TableError(f"Unknown column group '{column}'.")
        levels = self.row_levels
        if row not in levels:
            raise TableError(f"Unknown row '{row}'.")
        arr = (
            self.analysis_frame()[self.group_map[column]].iloc[levels.index(row)].to_numpy(dtype=float)
        )
        return arr[~np.isnan(arr)] if dropna else arr

    def to_long(self) -> pd.DataFrame:
        df = self.analysis_frame()
        frames = []
        for cname, cols in self.group_map.items():
            for i, col in enumerate(cols):
                frames.append(
                    pd.DataFrame(
                        {
                            "row": df[self._row_col].astype(str),
                            "column": cname,
                            "replicate": i + 1,
                            "value": df[col],
                        }
                    )
                )
        long = pd.concat(frames, ignore_index=True).dropna(subset=["value"]).reset_index(drop=True)
        long["row"] = pd.Categorical(long["row"], categories=self.row_levels, ordered=True)
        long["column"] = pd.Categorical(long["column"], categories=self.column_levels, ordered=True)
        return long

    def _extra_meta(self) -> dict[str, Any]:
        return {
            "groups": {k: list(v) for k, v in self.group_map.items()},
            "row_label": self.row_label,
            "column_label": self.column_label,
        }


# ================================================================ Multiple variables
class MultipleVariablesTable(DataTable):
    """Tidy data: one row per observation, one column per variable.

    Columns can be numeric or categorical (text). Roles (outcome, factor, covariate,
    subject, time, event, replicate ...) are chosen at analysis time.
    """

    kind = "multiple_variables"

    def __init__(self, name: str, data: pd.DataFrame, **kwargs: Any) -> None:
        data = data.copy()
        data.columns = _unique_names(data.columns)
        frame = pd.DataFrame(index=range(len(data)))
        for col in data.columns:
            raw = data[col].reset_index(drop=True)
            numeric = pd.to_numeric(raw, errors="coerce")
            non_blank = raw.notna() & (raw.astype(str).str.strip() != "")
            if non_blank.sum() > 0 and numeric[non_blank].notna().all():
                frame[col] = numeric.astype(float)
            else:
                frame[col] = [
                    None if (v is None or (isinstance(v, float) and np.isnan(v)) or str(v).strip() == "")
                    else str(v).strip()
                    for v in raw.tolist()
                ]
                frame[col] = frame[col].astype(object)
        super().__init__(name, frame, **kwargs)
        if self.data.shape[1] == 0:
            raise TableError("A Multiple variables table needs at least one column.")

    @property
    def numeric_columns(self) -> list[str]:
        return [c for c in self.data.columns if pd.api.types.is_numeric_dtype(self.data[c])]

    @property
    def categorical_columns(self) -> list[str]:
        return [c for c in self.data.columns if c not in self.numeric_columns]

    def frame(self, columns: list[str] | None = None, dropna: bool = True) -> pd.DataFrame:
        """Selected columns with excluded rows/cells removed (complete cases by default)."""
        df = self.analysis_frame()
        if columns is not None:
            missing = [c for c in columns if c not in df.columns]
            if missing:
                raise TableError(f"Unknown column(s): {missing}")
            df = df[columns]
        return df.dropna(axis=0, how="any").reset_index(drop=True) if dropna else df

    def analysis_frame(self) -> pd.DataFrame:
        df = self.data.copy()
        drop_rows = [r for r, c in self.excluded if c == "*" and 0 <= r < len(df)]
        for row, col in self.excluded:
            if col != "*" and col in df.columns and 0 <= row < len(df):
                df.loc[row, col] = np.nan if col in self.numeric_columns else None
        return df.drop(index=drop_rows)

    def to_long(self) -> pd.DataFrame:
        return self.analysis_frame().reset_index(drop=True)

    def _extra_meta(self) -> dict[str, Any]:
        return {
            "dtypes": {
                c: ("number" if c in self.numeric_columns else "category") for c in self.data.columns
            }
        }


# ======================================================================= Contingency
class ContingencyTable(DataTable):
    """Counts in rows x columns categories. First column holds the row labels.

    Long format columns: ``row``, ``column``, ``count``.
    """

    kind = "contingency"

    def __init__(
        self,
        name: str,
        data: pd.DataFrame,
        *,
        row_label: str | None = None,
        column_label: str = "Outcome",
        **kwargs: Any,
    ) -> None:
        data = data.copy()
        data.columns = _unique_names(data.columns)
        if data.shape[1] < 2:
            raise TableError("A Contingency table needs a row-label column and count columns.")
        first = data.columns[0]
        body = _to_numeric_frame(data.iloc[:, 1:])
        labels = _unique_names(data[first].tolist())
        frame = pd.concat([pd.Series(labels, name=first), body.reset_index(drop=True)], axis=1)
        kwargs.pop("excluded", None)
        super().__init__(name, frame, **kwargs)
        self._row_col = first
        self.row_label = str(row_label) if row_label else first
        self.column_label = str(column_label)
        values = self.data.iloc[:, 1:].to_numpy(dtype=float)
        finite = values[~np.isnan(values)]
        if (finite < 0).any():
            raise TableError("Contingency tables hold counts; negative numbers are not allowed.")

    @classmethod
    def from_counts(
        cls,
        name: str,
        counts: Any,
        row_names: list[str],
        column_names: list[str],
        row_label: str = "Group",
        column_label: str = "Outcome",
        **kwargs: Any,
    ) -> ContingencyTable:
        arr = np.asarray(counts, dtype=float)
        if arr.shape != (len(row_names), len(column_names)):
            raise TableError("counts must have shape (len(row_names), len(column_names)).")
        frame = pd.DataFrame(arr, columns=[str(c) for c in column_names])
        frame.insert(0, row_label, [str(r) for r in row_names])
        return cls(name, frame, row_label=row_label, column_label=column_label, **kwargs)

    @property
    def row_levels(self) -> list[str]:
        return [str(v) for v in self.data[self._row_col].tolist()]

    @property
    def column_levels(self) -> list[str]:
        return list(self.data.columns[1:])

    @property
    def counts(self) -> pd.DataFrame:
        """Counts indexed by row label (missing cells count as 0)."""
        df = self.data.iloc[:, 1:].fillna(0.0)
        df.index = pd.Index(self.row_levels, name=self.row_label)
        df.columns.name = self.column_label
        return df

    def has_non_integer_counts(self) -> bool:
        values = self.counts.to_numpy(dtype=float)
        return bool(np.any(np.abs(values - np.round(values)) > 1e-9))

    def to_long(self) -> pd.DataFrame:
        long = self.counts.reset_index(names="row").melt(
            id_vars="row", var_name="column", value_name="count"
        )
        return long

    def _extra_meta(self) -> dict[str, Any]:
        return {"row_label": self.row_label, "column_label": self.column_label}


# ===================================================================== construction
_KIND_TO_CLASS: dict[str, type[DataTable]] = {
    "column": ColumnTable,
    "xy": XYTable,
    "grouped": GroupedTable,
    "multiple_variables": MultipleVariablesTable,
    "contingency": ContingencyTable,
}


def table_from_meta(data: pd.DataFrame, meta: dict[str, Any]) -> DataTable:
    """Rebuild a table from its wide data and the dict produced by ``DataTable.meta()``."""
    kind = meta.get("kind")
    if kind not in _KIND_TO_CLASS:
        raise TableError(f"Unknown table kind '{kind}'. Expected one of {TABLE_KINDS}.")
    common = {
        "approximate": bool(meta.get("approximate", False)),
        "notes": meta.get("notes", ""),
        "excluded": [tuple(x) for x in meta.get("excluded", [])],
    }
    name = meta.get("name", "Table")
    if kind == "column":
        return ColumnTable(name, data, **common)
    if kind == "xy":
        return XYTable(name, data, x=meta.get("x"), series=meta.get("series"), **common)
    if kind == "grouped":
        return GroupedTable(
            name,
            data,
            groups=meta.get("groups"),
            row_label=meta.get("row_label"),
            column_label=meta.get("column_label", "Group"),
            **common,
        )
    if kind == "multiple_variables":
        return MultipleVariablesTable(name, data, **common)
    return ContingencyTable(
        name,
        data,
        row_label=meta.get("row_label"),
        column_label=meta.get("column_label", "Outcome"),
        **common,
    )

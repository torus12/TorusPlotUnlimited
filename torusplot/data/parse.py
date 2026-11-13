"""Turn pasted spreadsheet text (Excel / Google Sheets / CSV) into tables.

Copying cells from Excel puts tab-separated text on the clipboard. This module
detects the delimiter, an optional header row, European decimal commas, and common
"missing value" markers, then builds the requested table type.
"""

from __future__ import annotations

import csv
import io
import re
from typing import Any

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

MISSING_MARKERS = {"", "na", "n/a", "nan", "null", "none", "-", "--", ".", "#n/a", "#div/0!", "#value!"}

_NUM_DOT = re.compile(r"^[+-]?(\d{1,3}(,\d{3})+|\d+)?(\.\d*)?([eE][+-]?\d+)?$")
_NUM_COMMA = re.compile(r"^[+-]?(\d{1,3}(\.\d{3})+|\d+)?(,\d*)?([eE][+-]?\d+)?$")


def detect_delimiter(text: str) -> str:
    """Pick the most plausible delimiter: tab, semicolon, comma, or runs of spaces."""
    lines = [ln for ln in text.splitlines() if ln.strip()][:25]
    if not lines:
        return "\t"
    for delim in ("\t", ";", ","):
        counts = [ln.count(delim) for ln in lines]
        if min(counts) >= 1 and (max(counts) - min(counts) <= 1 or delim == "\t"):
            return delim
    if any("\t" in ln for ln in lines):
        return "\t"
    if all(re.search(r"\S\s+\S", ln) for ln in lines):
        return " "
    return "\t"


def _split(text: str, delimiter: str) -> list[list[str]]:
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip("\n")
    if delimiter == " ":
        return [re.split(r"\s{1,}", ln.strip()) for ln in text.split("\n") if ln.strip()]
    reader = csv.reader(io.StringIO(text), delimiter=delimiter)
    return [row for row in reader if any(cell.strip() for cell in row)]


def parse_number(cell: Any, decimal_comma: bool = False) -> float:
    """Parse one cell to float; returns NaN for blanks, text and missing markers."""
    if cell is None:
        return float("nan")
    if isinstance(cell, int | float | np.integer | np.floating):
        return float(cell)
    s = str(cell).strip().replace("−", "-").replace(" ", "").replace(" ", "")
    if s.lower() in MISSING_MARKERS:
        return float("nan")
    percent = s.endswith("%")
    if percent:
        s = s[:-1]
    if decimal_comma:
        if not _NUM_COMMA.match(s) or s in {"+", "-"}:
            return float("nan")
        s = s.replace(".", "").replace(",", ".")
    else:
        if not _NUM_DOT.match(s) or s in {"+", "-"}:
            return float("nan")
        s = s.replace(",", "")
    try:
        value = float(s)
    except ValueError:
        return float("nan")
    return value


def _looks_numeric(cell: str, decimal_comma: bool) -> bool:
    return not np.isnan(parse_number(cell, decimal_comma))


def detect_decimal_comma(rows: list[list[str]], delimiter: str) -> bool:
    """True when numbers are written like ``1,5`` (and the delimiter is not a comma)."""
    if delimiter == ",":
        return False
    comma_only = dot_only = 0
    for row in rows:
        for cell in row:
            s = cell.strip()
            if re.fullmatch(r"[+-]?\d+,\d+", s):
                comma_only += 1
            elif re.fullmatch(r"[+-]?\d*\.\d+", s):
                dot_only += 1
    return comma_only > 0 and dot_only == 0


def parse_pasted_text(
    text: str,
    header: bool | None = None,
    delimiter: str | None = None,
    decimal_comma: bool | None = None,
) -> pd.DataFrame:
    """Parse pasted text into a DataFrame of strings/floats (no table semantics yet).

    Parameters
    ----------
    header:
        ``True``/``False`` to force, ``None`` to auto-detect (a first row whose cells
        are mostly non-numeric is treated as column titles).
    """
    if text is None or not str(text).strip():
        raise TableError("Nothing to parse: the pasted text is empty.")
    delimiter = delimiter or detect_delimiter(text)
    rows = _split(str(text), delimiter)
    if not rows:
        raise TableError("Nothing to parse: the pasted text is empty.")
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    if decimal_comma is None:
        decimal_comma = detect_decimal_comma(rows, delimiter)

    if header is None:
        first = [c for c in rows[0] if c.strip()]
        numeric_first = sum(_looks_numeric(c, decimal_comma) for c in first)
        header = bool(first) and numeric_first < len(first) / 2 and len(rows) > 1
        if header and len(rows) > 1:
            # a header row should be "more texty" than the body
            body_cells = [c for r in rows[1:] for c in r if c.strip()]
            body_numeric = sum(_looks_numeric(c, decimal_comma) for c in body_cells)
            if body_cells and body_numeric == 0 and numeric_first == 0:
                header = True  # all-text table (multiple variables): first row = titles
    if header:
        columns = [c.strip() for c in rows[0]]
        body = rows[1:]
    else:
        columns = [f"Column {i + 1}" for i in range(width)]
        body = rows

    data: dict[str, list[Any]] = {}
    used: dict[str, int] = {}
    for j, raw_name in enumerate(columns):
        name = raw_name or f"Column {j + 1}"
        if name in used:
            used[name] += 1
            name = f"{name} ({used[name]})"
        used.setdefault(name, 1)
        cells = [r[j].strip() for r in body]
        non_missing = [c for c in cells if c.lower() not in MISSING_MARKERS]
        numeric = [_looks_numeric(c, decimal_comma) for c in non_missing]
        if non_missing and all(numeric):
            data[name] = [parse_number(c, decimal_comma) for c in cells]
        elif non_missing and sum(numeric) / len(non_missing) >= 0.8 and len(non_missing) >= 5:
            # mostly numbers with a few stray text cells (e.g. "outlier?"): keep numbers
            data[name] = [parse_number(c, decimal_comma) for c in cells]
        else:
            data[name] = [None if c.lower() in MISSING_MARKERS else c for c in cells]
    frame = pd.DataFrame(data)
    frame.attrs["header_detected"] = bool(header)
    frame.attrs["decimal_comma"] = bool(decimal_comma)
    frame.attrs["delimiter"] = delimiter
    return frame


def _replicate_groups(columns: list[str], replicates: int | None) -> dict[str, list[str]]:
    """Group side-by-side replicate columns.

    * ``replicates=n`` groups every ``n`` consecutive columns.
    * otherwise columns named ``Name``, ``Name (2)``... or ``Name:1``, ``Name_2``,
      ``Name rep 3`` are grouped by their base name.
    """
    if replicates and replicates > 1:
        if len(columns) % replicates != 0:
            raise TableError(
                f"{len(columns)} data columns cannot be split into groups of {replicates} replicates."
            )
        groups: dict[str, list[str]] = {}
        for i in range(0, len(columns), replicates):
            chunk = columns[i : i + replicates]
            base = _base_name(chunk[0])
            if base in groups:
                base = chunk[0]
            groups[base] = chunk
        return groups
    groups = {}
    for col in columns:
        groups.setdefault(_base_name(col), []).append(col)
    if all(len(v) == 1 for v in groups.values()):
        return {c: [c] for c in columns}
    return groups


def _base_name(col: str) -> str:
    base = re.sub(r"\s*\(\d+\)$", "", col)
    base = re.sub(r"(\s*[:_\-#]\s*|\s+(rep|replicate|n)\s*)\d+$", "", base, flags=re.IGNORECASE)
    return base.strip() or col


def build_table(
    kind: str,
    frame: pd.DataFrame,
    name: str = "Table",
    *,
    replicates: int | None = None,
    row_label: str | None = None,
    column_label: str | None = None,
    approximate: bool = False,
) -> DataTable:
    """Give a parsed frame table semantics.

    ``kind`` is one of ``column``, ``xy``, ``grouped``, ``multiple_variables``,
    ``contingency``. For XY and Grouped tables ``replicates`` (or repeated column
    titles) defines side-by-side replicate sub-columns.
    """
    frame = frame.copy()
    frame.attrs = {}
    if kind == "column":
        return ColumnTable(name, frame, approximate=approximate)
    if kind == "multiple_variables":
        return MultipleVariablesTable(name, frame, approximate=approximate)
    if kind == "xy":
        if frame.shape[1] < 2:
            raise TableError("An XY table needs an X column and at least one Y column.")
        cols = [str(c) for c in frame.columns]
        return XYTable(
            name, frame, x=cols[0], series=_replicate_groups(cols[1:], replicates), approximate=approximate
        )
    if kind == "grouped":
        if frame.shape[1] < 2:
            raise TableError("A Grouped table needs a row-label column and data columns.")
        cols = [str(c) for c in frame.columns]
        return GroupedTable(
            name,
            frame,
            groups=_replicate_groups(cols[1:], replicates),
            row_label=row_label or cols[0],
            column_label=column_label or "Group",
            approximate=approximate,
        )
    if kind == "contingency":
        return ContingencyTable(
            name,
            frame,
            row_label=row_label,
            column_label=column_label or "Outcome",
            approximate=approximate,
        )
    raise TableError(f"Unknown table kind '{kind}'.")


def table_from_text(kind: str, text: str, name: str = "Table", **kwargs: Any) -> DataTable:
    """Parse pasted text and build a table in one step."""
    header = kwargs.pop("header", None)
    delimiter = kwargs.pop("delimiter", None)
    decimal_comma = kwargs.pop("decimal_comma", None)
    frame = parse_pasted_text(text, header=header, delimiter=delimiter, decimal_comma=decimal_comma)
    return build_table(kind, frame, name, **kwargs)


def guess_table_kind(frame: pd.DataFrame) -> str:
    """Suggest a table type for a parsed frame (the user can always override)."""
    if frame.shape[1] == 0:
        return "column"
    numeric = [pd.api.types.is_numeric_dtype(frame[c]) for c in frame.columns]
    if all(numeric):
        first = frame.iloc[:, 0].dropna().to_numpy(dtype=float)
        monotonic = len(first) >= 3 and (np.all(np.diff(first) > 0) or np.all(np.diff(first) < 0))
        return "xy" if monotonic and frame.shape[1] >= 2 else "column"
    if not numeric[0] and all(numeric[1:]) and frame.shape[1] >= 2:
        body = frame.iloc[:, 1:].to_numpy(dtype=float)
        finite = body[~np.isnan(body)]
        is_counts = finite.size > 0 and np.all(finite >= 0) and np.allclose(finite, np.round(finite))
        if is_counts and frame.shape[0] <= 6 and frame.shape[1] <= 7 and frame.iloc[:, 0].is_unique:
            return "contingency"
        return "grouped" if frame.iloc[:, 0].is_unique else "multiple_variables"
    return "multiple_variables"

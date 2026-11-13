"""File import: CSV / TSV / TXT and Excel workbooks (via openpyxl)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .parse import build_table, parse_pasted_text
from .tables import DataTable, TableError


def read_text_file(path: str | Path, header: bool | None = None) -> pd.DataFrame:
    """Read a delimited text file (comma, semicolon, tab) into a parsed frame."""
    raw = Path(path).read_bytes()
    for encoding in ("utf-8-sig", "utf-16", "cp1252", "latin-1"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeError:
            continue
    else:  # pragma: no cover - latin-1 never fails
        raise TableError(f"Could not decode {path}")
    return parse_pasted_text(text, header=header)


def list_sheets(path: str | Path) -> list[str]:
    """Sheet names of an Excel workbook."""
    from openpyxl import load_workbook

    wb = load_workbook(Path(path), read_only=True, data_only=True)
    try:
        return list(wb.sheetnames)
    finally:
        wb.close()


def read_excel_sheet(
    path: str | Path, sheet: str | int | None = None, header: bool | None = None
) -> pd.DataFrame:
    """Read one worksheet (computed values, not formulas) into a parsed frame."""
    from openpyxl import load_workbook

    wb = load_workbook(Path(path), read_only=True, data_only=True)
    try:
        if sheet is None:
            ws = wb.worksheets[0]
        elif isinstance(sheet, int):
            ws = wb.worksheets[sheet]
        else:
            if sheet not in wb.sheetnames:
                raise TableError(f"Sheet '{sheet}' not found. Available: {wb.sheetnames}")
            ws = wb[sheet]
        rows = [list(r) for r in ws.iter_rows(values_only=True)]
    finally:
        wb.close()
    rows = [r for r in rows if any(c is not None and str(c).strip() != "" for c in r)]
    if not rows:
        raise TableError("The worksheet is empty.")
    width = max(len(r) for r in rows)
    keep = [j for j in range(width) if any(j < len(r) and r[j] not in (None, "") for r in rows)]
    lines = []
    for r in rows:
        cells = []
        for j in keep:
            v = r[j] if j < len(r) else None
            cells.append("" if v is None else repr(float(v)) if isinstance(v, int | float) and not isinstance(v, bool) else str(v))
        lines.append("\t".join(c.replace("\t", " ").replace("\n", " ") for c in cells))
    return parse_pasted_text("\n".join(lines), header=header, delimiter="\t", decimal_comma=False)


def read_frame(path: str | Path, sheet: str | int | None = None, header: bool | None = None) -> pd.DataFrame:
    suffix = Path(path).suffix.lower()
    if suffix in {".xlsx", ".xlsm"}:
        return read_excel_sheet(path, sheet=sheet, header=header)
    if suffix in {".csv", ".tsv", ".txt", ".dat"}:
        return read_text_file(path, header=header)
    raise TableError(f"Unsupported file type '{suffix}'. Use .csv, .tsv, .txt or .xlsx.")


def import_table(
    path: str | Path,
    kind: str,
    name: str | None = None,
    sheet: str | int | None = None,
    header: bool | None = None,
    **kwargs: Any,
) -> DataTable:
    """Import a CSV/TSV/XLSX file as a table of the given kind."""
    frame = read_frame(path, sheet=sheet, header=header)
    return build_table(kind, frame, name or Path(path).stem, **kwargs)


def export_table_csv(table: DataTable, path: str | Path) -> Path:
    """Write the table's wide data to CSV (UTF-8)."""
    path = Path(path)
    table.data.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
    return path

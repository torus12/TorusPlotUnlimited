"""TorusPlot project files (``.tplot``).

A ``.tplot`` file is an ordinary zip archive:

``project.yaml``
    Manifest: format version, table metadata, every FigureSpec, panel compositions and
    the analysis log.
``data/<table>.csv``
    One UTF-8 CSV per data table (the wide grid exactly as shown in the app).
``results.json``
    Every StatResult produced by the statistics engine, keyed by result ID. Figures
    refer to these IDs for their annotations, so a saved figure re-renders with exactly
    the numbers that were computed -- nothing is recalculated or re-typed.

The archive is written deterministically (fixed timestamps, sorted keys), so saving
the same project twice produces identical bytes.
"""

from __future__ import annotations

import io
import json
import re
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .. import __version__
from .tables import DataTable, TableError, table_from_meta

FORMAT_NAME = "torusplot-project"
FORMAT_VERSION = 1
_FIXED_DATE = (2026, 1, 1, 0, 0, 0)


class ProjectError(ValueError):
    """Raised when a project file is missing parts or was written by a newer version."""


def _slug(name: str, used: set[str]) -> str:
    base = re.sub(r"[^A-Za-z0-9_\-]+", "_", name).strip("_") or "table"
    slug, i = base, 2
    while slug.lower() in used:
        slug = f"{base}_{i}"
        i += 1
    used.add(slug.lower())
    return slug


def _dump(obj: Any) -> Any:
    """Pydantic model -> plain JSON-able data; dicts pass through."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    return obj


class Project:
    """In-memory project: tables, figure specs, compositions, statistics results."""

    def __init__(self, name: str = "Untitled project") -> None:
        self.name = name
        self.tables: dict[str, DataTable] = {}
        #: FigureSpec objects (or plain dicts) by figure name.
        self.figures: dict[str, Any] = {}
        #: CompositionSpec objects (or plain dicts) by name.
        self.compositions: dict[str, Any] = {}
        #: StatResult objects (or plain dicts) by result ID.
        self.results: dict[str, Any] = {}
        #: Human-readable audit trail (analyses run, exclusions, AI actions).
        self.log: list[str] = []

    # ----------------------------------------------------------------- tables
    def add_table(self, table: DataTable, replace: bool = True) -> DataTable:
        if not replace and table.name in self.tables:
            raise ProjectError(f"A table named '{table.name}' already exists.")
        self.tables[table.name] = table
        return table

    def unique_table_name(self, base: str) -> str:
        name, i = base, 2
        while name in self.tables:
            name = f"{base} {i}"
            i += 1
        return name

    def add_result(self, result: Any) -> Any:
        rid = result["id"] if isinstance(result, dict) else result.id
        self.results[rid] = result
        return result

    # ------------------------------------------------------------------- save
    def to_manifest(self) -> tuple[dict[str, Any], dict[str, str]]:
        """Return (manifest dict, {archive path: csv text})."""
        used: set[str] = set()
        files: dict[str, str] = {}
        tables_meta = []
        for table in self.tables.values():
            path = f"data/{_slug(table.name, used)}.csv"
            files[path] = table.data.to_csv(index=False, lineterminator="\n")
            tables_meta.append({"file": path, **table.meta()})
        manifest = {
            "format": FORMAT_NAME,
            "format_version": FORMAT_VERSION,
            "app_version": __version__,
            "name": self.name,
            "tables": tables_meta,
            "figures": {k: _dump(v) for k, v in self.figures.items()},
            "compositions": {k: _dump(v) for k, v in self.compositions.items()},
            "log": list(self.log),
        }
        return manifest, files

    def save(self, path: str | Path) -> Path:
        """Write the project to ``path`` (a ``.tplot`` zip archive)."""
        path = Path(path)
        if path.suffix.lower() != ".tplot":
            path = path.with_suffix(".tplot")
        manifest, files = self.to_manifest()
        results = {k: _dump(v) for k, v in self.results.items()}
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:

            def write(name: str, text: str) -> None:
                info = zipfile.ZipInfo(name, date_time=_FIXED_DATE)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16
                zf.writestr(info, text.encode("utf-8"))

            write("project.yaml", yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True))
            for name in sorted(files):
                write(name, files[name])
            write("results.json", json.dumps(results, indent=1, sort_keys=True, allow_nan=True))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(buffer.getvalue())
        return path

    # ------------------------------------------------------------------- load
    @classmethod
    def load(cls, path: str | Path | io.BytesIO) -> Project:
        """Read a ``.tplot`` archive written by :meth:`save`."""
        try:
            zf = zipfile.ZipFile(path)
        except (zipfile.BadZipFile, FileNotFoundError) as exc:
            raise ProjectError(f"Not a readable .tplot file: {exc}") from exc
        with zf:
            names = set(zf.namelist())
            if "project.yaml" not in names:
                raise ProjectError("Not a TorusPlot project: project.yaml is missing.")
            manifest = yaml.safe_load(zf.read("project.yaml").decode("utf-8")) or {}
            if manifest.get("format") != FORMAT_NAME:
                raise ProjectError("Not a TorusPlot project file.")
            if int(manifest.get("format_version", 0)) > FORMAT_VERSION:
                raise ProjectError(
                    "This project was saved by a newer TorusPlot version. Please update TorusPlot."
                )
            project = cls(manifest.get("name", "Untitled project"))
            for meta in manifest.get("tables", []):
                meta = dict(meta)
                file = meta.pop("file")
                if file not in names:
                    raise ProjectError(f"Project is missing data file '{file}'.")
                dtypes = meta.get("dtypes") or {}
                str_cols = {c: str for c, t in dtypes.items() if t != "number"}
                if meta.get("kind") in {"grouped", "contingency"}:
                    header = pd.read_csv(io.BytesIO(zf.read(file)), nrows=0)
                    if len(header.columns):
                        str_cols[header.columns[0]] = str
                frame = pd.read_csv(
                    io.BytesIO(zf.read(file)), dtype=str_cols, keep_default_na=True, na_values=[""]
                )
                try:
                    project.add_table(table_from_meta(frame, meta))
                except TableError as exc:
                    raise ProjectError(f"Table '{meta.get('name')}' could not be loaded: {exc}") from exc
            project.figures = _parse_models(manifest.get("figures") or {}, "figure")
            project.compositions = _parse_models(manifest.get("compositions") or {}, "composition")
            raw_results = json.loads(zf.read("results.json").decode("utf-8")) if "results.json" in names else {}
            project.results = _parse_models(raw_results, "result")
            project.log = list(manifest.get("log") or [])
        return project


def _parse_models(raw: dict[str, Any], what: str) -> dict[str, Any]:
    """Upgrade plain dicts to typed models when the model classes are available."""
    model = None
    try:
        if what == "figure":
            from ..spec import FigureSpec as model
        elif what == "composition":
            from ..spec import CompositionSpec as model
        elif what == "result":
            from ..stats.result import StatResult as model
    except ImportError:  # the data layer also works stand-alone
        model = None
    if model is None:
        return dict(raw)
    out = {}
    for key, value in raw.items():
        out[key] = model.model_validate(value) if isinstance(value, dict) else value
    return out

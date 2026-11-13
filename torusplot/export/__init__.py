"""Exports: figures (SVG/PDF/TIFF/PNG), statistics reports, methods paragraphs."""

from .figure import (
    FORMATS,
    ExportError,
    export_all,
    export_figure,
    figure_bytes,
    figure_size_mm,
    validate_svg,
)
from .methods import data_presentation_sentence, methods_paragraph, significance_legend
from .report import stats_report_csv, stats_report_markdown, stats_report_rows, write_stats_report

__all__ = [
    "FORMATS",
    "ExportError",
    "data_presentation_sentence",
    "export_all",
    "export_figure",
    "figure_bytes",
    "figure_size_mm",
    "methods_paragraph",
    "significance_legend",
    "stats_report_csv",
    "stats_report_markdown",
    "stats_report_rows",
    "validate_svg",
    "write_stats_report",
]

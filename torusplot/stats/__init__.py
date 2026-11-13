"""TorusPlot statistics engine.

Every analysis returns a typed :class:`~torusplot.stats.result.StatResult`. Use
:func:`run_analysis` for table-level access (what the app does) or import the
array-level functions from the individual modules.
"""

from ._utils import ApproximateDataError, StatsError
from .registry import (
    CATEGORIES,
    REGISTRY,
    AnalysisInfo,
    ParamSpec,
    analyses_for,
    get_analysis,
    load_all,
    run_analysis,
)
from .result import (
    AssumptionCheck,
    Comparison,
    EffectSize,
    Parameter,
    StatResult,
    format_p,
    new_result_id,
    p_to_stars,
)

__all__ = [
    "CATEGORIES",
    "REGISTRY",
    "AnalysisInfo",
    "ApproximateDataError",
    "AssumptionCheck",
    "Comparison",
    "EffectSize",
    "ParamSpec",
    "Parameter",
    "StatResult",
    "StatsError",
    "analyses_for",
    "format_p",
    "get_analysis",
    "load_all",
    "new_result_id",
    "p_to_stars",
    "run_analysis",
]

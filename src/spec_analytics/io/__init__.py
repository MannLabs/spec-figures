"""IO subpackage: engine-agnostic experiment dispatch and the concrete
search-engine loaders. See REFACTOR_PLAN.md §3."""

from __future__ import annotations

from .experiments import (  # noqa: F401
    detect_engine,
    load_experiments,
    sample_info_from_experiments,
    split_by_engine,
)
from .peaks import load_peaks  # noqa: F401
from .diann import load_diann  # noqa: F401

from . import experiments, peaks, diann  # noqa: F401

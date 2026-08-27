"""IO subpackage: engine-agnostic experiment dispatch and the concrete"""
from __future__ import annotations
from .experiments import (
    detect_engine,
    load_experiments,
    sample_info_from_experiments,
    split_by_engine,
)
from .peaks import load_peaks
from .diann import load_diann
from . import experiments, peaks, diann

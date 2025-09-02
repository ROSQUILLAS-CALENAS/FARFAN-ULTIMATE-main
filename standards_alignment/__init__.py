"""Standards Alignment System using Gromov-Wasserstein Optimal Transport.

This module provides tools for aligning standards graphs to document graphs
using entropic Gromov-Wasserstein optimal transport with provable stability
guarantees based on JMLR 2024 and JCGS 2023 theoretical foundations.
"""

from .api import (
    get_dimension_patterns,
    get_point_requirements,
    get_verification_criteria,
    load_standards,
)
from .graph_ops import DocumentGraph, StandardsGraph
from .gw_alignment import gw_align, sparse_align
from .patterns import Criterion, PatternSpec, Requirement
from .stable_gw_aligner import StableEGWAligner, StableGWConfig, TransportPlan

__version__ = "0.1.0"
__all__ = [
    "load_standards",
    "get_dimension_patterns",
    "get_point_requirements",
    "get_verification_criteria",
    "StandardsGraph",
    "DocumentGraph",
    "gw_align",
    "sparse_align",
    "PatternSpec",
    "Requirement",
    "Criterion",
    "StableEGWAligner",
    "StableGWConfig",
    "TransportPlan",
]

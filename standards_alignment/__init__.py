"""Standards Alignment System using Gromov-Wasserstein Optimal Transport.

This module provides tools for aligning standards graphs to document graphs
using entropic Gromov-Wasserstein optimal transport with provable stability
guarantees based on JMLR 2024 and JCGS 2023 theoretical foundations.
"""

# # # from .api import (  # Module not found  # Module not found  # Module not found
    get_dimension_patterns,
    get_point_requirements,
    get_verification_criteria,
    load_standards,
)
# # # from .graph_ops import DocumentGraph, StandardsGraph  # Module not found  # Module not found  # Module not found
# # # from .gw_alignment import gw_align, sparse_align  # Module not found  # Module not found  # Module not found
# # # from .patterns import Criterion, PatternSpec, Requirement  # Module not found  # Module not found  # Module not found
# # # from .stable_gw_aligner import StableEGWAligner, StableGWConfig, TransportPlan  # Module not found  # Module not found  # Module not found

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

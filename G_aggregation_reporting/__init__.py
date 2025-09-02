"""
G_aggregation_reporting - Stage: Aggregation and Reporting

This module contains components for evidence aggregation and comprehensive 
report compilation in the EGW Query Expansion system.

Components:
- MesoAggregator: Aggregates evidence from multiple sources and dimensions
- ReportCompiler: Compiles comprehensive reports with structured outputs
"""

# Stage marker comment: G - Aggregation and Reporting Stage
__stage__ = "aggregation_reporting"
__stage_code__ = "G"

from .meso_aggregator import MesoAggregator
from .report_compiler import ReportCompiler

__all__ = [
    'MesoAggregator',
    'ReportCompiler'
]
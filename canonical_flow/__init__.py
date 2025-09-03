"""
Canonical Flow - Organized Pipeline View

This directory contains auto-generated alias files that provide a canonical,
ordered view of the pipeline. Each alias re-exports the original module.

DO NOT EDIT FILES IN THIS DIRECTORY - they will be overwritten.
Edit original files and regenerate using tools/organize_canonical_structure.py
"""

# Import calibration dashboard for orchestrator integration
try:
# # #     from .calibration_dashboard import CalibrationDashboard, CalibrationReport  # Module not found  # Module not found  # Module not found
    __all__ = ['CalibrationDashboard', 'CalibrationReport']
except ImportError:
    __all__ = []

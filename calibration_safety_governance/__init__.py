"""
Auto-Enhancement Orchestration System
=====================================

This package provides comprehensive preflight validation, auto-deactivation monitoring,
and provenance tracking for the calibration safety governance system.

Components:
- preflight_validator: Input schema verification and compatibility checks
- auto_deactivation_monitor: Stability drift and performance regression detection
- provenance_tracker: Enhancement metadata and audit trail generation
- orchestrator: Main coordination system for all enhancement operations
"""

from .preflight_validator import PreflightValidator
from .auto_deactivation_monitor import AutoDeactivationMonitor
from .provenance_tracker import ProvenanceTracker
from .orchestrator import EnhancementOrchestrator

__version__ = "1.0.0"
__all__ = [
    "PreflightValidator",
    "AutoDeactivationMonitor", 
    "ProvenanceTracker",
    "EnhancementOrchestrator"
]
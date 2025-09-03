"""
Compatibility shim: Principal orchestrator entrypoint.

This module provides a stable, clean interface named ComprehensivePipelineOrchestrator
and helpers, delegating to the maintained implementation in pipeline_orchestrator.py.
This removes previous syntax/formatting issues and keeps the project executable.
"""
# # # from __future__ import annotations  # Module not found  # Module not found  # Module not found
# # # from typing import Any, Dict  # Module not found  # Module not found  # Module not found

# Delegate to the maintained orchestrator implementation
# # # from pipeline_orchestrator import (  # Module not found  # Module not found  # Module not found

# Mandatory Pipeline Contract Annotations
__phase__ = "O"
__code__ = "96O"
__stage_order__ = 7

    PipelineOrchestrator as _PipelineOrchestrator,
    ConfigurationValidator as _ConfigurationValidator,
    main as _main,
)


class ComprehensivePipelineOrchestrator(_PipelineOrchestrator):
    """Compatibility alias acting as the principal orchestrator."""


# Backward-compatible export of the original name
PipelineOrchestrator = _PipelineOrchestrator


def run_pipeline(configuration: Dict[str, Any], input_data: Dict[str, Any], *, validate: bool = True) -> Dict[str, Any]:
    """Run the pipeline end-to-end with optional configuration validation.

    Args:
        configuration: Pipeline configuration mapping.
        input_data: Initial input data for the pipeline.
        validate: If True, validates/fixes configuration before running.

    Returns:
        The pipeline execution result as produced by PipelineOrchestrator.execute_pipeline.
    """
    cfg = (
        _ConfigurationValidator.validate_and_fix(configuration)
        if validate
        else configuration
    )
    orchestrator = _PipelineOrchestrator(cfg)
    return orchestrator.execute_pipeline(input_data)


__all__ = [
    "ComprehensivePipelineOrchestrator",
    "PipelineOrchestrator",
    "run_pipeline",
]


if __name__ == "__main__":
    _main()

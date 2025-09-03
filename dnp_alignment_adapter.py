"""
DNP Alignment Adapter

Purpose:
- Provide a stable, callable entrypoint (process) for the deterministic orchestrator
- Enforce DNP standards between evaluation and scoring stages
- Be robust to partial inputs and operate in pass-through mode if needed
- Implement version pinning for DNP standards to ensure consistent mappings
- Provide neutral fallback handling for missing or incomplete alignment data

Inputs (data: dict expected, but tolerant to others):
- Should contain PDT document/package data under keys like 'pdt_data', 'document', or 'result'
- Should contain evaluation results under keys like 'evaluation_results' or 'results'

Outputs:
- Returns the input data enriched with:
  - 'dnp_compliance': raw compliance results by standard
  - 'dnp_report': synthesized DNP compliance report
  - 'dnp_guardian': {'enforced': True, 'timestamp': ..., 'engine': 'DNPAlignmentEngine', 'version': ...}
  - 'dnp_alignment': standardized alignment scores and mappings

This adapter avoids hard crashes and logs errors while keeping the pipeline flowing.
"""
# # # from __future__ import annotations  # Module not found  # Module not found  # Module not found

# Import audit logger for execution tracing

# Mandatory Pipeline Contract Annotations
__phase__ = "K"
__code__ = "16K"
__stage_order__ = 3

try:
# # #     from canonical_flow.analysis.audit_logger import get_audit_logger  # Module not found  # Module not found  # Module not found
except ImportError:
    # Fallback when audit logger is not available
    get_audit_logger = None

# # # from datetime import datetime, timezone  # Module not found  # Module not found  # Module not found
# # # from typing import Any, Dict, Optional, Union  # Module not found  # Module not found  # Module not found
import uuid
import logging

logger = logging.getLogger(__name__)

# Try to import centralized configuration
try:
# # #     from config_loader import get_thresholds  # Module not found  # Module not found  # Module not found
    THRESHOLDS_AVAILABLE = True
except ImportError:
    THRESHOLDS_AVAILABLE = False
    logger.warning("Centralized thresholds not available, using default values")

# DNP Standards Version Configuration
DNP_VERSION = "1.2.0"
DNP_SCHEMA_VERSION = "2024.1"
DNP_COMPATIBILITY_MATRIX = {
    "1.0.0": {"schema": "2023.1", "deprecated": True},
    "1.1.0": {"schema": "2023.2", "deprecated": True},
    "1.2.0": {"schema": "2024.1", "deprecated": False}
}

def _load_default_values() -> tuple[Dict[str, float], Dict[str, Any]]:
# # #     """Load default values from centralized configuration or fallback to hardcoded."""  # Module not found  # Module not found  # Module not found
    if THRESHOLDS_AVAILABLE:
        try:
            config = get_thresholds()
            dnp_config = config.dnp_alignment
            
            alignment_scores = dnp_config.default_alignment_scores.copy()
            mappings = {
                "pnd_alignment": {"score": 0.5, "mapped_objectives": [], "gaps": []},
                "ods_mapping": {"score": 0.5, "mapped_targets": [], "coverage": 0.0},
                "sectoral_consistency": {"score": 0.5, "consistent_sectors": [], "conflicts": []},
                "competencias_mapping": {"score": 0.5, "valid_competencies": [], "violations": []}
            }
            return alignment_scores, mappings
        except Exception as e:
            logger.debug(f"Failed to load centralized defaults: {e}")
    
    # Fallback to hardcoded values
    alignment_scores = {
        "gpr_alignment": 0.5,
        "sgp_compliance": 0.5,
        "sinergia_integration": 0.5,
        "competencias_score": 0.5,
        "territorial_coherence": 0.5,
        "overall_alignment": 0.5
    }
    
    mappings = {
        "pnd_alignment": {"score": 0.5, "mapped_objectives": [], "gaps": []},
        "ods_mapping": {"score": 0.5, "mapped_targets": [], "coverage": 0.0},
        "sectoral_consistency": {"score": 0.5, "consistent_sectors": [], "conflicts": []},
        "competencias_mapping": {"score": 0.5, "valid_competencies": [], "violations": []}
    }
    
    return alignment_scores, mappings

# Load default values
DEFAULT_ALIGNMENT_SCORES, DEFAULT_MAPPINGS = _load_default_values()


def _extract_pdt_and_eval(payload: Any) -> (Dict[str, Any], Dict[str, Any]):
# # #     """Best-effort extraction of PDT data and evaluation results from arbitrary payloads."""  # Module not found  # Module not found  # Module not found
    pdt_data: Dict[str, Any] = {}
    eval_results: Dict[str, Any] = {}

    if isinstance(payload, dict):
        # Common locations
        pdt_data = (
            payload.get("pdt_data")
            or payload.get("document")
            or payload.get("result")
            or payload.get("data")
            or {}
        )
        if not isinstance(pdt_data, dict):
            pdt_data = {"raw": pdt_data}

        eval_results = (
            payload.get("evaluation_results")
            or payload.get("results")
            or payload.get("analysis")
            or {}
        )
        if not isinstance(eval_results, dict):
            eval_results = {"raw": eval_results}
    else:
        # Non-dict payloads
        pdt_data = {"raw": payload}
        eval_results = {}

    return pdt_data, eval_results


def _create_version_info() -> Dict[str, Any]:
    """Create version information for DNP standards tracking."""
    return {
        "dnp_version": DNP_VERSION,
        "schema_version": DNP_SCHEMA_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": str(uuid.uuid4()),
        "compatibility": DNP_COMPATIBILITY_MATRIX.get(DNP_VERSION, {}),
        "deprecated": DNP_COMPATIBILITY_MATRIX.get(DNP_VERSION, {}).get("deprecated", False)
    }

def _create_neutral_fallback(evidence_data: Dict[str, Any], process_timestamp: str) -> Dict[str, Any]:
    """Create neutral fallback alignment when DNP standards are missing or incomplete."""
    return {
        "dnp_alignment": {
            "status": "neutral_fallback",
            "scores": DEFAULT_ALIGNMENT_SCORES.copy(),
            "mappings": DEFAULT_MAPPINGS.copy(),
            "evidence_count": len(evidence_data.get("evidence", [])) if "evidence" in evidence_data else 0,
            "fallback_reason": "DNP standards missing or unavailable",
            "timestamp": process_timestamp
        },
        "dnp_compliance": {
            "overall_status": "not_evaluated",
            "standards_applied": [],
            "fallback_applied": True,
            "neutral_scores": DEFAULT_ALIGNMENT_SCORES
        },
        "dnp_report": {
            "summary": "DNP alignment not available - neutral fallback applied",
            "recommendations": [
                "Ensure DNP standards are properly loaded",
                "Verify alignment engine configuration",
                "Check evidence data completeness"
            ],
            "status": "fallback_mode"
        }
    }

def process(data: Any, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Standardized process() API contract for DNP alignment.
    
    Takes evidence data as input and returns DNP-aligned evidence with:
    - Version-pinned DNP standards tracking
    - Standardized status reporting
    - Neutral fallback handling for missing/incomplete data
    - Audit logging for component execution tracing
    
    Args:
        data: Evidence data (dict expected, but tolerant to other types)
        context: Optional processing context
        
    Returns:
        Dict with enriched evidence data including DNP alignment information
    """
    # Audit logging for component execution
    audit_logger = get_audit_logger() if get_audit_logger else None
    input_data = {
        "data_type": type(data).__name__,
        "data_keys": list(data.keys()) if isinstance(data, dict) else [],
        "context_provided": context is not None
    }
    
    if audit_logger:
        with audit_logger.audit_component_execution("20A", input_data) as audit_ctx:
            result = _process_internal(data, context)
            audit_ctx.set_output({
                "dnp_enforced": result.get("dnp_guardian", {}).get("enforced", False),
                "compliance_keys": list(result.keys()) if isinstance(result, dict) else []
            })
            return result
    else:
        return _process_internal(data, context)

def _process_internal(data: Any, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Internal implementation of DNP alignment processing with version pinning and fallback handling."""
    process_timestamp = datetime.now(timezone.utc).isoformat()
    
    # Normalize to dict output base
    base: Dict[str, Any] = data if isinstance(data, dict) else {"data": data}

    # Create version information for standards tracking
    version_info = _create_version_info()

    # Initialize guardian metadata with version pinning
    base.setdefault("dnp_guardian", {})
    base["dnp_guardian"].update({
        "enforced": False,
        "timestamp": process_timestamp,
        "engine": "DNPAlignmentEngine",
        "context": (context or {}),
        "version_info": version_info,
        "process_id": str(uuid.uuid4())
    })

    # Check for deprecated version warning
    if version_info.get("deprecated", False):
        logger.warning(f"Using deprecated DNP version {DNP_VERSION}. Consider upgrading.")
        base["dnp_guardian"]["deprecated_version_warning"] = True

    try:
        # Lazy import to avoid heavy cost when unused
# # #         from dnp_alignment_engine import (  # Module not found  # Module not found  # Module not found
            DNPAlignmentEngine,
            create_dnp_alignment_engine,
        )
    except Exception as e:  # pragma: no cover - robustness first
        logger.warning("DNP engine import failed: %s", e)
        base["dnp_guardian"]["error"] = f"import_error: {e}"
        base["dnp_guardian"]["fallback_applied"] = True
        
        # Apply neutral fallback
        fallback_data = _create_neutral_fallback(base, process_timestamp)
        base.update(fallback_data)
        return base

    # Build engine (factory preferred when present)
    try:
        engine: DNPAlignmentEngine = create_dnp_alignment_engine()  # type: ignore
    except Exception:
        try:
            engine = DNPAlignmentEngine()  # type: ignore
        except Exception as e:  # pragma: no cover
            logger.warning("DNP engine init failed: %s", e)
            base["dnp_guardian"]["error"] = f"init_error: {e}"
            base["dnp_guardian"]["fallback_applied"] = True
            
            # Apply neutral fallback
            fallback_data = _create_neutral_fallback(base, process_timestamp)
            base.update(fallback_data)
            return base

    # Extract inputs
    pdt_data, eval_results = _extract_pdt_and_eval(base)

    # Attempt DNP alignment with version tracking
    try:
        # Add version info to engine context
        engine_context = {"version_info": version_info, "timestamp": process_timestamp}
        
        compliance = engine.evaluate_dnp_compliance(
            pdt_data=pdt_data, 
            evaluation_results=eval_results
        )
        report = engine.generate_compliance_report(compliance)

        # Structure standardized output
        base["dnp_compliance"] = {
            k: (v.__dict__ if hasattr(v, "__dict__") else v) 
            for k, v in compliance.items()
        }
        base["dnp_compliance"]["version_tracking"] = version_info
        
        base["dnp_report"] = report
        base["dnp_report"]["version_info"] = version_info
        
        # Add standardized alignment scores
        base["dnp_alignment"] = {
            "status": "aligned",
            "scores": _extract_alignment_scores(compliance),
            "mappings": _extract_alignment_mappings(compliance),
            "evidence_processed": len(pdt_data) + len(eval_results),
            "timestamp": process_timestamp,
            "version": version_info
        }
        
        base["dnp_guardian"]["enforced"] = True
        base["dnp_guardian"]["status"] = "success"
        return base
        
    except Exception as e:
        logger.warning("DNP evaluation failed: %s", e)
        base["dnp_guardian"]["error"] = f"evaluation_error: {e}"
        base["dnp_guardian"]["fallback_applied"] = True
        base["dnp_guardian"]["status"] = "failed_with_fallback"
        
        # Apply neutral fallback when evaluation fails
        fallback_data = _create_neutral_fallback(base, process_timestamp)
        base.update(fallback_data)
        return base

def _extract_alignment_scores(compliance: Dict[str, Any]) -> Dict[str, float]:
# # #     """Extract standardized alignment scores from compliance results."""  # Module not found  # Module not found  # Module not found
    scores = DEFAULT_ALIGNMENT_SCORES.copy()
    
    try:
        for standard_id, result in compliance.items():
            if hasattr(result, 'score'):
                if 'gpr' in standard_id.lower():
                    scores["gpr_alignment"] = max(scores["gpr_alignment"], result.score)
                elif 'sgp' in standard_id.lower():
                    scores["sgp_compliance"] = max(scores["sgp_compliance"], result.score)
                elif 'sinergia' in standard_id.lower():
                    scores["sinergia_integration"] = max(scores["sinergia_integration"], result.score)
                elif 'competencia' in standard_id.lower():
                    scores["competencias_score"] = max(scores["competencias_score"], result.score)
        
        # Calculate overall alignment score
        scores["overall_alignment"] = sum(scores.values()) / len(scores)
        
    except Exception as e:
        logger.debug(f"Score extraction partial failure: {e}")
        
    return scores

def _extract_alignment_mappings(compliance: Dict[str, Any]) -> Dict[str, Any]:
# # #     """Extract standardized alignment mappings from compliance results."""  # Module not found  # Module not found  # Module not found
    mappings = DEFAULT_MAPPINGS.copy()
    
    try:
        for standard_id, result in compliance.items():
            if hasattr(result, 'evidence') and hasattr(result, 'score'):
                if 'sinergia' in standard_id.lower():
                    mappings["pnd_alignment"]["score"] = result.score
                    mappings["pnd_alignment"]["mapped_objectives"] = result.evidence[:5]  # Top 5
                elif 'competencia' in standard_id.lower():
                    mappings["competencias_mapping"]["score"] = result.score
                    mappings["competencias_mapping"]["valid_competencies"] = result.evidence
                    
    except Exception as e:
        logger.debug(f"Mapping extraction partial failure: {e}")
        
    return mappings
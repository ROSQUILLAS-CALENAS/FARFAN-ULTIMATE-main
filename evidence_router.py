"""
Evidence Router
Routes evidence to appropriate processors/pipelines by type, dimension, or confidence.

Design goals:
- Minimal dependencies; integrate with existing evidence_processor dataclasses if available.
- Robust to different evidence shapes: StructuredEvidence, dicts, or generic objects.
- Return string names of processors/pipelines present in this repo (normative_validator, rubric_validator,
  adaptive_scoring_engine, evidence_processor) and generic pipeline names for orchestration.
"""
from __future__ import annotations

from typing import Any, Optional

# Optional imports from evidence_processor for better typing and field access
try:
    from evidence_processor import EvidenceType, ConfidenceLevel
except Exception:  # pragma: no cover - make router resilient if import path differs
    EvidenceType = None  # type: ignore
    ConfidenceLevel = None  # type: ignore


class EvidenceRouter:
    """Routes evidence to appropriate processors"""

    def __init__(self) -> None:
        # Static maps for type-based routing
        self._type_to_processor = {
            "direct_quote": "quote_processor",
            "paraphrase": "paraphrase_processor",
            "statistical": "statistical_processor",
            "expert_opinion": "expert_review_processor",
            "case_study": "case_study_processor",
        }

    # ----------------------- Helpers -----------------------
    def _get_evidence_type(self, evidence: Any) -> Optional[str]:
        # StructuredEvidence.evidence_type is an Enum; support multiple shapes
        et = None
        try:
            # evidence.evidence_type may be Enum; get value if present
            et = getattr(evidence, "evidence_type", None)
            if et is not None and hasattr(et, "value"):
                et = et.value
        except Exception:
            et = None
        if et is None:
            # dict-like
            if isinstance(evidence, dict):
                et = evidence.get("evidence_type")
                if hasattr(et, "value"):
                    et = et.value
        # Normalize
        if isinstance(et, str):
            return et.lower()
        return None

    def _get_dimension(self, evidence: Any, dimension: Optional[str]) -> Optional[str]:
        dim = dimension
        if dim is None:
            try:
                dim = getattr(evidence, "dimension", None)
            except Exception:
                dim = None
            if dim is None and isinstance(evidence, dict):
                dim = evidence.get("dimension")
        return dim.lower() if isinstance(dim, str) else None

    def _get_confidence_level(self, evidence: Any) -> Optional[str]:
        # Prefer scoring.confidence_level if present
        level = None
        try:
            scoring = getattr(evidence, "scoring", None)
            if scoring is not None:
                level = getattr(scoring, "confidence_level", None)
                if hasattr(level, "value"):
                    level = level.value
        except Exception:
            level = None
        if level is None and isinstance(evidence, dict):
            scoring = evidence.get("scoring", {})
            if isinstance(scoring, dict):
                level = scoring.get("confidence_level")
                if hasattr(level, "value"):
                    level = getattr(level, "value")
        return level.lower() if isinstance(level, str) else None

    def _get_overall_score(self, evidence: Any) -> Optional[float]:
        try:
            scoring = getattr(evidence, "scoring", None)
            if scoring is not None:
                score = getattr(scoring, "overall_score", None)
                if isinstance(score, (int, float)):
                    return float(score)
        except Exception:
            pass
        if isinstance(evidence, dict):
            scoring = evidence.get("scoring", {})
            if isinstance(scoring, dict):
                score = scoring.get("overall_score")
                if isinstance(score, (int, float)):
                    return float(score)
        return None

    # ---------------------- Public API ----------------------
    def route_by_type(self, evidence: Any) -> str:
        """Route based on evidence type to a processor name.

        Returns one of: quote_processor, paraphrase_processor, statistical_processor,
        expert_review_processor, case_study_processor, or general_processor (default).
        """
        et = self._get_evidence_type(evidence)
        if et and et in self._type_to_processor:
            return self._type_to_processor[et]
        # If Enum class available, map by identity for robustness
        if EvidenceType is not None:
            try:
                et_obj = getattr(evidence, "evidence_type", None)
                if isinstance(et_obj, EvidenceType):  # type: ignore[arg-type]
                    return self._type_to_processor.get(et_obj.value, "general_processor")
            except Exception:
                pass
        return "general_processor"

    def route_by_dimension(self, evidence: Any, dimension: str) -> str:
        """Route to a processor based on the dimension/topic.

        Known processors: normative_validator, rubric_validator, adaptive_scoring_engine,
        evidence_processor. Falls back to general_processor if no match.
        """
        dim = self._get_dimension(evidence, dimension)
        if not dim:
            return "general_processor"
        d = dim.lower()
        # Keyword-based mapping
        if any(k in d for k in ["normative", "legal", "regulatory", "compliance", "law"]):
            return "normative_validator"
        if any(k in d for k in ["rubric", "quality", "scoring", "evaluation", "criteria"]):
            return "rubric_validator"
        if any(k in d for k in ["adaptive", "predict", "ml", "model", "forecast", "ai"]):
            return "adaptive_scoring_engine"
        if any(k in d for k in ["evidence", "citation", "trace", "source"]):
            return "evidence_processor"
        return "general_processor"

    def route_by_confidence(self, evidence: Any) -> str:
        """Route based on confidence/score.

        - High confidence (>=0.8 or HIGH) => high_conf_pipeline
        - Medium confidence (>=0.6 or MEDIUM) => standard_pipeline
        - Else => human_review
        """
        level = self._get_confidence_level(evidence)
        if level == "high":
            return "high_conf_pipeline"
        if level == "medium":
            return "standard_pipeline"
        if level == "low":
            return "human_review"

        score = self._get_overall_score(evidence)
        if score is None:
            return "standard_pipeline"
        if score >= 0.8:
            return "high_conf_pipeline"
        if score >= 0.6:
            return "standard_pipeline"
        return "human_review"

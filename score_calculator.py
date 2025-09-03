"""
Score Calculator
Calculates scores using DNP-style formulas with robust defaults.

Public API:
- ScoreCalculator.calculate_dimension_score(dimension: str, evidence: List) -> float
- ScoreCalculator.apply_weights(scores: Dict, weights: Dict) -> float
- ScoreCalculator.calculate_conjunctive_score(scores: Dict, rules: List) -> float

Notes:
# # # - Works with StructuredEvidence from evidence_processor.py but also accepts  # Module not found  # Module not found  # Module not found
  dict-like or generic objects exposing similar fields.
- All outputs are normalized to [0, 1].
- Integrated with contracts schema validation system.
"""
# # # from __future__ import annotations  # Module not found  # Module not found  # Module not found

# # # from typing import Any, Dict, List, Optional  # Module not found  # Module not found  # Module not found
# # # from datetime import datetime  # Module not found  # Module not found  # Module not found

# Optional import for typing/Enum usage; keep robust if not available
try:
# # #     from evidence_processor import ConfidenceLevel  # Module not found  # Module not found  # Module not found
except Exception:  # pragma: no cover
    ConfidenceLevel = None  # type: ignore

# Import schema validation components
try:
# # #     from contracts.schemas import (  # Module not found  # Module not found  # Module not found
        DimensionEvalOutput,
        PointEvalOutput,
        StageMeta,
        ComplianceStatus,
        ConfidenceLevel as SchemaConfidenceLevel,
        ProcessingStatus,
        validate_process_schemas,
        enforce_required_fields,
        create_stage_meta,
    )
    CONTRACTS_AVAILABLE = True
except ImportError:
    CONTRACTS_AVAILABLE = False


class ScoreCalculator:
    """Calculates scores using DNP formulas"""

    def __init__(self) -> None:
        # Default weights aligned with EvidenceScoringSystem
        self.default_component_weights = {
            "relevance": 0.4,
            "credibility": 0.3,
            "recency": 0.2,
            "authority": 0.1,
        }

    # --------------------- Helpers ---------------------
    def _get_dimension(self, item: Any) -> Optional[str]:
        if hasattr(item, "dimension"):
            dim = getattr(item, "dimension")
            return dim.lower() if isinstance(dim, str) else None
        if isinstance(item, dict):
            dim = item.get("dimension")
            return dim.lower() if isinstance(dim, str) else None
        return None

    def _get_scoring(self, item: Any) -> Dict[str, float]:
        # Extract scoring metrics with safe defaults
        scoring = {}
        src = None
        if hasattr(item, "scoring"):
            src = getattr(item, "scoring")
        elif isinstance(item, dict):
            src = item.get("scoring")
        if isinstance(src, dict):
            scoring = src
        elif src is not None:
            # Likely ScoringMetrics dataclass; fetch attributes
            for k in [
                "relevance_score",
                "credibility_score",
                "recency_score",
                "authority_score",
                "overall_score",
                "confidence_level",
                "classification_labels",
            ]:
                try:
                    scoring[k] = getattr(src, k)
                except Exception:
                    pass

        # Normalize and defaults
        def _get(name: str, default: float = 0.0) -> float:
            v = scoring.get(name)
            try:
                v = float(v)
            except Exception:
                v = default
            return max(0.0, min(1.0, v))

        return {
            "relevance": _get("relevance_score"),
            "credibility": _get("credibility_score"),
            "recency": _get("recency_score"),
            "authority": _get("authority_score"),
            "overall": _get("overall_score"),
            "_raw_conf_level": scoring.get("confidence_level"),
            "_labels": scoring.get("classification_labels", []),
        }

    # --------------------- Public API ---------------------
    def calculate_dimension_score(self, dimension: str, evidence: List[Any]) -> float:
        """Aggregate evidence into a single dimension score in [0,1].

        DNP-style logic:
        - Use component weights (relevance, credibility, recency, authority) aligned with
          EvidenceScoringSystem.
        - Emphasize items whose dimension matches the target dimension.
        - Adjust by confidence level and weakly boost when classification labels include the dimension.
        """
        if not evidence:
            return 0.0
        dim_l = (dimension or "").lower()
        scores = []
        weights = []

        for item in evidence:
            item_dim = self._get_dimension(item) or ""
            sc = self._get_scoring(item)
            # Base component score
            comp = (
                sc["relevance"] * self.default_component_weights["relevance"]
                + sc["credibility"] * self.default_component_weights["credibility"]
                + sc["recency"] * self.default_component_weights["recency"]
                + sc["authority"] * self.default_component_weights["authority"]
            )
            # Dimension alignment boost
            if dim_l and dim_l in item_dim:
                comp = min(1.0, comp + 0.05)
            # Classification label hint boost
            labels = sc.get("_labels") or []
            if (
                isinstance(labels, list)
                and dim_l
                and any(dim_l in str(lbl).lower() for lbl in labels)
            ):
                comp = min(1.0, comp + 0.05)
            # Confidence adjustment
            conf_level = sc.get("_raw_conf_level")
            try:
                # Enum to str if available
                if hasattr(conf_level, "value"):
                    conf_level = conf_level.value
            except Exception:
                pass
            if isinstance(conf_level, str):
                lv = conf_level.lower()
                if lv == "high":
                    comp = min(1.0, comp + 0.05)
                elif lv == "low":
                    comp = max(0.0, comp - 0.1)
            # Weight higher quality evidence more (based on overall)
            w = 0.5 + 0.5 * sc["overall"]  # in [0.5, 1.0]
            scores.append(comp)
            weights.append(w)

        # Weighted average
        total_w = sum(weights)
        if total_w <= 0:
            return 0.0
        agg = sum(s * w for s, w in zip(scores, weights)) / total_w
        return float(max(0.0, min(1.0, agg)))

    def apply_weights(
        self, scores: Dict[str, float], weights: Dict[str, float]
    ) -> float:
        """Apply weights to a dict of scores and return normalized weighted average in [0,1]."""
        if not scores or not weights:
            return 0.0
        denom = sum(float(w) for w in weights.values())
        if denom <= 0:
            return 0.0
        num = 0.0
        for k, w in weights.items():
            v = float(scores.get(k, 0.0))
            num += v * float(w)
        res = num / denom
        return float(max(0.0, min(1.0, res)))

    def calculate_conjunctive_score(
        self, scores: Dict[str, float], rules: List[Any]
    ) -> float:
# # #         """Calculate a conjunctive (AND) score from rule scores.  # Module not found  # Module not found  # Module not found

        rules: list of dicts or strings. Dict format example:
        {"id": "R1", "mandatory": True, "threshold": 0.6}
        - Mandatory rules form a hard AND via min-satisfaction.
        - Optional rules form a soft conjunction via average satisfaction.
        The final score multiplies hard-AND and soft-AND components (bounded in [0,1]).
        """
        if not rules:
            return 0.0
        mandatory_sats: List[float] = []
        optional_sats: List[float] = []

        for rule in rules:
            if isinstance(rule, str):
                rid = rule
                mandatory = False
                thr = 0.6
            elif isinstance(rule, dict):
                rid = str(rule.get("id") or rule.get("rule_id") or "").strip()
                mandatory = bool(rule.get("mandatory", False))
                thr = float(rule.get("threshold", 0.6))
            else:
                # Unknown format, skip
                continue
            val = float(scores.get(rid, 0.0))
            thr = max(1e-6, min(1.0, thr))
            sat = max(0.0, min(1.0, val / thr))  # satisfaction relative to threshold
            if mandatory:
                mandatory_sats.append(sat)
            else:
                optional_sats.append(sat)

        hard_and = min(mandatory_sats) if mandatory_sats else 1.0
        if optional_sats:
            soft_and = sum(optional_sats) / len(optional_sats)
        else:
            soft_and = 1.0
        final = hard_and * soft_and
        return float(max(0.0, min(1.0, final)))

    @enforce_required_fields("doc_id", "page_num") if CONTRACTS_AVAILABLE else lambda f: f
    def process(self, data: Optional[Dict[str, Any]] = None, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process scoring request with schema validation
        
        Args:
            data: Input data containing doc_id, page_num and scoring parameters
            context: Optional processing context
            
        Returns:
            Dict containing scoring results with proper schema validation
        """
        if data is None:
            data = {}
            
        start_time = datetime.now()
        stage_meta = None
        
        try:
            # Extract required document fields
            doc_id = data.get("doc_id", "unknown_doc")
            page_num = data.get("page_num", 1)
            
            # Extract scoring parameters
            dimension = data.get("dimension", "")
            evidence_list = data.get("evidence", [])
            scores_dict = data.get("scores", {})
            weights_dict = data.get("weights", {})
            rules_list = data.get("rules", [])
            operation = data.get("operation", "dimension_score")
            
            # Perform requested scoring operation
            result_score = 0.0
            operation_metadata = {}
            
            if operation == "dimension_score" and dimension:
                result_score = self.calculate_dimension_score(dimension, evidence_list)
                operation_metadata = {
                    "operation": "dimension_score",
                    "dimension": dimension,
                    "evidence_count": len(evidence_list)
                }
            elif operation == "weighted_score" and scores_dict and weights_dict:
                result_score = self.apply_weights(scores_dict, weights_dict)
                operation_metadata = {
                    "operation": "weighted_score",
                    "scores_count": len(scores_dict),
                    "weights_count": len(weights_dict)
                }
            elif operation == "conjunctive_score" and scores_dict and rules_list:
                result_score = self.calculate_conjunctive_score(scores_dict, rules_list)
                operation_metadata = {
                    "operation": "conjunctive_score", 
                    "scores_count": len(scores_dict),
                    "rules_count": len(rules_list)
                }
            else:
                # Default fallback - try dimension score if evidence provided
                if evidence_list:
                    result_score = self.calculate_dimension_score(dimension or "default", evidence_list)
                    operation_metadata = {
                        "operation": "fallback_dimension_score",
                        "evidence_count": len(evidence_list)
                    }
            
            # Determine compliance status based on score
            if result_score >= 0.75:
                compliance_status = ComplianceStatus.CUMPLE if CONTRACTS_AVAILABLE else "CUMPLE"
            elif result_score >= 0.50:
                compliance_status = ComplianceStatus.CUMPLE_PARCIAL if CONTRACTS_AVAILABLE else "CUMPLE_PARCIAL"
            else:
                compliance_status = ComplianceStatus.NO_CUMPLE if CONTRACTS_AVAILABLE else "NO_CUMPLE"
            
            # Determine confidence level based on evidence quality
            confidence_level = SchemaConfidenceLevel.MEDIUM if CONTRACTS_AVAILABLE else "medium"
            if len(evidence_list) >= 5:
                confidence_level = SchemaConfidenceLevel.HIGH if CONTRACTS_AVAILABLE else "high"
            elif len(evidence_list) < 2:
                confidence_level = SchemaConfidenceLevel.LOW if CONTRACTS_AVAILABLE else "low"
            
            # Create stage metadata
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            stage_meta = create_stage_meta(
                stage_name="score_calculator",
                processing_status=ProcessingStatus.SUCCESS,
                stage_version="2.0.0",
                execution_time_ms=execution_time,
                performance_metrics={
                    "calculated_score": result_score,
                    "evidence_processed": len(evidence_list)
                }
            ) if CONTRACTS_AVAILABLE else None
            
            # Create output based on dimension or point scoring
            output_data = None
            if CONTRACTS_AVAILABLE and dimension and dimension.startswith("DE"):
                # Create dimension evaluation output
                output_data = DimensionEvalOutput(
                    doc_id=doc_id,
                    page_num=page_num,
                    dimension_id=dimension,
                    dimension_name=f"Dimension {dimension}",
                    score=result_score,
                    compliance_status=compliance_status,
                    confidence_level=confidence_level,
                    evidence_count=len(evidence_list),
                    processing_metadata=operation_metadata
                ).dict()
            elif CONTRACTS_AVAILABLE and dimension and dimension.startswith("P"):
                # Create point evaluation output
                output_data = PointEvalOutput(
                    doc_id=doc_id,
                    page_num=page_num,
                    point_id=dimension,
                    point_title=f"Point {dimension}",
                    score=result_score,
                    compliance_status=compliance_status,
                    confidence_level=confidence_level,
                    evidence_count=len(evidence_list),
                    processing_metadata=operation_metadata
                ).dict()
            
            return {
                "doc_id": doc_id,
                "page_num": page_num,
                "status": "success",
                "score": result_score,
                "compliance_status": compliance_status.value if hasattr(compliance_status, 'value') else compliance_status,
                "confidence_level": confidence_level.value if hasattr(confidence_level, 'value') else confidence_level,
                "evaluation_output": output_data,
                "stage_metadata": stage_meta.dict() if stage_meta else {},
                "processing_timestamp": start_time.isoformat(),
                "operation_metadata": operation_metadata
            }
            
        except Exception as e:
            # Create error stage metadata
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            stage_meta = create_stage_meta(
                stage_name="score_calculator", 
                processing_status=ProcessingStatus.FAILED,
                stage_version="2.0.0",
                execution_time_ms=execution_time,
                error_details={"error_message": str(e), "error_type": type(e).__name__}
            ) if CONTRACTS_AVAILABLE else None
            
            return {
                "doc_id": data.get("doc_id", "unknown_doc"),
                "page_num": data.get("page_num", 1),
                "status": "failed",
                "error": str(e),
                "stage_metadata": stage_meta.dict() if stage_meta else {},
                "processing_timestamp": start_time.isoformat()
            }

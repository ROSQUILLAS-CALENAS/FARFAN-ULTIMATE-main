"""
Evidence Adapter Module
# # # Transforms structured evidence outputs from EvidenceProcessor into QuestionEvalInput objects  # Module not found  # Module not found  # Module not found
with deterministic evidence_completeness and page_reference_quality scores.
"""

import json
import logging
# # # from dataclasses import dataclass, field, asdict  # Module not found  # Module not found  # Module not found
# # # from typing import Dict, List, Any, Optional, Tuple  # Module not found  # Module not found  # Module not found
# # # from decimal import Decimal  # Module not found  # Module not found  # Module not found
# # # from collections import OrderedDict  # Module not found  # Module not found  # Module not found

logger = logging.getLogger(__name__)


@dataclass
class QuestionEvalInput:
    """Standardized input object for question evaluation with evidence quality metrics."""
    question_id: str
    response: str  # "Sí", "Parcial", "No", "NI"
    evidence_completeness: float  # Ratio of questions with valid evidence to total questions
    page_reference_quality: float  # Fraction of evidence items with both page_num and exact_text
    evidence_metadata: Dict[str, Any] = field(default_factory=dict)
    processing_timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with deterministic ordering."""
        return OrderedDict([
            ("question_id", self.question_id),
            ("response", self.response),
            ("evidence_completeness", self.evidence_completeness),
            ("page_reference_quality", self.page_reference_quality),
            ("evidence_metadata", self.evidence_metadata),
            ("processing_timestamp", self.processing_timestamp)
        ])

    def to_json(self) -> str:
        """Convert to deterministic JSON serialization."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(',', ':'))


@dataclass
class EvidenceQualityMetrics:
    """Aggregated evidence quality metrics for a set of questions."""
    total_questions: int
    questions_with_valid_evidence: int
    total_evidence_items: int
    evidence_items_with_complete_references: int
    evidence_completeness: float
    page_reference_quality: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with deterministic ordering."""
        return OrderedDict([
            ("total_questions", self.total_questions),
            ("questions_with_valid_evidence", self.questions_with_valid_evidence),
            ("total_evidence_items", self.total_evidence_items),
            ("evidence_items_with_complete_references", self.evidence_items_with_complete_references),
            ("evidence_completeness", self.evidence_completeness),
            ("page_reference_quality", self.page_reference_quality)
        ])


class EvidenceAdapter:
    """
# # #     Adapter that transforms structured evidence outputs from EvidenceProcessor  # Module not found  # Module not found  # Module not found
    into QuestionEvalInput objects with deterministic quality metrics.
    """

    def __init__(self, precision: int = 4):
        """
        Initialize evidence adapter.
        
        Args:
            precision: Number of decimal places for rounding calculations
        """
        self.precision = precision
        self.processing_stats = {
            "total_adaptations": 0,
            "valid_evidence_count": 0,
            "invalid_evidence_count": 0,
            "complete_reference_count": 0
        }

    def transform_evidence_to_eval_inputs(
        self,
        evidence_data: Dict[str, Any],
        question_responses: Dict[str, str]
    ) -> Tuple[List[QuestionEvalInput], EvidenceQualityMetrics]:
        """
        Transform evidence processor output into evaluation inputs with calculated quality metrics.
        
        Args:
# # #             evidence_data: Structured output from EvidenceProcessor  # Module not found  # Module not found  # Module not found
            question_responses: Dictionary mapping question_id to response ("Sí", "Parcial", "No", "NI")
        
        Returns:
            Tuple of (list of QuestionEvalInput objects, aggregated quality metrics)
        """
# # #         # Extract evidence items from evidence_data  # Module not found  # Module not found  # Module not found
        evidence_items = self._extract_evidence_items(evidence_data)
        
        # Calculate aggregated quality metrics
        quality_metrics = self._calculate_quality_metrics(evidence_items, len(question_responses))
        
        # Transform each question response into QuestionEvalInput
        eval_inputs = []
        for question_id, response in sorted(question_responses.items()):
            eval_input = self._create_question_eval_input(
                question_id=question_id,
                response=response,
                evidence_completeness=quality_metrics.evidence_completeness,
                page_reference_quality=quality_metrics.page_reference_quality,
                evidence_metadata=self._extract_question_evidence_metadata(question_id, evidence_items)
            )
            eval_inputs.append(eval_input)
        
        # Update processing stats
        self._update_processing_stats(evidence_items)
        
        return eval_inputs, quality_metrics

    def _extract_evidence_items(self, evidence_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
# # #         Extract evidence items from EvidenceProcessor output.  # Module not found  # Module not found  # Module not found
        
        Args:
            evidence_data: Raw evidence processor output
            
        Returns:
            List of evidence items with standardized structure
        """
        evidence_items = []
        
        # Handle different evidence data structures
        if "evidences" in evidence_data and isinstance(evidence_data["evidences"], list):
            for evidence in evidence_data["evidences"]:
                evidence_items.append(self._standardize_evidence_item(evidence))
        
        elif "processed_evidences" in evidence_data and isinstance(evidence_data["processed_evidences"], list):
            for evidence in evidence_data["processed_evidences"]:
                evidence_items.append(self._standardize_evidence_item(evidence))
        
        elif "evidence" in evidence_data:
            evidence_items.append(self._standardize_evidence_item(evidence_data["evidence"]))
        
        else:
# # #             # Try to extract any evidence-like objects from the data  # Module not found  # Module not found  # Module not found
            for key, value in evidence_data.items():
                if isinstance(value, dict) and ("text" in value or "processed_text" in value):
                    evidence_items.append(self._standardize_evidence_item(value))
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict) and ("text" in item or "processed_text" in item):
                            evidence_items.append(self._standardize_evidence_item(item))
        
        return evidence_items

    def _standardize_evidence_item(self, evidence: Dict[str, Any]) -> Dict[str, Any]:
        """
        Standardize evidence item structure for consistent processing.
        
        Args:
            evidence: Raw evidence item
            
        Returns:
            Standardized evidence item
        """
        standardized = {
            "evidence_id": evidence.get("evidence_id", evidence.get("id", "")),
            "text": evidence.get("processed_text", evidence.get("text", "")),
            "has_valid_content": bool(evidence.get("processed_text", evidence.get("text", "")).strip()),
            "page_num": None,
            "exact_text": None,
            "source_metadata": {}
        }
        
# # #         # Extract page number from various possible locations  # Module not found  # Module not found  # Module not found
        if "source_metadata" in evidence:
            source_meta = evidence["source_metadata"]
            standardized["page_num"] = source_meta.get("page_number")
            standardized["source_metadata"] = source_meta
            
        elif "metadata" in evidence:
            meta = evidence["metadata"]
            standardized["page_num"] = meta.get("page_number", meta.get("page_num"))
            standardized["source_metadata"] = meta
            
        else:
            # Check direct fields
            standardized["page_num"] = evidence.get("page_number", evidence.get("page_num"))
        
        # Extract exact text reference
        if "evidence_chunk" in evidence:
            chunk = evidence["evidence_chunk"]
            if isinstance(chunk, dict):
                standardized["exact_text"] = chunk.get("text", chunk.get("raw_text"))
        
        standardized["exact_text"] = standardized["exact_text"] or evidence.get("original_text", evidence.get("exact_text"))
        
        return standardized

    def _calculate_quality_metrics(
        self, 
        evidence_items: List[Dict[str, Any]], 
        total_questions: int
    ) -> EvidenceQualityMetrics:
        """
        Calculate aggregated evidence quality metrics.
        
        Args:
            evidence_items: List of standardized evidence items
            total_questions: Total number of questions being evaluated
            
        Returns:
            Calculated quality metrics
        """
        # Count questions with valid evidence
        questions_with_valid_evidence = sum(
            1 for item in evidence_items if item["has_valid_content"]
        )
        
        # Count evidence items with complete references (both page_num and exact_text)
        complete_reference_count = sum(
            1 for item in evidence_items 
            if (item["page_num"] is not None and 
                item["exact_text"] is not None and 
                str(item["exact_text"]).strip())
        )
        
        total_evidence_items = len(evidence_items)
        
        # Calculate evidence_completeness: ratio of questions with valid evidence to total questions
        if total_questions > 0:
            evidence_completeness = questions_with_valid_evidence / total_questions
        else:
            evidence_completeness = 0.0
        
        # Calculate page_reference_quality: fraction of evidence items with complete references
        if total_evidence_items > 0:
            page_reference_quality = complete_reference_count / total_evidence_items
        else:
            page_reference_quality = 0.0
        
        # Round to specified precision
        evidence_completeness = self._round_to_precision(evidence_completeness)
        page_reference_quality = self._round_to_precision(page_reference_quality)
        
        return EvidenceQualityMetrics(
            total_questions=total_questions,
            questions_with_valid_evidence=questions_with_valid_evidence,
            total_evidence_items=total_evidence_items,
            evidence_items_with_complete_references=complete_reference_count,
            evidence_completeness=evidence_completeness,
            page_reference_quality=page_reference_quality
        )

    def _extract_question_evidence_metadata(
        self, 
        question_id: str, 
        evidence_items: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Extract relevant evidence metadata for a specific question.
        
        Args:
            question_id: Question identifier
            evidence_items: List of evidence items
            
        Returns:
            Metadata dictionary for the question
        """
        # Filter evidence items relevant to this question
        # (In a more sophisticated system, this would use question-evidence mapping)
        relevant_items = [
            item for item in evidence_items 
            if question_id in item.get("evidence_id", "") or 
               question_id in str(item.get("source_metadata", {}))
        ]
        
        if not relevant_items:
            # Use first available evidence item as fallback
            relevant_items = evidence_items[:1]
        
        metadata = OrderedDict()
        if relevant_items:
            item = relevant_items[0]
            metadata.update({
                "evidence_id": item["evidence_id"],
                "has_page_reference": item["page_num"] is not None,
                "has_exact_text": item["exact_text"] is not None,
                "source_metadata": item["source_metadata"]
            })
        
        return metadata

    def _create_question_eval_input(
        self,
        question_id: str,
        response: str,
        evidence_completeness: float,
        page_reference_quality: float,
        evidence_metadata: Dict[str, Any]
    ) -> QuestionEvalInput:
        """
        Create a QuestionEvalInput object with all required fields.
        
        Args:
            question_id: Question identifier
            response: Question response
            evidence_completeness: Calculated evidence completeness score
            page_reference_quality: Calculated page reference quality score
            evidence_metadata: Associated evidence metadata
            
        Returns:
            QuestionEvalInput object
        """
# # #         from datetime import datetime  # Module not found  # Module not found  # Module not found
        
        return QuestionEvalInput(
            question_id=question_id,
            response=response,
            evidence_completeness=evidence_completeness,
            page_reference_quality=page_reference_quality,
            evidence_metadata=evidence_metadata,
            processing_timestamp=datetime.now().isoformat()
        )

    def _update_processing_stats(self, evidence_items: List[Dict[str, Any]]) -> None:
        """Update internal processing statistics."""
        self.processing_stats["total_adaptations"] += 1
        self.processing_stats["valid_evidence_count"] += sum(
            1 for item in evidence_items if item["has_valid_content"]
        )
        self.processing_stats["invalid_evidence_count"] += sum(
            1 for item in evidence_items if not item["has_valid_content"]
        )
        self.processing_stats["complete_reference_count"] += sum(
            1 for item in evidence_items 
            if (item["page_num"] is not None and 
                item["exact_text"] is not None and 
                str(item["exact_text"]).strip())
        )

    def _round_to_precision(self, value: float) -> float:
        """Round value to specified precision."""
        if self.precision == 0:
            return round(value)
        
        factor = 10 ** self.precision
        return round(value * factor) / factor

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return OrderedDict([
            ("total_adaptations", self.processing_stats["total_adaptations"]),
            ("valid_evidence_count", self.processing_stats["valid_evidence_count"]),
            ("invalid_evidence_count", self.processing_stats["invalid_evidence_count"]),
            ("complete_reference_count", self.processing_stats["complete_reference_count"])
        ])

    def serialize_eval_inputs(self, eval_inputs: List[QuestionEvalInput]) -> str:
        """
        Serialize evaluation inputs to deterministic JSON.
        
        Args:
            eval_inputs: List of QuestionEvalInput objects
            
        Returns:
            Deterministic JSON string
        """
        # Convert to dictionaries with deterministic ordering
        data = [eval_input.to_dict() for eval_input in eval_inputs]
        
        # Serialize with consistent formatting
        return json.dumps(data, sort_keys=True, separators=(',', ':'), indent=None)


def create_evidence_adapter(precision: int = 4) -> EvidenceAdapter:
    """
    Factory function to create evidence adapter with specified precision.
    
    Args:
        precision: Number of decimal places for calculations
        
    Returns:
        Configured EvidenceAdapter instance
    """
    return EvidenceAdapter(precision=precision)


# Example usage and testing functions
def demo_evidence_adaptation():
    """Demonstrate evidence adaptation with sample data."""
    
    # Sample evidence processor output
    sample_evidence_data = {
        "processed_evidences": [
            {
                "evidence_id": "ev_001",
                "processed_text": "El programa incluye medidas específicas de seguimiento",
                "source_metadata": {
                    "page_number": 15,
                    "document_id": "doc_001"
                },
                "evidence_chunk": {
                    "text": "El programa incluye medidas específicas de seguimiento",
                    "raw_text": "El programa incluye medidas específicas de seguimiento"
                }
            },
            {
                "evidence_id": "ev_002", 
                "processed_text": "Se establecen indicadores de evaluación",
                "source_metadata": {
                    "document_id": "doc_001"
                    # Missing page_number
                },
                "evidence_chunk": {
                    "text": "Se establecen indicadores de evaluación"
                }
            },
            {
                "evidence_id": "ev_003",
                "processed_text": "",  # Invalid/empty evidence
                "source_metadata": {
                    "page_number": 20,
                    "document_id": "doc_002"
                }
            }
        ]
    }
    
    # Sample question responses
    sample_responses = {
        "DE1_Q1": "Sí",
        "DE1_Q2": "Parcial", 
        "DE2_Q1": "No",
        "DE2_Q2": "NI"
    }
    
    # Create adapter and transform
    adapter = create_evidence_adapter(precision=4)
    eval_inputs, quality_metrics = adapter.transform_evidence_to_eval_inputs(
        sample_evidence_data, sample_responses
    )
    
    print("Evidence Quality Metrics:")
    print(json.dumps(quality_metrics.to_dict(), indent=2))
    print("\nEvaluation Inputs:")
    print(adapter.serialize_eval_inputs(eval_inputs))
    print("\nProcessing Stats:")
    print(json.dumps(adapter.get_processing_stats(), indent=2))


if __name__ == "__main__":
    demo_evidence_adaptation()
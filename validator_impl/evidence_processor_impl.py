"""
Evidence Processor Implementation

Concrete implementation of EvidenceProcessorPort that processes evidence 
into structured format. Only depends on validator_api interfaces.
"""

import hashlib
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from enum import Enum
from dataclasses import dataclass, asdict

# Import only from validator_api - no pipeline dependencies
from validator_api.validation_interfaces import EvidenceProcessorPort, ValidationResult, ValidationStatus
from validator_api.dtos import (
    EvidenceProcessingRequest, 
    EvidenceProcessingResponse,
    EvidenceItem,
    EvidenceMetadata
)


class EvidenceType(Enum):
    """Types of evidence that can be processed."""
    DIRECT_QUOTE = "direct_quote"
    PARAPHRASE = "paraphrase"
    STATISTICAL = "statistical"
    EXPERT_OPINION = "expert_opinion"
    CASE_STUDY = "case_study"


class ConfidenceLevel(Enum):
    """Confidence levels for evidence scoring."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class ProcessingStatistics:
    """Statistics for evidence processing operations."""
    total_processed: int = 0
    successful_items: int = 0
    failed_items: int = 0
    high_confidence_items: int = 0
    medium_confidence_items: int = 0
    low_confidence_items: int = 0
    average_quality_score: float = 0.0
    processing_time_ms: float = 0.0


class EvidenceProcessorImpl(EvidenceProcessorPort):
    """Processes raw evidence into structured evidence items."""
    
    def __init__(self):
        # Processing configuration
        self.min_confidence_threshold = 0.5
        self.max_evidences_per_batch = 100
        self.context_window_size = 200
        
        # Quality scoring weights
        self.quality_weights = {
            "length_factor": 0.3,
            "type_factor": 0.2,
            "confidence_factor": 0.3,
            "metadata_factor": 0.2
        }
        
        # Processing statistics
        self.processing_stats = ProcessingStatistics()
        
        # Evidence ID counter for deterministic IDs
        self._evidence_counter = 0
    
    def process_evidence(self, request: EvidenceProcessingRequest) -> EvidenceProcessingResponse:
        """
        Process raw evidence into structured format.
        
        Args:
            request: Evidence processing request
            
        Returns:
            EvidenceProcessingResponse with processed evidence
        """
        start_time = datetime.now()
        
        try:
            # Extract raw evidence candidates
            raw_candidates = self._extract_evidence_candidates(request.raw_evidence)
            
            # Process each candidate
            processed_evidence = []
            errors = []
            warnings = []
            
            for candidate in raw_candidates[:self.max_evidences_per_batch]:
                try:
                    evidence_item = self._process_single_candidate(candidate, request)
                    
                    # Filter by quality threshold
                    if evidence_item.quality_score >= self.min_confidence_threshold:
                        processed_evidence.append(evidence_item)
                    else:
                        warnings.append(f"Evidence {evidence_item.evidence_id} below quality threshold: {evidence_item.quality_score}")
                
                except Exception as e:
                    errors.append(f"Failed to process evidence candidate: {str(e)}")
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Update statistics
            self._update_processing_stats(processed_evidence, processing_time)
            
            # Generate processing metadata
            processing_metadata = {
                "raw_candidates_count": len(raw_candidates),
                "processed_count": len(processed_evidence),
                "filtered_count": len(raw_candidates) - len(processed_evidence),
                "processing_type": request.processing_type,
                "quality_threshold": self.min_confidence_threshold,
                "processing_stats": asdict(self.processing_stats)
            }
            
            return EvidenceProcessingResponse(
                request_id=request.request_id,
                success=len(processed_evidence) > 0,
                processed_evidence=processed_evidence,
                processing_metadata=processing_metadata,
                errors=errors,
                warnings=warnings,
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            return EvidenceProcessingResponse(
                request_id=request.request_id,
                success=False,
                processed_evidence=[],
                processing_metadata={"error": str(e)},
                errors=[str(e)],
                warnings=[],
                processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000
            )
    
    def validate_evidence_structure(self, evidence: Dict[str, Any]) -> ValidationResult:
        """
        Validate the structure of evidence data.
        
        Args:
            evidence: Evidence data to validate
            
        Returns:
            ValidationResult indicating if structure is valid
        """
        errors = []
        warnings = []
        
        # Required fields
        required_fields = ["evidence_id", "text", "metadata"]
        for field in required_fields:
            if field not in evidence:
                errors.append(f"Missing required field: {field}")
        
        # Validate text content
        if "text" in evidence:
            if not isinstance(evidence["text"], str) or len(evidence["text"].strip()) == 0:
                errors.append("Text field must be a non-empty string")
            elif len(evidence["text"]) < 10:
                warnings.append("Text content is very short (< 10 characters)")
        
        # Validate metadata structure
        if "metadata" in evidence:
            if not isinstance(evidence["metadata"], dict):
                errors.append("Metadata must be a dictionary")
            else:
                metadata = evidence["metadata"]
                if "document_id" not in metadata:
                    warnings.append("Metadata missing document_id")
                if "title" not in metadata:
                    warnings.append("Metadata missing title")
        
        # Validate optional fields
        if "quality_score" in evidence:
            score = evidence["quality_score"]
            if not isinstance(score, (int, float)) or not (0.0 <= score <= 1.0):
                errors.append("Quality score must be a number between 0.0 and 1.0")
        
        # Determine status
        if errors:
            status = ValidationStatus.FAILED
            message = f"Evidence structure validation failed with {len(errors)} errors"
        elif warnings:
            status = ValidationStatus.WARNING
            message = f"Evidence structure validation passed with {len(warnings)} warnings"
        else:
            status = ValidationStatus.PASSED
            message = "Evidence structure validation passed"
        
        return ValidationResult(
            status=status,
            message=message,
            details={"validated_fields": list(evidence.keys())},
            errors=errors,
            warnings=warnings,
            confidence_score=1.0 if not errors else 0.0
        )
    
    def extract_metadata(self, raw_data: Any) -> Dict[str, Any]:
        """
        Extract metadata from raw evidence data.
        
        Args:
            raw_data: Raw evidence data
            
        Returns:
            Dictionary containing extracted metadata
        """
        metadata = {
            "extraction_timestamp": datetime.now().isoformat(),
            "data_type": type(raw_data).__name__,
            "data_hash": self._calculate_data_hash(raw_data)
        }
        
        if isinstance(raw_data, dict):
            # Extract from dictionary structure
            metadata.update({
                "document_id": raw_data.get("document_id", "unknown"),
                "title": raw_data.get("title", "Untitled"),
                "author": raw_data.get("author", ""),
                "publication_date": raw_data.get("publication_date"),
                "page_number": raw_data.get("page_number"),
                "section_header": raw_data.get("section_header"),
                "document_type": raw_data.get("document_type", "document"),
                "url": raw_data.get("url"),
                "confidence_level": raw_data.get("confidence_level"),
                "content_length": len(str(raw_data.get("text", "")))
            })
        elif isinstance(raw_data, str):
            # Extract from string data
            metadata.update({
                "document_id": f"text_{self._calculate_data_hash(raw_data)[:8]}",
                "title": "Text Document",
                "author": "",
                "document_type": "text",
                "content_length": len(raw_data)
            })
        elif isinstance(raw_data, list):
            # Extract from list data
            metadata.update({
                "document_id": f"list_{len(raw_data)}_items",
                "title": f"List of {len(raw_data)} items",
                "document_type": "list",
                "content_length": sum(len(str(item)) for item in raw_data),
                "item_count": len(raw_data)
            })
        
        # Remove None values
        return {k: v for k, v in metadata.items() if v is not None}
    
    def _extract_evidence_candidates(self, raw_evidence: Any) -> List[Dict[str, Any]]:
        """Extract individual evidence candidates from raw data."""
        candidates = []
        
        if isinstance(raw_evidence, dict):
            # Single evidence item
            if "text" in raw_evidence:
                candidates.append(raw_evidence)
            # Multiple evidence items
            elif "evidences" in raw_evidence:
                candidates.extend(raw_evidence["evidences"])
            # Evidence list
            elif "evidence_list" in raw_evidence:
                candidates.extend(raw_evidence["evidence_list"])
            else:
                # Treat entire dict as evidence
                candidates.append(raw_evidence)
        
        elif isinstance(raw_evidence, list):
            # List of evidence items
            for item in raw_evidence:
                if isinstance(item, dict):
                    candidates.append(item)
                else:
                    # Convert non-dict items
                    candidates.append({"text": str(item), "metadata": {}})
        
        elif isinstance(raw_evidence, str):
            # Single text evidence
            candidates.append({"text": raw_evidence, "metadata": {}})
        
        else:
            # Convert anything else to string
            candidates.append({"text": str(raw_evidence), "metadata": {}})
        
        return candidates
    
    def _process_single_candidate(self, candidate: Dict[str, Any], request: EvidenceProcessingRequest) -> EvidenceItem:
        """Process a single evidence candidate into an EvidenceItem."""
        self._evidence_counter += 1
        
        # Generate deterministic evidence ID
        evidence_id = self._generate_evidence_id(candidate)
        
        # Extract text content
        text = candidate.get("text", "")
        if not isinstance(text, str):
            text = str(text)
        
        # Create metadata object
        metadata = self._create_evidence_metadata(candidate, evidence_id)
        
        # Determine evidence type
        evidence_type = self._classify_evidence_type(text)
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(text, evidence_type, metadata)
        
        # Extract context
        context_before = candidate.get("context_before", "")
        context_after = candidate.get("context_after", "")
        
        # Create processing notes
        processing_notes = []
        if quality_score < 0.7:
            processing_notes.append("Low quality score detected")
        if len(text) < 50:
            processing_notes.append("Short text content")
        
        return EvidenceItem(
            evidence_id=evidence_id,
            text=text,
            metadata=metadata,
            context_before=context_before,
            context_after=context_after,
            start_position=candidate.get("start_position", 0),
            end_position=candidate.get("end_position", len(text)),
            evidence_type=evidence_type.value,
            quality_score=quality_score,
            processing_notes=processing_notes
        )
    
    def _generate_evidence_id(self, candidate: Dict[str, Any]) -> str:
        """Generate a deterministic evidence ID."""
        # Use existing ID if available
        if "evidence_id" in candidate:
            return candidate["evidence_id"]
        
        # Generate from content hash
        content = json.dumps(candidate, sort_keys=True)
        hash_value = hashlib.sha256(content.encode()).hexdigest()[:12]
        return f"ev_{hash_value}"
    
    def _create_evidence_metadata(self, candidate: Dict[str, Any], evidence_id: str) -> EvidenceMetadata:
        """Create metadata object for evidence item."""
        raw_metadata = candidate.get("metadata", {})
        
        return EvidenceMetadata(
            document_id=raw_metadata.get("document_id", evidence_id),
            title=raw_metadata.get("title", "Untitled Document"),
            author=raw_metadata.get("author", ""),
            publication_date=raw_metadata.get("publication_date"),
            page_number=raw_metadata.get("page_number"),
            section_header=raw_metadata.get("section_header"),
            document_type=raw_metadata.get("document_type", "document"),
            url=raw_metadata.get("url"),
            confidence_level=raw_metadata.get("confidence_level"),
            source_hash=self._calculate_data_hash(candidate)
        )
    
    def _classify_evidence_type(self, text: str) -> EvidenceType:
        """Classify evidence type based on text content."""
        text_lower = text.lower()
        
        # Check for direct quotes
        if '"' in text or "'" in text or text.strip().startswith(('"', "'")):
            return EvidenceType.DIRECT_QUOTE
        
        # Check for statistical evidence
        statistical_indicators = ["datos", "estadística", "número", "porcentaje", "cifra", "métrica", "%"]
        if any(indicator in text_lower for indicator in statistical_indicators):
            return EvidenceType.STATISTICAL
        
        # Check for expert opinion
        expert_indicators = ["experto", "especialista", "opinión", "considera", "afirma", "según"]
        if any(indicator in text_lower for indicator in expert_indicators):
            return EvidenceType.EXPERT_OPINION
        
        # Check for case study
        case_indicators = ["caso", "estudio", "ejemplo", "experiencia", "situación"]
        if any(indicator in text_lower for indicator in case_indicators):
            return EvidenceType.CASE_STUDY
        
        # Default to paraphrase
        return EvidenceType.PARAPHRASE
    
    def _calculate_quality_score(self, text: str, evidence_type: EvidenceType, metadata: EvidenceMetadata) -> float:
        """Calculate quality score for evidence item."""
        score = 0.0
        
        # Length factor (30%)
        length_factor = min(1.0, len(text) / 200)  # Normalize to 200 chars
        score += length_factor * self.quality_weights["length_factor"]
        
        # Type factor (20%)
        type_scores = {
            EvidenceType.DIRECT_QUOTE: 0.9,
            EvidenceType.STATISTICAL: 0.8,
            EvidenceType.EXPERT_OPINION: 0.7,
            EvidenceType.CASE_STUDY: 0.6,
            EvidenceType.PARAPHRASE: 0.5
        }
        type_factor = type_scores.get(evidence_type, 0.5)
        score += type_factor * self.quality_weights["type_factor"]
        
        # Confidence factor (30%) - based on text quality indicators
        confidence_factor = self._calculate_text_confidence(text)
        score += confidence_factor * self.quality_weights["confidence_factor"]
        
        # Metadata factor (20%)
        metadata_factor = self._calculate_metadata_completeness(metadata)
        score += metadata_factor * self.quality_weights["metadata_factor"]
        
        return min(1.0, max(0.0, score))
    
    def _calculate_text_confidence(self, text: str) -> float:
        """Calculate confidence based on text quality indicators."""
        confidence = 0.5  # Base confidence
        
        # Length bonus
        if len(text) > 100:
            confidence += 0.2
        elif len(text) < 20:
            confidence -= 0.3
        
        # Quality indicators
        quality_indicators = ["específico", "detallado", "preciso", "verificado", "documentado"]
        quality_count = sum(1 for indicator in quality_indicators if indicator in text.lower())
        confidence += quality_count * 0.1
        
        # Penalty for very short or very long text
        if len(text) < 10:
            confidence -= 0.4
        elif len(text) > 1000:
            confidence += 0.1
        
        return min(1.0, max(0.0, confidence))
    
    def _calculate_metadata_completeness(self, metadata: EvidenceMetadata) -> float:
        """Calculate completeness score for metadata."""
        completeness = 0.0
        total_fields = 0
        filled_fields = 0
        
        # Core fields
        core_fields = [
            ("document_id", metadata.document_id),
            ("title", metadata.title),
            ("author", metadata.author),
            ("document_type", metadata.document_type)
        ]
        
        for field_name, field_value in core_fields:
            total_fields += 1
            if field_value and field_value.strip():
                filled_fields += 1
        
        # Optional fields  
        optional_fields = [
            ("publication_date", metadata.publication_date),
            ("page_number", metadata.page_number),
            ("section_header", metadata.section_header),
            ("url", metadata.url)
        ]
        
        for field_name, field_value in optional_fields:
            total_fields += 1
            if field_value:
                filled_fields += 0.5  # Lower weight for optional fields
        
        completeness = filled_fields / total_fields if total_fields > 0 else 0.0
        return min(1.0, max(0.0, completeness))
    
    def _calculate_data_hash(self, data: Any) -> str:
        """Calculate hash for data."""
        if isinstance(data, str):
            content = data
        else:
            content = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _update_processing_stats(self, processed_evidence: List[EvidenceItem], processing_time: float):
        """Update processing statistics."""
        self.processing_stats.total_processed += len(processed_evidence)
        self.processing_stats.successful_items += len(processed_evidence)
        self.processing_stats.processing_time_ms = processing_time
        
        # Calculate confidence distribution
        high_confidence = sum(1 for ev in processed_evidence if ev.quality_score >= 0.8)
        medium_confidence = sum(1 for ev in processed_evidence if 0.5 <= ev.quality_score < 0.8)
        low_confidence = sum(1 for ev in processed_evidence if ev.quality_score < 0.5)
        
        self.processing_stats.high_confidence_items += high_confidence
        self.processing_stats.medium_confidence_items += medium_confidence
        self.processing_stats.low_confidence_items += low_confidence
        
        # Calculate average quality score
        if processed_evidence:
            avg_score = sum(ev.quality_score for ev in processed_evidence) / len(processed_evidence)
            self.processing_stats.average_quality_score = avg_score
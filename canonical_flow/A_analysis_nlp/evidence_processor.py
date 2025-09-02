"""
Canonical Flow Alias: 16A
Evidence Processor with Total Ordering and Deterministic Processing

Source: evidence_processor.py
Stage: analysis_nlp
Code: 16A
"""

import json
import logging
import sys
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Tuple
from collections import OrderedDict
from uuid import uuid4

# Import total ordering base
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from total_ordering_base import TotalOrderingBase, DeterministicCollectionMixin

logger = logging.getLogger(__name__)

# Optional validation model imports with fallbacks
try:
    from evidence_validation_model import (
        EvidenceValidationModel,
        ValidationSeverity,
        DNPEvidenceValidator,
        EvidenceValidationRequest,
        EvidenceValidationResponse
    )
except ImportError:
    EvidenceValidationModel = None
    ValidationSeverity = None
    DNPEvidenceValidator = None
    EvidenceValidationRequest = None
    EvidenceValidationResponse = None


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
class SourceMetadata:
    """Metadata for source documents."""
    document_id: str
    title: str
    author: str = ""
    publication_date: Optional[str] = None
    page_number: Optional[int] = None
    section_header: Optional[str] = None
    subsection_header: Optional[str] = None
    document_type: str = "document"
    url: Optional[str] = None
    doi: Optional[str] = None
    isbn: Optional[str] = None
    
    def __post_init__(self):
        # Ensure deterministic ordering and consistent types
        if self.publication_date and isinstance(self.publication_date, datetime):
            self.publication_date = self.publication_date.isoformat()


@dataclass
class EvidenceChunk:
    """Processed evidence chunk with context."""
    chunk_id: str
    text: str
    context_before: str = ""
    context_after: str = ""
    start_position: int = 0
    end_position: int = 0
    processing_timestamp: str = ""
    raw_text: str = ""
    
    def __post_init__(self):
        # Ensure consistent timestamp format
        if self.processing_timestamp and isinstance(self.processing_timestamp, datetime):
            self.processing_timestamp = self.processing_timestamp.isoformat()


@dataclass
class Citation:
    """Standardized citation information."""
    citation_id: str
    formatted_reference: str
    short_reference: str
    inline_citation: str
    metadata: SourceMetadata
    
    def to_apa_format(self) -> str:
        """Generate APA format citation."""
        author = self.metadata.author if self.metadata.author else "Unknown Author"
        title = self.metadata.title
        year = "n.d."
        
        if self.metadata.publication_date:
            try:
                if isinstance(self.metadata.publication_date, str):
                    # Try to parse ISO format
                    if "T" in self.metadata.publication_date:
                        year = self.metadata.publication_date.split("T")[0].split("-")[0]
                    else:
                        year = self.metadata.publication_date.split("-")[0]
                else:
                    year = str(self.metadata.publication_date)[:4]
            except:
                year = "n.d."

        if self.metadata.url:
            return f"{author} ({year}). {title}. Retrieved from {self.metadata.url}"
        else:
            return f"{author} ({year}). {title}."


@dataclass
class ProcessedEvidence:
    """Fully processed evidence with all metadata."""
    evidence_id: str
    original_text: str
    processed_text: str
    evidence_type: EvidenceType
    confidence_level: ConfidenceLevel
    confidence_score: float
    source_metadata: SourceMetadata
    evidence_chunk: EvidenceChunk
    citation: Citation
    keywords: List[str] = field(default_factory=list)
    quality_score: float = 0.0
    validation_status: str = "pending"
    processing_notes: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        # Ensure deterministic ordering
        self.keywords = sorted(self.keywords)
        self.processing_notes = sorted(self.processing_notes)


class EvidenceProcessor(TotalOrderingBase, DeterministicCollectionMixin):
    """
    Evidence Processor with deterministic processing and total ordering.
    
    Provides consistent evidence processing results and stable ID generation across runs.
    """
    
    def __init__(self):
        super().__init__("EvidenceProcessor")
        
        # Configuration with deterministic defaults
        self.min_confidence_threshold = 0.5
        self.max_evidences_per_batch = 100
        self.context_window_size = 200
        
        # Processing statistics
        self.processing_stats = {
            "evidences_processed": 0,
            "high_confidence_count": 0,
            "medium_confidence_count": 0,
            "low_confidence_count": 0,
            "validation_errors": 0,
        }
        
        # Initialize validation model if available
        self.validation_model = None
        if EvidenceValidationModel:
            try:
                self.validation_model = EvidenceValidationModel()
            except:
                pass
        
        # Update state hash
        self.update_state_hash(self._get_initial_state())
    
    def _get_comparison_key(self) -> Tuple[str, ...]:
        """Return comparison key for deterministic ordering"""
        return (
            self.component_id,
            str(self.min_confidence_threshold),
            str(self.max_evidences_per_batch),
            str(self.context_window_size),
            str(self.validation_model is not None),
            str(self._state_hash or "")
        )
    
    def _get_initial_state(self) -> Dict[str, Any]:
        """Get initial state for hash calculation"""
        return {
            "context_window_size": self.context_window_size,
            "has_validation_model": self.validation_model is not None,
            "max_evidences_per_batch": self.max_evidences_per_batch,
            "min_confidence_threshold": self.min_confidence_threshold,
        }
    
    def process(self, data: Any = None, context: Any = None) -> Dict[str, Any]:
        """
        Main processing function with deterministic output.
        
        Args:
            data: Input data containing evidences to process
            context: Processing context
            
        Returns:
            Deterministic evidence processing results
        """
        operation_id = self.generate_operation_id("process", {"data": data, "context": context})
        
        try:
            # Canonicalize inputs
            canonical_data = self.canonicalize_data(data) if data else {}
            canonical_context = self.canonicalize_data(context) if context else {}
            
            # Extract evidences to process
            evidences_to_process = self._extract_evidences_deterministic(canonical_data)
            
            # Process each evidence
            processed_evidences = []
            for evidence_data in evidences_to_process:
                processed_evidence = self._process_single_evidence_deterministic(evidence_data)
                processed_evidences.append(processed_evidence)
            
            # Filter and rank processed evidences
            filtered_evidences = self._filter_and_rank_evidences_deterministic(processed_evidences)
            
            # Generate deterministic output
            output = self._generate_deterministic_output(filtered_evidences, operation_id)
            
            # Update statistics
            self._update_processing_stats(filtered_evidences)
            
            # Update state hash
            self.update_state_hash(output)
            
            return output
            
        except Exception as e:
            error_output = {
                "component": self.component_name,
                "error": str(e),
                "operation_id": operation_id,
                "status": "error",
                "timestamp": self._get_deterministic_timestamp(),
            }
            return self.sort_dict_by_keys(error_output)
    
    def _extract_evidences_deterministic(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract evidences from input data with stable ordering"""
        evidences = []
        
        # Handle single evidence
        if "evidence" in data:
            evidence_data = data["evidence"]
            if isinstance(evidence_data, str):
                evidences.append({
                    "id": self.generate_stable_id(evidence_data, prefix="ev"),
                    "text": evidence_data,
                    "metadata": {}
                })
            elif isinstance(evidence_data, dict):
                evidences.append({
                    "id": evidence_data.get("id", self.generate_stable_id(str(evidence_data), prefix="ev")),
                    "text": evidence_data.get("text", ""),
                    "metadata": evidence_data.get("metadata", {})
                })
        
        # Handle multiple evidences
        if "evidences" in data and isinstance(data["evidences"], list):
            for i, ev in enumerate(data["evidences"]):
                if isinstance(ev, str):
                    evidences.append({
                        "id": self.generate_stable_id(ev, prefix="ev"),
                        "text": ev,
                        "metadata": {}
                    })
                elif isinstance(ev, dict):
                    evidences.append({
                        "id": ev.get("id", self.generate_stable_id(str(ev), prefix="ev")),
                        "text": ev.get("text", ""),
                        "metadata": ev.get("metadata", {})
                    })
        
        # Handle text directly (convert to evidence)
        if "text" in data and not evidences:
            evidences.append({
                "id": self.generate_stable_id(data["text"], prefix="ev"),
                "text": str(data["text"]),
                "metadata": {}
            })
        
        # If no evidences found, create empty evidence
        if not evidences:
            evidences.append({
                "id": "default_empty",
                "text": "",
                "metadata": {}
            })
        
        # Limit batch size
        if len(evidences) > self.max_evidences_per_batch:
            evidences = evidences[:self.max_evidences_per_batch]
        
        return evidences
    
    def _process_single_evidence_deterministic(self, evidence_data: Dict[str, Any]) -> ProcessedEvidence:
        """Process a single evidence with deterministic processing"""
        evidence_id = evidence_data["id"]
        evidence_text = evidence_data["text"]
        evidence_metadata = evidence_data.get("metadata", {})
        
        # Determine evidence type
        evidence_type = self._classify_evidence_type_deterministic(evidence_text)
        
        # Calculate confidence
        confidence_level, confidence_score = self._calculate_confidence_deterministic(evidence_text, evidence_type)
        
        # Create source metadata
        source_metadata = self._create_source_metadata_deterministic(evidence_metadata, evidence_id)
        
        # Create evidence chunk
        evidence_chunk = self._create_evidence_chunk_deterministic(evidence_text, evidence_id)
        
        # Create citation
        citation = self._create_citation_deterministic(source_metadata, evidence_id)
        
        # Extract keywords
        keywords = self._extract_keywords_deterministic(evidence_text)
        
        # Calculate quality score
        quality_score = self._calculate_quality_score_deterministic(evidence_text, confidence_score)
        
        # Validate evidence if model available
        validation_status = "validated"
        if self.validation_model:
            try:
                validation_result = self._validate_evidence(evidence_text)
                validation_status = validation_result.get("status", "validated")
            except:
                validation_status = "validation_failed"
        
        # Create processed evidence
        processed_evidence = ProcessedEvidence(
            evidence_id=evidence_id,
            original_text=evidence_text,
            processed_text=evidence_text.strip(),
            evidence_type=evidence_type,
            confidence_level=confidence_level,
            confidence_score=confidence_score,
            source_metadata=source_metadata,
            evidence_chunk=evidence_chunk,
            citation=citation,
            keywords=keywords,
            quality_score=quality_score,
            validation_status=validation_status,
            processing_notes=[],
        )
        
        return processed_evidence
    
    def _classify_evidence_type_deterministic(self, text: str) -> EvidenceType:
        """Classify evidence type with deterministic logic"""
        text_lower = text.lower()
        
        # Check for direct quotes
        if ('"' in text or "'" in text or 
            text.strip().startswith('"') or text.strip().startswith("'")):
            return EvidenceType.DIRECT_QUOTE
        
        # Check for statistical evidence
        statistical_indicators = ["datos", "estadística", "número", "porcentaje", "cifra", "métrica"]
        if any(indicator in text_lower for indicator in sorted(statistical_indicators)):
            return EvidenceType.STATISTICAL
        
        # Check for expert opinion
        expert_indicators = ["experto", "especialista", "opinión", "considera", "afirma"]
        if any(indicator in text_lower for indicator in sorted(expert_indicators)):
            return EvidenceType.EXPERT_OPINION
        
        # Check for case study
        case_indicators = ["caso", "estudio", "ejemplo", "experiencia", "situación"]
        if any(indicator in text_lower for indicator in sorted(case_indicators)):
            return EvidenceType.CASE_STUDY
        
        # Default to paraphrase
        return EvidenceType.PARAPHRASE
    
    def _calculate_confidence_deterministic(self, text: str, evidence_type: EvidenceType) -> tuple[ConfidenceLevel, float]:
        """Calculate confidence with deterministic logic"""
        
        # Base confidence score
        base_score = 0.5
        
        # Adjust based on text length
        text_length = len(text.strip())
        if text_length > 200:
            base_score += 0.2
        elif text_length > 100:
            base_score += 0.1
        elif text_length < 20:
            base_score -= 0.3
        
        # Adjust based on evidence type
        type_adjustments = {
            EvidenceType.DIRECT_QUOTE: 0.2,
            EvidenceType.STATISTICAL: 0.3,
            EvidenceType.EXPERT_OPINION: 0.1,
            EvidenceType.CASE_STUDY: 0.1,
            EvidenceType.PARAPHRASE: 0.0,
        }
        base_score += type_adjustments.get(evidence_type, 0.0)
        
        # Check for quality indicators
        quality_indicators = ["específico", "detallado", "preciso", "verificado", "documentado"]
        quality_count = sum(1 for indicator in quality_indicators if indicator in text.lower())
        base_score += quality_count * 0.05
        
        # Clamp score
        confidence_score = max(0.0, min(1.0, base_score))
        
        # Determine confidence level
        if confidence_score >= 0.8:
            confidence_level = ConfidenceLevel.HIGH
        elif confidence_score >= 0.6:
            confidence_level = ConfidenceLevel.MEDIUM
        else:
            confidence_level = ConfidenceLevel.LOW
        
        return confidence_level, confidence_score
    
    def _create_source_metadata_deterministic(self, metadata: Dict[str, Any], evidence_id: str) -> SourceMetadata:
        """Create source metadata with deterministic values"""
        return SourceMetadata(
            document_id=metadata.get("document_id", evidence_id),
            title=metadata.get("title", "Untitled Document"),
            author=metadata.get("author", ""),
            publication_date=metadata.get("publication_date"),
            page_number=metadata.get("page_number"),
            section_header=metadata.get("section_header", ""),
            subsection_header=metadata.get("subsection_header", ""),
            document_type=metadata.get("document_type", "document"),
            url=metadata.get("url"),
            doi=metadata.get("doi"),
            isbn=metadata.get("isbn"),
        )
    
    def _create_evidence_chunk_deterministic(self, text: str, evidence_id: str) -> EvidenceChunk:
        """Create evidence chunk with deterministic values"""
        return EvidenceChunk(
            chunk_id=self.generate_stable_id({"text": text, "evidence_id": evidence_id}, prefix="chunk"),
            text=text.strip(),
            context_before="",
            context_after="",
            start_position=0,
            end_position=len(text),
            processing_timestamp=self._get_deterministic_timestamp(),
            raw_text=text,
        )
    
    def _create_citation_deterministic(self, source_metadata: SourceMetadata, evidence_id: str) -> Citation:
        """Create citation with deterministic values"""
        citation_id = self.generate_stable_id(
            {"document_id": source_metadata.document_id, "evidence_id": evidence_id},
            prefix="cite"
        )
        
        # Generate formatted reference
        author = source_metadata.author or "Unknown Author"
        title = source_metadata.title
        year = "n.d."
        
        if source_metadata.publication_date:
            try:
                year = source_metadata.publication_date.split("-")[0] if "-" in source_metadata.publication_date else str(source_metadata.publication_date)[:4]
            except:
                year = "n.d."
        
        formatted_reference = f"{author} ({year}). {title}."
        short_reference = f"{author}, {year}"
        inline_citation = f"({author}, {year})"
        
        return Citation(
            citation_id=citation_id,
            formatted_reference=formatted_reference,
            short_reference=short_reference,
            inline_citation=inline_citation,
            metadata=source_metadata,
        )
    
    def _extract_keywords_deterministic(self, text: str) -> List[str]:
        """Extract keywords deterministically"""
        import re
        
        # Simple keyword extraction
        words = re.findall(r'\b[a-zA-ZáéíóúñÁÉÍÓÚÑ]+\b', text.lower())
        
        # Filter common words
        stop_words = {"y", "o", "pero", "de", "la", "el", "en", "un", "una", "que", "se", "es", "no", "a", "su", "por", "con", "para"}
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Get top keywords by frequency
        word_freq = {}
        for word in keywords:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency, then alphabetically for deterministic ordering
        top_keywords = sorted(word_freq.items(), key=lambda x: (-x[1], x[0]))[:10]
        
        return [word for word, freq in top_keywords]
    
    def _calculate_quality_score_deterministic(self, text: str, confidence_score: float) -> float:
        """Calculate quality score deterministically"""
        # Start with confidence score as base
        quality_score = confidence_score
        
        # Adjust for text characteristics
        text_length = len(text.strip())
        if 50 <= text_length <= 500:  # Optimal length range
            quality_score += 0.1
        elif text_length > 500:
            quality_score -= 0.1
        
        # Check for quality markers
        quality_markers = ["específicamente", "según", "de acuerdo", "evidencia", "datos", "investigación"]
        marker_count = sum(1 for marker in quality_markers if marker in text.lower())
        quality_score += marker_count * 0.05
        
        # Clamp score
        return max(0.0, min(1.0, quality_score))
    
    def _validate_evidence(self, text: str) -> Dict[str, Any]:
        """Validate evidence using validation model if available"""
        if not self.validation_model or not EvidenceValidationRequest:
            return {"status": "skipped"}
        
        try:
            request = EvidenceValidationRequest(
                evidence_text=text,
                context="",
                validation_type="basic"
            )
            response = self.validation_model.validate_evidence(request)
            return {
                "status": "validated",
                "validation_score": response.validation_score if response else 0.5
            }
        except:
            return {"status": "validation_failed"}
    
    def _filter_and_rank_evidences_deterministic(self, evidences: List[ProcessedEvidence]) -> List[ProcessedEvidence]:
        """Filter and rank evidences with deterministic ordering"""
        
        # Filter by minimum confidence threshold
        filtered_evidences = [
            ev for ev in evidences 
            if ev.confidence_score >= self.min_confidence_threshold
        ]
        
        # Sort by multiple criteria for stable ordering
        sorted_evidences = sorted(
            filtered_evidences,
            key=lambda e: (
                -e.confidence_score,  # Higher confidence first
                -e.quality_score,     # Higher quality first
                e.evidence_id         # Stable secondary sort
            )
        )
        
        return sorted_evidences
    
    def _update_processing_stats(self, evidences: List[ProcessedEvidence]):
        """Update processing statistics"""
        self.processing_stats["evidences_processed"] += len(evidences)
        
        for evidence in evidences:
            if evidence.confidence_level == ConfidenceLevel.HIGH:
                self.processing_stats["high_confidence_count"] += 1
            elif evidence.confidence_level == ConfidenceLevel.MEDIUM:
                self.processing_stats["medium_confidence_count"] += 1
            elif evidence.confidence_level == ConfidenceLevel.LOW:
                self.processing_stats["low_confidence_count"] += 1
    
    def _generate_deterministic_output(self, evidences: List[ProcessedEvidence], operation_id: str) -> Dict[str, Any]:
        """Generate deterministic output structure"""
        
        # Convert evidences to dictionaries for JSON serialization
        evidence_dicts = []
        for evidence in evidences:
            evidence_dict = {
                "citation": {
                    "citation_id": evidence.citation.citation_id,
                    "formatted_reference": evidence.citation.formatted_reference,
                    "inline_citation": evidence.citation.inline_citation,
                    "metadata": {
                        "author": evidence.citation.metadata.author,
                        "document_id": evidence.citation.metadata.document_id,
                        "document_type": evidence.citation.metadata.document_type,
                        "page_number": evidence.citation.metadata.page_number,
                        "publication_date": evidence.citation.metadata.publication_date,
                        "title": evidence.citation.metadata.title,
                        "url": evidence.citation.metadata.url,
                    },
                    "short_reference": evidence.citation.short_reference,
                },
                "confidence_level": evidence.confidence_level.value,
                "confidence_score": evidence.confidence_score,
                "evidence_chunk": {
                    "chunk_id": evidence.evidence_chunk.chunk_id,
                    "context_after": evidence.evidence_chunk.context_after,
                    "context_before": evidence.evidence_chunk.context_before,
                    "end_position": evidence.evidence_chunk.end_position,
                    "processing_timestamp": evidence.evidence_chunk.processing_timestamp,
                    "raw_text": evidence.evidence_chunk.raw_text,
                    "start_position": evidence.evidence_chunk.start_position,
                    "text": evidence.evidence_chunk.text,
                },
                "evidence_id": evidence.evidence_id,
                "evidence_type": evidence.evidence_type.value,
                "keywords": evidence.keywords,
                "original_text": evidence.original_text,
                "processed_text": evidence.processed_text,
                "processing_notes": evidence.processing_notes,
                "quality_score": evidence.quality_score,
                "validation_status": evidence.validation_status,
            }
            evidence_dicts.append(evidence_dict)
        
        # Generate summary statistics
        summary = {
            "confidence_level_counts": {},
            "evidence_type_counts": {},
            "total_evidences": len(evidences),
        }
        
        # Count by confidence level
        for evidence in evidences:
            level = evidence.confidence_level.value
            summary["confidence_level_counts"][level] = (
                summary["confidence_level_counts"].get(level, 0) + 1
            )
        
        # Count by evidence type
        for evidence in evidences:
            ev_type = evidence.evidence_type.value
            summary["evidence_type_counts"][ev_type] = (
                summary["evidence_type_counts"].get(ev_type, 0) + 1
            )
        
        # Sort counts for deterministic output
        summary["confidence_level_counts"] = self.sort_dict_by_keys(
            summary["confidence_level_counts"]
        )
        summary["evidence_type_counts"] = self.sort_dict_by_keys(
            summary["evidence_type_counts"]
        )
        
        # Add confidence and quality metrics
        from confidence_quality_metrics import ArtifactMetricsIntegrator
        
        integrator = ArtifactMetricsIntegrator()
        
        # Calculate evidence-level confidence and quality metrics
        evidence_data = {
            'evidence': evidence_dicts,
            'total_evidence': len(evidences),
            'nlp_score': sum(e.confidence_score for e in evidences) / len(evidences) if evidences else 0.0,
        }
        
        enhanced_output = integrator.add_metrics_to_question_artifact(evidence_data)
        
        output = {
            "component": self.component_name,
            "component_id": self.component_id,
            "confidence_score": enhanced_output['confidence_score'],
            "quality_score": enhanced_output['quality_score'],
            "evidences": evidence_dicts,
            "metadata": self.get_deterministic_metadata(),
            "metrics_metadata": enhanced_output['metrics_metadata'],
            "operation_id": operation_id,
            "processing_stats": self.sort_dict_by_keys(self.processing_stats),
            "status": "success",
            "summary": summary,
            "timestamp": self._get_deterministic_timestamp(),
        }
        
        return self.sort_dict_by_keys(output)
    
    def save_artifact(self, data: Dict[str, Any], document_stem: str, output_dir: str = "canonical_flow/analysis") -> str:
        """
        Save evidence processor output to canonical_flow/analysis/ directory with standardized naming.
        
        Args:
            data: Evidence data to save
            document_stem: Base document identifier (without extension)
            output_dir: Output directory (defaults to canonical_flow/analysis)
            
        Returns:
            Path to saved artifact
        """
        try:
            # Create output directory if it doesn't exist
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            # Generate standardized filename
            filename = f"{document_stem}_evidence.json"
            artifact_path = Path(output_dir) / filename
            
            # Save with consistent JSON formatting
            with open(artifact_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"EvidenceProcessor artifact saved: {artifact_path}")
            return str(artifact_path)
            
        except Exception as e:
            error_msg = f"Failed to save EvidenceProcessor artifact to {output_dir}/{document_stem}_evidence.json: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)


# Backward compatibility functions
def process_evidences(evidences: List[str]) -> Dict[str, Any]:
    """Process multiple evidences"""
    processor = EvidenceProcessor()
    return processor.process({"evidences": evidences})


def process_single_evidence(evidence: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
    """Process single evidence"""
    processor = EvidenceProcessor()
    evidence_data = {"evidence": {"text": evidence, "metadata": metadata or {}}}
    return processor.process(evidence_data)


def process(data=None, context=None):
    """Backward compatible process function"""
    processor = EvidenceProcessor()
    result = processor.process(data, context)
    
    # Save artifact if document_stem is provided
    if data and isinstance(data, dict) and 'document_stem' in data:
        processor.save_artifact(result, data['document_stem'])
    
    return result
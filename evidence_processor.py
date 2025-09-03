"""
Evidence processing system for generating structured evidence objects with citation information.
Integrates with scoring and classification systems for full audit trails.
"""

import json
import hashlib
import os
# # # from dataclasses import dataclass, field, asdict  # Module not found  # Module not found  # Module not found
# # # from datetime import datetime  # Module not found  # Module not found  # Module not found
# # # from enum import Enum  # Module not found  # Module not found  # Module not found
# # # from pathlib import Path  # Module not found  # Module not found  # Module not found
# # # from typing import Any, Dict, List, Optional, Union, Callable  # Module not found  # Module not found  # Module not found
# # # from uuid import uuid4  # Module not found  # Module not found  # Module not found

# Import audit logger for execution tracing
try:
# # #     from canonical_flow.analysis.audit_logger import get_audit_logger  # Module not found  # Module not found  # Module not found
except ImportError:
    # Fallback when audit logger is not available
    get_audit_logger = None

# Import validation model for integration
try:
# # #     from evidence_validation_model import (  # Module not found  # Module not found  # Module not found
        EvidenceValidationModel, 
        ValidationSeverity,
        DNPEvidenceValidator,
        EvidenceValidationRequest,
        EvidenceValidationResponse
    )
    VALIDATION_MODEL_AVAILABLE = True
except (ImportError, NameError, AttributeError) as e:
    # Fallback when validation model is not available
    print(f"Warning: Validation model not available ({e}). Proceeding without DNP validation.")
    EvidenceValidationModel = None
    ValidationSeverity = None
    DNPEvidenceValidator = None
    EvidenceValidationRequest = None
    EvidenceValidationResponse = None
    VALIDATION_MODEL_AVAILABLE = False


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
    publication_date: Optional[datetime] = None
    page_number: Optional[int] = None
    section_header: Optional[str] = None
    subsection_header: Optional[str] = None
    document_type: str = "document"
    url: Optional[str] = None
    doi: Optional[str] = None
    isbn: Optional[str] = None


@dataclass
class EvidenceChunk:
    """Processed evidence chunk with context."""

    chunk_id: str
    text: str
    context_before: str = ""
    context_after: str = ""
    start_position: int = 0
    end_position: int = 0
    processing_timestamp: datetime = field(default_factory=datetime.now)
    raw_text: str = ""


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
        year = (
            self.metadata.publication_date.year
            if self.metadata.publication_date
            else "n.d."
        )

        if self.metadata.url:
# # #             return f"{author} ({year}). {title}. Retrieved from {self.metadata.url}"  # Module not found  # Module not found  # Module not found
        else:
            return f"{author} ({year}). {title}."

    def to_mla_format(self) -> str:
        """Generate MLA format citation."""
        author = self.metadata.author if self.metadata.author else "Unknown Author"
        title = self.metadata.title

        if self.metadata.publication_date:
            date_str = self.metadata.publication_date.strftime("%d %b %Y")
        else:
            date_str = "n.d."

        if self.metadata.url:
            return f'{author}. "{title}." Web. {date_str}.'
        else:
            return f'{author}. "{title}." {date_str}.'


@dataclass
class ScoringMetrics:
    """Scoring and classification metrics for evidence."""

    relevance_score: float = 0.0
    credibility_score: float = 0.0
    recency_score: float = 0.0
    authority_score: float = 0.0
    overall_score: float = 0.0
    quality_tag: str = "medium"  # "high", "medium", or "low"
    confidence_level: ConfidenceLevel = ConfidenceLevel.MEDIUM
    classification_labels: List[str] = field(default_factory=list)
    processing_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Result of evidence validation against DNP standards."""
    
    evidence_id: str
    is_valid: bool
    validation_score: float
    validation_messages: List[str] = field(default_factory=list)
    severity_level: Optional[str] = None
    validation_timestamp: datetime = field(default_factory=datetime.now)
    traceability_id: Optional[str] = None


@dataclass
class StructuredEvidence:
    """Complete structured evidence object with all metadata and traceability."""

    evidence_id: str
    question_id: str
    dimension: str
    evidence_type: EvidenceType
    chunk: EvidenceChunk
    citation: Citation
    scoring: ScoringMetrics
    exact_text: str
    context_text: str
    supporting_snippets: List[str] = field(default_factory=list)
    contradicting_snippets: List[str] = field(default_factory=list)
    creation_timestamp: datetime = field(default_factory=datetime.now)
    audit_trail: List[Dict[str, Any]] = field(default_factory=list)
    validation_result: Optional[ValidationResult] = None

    def add_audit_entry(self, action: str, details: Dict[str, Any]):
        """Add an entry to the audit trail."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "details": details,
        }
        self.audit_trail.append(entry)

    def get_traceability_path(self) -> Dict[str, str]:
# # #         """Get the full traceability path from source to evidence."""  # Module not found  # Module not found  # Module not found
        return {
            "document_id": self.citation.metadata.document_id,
            "chunk_id": self.chunk.chunk_id,
            "evidence_id": self.evidence_id,
            "question_id": self.question_id,
            "dimension": self.dimension,
        }


class EvidenceProcessor:
    """Main processor for generating structured evidence objects."""

    def __init__(self, validation_model: Optional[EvidenceValidationModel] = None):
        self.citation_formatter = CitationFormatter()
        self.scoring_system = EvidenceScoringSystem()
        self.validation_model = validation_model
        self.dnp_validator = None
        if validation_model and DNPEvidenceValidator:
            self.dnp_validator = DNPEvidenceValidator(validation_model)
        self._evidence_id_counter = 0
    
    def process(
        self,
        raw_evidence_candidates: List[Dict[str, Any]],
        validation_hooks: Optional[Dict[str, Callable]] = None,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Standardized process method that accepts raw evidence candidates and returns structured evidence.
        
        Args:
            raw_evidence_candidates: List of raw evidence data dictionaries
            validation_hooks: Optional dictionary of validation functions to apply
                               Keys: validation stage names, Values: validation functions
            output_dir: Optional output directory for structured evidence files
        
        Returns:
            Dictionary containing structured evidence objects and processing metadata
        """
        # Set default output directory
        if output_dir is None:
            output_dir = "canonical_flow/analysis"
        
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        structured_evidence_list = []
        processing_metadata = {
            "processed_at": datetime.now().isoformat(),
            "total_candidates": len(raw_evidence_candidates),
            "validation_hooks_applied": list(validation_hooks.keys()) if validation_hooks else [],
            "output_directory": output_dir,
            "structured_evidence_ids": []
        }
        
        for raw_evidence in raw_evidence_candidates:
            try:
                # Generate hash-based deterministic ID
                evidence_id = self._generate_hash_based_id(raw_evidence)
                
                # Convert raw evidence to structured evidence
                structured_evidence = self._convert_raw_to_structured(
                    raw_evidence, 
                    evidence_id,
                    validation_hooks
                )
                
                structured_evidence_list.append(structured_evidence)
                processing_metadata["structured_evidence_ids"].append(evidence_id)
                
            except Exception as e:
                processing_metadata.setdefault("errors", []).append({
                    "raw_evidence": str(raw_evidence)[:100] + "..." if len(str(raw_evidence)) > 100 else str(raw_evidence),
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
        
        # Package results
        result = {
            "structured_evidence": structured_evidence_list,
            "processing_metadata": processing_metadata,
            "summary": self._generate_process_summary(structured_evidence_list)
        }
        
        # Write to output directory
        output_file = Path(output_dir) / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_evidence.json"
        self._write_evidence_file(result, output_file)
        processing_metadata["output_file"] = str(output_file)
        
        return result
    
    def _convert_raw_to_structured(
        self, 
        raw_evidence: Dict[str, Any], 
        evidence_id: str,
        validation_hooks: Optional[Dict[str, Callable]] = None
    ) -> StructuredEvidence:
        """Convert raw evidence candidate to structured evidence object."""
        
        # Extract required fields with defaults
        text = raw_evidence.get("text", "")
        question_id = raw_evidence.get("question_id", "unknown")
        dimension = raw_evidence.get("dimension", "default")
        evidence_type_str = raw_evidence.get("evidence_type", "direct_quote")
        
        # Convert evidence type string to enum
        try:
            evidence_type = EvidenceType(evidence_type_str)
        except ValueError:
            evidence_type = EvidenceType.DIRECT_QUOTE
        
        # Create evidence chunk
        chunk = EvidenceChunk(
            chunk_id=f"chunk_{evidence_id}",
            text=text,
            context_before=raw_evidence.get("context_before", ""),
            context_after=raw_evidence.get("context_after", ""),
            start_position=raw_evidence.get("start_position", 0),
            end_position=raw_evidence.get("end_position", len(text)),
            raw_text=raw_evidence.get("raw_text", text)
        )
        
        # Create source metadata
        metadata = SourceMetadata(
            document_id=raw_evidence.get("document_id", "unknown_doc"),
            title=raw_evidence.get("title", "Untitled Document"),
            author=raw_evidence.get("author", ""),
            publication_date=self._parse_date(raw_evidence.get("publication_date")),
            page_number=raw_evidence.get("page_number"),
            section_header=raw_evidence.get("section_header"),
            subsection_header=raw_evidence.get("subsection_header"),
            document_type=raw_evidence.get("document_type", "document"),
            url=raw_evidence.get("url"),
            doi=raw_evidence.get("doi"),
            isbn=raw_evidence.get("isbn")
        )
        
        # Create citation
        citation = self.citation_formatter.create_citation(metadata)
        
        # Generate scoring metrics
        scoring = self.scoring_system.score_evidence(chunk, metadata, question_id, dimension)
        
        # Build context text
        context_text = self._build_context_text(chunk)
        
        # Create structured evidence
        evidence = StructuredEvidence(
            evidence_id=evidence_id,
            question_id=question_id,
            dimension=dimension,
            evidence_type=evidence_type,
            chunk=chunk,
            citation=citation,
            scoring=scoring,
            exact_text=text,
            context_text=context_text,
            supporting_snippets=raw_evidence.get("supporting_snippets", []),
            contradicting_snippets=raw_evidence.get("contradicting_snippets", [])
        )
        
        # Add initial audit trail
        evidence.add_audit_entry("created_from_raw", {
            "processor": "EvidenceProcessor.process",
            "evidence_id": evidence_id,
            "raw_evidence_hash": hashlib.sha256(str(raw_evidence).encode()).hexdigest()[:8]
        })
        
        # Apply validation hooks if provided
        if validation_hooks:
            for hook_name, hook_func in validation_hooks.items():
                try:
                    hook_result = hook_func(evidence, raw_evidence)
                    evidence.add_audit_entry(f"validation_hook_{hook_name}", {
                        "hook_name": hook_name,
                        "result": hook_result,
                        "timestamp": datetime.now().isoformat()
                    })
                except Exception as e:
                    evidence.add_audit_entry(f"validation_hook_error_{hook_name}", {
                        "hook_name": hook_name,
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    })
        
        # Apply DNP validation if available
        if self.dnp_validator and EvidenceValidationRequest:
            validation_request = EvidenceValidationRequest(
                evidence_id=evidence_id,
                text=evidence.exact_text,
                context=evidence.context_text,
                metadata={
                    'document_id': metadata.document_id,
                    'author': metadata.author,
                    'publication_date': metadata.publication_date,
                    'doi': metadata.doi,
                    'isbn': metadata.isbn,
                    'title': metadata.title,
                },
                question_id=question_id,
                dimension=dimension,
            )
            
            dnp_response = self.dnp_validator.validate_evidence(validation_request)
            
            validation_result = ValidationResult(
                evidence_id=evidence_id,
                is_valid=dnp_response.is_valid,
                validation_score=dnp_response.validation_score,
                validation_messages=dnp_response.validation_messages,
                severity_level=dnp_response.severity_level.value if dnp_response.severity_level else "MEDIUM",
                traceability_id=dnp_response.traceability_id,
            )
            
            evidence.validation_result = validation_result
            evidence.add_audit_entry("dnp_validated", {
                "validation_score": validation_result.validation_score,
                "is_valid": validation_result.is_valid,
                "traceability_id": validation_result.traceability_id
            })
        
        return evidence
    
    def _parse_date(self, date_input: Union[str, datetime, None]) -> Optional[datetime]:
        """Parse various date input formats to datetime object."""
        if date_input is None:
            return None
        if isinstance(date_input, datetime):
            return date_input
        if isinstance(date_input, str):
            try:
                return datetime.fromisoformat(date_input)
            except ValueError:
                try:
                    return datetime.strptime(date_input, "%Y-%m-%d")
                except ValueError:
                    return None
        return None
    
    def _generate_process_summary(self, evidence_list: List[StructuredEvidence]) -> Dict[str, Any]:
        """Generate summary for the process() method results."""
        if not evidence_list:
            return {"total_processed": 0, "success": False}
        
        return {
            "total_processed": len(evidence_list),
            "success": True,
            "unique_dimensions": len(set(e.dimension for e in evidence_list)),
            "unique_questions": len(set(e.question_id for e in evidence_list)),
            "evidence_types": list(set(e.evidence_type.value for e in evidence_list)),
            "validation_coverage": sum(1 for e in evidence_list if e.validation_result is not None),
            "average_score": sum(e.scoring.overall_score for e in evidence_list) / len(evidence_list)
        }
    
    def _write_evidence_file(self, result: Dict[str, Any], output_file: Path) -> None:
        """Write structured evidence to JSON file with proper serialization."""
        # Convert dataclasses to dictionaries for JSON serialization
        serializable_result = self._make_serializable(result)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_result, f, indent=2, ensure_ascii=False, default=str)
    
    def _make_serializable(self, obj: Any) -> Any:
        """Convert objects to JSON-serializable format."""
        if hasattr(obj, '__dict__'):
            # Handle dataclass objects
            if hasattr(obj, '__dataclass_fields__'):
                return asdict(obj)
            else:
                return {k: self._make_serializable(v) for k, v in obj.__dict__.items()}
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, Enum):
            return obj.value
        elif isinstance(obj, datetime):
            return obj.isoformat()
        else:
            return obj

    def process_evidence_chunks(
        self,
        chunks: List[EvidenceChunk],
        metadata_list: List[SourceMetadata],
        question_id: str,
        dimension: str,
        evidence_type: EvidenceType = EvidenceType.DIRECT_QUOTE,
    ) -> List[StructuredEvidence]:
        """
        Process evidence chunks and generate structured evidence objects.

        Args:
            chunks: List of processed evidence chunks
            metadata_list: List of source metadata for each chunk
            question_id: Identifier for the question being answered
            dimension: Dimension/aspect being evaluated
            evidence_type: Type of evidence being processed

        Returns:
            List of structured evidence objects
        """
        # Audit logging for component execution
        audit_logger = get_audit_logger() if get_audit_logger else None
        input_data = {
            "chunks_count": len(chunks),
            "metadata_count": len(metadata_list),
            "question_id": question_id,
            "dimension": dimension,
            "evidence_type": evidence_type.value
        }
        
        if audit_logger:
            with audit_logger.audit_component_execution("16A", input_data) as audit_ctx:
                result = self._process_evidence_chunks_internal(
                    chunks, metadata_list, question_id, dimension, evidence_type
                )
                audit_ctx.set_output({
                    "structured_evidence_count": len(result),
                    "total_validations": sum(1 for e in result if e.validation_result)
                })
                return result
        else:
            return self._process_evidence_chunks_internal(
                chunks, metadata_list, question_id, dimension, evidence_type
            )

    def _process_evidence_chunks_internal(
        self,
        chunks: List[EvidenceChunk],
        metadata_list: List[SourceMetadata],
        question_id: str,
        dimension: str,
        evidence_type: EvidenceType,
    ) -> List[StructuredEvidence]:
        """Internal implementation of evidence chunk processing."""
        if len(chunks) != len(metadata_list):
            raise ValueError("Number of chunks must match number of metadata entries")

        structured_evidence_list = []

        for chunk, metadata in zip(chunks, metadata_list):
            # Generate unique evidence ID with counter for better traceability
            evidence_id = self._generate_evidence_id()

            # Create citation
            citation = self.citation_formatter.create_citation(metadata)

            # Generate scoring metrics
            scoring = self.scoring_system.score_evidence(
                chunk, metadata, question_id, dimension
            )

            # Determine context text
            context_text = self._build_context_text(chunk)

            # Create structured evidence object
            evidence = StructuredEvidence(
                evidence_id=evidence_id,
                question_id=question_id,
                dimension=dimension,
                evidence_type=evidence_type,
                chunk=chunk,
                citation=citation,
                scoring=scoring,
                exact_text=chunk.text,
                context_text=context_text,
            )

            # Add initial audit trail entry
            evidence.add_audit_entry(
                "created",
                {
                    "processor": "EvidenceProcessor",
                    "source_document": metadata.document_id,
                    "chunk_processed": chunk.chunk_id,
                    "evidence_id": evidence_id,
                },
            )

            # Validate evidence if DNP validator is available
            if self.dnp_validator and EvidenceValidationRequest:
                validation_request = EvidenceValidationRequest(
                    evidence_id=evidence_id,
                    text=evidence.exact_text,
                    context=evidence.context_text,
                    metadata={
                        'document_id': metadata.document_id,
                        'author': metadata.author,
                        'publication_date': metadata.publication_date,
                        'doi': metadata.doi,
                        'isbn': metadata.isbn,
                        'title': metadata.title,
                    },
                    question_id=question_id,
                    dimension=dimension,
                )
                
                dnp_response = self.dnp_validator.validate_evidence(validation_request)
                
                # Convert DNP response to our ValidationResult format
                validation_result = ValidationResult(
                    evidence_id=evidence_id,
                    is_valid=dnp_response.is_valid,
                    validation_score=dnp_response.validation_score,
                    validation_messages=dnp_response.validation_messages,
                    severity_level=dnp_response.severity_level.value if dnp_response.severity_level else "MEDIUM",
                    traceability_id=dnp_response.traceability_id,
                )
                
                evidence.validation_result = validation_result
                
                # Add validation audit trail entry
                evidence.add_audit_entry(
                    "validated",
                    {
                        "validation_model": "DNPStandards",
                        "validation_score": validation_result.validation_score,
                        "dnp_compliance_score": dnp_response.dnp_compliance_score,
                        "is_valid": validation_result.is_valid,
                        "evidence_id": evidence_id,
                        "traceability_id": validation_result.traceability_id,
                        "rule_violations": dnp_response.rule_violations,
                    },
                )

            structured_evidence_list.append(evidence)

        return structured_evidence_list

    def _build_context_text(self, chunk: EvidenceChunk) -> str:
        """Build context text with surrounding information."""
        context_parts = []

        if chunk.context_before:
            context_parts.append(f"...{chunk.context_before}")

        context_parts.append(f"**{chunk.text}**")

        if chunk.context_after:
            context_parts.append(f"{chunk.context_after}...")

        return " ".join(context_parts)

    def _generate_evidence_id(self) -> str:
        """Generate unique evidence ID with sequential counter for traceability."""
        self._evidence_id_counter += 1
        timestamp_part = int(datetime.now().timestamp() * 1000)  # milliseconds
        unique_part = uuid4().hex[:6]
        return f"ev_{timestamp_part}_{self._evidence_id_counter:04d}_{unique_part}"
    
    def _generate_hash_based_id(self, raw_evidence: Dict[str, Any]) -> str:
# # #         """Generate deterministic hash-based identifier from raw evidence."""  # Module not found  # Module not found  # Module not found
        # Create a consistent string representation
        hash_content = json.dumps(raw_evidence, sort_keys=True, default=str)
        
        # Generate SHA-256 hash
        hash_obj = hashlib.sha256(hash_content.encode('utf-8'))
        hash_digest = hash_obj.hexdigest()[:16]  # First 16 characters for readability
        
        return f"ev_hash_{hash_digest}"

    def validate_evidence_batch(self, evidence_list: List[StructuredEvidence]) -> List[StructuredEvidence]:
        """Batch validate multiple evidence items using DNP validator."""
        if not self.dnp_validator or not EvidenceValidationRequest:
            return evidence_list
        
        # Create validation requests for all evidence
        requests = []
        for evidence in evidence_list:
            if evidence.validation_result is None:  # Only validate if not already validated
                request = EvidenceValidationRequest(
                    evidence_id=evidence.evidence_id,
                    text=evidence.exact_text,
                    context=evidence.context_text,
                    metadata={
                        'document_id': evidence.citation.metadata.document_id,
                        'author': evidence.citation.metadata.author,
                        'publication_date': evidence.citation.metadata.publication_date,
                        'doi': evidence.citation.metadata.doi,
                        'isbn': evidence.citation.metadata.isbn,
                        'title': evidence.citation.metadata.title,
                    },
                    question_id=evidence.question_id,
                    dimension=evidence.dimension,
                )
                requests.append(request)
        
        if not requests:
            return evidence_list
        
        # Batch validate
        responses = self.dnp_validator.batch_validate_evidence(requests)
        
        # Map responses back to evidence
        response_map = {resp.evidence_id: resp for resp in responses}
        
        for evidence in evidence_list:
            if evidence.evidence_id in response_map:
                dnp_response = response_map[evidence.evidence_id]
                
                # Convert to our ValidationResult format
                validation_result = ValidationResult(
                    evidence_id=evidence.evidence_id,
                    is_valid=dnp_response.is_valid,
                    validation_score=dnp_response.validation_score,
                    validation_messages=dnp_response.validation_messages,
                    severity_level=dnp_response.severity_level.value if dnp_response.severity_level else "MEDIUM",
                    traceability_id=dnp_response.traceability_id,
                )
                
                evidence.validation_result = validation_result
                
                # Add audit trail entry
                evidence.add_audit_entry(
                    "batch_validated",
                    {
                        "validation_model": "DNPStandards",
                        "validation_score": validation_result.validation_score,
                        "dnp_compliance_score": dnp_response.dnp_compliance_score,
                        "is_valid": validation_result.is_valid,
                        "evidence_id": evidence.evidence_id,
                        "traceability_id": validation_result.traceability_id,
                        "rule_violations": dnp_response.rule_violations,
                    },
                )
        
        return evidence_list

    def get_evidence_by_id(self, evidence_list: List[StructuredEvidence], evidence_id: str) -> Optional[StructuredEvidence]:
        """Retrieve evidence by its unique identifier."""
        for evidence in evidence_list:
            if evidence.evidence_id == evidence_id:
                return evidence
        return None

    def get_validation_summary(self, evidence_list: List[StructuredEvidence]) -> Dict[str, Any]:
        """Generate validation summary for processed evidence."""
        validated_evidence = [e for e in evidence_list if e.validation_result is not None]
        
        if not validated_evidence:
            return {"total_validated": 0, "validation_enabled": False}
        
        valid_count = sum(1 for e in validated_evidence if e.validation_result.is_valid)
        avg_score = sum(e.validation_result.validation_score for e in validated_evidence) / len(validated_evidence)
        
        severity_counts = {}
        for evidence in validated_evidence:
            severity = evidence.validation_result.severity_level
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
        return {
            "total_validated": len(validated_evidence),
            "valid_count": valid_count,
            "invalid_count": len(validated_evidence) - valid_count,
            "average_validation_score": avg_score,
            "severity_distribution": severity_counts,
            "validation_enabled": True,
            "traceability_ids": [e.validation_result.traceability_id for e in validated_evidence if e.validation_result.traceability_id],
        }

    def aggregate_evidence_by_dimension(
        self, evidence_list: List[StructuredEvidence]
    ) -> Dict[str, List[StructuredEvidence]]:
        """Aggregate evidence by dimension for analysis."""
        aggregated = {}

        for evidence in evidence_list:
            if evidence.dimension not in aggregated:
                aggregated[evidence.dimension] = []
            aggregated[evidence.dimension].append(evidence)

        return aggregated

    def generate_evidence_summary(
        self, evidence_list: List[StructuredEvidence]
    ) -> Dict[str, Any]:
        """Generate summary statistics for processed evidence."""
        if not evidence_list:
            return {"total_evidence": 0}

        dimensions = set(e.dimension for e in evidence_list)
        evidence_types = set(e.evidence_type.value for e in evidence_list)
        avg_score = sum(e.scoring.overall_score for e in evidence_list) / len(
            evidence_list
        )

        base_summary = {
            "total_evidence": len(evidence_list),
            "dimensions_covered": list(dimensions),
            "evidence_types": list(evidence_types),
            "average_overall_score": avg_score,
            "high_confidence_count": sum(
                1
                for e in evidence_list
                if e.scoring.confidence_level == ConfidenceLevel.HIGH
            ),
            "unique_sources": len(
                set(e.citation.metadata.document_id for e in evidence_list)
            ),
            "evidence_ids": [e.evidence_id for e in evidence_list],
        }
        
        # Add validation summary if available
        validation_summary = self.get_validation_summary(evidence_list)
        base_summary.update(validation_summary)
        
        return base_summary


class CitationFormatter:
    """Handles creation and formatting of citations."""

    def create_citation(self, metadata: SourceMetadata) -> Citation:
# # #         """Create a standardized citation from metadata."""  # Module not found  # Module not found  # Module not found
        citation_id = f"cite_{uuid4().hex[:8]}"

        # Generate formatted reference
        formatted_ref = self._generate_formatted_reference(metadata)
        short_ref = self._generate_short_reference(metadata)
        inline_citation = self._generate_inline_citation(metadata)

        return Citation(
            citation_id=citation_id,
            formatted_reference=formatted_ref,
            short_reference=short_ref,
            inline_citation=inline_citation,
            metadata=metadata,
        )

    def _generate_formatted_reference(self, metadata: SourceMetadata) -> str:
        """Generate a complete formatted reference."""
        parts = []

        if metadata.author:
            parts.append(metadata.author)

        if metadata.publication_date:
            parts.append(f"({metadata.publication_date.year})")

        parts.append(f'"{metadata.title}"')

        if metadata.page_number:
            parts.append(f"p. {metadata.page_number}")

        if metadata.section_header:
            parts.append(f"Section: {metadata.section_header}")

        return ", ".join(parts)

    def _generate_short_reference(self, metadata: SourceMetadata) -> str:
        """Generate a short reference for inline use."""
        author = (
            metadata.author.split()[-1] if metadata.author else "Unknown"
        )  # Use last name
        year = metadata.publication_date.year if metadata.publication_date else "n.d."

        if metadata.page_number:
            return f"{author} {year}, p. {metadata.page_number}"
        else:
            return f"{author} {year}"

    def _generate_inline_citation(self, metadata: SourceMetadata) -> str:
        """Generate inline citation format."""
        author = (
            metadata.author.split()[-1] if metadata.author else "Unknown"
        )  # Use last name
        year = metadata.publication_date.year if metadata.publication_date else "n.d."

        if metadata.page_number:
            return f"({author}, {year}, p. {metadata.page_number})"
        else:
            return f"({author}, {year})"


class EvidenceScoringSystem:
    """System for scoring and classifying evidence."""

    def score_evidence(
        self,
        chunk: EvidenceChunk,
        metadata: SourceMetadata,
        question_id: str,
        dimension: str,
    ) -> ScoringMetrics:
        """
        Score evidence across four weighted metrics and assign quality tags.
        
        Metrics:
        - Relevance (0.4 weight): How well the evidence relates to the question/dimension
        - Credibility (0.3 weight): Source reliability and academic rigor
        - Recency (0.2 weight): How recent the evidence is
        - Authority (0.1 weight): Author expertise and source authority
        
        Quality Tags:
        - "high": Overall score ≥ 0.8
        - "medium": Overall score 0.5-0.79
        - "low": Overall score < 0.5
        """

        # Calculate individual scores (0-1 scale)
        relevance_score = self._calculate_relevance_score(
            chunk.text, question_id, dimension
        )
        credibility_score = self._calculate_credibility_score(metadata)
        recency_score = self._calculate_recency_score(metadata.publication_date)
        authority_score = self._calculate_authority_score(metadata)

        # Calculate weighted composite score
        weights = {
            "relevance": 0.4,
            "credibility": 0.3,
            "recency": 0.2,
            "authority": 0.1,
        }
        overall_score = (
            relevance_score * weights["relevance"]
            + credibility_score * weights["credibility"]
            + recency_score * weights["recency"]
            + authority_score * weights["authority"]
        )

        # Determine quality tag based on composite score
        quality_tag = self._determine_quality_tag(overall_score)

        # Determine confidence level
        confidence_level = self._determine_confidence_level(overall_score)

        # Generate classification labels
        labels = self._classify_evidence(chunk.text, dimension)

        return ScoringMetrics(
            relevance_score=relevance_score,
            credibility_score=credibility_score,
            recency_score=recency_score,
            authority_score=authority_score,
            overall_score=overall_score,
            quality_tag=quality_tag,
            confidence_level=confidence_level,
            classification_labels=labels,
            processing_metadata={
                "scoring_timestamp": datetime.now().isoformat(),
                "weights_used": weights,
                "quality_thresholds": {"high": 0.8, "medium": 0.5}
            },
        )

    def _calculate_relevance_score(
        self, text: str, question_id: str, dimension: str
    ) -> float:
        """
        Calculate relevance score (0-1 scale) based on text alignment with question/dimension.
        
        Factors:
        - Direct keyword/concept matches with dimension
        - Text length appropriateness
        - Semantic relevance indicators
        - Question context alignment
        """
        score = 0.2  # Base score
        text_lower = text.lower()
        dimension_lower = dimension.lower()

        # Direct dimension keyword match
        if dimension_lower in text_lower:
            score += 0.4

        # Semantic relevance (simple keyword expansion)
        dimension_keywords = self._get_dimension_keywords(dimension)
        keyword_matches = sum(1 for keyword in dimension_keywords if keyword in text_lower)
        score += min(keyword_matches * 0.1, 0.3)

        # Text length appropriateness (optimal range for evidence)
        text_length = len(text.split())
        if 15 <= text_length <= 150:
            score += 0.2
        elif 5 <= text_length <= 300:
            score += 0.1

        # Question ID context (if contains meaningful context)
        if question_id and question_id.lower() != "unknown":
            question_terms = question_id.lower().replace("_", " ").split()
            question_matches = sum(1 for term in question_terms if term in text_lower)
            score += min(question_matches * 0.05, 0.1)

        return min(score, 1.0)
    
    def _get_dimension_keywords(self, dimension: str) -> List[str]:
        """Get related keywords for a dimension to improve relevance scoring."""
        keyword_map = {
            "accuracy": ["precise", "correct", "accurate", "error", "performance"],
            "effectiveness": ["efficient", "successful", "improvement", "impact"],
            "safety": ["safe", "risk", "hazard", "secure", "protection"],
            "usability": ["user", "interface", "experience", "ease", "intuitive"],
            "reliability": ["consistent", "dependable", "stable", "robust"],
            "quality": ["standard", "grade", "excellence", "benchmark"],
            "performance": ["speed", "throughput", "latency", "response", "efficiency"],
            "cost": ["price", "expense", "budget", "economic", "financial"]
        }
        
        dimension_lower = dimension.lower()
        for key, keywords in keyword_map.items():
            if key in dimension_lower or dimension_lower in key:
                return keywords
        return []

    def _calculate_credibility_score(self, metadata: SourceMetadata) -> float:
        """
        Calculate credibility score (0-1 scale) based on source reliability indicators.
        
        Factors:
        - Academic/peer-reviewed sources (DOI, journal type)
        - Author credentials and expertise
        - Publication venue credibility
        - Source type authority
        """
        score = 0.1  # Minimal base score

        # Peer-reviewed academic sources (highest credibility)
        if metadata.doi:
            score += 0.4
        elif "journal" in metadata.document_type.lower():
            score += 0.3
        elif "conference" in metadata.document_type.lower():
            score += 0.25

        # Author credibility
        if metadata.author and metadata.author.strip() and metadata.author != "Unknown Author":
            score += 0.2
            # Additional credit for academic titles
            author_lower = metadata.author.lower()
            if any(title in author_lower for title in ["dr.", "prof.", "phd", "md"]):
                score += 0.1

        # ISBN suggests published work
        if metadata.isbn:
            score += 0.15

        # URL credibility (basic domain analysis)
        if metadata.url:
            url_lower = metadata.url.lower()
            if any(domain in url_lower for domain in [".edu", ".gov", ".org"]):
                score += 0.1
            elif any(domain in url_lower for domain in [".com", ".net"]):
                score += 0.05

        # Document type credibility
        doc_type_lower = metadata.document_type.lower()
        if "academic" in doc_type_lower or "research" in doc_type_lower:
            score += 0.1
        elif "report" in doc_type_lower or "whitepaper" in doc_type_lower:
            score += 0.05

        return min(score, 1.0)

    def _calculate_recency_score(self, publication_date: Optional[datetime]) -> float:
        """
        Calculate recency score (0-1 scale) based on publication date.
        
        Recent evidence is generally more valuable for evolving fields.
        
        Scoring:
        - ≤ 1 year: 1.0
        - ≤ 2 years: 0.9
        - ≤ 3 years: 0.8
        - ≤ 5 years: 0.7
        - ≤ 7 years: 0.6
        - ≤ 10 years: 0.5
        - > 10 years: 0.3
        - Unknown: 0.4 (neutral)
        """
        if not publication_date:
            return 0.4  # Neutral score for unknown dates

        years_old = (datetime.now() - publication_date).days / 365.25

        if years_old <= 1:
            return 1.0
        elif years_old <= 2:
            return 0.9
        elif years_old <= 3:
            return 0.8
        elif years_old <= 5:
            return 0.7
        elif years_old <= 7:
            return 0.6
        elif years_old <= 10:
            return 0.5
        else:
            return 0.3

    def _calculate_authority_score(self, metadata: SourceMetadata) -> float:
        """
        Calculate authority score (0-1 scale) based on source and author authority.
        
        Factors:
        - Author credentials and reputation
        - Institutional affiliations
        - Publication venue prestige
        - Formal identifiers (DOI, ISBN)
        """
        score = 0.2  # Base score

        # Author authority
        if metadata.author and metadata.author.strip() and metadata.author != "Unknown Author":
            author_lower = metadata.author.lower()
            
            # Academic titles indicate authority
            if any(title in author_lower for title in ["dr.", "prof.", "professor"]):
                score += 0.3
            elif any(title in author_lower for title in ["phd", "md", "ph.d."]):
                score += 0.25
            else:
                # At least has a real author name
                score += 0.15

        # Formal publication identifiers indicate authority
        if metadata.doi:
            score += 0.2
        if metadata.isbn:
            score += 0.15

# # #         # Institutional authority (from URL or title)  # Module not found  # Module not found  # Module not found
        if metadata.url:
            url_lower = metadata.url.lower()
            if any(domain in url_lower for domain in [".edu", ".gov"]):
                score += 0.15
            elif ".org" in url_lower:
                score += 0.1

        # Title/source authority indicators
        title_lower = metadata.title.lower()
        if any(term in title_lower for term in ["study", "research", "analysis", "investigation"]):
            score += 0.1

        # Document type authority
        doc_type_lower = metadata.document_type.lower()
        if any(term in doc_type_lower for term in ["academic", "journal", "research"]):
            score += 0.1

        return min(score, 1.0)

    def _determine_quality_tag(self, overall_score: float) -> str:
        """
        Determine quality tag based on composite weighted score.
        
        Returns:
            "high" for scores ≥ 0.8
            "medium" for scores 0.5-0.79
            "low" for scores < 0.5
        """
        if overall_score >= 0.8:
            return "high"
        elif overall_score >= 0.5:
            return "medium"
        else:
            return "low"

    def _determine_confidence_level(self, overall_score: float) -> ConfidenceLevel:
        """Determine confidence level based on overall score."""
        if overall_score >= 0.8:
            return ConfidenceLevel.HIGH
        elif overall_score >= 0.6:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW

    def _classify_evidence(self, text: str, dimension: str) -> List[str]:
        """Generate classification labels for evidence."""
        labels = []

        # Simple keyword-based classification
        if any(word in text.lower() for word in ["statistics", "data", "percent", "%"]):
            labels.append("quantitative")

        if any(
            word in text.lower()
            for word in ["expert", "professor", "dr.", "researcher"]
        ):
            labels.append("expert_opinion")

        if any(word in text.lower() for word in ["case study", "example", "instance"]):
            labels.append("case_study")

        labels.append(dimension.lower().replace(" ", "_"))

        return labels


def create_sample_evidence_with_validation() -> List[StructuredEvidence]:
    """Create sample structured evidence with DNP validation for demonstration."""
    # Create sample validation model
    if EvidenceValidationModel:
# # #         from evidence_validation_model import create_validation_model, EvidenceType as VEvidenceType, QuestionType, ValidationSeverity  # Module not found  # Module not found  # Module not found
        
        validation_model = create_validation_model(
            questions=[(QuestionType.TECHNICAL, "medical_ai", 1, "AI accuracy in medical diagnosis")],
            standards_dict={"accuracy": "Minimum 80% accuracy required for medical applications"},
            evidence_types=[VEvidenceType.SCIENTIFIC_STUDY],
            queries=[("medical AI accuracy", "en", None, 1.0)],
            rules=[
                ("accuracy_rule", "Evidence must demonstrate quantifiable accuracy", ValidationSeverity.HIGH, 0.8, 0.8),
                ("authorship_rule", "Evidence must have clear authorship", ValidationSeverity.MEDIUM, 0.7, None),
            ],
        )
        
        processor = EvidenceProcessor(validation_model=validation_model)
    else:
        processor = EvidenceProcessor()

    # Sample data
    sample_chunks = [
        EvidenceChunk(
            chunk_id="chunk_001",
            text="Machine learning algorithms have shown 85% accuracy in medical diagnosis applications.",
            context_before="Recent studies in healthcare technology indicate that",
            context_after="This represents a significant improvement over traditional methods.",
            start_position=150,
            end_position=250,
        ),
        EvidenceChunk(
            chunk_id="chunk_002", 
            text="The research demonstrates consistent results across multiple hospital systems.",
            context_before="Validation studies conducted over 12 months show that",
            context_after="These findings suggest broad applicability of the approach.",
            start_position=300,
            end_position=380,
        )
    ]

    sample_metadata = [
        SourceMetadata(
            document_id="doc_001",
            title="Advances in Medical AI Systems",
            author="Dr. Sarah Johnson",
            publication_date=datetime(2023, 6, 15),
            page_number=42,
            section_header="Results and Analysis",
            document_type="academic_journal",
            doi="10.1234/medical-ai-2023",
        ),
        SourceMetadata(
            document_id="doc_002",
            title="Clinical Validation of AI Diagnostic Tools",
            author="Dr. Michael Chen",
            publication_date=datetime(2023, 8, 20),
            page_number=67,
            section_header="Multi-site Validation",
            document_type="academic_journal",
            doi="10.5678/clinical-ai-validation-2023",
        )
    ]

    evidence_list = processor.process_evidence_chunks(
        chunks=sample_chunks,
        metadata_list=sample_metadata,
        question_id="q_001",
        dimension="accuracy",
        evidence_type=EvidenceType.STATISTICAL,
    )
    
    return evidence_list


def create_sample_evidence() -> List[StructuredEvidence]:
    """Create sample structured evidence for demonstration (backward compatibility)."""
    return create_sample_evidence_with_validation()


if __name__ == "__main__":
    # Demonstration with validation integration
    print("Creating sample evidence with DNP validation...")
    sample_evidence = create_sample_evidence_with_validation()

    for evidence in sample_evidence:
        print(f"Evidence ID: {evidence.evidence_id}")
        print(f"Citation: {evidence.citation.formatted_reference}")
        print(f"Score: {evidence.scoring.overall_score:.2f}")
        print(f"Confidence: {evidence.scoring.confidence_level.value}")
        print(f"Traceability: {evidence.get_traceability_path()}")
        
        # Show validation results if available
        if evidence.validation_result:
            print(f"Validation: {evidence.validation_result.is_valid} (Score: {evidence.validation_result.validation_score:.2f})")
            print(f"Severity: {evidence.validation_result.severity_level}")
            print(f"Messages: {', '.join(evidence.validation_result.validation_messages)}")
            print(f"Traceability ID: {evidence.validation_result.traceability_id}")
        else:
            print("No validation performed")
        
        print("-" * 50)
    
    # Show summary with validation statistics
    processor = EvidenceProcessor()
    summary = processor.generate_evidence_summary(sample_evidence)
    print("\nEvidence Summary with Validation:")
    print(json.dumps(summary, indent=2, default=str))

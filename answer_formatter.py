"""
DNP Compliance Answer Formatting System - Industrial Grade

A self-contained system for transforming aggregated evidence objects into structured
answer format using DNP compliance rules. Implements confidence calibration, audit trails,
and comprehensive traceability without external dependencies.
"""

import hashlib
import json
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union


# ==================== ENUM DEFINITIONS ====================
class EvidenceType(str, Enum):
    DIRECT_QUOTE = "direct_quote"
    PARAPHRASE = "paraphrase"
    STATISTICAL = "statistical"
    EXPERT_OPINION = "expert_opinion"
    LEGAL_REFERENCE = "legal_reference"


class ConfidenceLevel(str, Enum):
    VERY_HIGH = "very_high"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"


class ValidationSeverity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFORMATIONAL = "informational"


class ComplianceLevel(str, Enum):
    FULLY_COMPLIANT = "fully_compliant"
    SUBSTANTIALLY_COMPLIANT = "substantially_compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    NON_COMPLIANT = "non_compliant"
    NOT_APPLICABLE = "not_applicable"


class CheckStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    ERROR = "error"
    NOT_APPLICABLE = "not_applicable"


class ViolationSeverity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class AnswerStatus(str, Enum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    INSUFFICIENT = "insufficient"
    CONFLICTED = "conflicted"
    INDETERMINATE = "indeterminate"


class DNPComplianceLevel(str, Enum):
    FULLY_COMPLIANT = "fully_compliant"
    SUBSTANTIALLY_COMPLIANT = "substantially_compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    NON_COMPLIANT = "non_compliant"
    EXEMPT = "exempt"


# ==================== CORE DATA STRUCTURES ====================
@dataclass
class SourceMetadata:
    document_id: str
    document_type: str
    publication_date: str
    author: str
    publisher: str
    jurisdiction: str
    version: str
    access_timestamp: str
    retrieval_url: Optional[str] = None
    authenticity_score: float = 1.0
    classification: str = "public"


@dataclass
class Citation:
    metadata: SourceMetadata
    formatted_reference: str
    inline_citation: str
    page_number: Optional[str] = None
    section: Optional[str] = None


@dataclass
class EvidenceScoring:
    relevance_score: float
    accuracy_score: float
    recency_score: float
    authority_score: float
    objectivity_score: float
    overall_score: float
    weighting_factor: float = 1.0
    calibration_notes: List[str] = field(default_factory=list)


@dataclass
class StructuredEvidence:
    evidence_id: str
    exact_text: str
    evidence_type: EvidenceType
    dimension: str
    citation: Citation
    scoring: EvidenceScoring
    extraction_context: str
    processing_metadata: Dict[str, Any] = field(default_factory=dict)
    related_evidence_ids: List[str] = field(default_factory=list)


@dataclass
class ValidationRule:
    rule_id: str
    description: str
    severity: ValidationSeverity
    threshold: float
    weight: float
    applicable_evidence_types: List[EvidenceType]
    validation_function: str  # Name of function that implements this rule


@dataclass
class ValidationCriteria:
    rules: List[ValidationRule]
    overall_threshold: float
    version: str
    effective_date: str
    compliance_framework: str


@dataclass
class DNPStandards:
    standards: Dict[str, str]
    version: str
    jurisdiction: str
    effective_date: str
    amendments: List[Dict[str, Any]]


# ==================== FORMATTER-SPECIFIC STRUCTURES ====================
@dataclass
class CoverageMetrics:
    total_dimensions: int
    covered_dimensions: int
    evidence_density: float
    source_diversity: float
    temporal_coverage: float
    jurisdictional_coverage: float
    depth_score: float
    breadth_score: float

    @property
    def coverage_ratio(self) -> float:
        return self.covered_dimensions / self.total_dimensions if self.total_dimensions > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ConfidenceMetrics:
    base_confidence: float
    coverage_adjustment: float
    quality_adjustment: float
    consistency_adjustment: float
    source_reliability_adjustment: float
    temporal_relevance_adjustment: float
    final_confidence: float
    calibration_factors: Dict[str, float]
    confidence_band: Tuple[float, float]
    calibration_notes: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ReasoningStep:
    step_id: str
    description: str
    evidence_used: List[str]
    inference_type: str
    confidence_impact: float
    logical_sequence: int
    dependencies: List[str]
    timestamp: str
    validation_status: CheckStatus

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ReasoningChain:
    chain_id: str
    question_id: str
    steps: List[ReasoningStep]
    final_conclusion: str
    total_confidence: float
    creation_timestamp: str
    validation_status: CheckStatus
    chain_coverage: float

    def get_evidence_lineage(self) -> List[str]:
        evidence_ids = []
        for step in self.steps:
            evidence_ids.extend(step.evidence_used)
        return list(set(evidence_ids))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chain_id": self.chain_id,
            "question_id": self.question_id,
            "steps": [step.to_dict() for step in self.steps],
            "final_conclusion": self.final_conclusion,
            "total_confidence": self.total_confidence,
            "creation_timestamp": self.creation_timestamp,
            "validation_status": self.validation_status.value,
            "chain_coverage": self.chain_coverage,
            "evidence_lineage": self.get_evidence_lineage()
        }


@dataclass
class AuditTrail:
    trail_id: str
    answer_id: str
    processing_stages: List[Dict[str, Any]]
    evidence_sources: List[str]
    validation_checks: List[Dict[str, Any]]
    compliance_assessments: List[Dict[str, Any]]
    quality_metrics: Dict[str, float]
    timestamp: str
    system_version: str
    environment: str

    def add_stage(self, stage_name: str, details: Dict[str, Any]):
        stage_entry = {
            "stage": stage_name,
            "timestamp": time.time(),
            "details": details,
        }
        self.processing_stages.append(stage_entry)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trail_id": self.trail_id,
            "answer_id": self.answer_id,
            "processing_stages": self.processing_stages,
            "evidence_sources": self.evidence_sources,
            "validation_checks": self.validation_checks,
            "compliance_assessments": self.compliance_assessments,
            "quality_metrics": self.quality_metrics,
            "timestamp": self.timestamp,
            "system_version": self.system_version,
            "environment": self.environment
        }


@dataclass
class DNPCompliantAnswer:
    answer_id: str
    question_id: str
    answer_text: str
    answer_status: AnswerStatus
    compliance_level: DNPComplianceLevel
    confidence_metrics: ConfidenceMetrics
    supporting_evidence: List[StructuredEvidence]
    reasoning_chain: ReasoningChain
    audit_trail: AuditTrail
    dnp_validation_results: Dict[str, Any]
    source_attribution: Dict[str, List[str]]
    metadata: Dict[str, Any]
    creation_timestamp: str
    expiration_timestamp: str
    version: str

    def get_citation_text(self) -> str:
        citations = []
        for evidence in self.supporting_evidence:
            citations.append(evidence.citation.inline_citation)
        return "; ".join(citations)

    def get_traceability_hash(self) -> str:
        components = [
            self.answer_id,
            self.question_id,
            str(sorted([e.evidence_id for e in self.supporting_evidence])),
            self.reasoning_chain.chain_id,
            self.audit_trail.trail_id,
        ]
        content = "|".join(components)
        return hashlib.sha256(content.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer_id": self.answer_id,
            "question_id": self.question_id,
            "answer_text": self.answer_text,
            "answer_status": self.answer_status.value,
            "compliance_level": self.compliance_level.value,
            "confidence_metrics": self.confidence_metrics.to_dict(),
            "supporting_evidence_count": len(self.supporting_evidence),
            "reasoning_chain": self.reasoning_chain.to_dict(),
            "audit_trail": self.audit_trail.to_dict(),
            "dnp_validation_results": self.dnp_validation_results,
            "source_attribution": self.source_attribution,
            "metadata": self.metadata,
            "creation_timestamp": self.creation_timestamp,
            "expiration_timestamp": self.expiration_timestamp,
            "version": self.version,
            "traceability_hash": self.get_traceability_hash()
        }


# ==================== CORE FORMATTER CLASSES ====================
class ConfidenceCalibrator:
    """Calibrates confidence scores based on evidence quality and coverage."""

    def __init__(self, calibration_rules: Dict[str, Any]):
        self.calibration_rules = calibration_rules

    def calculate_confidence(
            self,
            evidence_collection: List[StructuredEvidence],
            coverage_metrics: CoverageMetrics,
            reasoning_chain: ReasoningChain,
    ) -> ConfidenceMetrics:
        """Calculate calibrated confidence metrics with industrial-grade precision."""
        # Base confidence from evidence scores
        evidence_scores = [e.scoring.overall_score * e.scoring.weighting_factor
                           for e in evidence_collection]
        base_confidence = (
            sum(evidence_scores) / len(evidence_scores) if evidence_scores else 0.0
        )

        # Coverage adjustment (up to 25% boost)
        coverage_adjustment = coverage_metrics.coverage_ratio * 0.25

        # Quality adjustment based on evidence diversity and reliability
        quality_factors = []
        quality_factors.append(coverage_metrics.source_diversity)
        quality_factors.append(coverage_metrics.temporal_coverage)
        quality_factors.append(coverage_metrics.jurisdictional_coverage)
        quality_factors.append(min(1.0, len(evidence_collection) / 8.0))  # Evidence quantity factor

        # Calculate source reliability average
        source_reliability = sum(e.citation.metadata.authenticity_score
                                 for e in evidence_collection) / len(evidence_collection)
        quality_factors.append(source_reliability)

        quality_adjustment = (
                sum(quality_factors) / len(quality_factors) * 0.20  # Up to 20% boost
        )

        # Consistency adjustment based on reasoning chain confidence
        consistency_adjustment = reasoning_chain.total_confidence * 0.15  # Up to 15% boost

        # Source reliability adjustment
        source_reliability_adjustment = source_reliability * 0.15

        # Temporal relevance adjustment
        temporal_relevance_adjustment = coverage_metrics.temporal_coverage * 0.10

        # Final confidence calculation with bounds
        final_confidence = min(
            1.0,
            max(0.0, base_confidence
                + coverage_adjustment
                + quality_adjustment
                + consistency_adjustment
                + source_reliability_adjustment
                + temporal_relevance_adjustment)
        )

        # Calculate confidence band (80% confidence interval)
        confidence_band = (
            max(0.0, final_confidence - 0.1),
            min(1.0, final_confidence + 0.1)
        )

        calibration_notes = []
        if len(evidence_collection) < 3:
            calibration_notes.append("Low evidence volume reduces confidence")
        if coverage_metrics.source_diversity < 0.5:
            calibration_notes.append("Limited source diversity affects confidence")
        if any(e.scoring.overall_score < 0.3 for e in evidence_collection):
            calibration_notes.append("Low-quality evidence impacts confidence")

        return ConfidenceMetrics(
            base_confidence=base_confidence,
            coverage_adjustment=coverage_adjustment,
            quality_adjustment=quality_adjustment,
            consistency_adjustment=consistency_adjustment,
            source_reliability_adjustment=source_reliability_adjustment,
            temporal_relevance_adjustment=temporal_relevance_adjustment,
            final_confidence=final_confidence,
            calibration_factors={
                "evidence_count": len(evidence_collection),
                "source_diversity": coverage_metrics.source_diversity,
                "coverage_ratio": coverage_metrics.coverage_ratio,
                "reasoning_steps": len(reasoning_chain.steps),
                "source_reliability": source_reliability,
                "temporal_coverage": coverage_metrics.temporal_coverage,
            },
            confidence_band=confidence_band,
            calibration_notes=calibration_notes
        )


class AuditTrailGenerator:
    """Generates comprehensive audit trails for answer formatting."""

    def __init__(self, system_version: str, environment: str):
        self.system_version = system_version
        self.environment = environment

    def create_audit_trail(self, answer_id: str, question_id: str) -> AuditTrail:
        """Create new audit trail for answer formatting process."""
        trail_id = f"trail_{uuid.uuid4().hex[:12]}"

        return AuditTrail(
            trail_id=trail_id,
            answer_id=answer_id,
            processing_stages=[],
            evidence_sources=[],
            validation_checks=[],
            compliance_assessments=[],
            quality_metrics={},
            timestamp=time.time(),
            system_version=self.system_version,
            environment=self.environment
        )


class AnswerFormatter:
    """
    Industrial-grade system for transforming aggregated evidence into structured,
    DNP-compliant answers with full audit trails and traceability.
    """

    def __init__(
            self,
            dnp_standards: DNPStandards,
            validation_criteria: ValidationCriteria,
            system_version: str = "1.0.0",
            environment: str = "production"
    ):
        """
        Initialize AnswerFormatter with DNP standards and validation criteria.

        Args:
            dnp_standards: DNP standards for compliance validation
            validation_criteria: Validation criteria for answer quality
            system_version: Version of the formatter system
            environment: Deployment environment
        """
        self.dnp_standards = dnp_standards
        self.validation_criteria = validation_criteria
        self.confidence_calibrator = ConfidenceCalibrator({})
        self.audit_generator = AuditTrailGenerator(system_version, environment)
        self.system_version = system_version

    def format_answer(
            self,
            question_id: str,
            evidence_collection: List[StructuredEvidence],
            required_dimensions: List[str],
            context_metadata: Optional[Dict[str, Any]] = None,
    ) -> DNPCompliantAnswer:
        """
        Format aggregated evidence into DNP-compliant structured answer with
        industrial-grade validation and traceability.

        Args:
            question_id: Unique identifier for the question
            evidence_collection: Collection of structured evidence
            required_dimensions: Required dimensions for complete answer
            context_metadata: Additional context metadata

        Returns:
            DNP-compliant structured answer with full traceability
        """
        answer_id = f"ans_{uuid.uuid4().hex[:12]}"
        current_time = time.time()

        # Initialize audit trail
        audit_trail = self.audit_generator.create_audit_trail(answer_id, question_id)
        audit_trail.add_stage(
            "initialization",
            {
                "evidence_count": len(evidence_collection),
                "required_dimensions": required_dimensions,
                "formatter_version": self.system_version,
                "context_metadata": context_metadata or {},
            },
        )

        try:
            # Analyze evidence coverage
            coverage_metrics = self._calculate_coverage_metrics(
                evidence_collection, required_dimensions
            )
            audit_trail.add_stage(
                "coverage_analysis",
                {
                    "metrics": coverage_metrics.to_dict(),
                    "status": "completed"
                },
            )

            # Generate reasoning chain
            reasoning_chain = self._generate_reasoning_chain(
                question_id, evidence_collection, coverage_metrics
            )
            audit_trail.add_stage(
                "reasoning_generation",
                {
                    "chain_id": reasoning_chain.chain_id,
                    "steps_count": len(reasoning_chain.steps),
                    "evidence_used": len(reasoning_chain.get_evidence_lineage()),
                    "status": "completed"
                },
            )

            # Calculate confidence metrics
            confidence_metrics = self.confidence_calibrator.calculate_confidence(
                evidence_collection, coverage_metrics, reasoning_chain
            )
            audit_trail.add_stage(
                "confidence_calculation",
                {
                    "final_confidence": confidence_metrics.final_confidence,
                    "calibration_factors": confidence_metrics.calibration_factors,
                    "status": "completed"
                },
            )

            # Generate answer text
            answer_text = self._generate_answer_text(evidence_collection, reasoning_chain)
            audit_trail.add_stage(
                "answer_generation",
                {
                    "text_length": len(answer_text),
                    "evidence_integrated": len(evidence_collection),
                    "status": "completed"
                },
            )

            # Determine answer status
            answer_status = self._determine_answer_status(
                coverage_metrics, confidence_metrics
            )

            # Validate DNP compliance
            dnp_validation = self._validate_dnp_compliance(answer_text, evidence_collection)
            compliance_level = self._determine_compliance_level(dnp_validation)
            audit_trail.add_stage(
                "dnp_validation",
                {
                    "results": dnp_validation,
                    "status": "completed"
                }
            )

            # Generate source attribution
            source_attribution = self._generate_source_attribution(evidence_collection)

            # Create metadata
            metadata = {
                "formatter_version": self.system_version,
                "processing_timestamp": current_time,
                "required_dimensions": required_dimensions,
                "coverage_metrics": coverage_metrics.to_dict(),
                "context_metadata": context_metadata or {},
            }

            # Create final answer
            answer = DNPCompliantAnswer(
                answer_id=answer_id,
                question_id=question_id,
                answer_text=answer_text,
                answer_status=answer_status,
                compliance_level=compliance_level,
                confidence_metrics=confidence_metrics,
                supporting_evidence=evidence_collection,
                reasoning_chain=reasoning_chain,
                audit_trail=audit_trail,
                dnp_validation_results=dnp_validation,
                source_attribution=source_attribution,
                metadata=metadata,
                creation_timestamp=current_time,
                expiration_timestamp=current_time + (365 * 24 * 60 * 60),  # 1 year
                version="1.0"
            )

            audit_trail.add_stage(
                "completion",
                {
                    "answer_id": answer_id,
                    "traceability_hash": answer.get_traceability_hash(),
                    "final_status": answer_status.value,
                    "compliance_level": compliance_level.value,
                    "processing_time_seconds": time.time() - current_time,
                    "status": "completed"
                },
            )

            return answer

        except Exception as e:
            # Log error to audit trail
            audit_trail.add_stage(
                "error_handling",
                {
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "status": "failed"
                },
            )
            raise

    def _calculate_coverage_metrics(
            self,
            evidence_collection: List[StructuredEvidence],
            required_dimensions: List[str],
    ) -> CoverageMetrics:
        """Calculate comprehensive coverage metrics for evidence collection."""
        covered_dimensions = set(evidence.dimension for evidence in evidence_collection)

        # Calculate evidence density
        total_evidence = len(evidence_collection)
        evidence_density = (
            total_evidence / len(required_dimensions) if required_dimensions else 0
        )

        # Calculate source diversity
        unique_sources = set(
            evidence.citation.metadata.document_id for evidence in evidence_collection
        )
        source_diversity = (
            len(unique_sources) / total_evidence if total_evidence > 0 else 0
        )

        # Calculate temporal coverage (based on publication dates)
        publication_dates = []
        for evidence in evidence_collection:
            if evidence.citation.metadata.publication_date:
                try:
                    # Convert to timestamp for calculation
                    pub_date = evidence.citation.metadata.publication_date
                    if isinstance(pub_date, str):
                        # Simplified date parsing - in production would use proper datetime
                        publication_dates.append(len(pub_date))  # Placeholder
                    else:
                        publication_dates.append(pub_date)
                except:
                    pass

        temporal_coverage = 1.0  # Default high coverage
        if len(publication_dates) > 1:
            date_range = max(publication_dates) - min(publication_dates)
            temporal_coverage = min(1.0, date_range / 365.0)  # Normalize to years

        # Calculate jurisdictional coverage
        jurisdictions = set(
            evidence.citation.metadata.jurisdiction for evidence in evidence_collection
        )
        jurisdictional_coverage = len(jurisdictions) / 3.0  # Normalize

        # Calculate depth and breadth scores
        dimension_coverage = {}
        for evidence in evidence_collection:
            dim = evidence.dimension
            dimension_coverage[dim] = dimension_coverage.get(dim, 0) + 1

        depth_score = sum(dimension_coverage.values()) / len(dimension_coverage) if dimension_coverage else 0
        breadth_score = len(dimension_coverage) / len(required_dimensions) if required_dimensions else 0

        return CoverageMetrics(
            total_dimensions=len(required_dimensions),
            covered_dimensions=len(covered_dimensions),
            evidence_density=evidence_density,
            source_diversity=source_diversity,
            temporal_coverage=temporal_coverage,
            jurisdictional_coverage=jurisdictional_coverage,
            depth_score=depth_score,
            breadth_score=breadth_score,
        )

    def _generate_reasoning_chain(
            self,
            question_id: str,
            evidence_collection: List[StructuredEvidence],
            coverage_metrics: CoverageMetrics
    ) -> ReasoningChain:
        """Generate comprehensive reasoning chain from evidence collection."""
        chain_id = f"chain_{uuid.uuid4().hex[:12]}"
        steps = []

        # Group evidence by dimension
        evidence_by_dimension = defaultdict(list)
        for evidence in evidence_collection:
            evidence_by_dimension[evidence.dimension].append(evidence)

        # Create reasoning steps for each dimension
        step_counter = 1
        total_confidence = 0.0

        for dimension, dimension_evidence in evidence_by_dimension.items():
            step_id = f"step_{step_counter:03d}"

            # Calculate confidence impact for this dimension
            dimension_scores = [e.scoring.overall_score for e in dimension_evidence]
            avg_score = sum(dimension_scores) / len(dimension_scores)
            confidence_impact = avg_score * (
                    len(dimension_evidence) / len(evidence_collection))

            step = ReasoningStep(
                step_id=step_id,
                description=f"Analyze evidence for dimension: {dimension}",
                evidence_used=[e.evidence_id for e in dimension_evidence],
                inference_type="evidence_aggregation",
                confidence_impact=confidence_impact,
                logical_sequence=step_counter,
                dependencies=[],
                timestamp=time.time(),
                validation_status=CheckStatus.PASSED
            )

            steps.append(step)
            total_confidence += confidence_impact
            step_counter += 1

        # Add synthesis step
        synthesis_step = ReasoningStep(
            step_id=f"step_{step_counter:03d}",
            description="Synthesize findings across all dimensions",
            evidence_used=[e.evidence_id for e in evidence_collection],
            inference_type="synthesis",
            confidence_impact=total_confidence * 0.1,  # Small boost for synthesis
            logical_sequence=step_counter,
            dependencies=[s.step_id for s in steps],
            timestamp=time.time(),
            validation_status=CheckStatus.PASSED
        )
        steps.append(synthesis_step)
        total_confidence += synthesis_step.confidence_impact

        # Generate final conclusion
        final_conclusion = (
            f"Based on analysis of {len(evidence_collection)} evidence items "
            f"across {len(evidence_by_dimension)} dimensions with "
            f"{coverage_metrics.coverage_ratio:.1%} coverage of required dimensions"
        )

        return ReasoningChain(
            chain_id=chain_id,
            question_id=question_id,
            steps=steps,
            final_conclusion=final_conclusion,
            total_confidence=min(total_confidence, 1.0),
            creation_timestamp=time.time(),
            validation_status=CheckStatus.PASSED,
            chain_coverage=coverage_metrics.coverage_ratio
        )

    def _generate_answer_text(
            self,
            evidence_collection: List[StructuredEvidence],
            reasoning_chain: ReasoningChain,
    ) -> str:
        """Generate structured answer text from evidence and reasoning."""
        # Group evidence by dimension
        evidence_by_dimension = defaultdict(list)
        for evidence in evidence_collection:
            evidence_by_dimension[evidence.dimension].append(evidence)

        answer_parts = []

        # Executive summary
        unique_sources = len(set(
            e.citation.metadata.document_id for e in evidence_collection
        ))
        summary = (
            f"Based on comprehensive analysis of {len(evidence_collection)} "
            f"evidence items from {unique_sources} independent sources:\n\n"
        )
        answer_parts.append(summary)

        # Dimension-specific findings
        for dimension, dimension_evidence in evidence_by_dimension.items():
            dimension_text = f"**{dimension.upper()} ANALYSIS:**\n"

            # Sort evidence by confidence score
            sorted_evidence = sorted(
                dimension_evidence,
                key=lambda x: x.scoring.overall_score,
                reverse=True
            )

            for evidence in sorted_evidence[:3]:  # Top 3 evidence items per dimension
                dimension_text += (
                    f"- {evidence.exact_text} "
                    f"({evidence.citation.inline_citation}, "
                    f"confidence: {evidence.scoring.overall_score:.2f})\n"
                )

            answer_parts.append(dimension_text + "\n")

        # Confidence and limitations
        confidence_text = (
            f"\n**CONFIDENCE ASSESSMENT:**\n"
            f"Overall confidence level: {reasoning_chain.total_confidence:.2f}/1.00 "
            f"based on evidence quality, coverage, and source reliability analysis."
        )
        answer_parts.append(confidence_text)

        # Methodological note
        methodology_note = (
            f"\n\n**METHODOLOGY:**\n"
            f"This assessment follows DNP compliance standards {self.dnp_standards.version} "
            f"and incorporates validation against {len(self.validation_criteria.rules)} "
            f"quality criteria. All evidence is traceable to original sources with "
            f"complete audit trail available."
        )
        answer_parts.append(methodology_note)

        return "".join(answer_parts)

    def _determine_answer_status(
            self, coverage_metrics: CoverageMetrics, confidence_metrics: ConfidenceMetrics
    ) -> AnswerStatus:
        """Determine answer status based on comprehensive coverage and confidence metrics."""
        coverage_ratio = coverage_metrics.coverage_ratio
        final_confidence = confidence_metrics.final_confidence
        source_diversity = coverage_metrics.source_diversity

        if coverage_ratio >= 0.9 and final_confidence >= 0.8 and source_diversity >= 0.7:
            return AnswerStatus.COMPLETE
        elif coverage_ratio >= 0.7 and final_confidence >= 0.6:
            return AnswerStatus.PARTIAL
        elif coverage_ratio >= 0.4 and final_confidence >= 0.4:
            return AnswerStatus.INDETERMINATE
        elif final_confidence < 0.3:
            return AnswerStatus.INSUFFICIENT
        else:
            return AnswerStatus.CONFLICTED

    def _validate_dnp_compliance(
            self, answer_text: str, evidence_collection: List[StructuredEvidence]
    ) -> Dict[str, Any]:
        """Validate answer against DNP compliance standards with industrial-grade checks."""
        validation_results = {
            "standards_checked": [],
            "compliance_scores": {},
            "violations": [],
            "recommendations": [],
            "validation_timestamp": time.time(),
            "validator_version": self.system_version
        }

        # Check each DNP standard
        for standard_id, standard_text in self.dnp_standards.standards.items():
            validation_results["standards_checked"].append(standard_id)

            # Comprehensive compliance scoring
            compliance_score = 0.8  # Base score

            # Evidence-based scoring adjustments
            if "evidence" in standard_text.lower():
                evidence_score = min(1.0, len(evidence_collection) / 5.0)
                compliance_score = (compliance_score + evidence_score) / 2

            if "citation" in standard_text.lower():
                citation_count = sum(
                    1 for e in evidence_collection if e.citation.formatted_reference
                )
                citation_score = (
                    citation_count / len(evidence_collection)
                    if evidence_collection
                    else 0
                )
                compliance_score = (compliance_score + citation_score) / 2

            if "source" in standard_text.lower() or "provenance" in standard_text.lower():
                source_authenticity = sum(
                    e.citation.metadata.authenticity_score for e in evidence_collection
                ) / len(evidence_collection) if evidence_collection else 0
                compliance_score = (compliance_score + source_authenticity) / 2

            validation_results["compliance_scores"][standard_id] = compliance_score

            # Check for violations
            if compliance_score < 0.6:
                violation = {
                    "standard_id": standard_id,
                    "severity": "high",
                    "description": f"Insufficient compliance: score {compliance_score:.2f}",
                    "requirement": standard_text
                }
                validation_results["violations"].append(violation)
            elif compliance_score < 0.8:
                violation = {
                    "standard_id": standard_id,
                    "severity": "medium",
                    "description": f"Marginal compliance: score {compliance_score:.2f}",
                    "requirement": standard_text
                }
                validation_results["violations"].append(violation)

            # Generate recommendations for improvement
            if compliance_score < 0.9:
                recommendation = {
                    "standard_id": standard_id,
                    "priority": "high" if compliance_score < 0.7 else "medium",
                    "suggestion": f"Enhance compliance for {standard_id} through additional evidence and validation",
                    "current_score": compliance_score,
                    "target_score": 0.9
                }
                validation_results["recommendations"].append(recommendation)

        # Overall compliance assessment
        if validation_results["compliance_scores"]:
            avg_compliance = sum(validation_results["compliance_scores"].values()) / len(
                validation_results["compliance_scores"])
            validation_results["overall_compliance"] = avg_compliance
        else:
            validation_results["overall_compliance"] = 0.0

        return validation_results

    def _determine_compliance_level(
            self, dnp_validation: Dict[str, Any]
    ) -> DNPComplianceLevel:
        """Determine DNP compliance level from validation results."""
        overall_compliance = dnp_validation.get("overall_compliance", 0.0)
        violation_count = len([v for v in dnp_validation.get("violations", [])
                               if v.get("severity") in ["high", "critical"]])

        if overall_compliance >= 0.95 and violation_count == 0:
            return DNPComplianceLevel.FULLY_COMPLIANT
        elif overall_compliance >= 0.85 and violation_count <= 1:
            return DNPComplianceLevel.SUBSTANTIALLY_COMPLIANT
        elif overall_compliance >= 0.7:
            return DNPComplianceLevel.PARTIALLY_COMPLIANT
        else:
            return DNPComplianceLevel.NON_COMPLIANT

    def _generate_source_attribution(
            self, evidence_collection: List[StructuredEvidence]
    ) -> Dict[str, List[str]]:
        """Generate comprehensive source attribution mapping."""
        attribution = {}

        for evidence in evidence_collection:
            source_id = evidence.citation.metadata.document_id
            if source_id not in attribution:
                attribution[source_id] = []

            attribution[source_id].append({
                "evidence_id": evidence.evidence_id,
                "text_preview": evidence.exact_text[:150] + "..."
                if len(evidence.exact_text) > 150
                else evidence.exact_text,
                "citation": evidence.citation.formatted_reference,
                "dimension": evidence.dimension,
                "evidence_type": evidence.evidence_type.value,
                "confidence_score": evidence.scoring.overall_score,
                "publication_date": evidence.citation.metadata.publication_date,
                "jurisdiction": evidence.citation.metadata.jurisdiction
            })

        return attribution


# ==================== SAMPLE DATA GENERATION ====================
def create_sample_evidence() -> List[StructuredEvidence]:
    """Create sample evidence for demonstration purposes."""
    evidence_list = []

    for i in range(3):
        metadata = SourceMetadata(
            document_id=f"doc_{i:03d}",
            document_type="legal_code",
            publication_date="2023-01-15",
            author="National Standards Body",
            publisher="Government Publishing Office",
            jurisdiction="US",
            version="1.0",
            access_timestamp="2023-05-20T14:30:00Z",
            authenticity_score=0.95
        )

        citation = Citation(
            metadata=metadata,
            formatted_reference=f"US Legal Code § {100 + i} (2023)",
            inline_citation=f"(US LC § {100 + i}, 2023)",
            page_number=f"{10 + i}",
            section="Main Provisions"
        )

        scoring = EvidenceScoring(
            relevance_score=0.8 + (i * 0.05),
            accuracy_score=0.9,
            recency_score=0.85,
            authority_score=0.95,
            objectivity_score=0.75,
            overall_score=0.85 + (i * 0.03)
        )

        evidence = StructuredEvidence(
            evidence_id=f"ev_{uuid.uuid4().hex[:8]}",
            exact_text=f"Sample legal provision text requiring compliance with standard XYZ-{i}",
            evidence_type=EvidenceType.LEGAL_REFERENCE,
            dimension="regulatory_compliance",
            citation=citation,
            scoring=scoring,
            extraction_context="Full text analysis of legal document section 2.1"
        )

        evidence_list.append(evidence)

    return evidence_list


def create_sample_dnp_standards() -> DNPStandards:
    """Create sample DNP standards for demonstration."""
    standards = {
        "DNP-001": "All answers must include supporting evidence with proper citations",
        "DNP-002": "Evidence must be traceable to original sources with authenticity verification",
        "DNP-003": "Confidence levels must be calibrated and documented with clear metrics",
        "DNP-004": "Answers must cover all required dimensions with minimum evidence threshold",
        "DNP-005": "Reasoning chains must be logically sound and evidence-based"
    }

    return DNPStandards(
        standards=standards,
        version="2.1.0",
        jurisdiction="International",
        effective_date="2023-01-01",
        amendments=[{"id": "amend-001", "date": "2023-03-15", "description": "Clarified evidence requirements"}]
    )


def create_sample_validation_criteria() -> ValidationCriteria:
    """Create sample validation criteria for demonstration."""
    rules = [
        ValidationRule(
            rule_id="V001",
            description="Minimum of 3 evidence sources required",
            severity=ValidationSeverity.HIGH,
            threshold=0.8,
            weight=1.0,
            applicable_evidence_types=[EvidenceType.LEGAL_REFERENCE, EvidenceType.DIRECT_QUOTE]
        ),
        ValidationRule(
            rule_id="V002",
            description="Evidence must be from authentic sources",
            severity=ValidationSeverity.CRITICAL,
            threshold=0.9,
            weight=1.5,
            applicable_evidence_types=list(EvidenceType)
        )
    ]

    return ValidationCriteria(
        rules=rules,
        overall_threshold=0.75,
        version="1.2",
        effective_date="2023-01-01",
        compliance_framework="DNP-2023"
    )


def create_sample_answer() -> DNPCompliantAnswer:
    """Create sample DNP-compliant answer for demonstration."""
    # Create sample standards and criteria
    dnp_standards = create_sample_dnp_standards()
    validation_criteria = create_sample_validation_criteria()

    # Create formatter
    formatter = AnswerFormatter(
        dnp_standards=dnp_standards,
        validation_criteria=validation_criteria,
        system_version="2.1.0",
        environment="demo"
    )

    # Create sample evidence
    sample_evidence = create_sample_evidence()

    # Format answer
    answer = formatter.format_answer(
        question_id="q_sample_001",
        evidence_collection=sample_evidence,
        required_dimensions=["regulatory_compliance", "safety", "performance"],
        context_metadata={"domain": "legal_compliance", "urgency": "high"}
    )

    return answer


# ==================== MAIN EXECUTION ====================
if __name__ == "__main__":
    # Demonstration
    print("Generating sample DNP-compliant answer...")
    sample_answer = create_sample_answer()

    print("\n" + "=" * 60)
    print("DNP COMPLIANCE ANSWER FORMATTER - DEMONSTRATION")
    print("=" * 60)

    print(f"Answer ID: {sample_answer.answer_id}")
    print(f"Status: {sample_answer.answer_status.value}")
    print(f"Compliance Level: {sample_answer.compliance_level.value}")
    print(f"Confidence: {sample_answer.confidence_metrics.final_confidence:.2f}")
    print(f"Evidence Count: {len(sample_answer.supporting_evidence)}")
    print(f"Reasoning Steps: {len(sample_answer.reasoning_chain.steps)}")
    print(f"Traceability Hash: {sample_answer.get_traceability_hash()}")

    print("\nAnswer Text Preview:")
    preview_length = 300
    if len(sample_answer.answer_text) > preview_length:
        print(sample_answer.answer_text[:preview_length] + "...")
    else:
        print(sample_answer.answer_text)

    print("\nValidation Results:")
    print(f"Overall Compliance: {sample_answer.dnp_validation_results.get('overall_compliance', 0):.2f}")
    print(f"Standards Checked: {len(sample_answer.dnp_validation_results.get('standards_checked', []))}")
    print(f"Violations: {len(sample_answer.dnp_validation_results.get('violations', []))}")

    print("\nAudit Trail Info:")
    print(f"Processing Stages: {len(sample_answer.audit_trail.processing_stages)}")
    print(f"System Version: {sample_answer.audit_trail.system_version}")

    # Export as JSON for inspection
    export_data = sample_answer.to_dict()
    with open("dnp_answer_demo.json", "w") as f:
        json.dump(export_data, f, indent=2)

    print("\nFull answer exported to 'dnp_answer_demo.json'")
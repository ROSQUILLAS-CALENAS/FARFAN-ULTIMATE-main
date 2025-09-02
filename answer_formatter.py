"""
AnswerFormatter - DNP Compliance Answer Formatting System

Transforms aggregated evidence objects into structured answer format using DNP compliance
rules for content validation, implements confidence calibration based on evidence quality
and coverage metrics, and generates audit-ready output with source traceability and
reasoning chain documentation.

Dependencies: evidence_processor, evidence_validation_model, normative_validator
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import uuid4

# Import from existing modules
try:
    from evidence_processor import (
        ConfidenceLevel,
        EvidenceType,
        SourceMetadata,
        StructuredEvidence,
    )
    from evidence_validation_model import (
        DNPStandards,
        ValidationCriteria,
        ValidationRule,
        ValidationSeverity,
    )
    from normative_validator import CheckStatus, ComplianceLevel, ViolationSeverity
except ImportError as e:
    print(
        f"Warning: Import error {e}. Running in standalone mode with minimal dependencies."
    )

    # Minimal fallback definitions
    class EvidenceType(str, Enum):
        DIRECT_QUOTE = "direct_quote"

    class ConfidenceLevel(str, Enum):
        HIGH = "high"
        MEDIUM = "medium"
        LOW = "low"

    class ValidationSeverity(str, Enum):
        HIGH = "high"
        MEDIUM = "medium"
        LOW = "low"

    class ComplianceLevel(str, Enum):
        CUMPLE = "CUMPLE"

    class CheckStatus(str, Enum):
        PASSED = "PASSED"

    class ViolationSeverity(str, Enum):
        CRITICAL = "critical"


class AnswerStatus(str, Enum):
    """Status levels for formatted answers."""

    COMPLETE = "complete"
    PARTIAL = "partial"
    INSUFFICIENT = "insufficient"
    CONFLICTED = "conflicted"


class DNPComplianceLevel(str, Enum):
    """DNP-specific compliance levels."""

    FULLY_COMPLIANT = "fully_compliant"
    SUBSTANTIALLY_COMPLIANT = "substantially_compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    NON_COMPLIANT = "non_compliant"


@dataclass
class CoverageMetrics:
    """Metrics for evidence coverage analysis."""

    total_dimensions: int
    covered_dimensions: int
    evidence_density: float
    source_diversity: float
    temporal_coverage: float

    @property
    def coverage_ratio(self) -> float:
        """Calculate coverage ratio."""
        return (
            self.covered_dimensions / self.total_dimensions
            if self.total_dimensions > 0
            else 0.0
        )


@dataclass
class ConfidenceMetrics:
    """Comprehensive confidence metrics for answers."""

    base_confidence: float
    coverage_adjustment: float
    quality_adjustment: float
    consistency_adjustment: float
    final_confidence: float
    calibration_factors: Dict[str, float] = field(default_factory=dict)


@dataclass
class ReasoningStep:
    """Individual step in reasoning chain."""

    step_id: str
    description: str
    evidence_used: List[str]
    inference_type: str
    confidence_impact: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ReasoningChain:
    """Complete reasoning chain with audit trail."""

    chain_id: str
    question_id: str
    steps: List[ReasoningStep]
    final_conclusion: str
    total_confidence: float
    creation_timestamp: datetime = field(default_factory=datetime.now)

    def get_evidence_lineage(self) -> List[str]:
        """Get all evidence IDs used in reasoning chain."""
        evidence_ids = []
        for step in self.steps:
            evidence_ids.extend(step.evidence_used)
        return list(set(evidence_ids))


@dataclass
class AuditTrail:
    """Comprehensive audit trail for answer generation."""

    trail_id: str
    answer_id: str
    processing_stages: List[Dict[str, Any]]
    evidence_sources: List[str]
    validation_checks: List[Dict[str, Any]]
    compliance_assessments: List[Dict[str, Any]]
    quality_metrics: Dict[str, float]
    timestamp: datetime = field(default_factory=datetime.now)

    def add_stage(self, stage_name: str, details: Dict[str, Any]):
        """Add processing stage to audit trail."""
        stage_entry = {
            "stage": stage_name,
            "timestamp": datetime.now().isoformat(),
            "details": details,
        }
        self.processing_stages.append(stage_entry)


@dataclass
class DNPCompliantAnswer:
    """DNP-compliant structured answer with full traceability."""

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
    metadata: Dict[str, Any] = field(default_factory=dict)
    creation_timestamp: datetime = field(default_factory=datetime.now)

    def get_citation_text(self) -> str:
        """Generate formatted citation text for the answer."""
        citations = []
        for evidence in self.supporting_evidence:
            citations.append(evidence.citation.inline_citation)
        return "; ".join(citations)

    def get_traceability_hash(self) -> str:
        """Generate hash for complete traceability."""
        components = [
            self.answer_id,
            self.question_id,
            str(sorted([e.evidence_id for e in self.supporting_evidence])),
            self.reasoning_chain.chain_id,
            self.audit_trail.trail_id,
        ]
        content = "|".join(components)
        return hashlib.sha256(content.encode()).hexdigest()


class AnswerFormatter:
    """
    Transforms aggregated evidence into structured, DNP-compliant answers with full audit trails.
    """

    def __init__(
        self, dnp_standards: DNPStandards, validation_criteria: ValidationCriteria
    ):
        """
        Initialize AnswerFormatter with DNP standards and validation criteria.

        Args:
            dnp_standards: DNP standards for compliance validation
            validation_criteria: Validation criteria for answer quality
        """
        self.dnp_standards = dnp_standards
        self.validation_criteria = validation_criteria
        self.confidence_calibrator = ConfidenceCalibrator()
        self.audit_generator = AuditTrailGenerator()

    def format_answer(
        self,
        question_id: str,
        evidence_collection: List[StructuredEvidence],
        required_dimensions: List[str],
        context_metadata: Optional[Dict[str, Any]] = None,
    ) -> DNPCompliantAnswer:
        """
        Format aggregated evidence into DNP-compliant structured answer.

        Args:
            question_id: Unique identifier for the question
            evidence_collection: Collection of structured evidence
            required_dimensions: Required dimensions for complete answer
            context_metadata: Additional context metadata

        Returns:
            DNP-compliant structured answer with full traceability
        """
        answer_id = f"ans_{uuid4().hex[:8]}"

        # Initialize audit trail
        audit_trail = self.audit_generator.create_audit_trail(answer_id, question_id)
        audit_trail.add_stage(
            "initialization",
            {
                "evidence_count": len(evidence_collection),
                "required_dimensions": required_dimensions,
                "formatter_version": "1.0",
            },
        )

        # Analyze evidence coverage
        coverage_metrics = self._calculate_coverage_metrics(
            evidence_collection, required_dimensions
        )
        audit_trail.add_stage(
            "coverage_analysis",
            {
                "coverage_ratio": coverage_metrics.coverage_ratio,
                "evidence_density": coverage_metrics.evidence_density,
                "source_diversity": coverage_metrics.source_diversity,
            },
        )

        # Generate reasoning chain
        reasoning_chain = self._generate_reasoning_chain(
            question_id, evidence_collection
        )
        audit_trail.add_stage(
            "reasoning_generation",
            {
                "chain_id": reasoning_chain.chain_id,
                "steps_count": len(reasoning_chain.steps),
                "evidence_used": len(reasoning_chain.get_evidence_lineage()),
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
            },
        )

        # Generate answer text
        answer_text = self._generate_answer_text(evidence_collection, reasoning_chain)
        audit_trail.add_stage(
            "answer_generation",
            {
                "text_length": len(answer_text),
                "evidence_integrated": len(evidence_collection),
            },
        )

        # Determine answer status
        answer_status = self._determine_answer_status(
            coverage_metrics, confidence_metrics
        )

        # Validate DNP compliance
        dnp_validation = self._validate_dnp_compliance(answer_text, evidence_collection)
        compliance_level = self._determine_compliance_level(dnp_validation)
        audit_trail.add_stage("dnp_validation", dnp_validation)

        # Generate source attribution
        source_attribution = self._generate_source_attribution(evidence_collection)

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
            metadata=context_metadata or {},
        )

        audit_trail.add_stage(
            "completion",
            {
                "answer_id": answer_id,
                "traceability_hash": answer.get_traceability_hash(),
                "final_status": answer_status.value,
            },
        )

        return answer

    def _calculate_coverage_metrics(
        self,
        evidence_collection: List[StructuredEvidence],
        required_dimensions: List[str],
    ) -> CoverageMetrics:
        """Calculate coverage metrics for evidence collection."""
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
        publication_dates = [
            evidence.citation.metadata.publication_date
            for evidence in evidence_collection
            if evidence.citation.metadata.publication_date
        ]

        temporal_coverage = 1.0  # Default high coverage
        if len(publication_dates) > 1:
            date_range = max(publication_dates) - min(publication_dates)
            temporal_coverage = min(1.0, date_range.days / 365.0)  # Normalize to years

        return CoverageMetrics(
            total_dimensions=len(required_dimensions),
            covered_dimensions=len(covered_dimensions),
            evidence_density=evidence_density,
            source_diversity=source_diversity,
            temporal_coverage=temporal_coverage,
        )

    def _generate_reasoning_chain(
        self, question_id: str, evidence_collection: List[StructuredEvidence]
    ) -> ReasoningChain:
        """Generate reasoning chain from evidence collection."""
        chain_id = f"chain_{uuid4().hex[:8]}"
        steps = []

        # Group evidence by dimension
        evidence_by_dimension = {}
        for evidence in evidence_collection:
            if evidence.dimension not in evidence_by_dimension:
                evidence_by_dimension[evidence.dimension] = []
            evidence_by_dimension[evidence.dimension].append(evidence)

        # Create reasoning steps for each dimension
        step_counter = 1
        total_confidence = 0.0

        for dimension, dimension_evidence in evidence_by_dimension.items():
            step_id = f"step_{step_counter:02d}"

            # Calculate confidence impact for this dimension
            dimension_scores = [e.scoring.overall_score for e in dimension_evidence]
            avg_score = sum(dimension_scores) / len(dimension_scores)
            confidence_impact = avg_score * (
                len(dimension_evidence) / len(evidence_collection)
            )

            step = ReasoningStep(
                step_id=step_id,
                description=f"Analyze evidence for dimension: {dimension}",
                evidence_used=[e.evidence_id for e in dimension_evidence],
                inference_type="evidence_aggregation",
                confidence_impact=confidence_impact,
            )

            steps.append(step)
            total_confidence += confidence_impact
            step_counter += 1

        # Add synthesis step
        synthesis_step = ReasoningStep(
            step_id=f"step_{step_counter:02d}",
            description="Synthesize findings across all dimensions",
            evidence_used=[e.evidence_id for e in evidence_collection],
            inference_type="synthesis",
            confidence_impact=total_confidence * 0.1,  # Small boost for synthesis
        )
        steps.append(synthesis_step)
        total_confidence += synthesis_step.confidence_impact

        # Generate final conclusion
        final_conclusion = f"Based on analysis of {len(evidence_collection)} evidence items across {len(evidence_by_dimension)} dimensions"

        return ReasoningChain(
            chain_id=chain_id,
            question_id=question_id,
            steps=steps,
            final_conclusion=final_conclusion,
            total_confidence=min(total_confidence, 1.0),
        )

    def _generate_answer_text(
        self,
        evidence_collection: List[StructuredEvidence],
        reasoning_chain: ReasoningChain,
    ) -> str:
        """Generate structured answer text from evidence and reasoning."""
        # Group evidence by dimension
        evidence_by_dimension = {}
        for evidence in evidence_collection:
            if evidence.dimension not in evidence_by_dimension:
                evidence_by_dimension[evidence.dimension] = []
            evidence_by_dimension[evidence.dimension].append(evidence)

        answer_parts = []

        # Executive summary
        summary = f"Based on analysis of {len(evidence_collection)} evidence items from {len(set(e.citation.metadata.document_id for e in evidence_collection))} sources:"
        answer_parts.append(summary)

        # Dimension-specific findings
        for dimension, dimension_evidence in evidence_by_dimension.items():
            dimension_text = f"\n**{dimension.title()}:**\n"

            # Sort evidence by confidence score
            sorted_evidence = sorted(
                dimension_evidence, key=lambda x: x.scoring.overall_score, reverse=True
            )

            for evidence in sorted_evidence[:3]:  # Top 3 evidence items per dimension
                dimension_text += (
                    f"- {evidence.exact_text} {evidence.citation.inline_citation}\n"
                )

            answer_parts.append(dimension_text)

        # Confidence and limitations
        confidence_text = f"\n**Confidence Level:** {reasoning_chain.total_confidence:.2f} based on evidence quality and coverage analysis."
        answer_parts.append(confidence_text)

        return "".join(answer_parts)

    def _determine_answer_status(
        self, coverage_metrics: CoverageMetrics, confidence_metrics: ConfidenceMetrics
    ) -> AnswerStatus:
        """Determine answer status based on coverage and confidence."""
        coverage_ratio = coverage_metrics.coverage_ratio
        final_confidence = confidence_metrics.final_confidence

        if coverage_ratio >= 0.9 and final_confidence >= 0.8:
            return AnswerStatus.COMPLETE
        elif coverage_ratio >= 0.6 and final_confidence >= 0.6:
            return AnswerStatus.PARTIAL
        elif final_confidence < 0.4:
            return AnswerStatus.INSUFFICIENT
        else:
            return AnswerStatus.CONFLICTED

    def _validate_dnp_compliance(
        self, answer_text: str, evidence_collection: List[StructuredEvidence]
    ) -> Dict[str, Any]:
        """Validate answer against DNP compliance standards."""
        validation_results = {
            "standards_checked": [],
            "compliance_scores": {},
            "violations": [],
            "recommendations": [],
        }

        # Check each DNP standard
        for standard_id, standard_text in self.dnp_standards.standards.items():
            validation_results["standards_checked"].append(standard_id)

            # Simple compliance check (in production, this would be more sophisticated)
            compliance_score = 0.8  # Default high compliance

            # Check for required elements based on standard
            if "evidence" in standard_text.lower():
                evidence_score = (
                    len(evidence_collection) / 10.0
                )  # Normalize to 10 evidence items
                compliance_score *= min(1.0, evidence_score)

            if "citation" in standard_text.lower():
                citation_count = sum(
                    1 for e in evidence_collection if e.citation.formatted_reference
                )
                citation_score = (
                    citation_count / len(evidence_collection)
                    if evidence_collection
                    else 0
                )
                compliance_score *= citation_score

            validation_results["compliance_scores"][standard_id] = compliance_score

            # Check for violations
            if compliance_score < 0.7:
                violation = {
                    "standard_id": standard_id,
                    "severity": "moderate",
                    "description": f"Low compliance score: {compliance_score:.2f}",
                }
                validation_results["violations"].append(violation)

        # Overall compliance assessment
        avg_compliance = (
            sum(validation_results["compliance_scores"].values())
            / len(validation_results["compliance_scores"])
            if validation_results["compliance_scores"]
            else 0
        )
        validation_results["overall_compliance"] = avg_compliance

        return validation_results

    def _determine_compliance_level(
        self, dnp_validation: Dict[str, Any]
    ) -> DNPComplianceLevel:
        """Determine DNP compliance level from validation results."""
        overall_compliance = dnp_validation.get("overall_compliance", 0.0)
        violation_count = len(dnp_validation.get("violations", []))

        if overall_compliance >= 0.95 and violation_count == 0:
            return DNPComplianceLevel.FULLY_COMPLIANT
        elif overall_compliance >= 0.8 and violation_count <= 1:
            return DNPComplianceLevel.SUBSTANTIALLY_COMPLIANT
        elif overall_compliance >= 0.6:
            return DNPComplianceLevel.PARTIALLY_COMPLIANT
        else:
            return DNPComplianceLevel.NON_COMPLIANT

    def _generate_source_attribution(
        self, evidence_collection: List[StructuredEvidence]
    ) -> Dict[str, List[str]]:
        """Generate source attribution mapping."""
        attribution = {}

        for evidence in evidence_collection:
            source_id = evidence.citation.metadata.document_id
            if source_id not in attribution:
                attribution[source_id] = []

            attribution[source_id].append(
                {
                    "evidence_id": evidence.evidence_id,
                    "text": evidence.exact_text[:100] + "..."
                    if len(evidence.exact_text) > 100
                    else evidence.exact_text,
                    "citation": evidence.citation.formatted_reference,
                    "dimension": evidence.dimension,
                }
            )

        return attribution


class ConfidenceCalibrator:
    """Calibrates confidence scores based on evidence quality and coverage."""

    def calculate_confidence(
        self,
        evidence_collection: List[StructuredEvidence],
        coverage_metrics: CoverageMetrics,
        reasoning_chain: ReasoningChain,
    ) -> ConfidenceMetrics:
        """Calculate calibrated confidence metrics."""
        # Base confidence from evidence scores
        evidence_scores = [e.scoring.overall_score for e in evidence_collection]
        base_confidence = (
            sum(evidence_scores) / len(evidence_scores) if evidence_scores else 0.0
        )

        # Coverage adjustment
        coverage_adjustment = coverage_metrics.coverage_ratio * 0.2  # Up to 20% boost

        # Quality adjustment based on evidence diversity
        quality_factors = []
        quality_factors.append(coverage_metrics.source_diversity)
        quality_factors.append(coverage_metrics.temporal_coverage)
        quality_factors.append(
            min(1.0, len(evidence_collection) / 5.0)
        )  # Evidence quantity factor

        quality_adjustment = (
            sum(quality_factors) / len(quality_factors) * 0.15
        )  # Up to 15% boost

        # Consistency adjustment (simplified - based on reasoning chain confidence)
        consistency_adjustment = (
            reasoning_chain.total_confidence * 0.1
        )  # Up to 10% boost

        # Final confidence calculation
        final_confidence = min(
            1.0,
            base_confidence
            + coverage_adjustment
            + quality_adjustment
            + consistency_adjustment,
        )

        return ConfidenceMetrics(
            base_confidence=base_confidence,
            coverage_adjustment=coverage_adjustment,
            quality_adjustment=quality_adjustment,
            consistency_adjustment=consistency_adjustment,
            final_confidence=final_confidence,
            calibration_factors={
                "evidence_count": len(evidence_collection),
                "source_diversity": coverage_metrics.source_diversity,
                "coverage_ratio": coverage_metrics.coverage_ratio,
                "reasoning_steps": len(reasoning_chain.steps),
            },
        )


class AuditTrailGenerator:
    """Generates comprehensive audit trails for answer formatting."""

    def create_audit_trail(self, answer_id: str, question_id: str) -> AuditTrail:
        """Create new audit trail for answer formatting process."""
        trail_id = f"trail_{uuid4().hex[:8]}"

        return AuditTrail(
            trail_id=trail_id,
            answer_id=answer_id,
            processing_stages=[],
            evidence_sources=[],
            validation_checks=[],
            compliance_assessments=[],
            quality_metrics={},
        )


def create_sample_answer() -> DNPCompliantAnswer:
    """Create sample DNP-compliant answer for demonstration."""
    try:
        from evidence_processor import create_sample_evidence
        from evidence_validation_model import (
            DNPStandards,
            ValidationCriteria,
            ValidationRule,
        )

        # Create sample DNP standards
        sample_standards = {
            "DNP-001": "All answers must include supporting evidence with proper citations",
            "DNP-002": "Evidence must be traceable to original sources",
            "DNP-003": "Confidence levels must be clearly documented",
        }

        dnp_standards = DNPStandards(standards=sample_standards)

        # Create sample validation criteria
        rules = [
            ValidationRule(
                rule_id="rule_001",
                description="Minimum evidence requirement",
                severity=ValidationSeverity.HIGH,
                prior_probability=0.8,
                threshold=0.7,
            )
        ]

        validation_criteria = ValidationCriteria(rules=frozenset(rules))

        # Create formatter
        formatter = AnswerFormatter(dnp_standards, validation_criteria)

        # Create sample evidence
        sample_evidence = create_sample_evidence()

        # Format answer
        answer = formatter.format_answer(
            question_id="q_sample_001",
            evidence_collection=sample_evidence,
            required_dimensions=["accuracy", "reliability", "performance"],
        )

        return answer

    except ImportError:
        print("Cannot create sample answer: Required dependencies not available")
        return None


if __name__ == "__main__":
    # Demonstration
    sample_answer = create_sample_answer()

    print(f"Answer ID: {sample_answer.answer_id}")
    print(f"Status: {sample_answer.answer_status.value}")
    print(f"Compliance Level: {sample_answer.compliance_level.value}")
    print(f"Confidence: {sample_answer.confidence_metrics.final_confidence:.2f}")
    print(f"Evidence Count: {len(sample_answer.supporting_evidence)}")
    print(f"Reasoning Steps: {len(sample_answer.reasoning_chain.steps)}")
    print(f"Traceability Hash: {sample_answer.get_traceability_hash()}")
    print(f"Citations: {sample_answer.get_citation_text()}")
    print("\nAnswer Text Preview:")
    print(
        sample_answer.answer_text[:200] + "..."
        if len(sample_answer.answer_text) > 200
        else sample_answer.answer_text
    )

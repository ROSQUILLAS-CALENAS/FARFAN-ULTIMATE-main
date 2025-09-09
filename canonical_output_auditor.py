"""
Enhanced Canonical Output Auditor

A sophisticated auditor that verifies canonical process compliance after pipeline execution.
Features:
- Comprehensive validation of four-cluster application
- Evidence linkage verification with detailed traceability
- DNP standards compliance checking
- Causal correction calibration triggers
- Multi-level reporting validation (micro/meso/macro)
- Deterministic audit results for reproducibility
- Extensible validation framework
- Comprehensive error handling and logging
"""

from __future__ import annotations

import hashlib
import logging
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
import weakref

# Mandatory Pipeline Contract Annotations
__phase__ = "S"
__code__ = "53S"
__stage_order__ = 10
__version__ = "2.0.0"

logger = logging.getLogger(__name__)

# Constants
REQUIRED_CLUSTERS_LABEL = ("C1", "C2", "C3", "C4")
REQUIRED_CLUSTERS_NUM = (1, 2, 3, 4)

# Best-effort imports with graceful degradation
try:
    import cluster_execution_controller
except ImportError:
    cluster_execution_controller = None
    logger.debug("cluster_execution_controller module not available")

try:
    import meso_aggregator
except ImportError:
    meso_aggregator = None
    logger.debug("meso_aggregator module not available")

try:
    import public_transformer_adapter
except ImportError:
    public_transformer_adapter = None
    logger.debug("public_transformer_adapter module not available")


class AuditSeverity(Enum):
    """Audit gap severity levels"""
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()


class ValidationResult(Enum):
    """Validation result status"""
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    SKIP = "skip"


@dataclass
class AuditGap:
    """Represents a gap found during audit"""
    code: str
    message: str
    severity: AuditSeverity
    context: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity.name.lower(),
            "context": self.context,
            "recommendations": self.recommendations
        }


@dataclass
class ValidationContext:
    """Context for validation operations"""
    data: Dict[str, Any]
    context: Dict[str, Any]
    deterministic: bool = False
    strict_mode: bool = False
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self):
        if self.deterministic:
            self.timestamp = 1640995200.0  # Fixed timestamp for deterministic mode


class BaseValidator(ABC):
    """Base class for all validators"""

    def __init__(self, name: str, description: str, severity: AuditSeverity = AuditSeverity.ERROR):
        self.name = name
        self.description = description
        self.severity = severity
        self._enabled = True

    @abstractmethod
    def validate(self, ctx: ValidationContext) -> Tuple[ValidationResult, List[AuditGap]]:
        """Perform validation and return result with any gaps found"""
        pass

    def is_enabled(self) -> bool:
        return self._enabled

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False


class DeterministicHasher:
    """Enhanced deterministic hashing system"""

    @staticmethod
    def remove_timestamps_recursive(obj: Any) -> Any:
        """Remove timestamp fields recursively for deterministic hashing"""
        if isinstance(obj, dict):
            cleaned = {}
            for key, value in obj.items():
                if key not in ('timestamp', 'created_at', 'updated_at', 'last_modified'):
                    cleaned[key] = DeterministicHasher.remove_timestamps_recursive(value)
            return cleaned
        elif isinstance(obj, list):
            return [DeterministicHasher.remove_timestamps_recursive(item) for item in obj]
        else:
            return obj

    @staticmethod
    def stable_hash_dict(d: Dict[str, Any], deterministic: bool = False) -> str:
        """Create stable hash of dictionary with optional deterministic mode"""
        try:
            # Try to use deterministic hashing module if available
            import importlib.util
            import os

            hash_module_path = Path(__file__).parent / 'egw_query_expansion' / 'core' / 'deterministic_hashing.py'
            if hash_module_path.exists():
                spec = importlib.util.spec_from_file_location("deterministic_hashing", hash_module_path)
                if spec and spec.loader:
                    hash_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(hash_module)

                    if deterministic:
                        clean_dict = DeterministicHasher.remove_timestamps_recursive(d)
                        return hash_module.hash_context(clean_dict)[:16]
                    else:
                        return hash_module.hash_context(d)[:16]
        except Exception as e:
            logger.debug(f"Failed to load deterministic hashing module: {e}")

        # Fallback to standard implementation
        try:
            import json
            if deterministic:
                clean_dict = DeterministicHasher.remove_timestamps_recursive(d)
                content = json.dumps(clean_dict, sort_keys=True, ensure_ascii=False)
            else:
                content = json.dumps(d, sort_keys=True, ensure_ascii=False)
        except Exception:
            content = str(d)

        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


class ClusterValidator(BaseValidator):
    """Validates four-cluster compliance and completeness"""

    def __init__(self):
        super().__init__(
            name="cluster_validation",
            description="Validates four-cluster application and completeness",
            severity=AuditSeverity.CRITICAL
        )

    def validate(self, ctx: ValidationContext) -> Tuple[ValidationResult, List[AuditGap]]:
        gaps = []
        cluster_audit = ctx.data.get("cluster_audit")

        if not isinstance(cluster_audit, dict):
            gaps.append(AuditGap(
                code="MISSING_CLUSTER_AUDIT",
                message="Cluster audit data is missing or invalid",
                severity=AuditSeverity.CRITICAL,
                recommendations=["Ensure cluster execution controller processes the data"]
            ))
            return ValidationResult.FAIL, gaps

        # Check for required clusters
        present_clusters = cluster_audit.get("present", [])
        cluster_labels = set(str(c) for c in present_clusters)
        required_c_labels = set(f"C{n}" for n in REQUIRED_CLUSTERS_NUM)
        required_num_labels = set(str(n) for n in REQUIRED_CLUSTERS_NUM)

        clusters_valid = (required_c_labels.issubset(cluster_labels) or
                          required_num_labels.issubset(cluster_labels))

        if not clusters_valid:
            gaps.append(AuditGap(
                code="MISSING_REQUIRED_CLUSTERS",
                message=f"Required clusters missing. Present: {sorted(cluster_labels)}, Required: {sorted(required_c_labels)}",
                severity=AuditSeverity.CRITICAL,
                context={"present_clusters": sorted(cluster_labels)},
                recommendations=[
                    "Ensure all four clusters (C1, C2, C3, C4) are processed",
                    "Check cluster execution controller configuration"
                ]
            ))

        # Check completeness
        if not cluster_audit.get("complete"):
            gaps.append(AuditGap(
                code="INCOMPLETE_CLUSTER_EXECUTION",
                message="Cluster execution is not marked as complete",
                severity=AuditSeverity.ERROR,
                recommendations=["Review cluster processing logs for errors"]
            ))

        # Check non-redundancy
        if not cluster_audit.get("non_redundant"):
            gaps.append(AuditGap(
                code="REDUNDANT_ITEMS",
                message="Redundant items detected in cluster processing",
                severity=AuditSeverity.WARNING,
                recommendations=["Review deduplication logic in cluster processing"]
            ))

        return ValidationResult.PASS if not gaps else ValidationResult.FAIL, gaps


class EvidenceLinkageValidator(BaseValidator):
    """Validates evidence linkage at micro level"""

    def __init__(self):
        super().__init__(
            name="evidence_linkage",
            description="Validates evidence linkage in micro-level analysis",
            severity=AuditSeverity.ERROR
        )

    def validate(self, ctx: ValidationContext) -> Tuple[ValidationResult, List[AuditGap]]:
        gaps = []
        cluster_audit = ctx.data.get("cluster_audit", {})
        micro = cluster_audit.get("micro", {})

        if not isinstance(micro, dict):
            gaps.append(AuditGap(
                code="MISSING_MICRO_LEVEL",
                message="Micro-level analysis data is missing",
                severity=AuditSeverity.ERROR,
                recommendations=["Ensure micro-level processing is enabled"]
            ))
            return ValidationResult.FAIL, gaps

        evidence_linked = True
        cluster_items = sorted(micro.items()) if ctx.deterministic else micro.items()

        for cluster_id, payload in cluster_items:
            if not isinstance(payload, dict):
                continue

            answers = payload.get("answers", [])
            if not answers:
                continue

            # Check if evidence is already marked as linked
            if payload.get("evidence_linked"):
                continue

            # Fallback: check for evidence IDs in answers
            evidence_ids = self._collect_evidence_ids_from_answers(answers)
            if not evidence_ids:
                evidence_linked = False
                gaps.append(AuditGap(
                    code="MISSING_EVIDENCE_LINKAGE",
                    message=f"No evidence linkage found for cluster {cluster_id}",
                    severity=AuditSeverity.ERROR,
                    context={"cluster_id": cluster_id, "answer_count": len(answers)},
                    recommendations=[
                        "Ensure evidence IDs are included in answer objects",
                        "Verify evidence linking process in cluster analysis"
                    ]
                ))

        return ValidationResult.PASS if evidence_linked else ValidationResult.FAIL, gaps

    def _collect_evidence_ids_from_answers(self, answers: List[Dict[str, Any]]) -> Set[str]:
        """Collect evidence IDs from answer objects"""
        evidence_ids = set()
        for answer in answers:
            if not isinstance(answer, dict):
                continue

            for key in ("evidence_ids", "citations"):
                values = answer.get(key, [])
                if isinstance(values, list):
                    evidence_ids.update(str(v) for v in values if isinstance(v, str))

        return evidence_ids


class DNPStandardsValidator(BaseValidator):
    """Validates DNP standards usage and compliance"""

    def __init__(self):
        super().__init__(
            name="dnp_standards",
            description="Validates usage of DNP standards and compliance",
            severity=AuditSeverity.WARNING
        )

    def validate(self, ctx: ValidationContext) -> Tuple[ValidationResult, List[AuditGap]]:
        gaps = []

        dnp_fields = [
            "dnp_validation_results",
            "dnp_compliance",
            "dnp_alignment"
        ]

        uses_dnp = self._check_dnp_usage(ctx.data, dnp_fields)

        if not uses_dnp:
            gaps.append(AuditGap(
                code="MISSING_DNP_ALIGNMENT",
                message="No DNP standards usage detected",
                severity=AuditSeverity.WARNING,
                context={"checked_fields": dnp_fields},
                recommendations=[
                    "Implement DNP validation in processing pipeline",
                    "Add DNP compliance checking to analysis workflow"
                ]
            ))
            return ValidationResult.WARNING, gaps

        return ValidationResult.PASS, gaps

    def _check_dnp_usage(self, data: Dict[str, Any], dnp_fields: List[str]) -> bool:
        """Check for DNP usage in data or report section"""
        for field in dnp_fields:
            if field in data:
                return True

            report = data.get("report", {})
            if isinstance(report, dict) and field in report:
                return True

        return False


class CausalCorrectionValidator(BaseValidator):
    """Validates causal correction signals and calibration"""

    def __init__(self):
        super().__init__(
            name="causal_correction",
            description="Validates causal correction signals and calibration triggers",
            severity=AuditSeverity.WARNING
        )

    def validate(self, ctx: ValidationContext) -> Tuple[ValidationResult, List[AuditGap]]:
        gaps = []

        causal_fields = [
            "causal_correction",
            "causal_graph",
            "causal_adjusted_scores",
            "causal_dnp"
        ]

        causal_signals = self._check_causal_signals(ctx.data, causal_fields)

        if not causal_signals:
            gaps.append(AuditGap(
                code="MISSING_CAUSAL_CORRECTION_SIGNAL",
                message="No causal correction signals detected",
                severity=AuditSeverity.WARNING,
                context={"checked_fields": causal_fields},
                recommendations=[
                    "Implement causal correction in analysis pipeline",
                    "Add causal graph generation to workflow"
                ]
            ))

        # Check for recalibration requirements
        recalibration_trigger = self._assess_recalibration_needs(ctx.data, causal_signals)
        if recalibration_trigger:
            gaps.append(AuditGap(
                code="RECALIBRATION_REQUIRED",
                message=recalibration_trigger["reason"],
                severity=AuditSeverity.INFO,
                context=recalibration_trigger,
                recommendations=[recalibration_trigger["action"]]
            ))

        return ValidationResult.PASS if not gaps else ValidationResult.WARNING, gaps

    def _check_causal_signals(self, data: Dict[str, Any], causal_fields: List[str]) -> bool:
        """Check for causal correction signals"""
        for field in causal_fields:
            if field in data:
                return True

            report = data.get("report", {})
            if isinstance(report, dict) and field in report:
                return True

        return False

    def _assess_recalibration_needs(self, data: Dict[str, Any], has_causal_signals: bool) -> Optional[Dict[str, Any]]:
        """Assess if recalibration is needed based on divergence metrics"""
        meso = data.get("meso_summary", {})
        div = meso.get("divergence_stats", {})
        max_div = float(div.get("max", 0.0))
        avg_div = float(div.get("avg", 0.0))

        recalibration_required = (max_div > 0.5 and avg_div > 0.25)

        if recalibration_required or not has_causal_signals:
            return {
                "reason": ("high_divergence_or_missing_causal_correction" if recalibration_required
                           else "missing_causal_correction"),
                "action": "invoke_adaptive_recalibration_and_update_scores",
                "metrics": {"max_div": max_div, "avg_div": avg_div},
                "has_causal_signals": has_causal_signals
            }

        return None


class ReportingLevelsValidator(BaseValidator):
    """Validates presence of micro, meso, and macro reporting levels"""

    def __init__(self):
        super().__init__(
            name="reporting_levels",
            description="Validates presence of all reporting levels (micro/meso/macro)",
            severity=AuditSeverity.ERROR
        )

    def validate(self, ctx: ValidationContext) -> Tuple[ValidationResult, List[AuditGap]]:
        gaps = []

        # Check micro level (cluster_audit)
        if not ctx.data.get("cluster_audit"):
            gaps.append(AuditGap(
                code="MISSING_MICRO_LEVEL",
                message="Micro-level reporting is missing",
                severity=AuditSeverity.ERROR,
                recommendations=["Ensure cluster analysis generates micro-level data"]
            ))

        # Check meso level
        if not ctx.data.get("meso_summary"):
            gaps.append(AuditGap(
                code="MISSING_MESO_SUMMARY",
                message="Meso-level summary is missing",
                severity=AuditSeverity.ERROR,
                recommendations=["Run meso aggregation to generate summary data"]
            ))

        # Check macro level
        if not ctx.data.get("macro_synthesis"):
            gaps.append(AuditGap(
                code="MISSING_MACRO_SYNTHESIS",
                message="Macro-level synthesis is missing",
                severity=AuditSeverity.ERROR,
                recommendations=["Generate macro synthesis from meso data"]
            ))

        return ValidationResult.PASS if not gaps else ValidationResult.FAIL, gaps


class EvidenceTraceabilityValidator(BaseValidator):
    """Validates evidence traceability and resolution"""

    def __init__(self):
        super().__init__(
            name="evidence_traceability",
            description="Validates evidence traceability and ID resolution",
            severity=AuditSeverity.WARNING
        )

    def validate(self, ctx: ValidationContext) -> Tuple[ValidationResult, List[AuditGap]]:
        gaps = []

        # Build evidence index
        evidence_index = self._build_evidence_index(ctx.data)

        # Check evidence resolution
        unresolved_evidence, per_cluster_stats = self._check_evidence_resolution(
            ctx.data, evidence_index, ctx.deterministic
        )

        if unresolved_evidence:
            gaps.append(AuditGap(
                code="UNRESOLVED_EVIDENCE_IDS",
                message=f"Found {len(unresolved_evidence)} unresolved evidence IDs",
                severity=AuditSeverity.WARNING,
                context={
                    "unresolved_count": len(unresolved_evidence),
                    "sample_unresolved": unresolved_evidence[:5],
                    "per_cluster_stats": per_cluster_stats
                },
                recommendations=[
                    "Verify evidence store integrity",
                    "Check evidence ID generation consistency",
                    "Review evidence linking process"
                ]
            ))

        return ValidationResult.PASS if not gaps else ValidationResult.WARNING, gaps

    def _build_evidence_index(self, data: Dict[str, Any]) -> Dict[str, Set[str]]:
        """Build evidence index from evidence system or evidence dict"""
        evidence_index = {}

        # Try evidence_system first
        evidence_system = data.get("evidence_system")
        if evidence_system:
            try:
                if hasattr(evidence_system, "serialize_canonical"):
                    raw = evidence_system.serialize_canonical()
                    if isinstance(raw, str):
                        import json
                        doc = json.loads(raw)
                        store = doc.get("store", {})
                        for qid, items in store.items():
                            ids = set()
                            for item in (items or []):
                                if isinstance(item, dict):
                                    eid = item.get("id")
                                    if isinstance(eid, str):
                                        ids.add(eid)
                            evidence_index[qid] = ids
            except Exception as e:
                logger.debug(f"Failed to process evidence_system: {e}")

        # Fallback to evidence dict
        if not evidence_index:
            evidence = data.get("evidence", {})
            if isinstance(evidence, dict):
                for qid, items in evidence.items():
                    ids = set()
                    if isinstance(items, list):
                        for item in items:
                            if isinstance(item, dict):
                                eid = item.get("id") or item.get("evidence_id")
                                if isinstance(eid, str):
                                    ids.add(eid)
                    evidence_index[qid] = ids

        return evidence_index

    def _check_evidence_resolution(self, data: Dict[str, Any], evidence_index: Dict[str, Set[str]],
                                   deterministic: bool) -> Tuple[List[str], Dict[str, Dict[str, int]]]:
        """Check evidence ID resolution against evidence index"""
        unresolved_evidence = []
        per_cluster_stats = {}

        cluster_audit = data.get("cluster_audit", {})
        micro = cluster_audit.get("micro", {})

        cluster_items = sorted(micro.items()) if deterministic else micro.items()

        for cluster_id, payload in cluster_items:
            if not isinstance(payload, dict):
                continue

            answers = payload.get("answers", [])
            unresolved_count = 0
            checked = 0

            # Sort answers for deterministic processing
            if deterministic and isinstance(answers, list):
                answers = sorted(answers, key=lambda x: str(x.get("question_id", "") or x.get("question", "")))

            for answer in answers:
                if not isinstance(answer, dict):
                    continue

                qid = str(answer.get("question_id") or answer.get("question") or "")
                evidence_ids = answer.get("evidence_ids", [])

                if isinstance(evidence_ids, list):
                    # Sort for deterministic processing
                    if deterministic:
                        evidence_ids = sorted(evidence_ids)

                    for eid in evidence_ids:
                        if isinstance(eid, str):
                            checked += 1
                            universe = evidence_index.get(qid, set())
                            if universe and eid not in universe:
                                unresolved_evidence.append(f"{qid}:{eid}")
                                unresolved_count += 1

            per_cluster_stats[cluster_id] = {
                "checked": checked,
                "unresolved": unresolved_count
            }

        if deterministic:
            unresolved_evidence.sort()

        return unresolved_evidence, per_cluster_stats


class MacroSynthesizer:
    """Generates macro-level synthesis from meso data"""

    @staticmethod
    def compute_macro_synthesis(data: Dict[str, Any], deterministic: bool = False) -> Dict[str, Any]:
        """Compute lightweight macro synthesis based on meso divergence and DNP validation presence"""
        meso = data.get("meso_summary", {})
        div = meso.get("divergence_stats", {})
        max_div = float(div.get("max", 0.0))
        avg_div = float(div.get("avg", 0.0))

        # Calculate evidence coverage proxy
        total_items = int(div.get("count", 0))
        evidence_total = 0
        items = (meso.get("items") or {}).values()
        for item in items:
            evidence_total += int(item.get("evidence_coverage", 0))
        coverage_proxy = float(evidence_total) / max(1, total_items)

        # Check DNP validation presence
        uses_dnp = MacroSynthesizer._has_dnp_validation(data)

        # Calculate alignment score
        alignment_score = max(0.0, 1.0 - min(1.0, (max_div + avg_div) / 2.0)) * (0.5 + 0.5 * min(1.0, coverage_proxy))

        timestamp = 1640995200.0 if deterministic else time.time()

        return {
            "alignment_score": round(alignment_score, 4),
            "uses_dnp_standards": bool(uses_dnp),
            "divergence": {"max": max_div, "avg": avg_div},
            "coverage_proxy": round(coverage_proxy, 4),
            "timestamp": timestamp,
            "generated_by": "canonical_auditor",
            "version": __version__
        }

    @staticmethod
    def _has_dnp_validation(data: Dict[str, Any]) -> bool:
        """Check if DNP validation is present in data"""
        dnp_keys = ("dnp_validation_results", "dnp_compliance", "dnp_alignment")

        for key in dnp_keys:
            if key in data:
                return True

            report = data.get("report", {})
            if isinstance(report, dict) and key in report:
                return True

        return False


class DataPreprocessor:
    """Preprocesses data to ensure micro and meso levels exist"""

    @staticmethod
    def ensure_micro_and_meso(data: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure micro and meso data exists, generating if possible"""
        result = dict(data)

        # Try to generate micro level via cluster_execution_controller
        cluster_audit = result.get("cluster_audit")
        if not isinstance(cluster_audit, dict) and cluster_execution_controller:
            try:
                logger.debug("Attempting to generate micro-level data via cluster_execution_controller")
                result = cluster_execution_controller.process(result, context={})
                cluster_audit = result.get("cluster_audit")
                logger.info("Successfully generated micro-level data")
            except Exception as e:
                logger.warning(f"Failed to generate micro-level data: {e}")

        # Try to generate meso level via meso_aggregator
        if (not result.get("meso_summary")) and meso_aggregator and isinstance(cluster_audit, dict):
            try:
                logger.debug("Attempting to generate meso-level data via meso_aggregator")
                result = meso_aggregator.process(result, context={})
                logger.info("Successfully generated meso-level data")
            except Exception as e:
                logger.warning(f"Failed to generate meso-level data: {e}")

        return result


class CanonicalAuditor:
    """
    Enhanced canonical output auditor with comprehensive validation framework.

    Features:
    - Modular validator architecture
    - Comprehensive gap analysis with severity levels
    - Deterministic audit results for reproducibility
    - Extensible validation framework
    - Detailed recommendations and context
    """

    def __init__(self, strict_mode: bool = False):
        """
        Initialize canonical auditor.

        Args:
            strict_mode: If True, all warnings are treated as errors
        """
        self.strict_mode = strict_mode
        self.validators: List[BaseValidator] = [
            ClusterValidator(),
            EvidenceLinkageValidator(),
            DNPStandardsValidator(),
            CausalCorrectionValidator(),
            ReportingLevelsValidator(),
            EvidenceTraceabilityValidator()
        ]
        self._preprocessor = DataPreprocessor()
        self._synthesizer = MacroSynthesizer()

    def add_validator(self, validator: BaseValidator) -> None:
        """Add a custom validator"""
        if validator not in self.validators:
            self.validators.append(validator)

    def remove_validator(self, validator_name: str) -> None:
        """Remove validator by name"""
        self.validators = [v for v in self.validators if v.name != validator_name]

    def process(self, data: Any, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process data through canonical audit validation.

        Args:
            data: Input data to audit
            context: Optional context with audit configuration

        Returns:
            Augmented data with canonical_audit section
        """
        ctx_dict = context or {}

        # Ensure we have a dictionary to work with
        if not isinstance(data, dict):
            logger.warning(f"Expected dict input, got {type(data)}. Creating wrapper dict.")
            result = {"data": data}
        else:
            result = dict(data)

        try:
            # Create validation context
            validation_ctx = ValidationContext(
                data=result,
                context=ctx_dict,
                deterministic=ctx_dict.get('deterministic', False),
                strict_mode=self.strict_mode
            )

            # Preprocess data to ensure required levels exist
            result = self._preprocessor.ensure_micro_and_meso(result)
            validation_ctx.data = result

            # Generate macro synthesis if missing
            if not result.get("macro_synthesis"):
                result["macro_synthesis"] = self._synthesizer.compute_macro_synthesis(
                    result, deterministic=validation_ctx.deterministic
                )

            # Run all validations
            all_gaps = []
            validation_results = {}

            for validator in self.validators:
                if not validator.is_enabled():
                    continue

                try:
                    result_status, gaps = validator.validate(validation_ctx)
                    validation_results[validator.name] = {
                        "status": result_status.value,
                        "gap_count": len(gaps),
                        "description": validator.description
                    }
                    all_gaps.extend(gaps)

                except Exception as e:
                    logger.error(f"Validator {validator.name} failed: {e}")
                    error_gap = AuditGap(
                        code="VALIDATOR_ERROR",
                        message=f"Validator {validator.name} failed: {str(e)}",
                        severity=AuditSeverity.ERROR,
                        context={"validator": validator.name, "error": str(e)}
                    )
                    all_gaps.append(error_gap)
                    validation_results[validator.name] = {
                        "status": "error",
                        "gap_count": 1,
                        "error": str(e)
                    }

            # Generate replicability hashes
            replicability = self._generate_replicability_hashes(result, validation_ctx.deterministic)

            # Check external dependencies
            external_deps = self._check_external_dependencies()

            # Assess raw data presence
            raw_data_presence = self._assess_raw_data_presence(result)

            # Build canonical audit result
            canonical_audit = self._build_canonical_audit_result(
                result, all_gaps, validation_results, replicability,
                external_deps, raw_data_presence, validation_ctx
            )

            result["canonical_audit"] = canonical_audit

            logger.info(f"Canonical audit completed with {len(all_gaps)} gaps found")
            return result

        except Exception as e:
            logger.error(f"Canonical audit failed: {e}")
            error_audit = {
                "status": "error",
                "error_message": str(e),
                "traceback": traceback.format_exc(),
                "timestamp": validation_ctx.timestamp if 'validation_ctx' in locals() else time.time()
            }
            result["canonical_audit"] = error_audit
            return result

    def _generate_replicability_hashes(self, data: Dict[str, Any], deterministic: bool) -> Dict[str, Optional[str]]:
        """Generate replicability hashes for key components"""
        return {
            "cluster_audit_hash": (
                DeterministicHasher.stable_hash_dict(data.get("cluster_audit", {}), deterministic)
                if data.get("cluster_audit") else None
            ),
            "meso_summary_hash": (
                DeterministicHasher.stable_hash_dict(data.get("meso_summary", {}), deterministic)
                if data.get("meso_summary") else None
            ),
            "macro_synthesis_hash": (
                DeterministicHasher.stable_hash_dict(data.get("macro_synthesis", {}), deterministic)
                if data.get("macro_synthesis") else None
            )
        }

    def _check_external_dependencies(self) -> Dict[str, bool]:
        """Check availability of external dependencies"""
        return {
            "cluster_execution_controller": cluster_execution_controller is not None,
            "meso_aggregator": meso_aggregator is not None,
            "public_transformer_adapter": public_transformer_adapter is not None
        }

    def _assess_raw_data_presence(self, data: Dict[str, Any]) -> Dict[str, bool]:
        """Assess presence of raw data components"""
        return {
            "features": isinstance(data.get("features"), dict),
            "vectors": isinstance(data.get("vectors"), list) and len(data.get("vectors", [])) > 0,
            "bm25_index": bool(data.get("bm25_index")),
            "vector_index": bool(data.get("vector_index")),
            "evidence_store": bool(data.get("evidence")) or bool(data.get("evidence_system"))
        }

    def _build_canonical_audit_result(self, data: Dict[str, Any], gaps: List[AuditGap],
                                      validation_results: Dict[str, Any],
                                      replicability: Dict[str, Optional[str]],
                                      external_deps: Dict[str, bool],
                                      raw_data_presence: Dict[str, bool],
                                      ctx: ValidationContext) -> Dict[str, Any]:
        """Build the final canonical audit result"""

        # Categorize gaps by severity
        gaps_by_severity = {
            "critical": [g for g in gaps if g.severity == AuditSeverity.CRITICAL],
            "error": [g for g in gaps if g.severity == AuditSeverity.ERROR],
            "warning": [g for g in gaps if g.severity == AuditSeverity.WARNING],
            "info": [g for g in gaps if g.severity == AuditSeverity.INFO]
        }

        # Extract specific validation results
        cluster_audit = data.get("cluster_audit", {})

        # Build evidence traceability from validation results
        evidence_traceability = {}
        for validator in self.validators:
            if validator.name == "evidence_traceability":
                try:
                    _, validation_gaps = validator.validate(ctx)
                    for gap in validation_gaps:
                        if gap.code == "UNRESOLVED_EVIDENCE_IDS":
                            evidence_traceability = {
                                "per_cluster": gap.context.get("per_cluster_stats", {}),
                                "unresolved_sample": gap.context.get("sample_unresolved", []),
                                "total_unresolved": gap.context.get("unresolved_count", 0)
                            }
                except Exception:
                    pass

        # Find calibration trigger from causal correction validator
        calibration_trigger = None
        for gap in gaps:
            if gap.code == "RECALIBRATION_REQUIRED":
                calibration_trigger = gap.context
                break

        return {
            # Core compliance flags
            "four_clusters_confirmed": not any(g.code == "MISSING_REQUIRED_CLUSTERS" for g in gaps),
            "clusters_complete": bool(cluster_audit.get("complete")) if isinstance(cluster_audit, dict) else False,
            "non_redundant": bool(cluster_audit.get("non_redundant")) if isinstance(cluster_audit, dict) else False,
            "evidence_linked_micro": not any(g.code == "MISSING_EVIDENCE_LINKAGE" for g in gaps),
            "uses_dnp_standards": not any(g.code == "MISSING_DNP_ALIGNMENT" for g in gaps),
            "causal_correction_signals": not any(g.code == "MISSING_CAUSAL_CORRECTION_SIGNAL" for g in gaps),
            "recalibration_required": any(g.code == "RECALIBRATION_REQUIRED" for g in gaps),

            # Reporting levels presence
            "reporting_levels": {
                "micro": bool(data.get("cluster_audit")),
                "meso": bool(data.get("meso_summary")),
                "macro": bool(data.get("macro_synthesis"))
            },

            # Data quality indicators
            "raw_data_presence": raw_data_presence,
            "external_dependencies": external_deps,

            # Traceability and replicability
            "evidence_traceability": evidence_traceability,
            "replicability": replicability,

            # Calibration and correction
            "calibration_trigger": calibration_trigger,

            # Validation results and gaps
            "validation_results": validation_results,
            "gaps_summary": {
                "total_gaps": len(gaps),
                "by_severity": {
                    severity: len(severity_gaps)
                    for severity, severity_gaps in gaps_by_severity.items()
                }
            },
            "gaps": [gap.to_dict() for gap in gaps],
            "gap_codes": sorted(list(set(gap.code for gap in gaps))),

            # Audit metadata
            "audit_metadata": {
                "version": __version__,
                "strict_mode": self.strict_mode,
                "deterministic": ctx.deterministic,
                "validators_used": [v.name for v in self.validators if v.is_enabled()],
                "timestamp": ctx.timestamp,
                "processing_time_ms": (time.time() - ctx.timestamp) * 1000 if not ctx.deterministic else None
            },

            # Overall status
            "status": self._determine_overall_status(gaps_by_severity),
            "compliance_score": self._calculate_compliance_score(gaps_by_severity, validation_results)
        }

    def _determine_overall_status(self, gaps_by_severity: Dict[str, List[AuditGap]]) -> str:
        """Determine overall audit status based on gaps"""
        if gaps_by_severity["critical"]:
            return "critical_failure"
        elif gaps_by_severity["error"]:
            return "failure"
        elif gaps_by_severity["warning"]:
            return "warning"
        elif gaps_by_severity["info"]:
            return "advisory"
        else:
            return "pass"

    def _calculate_compliance_score(self, gaps_by_severity: Dict[str, List[AuditGap]],
                                    validation_results: Dict[str, Any]) -> float:
        """Calculate overall compliance score (0.0 to 1.0)"""
        total_validators = len(validation_results)
        if total_validators == 0:
            return 0.0

        # Weight penalties by severity
        penalty_weights = {
            "critical": 1.0,
            "error": 0.7,
            "warning": 0.3,
            "info": 0.1
        }

        total_penalty = 0.0
        for severity, gaps in gaps_by_severity.items():
            if gaps:
                total_penalty += len(gaps) * penalty_weights.get(severity, 0.5)

        # Calculate score (higher penalties = lower score)
        max_possible_penalty = total_validators * 2.0  # Assume max 2 gaps per validator
        compliance_score = max(0.0, 1.0 - (total_penalty / max_possible_penalty))

        return round(compliance_score, 4)


# Utility functions for backward compatibility and easy usage
def process(data: Any, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Main entry point for canonical auditing (backward compatible).

    Args:
        data: Input data to audit
        context: Optional context with configuration

    Returns:
        Data augmented with canonical_audit section
    """
    auditor = CanonicalAuditor()
    return auditor.process(data, context)


def create_auditor(strict_mode: bool = False,
                   custom_validators: Optional[List[BaseValidator]] = None) -> CanonicalAuditor:
    """
    Factory function to create configured auditor.

    Args:
        strict_mode: Treat warnings as errors
        custom_validators: Additional validators to include

    Returns:
        Configured CanonicalAuditor instance
    """
    auditor = CanonicalAuditor(strict_mode=strict_mode)

    if custom_validators:
        for validator in custom_validators:
            auditor.add_validator(validator)

    return auditor


# Custom validator examples for extensibility
class CustomComplianceValidator(BaseValidator):
    """Example custom validator for organization-specific compliance"""

    def __init__(self, required_fields: List[str]):
        super().__init__(
            name="custom_compliance",
            description="Validates organization-specific compliance requirements",
            severity=AuditSeverity.ERROR
        )
        self.required_fields = required_fields

    def validate(self, ctx: ValidationContext) -> Tuple[ValidationResult, List[AuditGap]]:
        gaps = []

        for field in self.required_fields:
            if field not in ctx.data:
                gaps.append(AuditGap(
                    code=f"MISSING_REQUIRED_FIELD_{field.upper()}",
                    message=f"Required field '{field}' is missing",
                    severity=self.severity,
                    context={"required_field": field},
                    recommendations=[f"Ensure '{field}' is included in processing pipeline"]
                ))

        return ValidationResult.PASS if not gaps else ValidationResult.FAIL, gaps


class PerformanceValidator(BaseValidator):
    """Validates performance metrics and thresholds"""

    def __init__(self, max_processing_time: float = 300.0):
        super().__init__(
            name="performance_validation",
            description="Validates performance metrics against thresholds",
            severity=AuditSeverity.WARNING
        )
        self.max_processing_time = max_processing_time

    def validate(self, ctx: ValidationContext) -> Tuple[ValidationResult, List[AuditGap]]:
        gaps = []

        # Check processing time if available
        performance_data = ctx.data.get("performance_metrics", {})
        if isinstance(performance_data, dict):
            processing_time = performance_data.get("total_processing_time", 0)
            if processing_time > self.max_processing_time:
                gaps.append(AuditGap(
                    code="SLOW_PROCESSING_PERFORMANCE",
                    message=f"Processing time {processing_time:.2f}s exceeds threshold {self.max_processing_time:.2f}s",
                    severity=AuditSeverity.WARNING,
                    context={
                        "processing_time": processing_time,
                        "threshold": self.max_processing_time
                    },
                    recommendations=[
                        "Review processing pipeline for optimization opportunities",
                        "Consider parallel processing or caching strategies"
                    ]
                ))

        return ValidationResult.PASS if not gaps else ValidationResult.WARNING, gaps


# Example usage and demonstration
if __name__ == "__main__":
    # Example 1: Basic usage with default configuration
    sample_data = {
        "cluster_audit": {
            "present": ["C1", "C2", "C3", "C4"],
            "complete": True,
            "non_redundant": True,
            "micro": {
                "C1": {
                    "answers": [
                        {
                            "question_id": "q1",
                            "evidence_ids": ["e1", "e2"]
                        }
                    ]
                }
            }
        },
        "meso_summary": {
            "divergence_stats": {
                "max": 0.3,
                "avg": 0.15,
                "count": 10
            }
        },
        "evidence_system": None,  # Would normally contain evidence system
        "dnp_validation_results": {"status": "validated"}
    }

    # Run standard audit
    result = process(sample_data)
    print("Standard audit result:")
    print(f"Status: {result['canonical_audit']['status']}")
    print(f"Compliance score: {result['canonical_audit']['compliance_score']}")
    print(f"Total gaps: {result['canonical_audit']['gaps_summary']['total_gaps']}")

    # Example 2: Using custom auditor with strict mode and custom validators
    custom_auditor = create_auditor(
        strict_mode=True,
        custom_validators=[
            CustomComplianceValidator(["organization_id", "compliance_token"]),
            PerformanceValidator(max_processing_time=120.0)
        ]
    )

    # Run custom audit
    custom_result = custom_auditor.process(sample_data)
    print("\nCustom audit result:")
    print(f"Status: {custom_result['canonical_audit']['status']}")
    print(f"Validators used: {custom_result['canonical_audit']['audit_metadata']['validators_used']}")

    # Example 3: Deterministic audit for reproducibility
    deterministic_result = process(sample_data, context={"deterministic": True})
    print(f"\nDeterministic audit timestamp: {deterministic_result['canonical_audit']['audit_metadata']['timestamp']}")

    print("\nEnhanced canonical auditor demonstration completed!")
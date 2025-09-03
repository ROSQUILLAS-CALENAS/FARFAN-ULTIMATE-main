"""
AuditLogger for Decálogo Point Evaluation

Captures detailed execution traces, coverage validation, and scoring consistency
validation with comprehensive diagnostic information.
"""

import json
import time
import hashlib
# # # from datetime import datetime, timezone  # Module not found  # Module not found  # Module not found
# # # from pathlib import Path  # Module not found  # Module not found  # Module not found
# # # from typing import Any, Dict, List, Optional, Set, Union  # Module not found  # Module not found  # Module not found
# # # from dataclasses import dataclass, field, asdict  # Module not found  # Module not found  # Module not found
# # # from collections import defaultdict  # Module not found  # Module not found  # Module not found
import logging

logger = logging.getLogger(__name__)


@dataclass
class ComponentTrace:
    """Trace information for a single component execution."""
    component_name: str
    entry_timestamp: float
    exit_timestamp: Optional[float] = None
    duration_ms: Optional[float] = None
    status: str = "running"
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def complete(self, status: str = "completed", outputs: Optional[Dict[str, Any]] = None, 
                 errors: Optional[List[str]] = None):
        """Mark component execution as completed."""
        self.exit_timestamp = time.time()
        self.duration_ms = (self.exit_timestamp - self.entry_timestamp) * 1000
        self.status = status
        if outputs:
            self.outputs.update(outputs)
        if errors:
            self.errors.extend(errors)


@dataclass
class DecalogoPointTrace:
    """Comprehensive trace for a single Decálogo point evaluation."""
    point_id: int
    point_name: str
    evaluation_id: str
    start_timestamp: float
    end_timestamp: Optional[float] = None
    total_duration_ms: Optional[float] = None
    components: Dict[str, ComponentTrace] = field(default_factory=dict)
    question_processing: Dict[str, float] = field(default_factory=dict)
    evidence_validation: Dict[str, float] = field(default_factory=dict)
    scoring_computation: Dict[str, Any] = field(default_factory=dict)
    status: str = "running"
    errors: List[str] = field(default_factory=list)

    def complete(self):
        """Mark point evaluation as completed."""
        self.end_timestamp = time.time()
        self.total_duration_ms = (self.end_timestamp - self.start_timestamp) * 1000
        self.status = "completed"


@dataclass
class CoverageValidationResult:
    """Coverage validation results for question-dimension mapping."""
    dimension: str
    expected_question_count: int
    actual_question_count: int
    missing_question_ids: List[str] = field(default_factory=list)
    extra_question_ids: List[str] = field(default_factory=list)
    coverage_percentage: float = 0.0
    incomplete_evidence_count: int = 0
    missing_page_references: List[Dict[str, Any]] = field(default_factory=list)
    validation_status: str = "unknown"

    def __post_init__(self):
        """Calculate coverage percentage and status."""
        if self.expected_question_count > 0:
            self.coverage_percentage = (self.actual_question_count / self.expected_question_count) * 100
        
        if self.coverage_percentage >= 95:
            self.validation_status = "excellent"
        elif self.coverage_percentage >= 80:
            self.validation_status = "good"
        elif self.coverage_percentage >= 60:
            self.validation_status = "fair"
        else:
            self.validation_status = "poor"


@dataclass
class ScoringConsistencyResult:
    """Scoring consistency validation results."""
    dimension: str
    raw_scores: List[float] = field(default_factory=list)
    weighted_scores: List[float] = field(default_factory=list)
    final_classification: str = ""
    expected_classification: str = ""
    score_variance: float = 0.0
    inconsistencies: List[Dict[str, Any]] = field(default_factory=list)
    diagnostic_info: Dict[str, Any] = field(default_factory=dict)
    consistency_status: str = "unknown"

    def __post_init__(self):
        """Calculate consistency status based on variance and classification match."""
        if self.raw_scores:
            mean_score = sum(self.raw_scores) / len(self.raw_scores)
            self.score_variance = sum((x - mean_score) ** 2 for x in self.raw_scores) / len(self.raw_scores)
        
        classification_match = self.final_classification == self.expected_classification
        low_variance = self.score_variance < 0.01
        
        if classification_match and low_variance:
            self.consistency_status = "consistent"
        elif classification_match or low_variance:
            self.consistency_status = "acceptable"
        else:
            self.consistency_status = "inconsistent"


class AuditLogger:
    """
    Comprehensive audit logger for Decálogo point evaluation system.
    
    Captures detailed execution traces, validates coverage, and ensures
    scoring consistency with diagnostic information.
    """
    
    def __init__(self, audit_file: Optional[str] = None):
        """Initialize audit logger with optional custom audit file path."""
        self.audit_file = audit_file or "canonical_flow/evaluation/_audit.json"
        Path(self.audit_file).parent.mkdir(parents=True, exist_ok=True)
        
        # Core audit data structures
        self.session_id = self._generate_session_id()
        self.decalogo_traces: Dict[str, DecalogoPointTrace] = {}
        self.coverage_results: Dict[str, CoverageValidationResult] = {}
        self.scoring_results: Dict[str, ScoringConsistencyResult] = {}
        
        # Metrics tracking
        self.metrics = {
            "total_points_evaluated": 0,
            "total_questions_processed": 0,
            "total_evidence_validated": 0,
            "average_processing_time_ms": 0.0,
            "error_count": 0,
            "warning_count": 0
        }
        
        # Internal state
        self._active_point_id: Optional[str] = None
        self._start_time = time.time()

    def _generate_session_id(self) -> str:
        """Generate unique session identifier."""
        timestamp = datetime.now(timezone.utc).isoformat()
        hash_input = f"{timestamp}_{id(self)}".encode()
        return hashlib.md5(hash_input).hexdigest()[:16]

    def start_point_evaluation(self, point_id: int, point_name: str) -> str:
        """
        Start tracking evaluation of a Decálogo point.
        
        Args:
            point_id: Numeric ID of the Decálogo point
            point_name: Human-readable name of the point
            
        Returns:
            evaluation_id: Unique identifier for this evaluation session
        """
        evaluation_id = f"{self.session_id}_p{point_id}_{int(time.time() * 1000)}"
        
        trace = DecalogoPointTrace(
            point_id=point_id,
            point_name=point_name,
            evaluation_id=evaluation_id,
            start_timestamp=time.time()
        )
        
        self.decalogo_traces[evaluation_id] = trace
        self._active_point_id = evaluation_id
        self.metrics["total_points_evaluated"] += 1
        
        logger.info(f"Started evaluation for Decálogo point {point_id}: {point_name}")
        return evaluation_id

    def log_component_entry(self, component_name: str, inputs: Optional[Dict[str, Any]] = None,
                           evaluation_id: Optional[str] = None) -> str:
        """
        Log entry into a component during evaluation.
        
        Args:
            component_name: Name of the component being entered
            inputs: Input parameters to the component
            evaluation_id: Optional evaluation ID (uses active if not provided)
            
        Returns:
            component_trace_id: Unique identifier for this component execution
        """
        eval_id = evaluation_id or self._active_point_id
        if not eval_id or eval_id not in self.decalogo_traces:
            logger.warning(f"No active evaluation found for component entry: {component_name}")
            return ""
        
        trace = ComponentTrace(
            component_name=component_name,
            entry_timestamp=time.time(),
            inputs=inputs or {}
        )
        
        component_trace_id = f"{component_name}_{len(self.decalogo_traces[eval_id].components)}"
        self.decalogo_traces[eval_id].components[component_trace_id] = trace
        
        logger.debug(f"Component entry: {component_name} in evaluation {eval_id}")
        return component_trace_id

    def log_component_exit(self, component_name: str, outputs: Optional[Dict[str, Any]] = None,
                          errors: Optional[List[str]] = None, evaluation_id: Optional[str] = None):
        """
# # #         Log exit from a component during evaluation.  # Module not found  # Module not found  # Module not found
        
        Args:
            component_name: Name of the component being exited
# # #             outputs: Output data from the component  # Module not found  # Module not found  # Module not found
            errors: Any errors encountered during execution
            evaluation_id: Optional evaluation ID (uses active if not provided)
        """
        eval_id = evaluation_id or self._active_point_id
        if not eval_id or eval_id not in self.decalogo_traces:
            logger.warning(f"No active evaluation found for component exit: {component_name}")
            return
        
        # Find the most recent component trace with this name
        point_trace = self.decalogo_traces[eval_id]
        component_trace = None
        
        for trace_id, trace in reversed(list(point_trace.components.items())):
            if trace.component_name == component_name and trace.status == "running":
                component_trace = trace
                break
        
        if component_trace:
            status = "error" if errors else "completed"
            component_trace.complete(status=status, outputs=outputs, errors=errors)
            
            if errors:
                point_trace.errors.extend(errors)
                self.metrics["error_count"] += len(errors)
        else:
            logger.warning(f"No running component trace found for exit: {component_name}")

    def log_question_processing(self, question_id: str, duration_ms: float,
                               question_text: Optional[str] = None,
                               evaluation_id: Optional[str] = None):
        """
        Log question processing timing and details.
        
        Args:
            question_id: Unique identifier for the question
            duration_ms: Processing time in milliseconds
            question_text: Optional full text of the question
            evaluation_id: Optional evaluation ID (uses active if not provided)
        """
        eval_id = evaluation_id or self._active_point_id
        if not eval_id or eval_id not in self.decalogo_traces:
            logger.warning(f"No active evaluation found for question processing: {question_id}")
            return
        
        self.decalogo_traces[eval_id].question_processing[question_id] = duration_ms
        self.metrics["total_questions_processed"] += 1
        
        logger.debug(f"Question {question_id} processed in {duration_ms:.2f}ms")

    def log_evidence_validation(self, evidence_id: str, validation_time_ms: float,
                               page_references: Optional[List[str]] = None,
                               validation_status: str = "completed",
                               evaluation_id: Optional[str] = None):
        """
        Log evidence validation timing and details.
        
        Args:
            evidence_id: Unique identifier for the evidence
            validation_time_ms: Validation time in milliseconds
            page_references: List of page references in the evidence
            validation_status: Status of validation (completed, failed, incomplete)
            evaluation_id: Optional evaluation ID (uses active if not provided)
        """
        eval_id = evaluation_id or self._active_point_id
        if not eval_id or eval_id not in self.decalogo_traces:
            logger.warning(f"No active evaluation found for evidence validation: {evidence_id}")
            return
        
        self.decalogo_traces[eval_id].evidence_validation[evidence_id] = validation_time_ms
        self.metrics["total_evidence_validated"] += 1
        
        # Track incomplete evidence if page references are missing
        if validation_status == "incomplete" or not page_references:
            dimension = self._get_current_dimension(eval_id)
            if dimension and dimension in self.coverage_results:
                self.coverage_results[dimension].incomplete_evidence_count += 1
                
                if not page_references:
                    self.coverage_results[dimension].missing_page_references.append({
                        "evidence_id": evidence_id,
                        "evaluation_id": eval_id,
                        "timestamp": time.time()
                    })
        
        logger.debug(f"Evidence {evidence_id} validated in {validation_time_ms:.2f}ms")

    def log_scoring_computation(self, dimension: str, raw_scores: List[float],
                               weighted_scores: List[float], weights: List[float],
                               final_score: float, classification: str,
                               evaluation_id: Optional[str] = None):
        """
        Log scoring computation details with diagnostic information.
        
        Args:
            dimension: Evaluation dimension being scored
            raw_scores: List of raw component scores
            weighted_scores: List of weighted scores
            weights: List of weights applied
            final_score: Final aggregated score
            classification: Final classification result
            evaluation_id: Optional evaluation ID (uses active if not provided)
        """
        eval_id = evaluation_id or self._active_point_id
        if not eval_id or eval_id not in self.decalogo_traces:
            logger.warning(f"No active evaluation found for scoring: {dimension}")
            return
        
        scoring_data = {
            "raw_scores": raw_scores,
            "weighted_scores": weighted_scores,
            "weights": weights,
            "final_score": final_score,
            "classification": classification,
            "computation_timestamp": time.time()
        }
        
        self.decalogo_traces[eval_id].scoring_computation[dimension] = scoring_data
        
        # Update scoring consistency tracking
        if dimension not in self.scoring_results:
            self.scoring_results[dimension] = ScoringConsistencyResult(dimension=dimension)
        
        scoring_result = self.scoring_results[dimension]
        scoring_result.raw_scores.extend(raw_scores)
        scoring_result.weighted_scores.extend(weighted_scores)
        scoring_result.final_classification = classification
        
        # Check for inconsistencies
        if len(raw_scores) > 1:
            score_range = max(raw_scores) - min(raw_scores)
            if score_range > 0.3:  # High variance threshold
                scoring_result.inconsistencies.append({
                    "type": "high_score_variance",
                    "range": score_range,
                    "scores": raw_scores,
                    "evaluation_id": eval_id,
                    "timestamp": time.time()
                })
        
        logger.debug(f"Scoring computed for {dimension}: {final_score:.3f} -> {classification}")

    def complete_point_evaluation(self, evaluation_id: Optional[str] = None):
        """
        Mark a Decálogo point evaluation as completed.
        
        Args:
            evaluation_id: Optional evaluation ID (uses active if not provided)
        """
        eval_id = evaluation_id or self._active_point_id
        if not eval_id or eval_id not in self.decalogo_traces:
            logger.warning(f"No evaluation found to complete: {eval_id}")
            return
        
        self.decalogo_traces[eval_id].complete()
        
        if self._active_point_id == eval_id:
            self._active_point_id = None
        
        logger.info(f"Completed evaluation {eval_id}")

    def validate_coverage(self, dimension: str, expected_questions: List[str],
                         actual_questions: List[str]) -> CoverageValidationResult:
        """
        Validate question coverage for a dimension.
        
        Args:
            dimension: Evaluation dimension name
            expected_questions: List of expected question IDs
            actual_questions: List of actual question IDs processed
            
        Returns:
            CoverageValidationResult with detailed validation information
        """
        expected_set = set(expected_questions)
        actual_set = set(actual_questions)
        
        missing_questions = list(expected_set - actual_set)
        extra_questions = list(actual_set - expected_set)
        
        result = CoverageValidationResult(
            dimension=dimension,
            expected_question_count=len(expected_questions),
            actual_question_count=len(actual_questions),
            missing_question_ids=sorted(missing_questions),
            extra_question_ids=sorted(extra_questions)
        )
        
        self.coverage_results[dimension] = result
        
        if result.validation_status in ["poor", "fair"]:
            self.metrics["warning_count"] += 1
            logger.warning(f"Coverage validation for {dimension}: {result.validation_status} "
                          f"({result.coverage_percentage:.1f}%)")
        
        return result

    def validate_scoring_consistency(self, dimension: str, expected_classification: str) -> ScoringConsistencyResult:
        """
        Validate scoring consistency for a dimension.
        
        Args:
            dimension: Evaluation dimension name
            expected_classification: Expected classification result
            
        Returns:
            ScoringConsistencyResult with detailed consistency analysis
        """
        if dimension not in self.scoring_results:
            result = ScoringConsistencyResult(dimension=dimension)
            result.expected_classification = expected_classification
            self.scoring_results[dimension] = result
            return result
        
        result = self.scoring_results[dimension]
        result.expected_classification = expected_classification
        
        # Add diagnostic information
        if result.raw_scores:
            result.diagnostic_info = {
                "score_count": len(result.raw_scores),
                "score_mean": sum(result.raw_scores) / len(result.raw_scores),
                "score_min": min(result.raw_scores),
                "score_max": max(result.raw_scores),
                "classification_match": result.final_classification == expected_classification
            }
            
            # Check for classification inconsistencies
            if result.final_classification != expected_classification:
                result.inconsistencies.append({
                    "type": "classification_mismatch",
                    "expected": expected_classification,
                    "actual": result.final_classification,
                    "scores": result.raw_scores,
                    "timestamp": time.time()
                })
        
        # Force recalculation of status
        result.__post_init__()
        
        if result.consistency_status == "inconsistent":
            self.metrics["error_count"] += 1
            logger.error(f"Scoring inconsistency detected in {dimension}: "
                        f"expected {expected_classification}, got {result.final_classification}")
        
        return result

    def _get_current_dimension(self, evaluation_id: str) -> Optional[str]:
        """Get the current dimension being evaluated."""
        if evaluation_id not in self.decalogo_traces:
            return None
        
# # #         # Extract dimension from active component traces  # Module not found  # Module not found  # Module not found
        trace = self.decalogo_traces[evaluation_id]
        for component_trace in trace.components.values():
            if "dimension" in component_trace.metadata:
                return component_trace.metadata["dimension"]
        
        return None

    def generate_audit_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive audit report.
        
        Returns:
            Complete audit report with all tracked information
        """
        # Update final metrics
        if self.decalogo_traces:
            total_duration = sum(
                trace.total_duration_ms or 0 
                for trace in self.decalogo_traces.values()
            )
            self.metrics["average_processing_time_ms"] = (
                total_duration / len(self.decalogo_traces)
            )
        
        report = {
            "session_id": self.session_id,
            "audit_timestamp": datetime.now(timezone.utc).isoformat(),
            "session_duration_s": time.time() - self._start_time,
            
            # Core tracking data
            "decalogo_traces": {
                eval_id: asdict(trace) 
                for eval_id, trace in self.decalogo_traces.items()
            },
            
            "coverage_validation": {
                dim: asdict(result) 
                for dim, result in self.coverage_results.items()
            },
            
            "scoring_consistency": {
                dim: asdict(result) 
                for dim, result in self.scoring_results.items()
            },
            
            # Summary metrics
            "metrics": self.metrics.copy(),
            
            # Analysis summaries
            "summary": self._generate_summary_analysis(),
            "recommendations": self._generate_recommendations()
        }
        
        return report

    def _generate_summary_analysis(self) -> Dict[str, Any]:
        """Generate high-level summary analysis."""
        summary = {
            "overall_status": "completed",
            "points_evaluated": len(self.decalogo_traces),
            "dimensions_covered": len(self.coverage_results),
            "average_coverage_percentage": 0.0,
            "consistency_issues": 0,
            "error_rate": 0.0
        }
        
        # Calculate average coverage
        if self.coverage_results:
            total_coverage = sum(r.coverage_percentage for r in self.coverage_results.values())
            summary["average_coverage_percentage"] = total_coverage / len(self.coverage_results)
        
        # Count consistency issues
        summary["consistency_issues"] = sum(
            len(r.inconsistencies) for r in self.scoring_results.values()
        )
        
        # Calculate error rate
        total_operations = (self.metrics["total_questions_processed"] + 
                          self.metrics["total_evidence_validated"])
        if total_operations > 0:
            summary["error_rate"] = self.metrics["error_count"] / total_operations
        
        # Determine overall status
        if summary["error_rate"] > 0.1 or summary["consistency_issues"] > 5:
            summary["overall_status"] = "issues_detected"
        elif summary["average_coverage_percentage"] < 80:
            summary["overall_status"] = "coverage_warning"
        
        return summary

    def _generate_recommendations(self) -> List[Dict[str, Any]]:
        """Generate actionable recommendations based on audit findings."""
        recommendations = []
        
        # Coverage recommendations
        for dim, result in self.coverage_results.items():
            if result.coverage_percentage < 80:
                recommendations.append({
                    "type": "coverage_improvement",
                    "priority": "high" if result.coverage_percentage < 60 else "medium",
                    "dimension": dim,
                    "message": f"Improve question coverage for {dim}: {result.coverage_percentage:.1f}%",
                    "missing_questions": len(result.missing_question_ids),
                    "action": "Add missing questions or validate mapping"
                })
        
        # Consistency recommendations
        for dim, result in self.scoring_results.items():
            if result.consistency_status == "inconsistent":
                recommendations.append({
                    "type": "scoring_consistency",
                    "priority": "high",
                    "dimension": dim,
                    "message": f"Address scoring inconsistencies in {dim}",
                    "inconsistencies": len(result.inconsistencies),
                    "action": "Review scoring algorithms and weight distributions"
                })
        
        # Performance recommendations
        if self.metrics["average_processing_time_ms"] > 5000:
            recommendations.append({
                "type": "performance",
                "priority": "medium",
                "message": f"High average processing time: {self.metrics['average_processing_time_ms']:.0f}ms",
                "action": "Optimize slow components or increase timeout thresholds"
            })
        
        return recommendations

    def save_audit(self, file_path: Optional[str] = None) -> str:
        """
        Save audit report to JSON file.
        
        Args:
            file_path: Optional custom file path
            
        Returns:
            Path where audit was saved
        """
        audit_path = file_path or self.audit_file
        Path(audit_path).parent.mkdir(parents=True, exist_ok=True)
        
        report = self.generate_audit_report()
        
        with open(audit_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Audit report saved to {audit_path}")
        return audit_path

    def load_audit(self, file_path: Optional[str] = None) -> Dict[str, Any]:
        """
# # #         Load previous audit report from JSON file.  # Module not found  # Module not found  # Module not found
        
        Args:
            file_path: Optional custom file path
            
        Returns:
            Loaded audit report data
        """
        audit_path = file_path or self.audit_file
        
        try:
            with open(audit_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Audit file not found: {audit_path}")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in audit file {audit_path}: {e}")
            return {}


# Convenience functions for global audit logging
_global_audit_logger: Optional[AuditLogger] = None


def get_audit_logger() -> AuditLogger:
    """Get or create global audit logger instance."""
    global _global_audit_logger
    if _global_audit_logger is None:
        _global_audit_logger = AuditLogger()
    return _global_audit_logger


def start_point_evaluation(point_id: int, point_name: str) -> str:
    """Start tracking evaluation of a Decálogo point (global)."""
    return get_audit_logger().start_point_evaluation(point_id, point_name)


def log_component_entry(component_name: str, inputs: Optional[Dict[str, Any]] = None) -> str:
    """Log component entry (global)."""
    return get_audit_logger().log_component_entry(component_name, inputs)


def log_component_exit(component_name: str, outputs: Optional[Dict[str, Any]] = None,
                      errors: Optional[List[str]] = None):
    """Log component exit (global)."""
    get_audit_logger().log_component_exit(component_name, outputs, errors)


def save_audit_report(file_path: Optional[str] = None) -> str:
    """Save global audit report."""
    return get_audit_logger().save_audit(file_path)
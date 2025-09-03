"""
L-Classification Evaluation Stage Orchestrator

Implements the standardized process() API contract for the L_classification_evaluation stage,
consuming classification_input/<doc>/P{n}_questions.json files and generating complete
artifact sets with deterministic JSON serialization.

Author: Stage Orchestrator
Date: December 2024
Stage: L_classification_evaluation
Code: 24L
"""

import json
import logging
import traceback
import uuid
# # # from collections import OrderedDict  # Module not found  # Module not found  # Module not found
# # # from datetime import datetime, timezone  # Module not found  # Module not found  # Module not found
# # # from pathlib import Path  # Module not found  # Module not found  # Module not found
# # # from typing import Any, Dict, List, Optional, Union  # Module not found  # Module not found  # Module not found

# Import base classes for deterministic behavior
import sys

# Mandatory Pipeline Contract Annotations
__phase__ = "L"
__code__ = "28L"
__stage_order__ = 5

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
# # # from total_ordering_base import TotalOrderingBase, DeterministicCollectionMixin  # Module not found  # Module not found  # Module not found

# Import L-stage components
try:
# # #     from .decalogo_scoring_system import ScoringSystem, PointScore  # Module not found  # Module not found  # Module not found
# # #     from ..A_analysis_nlp.evidence_validation_model import EvidenceValidationModel as EvidenceValidator  # Module not found  # Module not found  # Module not found
# # #     from ..A_analysis_nlp.dnp_alignment_adapter import DNPAlignmentAdapter  # Module not found  # Module not found  # Module not found
except ImportError:
    # Fallback for standalone execution
# # #     from decalogo_scoring_system import ScoringSystem, PointScore  # Module not found  # Module not found  # Module not found
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "A_analysis_nlp"))
# # #     from evidence_validation_model import EvidenceValidationModel as EvidenceValidator  # Module not found  # Module not found  # Module not found
# # #     from dnp_alignment_adapter import DNPAlignmentAdapter  # Module not found  # Module not found  # Module not found


class LClassificationStageOrchestrator(TotalOrderingBase, DeterministicCollectionMixin):
    """
    Stage orchestrator for L_classification_evaluation implementing standardized process() API.
    
    Coordinates execution of all L-stage components while maintaining deterministic ordering
    through stable sorting of all output collections and comprehensive error handling.
    """
    
    def __init__(self, precision: int = 4):
        super().__init__("LClassificationStageOrchestrator")
        
        # Initialize component systems
        self.scoring_system = ScoringSystem(precision=precision)
        self.evidence_validator = EvidenceValidator()
        self.dnp_alignment = DNPAlignmentAdapter()
        
        # Configuration
        self.precision = precision
        self.max_processing_time_seconds = 300
        self.enable_parallel_processing = True
        
        # State tracking
        self.execution_id = None
        self.start_time = None
        self.audit_log = []
        
        # Setup logging
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Update state hash
        self.update_state_hash({
            "precision": precision,
            "components": ["ScoringSystem", "EvidenceValidator", "DNPAlignmentAdapter"],
            "version": "1.0.0"
        })
    
    def process(self, data: Any = None, context: Any = None) -> Dict[str, Any]:
        """
        Standardized process() API contract for L_classification_evaluation stage.
        
        Args:
            data: Input data containing classification inputs or file paths
            context: Optional execution context
            
        Returns:
            Dict containing all required artifacts with deterministic JSON serialization
        """
        # Initialize execution
        self.execution_id = str(uuid.uuid4())
        self.start_time = datetime.now(timezone.utc)
        self._log_audit("orchestrator_started", {"execution_id": self.execution_id})
        
        try:
            # Process input data
            input_files = self._resolve_input_files(data, context)
            
            # Initialize result structure
            results = {
                "execution_metadata": {
                    "execution_id": self.execution_id,
                    "stage": "L_classification_evaluation",
                    "start_time": self.start_time.isoformat(),
                    "orchestrator_version": "1.0.0"
                },
                "artifacts": {
                    "dimension_evaluations": OrderedDict(),
                    "point_summaries": OrderedDict(),
                    "composition_traces": OrderedDict(),
                    "confidence_intervals": OrderedDict(),
                    "guard_reports": OrderedDict(),
                    "points_index": OrderedDict()
                },
                "status_report": {
                    "successful_points": [],
                    "failed_points": [],
                    "processing_summary": {}
                },
                "audit_log": []
            }
            
            # Process each point with error isolation
            for file_path in input_files:
                point_id = self._extract_point_id(file_path)
                
                try:
                    self._log_audit("processing_point_started", {"point_id": point_id, "file": str(file_path)})
                    
                    # Process individual point
                    point_results = self._process_point(file_path, point_id)
                    
                    # Add results to artifacts
                    self._add_point_artifacts(results["artifacts"], point_id, point_results)
                    
                    # Track success
                    results["status_report"]["successful_points"].append(point_id)
                    
                    self._log_audit("processing_point_completed", {"point_id": point_id})
                    
                except Exception as e:
                    # Isolate failure - allow other points to continue
                    error_details = {
                        "point_id": point_id,
                        "file": str(file_path),
                        "error": str(e),
                        "traceback": traceback.format_exc()
                    }
                    
                    results["status_report"]["failed_points"].append(error_details)
                    self._log_audit("processing_point_failed", error_details)
                    
                    self.logger.error(f"Point {point_id} processing failed: {e}")
            
            # Generate final indices and summaries
            self._finalize_results(results)
            
            # Apply deterministic ordering to all collections
            self._apply_deterministic_ordering(results)
            
            return results
            
        except Exception as e:
            self._log_audit("orchestrator_failed", {"error": str(e), "traceback": traceback.format_exc()})
            raise
        finally:
            # Always add audit log to results if available
            if hasattr(self, 'audit_log') and 'audit_log' in locals():
                results["audit_log"] = self.audit_log
    
    def _resolve_input_files(self, data: Any, context: Any) -> List[Path]:
        """
# # #         Resolve input files from data and context.  # Module not found  # Module not found  # Module not found
        
        Args:
            data: Input data (can be dict, list, or path)
            context: Optional context with additional paths
            
        Returns:
            List of input file paths sorted deterministically
        """
        input_files = []
        
        if isinstance(data, dict):
            if "classification_input_path" in data:
                base_path = Path(data["classification_input_path"])
                input_files.extend(base_path.rglob("P*_questions.json"))
            elif "input_files" in data:
                input_files.extend(Path(f) for f in data["input_files"])
        
        elif isinstance(data, (list, tuple)):
            input_files.extend(Path(f) for f in data)
        
        elif isinstance(data, (str, Path)):
            path = Path(data)
            if path.is_dir():
                input_files.extend(path.rglob("P*_questions.json"))
            elif path.suffix == ".json" and "questions" in path.name:
                input_files.append(path)
        
        # Add context paths if available
        if context and isinstance(context, dict):
            if "input_paths" in context:
                for ctx_path in context["input_paths"]:
                    path = Path(ctx_path)
                    if path.is_dir():
                        input_files.extend(path.rglob("P*_questions.json"))
                    else:
                        input_files.append(path)
        
        # Remove duplicates and sort deterministically
        unique_files = list(set(input_files))
        return sorted(unique_files, key=lambda p: (str(p.parent), p.name))
    
    def _extract_point_id(self, file_path: Path) -> int:
        """
# # #         Extract point ID from file path.  # Module not found  # Module not found  # Module not found
        
        Args:
            file_path: Path to P{n}_questions.json file
            
        Returns:
            Point ID as integer
        """
# # #         # Extract from filename pattern P{n}_questions.json  # Module not found  # Module not found  # Module not found
        filename = file_path.name
        if filename.startswith("P") and "_questions.json" in filename:
            point_str = filename[1:filename.index("_questions.json")]
            return int(point_str)
        
# # #         raise ValueError(f"Cannot extract point ID from filename: {filename}")  # Module not found  # Module not found  # Module not found
    
    def _process_point(self, file_path: Path, point_id: int) -> Dict[str, Any]:
        """
        Process a single Decálogo point with all L-stage components.
        
        Args:
            file_path: Path to point questions file
            point_id: Point ID
            
        Returns:
            Complete point processing results
        """
        # Load question data
        with open(file_path, 'r', encoding='utf-8') as f:
            question_data = json.load(f)
        
        results = {
            "point_id": point_id,
            "input_file": str(file_path),
            "components": OrderedDict()
        }
        
        # Execute components in proper sequence
        try:
            # 1. Scoring System - Core evaluation
            self._log_audit("scoring_started", {"point_id": point_id})
            scoring_result = self._execute_scoring_system(point_id, question_data)
            results["components"]["scoring_system"] = scoring_result
            self._log_audit("scoring_completed", {"point_id": point_id, "final_score": scoring_result.get("final_score", 0)})
            
            # 2. Evidence Quality Validation
            self._log_audit("evidence_validation_started", {"point_id": point_id})
            evidence_result = self._execute_evidence_validation(point_id, question_data, scoring_result)
            results["components"]["evidence_validation"] = evidence_result
            self._log_audit("evidence_validation_completed", {"point_id": point_id})
            
            # 3. DNP Alignment
            self._log_audit("dnp_alignment_started", {"point_id": point_id})
            dnp_result = self._execute_dnp_alignment(point_id, question_data, scoring_result)
            results["components"]["dnp_alignment"] = dnp_result
            self._log_audit("dnp_alignment_completed", {"point_id": point_id})
            
            # 4. Generate derived artifacts
            results["artifacts"] = self._generate_point_artifacts(results["components"], point_id)
            
        except Exception as e:
            results["error"] = {
                "message": str(e),
                "traceback": traceback.format_exc(),
                "component": self._identify_failed_component(traceback.format_exc())
            }
            raise
        
        return results
    
    def _execute_scoring_system(self, point_id: int, question_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute scoring system with deterministic evaluation."""
        # Convert question data to expected format
        evaluation_data = self._convert_to_evaluation_format(question_data)
        
        # Process with scoring system
        point_score = self.scoring_system.process_point_evaluation(point_id, evaluation_data)
        
        # Convert to serializable format
        return {
            "point_id": point_score.point_id,
            "final_score": point_score.final_score,
            "total_questions": point_score.total_questions,
            "dimension_scores": [
                {
                    "dimension_id": ds.dimension_id,
                    "weighted_average": ds.weighted_average,
                    "total_questions": ds.total_questions,
                    "question_responses": [
                        {
                            "question_id": qr.question_id,
                            "response": qr.response,
                            "base_score": qr.base_score,
                            "evidence_completeness": qr.evidence_completeness,
                            "page_reference_quality": qr.page_reference_quality,
                            "final_score": qr.final_score
                        }
                        for qr in ds.question_responses
                    ]
                }
                for ds in point_score.dimension_scores
            ]
        }
    
    def _execute_evidence_validation(self, point_id: int, question_data: Dict[str, Any], 
                                   scoring_result: Dict[str, Any]) -> Dict[str, Any]:
        """Execute evidence quality validation."""
        try:
            # Extract evidence texts for validation
            evidence_texts = []
            for dimension_data in question_data.values():
                if isinstance(dimension_data, list):
                    for question in dimension_data:
                        if isinstance(question, dict) and "evidence_text" in question:
                            evidence_texts.append(question["evidence_text"])
            
            # Run evidence validation
            validation_results = []
            for i, evidence_text in enumerate(evidence_texts):
                result = self.evidence_validator.process(
                    data=evidence_text,
                    context={"point_id": point_id, "evidence_index": i}
                )
                validation_results.append(result)
            
            return {
                "point_id": point_id,
                "validation_results": validation_results,
                "summary": {
                    "total_evidence_items": len(evidence_texts),
                    "validation_status": "completed"
                }
            }
        
        except Exception as e:
            return {
                "point_id": point_id,
                "error": str(e),
                "validation_results": [],
                "summary": {
                    "total_evidence_items": 0,
                    "validation_status": "failed"
                }
            }
    
    def _execute_dnp_alignment(self, point_id: int, question_data: Dict[str, Any], 
                             scoring_result: Dict[str, Any]) -> Dict[str, Any]:
        """Execute DNP alignment assessment."""
        try:
            # Prepare data for DNP alignment
            alignment_data = {
                "point_id": point_id,
                "scoring_result": scoring_result,
                "question_data": question_data
            }
            
            # Run DNP alignment
            alignment_result = self.dnp_alignment.process(
                data=alignment_data,
                context={"stage": "classification_evaluation"}
            )
            
            return {
                "point_id": point_id,
                "alignment_result": alignment_result,
                "alignment_status": "completed"
            }
        
        except Exception as e:
            return {
                "point_id": point_id,
                "error": str(e),
                "alignment_result": {},
                "alignment_status": "failed"
            }
    
    def _convert_to_evaluation_format(self, question_data: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """Convert input question data to scoring system format."""
        evaluation_data = {
            "DE-1": [],
            "DE-2": [],
            "DE-3": [],
            "DE-4": []
        }
        
        # Convert question data to expected format
        for dimension, questions in question_data.items():
            if dimension in evaluation_data and isinstance(questions, list):
                for question in questions:
                    if isinstance(question, dict):
                        eval_question = {
                            "question_id": question.get("question_id", ""),
                            "response": question.get("response", "NI"),
                            "evidence_completeness": question.get("evidence_completeness", 0.0),
                            "page_reference_quality": question.get("page_reference_quality", 0.0)
                        }
                        evaluation_data[dimension].append(eval_question)
        
        return evaluation_data
    
    def _generate_point_artifacts(self, components: Dict[str, Any], point_id: int) -> Dict[str, Any]:
        """Generate all required artifacts for a point."""
        artifacts = {}
        
        # Dimension evaluations
        if "scoring_system" in components:
            artifacts["dimension_evaluation"] = {
                "point_id": point_id,
                "dimensions": components["scoring_system"].get("dimension_scores", []),
                "evaluation_timestamp": self._get_deterministic_timestamp()
            }
        
        # Point summary
        artifacts["point_summary"] = {
            "point_id": point_id,
            "final_score": components.get("scoring_system", {}).get("final_score", 0.0),
            "total_questions": components.get("scoring_system", {}).get("total_questions", 0),
            "evidence_validation_status": components.get("evidence_validation", {}).get("summary", {}).get("validation_status", "unknown"),
            "dnp_alignment_status": components.get("dnp_alignment", {}).get("alignment_status", "unknown"),
            "processing_timestamp": self._get_deterministic_timestamp()
        }
        
        # Composition trace
        artifacts["composition_trace"] = {
            "point_id": point_id,
            "component_execution_order": ["scoring_system", "evidence_validation", "dnp_alignment"],
            "dimension_weights": {
                "DE-1": 0.30,
                "DE-2": 0.25,
                "DE-3": 0.25,
                "DE-4": 0.20
            },
            "composition_method": "weighted_average",
            "trace_timestamp": self._get_deterministic_timestamp()
        }
        
        # Confidence intervals (placeholder for statistical analysis)
        artifacts["confidence_interval"] = {
            "point_id": point_id,
            "final_score": components.get("scoring_system", {}).get("final_score", 0.0),
            "confidence_level": 0.95,
            "lower_bound": max(0.0, components.get("scoring_system", {}).get("final_score", 0.0) - 0.05),
            "upper_bound": min(1.0, components.get("scoring_system", {}).get("final_score", 0.0) + 0.05),
            "interval_timestamp": self._get_deterministic_timestamp()
        }
        
        # Guard report
        artifacts["guard_report"] = {
            "point_id": point_id,
            "validation_checks": {
                "scoring_completed": "scoring_system" in components,
                "evidence_validated": "evidence_validation" in components,
                "dnp_aligned": "dnp_alignment" in components,
                "no_critical_errors": not any("error" in comp for comp in components.values())
            },
            "guard_status": "passed" if all([
                "scoring_system" in components,
                "evidence_validation" in components,
                "dnp_alignment" in components,
                not any("error" in comp for comp in components.values())
            ]) else "failed",
            "report_timestamp": self._get_deterministic_timestamp()
        }
        
        return artifacts
    
    def _add_point_artifacts(self, artifacts: Dict[str, Any], point_id: int, point_results: Dict[str, Any]):
        """Add point artifacts to main results with deterministic keys."""
        point_artifacts = point_results.get("artifacts", {})
        
        # Add to each artifact type
        if "dimension_evaluation" in point_artifacts:
            artifacts["dimension_evaluations"][f"P{point_id:02d}"] = point_artifacts["dimension_evaluation"]
        
        if "point_summary" in point_artifacts:
            artifacts["point_summaries"][f"P{point_id:02d}"] = point_artifacts["point_summary"]
        
        if "composition_trace" in point_artifacts:
            artifacts["composition_traces"][f"P{point_id:02d}"] = point_artifacts["composition_trace"]
        
        if "confidence_interval" in point_artifacts:
            artifacts["confidence_intervals"][f"P{point_id:02d}"] = point_artifacts["confidence_interval"]
        
        if "guard_report" in point_artifacts:
            artifacts["guard_reports"][f"P{point_id:02d}"] = point_artifacts["guard_report"]
    
    def _finalize_results(self, results: Dict[str, Any]):
        """Generate final indices and processing summaries."""
        # Generate points index
        points_index = OrderedDict()
        successful_points = results["status_report"]["successful_points"]
        failed_points = [fp["point_id"] for fp in results["status_report"]["failed_points"]]
        
        for point_id in sorted(successful_points + failed_points):
            points_index[f"P{point_id:02d}"] = {
                "point_id": point_id,
                "status": "success" if point_id in successful_points else "failed",
                "artifacts_available": [
                    artifact_type for artifact_type, artifacts in results["artifacts"].items()
                    if f"P{point_id:02d}" in artifacts
                ],
                "index_timestamp": self._get_deterministic_timestamp()
            }
        
        results["artifacts"]["points_index"] = points_index
        
        # Processing summary
        results["status_report"]["processing_summary"] = {
            "total_points": len(successful_points) + len(failed_points),
            "successful_points": len(successful_points),
            "failed_points": len(failed_points),
            "success_rate": len(successful_points) / (len(successful_points) + len(failed_points)) if (successful_points or failed_points) else 0.0,
            "processing_completed": datetime.now(timezone.utc).isoformat()
        }
    
    def _apply_deterministic_ordering(self, results: Dict[str, Any]):
        """Apply deterministic ordering to all collections in results."""
        # Sort successful and failed points
        results["status_report"]["successful_points"].sort()
        results["status_report"]["failed_points"].sort(key=lambda x: x["point_id"])
        
        # All artifacts are already OrderedDict with sorted keys
        # Audit log maintains chronological order
        
        # Add final metadata
        results["execution_metadata"]["end_time"] = datetime.now(timezone.utc).isoformat()
        results["execution_metadata"]["total_processing_time_seconds"] = (
            datetime.now(timezone.utc) - self.start_time
        ).total_seconds()
    
    def _identify_failed_component(self, traceback_str: str) -> str:
        """Identify which component failed based on traceback."""
        if "scoring_system" in traceback_str.lower():
            return "scoring_system"
        elif "evidence_validation" in traceback_str.lower():
            return "evidence_validation"
        elif "dnp_alignment" in traceback_str.lower():
            return "dnp_alignment"
        else:
            return "orchestrator"
    
    def _log_audit(self, event: str, details: Dict[str, Any]):
        """Log audit event with timestamp."""
        audit_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "execution_id": self.execution_id,
            "event": event,
            "details": details
        }
        self.audit_log.append(audit_entry)
    
    def serialize_results(self, results: Dict[str, Any], output_path: Optional[Path] = None) -> str:
        """
        Serialize results with deterministic JSON formatting.
        
        Args:
            results: Results dictionary to serialize
            output_path: Optional output file path
            
        Returns:
            JSON string with deterministic formatting
        """
        json_str = json.dumps(results, indent=2, ensure_ascii=False, sort_keys=True)
        
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(json_str)
        
        return json_str
    
    def get_orchestrator_info(self) -> Dict[str, Any]:
        """Get information about the orchestrator configuration."""
        return {
            "orchestrator": "LClassificationStageOrchestrator",
            "version": "1.0.0",
            "stage": "L_classification_evaluation",
            "components": ["ScoringSystem", "EvidenceValidator", "DNPAlignmentAdapter"],
            "precision": self.precision,
            "deterministic": True,
            "api_contract": "standardized_process",
            "artifact_types": [
                "dimension_evaluations",
                "point_summaries", 
                "composition_traces",
                "confidence_intervals",
                "guard_reports",
                "points_index"
            ]
        }


# Convenience function for direct usage
def process(data: Any = None, context: Any = None) -> Dict[str, Any]:
    """
    Standardized process() API entry point for L_classification_evaluation stage.
    
    Args:
        data: Input data containing classification inputs or file paths
        context: Optional execution context
        
    Returns:
        Complete L-stage processing results with all artifacts
    """
    orchestrator = LClassificationStageOrchestrator()
    return orchestrator.process(data, context)
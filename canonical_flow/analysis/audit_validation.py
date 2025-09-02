"""
Audit Validation Module for EGW Query Expansion Analysis Components

This module provides validation utilities to ensure that the generated _audit.json 
contains complete execution traces for all analysis components that were invoked 
during the analysis pipeline run.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Any
from datetime import datetime


class AuditValidationError(Exception):
    """Exception raised when audit validation fails."""
    pass


class AuditValidator:
    """
    Validator for audit data completeness and integrity.
    
    Ensures that all required components have complete execution traces
    and that the audit data meets quality standards.
    """
    
    def __init__(self, audit_file_path: str = "canonical_flow/analysis/_audit.json"):
        self.audit_file_path = Path(audit_file_path)
        self.expected_components = {
            "13A": "adaptive_analyzer",
            "14A": "question_analyzer", 
            "15A": "implementacion_mapeo",
            "16A": "evidence_processor",
            "17A": "extractor_evidencias_contextual",
            "18A": "evidence_validation_model",
            "19A": "evaluation_driven_processor",
            "20A": "dnp_alignment_adapter"
        }
    
    def validate_audit_file_exists(self) -> bool:
        """Validate that the audit file exists."""
        return self.audit_file_path.exists()
    
    def load_audit_data(self) -> Dict[str, Any]:
        """Load audit data from the JSON file."""
        if not self.validate_audit_file_exists():
            raise AuditValidationError(f"Audit file not found: {self.audit_file_path}")
        
        try:
            with open(self.audit_file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise AuditValidationError(f"Invalid JSON in audit file: {e}")
        except Exception as e:
            raise AuditValidationError(f"Error reading audit file: {e}")
    
    def validate_audit_structure(self, audit_data: Dict[str, Any]) -> List[str]:
        """
        Validate the basic structure of audit data.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Check required top-level keys
        required_keys = ["audit_metadata", "system_events", "components", "execution_summary"]
        for key in required_keys:
            if key not in audit_data:
                errors.append(f"Missing required top-level key: {key}")
        
        # Validate audit metadata
        if "audit_metadata" in audit_data:
            metadata = audit_data["audit_metadata"]
            required_metadata = ["generated_timestamp", "execution_id", "total_components_invoked", 
                                "total_events_recorded", "audit_format_version"]
            for key in required_metadata:
                if key not in metadata:
                    errors.append(f"Missing required metadata key: {key}")
        
        # Validate components structure
        if "components" in audit_data:
            components = audit_data["components"]
            if not isinstance(components, dict):
                errors.append("Components should be a dictionary")
            else:
                for comp_code, comp_data in components.items():
                    if not isinstance(comp_data, dict):
                        errors.append(f"Component {comp_code} data should be a dictionary")
                        continue
                    
                    required_comp_keys = ["component_name", "events", "metrics_summary"]
                    for key in required_comp_keys:
                        if key not in comp_data:
                            errors.append(f"Component {comp_code} missing key: {key}")
        
        return errors
    
    def validate_component_completeness(self, audit_data: Dict[str, Any], 
                                       invoked_components: Optional[Set[str]] = None) -> Dict[str, Any]:
        """
        Validate that all invoked components have complete execution traces.
        
        Args:
            audit_data: Loaded audit data
            invoked_components: Set of component codes that should be present
                              If None, validates all components that appear in audit data
        
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            "complete": True,
            "errors": [],
            "warnings": [],
            "component_status": {},
            "missing_traces": [],
            "incomplete_traces": [],
            "validation_timestamp": datetime.now().isoformat()
        }
        
        if "components" not in audit_data:
            validation_results["complete"] = False
            validation_results["errors"].append("No components data found in audit")
            return validation_results
        
        recorded_components = set(audit_data["components"].keys())
        
        # If no specific components to validate, use all recorded ones
        if invoked_components is None:
            invoked_components = recorded_components
        
        # Check for missing components
        missing_components = invoked_components - recorded_components
        if missing_components:
            validation_results["complete"] = False
            validation_results["missing_traces"] = list(missing_components)
            for comp in missing_components:
                validation_results["errors"].append(
                    f"Missing execution trace for component {comp} ({self.expected_components.get(comp, 'unknown')})"
                )
        
        # Validate each recorded component
        for comp_code, comp_data in audit_data["components"].items():
            comp_status = self._validate_single_component(comp_code, comp_data)
            validation_results["component_status"][comp_code] = comp_status
            
            if not comp_status["complete"]:
                validation_results["complete"] = False
                validation_results["incomplete_traces"].append(comp_code)
                validation_results["errors"].extend(comp_status["errors"])
            
            validation_results["warnings"].extend(comp_status["warnings"])
        
        return validation_results
    
    def _validate_single_component(self, comp_code: str, comp_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate execution trace for a single component.
        
        Returns:
            Dictionary with component-specific validation results
        """
        status = {
            "complete": True,
            "errors": [],
            "warnings": [],
            "start_events": 0,
            "end_events": 0,
            "error_events": 0,
            "has_performance_metrics": False,
            "has_input_schema": False,
            "has_output_schema": False
        }
        
        if "events" not in comp_data:
            status["complete"] = False
            status["errors"].append(f"Component {comp_code} has no events")
            return status
        
        events = comp_data["events"]
        if not isinstance(events, list):
            status["complete"] = False
            status["errors"].append(f"Component {comp_code} events should be a list")
            return status
        
        # Analyze events
        for event in events:
            if not isinstance(event, dict):
                status["errors"].append(f"Component {comp_code} has invalid event structure")
                continue
            
            event_type = event.get("event_type", "")
            
            if event_type == "component_start":
                status["start_events"] += 1
                if "input_schema" in event:
                    status["has_input_schema"] = True
            elif event_type == "component_end":
                status["end_events"] += 1
                if "output_schema" in event:
                    status["has_output_schema"] = True
                if "performance_metrics" in event:
                    status["has_performance_metrics"] = True
            elif event_type == "component_error":
                status["error_events"] += 1
        
        # Check for balanced start/end events
        if status["start_events"] != status["end_events"]:
            status["complete"] = False
            status["errors"].append(
                f"Component {comp_code} has unbalanced events: "
                f"{status['start_events']} starts, {status['end_events']} ends"
            )
        
        # Check for at least one complete execution
        if status["start_events"] == 0:
            status["complete"] = False
            status["errors"].append(f"Component {comp_code} has no start events")
        
        # Warnings for missing optional data
        if not status["has_performance_metrics"] and status["end_events"] > 0:
            status["warnings"].append(f"Component {comp_code} missing performance metrics")
        
        if not status["has_input_schema"] and status["start_events"] > 0:
            status["warnings"].append(f"Component {comp_code} missing input schema")
        
        if not status["has_output_schema"] and status["end_events"] > 0:
            status["warnings"].append(f"Component {comp_code} missing output schema")
        
        return status
    
    def validate_execution_summary(self, audit_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate the execution summary section.
        
        Returns:
            Dictionary with summary validation results
        """
        validation = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        if "execution_summary" not in audit_data:
            validation["valid"] = False
            validation["errors"].append("Missing execution_summary section")
            return validation
        
        summary = audit_data["execution_summary"]
        required_keys = ["total_execution_time", "components_with_errors", "successful_components"]
        
        for key in required_keys:
            if key not in summary:
                validation["valid"] = False
                validation["errors"].append(f"Missing execution summary key: {key}")
        
        # Validate consistency with components data
        if "components" in audit_data:
            recorded_components = set(audit_data["components"].keys())
            
            if "successful_components" in summary:
                successful = set(summary["successful_components"])
                extra_successful = successful - recorded_components
                if extra_successful:
                    validation["warnings"].append(
                        f"Execution summary lists unknown successful components: {extra_successful}"
                    )
            
            if "components_with_errors" in summary:
                error_components = set(summary["components_with_errors"])
                extra_errors = error_components - recorded_components
                if extra_errors:
                    validation["warnings"].append(
                        f"Execution summary lists unknown error components: {extra_errors}"
                    )
        
        return validation
    
    def run_full_validation(self, invoked_components: Optional[Set[str]] = None) -> Dict[str, Any]:
        """
        Run complete audit validation including all checks.
        
        Args:
            invoked_components: Set of component codes that should be present
        
        Returns:
            Comprehensive validation report
        """
        full_report = {
            "validation_timestamp": datetime.now().isoformat(),
            "audit_file_path": str(self.audit_file_path),
            "file_exists": False,
            "structure_valid": False,
            "components_complete": False,
            "execution_summary_valid": False,
            "overall_valid": False,
            "errors": [],
            "warnings": [],
            "detailed_results": {}
        }
        
        # Check file existence
        full_report["file_exists"] = self.validate_audit_file_exists()
        if not full_report["file_exists"]:
            full_report["errors"].append(f"Audit file not found: {self.audit_file_path}")
            return full_report
        
        # Load and validate structure
        try:
            audit_data = self.load_audit_data()
            
            # Validate structure
            structure_errors = self.validate_audit_structure(audit_data)
            full_report["structure_valid"] = len(structure_errors) == 0
            full_report["errors"].extend(structure_errors)
            full_report["detailed_results"]["structure_validation"] = {
                "valid": full_report["structure_valid"],
                "errors": structure_errors
            }
            
            # Validate component completeness
            completeness_results = self.validate_component_completeness(audit_data, invoked_components)
            full_report["components_complete"] = completeness_results["complete"]
            full_report["errors"].extend(completeness_results["errors"])
            full_report["warnings"].extend(completeness_results["warnings"])
            full_report["detailed_results"]["component_validation"] = completeness_results
            
            # Validate execution summary
            summary_results = self.validate_execution_summary(audit_data)
            full_report["execution_summary_valid"] = summary_results["valid"]
            full_report["errors"].extend(summary_results["errors"])
            full_report["warnings"].extend(summary_results["warnings"])
            full_report["detailed_results"]["summary_validation"] = summary_results
            
            # Overall validation status
            full_report["overall_valid"] = (
                full_report["structure_valid"] and 
                full_report["components_complete"] and 
                full_report["execution_summary_valid"]
            )
            
        except AuditValidationError as e:
            full_report["errors"].append(str(e))
        except Exception as e:
            full_report["errors"].append(f"Unexpected validation error: {e}")
        
        return full_report
    
    def generate_validation_report(self, validation_results: Dict[str, Any]) -> str:
        """
        Generate a human-readable validation report.
        
        Args:
            validation_results: Results from run_full_validation()
        
        Returns:
            Formatted validation report as string
        """
        report_lines = []
        
        report_lines.append("=" * 60)
        report_lines.append("EGW QUERY EXPANSION AUDIT VALIDATION REPORT")
        report_lines.append("=" * 60)
        report_lines.append(f"Generated: {validation_results['validation_timestamp']}")
        report_lines.append(f"Audit File: {validation_results['audit_file_path']}")
        report_lines.append()
        
        # Overall status
        overall_status = "✓ PASSED" if validation_results["overall_valid"] else "✗ FAILED"
        report_lines.append(f"Overall Validation Status: {overall_status}")
        report_lines.append()
        
        # Individual checks
        checks = [
            ("File Exists", "file_exists"),
            ("Structure Valid", "structure_valid"),
            ("Components Complete", "components_complete"),
            ("Execution Summary Valid", "execution_summary_valid")
        ]
        
        for check_name, check_key in checks:
            status = "✓" if validation_results.get(check_key, False) else "✗"
            report_lines.append(f"{status} {check_name}")
        
        # Errors
        if validation_results["errors"]:
            report_lines.append()
            report_lines.append("ERRORS:")
            for error in validation_results["errors"]:
                report_lines.append(f"  • {error}")
        
        # Warnings
        if validation_results["warnings"]:
            report_lines.append()
            report_lines.append("WARNINGS:")
            for warning in validation_results["warnings"]:
                report_lines.append(f"  • {warning}")
        
        # Component details
        if "detailed_results" in validation_results and "component_validation" in validation_results["detailed_results"]:
            comp_validation = validation_results["detailed_results"]["component_validation"]
            if "component_status" in comp_validation:
                report_lines.append()
                report_lines.append("COMPONENT STATUS:")
                for comp_code, status in comp_validation["component_status"].items():
                    comp_name = self.expected_components.get(comp_code, "unknown")
                    status_icon = "✓" if status["complete"] else "✗"
                    report_lines.append(f"  {status_icon} {comp_code} ({comp_name})")
                    if status["errors"]:
                        for error in status["errors"]:
                            report_lines.append(f"      • {error}")
        
        report_lines.append()
        report_lines.append("=" * 60)
        
        return "\n".join(report_lines)


def validate_audit_completeness(audit_file_path: Optional[str] = None,
                               invoked_components: Optional[Set[str]] = None) -> Dict[str, Any]:
    """
    Convenience function to run full audit validation.
    
    Args:
        audit_file_path: Path to audit file (uses default if None)
        invoked_components: Set of component codes that should be present
    
    Returns:
        Full validation results
    """
    if audit_file_path is None:
        audit_file_path = "canonical_flow/analysis/_audit.json"
    
    validator = AuditValidator(audit_file_path)
    return validator.run_full_validation(invoked_components)


def print_audit_validation_report(audit_file_path: Optional[str] = None,
                                 invoked_components: Optional[Set[str]] = None) -> bool:
    """
    Print a formatted audit validation report to console.
    
    Args:
        audit_file_path: Path to audit file (uses default if None)
        invoked_components: Set of component codes that should be present
    
    Returns:
        True if validation passed, False otherwise
    """
    validator = AuditValidator(audit_file_path or "canonical_flow/analysis/_audit.json")
    results = validator.run_full_validation(invoked_components)
    report = validator.generate_validation_report(results)
    print(report)
    return results["overall_valid"]
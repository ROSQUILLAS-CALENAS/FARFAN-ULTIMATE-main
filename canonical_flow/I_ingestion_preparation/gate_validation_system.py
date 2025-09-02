"""
Gate Validation System for I_ingestion_preparation Stage

This module enforces strict sequential execution order from 01I through 05I by checking 
for required input artifacts before each component runs. The system verifies dependencies
and prevents execution of downstream components when required inputs are missing or corrupted.

Components and Dependencies:
- 01I (pdf_reader.py): No dependencies → Produces _text.json files
- 02I (advanced_loader.py): Requires _text.json files → Produces _bundle.json files  
- 03I (feature_extractor.py): Requires _bundle.json files → Produces _features.json files
- 04I (normative_validator.py): Requires _features.json files → Produces _validation.json files
- 05I (raw_data_generator.py): Requires _validation.json files → Produces final artifacts

Author: Gate Validation System
Date: December 2024
Stage: I_ingestion_preparation
"""

import json
import hashlib
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# Import LibraryStatusReporter for pre-flight validation
try:
    from ..mathematical_enhancers.mathematical_compatibility_matrix import (
        LibraryStatusReporter, MathematicalCompatibilityMatrix
    )
    LIBRARY_STATUS_REPORTER_AVAILABLE = True
except ImportError:
    LIBRARY_STATUS_REPORTER_AVAILABLE = False


class ComponentState(str, Enum):
    """Execution states for pipeline components."""
    
    PENDING = "pending"           # Ready to execute
    BLOCKED = "blocked"           # Cannot execute due to missing dependencies
    RUNNING = "running"           # Currently executing  
    COMPLETED = "completed"       # Successfully completed
    FAILED = "failed"             # Failed during execution
    SKIPPED = "skipped"           # Skipped due to upstream failures
    

class GateStatus(str, Enum):
    """Gate validation statuses."""
    
    OPEN = "open"                 # All dependencies satisfied
    CLOSED = "closed"             # Dependencies not satisfied
    ERROR = "error"               # Error during validation


@dataclass
class ArtifactSpec:
    """Specification for a required artifact."""
    
    pattern: str                  # File pattern (e.g., "*_text.json")
    description: str              # Human-readable description
    required_fields: List[str] = field(default_factory=list)  # Required JSON fields
    min_size_bytes: int = 0       # Minimum file size
    max_age_hours: Optional[int] = None  # Maximum age in hours


@dataclass
class ValidationResult:
    """Result of artifact validation."""
    
    artifact_path: Path
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GateValidationReport:
    """Comprehensive report of gate validation results."""
    
    component_id: str
    gate_status: GateStatus
    dependencies_satisfied: bool
    missing_artifacts: List[str] = field(default_factory=list)
    corrupted_artifacts: List[str] = field(default_factory=list)
    validation_results: List[ValidationResult] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


class ArtifactValidator(ABC):
    """Abstract base class for artifact validators."""
    
    @abstractmethod
    def validate(self, artifact_path: Path) -> ValidationResult:
        """Validate a specific artifact."""
        pass
    
    
class JSONArtifactValidator(ArtifactValidator):
    """Validator for JSON artifacts."""
    
    def __init__(self, required_fields: List[str] = None, min_size_bytes: int = 0):
        self.required_fields = required_fields or []
        self.min_size_bytes = min_size_bytes
        
    def validate(self, artifact_path: Path) -> ValidationResult:
        """Validate JSON artifact structure and content."""
        errors = []
        warnings = []
        metadata = {}
        
        # Check file exists
        if not artifact_path.exists():
            return ValidationResult(
                artifact_path=artifact_path,
                is_valid=False,
                errors=[f"Artifact not found: {artifact_path}"]
            )
        
        # Check file size
        file_size = artifact_path.stat().st_size
        metadata['file_size_bytes'] = file_size
        
        if file_size < self.min_size_bytes:
            errors.append(f"File too small: {file_size} bytes < {self.min_size_bytes} bytes")
        
        if file_size == 0:
            return ValidationResult(
                artifact_path=artifact_path,
                is_valid=False,
                errors=errors + ["File is empty"]
            )
        
        # Validate JSON structure
        try:
            with open(artifact_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                metadata['json_keys'] = list(data.keys()) if isinstance(data, dict) else []
                
                # Check required fields
                if isinstance(data, dict):
                    for field in self.required_fields:
                        if field not in data:
                            errors.append(f"Required field missing: {field}")
                        elif not data[field]:  # Check if field is empty
                            warnings.append(f"Required field is empty: {field}")
                else:
                    errors.append("JSON data is not a dictionary")
                    
        except json.JSONDecodeError as e:
            errors.append(f"Invalid JSON format: {str(e)}")
        except Exception as e:
            errors.append(f"Error reading file: {str(e)}")
        
        # Calculate file hash for integrity
        try:
            metadata['sha256_hash'] = self._calculate_file_hash(artifact_path)
        except Exception as e:
            warnings.append(f"Could not calculate file hash: {str(e)}")
        
        return ValidationResult(
            artifact_path=artifact_path,
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metadata=metadata
        )
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()


class ComponentGate:
    """Gate that controls access to a pipeline component based on dependencies."""
    
    def __init__(
        self, 
        component_id: str,
        dependencies: List[ArtifactSpec],
        base_path: Path,
        validators: Dict[str, ArtifactValidator] = None
    ):
        self.component_id = component_id
        self.dependencies = dependencies
        self.base_path = Path(base_path)
        self.validators = validators or {}
        self.logger = logging.getLogger(f"ComponentGate.{component_id}")
        
    def validate_dependencies(self) -> GateValidationReport:
        """Validate all dependencies for this component."""
        report = GateValidationReport(
            component_id=self.component_id,
            gate_status=GateStatus.CLOSED,
            dependencies_satisfied=False
        )
        
        try:
            # Check each dependency
            all_satisfied = True
            
            for dependency in self.dependencies:
                dependency_satisfied = self._validate_dependency(dependency, report)
                if not dependency_satisfied:
                    all_satisfied = False
            
            # Set final status
            report.dependencies_satisfied = all_satisfied
            report.gate_status = GateStatus.OPEN if all_satisfied else GateStatus.CLOSED
            
            # Generate recommendations
            self._generate_recommendations(report)
            
        except Exception as e:
            self.logger.error(f"Error during dependency validation: {str(e)}")
            report.gate_status = GateStatus.ERROR
            report.validation_results.append(
                ValidationResult(
                    artifact_path=Path("unknown"),
                    is_valid=False,
                    errors=[f"Validation system error: {str(e)}"]
                )
            )
        
        return report
    
    def _validate_dependency(self, dependency: ArtifactSpec, report: GateValidationReport) -> bool:
        """Validate a single dependency."""
        # Find matching artifacts
        matching_files = list(self.base_path.glob(dependency.pattern))
        
        if not matching_files:
            report.missing_artifacts.append(dependency.pattern)
            self.logger.warning(f"Missing artifacts matching pattern: {dependency.pattern}")
            return False
        
        # Validate each matching file
        dependency_satisfied = True
        validator = self._get_validator_for_pattern(dependency.pattern)
        
        for artifact_path in matching_files:
            validation_result = validator.validate(artifact_path)
            report.validation_results.append(validation_result)
            
            if not validation_result.is_valid:
                report.corrupted_artifacts.append(str(artifact_path))
                dependency_satisfied = False
                self.logger.error(f"Invalid artifact: {artifact_path}, errors: {validation_result.errors}")
            else:
                self.logger.debug(f"Valid artifact: {artifact_path}")
        
        return dependency_satisfied
    
    def _get_validator_for_pattern(self, pattern: str) -> ArtifactValidator:
        """Get appropriate validator for file pattern."""
        # Use configured validator if available
        if pattern in self.validators:
            return self.validators[pattern]
        
        # Default validators based on file extension
        if pattern.endswith('.json'):
            return JSONArtifactValidator(min_size_bytes=10)  # Minimum 10 bytes for valid JSON
        
        # Fallback validator for other file types
        return JSONArtifactValidator()  # Will still check basic file properties
    
    def _generate_recommendations(self, report: GateValidationReport):
        """Generate actionable recommendations based on validation results."""
        if report.missing_artifacts:
            report.recommendations.append(
                f"Execute upstream components to generate missing artifacts: {', '.join(report.missing_artifacts)}"
            )
        
        if report.corrupted_artifacts:
            report.recommendations.append(
                f"Fix or regenerate corrupted artifacts: {', '.join(report.corrupted_artifacts)}"
            )
        
        if report.dependencies_satisfied:
            report.recommendations.append(f"All dependencies satisfied. Component {self.component_id} is ready for execution.")


class IngestionPipelineGatekeeper:
    """Main gatekeeper for the I_ingestion_preparation pipeline."""
    
    # Component execution order
    COMPONENT_ORDER = ["01I", "02I", "03I", "04I", "05I"]
    
    # Component dependency specifications
    COMPONENT_DEPENDENCIES = {
        "01I": [],  # pdf_reader has no dependencies
        "02I": [    # advanced_loader requires text files from pdf_reader
            ArtifactSpec(
                pattern="*_text.json",
                description="Text extraction results from PDF reader",
                required_fields=["text", "pages", "metadata"],
                min_size_bytes=100
            )
        ],
        "03I": [    # feature_extractor requires bundle files from advanced_loader
            ArtifactSpec(
                pattern="*_bundle.json", 
                description="Document bundles from advanced loader",
                required_fields=["document_features", "structure", "content"],
                min_size_bytes=200
            )
        ],
        "04I": [    # normative_validator requires feature files from feature_extractor
            ArtifactSpec(
                pattern="*_features.json",
                description="Extracted features from feature extractor", 
                required_fields=["textual_features", "structural_features", "compliance_score"],
                min_size_bytes=150
            )
        ],
        "05I": [    # raw_data_generator requires validation files from normative_validator
            ArtifactSpec(
                pattern="*_validation.json",
                description="Validation results from normative validator",
                required_fields=["compliance_score", "checklist", "summary"], 
                min_size_bytes=300
            )
        ]
    }
    
    def __init__(self, base_data_path: Union[str, Path], enable_strict_mode: bool = True,
                 enable_library_status_reporting: bool = True):
        self.base_data_path = Path(base_data_path)
        self.enable_strict_mode = enable_strict_mode
        self.enable_library_status_reporting = enable_library_status_reporting
        self.component_states: Dict[str, ComponentState] = {
            comp: ComponentState.PENDING for comp in self.COMPONENT_ORDER
        }
        self.gates: Dict[str, ComponentGate] = {}
        self.execution_log: List[Dict[str, Any]] = []
        self.logger = logging.getLogger(self.__class__.__name__)
        self.library_status_reporter = None
        self.pipeline_startup_report = None
        
        # Initialize library status reporter if available
        self._initialize_library_status_reporter()
        
        # Initialize gates for each component
        self._initialize_gates()
    
    def _initialize_library_status_reporter(self):
        """Initialize the library status reporter for pre-flight validation."""
        if not self.enable_library_status_reporting or not LIBRARY_STATUS_REPORTER_AVAILABLE:
            self.logger.info("Library status reporting disabled or not available")
            return
        
        try:
            compatibility_matrix = MathematicalCompatibilityMatrix()
            self.library_status_reporter = LibraryStatusReporter(
                matrix=compatibility_matrix,
                report_path=self.base_data_path / "library_status_report.json"
            )
            self.logger.info("Library status reporter initialized successfully")
        except Exception as e:
            self.logger.warning(f"Could not initialize library status reporter: {e}")
    
    def run_pre_flight_dependency_validation(self) -> Dict[str, Any]:
        """
        Run pre-flight dependency validation during pipeline startup.
        
        Returns:
            Dict with validation results and recommendations
        """
        if not self.library_status_reporter:
            return {
                'success': False,
                'error': 'Library status reporter not available',
                'timestamp': datetime.now().isoformat()
            }
        
        self.logger.info("Running pre-flight dependency validation...")
        
        try:
            # Generate comprehensive status report
            report_summary = self.library_status_reporter.invoke_status_reporting()
            
            # Store the startup report for later reference
            self.pipeline_startup_report = report_summary
            
            if report_summary['success']:
                self.logger.info(f"Pre-flight validation completed: {report_summary['report_path']}")
                
                # Log critical findings
                health = report_summary['system_health']
                if health['critical_issues']:
                    self.logger.error(f"Critical library issues detected: {health['required_missing_count']} required libraries missing")
                else:
                    self.logger.info(f"System health: {health['health_classification']} (score: {health['overall_score']:.2f})")
                
                # Log risk assessment
                risk_level = report_summary['risk_level']
                if risk_level in ['HIGH', 'MEDIUM']:
                    self.logger.warning(f"Risk level: {risk_level}")
                else:
                    self.logger.info(f"Risk level: {risk_level}")
                
                # Track the pre-flight validation call
                for lib_name in self.library_status_reporter.matrix.library_specs.keys():
                    self.library_status_reporter.track_library_call(lib_name, is_mock=False)
            
            return report_summary
            
        except Exception as e:
            error_msg = f"Pre-flight dependency validation failed: {str(e)}"
            self.logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'timestamp': datetime.now().isoformat()
            }
    
    def get_aggregated_status_report(self) -> Dict[str, Any]:
        """
        Get aggregated status report across all EGW components.
        
        Returns:
            Aggregated status report or empty dict if not available
        """
        if not self.library_status_reporter:
            return {}
        
        try:
            # Generate comprehensive report 
            full_report = self.library_status_reporter.generate_comprehensive_report()
            
            # Create aggregated summary
            aggregated = {
                'pipeline_startup_report': self.pipeline_startup_report,
                'system_health_summary': full_report['system_health'],
                'stage_readiness_summary': full_report['stage_readiness'],
                'risk_assessment_summary': full_report['risk_assessment'],
                'library_usage_statistics': {
                    lib_name: {
                        'fallback_rate': (counters['mock_calls'] / max(1, counters['mock_calls'] + counters['real_calls'])),
                        'total_calls': counters['mock_calls'] + counters['real_calls'],
                        'fallback_triggered': counters['fallback_triggered']
                    }
                    for lib_name, counters in self.library_status_reporter.fallback_counters.items()
                },
                'critical_recommendations': [
                    rec for rec in full_report['upgrade_recommendations'] 
                    if rec['priority'] in ['HIGH', 'CRITICAL']
                ],
                'aggregation_timestamp': datetime.now().isoformat()
            }
            
            return aggregated
            
        except Exception as e:
            self.logger.error(f"Failed to generate aggregated status report: {e}")
            return {}
        
    def _initialize_gates(self):
        """Initialize validation gates for all components."""
        for component_id in self.COMPONENT_ORDER:
            dependencies = self.COMPONENT_DEPENDENCIES.get(component_id, [])
            
            # Create component-specific validators
            validators = self._create_component_validators(component_id)
            
            self.gates[component_id] = ComponentGate(
                component_id=component_id,
                dependencies=dependencies, 
                base_path=self.base_data_path,
                validators=validators
            )
            
    def _create_component_validators(self, component_id: str) -> Dict[str, ArtifactValidator]:
        """Create specialized validators for component artifacts."""
        validators = {}
        
        dependencies = self.COMPONENT_DEPENDENCIES.get(component_id, [])
        for dep in dependencies:
            if dep.pattern.endswith('.json'):
                validators[dep.pattern] = JSONArtifactValidator(
                    required_fields=dep.required_fields,
                    min_size_bytes=dep.min_size_bytes
                )
        
        return validators
    
    def validate_component_readiness(self, component_id: str) -> GateValidationReport:
        """Validate if a component is ready for execution."""
        if component_id not in self.COMPONENT_ORDER:
            raise ValueError(f"Unknown component: {component_id}")
        
        gate = self.gates[component_id]
        report = gate.validate_dependencies()
        
        # Log validation attempt
        self.execution_log.append({
            'timestamp': datetime.now(),
            'component_id': component_id,
            'action': 'validation_check',
            'gate_status': report.gate_status.value,
            'dependencies_satisfied': report.dependencies_satisfied
        })
        
        self.logger.info(f"Component {component_id} validation: {report.gate_status.value}")
        
        return report
    
    def can_execute_component(self, component_id: str) -> Tuple[bool, GateValidationReport]:
        """Check if component can be executed based on dependencies and state."""
        report = self.validate_component_readiness(component_id)
        
        # In strict mode, all dependencies must be satisfied
        if self.enable_strict_mode:
            can_execute = (
                report.gate_status == GateStatus.OPEN and
                report.dependencies_satisfied and
                self.component_states[component_id] == ComponentState.PENDING
            )
        else:
            # In non-strict mode, allow execution with warnings
            can_execute = (
                report.gate_status != GateStatus.ERROR and
                self.component_states[component_id] == ComponentState.PENDING
            )
        
        return can_execute, report
    
    def execute_component_with_validation(
        self, 
        component_id: str, 
        execution_func: callable,
        *args, 
        **kwargs
    ) -> Dict[str, Any]:
        """Execute component with pre and post validation."""
        self.logger.info(f"Attempting to execute component {component_id}")
        
        # Pre-execution validation
        can_execute, validation_report = self.can_execute_component(component_id)
        
        if not can_execute:
            error_msg = f"Component {component_id} cannot execute due to failed dependencies"
            self.logger.error(error_msg)
            
            # Mark component as blocked
            self.component_states[component_id] = ComponentState.BLOCKED
            
            # Mark downstream components as skipped if in strict mode
            if self.enable_strict_mode:
                self._mark_downstream_components_skipped(component_id)
            
            return {
                'success': False,
                'component_id': component_id,
                'error': error_msg,
                'validation_report': validation_report,
                'execution_blocked': True
            }
        
        # Execute component
        try:
            # Update state
            self.component_states[component_id] = ComponentState.RUNNING
            
            self.logger.info(f"Executing component {component_id}")
            start_time = time.time()
            
            # Execute the actual component function
            result = execution_func(*args, **kwargs)
            
            execution_time = time.time() - start_time
            
            # Update state based on result
            if result and result.get('success', True):
                self.component_states[component_id] = ComponentState.COMPLETED
                self.logger.info(f"Component {component_id} completed successfully in {execution_time:.2f}s")
                
                # Log successful execution
                self.execution_log.append({
                    'timestamp': datetime.now(),
                    'component_id': component_id,
                    'action': 'execution_completed',
                    'execution_time_seconds': execution_time,
                    'success': True
                })
                
                return {
                    'success': True,
                    'component_id': component_id,
                    'result': result,
                    'execution_time_seconds': execution_time,
                    'validation_report': validation_report
                }
            else:
                # Component reported failure
                self.component_states[component_id] = ComponentState.FAILED
                error_msg = result.get('error', 'Component execution failed') if result else 'Component execution failed'
                self.logger.error(f"Component {component_id} failed: {error_msg}")
                
                # Mark downstream components as skipped
                if self.enable_strict_mode:
                    self._mark_downstream_components_skipped(component_id)
                
                # Log failed execution
                self.execution_log.append({
                    'timestamp': datetime.now(),
                    'component_id': component_id,
                    'action': 'execution_failed',
                    'execution_time_seconds': execution_time,
                    'success': False,
                    'error': error_msg
                })
                
                return {
                    'success': False,
                    'component_id': component_id,
                    'error': error_msg,
                    'result': result,
                    'execution_time_seconds': execution_time,
                    'validation_report': validation_report
                }
        
        except Exception as e:
            # Unexpected execution error
            self.component_states[component_id] = ComponentState.FAILED
            error_msg = f"Unexpected error during execution: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            
            # Mark downstream components as skipped
            if self.enable_strict_mode:
                self._mark_downstream_components_skipped(component_id)
            
            # Log exception
            self.execution_log.append({
                'timestamp': datetime.now(),
                'component_id': component_id,
                'action': 'execution_exception',
                'success': False,
                'error': error_msg
            })
            
            return {
                'success': False,
                'component_id': component_id,
                'error': error_msg,
                'validation_report': validation_report,
                'exception_occurred': True
            }
    
    def _mark_downstream_components_skipped(self, failed_component_id: str):
        """Mark all downstream components as skipped when a component fails."""
        component_index = self.COMPONENT_ORDER.index(failed_component_id)
        
        for i in range(component_index + 1, len(self.COMPONENT_ORDER)):
            downstream_component = self.COMPONENT_ORDER[i]
            if self.component_states[downstream_component] == ComponentState.PENDING:
                self.component_states[downstream_component] = ComponentState.SKIPPED
                self.logger.warning(f"Component {downstream_component} skipped due to upstream failure")
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get comprehensive status of the entire pipeline."""
        status = {
            'component_states': dict(self.component_states),
            'pipeline_health': self._calculate_pipeline_health(),
            'execution_log': self.execution_log.copy(),
            'next_executable_component': self._get_next_executable_component(),
            'blocked_components': self._get_blocked_components(),
            'timestamp': datetime.now()
        }
        
        return status
    
    def _calculate_pipeline_health(self) -> Dict[str, Any]:
        """Calculate overall pipeline health metrics."""
        state_counts = {}
        for state in ComponentState:
            state_counts[state.value] = sum(
                1 for comp_state in self.component_states.values() 
                if comp_state == state
            )
        
        total_components = len(self.COMPONENT_ORDER)
        completed_components = state_counts.get(ComponentState.COMPLETED.value, 0)
        failed_components = state_counts.get(ComponentState.FAILED.value, 0)
        
        health_percentage = (completed_components / total_components) * 100 if total_components > 0 else 0
        
        return {
            'state_counts': state_counts,
            'health_percentage': round(health_percentage, 1),
            'completion_status': f"{completed_components}/{total_components}",
            'has_failures': failed_components > 0,
            'is_healthy': failed_components == 0 and completed_components == total_components
        }
    
    def _get_next_executable_component(self) -> Optional[str]:
        """Get the next component that can be executed."""
        for component_id in self.COMPONENT_ORDER:
            if self.component_states[component_id] == ComponentState.PENDING:
                can_execute, _ = self.can_execute_component(component_id)
                if can_execute:
                    return component_id
        return None
    
    def _get_blocked_components(self) -> List[Dict[str, Any]]:
        """Get list of blocked components with reasons."""
        blocked = []
        
        for component_id in self.COMPONENT_ORDER:
            if self.component_states[component_id] == ComponentState.BLOCKED:
                report = self.validate_component_readiness(component_id)
                blocked.append({
                    'component_id': component_id,
                    'missing_artifacts': report.missing_artifacts,
                    'corrupted_artifacts': report.corrupted_artifacts,
                    'recommendations': report.recommendations
                })
        
        return blocked
    
    def reset_pipeline(self):
        """Reset pipeline to initial state."""
        self.logger.info("Resetting pipeline state")
        
        self.component_states = {
            comp: ComponentState.PENDING for comp in self.COMPONENT_ORDER
        }
        
        self.execution_log.clear()
        
        # Log reset
        self.execution_log.append({
            'timestamp': datetime.now(),
            'action': 'pipeline_reset'
        })
    
    def generate_dependency_report(self) -> Dict[str, Any]:
        """Generate comprehensive dependency analysis report."""
        report = {
            'pipeline_overview': {
                'total_components': len(self.COMPONENT_ORDER),
                'execution_order': self.COMPONENT_ORDER.copy(),
                'strict_mode_enabled': self.enable_strict_mode
            },
            'component_dependencies': {},
            'current_status': self.get_pipeline_status(),
            'validation_summary': {},
            'recommendations': []
        }
        
        # Analyze each component's dependencies
        for component_id in self.COMPONENT_ORDER:
            dependencies = self.COMPONENT_DEPENDENCIES.get(component_id, [])
            validation_report = self.validate_component_readiness(component_id)
            
            component_info = {
                'dependencies': [
                    {
                        'pattern': dep.pattern,
                        'description': dep.description,
                        'required_fields': dep.required_fields,
                        'min_size_bytes': dep.min_size_bytes
                    }
                    for dep in dependencies
                ],
                'current_state': self.component_states[component_id].value,
                'gate_status': validation_report.gate_status.value,
                'dependencies_satisfied': validation_report.dependencies_satisfied,
                'missing_artifacts': validation_report.missing_artifacts,
                'corrupted_artifacts': validation_report.corrupted_artifacts
            }
            
            report['component_dependencies'][component_id] = component_info
            
            # Add to validation summary
            report['validation_summary'][component_id] = {
                'valid_artifacts': len([r for r in validation_report.validation_results if r.is_valid]),
                'invalid_artifacts': len([r for r in validation_report.validation_results if not r.is_valid]),
                'total_artifacts': len(validation_report.validation_results)
            }
        
        # Generate global recommendations
        next_component = self._get_next_executable_component()
        if next_component:
            report['recommendations'].append(f"Execute component {next_component} next")
        
        blocked_components = self._get_blocked_components()
        if blocked_components:
            report['recommendations'].append(f"Resolve dependencies for blocked components: {[b['component_id'] for b in blocked_components]}")
        
        return report
"""
Mathematical Pipeline Coordinator

Orchestrates the 9 stage mathematical enhancers (math_stage1 through math_stage12)
by managing their execution dependencies and implementing theorem-based validation
to ensure mathematical consistency across all pipeline stages.

Features:
- Dependency resolution between mathematical enhancer stages
- Theorem-based validation at stage boundaries  
- Mathematical invariant preservation verification
- Rollback mechanisms for validation failures
- Unified interface for comprehensive_pipeline_orchestrator.py integration
- Mathematical consistency enforcement across pipeline flow

Architecture:
- Stage-based execution with dependency resolution
- Mathematical property validation hooks
- Rollback and recovery mechanisms
- Comprehensive logging and monitoring
"""

import logging
import traceback
# # # from abc import ABC, abstractmethod  # Module not found  # Module not found  # Module not found
# # # from dataclasses import dataclass, field  # Module not found  # Module not found  # Module not found
# # # from enum import Enum  # Module not found  # Module not found  # Module not found
# # # from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable  # Module not found  # Module not found  # Module not found
# # # from datetime import datetime  # Module not found  # Module not found  # Module not found
import numpy as np
import json

try:
    import sys
# # #     from pathlib import Path  # Module not found  # Module not found  # Module not found
    
    # Add project root to path for canonical imports
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
# # #     from egw_query_expansion.mathematical_foundations import (  # Module not found  # Module not found  # Module not found
        InformationTheory, 
        SemanticSimilarity, 
        BayesianInference,
        EntropyMeasures,
        SimilarityResult,
        BayesianResult
    )
except ImportError:
    # Fallback implementations for testing without full installation
    class InformationTheory:
        @staticmethod
        def shannon_entropy(probabilities):
            if len(probabilities) == 0:
                return 0.0
            return -np.sum(probabilities * np.log(probabilities + 1e-10))
    
    class SemanticSimilarity:
        pass
    
    class BayesianInference:
        pass
    
    class EntropyMeasures:
        pass
    
    class SimilarityResult:
        pass
    
    class BayesianResult:
        pass

logger = logging.getLogger(__name__)


class MathStageType(Enum):
    """Mathematical enhancement stage types"""
    INGESTION = "math_stage1_ingestion_enhancer"
    CONTEXT = "math_stage2_context_enhancer"  
    KNOWLEDGE = "math_stage3_knowledge_enhancer"
    ANALYSIS = "math_stage4_analysis_enhancer"
    CLASSIFICATION = "math_stage5_classification_enhancer"
    SEARCH = "math_stage6_search_enhancer"
    ORCHESTRATION = "math_stage7_orchestration_enhancer"
    AGGREGATION = "math_stage8_aggregation_enhancer"
    INTEGRATION = "math_stage9_integration_enhancer"


class ValidationSeverity(Enum):
    """Severity levels for mathematical validation"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class MathematicalInvariant:
    """Represents a mathematical property that must be preserved"""
    name: str
    description: str
    validation_function: Callable[[Dict[str, Any]], bool]
    severity: ValidationSeverity
    error_message: str


@dataclass
class ValidationResult:
    """Result of validation process with comprehensive status and metadata"""
    is_valid: bool
    error_messages: List[str] = field(default_factory=list)
    warning_messages: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    validation_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    severity_level: Optional[str] = None
    
    def add_error(self, message: str):
        """Add an error message to the validation result"""
        self.error_messages.append(message)
        self.is_valid = False
    
    def add_warning(self, message: str):
        """Add a warning message to the validation result"""
        self.warning_messages.append(message)
    
    def update_metadata(self, key: str, value: Any):
        """Update metadata with key-value pair"""
        self.metadata[key] = value
    
    def has_errors(self) -> bool:
        """Check if validation has any errors"""
        return len(self.error_messages) > 0
    
    def has_warnings(self) -> bool:
        """Check if validation has any warnings"""
        return len(self.warning_messages) > 0
    
    def get_summary(self) -> str:
        """Get a summary of the validation result"""
        status = "VALID" if self.is_valid else "INVALID"
        error_count = len(self.error_messages)
        warning_count = len(self.warning_messages)
        return f"Validation {status}: {error_count} errors, {warning_count} warnings"


@dataclass
class StageResult:
    """Result of stage execution with comprehensive status and timing information"""
    stage_name: str
    execution_status: str  # 'pending', 'running', 'completed', 'failed', 'rolled_back'
    output_data: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    validation_result: Optional[ValidationResult] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def mark_started(self):
        """Mark the stage as started"""
        self.execution_status = 'running'
        self.start_time = datetime.now()
    
    def mark_completed(self, output_data: Dict[str, Any] = None):
        """Mark the stage as completed"""
        self.execution_status = 'completed'
        self.end_time = datetime.now()
        if output_data:
            self.output_data.update(output_data)
        if self.start_time and self.end_time:
            self.execution_time = (self.end_time - self.start_time).total_seconds()
    
    def mark_failed(self, error_message: str):
        """Mark the stage as failed with error message"""
        self.execution_status = 'failed'
        self.end_time = datetime.now()
        self.error_message = error_message
        if self.start_time and self.end_time:
            self.execution_time = (self.end_time - self.start_time).total_seconds()
    
    def mark_rolled_back(self):
        """Mark the stage as rolled back"""
        self.execution_status = 'rolled_back'
        self.error_message = None
    
    def add_metric(self, name: str, value: float):
        """Add a performance metric"""
        self.metrics[name] = value
    
    def update_metadata(self, key: str, value: Any):
        """Update metadata with key-value pair"""
        self.metadata[key] = value
    
    def is_successful(self) -> bool:
        """Check if stage execution was successful"""
        return self.execution_status == 'completed'
    
    def has_failed(self) -> bool:
        """Check if stage execution failed"""
        return self.execution_status == 'failed'
    
    def get_duration_ms(self) -> float:
        """Get execution duration in milliseconds"""
        return self.execution_time * 1000.0


@dataclass
class StageValidationResult:
    """Result of mathematical validation for a stage"""
    stage: MathStageType
    success: bool
    violations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class StageExecutionState:
    """Execution state for a mathematical enhancement stage"""
    stage: MathStageType
    status: str  # 'pending', 'running', 'completed', 'failed', 'rolled_back'
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)
    validation_result: Optional[StageValidationResult] = None
    stage_result: Optional[StageResult] = None
    execution_time: float = 0.0
    error_message: Optional[str] = None
    rollback_checkpoint: Optional[Dict[str, Any]] = None


class MathematicalEnhancer(ABC):
    """Abstract base class for mathematical enhancement stages"""
    
    def __init__(self, stage_type: MathStageType):
        self.stage_type = stage_type
        self.dependencies: Set[MathStageType] = set()
        self.invariants: List[MathematicalInvariant] = []
        
    @abstractmethod
    def enhance(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply mathematical enhancement to input data"""
        pass
        
    @abstractmethod
    def validate_output(self, output_data: Dict[str, Any]) -> StageValidationResult:
        """Validate mathematical properties of output"""
        pass
        
    def add_invariant(self, invariant: MathematicalInvariant):
        """Add a mathematical invariant to verify"""
        self.invariants.append(invariant)
        
    def create_checkpoint(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a rollback checkpoint"""
        return {
            'timestamp': datetime.now().isoformat(),
            'data': json.loads(json.dumps(data, default=str)),
            'stage': self.stage_type.value
        }


class MathStage1IngestiorEnhancer(MathematicalEnhancer):
    """Mathematical enhancer for ingestion stage"""
    
    def __init__(self):
        super().__init__(MathStageType.INGESTION)
        
        # Add mathematical invariants for ingestion
        self.add_invariant(MathematicalInvariant(
            name="data_completeness",
            description="Input data must be non-empty and well-formed",
            validation_function=lambda data: len(data.get('text', '')) > 0,
            severity=ValidationSeverity.CRITICAL,
            error_message="Input text cannot be empty"
        ))
        
        self.add_invariant(MathematicalInvariant(
            name="encoding_consistency", 
            description="Text encoding must be consistent",
            validation_function=self._validate_encoding,
            severity=ValidationSeverity.ERROR,
            error_message="Text encoding is inconsistent"
        ))
    
    def _validate_encoding(self, data: Dict[str, Any]) -> bool:
        """Validate text encoding consistency"""
        try:
            text = data.get('text', '')
            if isinstance(text, str):
                # Try to encode/decode to verify consistency
                text.encode('utf-8').decode('utf-8')
                return True
            return False
        except (UnicodeError, AttributeError):
            return False
    
    def enhance(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply mathematical enhancements to ingestion data"""
        logger.info("Applying mathematical enhancements to ingestion stage")
        
        enhanced_data = input_data.copy()
        
        # Extract text content
        text = input_data.get('text', '')
        
        # Add information-theoretic measures
        if text:
            # Character frequency distribution
            char_counts = np.array([text.count(c) for c in set(text)])
            char_probs = char_counts / np.sum(char_counts)
            
            # Compute entropy measures
            info_theory = InformationTheory()
            text_entropy = info_theory.shannon_entropy(char_probs)
            
            enhanced_data.update({
                'mathematical_metrics': {
                    'text_entropy': float(text_entropy),
                    'text_length': len(text),
                    'char_diversity': len(set(text)),
                    'compression_ratio': text_entropy / np.log2(len(set(text))) if len(set(text)) > 1 else 0.0
                }
            })
        
        return enhanced_data
    
    def validate_output(self, output_data: Dict[str, Any]) -> StageValidationResult:
        """Validate mathematical properties of ingestion output"""
        violations = []
        warnings = []
        
        # Validate all invariants
        for invariant in self.invariants:
            try:
                if not invariant.validation_function(output_data):
                    if invariant.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]:
                        violations.append(f"{invariant.name}: {invariant.error_message}")
                    else:
                        warnings.append(f"{invariant.name}: {invariant.error_message}")
            except Exception as e:
                violations.append(f"{invariant.name}: Validation error - {str(e)}")
        
        # Extract metrics
        metrics = output_data.get('mathematical_metrics', {})
        
        return StageValidationResult(
            stage=self.stage_type,
            success=len(violations) == 0,
            violations=violations,
            warnings=warnings,
            metrics=metrics
        )


class MathStage2ContextEnhancer(MathematicalEnhancer):
    """Mathematical enhancer for context construction stage"""
    
    def __init__(self):
        super().__init__(MathStageType.CONTEXT)
        self.dependencies.add(MathStageType.INGESTION)
        
        # Add context-specific invariants
        self.add_invariant(MathematicalInvariant(
            name="context_coherence",
            description="Context must maintain semantic coherence",
            validation_function=self._validate_context_coherence,
            severity=ValidationSeverity.WARNING,
            error_message="Context lacks semantic coherence"
        ))
    
    def _validate_context_coherence(self, data: Dict[str, Any]) -> bool:
        """Validate semantic coherence of context"""
        context_data = data.get('context', {})
        if not context_data:
            return False
            
        # Check for required context fields
        required_fields = ['text', 'metadata', 'processed_content']
        return all(field in context_data for field in required_fields)
    
    def enhance(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply mathematical enhancements to context construction"""
        logger.info("Applying mathematical enhancements to context stage")
        
        enhanced_data = input_data.copy()
        
        # Build enhanced context with mathematical properties
        context = enhanced_data.get('context', {})
        
        # Add contextual mathematical measures
        text = input_data.get('text', '')
        if text:
            # Compute contextual complexity measures
            words = text.split()
            if words:
                word_lengths = [len(word) for word in words]
                
                context_metrics = {
                    'lexical_diversity': len(set(words)) / len(words) if words else 0.0,
                    'avg_word_length': np.mean(word_lengths) if word_lengths else 0.0,
                    'word_length_variance': np.var(word_lengths) if word_lengths else 0.0,
                    'context_complexity': len(set(words)) * np.mean(word_lengths) if words else 0.0
                }
                
                enhanced_data['context'] = {
                    **context,
                    'mathematical_context_metrics': context_metrics
                }
        
        return enhanced_data
    
    def validate_output(self, output_data: Dict[str, Any]) -> StageValidationResult:
        """Validate mathematical properties of context output"""
        violations = []
        warnings = []
        
        # Validate invariants
        for invariant in self.invariants:
            try:
                if not invariant.validation_function(output_data):
                    if invariant.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]:
                        violations.append(f"{invariant.name}: {invariant.error_message}")
                    else:
                        warnings.append(f"{invariant.name}: {invariant.error_message}")
            except Exception as e:
                violations.append(f"{invariant.name}: Validation error - {str(e)}")
        
        # Extract context metrics
        context = output_data.get('context', {})
        metrics = context.get('mathematical_context_metrics', {})
        
        return StageValidationResult(
            stage=self.stage_type,
            success=len(violations) == 0,
            violations=violations,
            warnings=warnings,
            metrics=metrics
        )


class MathematicalPipelineCoordinator:
    """Coordinates mathematical enhancement stages with theorem-based validation"""
    
    def __init__(self):
        self.enhancers: Dict[MathStageType, MathematicalEnhancer] = {}
        self.execution_states: Dict[MathStageType, StageExecutionState] = {}
        self.global_invariants: List[MathematicalInvariant] = []
        self.execution_history: List[Dict[str, Any]] = []
        
        # Initialize mathematical foundations
        self.info_theory = InformationTheory()
        self.semantic_similarity = SemanticSimilarity()
        self.bayesian_inference = BayesianInference()
        
        # Register enhancers
        self._register_enhancers()
        self._setup_global_invariants()
    
    def _register_enhancers(self):
        """Register all mathematical enhancers"""
        # Register implemented enhancers
        self.enhancers[MathStageType.INGESTION] = MathStage1IngestiorEnhancer()
        self.enhancers[MathStageType.CONTEXT] = MathStage2ContextEnhancer()
        
        # Initialize execution states
        for stage_type in MathStageType:
            self.execution_states[stage_type] = StageExecutionState(
                stage=stage_type,
                status='pending'
            )
    
    def _setup_global_invariants(self):
        """Setup global mathematical invariants that apply across all stages"""
        self.global_invariants = [
            MathematicalInvariant(
                name="data_flow_continuity",
                description="Data must flow continuously between stages",
                validation_function=self._validate_data_flow_continuity,
                severity=ValidationSeverity.CRITICAL,
                error_message="Data flow discontinuity detected"
            ),
            
            MathematicalInvariant(
                name="mathematical_consistency",
                description="Mathematical metrics must remain consistent",
                validation_function=self._validate_mathematical_consistency,
                severity=ValidationSeverity.ERROR,
                error_message="Mathematical inconsistency detected"
            ),
            
            MathematicalInvariant(
                name="information_preservation",
                description="Information content must be preserved or enhanced",
                validation_function=self._validate_information_preservation,
                severity=ValidationSeverity.WARNING,
                error_message="Information loss detected"
            )
        ]
    
    def _validate_data_flow_continuity(self, data: Dict[str, Any]) -> bool:
        """Validate that data flows continuously between stages"""
        return 'text' in data or 'context' in data or 'output' in data
    
    def _validate_mathematical_consistency(self, data: Dict[str, Any]) -> bool:
        """Validate mathematical consistency across stages"""
        # Check for mathematical metrics consistency
        metrics = data.get('mathematical_metrics', {})
        if not metrics:
            return True  # No metrics to validate
            
        # Validate metric ranges
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                if np.isnan(value) or np.isinf(value):
                    return False
                    
        return True
    
    def _validate_information_preservation(self, data: Dict[str, Any]) -> bool:
        """Validate that information is preserved during enhancement"""
        # Simplified check - ensure essential data structures are present
        essential_keys = ['text', 'context', 'mathematical_metrics']
        return any(key in data for key in essential_keys)
    
    def _resolve_dependencies(self, target_stage: MathStageType) -> List[MathStageType]:
        """Resolve execution dependencies for a target stage"""
        if target_stage not in self.enhancers:
            return []
            
        enhancer = self.enhancers[target_stage]
        dependencies = list(enhancer.dependencies)
        
        # Add transitive dependencies
        all_deps = set(dependencies)
        for dep in dependencies:
            all_deps.update(self._resolve_dependencies(dep))
        
        # Sort dependencies by stage order
        stage_order = list(MathStageType)
        sorted_deps = sorted(all_deps, key=lambda x: stage_order.index(x))
        
        return sorted_deps
    
    def _create_rollback_checkpoint(self, stage: MathStageType, data: Dict[str, Any]):
        """Create a rollback checkpoint for a stage"""
        if stage in self.enhancers:
            checkpoint = self.enhancers[stage].create_checkpoint(data)
            self.execution_states[stage].rollback_checkpoint = checkpoint
            logger.info(f"Created rollback checkpoint for stage {stage.value}")
    
    def _rollback_stage(self, stage: MathStageType) -> bool:
        """Rollback a stage to its previous checkpoint"""
        try:
            state = self.execution_states[stage]
            if state.rollback_checkpoint:
# # #                 # Restore data from checkpoint  # Module not found  # Module not found  # Module not found
                state.input_data = state.rollback_checkpoint.get('data', {})
                state.output_data = {}
                state.status = 'rolled_back'
                state.error_message = None
                
                logger.info(f"Successfully rolled back stage {stage.value}")
                return True
            else:
                logger.warning(f"No rollback checkpoint available for stage {stage.value}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to rollback stage {stage.value}: {str(e)}")
            return False
    
    def _validate_stage_boundaries(self, 
                                 from_stage: MathStageType, 
                                 to_stage: MathStageType, 
                                 data: Dict[str, Any]) -> StageValidationResult:
        """Validate mathematical properties at stage boundaries"""
        violations = []
        warnings = []
        metrics = {}
        
        # Validate global invariants
        for invariant in self.global_invariants:
            try:
                if not invariant.validation_function(data):
                    if invariant.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]:
                        violations.append(f"Boundary {from_stage.value}->{to_stage.value}: {invariant.error_message}")
                    else:
                        warnings.append(f"Boundary {from_stage.value}->{to_stage.value}: {invariant.error_message}")
            except Exception as e:
                violations.append(f"Boundary validation error: {str(e)}")
        
        # Compute boundary-specific metrics
        if 'mathematical_metrics' in data:
            prev_metrics = self.execution_states[from_stage].output_data.get('mathematical_metrics', {})
            curr_metrics = data.get('mathematical_metrics', {})
            
            # Measure information preservation
            if prev_metrics and curr_metrics:
                common_keys = set(prev_metrics.keys()) & set(curr_metrics.keys())
                if common_keys:
                    preservation_ratios = []
                    for key in common_keys:
                        if prev_metrics[key] != 0:
                            ratio = curr_metrics[key] / prev_metrics[key]
                            preservation_ratios.append(ratio)
                    
                    if preservation_ratios:
                        metrics['avg_preservation_ratio'] = np.mean(preservation_ratios)
                        metrics['preservation_variance'] = np.var(preservation_ratios)
        
        return StageValidationResult(
            stage=to_stage,
            success=len(violations) == 0,
            violations=violations,
            warnings=warnings,
            metrics=metrics
        )
    
    def execute_stage(self, stage: MathStageType, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single mathematical enhancement stage"""
        try:
            # Check if enhancer exists
            if stage not in self.enhancers:
                logger.warning(f"No enhancer registered for stage {stage.value}, passing through data")
                return input_data
            
            enhancer = self.enhancers[stage]
            state = self.execution_states[stage]
            
            # Update state
            state.status = 'running'
            state.input_data = input_data.copy()
            
            # Create rollback checkpoint
            self._create_rollback_checkpoint(stage, input_data)
            
            # Execute enhancement
            start_time = datetime.now()
            logger.info(f"Executing mathematical enhancement stage: {stage.value}")
            
            output_data = enhancer.enhance(input_data)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Validate output
            validation_result = enhancer.validate_output(output_data)
            
            # Update state
            state.output_data = output_data
            state.validation_result = validation_result
            state.execution_time = execution_time
            
            if validation_result.success:
                state.status = 'completed'
                logger.info(f"Successfully completed stage {stage.value} in {execution_time:.2f}s")
            else:
                state.status = 'failed'
                state.error_message = f"Validation failed: {'; '.join(validation_result.violations)}"
                logger.error(f"Stage {stage.value} validation failed: {state.error_message}")
                
                # Attempt rollback on critical failures
                critical_violations = [v for v in validation_result.violations if 'CRITICAL' in v]
                if critical_violations:
                    logger.info(f"Critical violations detected, attempting rollback for stage {stage.value}")
                    self._rollback_stage(stage)
                    raise ValueError(f"Critical validation failure in stage {stage.value}")
            
            # Record execution in history
            self.execution_history.append({
                'stage': stage.value,
                'timestamp': datetime.now().isoformat(),
                'execution_time': execution_time,
                'success': validation_result.success,
                'violations': validation_result.violations,
                'warnings': validation_result.warnings,
                'metrics': validation_result.metrics
            })
            
            return output_data
            
        except Exception as e:
            # Handle execution failure
            state = self.execution_states[stage]
            state.status = 'failed'
            state.error_message = str(e)
            
            logger.error(f"Stage {stage.value} execution failed: {str(e)}")
            logger.debug(traceback.format_exc())
            
            # Attempt rollback
            if self._rollback_stage(stage):
                logger.info(f"Rolled back stage {stage.value} due to execution failure")
            
            raise
    
    def execute_pipeline(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the complete mathematical enhancement pipeline"""
        logger.info("Starting mathematical enhancement pipeline execution")
        
        current_data = input_data.copy()
        executed_stages = []
        
        try:
            # Execute stages in dependency order
            for stage_type in MathStageType:
                # Resolve dependencies
                dependencies = self._resolve_dependencies(stage_type)
                
                # Check that all dependencies have been executed
                missing_deps = [dep for dep in dependencies if dep not in executed_stages]
                if missing_deps:
                    logger.warning(f"Stage {stage_type.value} has unmet dependencies: {[d.value for d in missing_deps]}")
                    # Skip stages with unmet dependencies or execute dependencies first
                    for dep in missing_deps:
                        if dep in MathStageType:
                            current_data = self.execute_stage(dep, current_data)
                            executed_stages.append(dep)
                
                # Validate stage boundaries if there was a previous stage
                if executed_stages:
                    prev_stage = executed_stages[-1]
                    boundary_validation = self._validate_stage_boundaries(prev_stage, stage_type, current_data)
                    
                    if not boundary_validation.success:
                        critical_violations = [v for v in boundary_validation.violations if 'CRITICAL' in v]
                        if critical_violations:
                            raise ValueError(f"Critical boundary validation failure: {'; '.join(critical_violations)}")
                
                # Execute current stage
                current_data = self.execute_stage(stage_type, current_data)
                executed_stages.append(stage_type)
            
            # Final validation
            self._validate_pipeline_completion(current_data)
            
            logger.info("Mathematical enhancement pipeline completed successfully")
            return current_data
            
        except Exception as e:
            logger.error(f"Mathematical enhancement pipeline failed: {str(e)}")
            
            # Attempt to rollback failed stages
            failed_stages = [stage for stage, state in self.execution_states.items() 
                           if state.status == 'failed']
            
            for stage in failed_stages:
                self._rollback_stage(stage)
            
            raise
    
    def _validate_pipeline_completion(self, final_data: Dict[str, Any]):
        """Validate the completed pipeline output"""
        violations = []
        
        # Validate final global invariants
        for invariant in self.global_invariants:
            try:
                if not invariant.validation_function(final_data):
                    violations.append(f"Final validation: {invariant.error_message}")
            except Exception as e:
                violations.append(f"Final validation error: {str(e)}")
        
        if violations:
            critical_violations = [v for v in violations if 'CRITICAL' in v]
            if critical_violations:
                raise ValueError(f"Critical final validation failures: {'; '.join(critical_violations)}")
            else:
                logger.warning(f"Final validation warnings: {'; '.join(violations)}")
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get comprehensive pipeline execution status"""
        status = {
            'execution_states': {},
            'overall_status': 'unknown',
            'execution_history': self.execution_history,
            'summary_metrics': {}
        }
        
        # Collect state information
        completed_stages = 0
        failed_stages = 0
        
        for stage_type, state in self.execution_states.items():
            status['execution_states'][stage_type.value] = {
                'status': state.status,
                'execution_time': state.execution_time,
                'error_message': state.error_message,
                'validation_success': state.validation_result.success if state.validation_result else None,
                'violations': state.validation_result.violations if state.validation_result else [],
                'warnings': state.validation_result.warnings if state.validation_result else [],
                'metrics': state.validation_result.metrics if state.validation_result else {}
            }
            
            if state.status == 'completed':
                completed_stages += 1
            elif state.status == 'failed':
                failed_stages += 1
        
        # Determine overall status
        total_stages = len(MathStageType)
        if completed_stages == total_stages:
            status['overall_status'] = 'completed'
        elif failed_stages > 0:
            status['overall_status'] = 'failed'
        elif completed_stages > 0:
            status['overall_status'] = 'partial'
        else:
            status['overall_status'] = 'pending'
        
        # Summary metrics
        total_execution_time = sum(state.execution_time for state in self.execution_states.values())
        status['summary_metrics'] = {
            'total_execution_time': total_execution_time,
            'completed_stages': completed_stages,
            'failed_stages': failed_stages,
            'success_rate': completed_stages / total_stages if total_stages > 0 else 0.0
        }
        
        return status
    
    def reset_pipeline(self):
        """Reset pipeline to initial state"""
        logger.info("Resetting mathematical pipeline coordinator")
        
        for stage_type in MathStageType:
            self.execution_states[stage_type] = StageExecutionState(
                stage=stage_type,
                status='pending'
            )
        
        self.execution_history.clear()
    
    def integrate_with_comprehensive_orchestrator(self, 
                                                comprehensive_data: Dict[str, Any]) -> Dict[str, Any]:
        """Integration point for comprehensive_pipeline_orchestrator.py"""
        logger.info("Integrating mathematical enhancements with comprehensive orchestrator")
        
        try:
            # Extract relevant data for mathematical enhancement
            math_input = {
                'text': comprehensive_data.get('text', ''),
                'context': comprehensive_data.get('context', {}),
                'metadata': comprehensive_data.get('metadata', {}),
                'stage_data': comprehensive_data.get('stage_data', {})
            }
            
            # Execute mathematical enhancement pipeline
            enhanced_data = self.execute_pipeline(math_input)
            
            # Merge enhanced data back into comprehensive data
            result = comprehensive_data.copy()
            result.update({
                'mathematical_enhancements': enhanced_data,
                'mathematical_pipeline_status': self.get_pipeline_status()
            })
            
            logger.info("Successfully integrated mathematical enhancements")
            return result
            
        except Exception as e:
            logger.error(f"Failed to integrate mathematical enhancements: {str(e)}")
            
            # Return original data with error information
            result = comprehensive_data.copy()
            result.update({
                'mathematical_enhancements': {},
                'mathematical_pipeline_error': str(e),
                'mathematical_pipeline_status': self.get_pipeline_status()
            })
            
            return result


# Convenience function for integration
def create_mathematical_pipeline_coordinator() -> MathematicalPipelineCoordinator:
    """Factory function to create a mathematical pipeline coordinator"""
    return MathematicalPipelineCoordinator()


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create coordinator
    coordinator = create_mathematical_pipeline_coordinator()
    
    # Example input data
    test_data = {
        'text': 'This is a sample text for mathematical enhancement testing.',
        'metadata': {'source': 'test', 'timestamp': '2024-01-01'},
        'context': {}
    }
    
    try:
        # Execute pipeline
        result = coordinator.execute_pipeline(test_data)
        
        # Display results
        print("Pipeline execution completed successfully")
        print(f"Enhanced data keys: {list(result.keys())}")
        
        # Display status
        status = coordinator.get_pipeline_status()
        print(f"Overall status: {status['overall_status']}")
        print(f"Success rate: {status['summary_metrics']['success_rate']:.2%}")
        
    except Exception as e:
        print(f"Pipeline execution failed: {str(e)}")
        
        # Display status for debugging
        status = coordinator.get_pipeline_status()
        print(f"Pipeline status: {status['overall_status']}")
        
        for stage, state in status['execution_states'].items():
            if state['status'] == 'failed':
                print(f"Failed stage {stage}: {state['error_message']}")
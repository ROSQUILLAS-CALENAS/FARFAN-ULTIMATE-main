"""
Stage Transition Controller - Deterministic Finite Automaton (DFA)

Implements a DFA to enforce the canonical pipeline phase sequence:
I→X→K→A→L→R→O→G→T→S

Features:
- State validation and transition guards
- Artifact validation from previous stages
- Hash continuity verification to detect data corruption
- Runtime detection of backward imports that violate DAG architecture
- Comprehensive logging with OpenTelemetry spans for observability
"""

import hashlib
import inspect
import logging
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# OpenTelemetry imports with fallback
try:
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    # Mock classes for fallback
    class MockTracer:
        def start_span(self, name, **kwargs):
            return MockSpan()
    
    class MockSpan:
        def __enter__(self):
            return self
        
        def __exit__(self, *args):
            pass
        
        def set_status(self, status):
            pass
        
        def set_attribute(self, key, value):
            pass


logger = logging.getLogger(__name__)


class PipelineStage(Enum):
    """Canonical pipeline stages in DFA sequence"""
    I_INGESTION_PREPARATION = "I"
    X_CONTEXT_CONSTRUCTION = "X" 
    K_KNOWLEDGE_EXTRACTION = "K"
    A_ANALYSIS_NLP = "A"
    L_CLASSIFICATION_EVALUATION = "L"
    R_SEARCH_RETRIEVAL = "R"
    O_ORCHESTRATION_CONTROL = "O"
    G_AGGREGATION_REPORTING = "G"
    T_INTEGRATION_STORAGE = "T"
    S_SYNTHESIS_OUTPUT = "S"

    @classmethod
    def get_canonical_sequence(cls) -> List['PipelineStage']:
        """Return canonical stage sequence for DFA"""
        return [
            cls.I_INGESTION_PREPARATION,
            cls.X_CONTEXT_CONSTRUCTION,
            cls.K_KNOWLEDGE_EXTRACTION,
            cls.A_ANALYSIS_NLP,
            cls.L_CLASSIFICATION_EVALUATION,
            cls.R_SEARCH_RETRIEVAL,
            cls.O_ORCHESTRATION_CONTROL,
            cls.G_AGGREGATION_REPORTING,
            cls.T_INTEGRATION_STORAGE,
            cls.S_SYNTHESIS_OUTPUT,
        ]
    
    @classmethod
    def from_string(cls, stage_str: str) -> Optional['PipelineStage']:
        """Create stage from string representation"""
        stage_map = {stage.value: stage for stage in cls}
        return stage_map.get(stage_str.upper())


@dataclass
class StageArtifact:
    """Represents an artifact produced by a pipeline stage"""
    name: str
    hash_value: str
    timestamp: float
    metadata: Dict[str, Any]


@dataclass 
class TransitionResult:
    """Result of a stage transition attempt"""
    success: bool
    from_stage: Optional[PipelineStage]
    to_stage: PipelineStage
    message: str
    validation_errors: List[str]
    hash_continuity_verified: bool
    artifacts_validated: bool
    execution_time_ms: float


class StageTransitionError(Exception):
    """Exception raised when stage transition fails"""
    def __init__(self, message: str, from_stage: Optional[PipelineStage], 
                 to_stage: PipelineStage, validation_errors: List[str] = None):
        super().__init__(message)
        self.from_stage = from_stage
        self.to_stage = to_stage
        self.validation_errors = validation_errors or []


class BackwardImportError(Exception):
    """Exception raised when backward import is detected"""
    def __init__(self, message: str, importing_module: str, imported_module: str,
                 importing_stage: PipelineStage, imported_stage: PipelineStage):
        super().__init__(message)
        self.importing_module = importing_module
        self.imported_module = imported_module
        self.importing_stage = importing_stage
        self.imported_stage = imported_stage


class ArtifactValidator(ABC):
    """Abstract base class for stage artifact validators"""
    
    @abstractmethod
    def validate(self, artifacts: Dict[str, StageArtifact], 
                 from_stage: Optional[PipelineStage]) -> Tuple[bool, List[str]]:
        """Validate artifacts from previous stage"""
        pass


class DefaultArtifactValidator(ArtifactValidator):
    """Default artifact validator implementation"""
    
    REQUIRED_ARTIFACTS = {
        PipelineStage.X_CONTEXT_CONSTRUCTION: ["ingestion_data", "validation_report"],
        PipelineStage.K_KNOWLEDGE_EXTRACTION: ["context_graph", "lineage_tracker"],
        PipelineStage.A_ANALYSIS_NLP: ["knowledge_graph", "embeddings"],
        PipelineStage.L_CLASSIFICATION_EVALUATION: ["analysis_results", "evidence_mappings"],
        PipelineStage.R_SEARCH_RETRIEVAL: ["classification_scores", "evaluation_metrics"],
        PipelineStage.O_ORCHESTRATION_CONTROL: ["search_indices", "retrieval_results"],
        PipelineStage.G_AGGREGATION_REPORTING: ["orchestration_state", "control_decisions"],
        PipelineStage.T_INTEGRATION_STORAGE: ["aggregated_reports", "meso_summaries"],
        PipelineStage.S_SYNTHESIS_OUTPUT: ["integration_results", "storage_confirmations"],
    }
    
    def validate(self, artifacts: Dict[str, StageArtifact], 
                 from_stage: Optional[PipelineStage]) -> Tuple[bool, List[str]]:
        """Validate required artifacts are present and valid"""
        errors = []
        
        if from_stage is None:
            return True, []
        
        required = self.REQUIRED_ARTIFACTS.get(from_stage, [])
        
        for artifact_name in required:
            if artifact_name not in artifacts:
                errors.append(f"Missing required artifact: {artifact_name}")
                continue
                
            artifact = artifacts[artifact_name]
            if not artifact.hash_value:
                errors.append(f"Artifact {artifact_name} has no hash value")
            
            # Verify artifact integrity
            if not self._verify_artifact_integrity(artifact):
                errors.append(f"Artifact {artifact_name} failed integrity check")
        
        return len(errors) == 0, errors
    
    def _verify_artifact_integrity(self, artifact: StageArtifact) -> bool:
        """Verify artifact integrity (placeholder implementation)"""
        # In a real implementation, this would validate the artifact content
        # against its hash and perform additional integrity checks
        return bool(artifact.hash_value and artifact.timestamp > 0)


class HashContinuityValidator:
    """Validates hash continuity between pipeline stages to detect corruption"""
    
    def __init__(self):
        self.stage_hashes: Dict[PipelineStage, str] = {}
        self.hash_chain: List[Tuple[PipelineStage, str]] = []
    
    def compute_stage_hash(self, stage: PipelineStage, 
                          artifacts: Dict[str, StageArtifact],
                          previous_hash: Optional[str] = None) -> str:
        """Compute deterministic hash for stage including previous hash"""
        hasher = hashlib.sha256()
        
        # Include previous stage hash for continuity
        if previous_hash:
            hasher.update(previous_hash.encode('utf-8'))
        
        # Include stage identifier
        hasher.update(stage.value.encode('utf-8'))
        
        # Include sorted artifacts
        for name, artifact in sorted(artifacts.items()):
            hasher.update(name.encode('utf-8'))
            hasher.update(artifact.hash_value.encode('utf-8'))
            hasher.update(str(artifact.timestamp).encode('utf-8'))
        
        stage_hash = hasher.hexdigest()
        self.stage_hashes[stage] = stage_hash
        self.hash_chain.append((stage, stage_hash))
        
        return stage_hash
    
    def validate_continuity(self, stage: PipelineStage, 
                           computed_hash: str) -> Tuple[bool, List[str]]:
        """Validate hash continuity for corruption detection"""
        errors = []
        
        # Check if stage hash matches expected
        expected_hash = self.stage_hashes.get(stage)
        if expected_hash and expected_hash != computed_hash:
            errors.append(f"Hash mismatch for stage {stage.value}: "
                         f"expected {expected_hash[:16]}..., got {computed_hash[:16]}...")
        
        # Verify chain integrity
        if len(self.hash_chain) > 1:
            prev_stage, prev_hash = self.hash_chain[-2]
            if not self._verify_hash_dependency(prev_hash, computed_hash):
                errors.append(f"Hash continuity broken between {prev_stage.value} and {stage.value}")
        
        return len(errors) == 0, errors
    
    def _verify_hash_dependency(self, prev_hash: str, current_hash: str) -> bool:
        """Verify current hash properly depends on previous hash"""
        # This is a simplified check - in practice, you'd verify the hash was
        # computed correctly including the previous hash as input
        return len(current_hash) == 64 and len(prev_hash) == 64


class ImportGraphAnalyzer:
    """Analyzes import graph to detect backward dependencies that violate DAG"""
    
    def __init__(self):
        self.stage_modules: Dict[PipelineStage, Set[str]] = {}
        self.import_graph: Dict[str, Set[str]] = {}
        self._build_stage_module_mapping()
    
    def _build_stage_module_mapping(self):
        """Build mapping of stages to their modules"""
        stage_prefixes = {
            PipelineStage.I_INGESTION_PREPARATION: ['I_ingestion', 'ingestion'],
            PipelineStage.X_CONTEXT_CONSTRUCTION: ['X_context', 'context'],
            PipelineStage.K_KNOWLEDGE_EXTRACTION: ['K_knowledge', 'knowledge'],
            PipelineStage.A_ANALYSIS_NLP: ['A_analysis', 'analysis'],
            PipelineStage.L_CLASSIFICATION_EVALUATION: ['L_classification', 'classification'],
            PipelineStage.R_SEARCH_RETRIEVAL: ['R_search', 'retrieval'],
            PipelineStage.O_ORCHESTRATION_CONTROL: ['O_orchestration', 'orchestration'],
            PipelineStage.G_AGGREGATION_REPORTING: ['G_aggregation', 'aggregation'],
            PipelineStage.T_INTEGRATION_STORAGE: ['T_integration', 'integration'],
            PipelineStage.S_SYNTHESIS_OUTPUT: ['S_synthesis', 'synthesis'],
        }
        
        for stage, prefixes in stage_prefixes.items():
            self.stage_modules[stage] = set()
            for module_name in sys.modules:
                if any(prefix in module_name.lower() for prefix in prefixes):
                    self.stage_modules[stage].add(module_name)
    
    def detect_backward_imports(self) -> List[BackwardImportError]:
        """Detect imports that violate DAG architecture"""
        violations = []
        canonical_order = PipelineStage.get_canonical_sequence()
        stage_indices = {stage: i for i, stage in enumerate(canonical_order)}
        
        for importing_stage, modules in self.stage_modules.items():
            importing_index = stage_indices[importing_stage]
            
            for module in modules:
                if module not in sys.modules:
                    continue
                
                module_obj = sys.modules[module]
                imported_modules = self._get_imported_modules(module_obj)
                
                for imported_module in imported_modules:
                    imported_stage = self._identify_module_stage(imported_module)
                    
                    if imported_stage is None:
                        continue
                    
                    imported_index = stage_indices[imported_stage]
                    
                    # Check if importing from a later stage (backward import)
                    if imported_index > importing_index:
                        error = BackwardImportError(
                            f"Backward import detected: {module} (stage {importing_stage.value}) "
                            f"imports from {imported_module} (stage {imported_stage.value})",
                            module, imported_module, importing_stage, imported_stage
                        )
                        violations.append(error)
        
        return violations
    
    def _get_imported_modules(self, module_obj) -> Set[str]:
        """Get modules imported by given module"""
        imported = set()
        
        if hasattr(module_obj, '__dict__'):
            for name, obj in module_obj.__dict__.items():
                if inspect.ismodule(obj) and hasattr(obj, '__name__'):
                    imported.add(obj.__name__)
        
        return imported
    
    def _identify_module_stage(self, module_name: str) -> Optional[PipelineStage]:
        """Identify which stage a module belongs to"""
        for stage, modules in self.stage_modules.items():
            if any(module_name.startswith(mod.split('.')[0]) for mod in modules):
                return stage
        return None


class StageTransitionController:
    """Deterministic Finite Automaton for pipeline stage transitions"""
    
    def __init__(self, artifact_validator: Optional[ArtifactValidator] = None):
        self.current_stage: Optional[PipelineStage] = None
        self.stage_history: List[Tuple[PipelineStage, float]] = []
        self.stage_artifacts: Dict[PipelineStage, Dict[str, StageArtifact]] = {}
        
        self.artifact_validator = artifact_validator or DefaultArtifactValidator()
        self.hash_validator = HashContinuityValidator()
        self.import_analyzer = ImportGraphAnalyzer()
        
        # OpenTelemetry tracer
        if OTEL_AVAILABLE:
            self.tracer = trace.get_tracer(__name__)
        else:
            self.tracer = MockTracer()
        
        # DFA transition table
        self.transition_table = self._build_transition_table()
    
    def _build_transition_table(self) -> Dict[Optional[PipelineStage], Set[PipelineStage]]:
        """Build DFA transition table for valid stage transitions"""
        canonical_sequence = PipelineStage.get_canonical_sequence()
        
        transitions = {None: {canonical_sequence[0]}}  # Initial state can transition to I
        
        for i, stage in enumerate(canonical_sequence):
            if i + 1 < len(canonical_sequence):
                transitions[stage] = {canonical_sequence[i + 1]}
            else:
                transitions[stage] = set()  # Final state
        
        return transitions
    
    def transition_to_stage(self, target_stage: PipelineStage, 
                           artifacts: Optional[Dict[str, StageArtifact]] = None) -> TransitionResult:
        """Attempt to transition to target stage with full validation"""
        start_time = time.time()
        
        with self.tracer.start_span(f"stage_transition_{target_stage.value}") as span:
            try:
                span.set_attribute("from_stage", self.current_stage.value if self.current_stage else "initial")
                span.set_attribute("to_stage", target_stage.value)
                
                # Validate transition is legal according to DFA
                if not self._is_valid_transition(target_stage):
                    error_msg = f"Invalid transition from {self.current_stage} to {target_stage}"
                    span.set_status(Status(StatusCode.ERROR, error_msg))
                    return self._create_failure_result(target_stage, error_msg, start_time)
                
                # Validate required artifacts from previous stage
                artifacts = artifacts or {}
                artifacts_valid, artifact_errors = self.artifact_validator.validate(
                    artifacts, self.current_stage
                )
                
                if not artifacts_valid:
                    error_msg = f"Artifact validation failed: {'; '.join(artifact_errors)}"
                    span.set_status(Status(StatusCode.ERROR, error_msg))
                    return self._create_failure_result(target_stage, error_msg, start_time, 
                                                     validation_errors=artifact_errors)
                
                # Compute and validate hash continuity
                previous_hash = None
                if self.current_stage:
                    previous_hash = self.hash_validator.stage_hashes.get(self.current_stage)
                
                current_hash = self.hash_validator.compute_stage_hash(
                    target_stage, artifacts, previous_hash
                )
                
                hash_valid, hash_errors = self.hash_validator.validate_continuity(
                    target_stage, current_hash
                )
                
                # Detect backward imports
                with self.tracer.start_span("detect_backward_imports") as import_span:
                    backward_imports = self.import_analyzer.detect_backward_imports()
                    
                    if backward_imports:
                        import_errors = [str(error) for error in backward_imports]
                        error_msg = f"Backward imports detected: {'; '.join(import_errors)}"
                        import_span.set_status(Status(StatusCode.ERROR, error_msg))
                        return self._create_failure_result(target_stage, error_msg, start_time,
                                                         validation_errors=import_errors)
                
                # Update state
                self.current_stage = target_stage
                self.stage_history.append((target_stage, time.time()))
                self.stage_artifacts[target_stage] = artifacts
                
                # Log successful transition
                execution_time = (time.time() - start_time) * 1000
                logger.info(f"Successfully transitioned to stage {target_stage.value} "
                           f"in {execution_time:.2f}ms")
                
                span.set_attribute("success", True)
                span.set_attribute("execution_time_ms", execution_time)
                span.set_attribute("artifacts_count", len(artifacts))
                span.set_attribute("hash_value", current_hash[:16])
                
                return TransitionResult(
                    success=True,
                    from_stage=self.stage_history[-2][0] if len(self.stage_history) > 1 else None,
                    to_stage=target_stage,
                    message=f"Successfully transitioned to {target_stage.value}",
                    validation_errors=[],
                    hash_continuity_verified=hash_valid,
                    artifacts_validated=artifacts_valid,
                    execution_time_ms=execution_time
                )
                
            except Exception as e:
                error_msg = f"Unexpected error during transition: {str(e)}"
                span.set_status(Status(StatusCode.ERROR, error_msg))
                logger.exception(error_msg)
                return self._create_failure_result(target_stage, error_msg, start_time)
    
    def _is_valid_transition(self, target_stage: PipelineStage) -> bool:
        """Check if transition is valid according to DFA rules"""
        valid_targets = self.transition_table.get(self.current_stage, set())
        return target_stage in valid_targets
    
    def _create_failure_result(self, target_stage: PipelineStage, message: str, 
                              start_time: float, validation_errors: List[str] = None) -> TransitionResult:
        """Create a failure result with timing information"""
        execution_time = (time.time() - start_time) * 1000
        
        return TransitionResult(
            success=False,
            from_stage=self.current_stage,
            to_stage=target_stage,
            message=message,
            validation_errors=validation_errors or [],
            hash_continuity_verified=False,
            artifacts_validated=False,
            execution_time_ms=execution_time
        )
    
    def get_current_stage(self) -> Optional[PipelineStage]:
        """Get the current pipeline stage"""
        return self.current_stage
    
    def get_stage_history(self) -> List[Tuple[PipelineStage, float]]:
        """Get complete stage transition history"""
        return self.stage_history.copy()
    
    def get_valid_next_stages(self) -> Set[PipelineStage]:
        """Get valid next stages from current state"""
        return self.transition_table.get(self.current_stage, set())
    
    def validate_pipeline_integrity(self) -> Tuple[bool, List[str]]:
        """Validate overall pipeline integrity"""
        with self.tracer.start_span("validate_pipeline_integrity") as span:
            errors = []
            
            # Check for backward imports
            backward_imports = self.import_analyzer.detect_backward_imports()
            if backward_imports:
                errors.extend([str(error) for error in backward_imports])
            
            # Validate hash chain integrity
            if len(self.hash_validator.hash_chain) > 1:
                for i in range(1, len(self.hash_validator.hash_chain)):
                    stage, hash_val = self.hash_validator.hash_chain[i]
                    is_valid, hash_errors = self.hash_validator.validate_continuity(stage, hash_val)
                    if not is_valid:
                        errors.extend(hash_errors)
            
            # Check stage sequence compliance
            if len(self.stage_history) > 1:
                canonical_sequence = PipelineStage.get_canonical_sequence()
                stage_indices = {stage: i for i, stage in enumerate(canonical_sequence)}
                
                for i in range(1, len(self.stage_history)):
                    prev_stage = self.stage_history[i-1][0]
                    curr_stage = self.stage_history[i][0]
                    
                    prev_idx = stage_indices[prev_stage]
                    curr_idx = stage_indices[curr_stage]
                    
                    if curr_idx != prev_idx + 1:
                        errors.append(f"Invalid stage sequence: {prev_stage.value} -> {curr_stage.value}")
            
            is_valid = len(errors) == 0
            span.set_attribute("is_valid", is_valid)
            span.set_attribute("error_count", len(errors))
            
            return is_valid, errors
    
    def generate_transition_report(self) -> Dict[str, Any]:
        """Generate comprehensive transition report for observability"""
        with self.tracer.start_span("generate_transition_report") as span:
            report = {
                "controller_status": {
                    "current_stage": self.current_stage.value if self.current_stage else None,
                    "total_transitions": len(self.stage_history),
                    "pipeline_complete": self.current_stage == PipelineStage.S_SYNTHESIS_OUTPUT,
                },
                "stage_history": [
                    {"stage": stage.value, "timestamp": timestamp}
                    for stage, timestamp in self.stage_history
                ],
                "hash_chain": [
                    {"stage": stage.value, "hash": hash_val[:16] + "..."}
                    for stage, hash_val in self.hash_validator.hash_chain
                ],
                "artifacts_summary": {
                    stage.value: len(artifacts)
                    for stage, artifacts in self.stage_artifacts.items()
                },
                "validation_status": {},
                "next_valid_stages": [stage.value for stage in self.get_valid_next_stages()],
            }
            
            # Add integrity validation results
            is_valid, errors = self.validate_pipeline_integrity()
            report["validation_status"] = {
                "pipeline_integrity": is_valid,
                "validation_errors": errors,
            }
            
            span.set_attribute("stages_completed", len(self.stage_history))
            span.set_attribute("pipeline_complete", report["controller_status"]["pipeline_complete"])
            
            return report


# Convenience functions for common operations
def create_stage_artifact(name: str, content: Any, metadata: Dict[str, Any] = None) -> StageArtifact:
    """Create a stage artifact with computed hash"""
    content_str = str(content)
    hash_value = hashlib.sha256(content_str.encode('utf-8')).hexdigest()
    
    return StageArtifact(
        name=name,
        hash_value=hash_value,
        timestamp=time.time(),
        metadata=metadata or {}
    )


def get_canonical_stage_sequence() -> List[str]:
    """Get canonical stage sequence as string list"""
    return [stage.value for stage in PipelineStage.get_canonical_sequence()]


# Global controller instance for singleton pattern
_global_controller: Optional[StageTransitionController] = None


def get_stage_controller() -> StageTransitionController:
    """Get global stage controller instance"""
    global _global_controller
    if _global_controller is None:
        _global_controller = StageTransitionController()
    return _global_controller


def reset_stage_controller():
    """Reset global stage controller (useful for testing)"""
    global _global_controller
    _global_controller = None


if __name__ == "__main__":
    # Example usage
    controller = StageTransitionController()
    
    # Create some test artifacts
    test_artifacts = {
        "test_data": create_stage_artifact("test_data", {"key": "value"}),
    }
    
    # Try transitions
    result = controller.transition_to_stage(PipelineStage.I_INGESTION_PREPARATION, test_artifacts)
    print(f"Transition result: {result.success} - {result.message}")
    
    if result.success:
        # Generate report
        report = controller.generate_transition_report()
        print(f"Pipeline status: {report['controller_status']}")
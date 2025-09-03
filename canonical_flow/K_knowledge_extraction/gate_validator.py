"""
GateValidator for K_knowledge_extraction stage.

Enforces execution sequence and validates input artifacts for knowledge extraction components.
Execution sequence: 06K→07K→11K→08K→09K→10K
"""

import json
import logging
# # # from pathlib import Path  # Module not found  # Module not found  # Module not found
# # # from typing import Dict, Any, List, Optional, Tuple  # Module not found  # Module not found  # Module not found
# # # from dataclasses import dataclass  # Module not found  # Module not found  # Module not found
# # # from enum import Enum  # Module not found  # Module not found  # Module not found
import traceback


class ComponentStatus(Enum):
    """Component execution status."""
    SUCCESS = "success"
    FAILED = "failed"
    DEPENDENCY_ERROR = "dependency_error"
    PENDING = "pending"
    RUNNING = "running"


@dataclass
class ValidationResult:
    """Result of artifact validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    artifact_path: Optional[Path] = None
    
    def add_error(self, error: str):
        """Add error to validation result."""
        self.errors.append(error)
        self.is_valid = False
    
    def add_warning(self, warning: str):
        """Add warning to validation result."""
        self.warnings.append(warning)


@dataclass 
class ComponentDependency:
    """Defines a component's dependencies."""
    component_id: str
    required_artifacts: List[str]  # List of JSON artifact filenames
    schema_validators: Dict[str, callable]  # Artifact name -> validator function


class GateValidator:
    """
    Validates execution sequence and input artifacts for knowledge extraction components.
    
    Enforces the sequence: 06K→07K→11K→08K→09K→10K
    Validates required artifacts exist and have proper schema compliance.
    """
    
    def __init__(self, canonical_flow_path: Path):
        """Initialize GateValidator.
        
        Args:
            canonical_flow_path: Path to canonical_flow directory
        """
        self.canonical_flow_path = Path(canonical_flow_path)
        self.knowledge_artifacts_path = self.canonical_flow_path / "knowledge"
        self.logger = logging.getLogger(__name__)
        
        # Ensure knowledge artifacts directory exists
        self.knowledge_artifacts_path.mkdir(parents=True, exist_ok=True)
        
        # Define execution sequence and dependencies
        self._setup_dependencies()
        
    def _setup_dependencies(self):
        """Set up component dependencies and execution sequence."""
        
        # Component execution sequence: 06K→07K→11K→08K→09K→10K
        self.execution_sequence = ["06K", "07K", "11K", "08K", "09K", "10K"]
        
        # Define dependencies for each component
        self.dependencies = {
            "06K": ComponentDependency(
                component_id="06K",
                required_artifacts=[],  # First component, no dependencies
                schema_validators={}
            ),
            "07K": ComponentDependency(
                component_id="07K", 
                required_artifacts=["terms.json"],  # Needs 06K output
                schema_validators={"terms.json": self._validate_terms_schema}
            ),
            "11K": ComponentDependency(
                component_id="11K",
                required_artifacts=["terms.json", "chunks.json"],  # Needs 06K and 07K
                schema_validators={
                    "terms.json": self._validate_terms_schema,
                    "chunks.json": self._validate_chunks_schema
                }
            ),
            "08K": ComponentDependency(
                component_id="08K", 
                required_artifacts=["terms.json", "chunks.json", "concepts.json"],  # Needs 06K, 07K, 11K
                schema_validators={
                    "terms.json": self._validate_terms_schema,
                    "chunks.json": self._validate_chunks_schema,
                    "concepts.json": self._validate_concepts_schema
                }
            ),
            "09K": ComponentDependency(
                component_id="09K",
                required_artifacts=["terms.json", "chunks.json", "concepts.json", "entities.json"],
                schema_validators={
                    "terms.json": self._validate_terms_schema,
                    "chunks.json": self._validate_chunks_schema,
                    "concepts.json": self._validate_concepts_schema,
                    "entities.json": self._validate_entities_schema
                }
            ),
            "10K": ComponentDependency(
                component_id="10K",
                required_artifacts=["terms.json", "chunks.json", "concepts.json", "entities.json", "relations.json"],
                schema_validators={
                    "terms.json": self._validate_terms_schema,
                    "chunks.json": self._validate_chunks_schema, 
                    "concepts.json": self._validate_concepts_schema,
                    "entities.json": self._validate_entities_schema,
                    "relations.json": self._validate_relations_schema
                }
            )
        }
    
    def validate_component_execution(self, component_id: str, document_id: str) -> ValidationResult:
        """
        Validate that a component can execute based on dependencies.
        
        Args:
            component_id: Component identifier (06K, 07K, etc.)
            document_id: Document identifier for artifact lookup
            
        Returns:
            ValidationResult with validation status and details
        """
        result = ValidationResult(is_valid=True, errors=[], warnings=[])
        
        try:
            # Check if component is in valid execution sequence
            if component_id not in self.execution_sequence:
                result.add_error(f"Unknown component {component_id}. Valid components: {self.execution_sequence}")
                return result
            
            # Check execution order
            component_index = self.execution_sequence.index(component_id)
            result = self._validate_execution_order(component_id, component_index, document_id, result)
            
            # Validate required artifacts
            dependency = self.dependencies[component_id]
            result = self._validate_required_artifacts(dependency, document_id, result)
            
            if result.is_valid:
                self.logger.info(f"Component {component_id} validation passed for document {document_id}")
            else:
                self.logger.error(f"Component {component_id} validation failed for document {document_id}: {result.errors}")
                
        except Exception as e:
            self.logger.error(f"Validation error for component {component_id}: {e}\n{traceback.format_exc()}")
            result.add_error(f"Validation system error: {str(e)}")
            
        return result
    
    def _validate_execution_order(self, component_id: str, component_index: int, 
                                document_id: str, result: ValidationResult) -> ValidationResult:
        """Validate that prerequisites have been executed."""
        
        # Check that all previous components in sequence have completed
        for i in range(component_index):
            prerequisite_id = self.execution_sequence[i]
            prerequisite_dependency = self.dependencies[prerequisite_id]
            
            # Check if prerequisite produced its expected outputs
            if prerequisite_dependency.required_artifacts:
                # For components that should produce output, check the last required artifact
                # This is a heuristic - in practice you might track completion differently
                pass
            
            # For the first component (06K), check if terms.json exists as it should produce it
            if prerequisite_id == "06K":
                terms_path = self._get_artifact_path(document_id, "terms.json")
                if not terms_path.exists():
                    result.add_error(f"Prerequisite component {prerequisite_id} has not completed - missing terms.json")
            
            # For 07K, check if chunks.json exists
            elif prerequisite_id == "07K":
                chunks_path = self._get_artifact_path(document_id, "chunks.json") 
                if not chunks_path.exists():
                    result.add_error(f"Prerequisite component {prerequisite_id} has not completed - missing chunks.json")
                    
            # For 11K, check if concepts.json exists
            elif prerequisite_id == "11K" and component_index > 2:
                concepts_path = self._get_artifact_path(document_id, "concepts.json")
                if not concepts_path.exists():
                    result.add_error(f"Prerequisite component {prerequisite_id} has not completed - missing concepts.json")
                    
        return result
        
    def _validate_required_artifacts(self, dependency: ComponentDependency, 
                                   document_id: str, result: ValidationResult) -> ValidationResult:
        """Validate that required artifacts exist and have valid schemas."""
        
        for artifact_name in dependency.required_artifacts:
            artifact_path = self._get_artifact_path(document_id, artifact_name)
            
            # Check if artifact exists
            if not artifact_path.exists():
                result.add_error(f"Required artifact missing: {artifact_path}")
                continue
                
            # Validate JSON structure
            try:
                with open(artifact_path, 'r', encoding='utf-8') as f:
                    artifact_data = json.load(f)
            except json.JSONDecodeError as e:
                result.add_error(f"Invalid JSON in {artifact_path}: {str(e)}")
                continue
            except Exception as e:
                result.add_error(f"Error reading {artifact_path}: {str(e)}")
                continue
                
            # Validate schema
            if artifact_name in dependency.schema_validators:
                validator = dependency.schema_validators[artifact_name]
                schema_result = validator(artifact_data, artifact_path)
                result.errors.extend(schema_result.errors)
                result.warnings.extend(schema_result.warnings)
                if not schema_result.is_valid:
                    result.is_valid = False
                    
        return result
    
    def _get_artifact_path(self, document_id: str, artifact_name: str) -> Path:
        """Get path to artifact for given document."""
        return self.knowledge_artifacts_path / document_id / artifact_name
    
    def mark_component_failed(self, component_id: str, document_id: str, 
                            status: ComponentStatus = ComponentStatus.DEPENDENCY_ERROR) -> None:
        """Mark component as failed and log detailed error information."""
        
        error_info = {
            "component_id": component_id,
            "document_id": document_id,
            "status": status.value,
            "timestamp": json.dumps({"timestamp": "now"}),  # Simplified timestamp
            "error_type": "dependency_validation_failure"
        }
        
        # Log error
        self.logger.error(f"Component {component_id} marked as {status.value} for document {document_id}")
        
        # Save error info to artifacts directory
        error_path = self.knowledge_artifacts_path / document_id / f"{component_id}_error.json"
        error_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(error_path, 'w', encoding='utf-8') as f:
                json.dump(error_info, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Failed to write error file {error_path}: {e}")
    
    def can_graceful_degrade(self, component_id: str, document_id: str) -> bool:
        """
        Check if system can gracefully degrade for failed component.
        
        Allows subsequent components to continue processing other documents.
        """
        # Always allow graceful degradation - system should continue processing
        # other documents even if one component fails for a specific document
        self.logger.info(f"Enabling graceful degradation for component {component_id}, document {document_id}")
        return True
    
    # Schema validation methods
    def _validate_terms_schema(self, data: Dict[str, Any], artifact_path: Path) -> ValidationResult:
        """Validate terms.json schema."""
        result = ValidationResult(is_valid=True, errors=[], warnings=[], artifact_path=artifact_path)
        
        required_fields = ["terms", "metadata"]
        for field in required_fields:
            if field not in data:
                result.add_error(f"Missing required field: {field}")
        
        if "terms" in data:
            if not isinstance(data["terms"], list):
                result.add_error("Field 'terms' must be a list")
            else:
                for i, term in enumerate(data["terms"]):
                    if not isinstance(term, dict):
                        result.add_error(f"Term at index {i} must be an object")
                    elif "text" not in term:
                        result.add_error(f"Term at index {i} missing 'text' field")
        
        return result
    
    def _validate_chunks_schema(self, data: Dict[str, Any], artifact_path: Path) -> ValidationResult:
        """Validate chunks.json schema."""
        result = ValidationResult(is_valid=True, errors=[], warnings=[], artifact_path=artifact_path)
        
        required_fields = ["chunks", "metadata"]
        for field in required_fields:
            if field not in data:
                result.add_error(f"Missing required field: {field}")
        
        if "chunks" in data:
            if not isinstance(data["chunks"], list):
                result.add_error("Field 'chunks' must be a list")
            else:
                for i, chunk in enumerate(data["chunks"]):
                    if not isinstance(chunk, dict):
                        result.add_error(f"Chunk at index {i} must be an object")
                    elif "content" not in chunk:
                        result.add_error(f"Chunk at index {i} missing 'content' field")
        
        return result
    
    def _validate_concepts_schema(self, data: Dict[str, Any], artifact_path: Path) -> ValidationResult:
        """Validate concepts.json schema."""
        result = ValidationResult(is_valid=True, errors=[], warnings=[], artifact_path=artifact_path)
        
        required_fields = ["concepts", "metadata"]  
        for field in required_fields:
            if field not in data:
                result.add_error(f"Missing required field: {field}")
        
        if "concepts" in data:
            if not isinstance(data["concepts"], list):
                result.add_error("Field 'concepts' must be a list")
            else:
                for i, concept in enumerate(data["concepts"]):
                    if not isinstance(concept, dict):
                        result.add_error(f"Concept at index {i} must be an object")
                    elif "name" not in concept:
                        result.add_error(f"Concept at index {i} missing 'name' field")
        
        return result
    
    def _validate_entities_schema(self, data: Dict[str, Any], artifact_path: Path) -> ValidationResult:
        """Validate entities.json schema."""
        result = ValidationResult(is_valid=True, errors=[], warnings=[], artifact_path=artifact_path)
        
        required_fields = ["entities", "metadata"]
        for field in required_fields:
            if field not in data:
                result.add_error(f"Missing required field: {field}")
                
        if "entities" in data:
            if not isinstance(data["entities"], list):
                result.add_error("Field 'entities' must be a list")
            else:
                for i, entity in enumerate(data["entities"]):
                    if not isinstance(entity, dict):
                        result.add_error(f"Entity at index {i} must be an object")
                    elif "id" not in entity:
                        result.add_error(f"Entity at index {i} missing 'id' field")
        
        return result
    
    def _validate_relations_schema(self, data: Dict[str, Any], artifact_path: Path) -> ValidationResult:
        """Validate relations.json schema."""
        result = ValidationResult(is_valid=True, errors=[], warnings=[], artifact_path=artifact_path)
        
        required_fields = ["relations", "metadata"]
        for field in required_fields:
            if field not in data:
                result.add_error(f"Missing required field: {field}")
        
        if "relations" in data:
            if not isinstance(data["relations"], list):
                result.add_error("Field 'relations' must be a list")
            else:
                for i, relation in enumerate(data["relations"]):
                    if not isinstance(relation, dict):
                        result.add_error(f"Relation at index {i} must be an object")
                    elif not all(field in relation for field in ["source", "target", "type"]):
                        result.add_error(f"Relation at index {i} missing required fields (source, target, type)")
        
        return result


class KnowledgeExtractionGate:
    """
    High-level gate interface for knowledge extraction components.
    
    Provides simple interface for components to validate before execution.
    """
    
    def __init__(self, canonical_flow_path: Path):
        """Initialize gate with validator."""
        self.validator = GateValidator(canonical_flow_path)
        self.logger = logging.getLogger(__name__)
    
    def can_execute(self, component_id: str, document_id: str) -> Tuple[bool, List[str]]:
        """
        Check if component can execute.
        
        Returns:
            Tuple of (can_execute: bool, error_messages: List[str])
        """
        result = self.validator.validate_component_execution(component_id, document_id)
        
        if not result.is_valid:
            # Mark component as failed due to dependency error
            self.validator.mark_component_failed(
                component_id, 
                document_id, 
                ComponentStatus.DEPENDENCY_ERROR
            )
            
            # Enable graceful degradation
            if self.validator.can_graceful_degrade(component_id, document_id):
                self.logger.warning(
                    f"Component {component_id} failed validation for document {document_id}, "
                    "but system will continue processing other documents"
                )
        
        return result.is_valid, result.errors
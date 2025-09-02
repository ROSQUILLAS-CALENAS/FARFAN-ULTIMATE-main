"""
Comprehensive Handoff Validation System

This system enforces data integrity and schema compliance between critical pipeline 
stage pairs, providing early failure detection and detailed logging for validation issues.
"""

import json
import logging
import hashlib
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union, Type
from pathlib import Path
from dataclasses import dataclass, field

from schemas.pipeline_schemas import (
    StageType, DataIntegrityLevel, ValidationError, StageHandoffValidationResult,
    IngestionOutput, ContextOutput, KnowledgeOutput, AnalysisOutput, 
    ClassificationOutput, RetrievalOutput, DocumentMetadata,
    HANDOFF_SCHEMAS, serialize_stage_output, deserialize_stage_output
)


@dataclass
class CheckpointValidationConfig:
    """Configuration for checkpoint validation"""
    stage_pair: Tuple[StageType, StageType]
    integrity_level: DataIntegrityLevel = DataIntegrityLevel.STRICT
    enforce_schema: bool = True
    validate_cross_dependencies: bool = True
    fail_fast: bool = True
    log_warnings_as_errors: bool = False


@dataclass
class ValidationCheckpoint:
    """Checkpoint for validating stage handoffs"""
    checkpoint_id: str
    source_stage: StageType
    target_stage: StageType
    config: CheckpointValidationConfig
    validation_timestamp: datetime = field(default_factory=datetime.now)
    validation_results: List[StageHandoffValidationResult] = field(default_factory=list)


class SchemaValidator:
    """Validates data against expected schemas"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
    
    def validate_required_fields(
        self, 
        data: Dict[str, Any], 
        required_fields: List[str],
        field_path: str = ""
    ) -> List[ValidationError]:
        """Validate that all required fields are present and non-empty"""
        errors = []
        
        for field in required_fields:
            field_full_path = f"{field_path}.{field}" if field_path else field
            
            if field not in data:
                errors.append(ValidationError(
                    field_name=field_full_path,
                    error_type="missing_field",
                    error_message=f"Required field '{field}' is missing",
                    severity="error",
                    suggested_fix=f"Ensure previous stage outputs '{field}' field"
                ))
            elif data[field] is None:
                errors.append(ValidationError(
                    field_name=field_full_path,
                    error_type="null_value",
                    error_message=f"Required field '{field}' is null",
                    severity="error",
                    suggested_fix=f"Ensure previous stage provides valid value for '{field}'"
                ))
            elif isinstance(data[field], (str, list, dict)) and len(data[field]) == 0:
                errors.append(ValidationError(
                    field_name=field_full_path,
                    error_type="empty_value", 
                    error_message=f"Required field '{field}' is empty",
                    severity="warning",
                    suggested_fix=f"Verify previous stage processing logic for '{field}'"
                ))
        
        return errors
    
    def validate_metadata_consistency(
        self, 
        current_metadata: DocumentMetadata,
        expected_doc_id: Optional[str] = None,
        expected_hash: Optional[str] = None
    ) -> List[ValidationError]:
        """Validate metadata consistency across stages"""
        errors = []
        
        # Validate doc_id consistency
        if expected_doc_id and current_metadata.doc_id != expected_doc_id:
            errors.append(ValidationError(
                field_name="metadata.doc_id",
                error_type="doc_id_mismatch",
                error_message=f"Document ID mismatch: expected {expected_doc_id}, got {current_metadata.doc_id}",
                severity="error",
                suggested_fix="Ensure doc_id is preserved across all stages"
            ))
        
        # Validate content hash consistency if provided
        if expected_hash and current_metadata.content_hash != expected_hash:
            errors.append(ValidationError(
                field_name="metadata.content_hash",
                error_type="hash_mismatch", 
                error_message=f"Content hash mismatch: expected {expected_hash}, got {current_metadata.content_hash}",
                severity="error",
                suggested_fix="Verify content integrity - document may have been modified"
            ))
        
        # Validate required metadata fields
        required_metadata_fields = ["doc_id", "source_path", "content_hash", "created_timestamp", "stage_processed"]
        metadata_dict = current_metadata.__dict__
        
        for field in required_metadata_fields:
            if not getattr(current_metadata, field, None):
                errors.append(ValidationError(
                    field_name=f"metadata.{field}",
                    error_type="missing_metadata",
                    error_message=f"Required metadata field '{field}' is missing or empty",
                    severity="error",
                    suggested_fix=f"Ensure all stages populate metadata.{field}"
                ))
        
        return errors
    
    def validate_schema_compliance(
        self, 
        data: Dict[str, Any], 
        expected_schema: Type,
        integrity_level: DataIntegrityLevel = DataIntegrityLevel.STRICT
    ) -> List[ValidationError]:
        """Validate data complies with expected schema"""
        errors = []
        
        try:
            # Attempt to deserialize to schema object
            schema_obj = deserialize_stage_output(data, expected_schema)
            
            # In strict mode, validate all schema fields
            if integrity_level == DataIntegrityLevel.STRICT:
                schema_fields = schema_obj.__dataclass_fields__.keys()
                for field in schema_fields:
                    if field not in data and not hasattr(schema_obj, field):
                        errors.append(ValidationError(
                            field_name=field,
                            error_type="schema_violation",
                            error_message=f"Schema field '{field}' missing from data",
                            severity="error" if integrity_level == DataIntegrityLevel.STRICT else "warning",
                            suggested_fix=f"Ensure stage outputs include '{field}' field"
                        ))
                        
        except TypeError as e:
            errors.append(ValidationError(
                field_name="schema_validation",
                error_type="schema_deserialization_error",
                error_message=f"Failed to deserialize data to expected schema: {str(e)}",
                severity="error",
                suggested_fix="Review data structure and ensure it matches expected schema"
            ))
        except Exception as e:
            errors.append(ValidationError(
                field_name="schema_validation",
                error_type="unexpected_schema_error",
                error_message=f"Unexpected error during schema validation: {str(e)}",
                severity="error",
                suggested_fix="Check data format and schema definition"
            ))
        
        return errors


class CrossStageValidator:
    """Validates dependencies and consistency across pipeline stages"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
    
    def validate_dependency_requirements(
        self,
        source_stage: StageType,
        target_stage: StageType, 
        output_data: Dict[str, Any]
    ) -> List[ValidationError]:
        """Validate cross-stage dependency requirements"""
        errors = []
        
        # Get dependency requirements for this stage pair
        stage_pair = (source_stage, target_stage)
        if stage_pair not in HANDOFF_SCHEMAS:
            errors.append(ValidationError(
                field_name="stage_pair",
                error_type="unsupported_handoff",
                error_message=f"Handoff from {source_stage.value} to {target_stage.value} is not supported",
                severity="error",
                suggested_fix="Review pipeline configuration for valid stage transitions"
            ))
            return errors
        
        handoff_config = HANDOFF_SCHEMAS[stage_pair]
        cross_dependencies = handoff_config.get("cross_stage_dependencies", [])
        
        # Validate each cross-stage dependency
        for dependency in cross_dependencies:
            dependency_errors = self._validate_single_dependency(dependency, output_data)
            errors.extend(dependency_errors)
        
        return errors
    
    def _validate_single_dependency(self, dependency: str, data: Dict[str, Any]) -> List[ValidationError]:
        """Validate a single cross-stage dependency"""
        errors = []
        
        # Handle nested field access (e.g., "metadata.doc_id")
        field_parts = dependency.split(".")
        current_data = data
        
        try:
            for part in field_parts:
                if isinstance(current_data, dict):
                    if part not in current_data:
                        errors.append(ValidationError(
                            field_name=dependency,
                            error_type="missing_dependency",
                            error_message=f"Cross-stage dependency '{dependency}' is missing",
                            severity="error",
                            suggested_fix=f"Ensure previous stage outputs '{dependency}'"
                        ))
                        break
                    current_data = current_data[part]
                elif hasattr(current_data, part):
                    current_data = getattr(current_data, part)
                else:
                    errors.append(ValidationError(
                        field_name=dependency,
                        error_type="inaccessible_dependency",
                        error_message=f"Cannot access dependency '{dependency}' in data",
                        severity="error",
                        suggested_fix=f"Verify data structure for '{dependency}'"
                    ))
                    break
            
            # Check if final value is valid
            if not errors and (current_data is None or (isinstance(current_data, (str, list, dict)) and len(current_data) == 0)):
                errors.append(ValidationError(
                    field_name=dependency,
                    error_type="empty_dependency",
                    error_message=f"Cross-stage dependency '{dependency}' is empty or null",
                    severity="warning",
                    suggested_fix=f"Verify previous stage processing for '{dependency}'"
                ))
                
        except Exception as e:
            errors.append(ValidationError(
                field_name=dependency,
                error_type="dependency_access_error",
                error_message=f"Error accessing dependency '{dependency}': {str(e)}",
                severity="error",
                suggested_fix=f"Check data structure and field names for '{dependency}'"
            ))
        
        return errors
    
    def validate_artifact_presence(
        self, 
        data: Dict[str, Any], 
        expected_artifacts: List[str]
    ) -> List[ValidationError]:
        """Validate presence of expected artifacts"""
        errors = []
        
        for artifact in expected_artifacts:
            if artifact not in data:
                errors.append(ValidationError(
                    field_name=artifact,
                    error_type="missing_artifact",
                    error_message=f"Expected artifact '{artifact}' is missing",
                    severity="error",
                    suggested_fix=f"Ensure previous stage generates '{artifact}'"
                ))
            elif not data[artifact]:
                errors.append(ValidationError(
                    field_name=artifact,
                    error_type="empty_artifact",
                    error_message=f"Expected artifact '{artifact}' is empty",
                    severity="warning",
                    suggested_fix=f"Verify previous stage processing for '{artifact}'"
                ))
        
        return errors


class HandoffValidationSystem:
    """Main system for validating pipeline stage handoffs"""
    
    def __init__(self, log_level: int = logging.INFO):
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.logger.setLevel(log_level)
        
        self.schema_validator = SchemaValidator()
        self.cross_stage_validator = CrossStageValidator()
        self.validation_history: List[StageHandoffValidationResult] = []
        
        # Checkpoint configurations for critical handoffs
        self.checkpoint_configs = {
            (StageType.INGESTION_PREPARATION, StageType.CONTEXT_ESTABLISHMENT): CheckpointValidationConfig(
                stage_pair=(StageType.INGESTION_PREPARATION, StageType.CONTEXT_ESTABLISHMENT),
                integrity_level=DataIntegrityLevel.STRICT,
                fail_fast=True
            ),
            (StageType.KNOWLEDGE_EXTRACTION, StageType.ANALYSIS_NLP): CheckpointValidationConfig(
                stage_pair=(StageType.KNOWLEDGE_EXTRACTION, StageType.ANALYSIS_NLP),
                integrity_level=DataIntegrityLevel.STRICT,
                fail_fast=True
            ),
            (StageType.CLASSIFICATION_EVALUATION, StageType.SEARCH_RETRIEVAL): CheckpointValidationConfig(
                stage_pair=(StageType.CLASSIFICATION_EVALUATION, StageType.SEARCH_RETRIEVAL),
                integrity_level=DataIntegrityLevel.STRICT,
                fail_fast=True
            )
        }
    
    def validate_handoff(
        self,
        source_stage: StageType,
        target_stage: StageType,
        output_data: Dict[str, Any],
        config: Optional[CheckpointValidationConfig] = None
    ) -> StageHandoffValidationResult:
        """Validate a complete stage handoff"""
        
        if config is None:
            config = self.checkpoint_configs.get((source_stage, target_stage))
            if config is None:
                # Create default config
                config = CheckpointValidationConfig(
                    stage_pair=(source_stage, target_stage),
                    integrity_level=DataIntegrityLevel.MODERATE
                )
        
        self.logger.info(f"Validating handoff from {source_stage.value} to {target_stage.value}")
        
        # Initialize validation result
        validation_result = StageHandoffValidationResult(
            source_stage=source_stage,
            target_stage=target_stage,
            is_valid=True,
            data_integrity_level=config.integrity_level
        )
        
        try:
            # 1. Validate mandatory fields exist
            self._validate_mandatory_fields(source_stage, target_stage, output_data, validation_result)
            
            # 2. Validate schema compliance
            if config.enforce_schema:
                self._validate_schema_compliance(source_stage, target_stage, output_data, validation_result, config)
            
            # 3. Validate cross-stage dependencies
            if config.validate_cross_dependencies:
                self._validate_cross_stage_dependencies(source_stage, target_stage, output_data, validation_result)
            
            # 4. Validate metadata consistency
            self._validate_metadata_consistency(output_data, validation_result)
            
            # Determine overall validity
            has_errors = any(error.severity == "error" for error in validation_result.validation_errors)
            has_warnings = any(error.severity == "warning" for error in validation_result.validation_errors)
            
            if has_errors:
                validation_result.is_valid = False
            elif has_warnings and config.log_warnings_as_errors:
                validation_result.is_valid = False
            
            # Log results
            self._log_validation_results(validation_result, config.fail_fast)
            
            # Store validation history
            self.validation_history.append(validation_result)
            
            # Fail fast if configured and validation failed
            if config.fail_fast and not validation_result.is_valid:
                error_messages = [error.error_message for error in validation_result.validation_errors 
                                if error.severity == "error"]
                raise HandoffValidationError(
                    f"Critical handoff validation failure from {source_stage.value} to {target_stage.value}: {'; '.join(error_messages)}",
                    validation_result
                )
                
        except HandoffValidationError:
            raise  # Re-raise validation errors
        except Exception as e:
            self.logger.error(f"Unexpected error during handoff validation: {str(e)}")
            validation_result.is_valid = False
            validation_result.validation_errors.append(ValidationError(
                field_name="system_error",
                error_type="validation_system_error",
                error_message=f"Validation system error: {str(e)}",
                severity="error"
            ))
        
        return validation_result
    
    def _validate_mandatory_fields(
        self,
        source_stage: StageType,
        target_stage: StageType, 
        output_data: Dict[str, Any],
        validation_result: StageHandoffValidationResult
    ):
        """Validate mandatory fields exist in stage outputs"""
        stage_pair = (source_stage, target_stage)
        
        if stage_pair in HANDOFF_SCHEMAS:
            handoff_config = HANDOFF_SCHEMAS[stage_pair]
            required_fields = handoff_config.get("required_fields", [])
            
            field_errors = self.schema_validator.validate_required_fields(output_data, required_fields)
            validation_result.validation_errors.extend(field_errors)
            
            # Track missing fields separately
            validation_result.missing_fields = [
                error.field_name for error in field_errors 
                if error.error_type in ["missing_field", "null_value"]
            ]
    
    def _validate_schema_compliance(
        self,
        source_stage: StageType,
        target_stage: StageType,
        output_data: Dict[str, Any],
        validation_result: StageHandoffValidationResult,
        config: CheckpointValidationConfig
    ):
        """Validate output data complies with expected schema"""
        stage_pair = (source_stage, target_stage)
        
        if stage_pair in HANDOFF_SCHEMAS:
            handoff_config = HANDOFF_SCHEMAS[stage_pair]
            input_schema = handoff_config.get("input_schema")
            
            if input_schema:
                schema_errors = self.schema_validator.validate_schema_compliance(
                    output_data, input_schema, config.integrity_level
                )
                validation_result.validation_errors.extend(schema_errors)
                
                # Track schema violations
                validation_result.schema_violations = [
                    error.error_message for error in schema_errors
                    if error.error_type.startswith("schema")
                ]
    
    def _validate_cross_stage_dependencies(
        self,
        source_stage: StageType,
        target_stage: StageType,
        output_data: Dict[str, Any], 
        validation_result: StageHandoffValidationResult
    ):
        """Validate cross-stage dependency requirements"""
        dependency_errors = self.cross_stage_validator.validate_dependency_requirements(
            source_stage, target_stage, output_data
        )
        validation_result.validation_errors.extend(dependency_errors)
        
        # Track dependency errors separately
        validation_result.cross_stage_dependency_errors = [
            error.error_message for error in dependency_errors
            if error.error_type.endswith("dependency") or error.error_type.endswith("artifact")
        ]
    
    def _validate_metadata_consistency(
        self,
        output_data: Dict[str, Any],
        validation_result: StageHandoffValidationResult
    ):
        """Validate metadata consistency"""
        if "metadata" in output_data:
            try:
                if isinstance(output_data["metadata"], dict):
                    metadata = DocumentMetadata(**output_data["metadata"])
                else:
                    metadata = output_data["metadata"]
                
                metadata_errors = self.schema_validator.validate_metadata_consistency(metadata)
                validation_result.validation_errors.extend(metadata_errors)
                
            except Exception as e:
                validation_result.validation_errors.append(ValidationError(
                    field_name="metadata",
                    error_type="metadata_parsing_error",
                    error_message=f"Failed to parse metadata: {str(e)}",
                    severity="error",
                    suggested_fix="Ensure metadata follows DocumentMetadata schema"
                ))
    
    def _log_validation_results(
        self, 
        validation_result: StageHandoffValidationResult, 
        fail_fast: bool
    ):
        """Log validation results with appropriate level"""
        stage_pair = f"{validation_result.source_stage.value} → {validation_result.target_stage.value}"
        
        if validation_result.is_valid:
            self.logger.info(f"✅ Handoff validation PASSED for {stage_pair}")
        else:
            error_count = len([e for e in validation_result.validation_errors if e.severity == "error"])
            warning_count = len([e for e in validation_result.validation_errors if e.severity == "warning"])
            
            self.logger.error(f"❌ Handoff validation FAILED for {stage_pair} - {error_count} errors, {warning_count} warnings")
            
            # Log individual errors
            for error in validation_result.validation_errors:
                if error.severity == "error":
                    self.logger.error(f"  ❌ {error.field_name}: {error.error_message}")
                    if error.suggested_fix:
                        self.logger.error(f"     💡 Suggested fix: {error.suggested_fix}")
                elif error.severity == "warning":
                    self.logger.warning(f"  ⚠️  {error.field_name}: {error.error_message}")
    
    def create_checkpoint(
        self,
        source_stage: StageType,
        target_stage: StageType,
        output_data: Dict[str, Any]
    ) -> ValidationCheckpoint:
        """Create validation checkpoint at critical handoff points"""
        checkpoint_id = f"{source_stage.value}_to_{target_stage.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        config = self.checkpoint_configs.get((source_stage, target_stage))
        if config is None:
            config = CheckpointValidationConfig(stage_pair=(source_stage, target_stage))
        
        validation_result = self.validate_handoff(source_stage, target_stage, output_data, config)
        
        checkpoint = ValidationCheckpoint(
            checkpoint_id=checkpoint_id,
            source_stage=source_stage,
            target_stage=target_stage,
            config=config,
            validation_results=[validation_result]
        )
        
        self.logger.info(f"Created validation checkpoint: {checkpoint_id}")
        return checkpoint
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of all validations performed"""
        total_validations = len(self.validation_history)
        successful_validations = len([v for v in self.validation_history if v.is_valid])
        failed_validations = total_validations - successful_validations
        
        # Group by stage pairs
        stage_pair_stats = {}
        for validation in self.validation_history:
            stage_pair = f"{validation.source_stage.value} → {validation.target_stage.value}"
            if stage_pair not in stage_pair_stats:
                stage_pair_stats[stage_pair] = {"total": 0, "successful": 0, "failed": 0}
            
            stage_pair_stats[stage_pair]["total"] += 1
            if validation.is_valid:
                stage_pair_stats[stage_pair]["successful"] += 1
            else:
                stage_pair_stats[stage_pair]["failed"] += 1
        
        return {
            "total_validations": total_validations,
            "successful_validations": successful_validations,
            "failed_validations": failed_validations,
            "success_rate": successful_validations / total_validations if total_validations > 0 else 0,
            "stage_pair_statistics": stage_pair_stats,
            "last_validation": self.validation_history[-1].validation_timestamp if self.validation_history else None
        }


class HandoffValidationError(Exception):
    """Exception raised when handoff validation fails"""
    
    def __init__(self, message: str, validation_result: StageHandoffValidationResult):
        super().__init__(message)
        self.validation_result = validation_result


# Convenience functions for common validation scenarios

def validate_ingestion_to_context(output_data: Dict[str, Any]) -> StageHandoffValidationResult:
    """Validate handoff from I_ingestion_preparation to C_context_establishment"""
    system = HandoffValidationSystem()
    return system.validate_handoff(
        StageType.INGESTION_PREPARATION,
        StageType.CONTEXT_ESTABLISHMENT, 
        output_data
    )


def validate_knowledge_to_analysis(output_data: Dict[str, Any]) -> StageHandoffValidationResult:
    """Validate handoff from K_knowledge_extraction to A_analysis_nlp"""
    system = HandoffValidationSystem()
    return system.validate_handoff(
        StageType.KNOWLEDGE_EXTRACTION,
        StageType.ANALYSIS_NLP,
        output_data
    )


def validate_classification_to_retrieval(output_data: Dict[str, Any]) -> StageHandoffValidationResult:
    """Validate handoff from L_classification_evaluation to S_search_retrieval"""
    system = HandoffValidationSystem()
    return system.validate_handoff(
        StageType.CLASSIFICATION_EVALUATION,
        StageType.SEARCH_RETRIEVAL,
        output_data
    )


def create_checkpoint_validator() -> HandoffValidationSystem:
    """Create a pre-configured handoff validation system with strict checkpoints"""
    return HandoffValidationSystem(log_level=logging.INFO)


if __name__ == "__main__":
    # Demo usage
    from datetime import datetime
    
    logging.basicConfig(level=logging.INFO)
    
    # Example: Validate ingestion to context handoff
    sample_ingestion_output = {
        "metadata": {
            "doc_id": "test_doc_001",
            "source_path": "/data/test.pdf",
            "content_hash": "abc123def456",
            "created_timestamp": datetime.now().isoformat(),
            "stage_processed": "I_ingestion_preparation",
            "content_length": 1500
        },
        "extracted_text": "Sample extracted text content...",
        "document_structure": {"pages": 5, "sections": 3},
        "page_count": 5,
        "extraction_confidence": 0.95,
        "content_blocks": [{"type": "paragraph", "content": "Block 1"}],
        "semantic_markers": ["introduction", "conclusion"]
    }
    
    validator = create_checkpoint_validator()
    result = validate_ingestion_to_context(sample_ingestion_output)
    
    print(f"\n=== Validation Result ===")
    print(f"Valid: {result.is_valid}")
    print(f"Errors: {len(result.validation_errors)}")
    
    if result.validation_errors:
        print("\nValidation Issues:")
        for error in result.validation_errors:
            print(f"  {error.severity}: {error.field_name} - {error.error_message}")
    
    print(f"\n=== Validation Summary ===")
    summary = validator.get_validation_summary() 
    for key, value in summary.items():
        print(f"{key}: {value}")
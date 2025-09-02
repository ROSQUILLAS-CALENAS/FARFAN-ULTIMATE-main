"""
Centralized Schema Registry for Pipeline Stage Validation

This module provides a comprehensive schema registry that manages typed Pydantic models
for all pipeline stages, implements version compatibility checking, and provides 
automatic schema evolution support.
"""

import json
import logging
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
try:
    from packaging import version
except ImportError:
    # Fallback for version comparison if packaging is not available
    version = None
import hashlib

from pydantic import BaseModel, Field, ValidationError, validator
from pydantic.version import VERSION as PYDANTIC_VERSION


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PipelineStage(str, Enum):
    """Pipeline stages as identified in the comprehensive orchestrator."""
    INGESTION = "ingestion_preparation"
    CONTEXT_BUILD = "context_construction" 
    KNOWLEDGE = "knowledge_extraction"
    ANALYSIS = "analysis_nlp"
    CLASSIFICATION = "classification_evaluation"
    ROUTING = "routing_decision"
    SEARCH = "search_retrieval"
    ORCHESTRATION = "orchestration_control"
    MONITORING = "monitoring_validation"
    VALIDATION = "validation_contracts"
    AGGREGATION = "aggregation_reporting"
    INTEGRATION = "integration_storage"


class SchemaVersion(BaseModel):
    """Version information for schemas."""
    major: int = Field(..., ge=0)
    minor: int = Field(..., ge=0) 
    patch: int = Field(..., ge=0)
    
    @property
    def version_string(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"
    
    def __str__(self) -> str:
        return self.version_string
    
    def is_compatible_with(self, other: "SchemaVersion") -> bool:
        """Check if this version is backward compatible with another."""
        if self.major != other.major:
            return False
        if self.minor < other.minor:
            return False
        return True


@dataclass
class CompatibilityResult:
    """Result of schema compatibility check."""
    compatible: bool
    issues: List[str] = field(default_factory=list)
    migration_path: Optional[Dict[str, Any]] = None


# Base schema models for each pipeline stage
class BaseStageSchema(BaseModel):
    """Base class for all stage schemas."""
    schema_version: SchemaVersion = Field(default=SchemaVersion(major=1, minor=0, patch=0))
    stage: PipelineStage = Field(...)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# Stage 1: Ingestion & Preparation Schemas
class IngestionInputSchema(BaseStageSchema):
    """Input schema for ingestion stage."""
    stage: PipelineStage = Field(default=PipelineStage.INGESTION)
    file_path: str = Field(..., description="Path to the document to be processed")
    file_type: str = Field(..., description="Type of file (pdf, txt, etc.)")
    processing_options: Dict[str, Any] = Field(default_factory=dict)


class IngestionOutputSchema(BaseStageSchema):
    """Output schema for ingestion stage."""
    stage: PipelineStage = Field(default=PipelineStage.INGESTION)
    extracted_text: str = Field(..., description="Extracted text content")
    document_metadata: Dict[str, Any] = Field(..., description="Document metadata")
    extraction_metrics: Dict[str, float] = Field(..., description="Extraction quality metrics")
    loaded_docs: List[Dict[str, Any]] = Field(default_factory=list)
    features: Dict[str, Any] = Field(default_factory=dict)
    vectors: List[List[float]] = Field(default_factory=list)
    validation_report: Dict[str, Any] = Field(default_factory=dict)
    compliance: bool = Field(default=False)


# Stage 2: Context Construction Schemas  
class ContextInputSchema(BaseStageSchema):
    """Input schema for context construction stage."""
    stage: PipelineStage = Field(default=PipelineStage.CONTEXT_BUILD)
    validated_data: Dict[str, Any] = Field(..., description="Validated data from ingestion")
    context_requirements: Dict[str, Any] = Field(default_factory=dict)


class ContextOutputSchema(BaseStageSchema):
    """Output schema for context construction stage."""
    stage: PipelineStage = Field(default=PipelineStage.CONTEXT_BUILD)
    immutable_context: Dict[str, Any] = Field(..., description="Immutable context structure")
    adapted_context: Dict[str, Any] = Field(..., description="Adapted context for processing")
    lineage_graph: Dict[str, Any] = Field(..., description="Lineage tracking information")
    trace: List[Dict[str, Any]] = Field(default_factory=list)


# Stage 3: Knowledge Extraction Schemas
class KnowledgeInputSchema(BaseStageSchema):
    """Input schema for knowledge extraction stage."""
    stage: PipelineStage = Field(default=PipelineStage.KNOWLEDGE)
    context_data: Dict[str, Any] = Field(..., description="Context data for knowledge extraction")
    extraction_parameters: Dict[str, Any] = Field(default_factory=dict)


class KnowledgeOutputSchema(BaseStageSchema):
    """Output schema for knowledge extraction stage."""
    stage: PipelineStage = Field(default=PipelineStage.KNOWLEDGE)
    knowledge_graph: Dict[str, Any] = Field(..., description="Constructed knowledge graph")
    causal_relations: Dict[str, Any] = Field(..., description="Identified causal relationships")
    embeddings: List[List[float]] = Field(..., description="Generated embeddings")
    topological_features: Dict[str, Any] = Field(default_factory=dict)
    dnp_model: Optional[Dict[str, Any]] = Field(default=None)


# Stage 4: Analysis & NLP Schemas
class AnalysisInputSchema(BaseStageSchema):
    """Input schema for analysis stage."""
    stage: PipelineStage = Field(default=PipelineStage.ANALYSIS)
    knowledge_data: Dict[str, Any] = Field(..., description="Knowledge data for analysis")
    analysis_config: Dict[str, Any] = Field(default_factory=dict)


class AnalysisOutputSchema(BaseStageSchema):
    """Output schema for analysis stage."""
    stage: PipelineStage = Field(default=PipelineStage.ANALYSIS)
    analysis_results: Dict[str, Any] = Field(..., description="Analysis results")
    questions: List[Dict[str, Any]] = Field(..., description="Analyzed questions")
    intents: Dict[str, Any] = Field(..., description="Identified intents")
    processed_evidence: Dict[str, Any] = Field(..., description="Processed evidence")
    contextual_evidence: Dict[str, Any] = Field(..., description="Contextual evidence")
    validated_evidence: Dict[str, Any] = Field(..., description="Validated evidence")
    evaluation_results: Dict[str, Any] = Field(..., description="Evaluation results")
    dnp_compliance: Dict[str, Any] = Field(default_factory=dict)


# Stage 5: Classification & Scoring Schemas
class ClassificationInputSchema(BaseStageSchema):
    """Input schema for classification stage."""
    stage: PipelineStage = Field(default=PipelineStage.CLASSIFICATION)
    analysis_data: Dict[str, Any] = Field(..., description="Analysis data for classification")
    scoring_parameters: Dict[str, Any] = Field(default_factory=dict)


class ClassificationOutputSchema(BaseStageSchema):
    """Output schema for classification stage.""" 
    stage: PipelineStage = Field(default=PipelineStage.CLASSIFICATION)
    adaptive_scores: Dict[str, float] = Field(..., description="Adaptive scoring results")
    final_scores: Dict[str, float] = Field(..., description="Final calculated scores")
    risk_bounds: Dict[str, float] = Field(..., description="Risk control bounds")
    certificates: List[Dict[str, Any]] = Field(default_factory=list)


# Stage 6: Routing & Decision Schemas
class RoutingInputSchema(BaseStageSchema):
    """Input schema for routing stage."""
    stage: PipelineStage = Field(default=PipelineStage.ROUTING)
    classification_data: Dict[str, Any] = Field(..., description="Classification data for routing")
    routing_config: Dict[str, Any] = Field(default_factory=dict)


class RoutingOutputSchema(BaseStageSchema):
    """Output schema for routing stage."""
    stage: PipelineStage = Field(default=PipelineStage.ROUTING)
    routing_decisions: Dict[str, Any] = Field(..., description="Routing decisions")
    evidence_routes: Dict[str, Any] = Field(..., description="Evidence routing information")
    control_signals: Dict[str, Any] = Field(default_factory=dict)


# Stage 7: Search & Retrieval Schemas  
class SearchInputSchema(BaseStageSchema):
    """Input schema for search stage."""
    stage: PipelineStage = Field(default=PipelineStage.SEARCH)
    query: str = Field(..., description="Search query")
    routing_data: Dict[str, Any] = Field(..., description="Routing data for search")
    search_parameters: Dict[str, Any] = Field(default_factory=dict)


class SearchOutputSchema(BaseStageSchema):
    """Output schema for search stage."""
    stage: PipelineStage = Field(default=PipelineStage.SEARCH)
    search_results: List[Dict[str, Any]] = Field(..., description="Search results")
    retrieval_metrics: Dict[str, float] = Field(..., description="Retrieval performance metrics")
    reranked_candidates: List[Dict[str, Any]] = Field(default_factory=list)
    recommendations: List[Dict[str, Any]] = Field(default_factory=list)


# Stage 8: Orchestration & Parallel Processing Schemas
class OrchestrationInputSchema(BaseStageSchema):
    """Input schema for orchestration stage."""
    stage: PipelineStage = Field(default=PipelineStage.ORCHESTRATION)
    search_data: Dict[str, Any] = Field(..., description="Search data for orchestration")
    orchestration_config: Dict[str, Any] = Field(default_factory=dict)


class OrchestrationOutputSchema(BaseStageSchema):
    """Output schema for orchestration stage."""
    stage: PipelineStage = Field(default=PipelineStage.ORCHESTRATION)
    orchestration_state: Dict[str, Any] = Field(..., description="Orchestration state")
    distributed_results: Dict[str, Any] = Field(..., description="Distributed processing results")
    workflow_status: Dict[str, Any] = Field(default_factory=dict)


# Stage 9: Monitoring & Validation Schemas
class MonitoringInputSchema(BaseStageSchema):
    """Input schema for monitoring stage."""
    stage: PipelineStage = Field(default=PipelineStage.MONITORING)
    orchestration_data: Dict[str, Any] = Field(..., description="Orchestration data for monitoring")
    monitoring_config: Dict[str, Any] = Field(default_factory=dict)


class MonitoringOutputSchema(BaseStageSchema):
    """Output schema for monitoring stage."""
    stage: PipelineStage = Field(default=PipelineStage.MONITORING)
    circuit_state: str = Field(..., description="Circuit breaker state")
    pressure_state: Dict[str, Any] = Field(..., description="Backpressure management state")
    alerts: List[Dict[str, Any]] = Field(default_factory=list)
    exceptions: List[Dict[str, Any]] = Field(default_factory=list)
    telemetry: Dict[str, Any] = Field(default_factory=dict)


# Stage 10: Validation & Contracts Schemas
class ValidationInputSchema(BaseStageSchema):
    """Input schema for validation stage."""
    stage: PipelineStage = Field(default=PipelineStage.VALIDATION)
    monitoring_data: Dict[str, Any] = Field(..., description="Monitoring data for validation")
    validation_rules: Dict[str, Any] = Field(default_factory=dict)


class ValidationOutputSchema(BaseStageSchema):
    """Output schema for validation stage."""
    stage: PipelineStage = Field(default=PipelineStage.VALIDATION)
    contract_validation: bool = Field(..., description="Contract validation result")
    constraint_validation: bool = Field(..., description="Constraint validation result") 
    rubric_scores: Dict[str, float] = Field(..., description="Rubric scoring results")


# Stage 11: Aggregation & Synthesis Schemas
class AggregationInputSchema(BaseStageSchema):
    """Input schema for aggregation stage."""
    stage: PipelineStage = Field(default=PipelineStage.AGGREGATION)
    validation_data: Dict[str, Any] = Field(..., description="Validation data for aggregation")
    aggregation_config: Dict[str, Any] = Field(default_factory=dict)


class AggregationOutputSchema(BaseStageSchema):
    """Output schema for aggregation stage."""
    stage: PipelineStage = Field(default=PipelineStage.AGGREGATION)
    synthesized_answer: str = Field(..., description="Synthesized answer")
    formatted_answer: str = Field(..., description="Formatted answer")
    report: Dict[str, Any] = Field(..., description="Generated report")
    report_pdf: Optional[bytes] = Field(default=None, description="PDF report bytes")


# Stage 12: Integration & Metrics Schemas
class IntegrationInputSchema(BaseStageSchema):
    """Input schema for integration stage."""
    stage: PipelineStage = Field(default=PipelineStage.INTEGRATION)
    aggregation_data: Dict[str, Any] = Field(..., description="Aggregation data for integration")
    integration_config: Dict[str, Any] = Field(default_factory=dict)


class IntegrationOutputSchema(BaseStageSchema):
    """Output schema for integration stage."""
    stage: PipelineStage = Field(default=PipelineStage.INTEGRATION)
    collected_metrics: Dict[str, float] = Field(..., description="Collected metrics")
    enhanced_analytics: Dict[str, Any] = Field(..., description="Enhanced analytics")
    feedback: Dict[str, Any] = Field(..., description="Feedback information")
    compensations: Dict[str, Any] = Field(default_factory=dict)
    optimizations: Dict[str, Any] = Field(default_factory=dict)


@dataclass
class SchemaEvolutionRule:
    """Rules for schema evolution and migration."""
    from_version: SchemaVersion
    to_version: SchemaVersion
    migration_function: str  # Name of migration function
    field_mappings: Dict[str, str] = field(default_factory=dict)  # old_field -> new_field
    default_values: Dict[str, Any] = field(default_factory=dict)  # new_field -> default_value
    removed_fields: List[str] = field(default_factory=list)


class SchemaRegistry:
    """
    Centralized registry for managing pipeline stage schemas with version compatibility
    and automatic evolution support.
    """
    
    def __init__(self, schemas_config_path: Optional[str] = None):
        self.schemas_config_path = schemas_config_path or "schemas.json"
        self.input_schemas: Dict[PipelineStage, Type[BaseStageSchema]] = {}
        self.output_schemas: Dict[PipelineStage, Type[BaseStageSchema]] = {}
        self.schema_versions: Dict[str, SchemaVersion] = {}
        self.evolution_rules: Dict[Tuple[str, str], SchemaEvolutionRule] = {}
        
        self._initialize_default_schemas()
        self._load_schemas_config()
    
    def _initialize_default_schemas(self):
        """Initialize default schema mappings."""
        schema_mappings = {
            PipelineStage.INGESTION: (IngestionInputSchema, IngestionOutputSchema),
            PipelineStage.CONTEXT_BUILD: (ContextInputSchema, ContextOutputSchema),
            PipelineStage.KNOWLEDGE: (KnowledgeInputSchema, KnowledgeOutputSchema),
            PipelineStage.ANALYSIS: (AnalysisInputSchema, AnalysisOutputSchema),
            PipelineStage.CLASSIFICATION: (ClassificationInputSchema, ClassificationOutputSchema),
            PipelineStage.ROUTING: (RoutingInputSchema, RoutingOutputSchema),
            PipelineStage.SEARCH: (SearchInputSchema, SearchOutputSchema),
            PipelineStage.ORCHESTRATION: (OrchestrationInputSchema, OrchestrationOutputSchema),
            PipelineStage.MONITORING: (MonitoringInputSchema, MonitoringOutputSchema),
            PipelineStage.VALIDATION: (ValidationInputSchema, ValidationOutputSchema),
            PipelineStage.AGGREGATION: (AggregationInputSchema, AggregationOutputSchema),
            PipelineStage.INTEGRATION: (IntegrationInputSchema, IntegrationOutputSchema),
        }
        
        for stage, (input_schema, output_schema) in schema_mappings.items():
            self.input_schemas[stage] = input_schema
            self.output_schemas[stage] = output_schema
    
    def _load_schemas_config(self):
        """Load schema configuration from JSON file."""
        config_path = Path(self.schemas_config_path)
        
        if not config_path.exists():
            logger.info(f"Schema config file {self.schemas_config_path} not found, creating default")
            self._create_default_config()
            return
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Load schema versions
            if 'versions' in config:
                for stage_name, version_data in config['versions'].items():
                    self.schema_versions[stage_name] = SchemaVersion(**version_data)
            
            # Load evolution rules
            if 'evolution_rules' in config:
                for rule_data in config['evolution_rules']:
                    from_version = SchemaVersion(**rule_data['from_version'])
                    to_version = SchemaVersion(**rule_data['to_version'])
                    rule = SchemaEvolutionRule(
                        from_version=from_version,
                        to_version=to_version,
                        migration_function=rule_data['migration_function'],
                        field_mappings=rule_data.get('field_mappings', {}),
                        default_values=rule_data.get('default_values', {}),
                        removed_fields=rule_data.get('removed_fields', [])
                    )
                    stage_name = rule_data['stage']
                    key = (stage_name, f"{from_version}->{to_version}")
                    self.evolution_rules[key] = rule
                    
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Error loading schema config: {e}")
            self._create_default_config()
    
    def _create_default_config(self):
        """Create default schema configuration file."""
        default_config = {
            "versions": {
                stage.value: {
                    "major": 1,
                    "minor": 0, 
                    "patch": 0
                }
                for stage in PipelineStage
            },
            "evolution_rules": [
                {
                    "stage": "ingestion_preparation",
                    "from_version": {"major": 1, "minor": 0, "patch": 0},
                    "to_version": {"major": 1, "minor": 1, "patch": 0},
                    "migration_function": "migrate_ingestion_v1_0_to_v1_1",
                    "field_mappings": {},
                    "default_values": {},
                    "removed_fields": []
                }
            ]
        }
        
        try:
            with open(self.schemas_config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
            logger.info(f"Created default schema config at {self.schemas_config_path}")
        except Exception as e:
            logger.error(f"Error creating default config: {e}")
    
    def get_input_schema(self, stage: PipelineStage) -> Type[BaseStageSchema]:
        """Get the input schema for a specific pipeline stage."""
        if stage not in self.input_schemas:
            raise ValueError(f"No input schema found for stage: {stage}")
        return self.input_schemas[stage]
    
    def get_output_schema(self, stage: PipelineStage) -> Type[BaseStageSchema]:
        """Get the output schema for a specific pipeline stage."""
        if stage not in self.output_schemas:
            raise ValueError(f"No output schema found for stage: {stage}")
        return self.output_schemas[stage]
    
    def validate_input(self, stage: PipelineStage, data: Dict[str, Any]) -> BaseStageSchema:
        """Validate input data against the stage's input schema."""
        schema_class = self.get_input_schema(stage)
        try:
            return schema_class(**data)
        except ValidationError as e:
            logger.error(f"Input validation failed for stage {stage}: {e}")
            raise
    
    def validate_output(self, stage: PipelineStage, data: Dict[str, Any]) -> BaseStageSchema:
        """Validate output data against the stage's output schema."""
        schema_class = self.get_output_schema(stage)
        try:
            return schema_class(**data)
        except ValidationError as e:
            logger.error(f"Output validation failed for stage {stage}: {e}")
            raise
    
    def check_compatibility(self, 
                          from_stage: PipelineStage, 
                          to_stage: PipelineStage) -> CompatibilityResult:
        """Check if the output of one stage is compatible with the input of another."""
        output_schema = self.get_output_schema(from_stage)
        input_schema = self.get_input_schema(to_stage)
        
        issues = []
        
        # Check if required fields in input schema can be satisfied by output schema
        output_fields = set(output_schema.__fields__.keys())
        input_required_fields = {
            name for name, field in input_schema.__fields__.items() 
            if field.required and name not in ['stage', 'timestamp', 'schema_version']
        }
        
        missing_fields = input_required_fields - output_fields
        if missing_fields:
            issues.append(f"Missing required fields: {missing_fields}")
        
        # Check type compatibility for overlapping fields
        for field_name in output_fields & set(input_schema.__fields__.keys()):
            output_field = output_schema.__fields__[field_name]
            input_field = input_schema.__fields__[field_name]
            
            if output_field.type_ != input_field.type_:
                issues.append(
                    f"Type mismatch for field '{field_name}': "
                    f"output={output_field.type_}, input={input_field.type_}"
                )
        
        return CompatibilityResult(
            compatible=len(issues) == 0,
            issues=issues
        )
    
    def validate_pipeline_compatibility(self, stages: List[PipelineStage]) -> Dict[str, CompatibilityResult]:
        """Validate compatibility across a sequence of pipeline stages."""
        results = {}
        
        for i in range(len(stages) - 1):
            from_stage = stages[i]
            to_stage = stages[i + 1]
            key = f"{from_stage.value} -> {to_stage.value}"
            results[key] = self.check_compatibility(from_stage, to_stage)
        
        return results
    
    def migrate_data(self, 
                    stage: PipelineStage,
                    data: Dict[str, Any],
                    from_version: SchemaVersion,
                    to_version: SchemaVersion,
                    schema_type: str = "input") -> Dict[str, Any]:
        """Migrate data from one schema version to another."""
        rule_key = (stage.value, f"{from_version}->{to_version}")
        
        if rule_key not in self.evolution_rules:
            raise ValueError(f"No migration rule found for {stage.value} {from_version} -> {to_version}")
        
        rule = self.evolution_rules[rule_key]
        migrated_data = data.copy()
        
        # Apply field mappings
        for old_field, new_field in rule.field_mappings.items():
            if old_field in migrated_data:
                migrated_data[new_field] = migrated_data.pop(old_field)
        
        # Add default values for new fields
        for field_name, default_value in rule.default_values.items():
            if field_name not in migrated_data:
                migrated_data[field_name] = default_value
        
        # Remove deprecated fields
        for field_name in rule.removed_fields:
            migrated_data.pop(field_name, None)
        
        # Update schema version
        migrated_data['schema_version'] = to_version.dict()
        
        return migrated_data
    
    def get_schema_hash(self, schema_class: Type[BaseStageSchema]) -> str:
        """Generate a hash for schema structure for integrity checking."""
        schema_structure = {
            field_name: {
                'type': str(field.type_),
                'required': field.required,
                'default': str(field.default) if field.default is not None else None
            }
            for field_name, field in schema_class.__fields__.items()
        }
        
        schema_json = json.dumps(schema_structure, sort_keys=True)
        return hashlib.sha256(schema_json.encode()).hexdigest()[:16]
    
    def validate_data_integrity(self, 
                              stage: PipelineStage, 
                              data: Dict[str, Any],
                              schema_type: str = "input") -> bool:
        """Validate data integrity and completeness."""
        try:
            if schema_type == "input":
                validated = self.validate_input(stage, data)
            else:
                validated = self.validate_output(stage, data)
            
            # Additional integrity checks
            if validated.timestamp > datetime.utcnow():
                logger.warning("Future timestamp detected in data")
                return False
            
            # Check for required metadata
            if not validated.metadata:
                logger.warning("No metadata found in schema")
            
            return True
            
        except ValidationError:
            return False
    
    def get_all_stages(self) -> List[PipelineStage]:
        """Get all available pipeline stages."""
        return list(PipelineStage)
    
    def get_schema_documentation(self, stage: PipelineStage) -> Dict[str, Any]:
        """Generate documentation for a stage's schemas."""
        input_schema = self.get_input_schema(stage)
        output_schema = self.get_output_schema(stage)
        
        return {
            "stage": stage.value,
            "input_schema": {
                "name": input_schema.__name__,
                "fields": {
                    name: {
                        "type": str(field.type_),
                        "required": field.required,
                        "description": field.field_info.description or "",
                        "default": str(field.default) if field.default is not None else None
                    }
                    for name, field in input_schema.__fields__.items()
                }
            },
            "output_schema": {
                "name": output_schema.__name__,
                "fields": {
                    name: {
                        "type": str(field.type_),
                        "required": field.required,
                        "description": field.field_info.description or "",
                        "default": str(field.default) if field.default is not None else None
                    }
                    for name, field in output_schema.__fields__.items()
                }
            }
        }
    
    def export_schemas_config(self, output_path: Optional[str] = None) -> None:
        """Export current schema configuration to file."""
        output_path = output_path or "schema_registry_export.json"
        
        config = {
            "generated_at": datetime.utcnow().isoformat(),
            "pydantic_version": PYDANTIC_VERSION,
            "stages": [stage.value for stage in PipelineStage],
            "schemas": {
                stage.value: self.get_schema_documentation(stage)
                for stage in PipelineStage
            },
            "versions": {
                stage: version.dict() 
                for stage, version in self.schema_versions.items()
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Schema configuration exported to {output_path}")


# Global registry instance
_registry_instance = None


def get_schema_registry() -> SchemaRegistry:
    """Get the global schema registry instance."""
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = SchemaRegistry()
    return _registry_instance


def validate_stage_input(stage: PipelineStage, data: Dict[str, Any]) -> BaseStageSchema:
    """Convenience function to validate stage input."""
    return get_schema_registry().validate_input(stage, data)


def validate_stage_output(stage: PipelineStage, data: Dict[str, Any]) -> BaseStageSchema:
    """Convenience function to validate stage output."""
    return get_schema_registry().validate_output(stage, data)


def check_stage_compatibility(from_stage: PipelineStage, to_stage: PipelineStage) -> CompatibilityResult:
    """Convenience function to check stage compatibility."""
    return get_schema_registry().check_compatibility(from_stage, to_stage)
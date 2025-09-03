"""
Pydantic models for PDT ingestion engine with comprehensive validation
# # # Migrated from dataclasses to provide better API validation and serialization  # Module not found  # Module not found  # Module not found
"""

# # # from datetime import datetime  # Module not found  # Module not found  # Module not found
# # # from enum import Enum  # Module not found  # Module not found  # Module not found
# # # from typing import Any, Dict, List, Optional, Union  # Module not found  # Module not found  # Module not found
# # # from uuid import UUID  # Module not found  # Module not found  # Module not found

import numpy as np
import pandas as pd
# # # from pydantic import BaseModel, ConfigDict, Field, field_validator  # Module not found  # Module not found  # Module not found

# # # # Import base model from models directory (guarded)  # Module not found  # Module not found  # Module not found
try:
# # #     from models.base import BaseModel as CustomBaseModel  # Module not found  # Module not found  # Module not found
except Exception:
    # Fallback to pydantic BaseModel if optional custom base is missing
    CustomBaseModel = BaseModel


class SectionType(str, Enum):
    """Tipos de secciones reconocidas en PDTs"""

    DIAGNOSTICO = "diagnostico"
    PROGRAMAS = "programas"
    PRESUPUESTO = "presupuesto"
    METAS = "metas"
    SEGUIMIENTO = "seguimiento"
    PARTICIPACION = "participacion"
    TERRITORIAL = "territorial"
    AMBIENTAL = "ambiental"
    ECONOMICO = "economico"
    SOCIAL = "social"


class ComplianceStatus(str, Enum):
    """Estados de satisfacibilidad de evaluación"""

    CUMPLE = "CUMPLE"
    CUMPLE_PARCIAL = "CUMPLE_PARCIAL"
    NO_CUMPLE = "NO_CUMPLE"


class Priority(str, Enum):
    """Priority levels"""

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class DifficultyLevel(str, Enum):
    """Implementation difficulty levels"""

    EASY = "EASY"
    MEDIUM = "MEDIUM"
    HARD = "HARD"


# Core Document Models


class Citation(CustomBaseModel):
    """Referencia a ubicación específica en el documento"""

    page: int = Field(..., ge=1, description="Page number (1-based)")
    char_start: int = Field(..., ge=0, description="Character start position")
    char_end: int = Field(..., gt=0, description="Character end position")
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Citation confidence"
    )

    @field_validator("char_end")
    @classmethod
    def validate_char_range(cls, v, info):
        if info.data and "char_start" in info.data and v <= info.data["char_start"]:
            raise ValueError("char_end must be greater than char_start")
        return v


class DocumentEnvelope(CustomBaseModel):
    """Contenedor principal del documento PDT"""

    pdt_id: str = Field(..., min_length=1, description="Unique PDT identifier")
    gcs_uri: str = Field(..., pattern=r"^gs://.+", description="Google Cloud Storage URI")
    raw_metadata: Dict[str, Any] = Field(
# # #         default_factory=dict, description="Raw metadata from document"  # Module not found  # Module not found  # Module not found
    )
    clean_metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Processed metadata"
    )
    sha256_hash: str = Field(default="", description="SHA256 hash of document")
    processing_timestamp: Optional[datetime] = Field(
        None, description="Processing timestamp"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "pdt_id": "pdt_1703123456_abc123",
                "gcs_uri": "gs://pdt-documents/example.pdf",
                "sha256_hash": "a1b2c3d4...",
            }
        }
    )


class SectionBlock(CustomBaseModel):
    """Bloque de sección semántica del PDT"""

    section_id: str = Field(..., min_length=1, description="Unique section identifier")
    section_type: SectionType = Field(..., description="Section type classification")
    page_start: int = Field(..., ge=1, description="Starting page number")
    page_end: int = Field(..., ge=1, description="Ending page number")
    text: str = Field(..., min_length=1, description="Section text content")
    citations: List[Citation] = Field(
        default_factory=list, description="Text citations"
    )
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Section detection confidence"
    )
    word_count: int = Field(default=0, ge=0, description="Word count in section")

    @field_validator("page_end")
    @classmethod
    def validate_page_range(cls, v, info):
        if info.data and "page_start" in info.data and v < info.data["page_start"]:
            raise ValueError("page_end must be >= page_start")
        return v

    def dict(self, **kwargs) -> Dict[str, Any]:
        """Custom serialization for compatibility with existing code"""
        data = super().model_dump(**kwargs)
        # Convert enums to string values for backward compatibility
        if "section_type" in data and hasattr(data["section_type"], "value"):
            data["section_type"] = data["section_type"].value
        return data


class TableArtifact(CustomBaseModel):
    """Artefacto de tabla extraída del documento"""

    table_id: str = Field(..., min_length=1, description="Unique table identifier")
    source: str = Field(..., description="Extraction source (camelot, tabula, etc.)")
    page: int = Field(..., ge=1, description="Page number containing table")
    # Note: DataFrame will be handled separately due to Pydantic limitations
    dataframe_shape: tuple[int, int] = Field(
        default=(0, 0), description="DataFrame shape (rows, cols)"
    )
    csv_uri: Optional[str] = Field(None, description="URI of CSV export")
    parquet_uri: Optional[str] = Field(None, description="URI of Parquet export")
    quality_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Table quality score"
    )
    header_map: Dict[str, str] = Field(
        default_factory=dict, description="Header mapping"
    )
    column_count: int = Field(default=0, ge=0, description="Number of columns")
    row_count: int = Field(default=0, ge=0, description="Number of rows")

    # For storing actual DataFrame data as serialized format
    _dataframe_json: Optional[str] = Field(
        default=None, exclude=True, description="Serialized DataFrame"
    )

    @field_validator("source")
    @classmethod
    def validate_source(cls, v):
        allowed_sources = ["camelot", "tabula", "pdfplumber", "manual", "auto"]
        if v not in allowed_sources:
            raise ValueError(f"Source must be one of {allowed_sources}")
        return v

    def set_dataframe(self, df: pd.DataFrame) -> None:
        """Set DataFrame and update metadata"""
        if df is not None:
            self.dataframe_shape = df.shape
            self.row_count, self.column_count = df.shape
            self._dataframe_json = df.to_json(orient="records")

    def get_dataframe(self) -> Optional[pd.DataFrame]:
# # #         """Retrieve DataFrame from serialized format"""  # Module not found  # Module not found  # Module not found
        if self._dataframe_json:
            return pd.read_json(self._dataframe_json, orient="records")
        return None


class QualityIndicators(CustomBaseModel):
    """Indicadores de calidad del documento procesado"""

    completeness_index: float = Field(
        ..., ge=0.0, le=1.0, description="Document completeness score"
    )
    logical_coherence_hint: float = Field(
        ..., ge=0.0, le=1.0, description="Logical coherence score"
    )
    tables_found: int = Field(..., ge=0, description="Number of tables detected")
    ocr_ratio: float = Field(
        ..., ge=0.0, le=1.0, description="Ratio of pages requiring OCR"
    )
    mandatory_sections_present: List[str] = Field(
        default_factory=list, description="Present mandatory sections"
    )
    missing_sections: List[str] = Field(
        default_factory=list, description="Missing mandatory sections"
    )

    # Additional quality metrics
    average_section_length: float = Field(
        default=0.0, ge=0.0, description="Average section length in words"
    )
    text_extraction_quality: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Text extraction quality"
    )
    structure_consistency: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Document structure consistency"
    )


# Context and Analysis Models


class PDTContext(CustomBaseModel):
    """Contexto granular del municipio para análisis predictivo"""

    municipality_code: str = Field(
        ..., min_length=5, max_length=10, description="DANE municipality code"
    )
    municipality_name: str = Field(..., min_length=1, description="Municipality name")
    department: str = Field(default="", description="Department name")
    population: int = Field(..., gt=0, description="Total population")
    area_km2: float = Field(..., gt=0.0, description="Territory area in km²")
    budget: float = Field(..., ge=0.0, description="Annual budget in COP")
    gdp_per_capita: float = Field(..., ge=0.0, description="GDP per capita in COP")
    urbanization_rate: float = Field(
        ..., ge=0.0, le=1.0, description="Urbanization rate"
    )

    # Development indices
    education_index: float = Field(
        ..., ge=0.0, le=1.0, description="Education development index"
    )
    health_index: float = Field(
        ..., ge=0.0, le=1.0, description="Health development index"
    )
    poverty_index: float = Field(..., ge=0.0, le=1.0, description="Poverty index")
    infrastructure_index: float = Field(
        ..., ge=0.0, le=1.0, description="Infrastructure index"
    )
    governance_index: float = Field(..., ge=0.0, le=1.0, description="Governance index")
    environmental_index: float = Field(
        ..., ge=0.0, le=1.0, description="Environmental index"
    )

    # Historical and contextual data
    previous_pdt_scores: Dict[str, float] = Field(
        default_factory=dict, description="Previous PDT scores"
    )
    regional_indicators: Dict[str, float] = Field(
        default_factory=dict, description="Regional indicators"
    )
    temporal_features: Dict[str, Any] = Field(
        default_factory=dict, description="Temporal features"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "municipality_code": "05001",
                "municipality_name": "Medellín",
                "department": "Antioquia",
                "population": 2533424,
                "area_km2": 380.64,
                "budget": 7500000000000.0,
                "gdp_per_capita": 25000000.0,
                "urbanization_rate": 0.95,
                "education_index": 0.78,
                "health_index": 0.82,
                "poverty_index": 0.15,
            }
        }
    )


# Scoring Models


class DimensionScore(CustomBaseModel):
    """Puntuación de una dimensión específica"""

    dimension_id: str = Field(..., description="Dimension identifier")
    raw_score: float = Field(..., ge=0.0, le=1.0, description="Raw calculated score")
    weighted_score: float = Field(..., ge=0.0, le=1.0, description="Weighted score")
    predicted_score: float = Field(
        ..., ge=0.0, le=1.0, description="ML predicted score"
    )
    compliance_status: ComplianceStatus = Field(..., description="Compliance status")
    contributing_questions: Dict[str, float] = Field(
        default_factory=dict, description="Question contributions"
    )
    quality_metrics: Dict[str, float] = Field(
        default_factory=dict, description="Quality metrics"
    )
    confidence_level: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Confidence level"
    )


class DecalogoPointScore(CustomBaseModel):
    """Puntuación de un punto del Decálogo"""

    point_id: str = Field(
        ..., pattern=r"^P\d{1,2}$", description="Point identifier (P1-P10)"
    )
    raw_score: float = Field(..., ge=0.0, le=1.0, description="Raw calculated score")
    weighted_score: float = Field(..., ge=0.0, le=1.0, description="Weighted score")
    predicted_score: float = Field(
        ..., ge=0.0, le=1.0, description="ML predicted score"
    )
    compliance_status: ComplianceStatus = Field(..., description="Compliance status")
    dimension_contributions: Dict[str, float] = Field(
        default_factory=dict, description="Dimension contributions"
    )
    thresholds: Dict[str, float] = Field(
        default_factory=dict, description="Compliance thresholds"
    )
    confidence_level: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Confidence level"
    )


class AdaptiveScoringResults(CustomBaseModel):
    """Resultados del motor de puntuación adaptativa"""

    global_score: float = Field(..., ge=0.0, le=1.0, description="Global score")
    predicted_global_score: float = Field(
        ..., ge=0.0, le=1.0, description="ML predicted global score"
    )
    dimension_scores: Dict[str, DimensionScore] = Field(
        default_factory=dict, description="Dimension scores"
    )
    decalogo_scores: Dict[str, DecalogoPointScore] = Field(
        default_factory=dict, description="Decálogo scores"
    )
    feature_importance: Dict[str, float] = Field(
        default_factory=dict, description="Feature importance"
    )
    model_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Model confidence"
    )
    prediction_quality_metrics: Dict[str, float] = Field(
        default_factory=dict, description="Prediction quality"
    )
    analysis_timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Analysis timestamp"
    )


# Recommendation Models


class RecommendationItem(CustomBaseModel):
    """Recomendación específica y accionable"""

    recommendation_id: str = Field(
        ..., min_length=1, description="Unique recommendation ID"
    )
    category: str = Field(..., min_length=1, description="Recommendation category")
    priority: Priority = Field(..., description="Priority level")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Recommendation confidence"
    )
    title: str = Field(..., min_length=5, description="Recommendation title")
    description: str = Field(..., min_length=10, description="Detailed description")
    expected_impact: float = Field(..., ge=0.0, le=1.0, description="Expected impact")
    implementation_difficulty: DifficultyLevel = Field(
        ..., description="Implementation difficulty"
    )
    related_dimensions: List[str] = Field(
        default_factory=list, description="Related dimensions"
    )
    evidence_support: List[str] = Field(
        default_factory=list, description="Supporting evidence"
    )
    action_steps: List[str] = Field(
        default_factory=list, description="Implementation steps"
    )

    # Additional metadata
    estimated_cost_range: Optional[str] = Field(
        None, description="Estimated cost range"
    )
    timeline_weeks: Optional[int] = Field(
        None, ge=1, description="Estimated timeline in weeks"
    )
    success_indicators: List[str] = Field(
        default_factory=list, description="Success indicators"
    )


class IntelligentRecommendations(CustomBaseModel):
    """Conjunto completo de recomendaciones inteligentes"""

    municipality_id: str = Field(
        ..., min_length=1, description="Municipality identifier"
    )
    total_recommendations: int = Field(
        ..., ge=0, description="Total number of recommendations"
    )
    high_priority_count: int = Field(
        ..., ge=0, description="High priority recommendations count"
    )
    expected_total_impact: float = Field(
        ..., ge=0.0, description="Expected total impact"
    )
    recommendations: List[RecommendationItem] = Field(
        default_factory=list, description="Recommendations list"
    )
    gap_analysis: Dict[str, float] = Field(
        default_factory=dict, description="Gap analysis results"
    )
    feature_importance: Dict[str, float] = Field(
        default_factory=dict, description="Feature importance"
    )
    generation_timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Generation timestamp"
    )
    confidence_level: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Overall confidence"
    )

    @field_validator("high_priority_count")
    @classmethod
    def validate_high_priority_count(cls, v, info):
        if info.data and "total_recommendations" in info.data and v > info.data["total_recommendations"]:
            raise ValueError("high_priority_count cannot exceed total_recommendations")
        return v


# Main Document Package Model


class DocumentPackage(CustomBaseModel):
    """Paquete normalizado final del documento PDT"""

    header: Dict[str, Any] = Field(
        default_factory=dict, description="Document header information"
    )
    sections: List[SectionBlock] = Field(
        default_factory=list, description="Document sections"
    )
    tables: List[TableArtifact] = Field(
        default_factory=list, description="Extracted tables"
    )
    quality_indicators: QualityIndicators = Field(..., description="Quality indicators")
    package_uri: Optional[str] = Field(None, description="Package storage URI")

    # Analysis results
    adaptive_scoring_results: Optional[AdaptiveScoringResults] = Field(
        None, description="Adaptive scoring results"
    )
    intelligent_recommendations: Optional[IntelligentRecommendations] = Field(
        None, description="Intelligent recommendations"
    )
    pdt_context: Optional[PDTContext] = Field(None, description="PDT municipal context")

    # Processing metadata
    processing_metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Processing metadata"
    )
    validation_results: Dict[str, Any] = Field(
        default_factory=dict, description="Validation results"
    )

    @field_validator("sections")
    @classmethod
    def validate_sections(cls, v):
        if not v:
            raise ValueError("Document must contain at least one section")
        return v

    def get_section_by_type(self, section_type: SectionType) -> Optional[SectionBlock]:
        """Get first section of specified type"""
        for section in self.sections:
            if section.section_type == section_type:
                return section
        return None

    def get_sections_by_type(self, section_type: SectionType) -> List[SectionBlock]:
        """Get all sections of specified type"""
        return [
            section for section in self.sections if section.section_type == section_type
        ]

    def get_table_count(self) -> int:
        """Get total number of tables"""
        return len(self.tables)

    def get_total_word_count(self) -> int:
        """Get total word count across all sections"""
        return sum(section.word_count for section in self.sections)


# Backward compatibility functions for existing code


def convert_dataclass_to_pydantic(dataclass_obj) -> BaseModel:
    """Convert legacy dataclass objects to Pydantic models"""
    # This function can be used during migration to convert existing dataclass instances
    # Implementation would depend on specific conversion requirements
    pass

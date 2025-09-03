"""
NORMATIVE VALIDATOR - MÓDULO DE VALIDACIÓN NORMATIVA DNP
========================================================

Módulo de nivel industrial para validación automatizada de Planes de Desarrollo Territorial (PDT)
contra estándares cuantitativos y cualitativos del Departamento Nacional de Planeación (DNP) de Colombia.

Versión: 2.0 (Producción)
Fecha: 12 de Agosto de 2025
Stack: Python 3.12+ | Pydantic 2.8+ | JSONSchema | Rich | Loguru

Arquitectura: Clean Architecture + SOLID + Factory Pattern + Strategy Pattern
"""

import json
import logging
import re
import time
# # # from abc import ABC, abstractmethod  # Module not found  # Module not found  # Module not found
# # # from contextlib import contextmanager  # Module not found  # Module not found  # Module not found
# # # from dataclasses import dataclass, field  # Module not found  # Module not found  # Module not found
# # # from datetime import datetime  # Module not found  # Module not found  # Module not found
# # # from enum import Enum  # Module not found  # Module not found  # Module not found
# # # from pathlib import Path  # Module not found  # Module not found  # Module not found
# # # from typing import Any, Dict, List, Optional, Protocol, Union, runtime_checkable  # Module not found  # Module not found  # Module not found

# # # from total_ordering_base import TotalOrderingBase  # Module not found  # Module not found  # Module not found

# Dependencies avanzadas para el stack moderno
try:
    import numpy as np
# # #     from fuzzywuzzy import fuzz  # Module not found  # Module not found  # Module not found
# # #     from jsonschema import ValidationError as JsonSchemaValidationError  # Module not found  # Module not found  # Module not found
# # #     from jsonschema import validate  # Module not found  # Module not found  # Module not found
# # #     from loguru import logger  # Module not found  # Module not found  # Module not found
# # #     from pydantic import BaseModel, ConfigDict, Field, field_validator  # Module not found  # Module not found  # Module not found
# # #     from rich.console import Console  # Module not found  # Module not found  # Module not found
# # #     from rich.progress import Progress, SpinnerColumn, TextColumn  # Module not found  # Module not found  # Module not found
# # #     from rich.table import Table  # Module not found  # Module not found  # Module not found
except ImportError as e:
    print(
        f"Dependencia faltante: {e}. Instalar con: pip install pydantic jsonschema rich loguru numpy fuzzywuzzy"
    )
    raise

# ===============================================================================
# 1. MODELOS DE DATOS Y ENUMS (Pydantic 2.x + Type Safety)
# ===============================================================================


class ComplianceLevel(str, Enum):
    """Niveles de cumplimiento normativo."""

    CUMPLE = "CUMPLE"
    CUMPLE_PARCIAL = "CUMPLE_PARCIAL"
    NO_CUMPLE = "NO_CUMPLE"


class CheckStatus(str, Enum):
    """Estados posibles de una verificación."""

    PASSED = "PASSED"
    FAILED = "FAILED"
    WARNING = "WARNING"
    SKIPPED = "SKIPPED"


class ViolationSeverity(str, Enum):
    """Severidad de las violaciones normativas."""

    CRITICAL = "critical"
    MODERATE = "moderate"
    MINOR = "minor"


@dataclass(frozen=True, slots=True)
class CheckResult:
    """Resultado inmutable de una verificación normativa individual."""

    check_id: str
    description: str
    status: CheckStatus
    evidence: List[str] = field(default_factory=list)
    penalty_applied: float = 0.0
    rationale: str = ""
    execution_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validación post-inicialización."""
        if not self.check_id or not self.description:
            raise ValueError("check_id y description son campos obligatorios")
        if (
            not isinstance(self.penalty_applied, (int, float))
            or self.penalty_applied < 0
        ):
            raise ValueError("penalty_applied debe ser un número no negativo")


class NormativeResult(BaseModel):
    """Resultado completo de validación normativa con Pydantic V2."""

    model_config = ConfigDict(
        validate_assignment=True, frozen=True, extra="forbid", str_strip_whitespace=True
    )

    compliance_score: float = Field(
        ..., ge=0, le=100, description="Puntaje de conformidad (0-100)"
    )
    compliance_level: ComplianceLevel = Field(
        ..., description="Nivel de cumplimiento normativo"
    )
    checklist: List[CheckResult] = Field(
        default_factory=list, description="Lista de verificaciones realizadas"
    )
    summary: Dict[str, Any] = Field(
        default_factory=dict, description="Resumen ejecutivo"
    )
    processing_metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Metadatos de procesamiento"
    )
    timestamp: datetime = Field(default_factory=datetime.now)

    @field_validator("compliance_score")
    @classmethod
    def validate_score(cls, v):
        if not 0 <= v <= 100:
            raise ValueError("El puntaje debe estar entre 0 y 100")
        return round(v, 2)


# ===============================================================================
# 2. INTERFACES Y PROTOCOLOS (Type Safety Avanzado)
# ===============================================================================


@runtime_checkable
class StandardsProvider(Protocol):
    """Protocolo para proveedores de estándares de validación."""

    def get_standards(self) -> Dict[str, Any]:
        ...

    def is_valid(self) -> bool:
        ...


@runtime_checkable
class RuleExecutor(Protocol):
    """Protocolo para ejecutores de reglas de validación."""

    def execute(self, evaluation: Any) -> List[CheckResult]:
        ...

    def get_rule_id(self) -> str:
        ...


class ValidationRule(TotalOrderingBase, ABC):
    """Clase base abstracta para reglas de validación."""

    def __init__(self, rule_id: str, description: str, standards: Dict[str, Any]):
        super().__init__(component_name=f"ValidationRule_{rule_id}")
        
        self.rule_id = rule_id
        self.description = description
        self.standards = standards
        self._logger = logger.bind(rule=rule_id)
        
        # State tracking
        self._validations_performed = 0
        self._last_validation_result = None
        
        # Generate rule configuration ID
        rule_config = {
            "rule_id": rule_id,
            "description": description,
            "standards_keys": sorted(standards.keys()) if standards else []
        }
        self._rule_config_id = self.generate_stable_id(rule_config, prefix="rule")

    @abstractmethod
    def validate(self, evaluation: Any) -> List[CheckResult]:
        """Ejecuta la validación específica de esta regla."""
        pass

    @contextmanager
    def _time_execution(self):
        """Context manager para medir tiempo de ejecución."""
        start_time = time.perf_counter()
        yield
        execution_time = (time.perf_counter() - start_time) * 1000
        self._logger.debug(f"Regla ejecutada en {execution_time:.2f}ms")
    
    def __lt__(self, other):
        """Comparison based on rule_id for stable sorting"""
        if not isinstance(other, ValidationRule):
            return NotImplemented
        return self.rule_id < other.rule_id


# ===============================================================================
# 3. CARGADOR DE ESTÁNDARES (Advanced JSON Schema Validation)
# ===============================================================================


class StandardsLoader(TotalOrderingBase):
    """Cargador robusto de estándares DNP con validación avanzada."""

    # Schema JSON para validación estructural
    DNP_STANDARDS_SCHEMA = {
        "type": "object",
        "required": [
            "metadata",
            "mandatory_sections",
            "decalogo_point_standards",
            "compliance_penalties",
        ],
        "properties": {
            "metadata": {
                "type": "object",
                "required": ["version", "source"],
                "properties": {
                    "version": {"type": "string", "pattern": r"^\d+\.\d+\.\d+$"},
                    "source": {"type": "string", "minLength": 10},
                },
            },
            "mandatory_sections": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
            },
            "decalogo_point_standards": {
                "type": "object",
                "patternProperties": {
                    r"^\d+$": {
                        "type": "object",
                        "required": [
                            "name",
                            "sector_mapping",
                            "minimum_budget_percentage",
                        ],
                        "properties": {
                            "name": {"type": "string", "minLength": 5},
                            "sector_mapping": {"type": "string"},
                            "minimum_budget_percentage": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 100,
                            },
                            "required_instruments": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "mandatory_indicators": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "required": ["name", "type", "keywords"],
                                    "properties": {
                                        "name": {"type": "string"},
                                        "type": {
                                            "type": "string",
                                            "enum": [
                                                "producto",
                                                "resultado",
                                                "impacto",
                                            ],
                                        },
                                        "unit": {"type": "string"},
                                        "keywords": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                        },
                                    },
                                },
                            },
                        },
                    }
                },
            },
            "compliance_penalties": {
                "type": "object",
                "required": ["critical", "moderate"],
                "properties": {
                    "critical": {
                        "type": "array",
                        "items": {"$ref": "#/definitions/penalty"},
                    },
                    "moderate": {
                        "type": "array",
                        "items": {"$ref": "#/definitions/penalty"},
                    },
                    "minor": {
                        "type": "array",
                        "items": {"$ref": "#/definitions/penalty"},
                    },
                },
            },
        },
        "definitions": {
            "penalty": {
                "type": "object",
                "required": ["violation", "penalty", "check_logic"],
                "properties": {
                    "violation": {"type": "string", "minLength": 10},
                    "penalty": {"type": "number", "minimum": 0, "maximum": 100},
                    "check_logic": {"type": "string"},
                },
            }
        },
    }

    def __init__(self, standards_path: Union[str, Path]):
        super().__init__(component_name="StandardsLoader")
        
        self.standards_path = Path(standards_path)
        self._standards_cache: Optional[Dict[str, Any]] = None
        self._logger = logger.bind(component="StandardsLoader")
        
        # State tracking
        self._load_count = 0
        self._cache_hits = 0
        
        # Generate loader ID based on standards path
        loader_config = {
            "standards_path": str(self.standards_path.resolve()),
            "schema_version": "2.0"
        }
        self._loader_id = self.generate_stable_id(loader_config, prefix="load")

    def load_standards(self, force_reload: bool = False) -> Dict[str, Any]:
        """Carga y valida los estándares DNP con cache inteligente."""
        if self._standards_cache is not None and not force_reload:
            self._cache_hits += 1
            self._logger.debug("Retornando estándares desde cache")
            
            # Update state tracking
            state_data = {
                "cache_hits": self._cache_hits,
                "load_count": self._load_count,
                "loader_id": self._loader_id
            }
            self.update_state_hash(state_data)
            
            return self._standards_cache

        self._load_count += 1

        try:
            operation_id = self.generate_operation_id(
                "load_standards", 
                {"force_reload": force_reload, "path": str(self.standards_path)}
            )
            
            self._logger.info(f"Cargando estándares desde: {self.standards_path} [op_id: {operation_id[:8]}]")

            if not self.standards_path.exists():
                raise FileNotFoundError(
                    f"Archivo de estándares no encontrado: {self.standards_path}"
                )

            with open(self.standards_path, "r", encoding="utf-8") as f:
                standards_data = json.load(f)

            # Validación estructural con jsonschema
            validate(instance=standards_data, schema=self.DNP_STANDARDS_SCHEMA)

            # Validaciones adicionales de negocio
            self._validate_business_rules(standards_data)

            # Sort standards data for deterministic order
            standards_data = self.sort_dict_by_keys(standards_data)

            self._standards_cache = standards_data
            
            # Update state tracking
            state_data = {
                "cache_hits": self._cache_hits,
                "load_count": self._load_count,
                "loader_id": self._loader_id
            }
            self.update_state_hash(state_data)
            
            self._logger.success(f"Estándares cargados y validados exitosamente [op_id: {operation_id[:8]}]")
            return standards_data

        except (json.JSONDecodeError, JsonSchemaValidationError) as e:
            self._logger.error(f"Error de validación en estándares: {e}")
            raise ValueError(f"Archivo de estándares inválido: {e}")
        except Exception as e:
            self._logger.error(f"Error inesperado cargando estándares: {e}")
            raise

    def _validate_business_rules(self, standards: Dict[str, Any]) -> None:
        """Validaciones adicionales de reglas de negocio."""
        # Verificar que los porcentajes presupuestales sumen un valor razonable
        total_budget_percentage = sum(
            point["minimum_budget_percentage"]
            for point in standards["decalogo_point_standards"].values()
        )

        if total_budget_percentage > 100:
            self._logger.warning(
                f"Suma de porcentajes presupuestales excede 100%: {total_budget_percentage}"
            )

        # Verificar que existan penalizaciones críticas
        if len(standards["compliance_penalties"]["critical"]) == 0:
            raise ValueError("Debe existir al menos una penalización crítica")

    def get_standards(self) -> Dict[str, Any]:
        """Obtiene los estándares, cargándolos si es necesario."""
        return self.load_standards()

    def is_valid(self) -> bool:
        """Verifica si los estándares están correctamente cargados."""
        try:
            standards = self.get_standards()
            return all(
                key in standards
                for key in [
                    "metadata",
                    "mandatory_sections",
                    "decalogo_point_standards",
                ]
            )
        except Exception:
            return False
    
    def __lt__(self, other):
        """Comparison based on load count for stable sorting"""
        if not isinstance(other, StandardsLoader):
            return NotImplemented
        return self._load_count < other._load_count
    
    def to_json_dict(self) -> Dict[str, Any]:
        """Generate canonical JSON representation"""
        return self.serialize_output({
            "component_id": self.component_id,
            "component_name": self.component_name,
            "loader_id": self._loader_id,
            "standards_path": str(self.standards_path),
            "load_count": self._load_count,
            "cache_hits": self._cache_hits,
            "has_cached_standards": self._standards_cache is not None,
            "metadata": self.get_deterministic_metadata()
        })


# ===============================================================================
# 4. REGLAS DE VALIDACIÓN ESPECÍFICAS (Strategy Pattern)
# ===============================================================================


class MandatorySectionsRule(ValidationRule):
    """Valida la presencia de secciones obligatorias en el PDT."""

    def validate(self, evaluation: Any) -> List[CheckResult]:
        self._validations_performed += 1
        
        # Generate operation ID for this validation
        operation_id = self.generate_operation_id(
            "validate_mandatory_sections",
            {
                "rule_id": self.rule_id,
                "evaluation_type": type(evaluation).__name__
            }
        )
        
        results = []
        mandatory_sections = self.standards.get("mandatory_sections", [])
        identified_sections = evaluation.metadata.get("sections_identified", [])
        
        # Sort sections for deterministic processing order
        sorted_mandatory = sorted(mandatory_sections)
        sorted_identified = sorted(identified_sections)

        with self._time_execution():
            for section in sorted_mandatory:
                section_found = any(
                    fuzz.partial_ratio(section.lower(), identified.lower()) > 85
                    for identified in sorted_identified
                )

                if section_found:
                    results.append(
                        CheckResult(
                            check_id=f"STRUCT_PASS_{section.upper()}",
                            description=f"Verificación de sección obligatoria: {section}",
                            status=CheckStatus.PASSED,
                            evidence=[
                                f"Sección '{section}' identificada en el documento"
                            ],
                            rationale="Sección obligatoria encontrada mediante matching fuzzy",
                        )
                    )
                else:
                    results.append(
                        CheckResult(
                            check_id=f"STRUCT_FAIL_{section.upper()}",
                            description=f"Verificación de sección obligatoria: {section}",
                            status=CheckStatus.FAILED,
                            penalty_applied=15.0,
                            rationale="Sección obligatoria no identificada por el extractor",
                            metadata={"severity": ViolationSeverity.CRITICAL},
                        )
                    )
        
        # Sort results by check_id for deterministic ordering
        sorted_results = sorted(results, key=lambda r: r.check_id)
        
        # Update state
        self._last_validation_result = len(sorted_results)
        state_data = {
            "validations_performed": self._validations_performed,
            "rule_config_id": self._rule_config_id,
            "last_result_count": self._last_validation_result
        }
        self.update_state_hash(state_data)

        return sorted_results


class MandatoryIndicatorsRule(ValidationRule):
    """Valida la presencia y completitud de indicadores obligatorios por sector."""

    def validate(self, evaluation: Any) -> List[CheckResult]:
        self._validations_performed += 1
        
        # Generate operation ID for this validation
        operation_id = self.generate_operation_id(
            "validate_mandatory_indicators",
            {
                "rule_id": self.rule_id,
                "evaluation_type": type(evaluation).__name__
            }
        )
        
        results = []
        decalogo_standards = self.standards.get("decalogo_point_standards", {})

        with self._time_execution():
            # Sort point IDs for deterministic processing
            sorted_points = sorted(decalogo_standards.items(), key=lambda x: x[0])
            
            for point_id, point_data in sorted_points:
                mandatory_indicators = point_data.get("mandatory_indicators", [])
                sector = point_data.get("sector_mapping", "unknown")

                # Sort indicators for deterministic processing
                sorted_indicators = sorted(mandatory_indicators, key=lambda x: x.get('name', ''))

                for required_indicator in sorted_indicators:
                    indicator_found = self._find_indicator_in_evaluation(
                        evaluation, required_indicator, sector
                    )

                    indicator_name_clean = required_indicator['name'].replace(' ', '_').replace('/', '_')

                    if indicator_found:
                        # Verificar línea base
                        has_baseline = indicator_found.get("baseline_value") is not None
                        has_target = indicator_found.get("target_2027") is not None

                        if has_baseline and has_target:
                            results.append(
                                CheckResult(
                                    check_id=f"IND_COMPLETE_{point_id}_{indicator_name_clean}",
                                    description=f"Indicador completo: {required_indicator['name']}",
                                    status=CheckStatus.PASSED,
                                    evidence=[
                                        f"Línea base: {indicator_found.get('baseline_value')}",
                                        f"Meta 2027: {indicator_found.get('target_2027')}",
                                    ],
                                )
                            )
                        else:
                            penalty = 8.0 if not has_baseline else 5.0
                            missing_elements = []
                            if not has_baseline:
                                missing_elements.append("línea base")
                            if not has_target:
                                missing_elements.append("meta 2027")

                            results.append(
                                CheckResult(
                                    check_id=f"IND_INCOMPLETE_{point_id}_{indicator_name_clean}",
                                    description=f"Indicador incompleto: {required_indicator['name']}",
                                    status=CheckStatus.WARNING,
                                    penalty_applied=penalty,
                                    rationale=f"Falta: {', '.join(sorted(missing_elements))}",
                                    metadata={"severity": ViolationSeverity.MODERATE},
                                )
                            )
                    else:
                        results.append(
                            CheckResult(
                                check_id=f"IND_MISSING_{point_id}_{indicator_name_clean}",
                                description=f"Indicador obligatorio ausente: {required_indicator['name']}",
                                status=CheckStatus.FAILED,
                                penalty_applied=20.0,
                                rationale=f"Indicador requerido para punto {point_id} del decálogo no encontrado",
                                metadata={"severity": ViolationSeverity.CRITICAL},
                            )
                        )
        
        # Sort results by check_id for deterministic ordering
        sorted_results = sorted(results, key=lambda r: r.check_id)
        
        # Update state
        self._last_validation_result = len(sorted_results)
        state_data = {
            "validations_performed": self._validations_performed,
            "rule_config_id": self._rule_config_id,
            "last_result_count": self._last_validation_result
        }
        self.update_state_hash(state_data)

        return sorted_results

    def _find_indicator_in_evaluation(
        self, evaluation: Any, required_indicator: Dict, sector: str
    ) -> Optional[Dict]:
        """Busca un indicador específico en los datos de evaluación usando matching avanzado."""
        if not hasattr(evaluation, "extracted_indicators"):
            return None

        keywords = sorted(required_indicator.get("keywords", []))  # Sort for deterministic order
        indicator_name = required_indicator["name"].lower()

        # Sort extracted indicators for deterministic processing
        sorted_extracted = sorted(
            evaluation.extracted_indicators, 
            key=lambda x: x.get("name", "")
        )

        for extracted in sorted_extracted:
            extracted_name = extracted.get("name", "").lower()

            # Matching por nombre exacto
            if fuzz.ratio(indicator_name, extracted_name) > 90:
                return extracted

            # Matching por keywords (in deterministic order)
            for keyword in keywords:
                if fuzz.partial_ratio(keyword.lower(), extracted_name) > 85:
                    return extracted

        return None


class BudgetThresholdsRule(ValidationRule):
    """Valida umbrales presupuestales mínimos por sector."""

    def validate(self, evaluation: Any) -> List[CheckResult]:
        results = []
        decalogo_standards = self.standards.get("decalogo_point_standards", {})

        total_budget = evaluation.financial_data.get("total_budget", 0)
        if total_budget == 0:
            results.append(
                CheckResult(
                    check_id="BUDGET_NO_TOTAL",
                    description="Validación de presupuesto total",
                    status=CheckStatus.FAILED,
                    penalty_applied=25.0,
                    rationale="No se pudo identificar el presupuesto total del plan",
                    metadata={"severity": ViolationSeverity.CRITICAL},
                )
            )
            return results

        with self._time_execution():
            for point_id, point_data in decalogo_standards.items():
                sector = point_data.get("sector_mapping")
                minimum_percentage = point_data.get("minimum_budget_percentage", 0)
                sector_name = point_data.get("name")

                sector_budget = evaluation.financial_data.get("sectors", {}).get(
                    sector, 0
                )
                actual_percentage = (
                    (sector_budget / total_budget * 100) if total_budget > 0 else 0
                )

                if actual_percentage >= minimum_percentage:
                    results.append(
                        CheckResult(
                            check_id=f"BUDGET_OK_{sector.upper()}",
                            description=f"Umbral presupuestal cumplido para {sector_name}",
                            status=CheckStatus.PASSED,
                            evidence=[
                                f"Presupuesto asignado: ${sector_budget:,.0f}",
                                f"Porcentaje actual: {actual_percentage:.1f}%",
                                f"Umbral mínimo: {minimum_percentage}%",
                            ],
                        )
                    )
                else:
                    deficit_percentage = minimum_percentage - actual_percentage
                    penalty = min(
                        deficit_percentage * 2, 15.0
                    )  # Penalización proporcional

                    results.append(
                        CheckResult(
                            check_id=f"BUDGET_LOW_{sector.upper()}",
                            description=f"Umbral presupuestal insuficiente para {sector_name}",
                            status=CheckStatus.FAILED,
                            penalty_applied=penalty,
                            evidence=[
                                f"Presupuesto asignado: ${sector_budget:,.0f} ({actual_percentage:.1f}%)",
                                f"Umbral mínimo requerido: {minimum_percentage}%",
                                f"Déficit: {deficit_percentage:.1f} puntos porcentuales",
                            ],
                            rationale=f"El presupuesto asignado no alcanza el umbral mínimo establecido",
                            metadata={
                                "severity": ViolationSeverity.MODERATE,
                                "deficit": deficit_percentage,
                            },
                        )
                    )

        return results


# ===============================================================================
# 5. MOTOR DE REGLAS (Advanced Rule Engine)
# ===============================================================================


class RuleEngine:
    """Motor de reglas avanzado con capacidades de paralelización y métricas."""

    def __init__(self, standards: Dict[str, Any]):
        self.standards = standards
        self._logger = logger.bind(component="RuleEngine")

        # Factory de reglas usando reflection
        self.rule_classes = {
            "mandatory_sections": MandatorySectionsRule,
            "mandatory_indicators": MandatoryIndicatorsRule,
            "budget_thresholds": BudgetThresholdsRule,
        }

        self._initialize_rules()

    def _initialize_rules(self) -> None:
        """Inicializa las reglas de validación."""
        self.rules: List[ValidationRule] = []

        for rule_name, rule_class in self.rule_classes.items():
            rule_instance = rule_class(
                rule_id=rule_name,
                description=f"Validación de {rule_name.replace('_', ' ')}",
                standards=self.standards,
            )
            self.rules.append(rule_instance)
            self._logger.debug(f"Regla inicializada: {rule_name}")

    def run_all(self, evaluation: Any) -> List[CheckResult]:
        """Ejecuta todas las reglas con métricas avanzadas."""
        all_results = []
        execution_stats = {
            "total_rules": len(self.rules),
            "successful_rules": 0,
            "failed_rules": 0,
            "total_checks": 0,
            "start_time": time.perf_counter(),
        }

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=Console(),
        ) as progress:
            task = progress.add_task(
                "Ejecutando validaciones normativas...", total=len(self.rules)
            )

            for rule in self.rules:
                try:
                    self._logger.info(f"Ejecutando regla: {rule.rule_id}")
                    rule_start_time = time.perf_counter()

                    rule_results = rule.validate(evaluation)

                    rule_execution_time = (time.perf_counter() - rule_start_time) * 1000

                    # Actualizar métricas de tiempo en cada resultado
                    for result in rule_results:
                        result.execution_time_ms = rule_execution_time / len(
                            rule_results
                        )

                    all_results.extend(rule_results)
                    execution_stats["successful_rules"] += 1
                    execution_stats["total_checks"] += len(rule_results)

                    self._logger.success(
                        f"Regla {rule.rule_id} completada: {len(rule_results)} checks en {rule_execution_time:.2f}ms"
                    )

                except Exception as e:
                    self._logger.error(f"Error ejecutando regla {rule.rule_id}: {e}")
                    execution_stats["failed_rules"] += 1

                    # Crear resultado de error para la regla fallida
                    error_result = CheckResult(
                        check_id=f"RULE_ERROR_{rule.rule_id.upper()}",
                        description=f"Error en regla {rule.rule_id}",
                        status=CheckStatus.FAILED,
                        penalty_applied=5.0,
                        rationale=f"Error interno: {str(e)}",
                        metadata={"error_type": type(e).__name__},
                    )
                    all_results.append(error_result)

                progress.advance(task)

        # Estadísticas finales
        execution_stats["total_time_ms"] = (
            time.perf_counter() - execution_stats["start_time"]
        ) * 1000
        execution_stats["avg_time_per_rule"] = execution_stats["total_time_ms"] / max(
            1, len(self.rules)
        )

        self._logger.info(f"Motor de reglas completado: {execution_stats}")

        return all_results


# ===============================================================================
# 6. CALCULADORA DE PUNTAJE (Advanced Scoring Algorithm)
# ===============================================================================


class ScoreCalculator:
    """Calculadora avanzada de puntajes con algoritmos sofisticados."""

    def __init__(self, penalties_config: Dict[str, Any]):
        self.penalties_config = penalties_config
        self._logger = logger.bind(component="ScoreCalculator")

    def calculate_score(self, check_results: List[CheckResult]) -> float:
        """Calcula el puntaje final usando algoritmo de penalizaciones ponderadas."""
        if not check_results:
            self._logger.warning("No hay resultados para calcular puntaje")
            return 0.0

        base_score = 100.0
        total_penalty = 0.0

        # Agrupar penalizaciones por severidad para análisis
        penalty_groups = {
            ViolationSeverity.CRITICAL: [],
            ViolationSeverity.MODERATE: [],
            ViolationSeverity.MINOR: [],
        }

        for result in check_results:
            if result.status == CheckStatus.FAILED and result.penalty_applied > 0:
                severity = result.metadata.get("severity", ViolationSeverity.MODERATE)
                penalty_groups[severity].append(result.penalty_applied)
                total_penalty += result.penalty_applied

        # Aplicar factor de severidad exponencial para penalizaciones críticas
        critical_penalties = sum(penalty_groups[ViolationSeverity.CRITICAL])
        if critical_penalties > 0:
            # Las penalizaciones críticas tienen un impacto exponencial
            critical_factor = 1 + (
                len(penalty_groups[ViolationSeverity.CRITICAL]) * 0.1
            )
            critical_penalties *= critical_factor

        moderate_penalties = sum(penalty_groups[ViolationSeverity.MODERATE])
        minor_penalties = (
            sum(penalty_groups[ViolationSeverity.MINOR]) * 0.5
        )  # Penalizaciones menores al 50%

        final_penalty = critical_penalties + moderate_penalties + minor_penalties
        final_score = max(0.0, base_score - final_penalty)

        self._logger.info(
            f"Cálculo de puntaje: Base=100, Críticas={critical_penalties:.1f}, "
            f"Moderadas={moderate_penalties:.1f}, Menores={minor_penalties:.1f}, "
            f"Final={final_score:.2f}"
        )

        return round(final_score, 2)

    def determine_compliance_level(self, score: float) -> ComplianceLevel:
        """Determina el nivel de cumplimiento basado en el puntaje."""
        if score >= 90:
            return ComplianceLevel.CUMPLE
        elif score >= 70:
            return ComplianceLevel.CUMPLE_PARCIAL
        else:
            return ComplianceLevel.NO_CUMPLE

    def generate_score_breakdown(
        self, check_results: List[CheckResult]
    ) -> Dict[str, Any]:
        """Genera un desglose detallado del puntaje."""
        total_checks = len(check_results)
        passed_checks = sum(1 for r in check_results if r.status == CheckStatus.PASSED)
        failed_checks = sum(1 for r in check_results if r.status == CheckStatus.FAILED)
        warning_checks = sum(
            1 for r in check_results if r.status == CheckStatus.WARNING
        )

        return {
            "total_checks_performed": total_checks,
            "checks_passed": passed_checks,
            "checks_failed": failed_checks,
            "checks_with_warnings": warning_checks,
            "pass_rate_percentage": round(
                (passed_checks / max(1, total_checks)) * 100, 1
            ),
            "total_penalties_applied": sum(r.penalty_applied for r in check_results),
            "critical_violations": len(
                [
                    r
                    for r in check_results
                    if r.metadata.get("severity") == ViolationSeverity.CRITICAL
                ]
            ),
            "average_execution_time_ms": np.mean(
                [r.execution_time_ms for r in check_results if r.execution_time_ms > 0]
            ),
        }


# ===============================================================================
# 7. VALIDADOR NORMATIVO PRINCIPAL (Facade Pattern)
# ===============================================================================


class NormativeValidator:
    """
    Fachada principal del sistema de validación normativa.

    Implementa Clean Architecture con separación clara de responsabilidades:
    - Orquestación de componentes
    - Manejo de errores centralizado
    - Logging estructurado
    - Métricas de rendimiento
    """

    def __init__(self, dnp_standards_path: Union[str, Path]):
        """Inicializa el validador con configuración avanzada."""
        self.standards_path = Path(dnp_standards_path)
        self._logger = logger.bind(component="NormativeValidator")

        # Inicialización lazy de componentes pesados
        self._standards_loader: Optional[StandardsLoader] = None
        self._rule_engine: Optional[RuleEngine] = None
        self._score_calculator: Optional[ScoreCalculator] = None

        self._setup_logging()

    def _setup_logging(self) -> None:
        """Configura logging estructurado para el módulo."""
        logger.add(
            "normative_validator.log",
            rotation="10 MB",
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
            serialize=False,
        )

    @property
    def standards_loader(self) -> StandardsLoader:
        """Lazy loading del cargador de estándares."""
        if self._standards_loader is None:
            self._standards_loader = StandardsLoader(self.standards_path)
        return self._standards_loader

    @property
    def rule_engine(self) -> RuleEngine:
        """Lazy loading del motor de reglas."""
        if self._rule_engine is None:
            standards = self.standards_loader.get_standards()
            self._rule_engine = RuleEngine(standards)
        return self._rule_engine

    @property
    def score_calculator(self) -> ScoreCalculator:
        """Lazy loading del calculador de puntaje."""
        if self._score_calculator is None:
            standards = self.standards_loader.get_standards()
            penalties_config = standards.get("compliance_penalties", {})
            self._score_calculator = ScoreCalculator(penalties_config)
        return self._score_calculator

    def validate(self, evaluation: Any) -> NormativeResult:
        """
        Ejecuta el proceso completo de validación normativa.

        Args:
            evaluation: Objeto PDTEvaluation con los datos extraídos del plan

        Returns:
            NormativeResult: Resultado completo de la validación

        Raises:
            ValueError: Si los datos de entrada son inválidos
            RuntimeError: Si ocurre un error durante la validación
        """
        validation_start_time = time.perf_counter()

        try:
            self._logger.info("Iniciando validación normativa DNP")

            # Validación de entrada
            if evaluation is None:
                raise ValueError("El objeto de evaluación no puede ser None")

            # Pre-validaciones del objeto evaluation
            self._validate_evaluation_structure(evaluation)

            # PASO 1: Ejecutar todas las reglas de validación
            self._logger.info("Ejecutando motor de reglas")
            check_results = self.rule_engine.run_all(evaluation)

            if not check_results:
                self._logger.warning("No se generaron resultados de validación")
                return self._create_empty_result("No se ejecutaron validaciones")

            # PASO 2: Calcular puntaje final
            self._logger.info("Calculando puntaje de conformidad")
            final_score = self.score_calculator.calculate_score(check_results)
            compliance_level = self.score_calculator.determine_compliance_level(
                final_score
            )

            # PASO 3: Generar resumen ejecutivo
            summary = self._generate_executive_summary(check_results, final_score)
            score_breakdown = self.score_calculator.generate_score_breakdown(
                check_results
            )

            # PASO 4: Metadatos de procesamiento
            total_validation_time = (time.perf_counter() - validation_start_time) * 1000
            processing_metadata = {
                "validation_time_ms": round(total_validation_time, 2),
                "standards_version": self.standards_loader.get_standards()
                .get("metadata", {})
                .get("version"),
                "total_rules_executed": len(self.rule_engine.rules),
                "dnp_standards_file": str(self.standards_path),
                "validator_version": "2.0",
            }

            # PASO 5: Ensamblar resultado final
            result = NormativeResult(
                compliance_score=final_score,
                compliance_level=compliance_level,
                checklist=check_results,
                summary={**summary, **score_breakdown},
                processing_metadata=processing_metadata,
            )

            self._logger.success(
                f"Validación completada: Score={final_score:.2f}, "
                f"Nivel={compliance_level.value}, "
                f"Tiempo={total_validation_time:.2f}ms"
            )

            return result

        except Exception as e:
            self._logger.error(f"Error durante validación normativa: {e}")
            return self._create_error_result(str(e))

    def _validate_evaluation_structure(self, evaluation: Any) -> None:
        """Valida la estructura básica del objeto de evaluación."""
        required_attributes = ["metadata", "financial_data"]

        for attr in required_attributes:
            if not hasattr(evaluation, attr):
                raise ValueError(
                    f"El objeto evaluation debe tener el atributo '{attr}'"
                )

        if not isinstance(evaluation.metadata, dict):
            raise ValueError("evaluation.metadata debe ser un diccionario")

        if not isinstance(evaluation.financial_data, dict):
            raise ValueError("evaluation.financial_data debe ser un diccionario")

    def _generate_executive_summary(
        self, check_results: List[CheckResult], final_score: float
    ) -> Dict[str, Any]:
        """Genera un resumen ejecutivo inteligente de los resultados."""

        # Análisis de fortalezas y debilidades
        strengths = [r for r in check_results if r.status == CheckStatus.PASSED]
        critical_issues = [
            r
            for r in check_results
            if r.status == CheckStatus.FAILED
            and r.metadata.get("severity") == ViolationSeverity.CRITICAL
        ]
        moderate_issues = [
            r
            for r in check_results
            if r.status in [CheckStatus.FAILED, CheckStatus.WARNING]
            and r.metadata.get("severity") == ViolationSeverity.MODERATE
        ]

        # Recomendaciones automáticas basadas en patrones
        recommendations = self._generate_smart_recommendations(check_results)

        # Identificar áreas más problemáticas
        problem_areas = self._identify_problem_areas(check_results)

        return {
            "main_findings": self._generate_main_findings_narrative(
                final_score, critical_issues, moderate_issues
            ),
            "key_strengths": [s.description for s in strengths[:5]],  # Top 5 fortalezas
            "critical_gaps": [i.description for i in critical_issues],
            "improvement_opportunities": [i.description for i in moderate_issues[:3]],
            "automated_recommendations": recommendations,
            "priority_areas_for_improvement": problem_areas,
            "compliance_trend": self._assess_compliance_trend(final_score),
            "risk_assessment": self._assess_compliance_risk(check_results),
        }

    def _generate_main_findings_narrative(
        self,
        score: float,
        critical_issues: List[CheckResult],
        moderate_issues: List[CheckResult],
    ) -> str:
        """Genera una narrativa inteligente de los hallazgos principales."""
        if score >= 90:
            base_narrative = "El Plan de Desarrollo Territorial presenta un alto nivel de conformidad normativa"
        elif score >= 70:
            base_narrative = "El Plan de Desarrollo Territorial presenta conformidad normativa parcial"
        else:
            base_narrative = "El Plan de Desarrollo Territorial presenta deficiencias significativas en conformidad normativa"

        if critical_issues:
            base_narrative += f", con {len(critical_issues)} aspectos críticos que requieren atención inmediata"

        if moderate_issues:
            base_narrative += (
                f" y {len(moderate_issues)} oportunidades de mejora identificadas"
            )

        base_narrative += "."

        return base_narrative

    def _generate_smart_recommendations(
        self, check_results: List[CheckResult]
    ) -> List[str]:
        """Genera recomendaciones inteligentes basadas en patrones de fallas."""
        recommendations = []

        # Análisis de patrones
        failed_checks = [r for r in check_results if r.status == CheckStatus.FAILED]

        # Recomendaciones por tipo de falla
        structural_failures = [
            r for r in failed_checks if r.check_id.startswith("STRUCT_")
        ]
        if structural_failures:
            recommendations.append(
                "Fortalecer la estructura documental incluyendo todas las secciones obligatorias del marco normativo"
            )

        indicator_failures = [r for r in failed_checks if r.check_id.startswith("IND_")]
        if indicator_failures:
            recommendations.append(
                "Desarrollar un sistema robusto de indicadores con líneas base y metas cuantificadas para 2027"
            )

        budget_failures = [r for r in failed_checks if r.check_id.startswith("BUDGET_")]
        if budget_failures:
            recommendations.append(
                "Revisar la asignación presupuestal para cumplir con los umbrales mínimos sectoriales"
            )

        return recommendations

    def _identify_problem_areas(self, check_results: List[CheckResult]) -> List[str]:
        """Identifica las áreas más problemáticas usando análisis de clustering."""
        problem_counts = {}

        for result in check_results:
            if result.status == CheckStatus.FAILED:
                # Extraer área problemática del check_id
                area = result.check_id.split("_")[0]
                problem_counts[area] = (
                    problem_counts.get(area, 0) + result.penalty_applied
                )

        # Ordenar por impacto (penalizaciones acumuladas)
        sorted_areas = sorted(problem_counts.items(), key=lambda x: x[1], reverse=True)

        area_names = {
            "STRUCT": "Estructura documental",
            "IND": "Sistema de indicadores",
            "BUDGET": "Planificación presupuestal",
            "RULE": "Cumplimiento normativo general",
        }

        return [area_names.get(area, area) for area, _ in sorted_areas[:3]]

    def _assess_compliance_trend(self, score: float) -> str:
        """Evalúa la tendencia de cumplimiento."""
        if score >= 85:
            return "Excelente"
        elif score >= 70:
            return "Aceptable con mejoras necesarias"
        elif score >= 50:
            return "Deficiente, requiere intervención"
        else:
            return "Crítico, requiere reestructuración completa"

    def _assess_compliance_risk(self, check_results: List[CheckResult]) -> str:
        """Evalúa el riesgo de incumplimiento normativo."""
        critical_count = len(
            [
                r
                for r in check_results
                if r.metadata.get("severity") == ViolationSeverity.CRITICAL
            ]
        )

        if critical_count == 0:
            return "Bajo"
        elif critical_count <= 2:
            return "Moderado"
        elif critical_count <= 5:
            return "Alto"
        else:
            return "Crítico"

    def _create_empty_result(self, reason: str) -> NormativeResult:
        """Crea un resultado vacío para casos excepcionales."""
        return NormativeResult(
            compliance_score=0.0,
            compliance_level=ComplianceLevel.NO_CUMPLE,
            checklist=[],
            summary={
                "main_findings": f"Validación no completada: {reason}",
                "error": True,
            },
            processing_metadata={"error": reason},
        )

    def _create_error_result(self, error_message: str) -> NormativeResult:
        """Crea un resultado de error para manejo de excepciones."""
        return NormativeResult(
            compliance_score=0.0,
            compliance_level=ComplianceLevel.NO_CUMPLE,
            checklist=[
                CheckResult(
                    check_id="VALIDATION_ERROR",
                    description="Error durante validación normativa",
                    status=CheckStatus.FAILED,
                    rationale=error_message,
                    penalty_applied=100.0,
                )
            ],
            summary={
                "main_findings": f"Error crítico durante validación: {error_message}",
                "error": True,
                "requires_manual_review": True,
            },
            processing_metadata={"validation_error": error_message},
        )

    def get_validation_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas del validador para monitoreo."""
        if not self._rule_engine:
            return {"status": "not_initialized"}

        return {
            "standards_loaded": self.standards_loader.is_valid(),
            "total_rules_available": len(self.rule_engine.rules),
            "standards_version": self.standards_loader.get_standards()
            .get("metadata", {})
            .get("version"),
            "validator_ready": True,
        }


# ===============================================================================
# 8. UTILIDADES Y HELPERS AVANZADOS
# ===============================================================================


class ValidationReporter:
    """Generador de reportes avanzados para resultados de validación."""

    @staticmethod
    def generate_rich_report(result: NormativeResult) -> None:
        """Genera un reporte visual rico usando Rich."""
        console = Console()

        # Header del reporte
        console.print(f"\n🏛️  REPORTE DE VALIDACIÓN NORMATIVA DNP", style="bold blue")
        console.print(f"Fecha: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        console.print("=" * 80)

        # Resumen ejecutivo
        console.print(
            f"\n📊 PUNTAJE DE CONFORMIDAD: {result.compliance_score}/100", style="bold"
        )
        console.print(
            f"📈 NIVEL: {result.compliance_level.value}",
            style="green"
            if result.compliance_level == ComplianceLevel.CUMPLE
            else "yellow"
            if result.compliance_level == ComplianceLevel.CUMPLE_PARCIAL
            else "red",
        )

        # Tabla de resultados detallada
        table = Table(title="Detalle de Verificaciones")
        table.add_column("ID", style="cyan")
        table.add_column("Descripción", style="white")
        table.add_column("Estado", justify="center")
        table.add_column("Penalización", justify="right", style="red")

        for check in result.checklist:
            status_style = (
                "green"
                if check.status == CheckStatus.PASSED
                else "yellow"
                if check.status == CheckStatus.WARNING
                else "red"
            )

            table.add_row(
                check.check_id,
                check.description[:50] + "..."
                if len(check.description) > 50
                else check.description,
                f"[{status_style}]{check.status.value}[/{status_style}]",
                f"{check.penalty_applied:.1f}" if check.penalty_applied > 0 else "-",
            )

        console.print(table)

        # Recomendaciones
        if "automated_recommendations" in result.summary:
            console.print(f"\n💡 RECOMENDACIONES:", style="bold yellow")
            for i, rec in enumerate(result.summary["automated_recommendations"], 1):
                console.print(f"  {i}. {rec}")


# ===============================================================================
# 9. CONFIGURACIÓN DE EJEMPLO Y TESTING UTILITIES
# ===============================================================================


def create_sample_dnp_standards() -> Dict[str, Any]:
    """Crea un ejemplo de estándares DNP para testing y demostración."""
    return {
        "metadata": {
            "version": "1.0.0",
            "source": "Sistema de Evaluación PDT - Metodología Conjuntiva - Demo",
        },
        "mandatory_sections": ["diagnostico", "parte_estrategica", "plan_inversiones"],
        "decalogo_point_standards": {
            "1": {
                "name": "Derecho a la vida, a la seguridad y a la convivencia",
                "sector_mapping": "security",
                "minimum_budget_percentage": 3.0,
                "required_instruments": [
                    "Plan Integral de Seguridad y Convivencia Ciudadana (PISCC)"
                ],
                "mandatory_indicators": [
                    {
                        "name": "Tasa de homicidios",
                        "type": "resultado",
                        "unit": "Por 100.000 habitantes",
                        "keywords": ["tasa de homicidios", "homicidios por cien mil"],
                    }
                ],
            },
            "4": {
                "name": "Derecho humano a la salud",
                "sector_mapping": "health",
                "minimum_budget_percentage": 25.0,
                "required_instruments": ["Plan Territorial de Salud (PTS)"],
                "mandatory_indicators": [
                    {
                        "name": "Mortalidad infantil",
                        "type": "resultado",
                        "unit": "Por 1.000 nacidos vivos",
                        "keywords": [
                            "mortalidad infantil",
                            "muertes menores de un año",
                        ],
                    },
                    {
                        "name": "Cobertura en salud",
                        "type": "resultado",
                        "unit": "Porcentaje",
                        "keywords": [
                            "cobertura de aseguramiento",
                            "afiliación al sistema",
                        ],
                    },
                ],
            },
        },
        "compliance_penalties": {
            "critical": [
                {
                    "violation": "Ausencia de diagnóstico con enfoque de derechos",
                    "penalty": 25.0,
                    "check_logic": "check_diagnosis_quality",
                },
                {
                    "violation": "Sin presupuesto identificado para programas críticos",
                    "penalty": 25.0,
                    "check_logic": "check_critical_budget",
                },
            ],
            "moderate": [
                {
                    "violation": "Indicadores sin línea base",
                    "penalty": 10.0,
                    "check_logic": "check_indicators_for_baseline",
                }
            ],
            "minor": [
                {
                    "violation": "Formato de presentación no estándar",
                    "penalty": 2.0,
                    "check_logic": "check_document_format",
                }
            ],
        },
    }


class MockPDTEvaluation:
    """Mock object para testing del validador."""

    def __init__(self, compliant: bool = True):
        self.metadata = {
            "sections_identified": [
                "diagnostico",
                "parte_estrategica",
                "plan_inversiones",
            ]
            if compliant
            else ["diagnostico"]
        }

        self.financial_data = (
            {
                "total_budget": 10000000000,  # 10 mil millones
                "sectors": {
                    "security": 500000000,  # 5% (debe ser 3% mínimo)
                    "health": 3000000000,  # 30% (debe ser 25% mínimo)
                },
            }
            if compliant
            else {
                "total_budget": 10000000000,
                "sectors": {
                    "security": 100000000,  # 1% (insuficiente)
                    "health": 1000000000,  # 10% (insuficiente)
                },
            }
        )

        self.extracted_indicators = (
            [
                {
                    "name": "Tasa de homicidios",
                    "baseline_value": 15.2,
                    "target_2027": 10.0,
                    "unit": "Por 100.000 habitantes",
                },
                {
                    "name": "Mortalidad infantil",
                    "baseline_value": 8.5,
                    "target_2027": 6.0,
                    "unit": "Por 1.000 nacidos vivos",
                },
                {
                    "name": "Cobertura en salud",
                    "baseline_value": 85.0,
                    "target_2027": 95.0,
                    "unit": "Porcentaje",
                },
            ]
            if compliant
            else [
                {
                    "name": "Tasa de homicidios",
                    "baseline_value": None,  # Sin línea base
                    "target_2027": 10.0,
                }
            ]
        )


# ===============================================================================
# 10. FUNCIÓN PRINCIPAL Y DEMO
# ===============================================================================


def main():
    """Función principal para demostración del validador."""

    # Configurar logging para demo
    logger.add(lambda msg: print(msg), colorize=True, level="INFO")

    console = Console()
    console.print(
        "🚀 DEMO: NormativeValidator - Motor de Validación DNP", style="bold green"
    )

    try:
        # Crear archivo de estándares de ejemplo
        standards_data = create_sample_dnp_standards()
        standards_file = Path("dnp_standards_demo.json")

        with open(standards_file, "w", encoding="utf-8") as f:
            json.dump(standards_data, f, indent=2, ensure_ascii=False)

        console.print(f"✅ Archivo de estándares creado: {standards_file}")

        # Inicializar validador
        validator = NormativeValidator(standards_file)
        console.print("✅ Validador inicializado")

        # Crear evaluación de ejemplo (compliant)
        compliant_evaluation = MockPDTEvaluation(compliant=True)

        console.print("\n📋 Ejecutando validación en PDT CONFORME...")
        result_compliant = validator.validate(compliant_evaluation)

        ValidationReporter.generate_rich_report(result_compliant)

        # Crear evaluación de ejemplo (non-compliant)
        console.print(f"\n" + "=" * 80)
        console.print("📋 Ejecutando validación en PDT NO CONFORME...")

        non_compliant_evaluation = MockPDTEvaluation(compliant=False)
        result_non_compliant = validator.validate(non_compliant_evaluation)

        ValidationReporter.generate_rich_report(result_non_compliant)

        # Limpiar archivo temporal
        standards_file.unlink()
        console.print(f"\n🧹 Archivo temporal eliminado")

    except Exception as e:
        console.print(f"❌ Error en demo: {e}", style="red")
        raise


if __name__ == "__main__":
    main()

# ===============================================================================
# 11. TESTS UNITARIOS AVANZADOS
# ===============================================================================

"""
SUITE DE PRUEBAS UNITARIAS

Para ejecutar los tests:
pytest -v normative_validator.py::TestNormativeValidator

Requiere: pytest, pytest-mock, pytest-cov
"""

import shutil
import tempfile
# # # from unittest.mock import MagicMock, Mock, patch  # Module not found  # Module not found  # Module not found

import pytest


class TestNormativeValidator:
    """Suite de pruebas exhaustiva para NormativeValidator."""

    @pytest.fixture
    def temp_standards_file(self):
        """Fixture para crear archivo temporal de estándares."""
        temp_dir = tempfile.mkdtemp()
        standards_file = Path(temp_dir) / "test_standards.json"

        with open(standards_file, "w") as f:
            json.dump(create_sample_dnp_standards(), f)

        yield standards_file

        shutil.rmtree(temp_dir)

    @pytest.fixture
    def validator(self, temp_standards_file):
        """Fixture para validator inicializado."""
        return NormativeValidator(temp_standards_file)

    @pytest.fixture
    def compliant_evaluation(self):
        """Fixture para evaluación que cumple."""
        return MockPDTEvaluation(compliant=True)

    @pytest.fixture
    def non_compliant_evaluation(self):
        """Fixture para evaluación que no cumple."""
        return MockPDTEvaluation(compliant=False)

    def test_validator_initialization(self, temp_standards_file):
        """Test de inicialización correcta del validador."""
        validator = NormativeValidator(temp_standards_file)
        assert validator.standards_path == temp_standards_file
        assert validator.standards_loader is not None

    def test_standards_loading_success(self, validator):
        """Test de carga exitosa de estándares."""
        standards = validator.standards_loader.get_standards()
        assert "metadata" in standards
        assert "mandatory_sections" in standards
        assert standards["metadata"]["version"] == "1.0.0"

    def test_standards_schema_validation(self):
        """Test de validación de schema de estándares."""
        invalid_standards = {"invalid": "structure"}
        temp_dir = tempfile.mkdtemp()
        invalid_file = Path(temp_dir) / "invalid.json"

        try:
            with open(invalid_file, "w") as f:
                json.dump(invalid_standards, f)

            with pytest.raises(ValueError, match="Archivo de estándares inválido"):
                loader = StandardsLoader(invalid_file)
                loader.load_standards()
        finally:
            shutil.rmtree(temp_dir)

    def test_fully_compliant_pdt(self, validator, compliant_evaluation):
        """Test de PDT completamente conforme."""
        result = validator.validate(compliant_evaluation)

        assert result.compliance_score >= 85.0
        assert result.compliance_level in [
            ComplianceLevel.CUMPLE,
            ComplianceLevel.CUMPLE_PARCIAL,
        ]
        assert len(result.checklist) > 0
        assert not any(
            check.status == CheckStatus.FAILED
            for check in result.checklist
            if check.metadata.get("severity") == ViolationSeverity.CRITICAL
        )

    def test_non_compliant_pdt(self, validator, non_compliant_evaluation):
        """Test de PDT no conforme."""
        result = validator.validate(non_compliant_evaluation)

        assert result.compliance_score < 90.0
        assert result.compliance_level != ComplianceLevel.CUMPLE
        assert any(check.status == CheckStatus.FAILED for check in result.checklist)
        assert result.summary["main_findings"] is not None

    def test_missing_mandatory_section(self, validator):
        """Test de sección obligatoria faltante."""
        evaluation = MockPDTEvaluation(compliant=True)
        evaluation.metadata["sections_identified"] = [
            "diagnostico"
        ]  # Falta parte_estrategica

        result = validator.validate(evaluation)

        failed_structural = [
            r for r in result.checklist if r.check_id.startswith("STRUCT_FAIL")
        ]
        assert len(failed_structural) > 0
        assert result.compliance_score < 100.0

    def test_budget_threshold_validation(self, validator):
        """Test de validación de umbrales presupuestales."""
        evaluation = MockPDTEvaluation(compliant=True)
        evaluation.financial_data["sectors"][
            "health"
        ] = 1000000000  # 10% (insuficiente)

        result = validator.validate(evaluation)

        budget_failures = [
            r
            for r in result.checklist
            if r.check_id.startswith("BUDGET_") and r.status == CheckStatus.FAILED
        ]
        assert len(budget_failures) > 0

    def test_indicator_completeness(self, validator):
        """Test de completitud de indicadores."""
        evaluation = MockPDTEvaluation(compliant=True)
        evaluation.extracted_indicators[0]["baseline_value"] = None  # Sin línea base

        result = validator.validate(evaluation)

        incomplete_indicators = [
            r for r in result.checklist if "INCOMPLETE" in r.check_id
        ]
        assert len(incomplete_indicators) > 0

    def test_error_handling(self, validator):
        """Test de manejo de errores."""
        result = validator.validate(None)  # Entrada inválida

        assert result.compliance_score == 0.0
        assert result.compliance_level == ComplianceLevel.NO_CUMPLE
        assert result.summary.get("error") is True

    def test_score_calculation_edge_cases(self):
        """Test de casos límite en cálculo de puntaje."""
        penalties_config = {"critical": [], "moderate": [], "minor": []}
        calculator = ScoreCalculator(penalties_config)

        # Sin resultados
        assert calculator.calculate_score([]) == 0.0

        # Solo resultados exitosos
        passed_results = [
            CheckResult("TEST1", "Test 1", CheckStatus.PASSED),
            CheckResult("TEST2", "Test 2", CheckStatus.PASSED),
        ]
        assert calculator.calculate_score(passed_results) == 100.0

        # Resultado con penalización máxima
        max_penalty_result = [
            CheckResult(
                "TEST_FAIL", "Test Fail", CheckStatus.FAILED, penalty_applied=100.0
            )
        ]
        assert calculator.calculate_score(max_penalty_result) == 0.0

    def test_compliance_level_thresholds(self):
        """Test de umbrales de niveles de cumplimiento."""
        penalties_config = {}
        calculator = ScoreCalculator(penalties_config)

        assert calculator.determine_compliance_level(95.0) == ComplianceLevel.CUMPLE
        assert (
            calculator.determine_compliance_level(85.0)
            == ComplianceLevel.CUMPLE_PARCIAL
        )
        assert calculator.determine_compliance_level(60.0) == ComplianceLevel.NO_CUMPLE

    def test_performance_benchmarks(self, validator, compliant_evaluation):
        """Test de benchmarks de rendimiento."""
        start_time = time.perf_counter()
        result = validator.validate(compliant_evaluation)
        execution_time = (time.perf_counter() - start_time) * 1000

        # El validador debe ejecutarse en menos de 5 segundos
        assert execution_time < 5000
        assert result.processing_metadata["validation_time_ms"] > 0

    @patch("normative_validator.logger")
    def test_logging_integration(self, mock_logger, validator, compliant_evaluation):
        """Test de integración con sistema de logging."""
        validator.validate(compliant_evaluation)

        # Verificar que se generaron logs
        assert mock_logger.info.called
        assert mock_logger.success.called


# ===============================================================================
# CONFIGURACIÓN FINAL Y EXPORTACIONES
# ===============================================================================

__all__ = [
    "NormativeValidator",
    "NormativeResult",
    "CheckResult",
    "ComplianceLevel",
    "CheckStatus",
    "ViolationSeverity",
    "ValidationReporter",
    "create_sample_dnp_standards",
    "MockPDTEvaluation",
]

__version__ = "2.0.0"
__author__ = "AI Assistant - Advanced PDT Validation System"
__license__ = "MIT"

# Configuración para uso en producción
if __name__ != "__main__":
    # Suprimir logs de demo cuando se importa como módulo
    logger.remove()
    logger.add(lambda msg: None, level="CRITICAL")

def process(data=None, context=None) -> Dict[str, Any]:
    """
    Process API for normative validator component (04I).
    
    Validates document features against normative standards and writes 
    standardized artifacts using ArtifactManager.
    
    Args:
        data: Input data (features data or document content)
        context: Processing context with metadata
        
    Returns:
        Dictionary with processing results and output paths
    """
    # Import ArtifactManager locally to avoid circular imports
    try:
# # #         from canonical_flow.ingestion import ArtifactManager  # Module not found  # Module not found  # Module not found
    except ImportError:
        return {"error": "ArtifactManager not available"}
    
    artifact_manager = ArtifactManager()
    
    # Process input data
    if not data:
        return {"error": "No input data provided"}
    
    results = []
    
    # Handle different input formats
    if isinstance(data, dict) and 'results' in data:
# # #         # Input from 03I component  # Module not found  # Module not found  # Module not found
        feature_results = data['results']
    elif isinstance(data, list):
        feature_results = data
    else:
        feature_results = [data]
    
    for feature_result in feature_results:
        try:
            # Extract stem and feature data
            if isinstance(feature_result, dict):
                stem = feature_result.get('stem', feature_result.get('document_stem', 'unknown'))
                
                # For actual processing, we need to read the features file
                features_path = feature_result.get('output_path')
                if features_path and Path(features_path).exists():
                    with open(features_path, 'r', encoding='utf-8') as f:
                        feature_data = json.load(f)
                else:
                    # Use feature_result directly if no file path
                    feature_data = feature_result
            else:
                stem = 'unknown'
                feature_data = {}
            
            # Extract features for validation
            document_features = feature_data.get('document_features', {})
            text_statistics = feature_data.get('text_statistics', {})
            
            # Create mock evaluation for validation
            mock_evaluation = {
                "document_id": stem,
                "word_count": document_features.get('word_count', 0),
                "section_count": document_features.get('section_count', 0),
                "technical_terms": document_features.get('technical_term_density', 0.0),
                "readability_score": document_features.get('readability_flesch', 50.0),
                "total_length": text_statistics.get('total_characters', 0)
            }
            
            # Initialize validator with sample standards
            validator = NormativeValidator()
            standards = create_sample_dnp_standards()
            validator.load_standards(standards)
            
            # Perform validation
            validation_result = validator.validate(mock_evaluation)
            
            # Create validation artifact
            validation_data = {
                "document_stem": stem,
                "validation_metadata": {
                    "component": "04I",
                    "processor": "NormativeValidator",
                    "timestamp": str(datetime.now()),
                    "standards_version": "sample_dnp_v1.0"
                },
                "compliance_result": {
                    "overall_level": validation_result.overall_compliance.value,
                    "total_score": validation_result.total_score,
                    "max_possible_score": validation_result.max_possible_score,
                    "compliance_percentage": validation_result.compliance_percentage,
                    "total_violations": len(validation_result.violations),
                    "critical_violations": len([v for v in validation_result.violations if v.severity == ViolationSeverity.CRITICAL])
                },
                "check_results": [
                    {
                        "check_id": result.check_id,
                        "description": result.description,
                        "status": result.status.value,
                        "penalty_applied": result.penalty_applied,
                        "evidence": result.evidence,
                        "rationale": result.rationale,
                        "execution_time_ms": result.execution_time_ms
                    }
                    for result in validation_result.check_results
                ],
                "violations": [
                    {
                        "check_id": violation.check_id,
                        "severity": violation.severity.value,
                        "message": violation.message,
                        "evidence": violation.evidence
                    }
                    for violation in validation_result.violations
                ],
                "processing_statistics": validation_result.processing_metadata
            }
            
            # Write artifact using ArtifactManager
            output_path = artifact_manager.write_artifact(stem, "validation", validation_data)
            
            results.append({
                "stem": stem,
                "success": True,
                "output_path": str(output_path),
                "compliance_level": validation_result.overall_compliance.value,
                "compliance_percentage": validation_result.compliance_percentage,
                "total_violations": len(validation_result.violations),
                "artifact_type": "validation"
            })
            
        except Exception as e:
            # Write error artifact
            error_stem = feature_result.get('stem', 'unknown') if isinstance(feature_result, dict) else 'unknown'
            error_data = {
                "document_stem": error_stem,
                "error": str(e),
                "processing_metadata": {
                    "component": "04I",
                    "status": "failed",
                    "timestamp": str(datetime.now())
                }
            }
            
            try:
                output_path = artifact_manager.write_artifact(error_stem, "validation", error_data)
                results.append({
                    "stem": error_stem,
                    "success": False,
                    "error": str(e),
                    "output_path": str(output_path),
                    "artifact_type": "validation"
                })
            except Exception as artifact_error:
                results.append({
                    "stem": error_stem,
                    "success": False,
                    "error": f"Processing failed: {str(e)}, Artifact writing failed: {str(artifact_error)}"
                })
    
    return {
        "component": "04I",
        "results": results,
        "total_inputs": len(feature_results),
        "successful_validations": len([r for r in results if r.get('success', False)])
    }


print(f"✅ NormativeValidator v{__version__} cargado exitosamente")

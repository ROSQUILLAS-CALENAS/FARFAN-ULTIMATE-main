"""
Validador de contratos para asegurar consistencia entre servicios
con esquemas JSON Schema, validación de APIs y trazabilidad
"""

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

import jsonschema
import yaml
from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)


class ContractType(str, Enum):
    """Tipos de contratos soportados"""

    API_REQUEST = "api_request"
    API_RESPONSE = "api_response"
    EVENT_SCHEMA = "event_schema"
    DATA_SCHEMA = "data_schema"
    WORKFLOW_INPUT = "workflow_input"
    WORKFLOW_OUTPUT = "workflow_output"
    MESSAGE_QUEUE = "message_queue"


class ValidationResult(BaseModel):
    """Resultado de validación de contrato"""

    valid: bool
    contract_name: str
    contract_version: str
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    validation_time: float
    data_hash: str
    timestamp: datetime = Field(default_factory=datetime.now)


class ContractVersion(BaseModel):
    """Versión específica de un contrato"""

    version: str
    schema: Dict[str, Any]
    created_at: datetime = Field(default_factory=datetime.now)
    deprecated: bool = False
    deprecation_date: Optional[datetime] = None
    migration_guide: Optional[str] = None
    breaking_changes: List[str] = Field(default_factory=list)


class Contract(BaseModel):
    """Definición de un contrato con versionado"""

    name: str
    contract_type: ContractType
    description: str
    versions: Dict[str, ContractVersion] = Field(default_factory=dict)
    current_version: str = "1.0.0"

    # Configuración de validación
    strict_mode: bool = True
    allow_additional_properties: bool = False
    require_all_properties: bool = True

    # Metadata
    owner: str = ""
    tags: List[str] = Field(default_factory=list)
    related_contracts: List[str] = Field(default_factory=list)


class ContractRegistry:
    """Registro centralizado de contratos"""

    def __init__(self):
        self.contracts: Dict[str, Contract] = {}
        self.validation_cache: Dict[str, ValidationResult] = {}
        self.validation_history: List[ValidationResult] = []

    def register_contract(self, contract: Contract):
        """Registra un nuevo contrato"""
        self.contracts[contract.name] = contract
        logger.info(
            f"Registered contract: {contract.name} (v{contract.current_version})"
        )

    def add_contract_version(
        self,
        contract_name: str,
        version: str,
        schema: Dict[str, Any],
        breaking_changes: List[str] = None,
    ):
        """Añade nueva versión a un contrato existente"""
        if contract_name not in self.contracts:
            raise ValueError(f"Contract {contract_name} not found")

        contract_version = ContractVersion(
            version=version, schema=schema, breaking_changes=breaking_changes or []
        )

        self.contracts[contract_name].versions[version] = contract_version
        logger.info(f"Added version {version} to contract {contract_name}")

    def deprecate_contract_version(
        self, contract_name: str, version: str, migration_guide: str = None
    ):
        """Marca una versión de contrato como deprecated"""
        if contract_name not in self.contracts:
            raise ValueError(f"Contract {contract_name} not found")

        if version not in self.contracts[contract_name].versions:
            raise ValueError(
                f"Version {version} not found for contract {contract_name}"
            )

        contract_version = self.contracts[contract_name].versions[version]
        contract_version.deprecated = True
        contract_version.deprecation_date = datetime.now()
        contract_version.migration_guide = migration_guide

        logger.warning(f"Deprecated contract {contract_name} version {version}")

    def get_contract(self, name: str) -> Optional[Contract]:
        """Obtiene un contrato por nombre"""
        return self.contracts.get(name)

    def get_contract_schema(
        self, name: str, version: str = None
    ) -> Optional[Dict[str, Any]]:
        """Obtiene el schema de un contrato en una versión específica"""
        contract = self.get_contract(name)
        if not contract:
            return None

        target_version = version or contract.current_version
        if target_version not in contract.versions:
            return None

        return contract.versions[target_version].schema

    def list_contracts(self) -> List[str]:
        """Lista todos los contratos registrados"""
        return list(self.contracts.keys())


class ContractValidator:
    """
    Validador principal que verifica adherencia a contratos
    usando JSON Schema y validaciones personalizadas
    """

    def __init__(self, registry: ContractRegistry = None, telemetry_collector=None):
        self.registry = registry or ContractRegistry()
        self.telemetry = telemetry_collector

        # Validadores personalizados por tipo de contrato
        self.custom_validators: Dict[ContractType, List[Callable]] = {}

        # Configuración
        self.enable_caching = True
        self.cache_ttl_seconds = 300  # 5 minutos
        self.max_validation_history = 1000

        # Estadísticas
        self.stats = {
            "validations_performed": 0,
            "validations_passed": 0,
            "validations_failed": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

    def register_custom_validator(
        self, contract_type: ContractType, validator: Callable
    ):
        """Registra validador personalizado para un tipo de contrato"""
        if contract_type not in self.custom_validators:
            self.custom_validators[contract_type] = []

        self.custom_validators[contract_type].append(validator)
        logger.info(f"Registered custom validator for {contract_type.value}")

    async def validate(
        self,
        contract_name: str,
        data: Any,
        version: str = None,
        context: Dict[str, Any] = None,
    ) -> ValidationResult:
        """
        Valida datos contra un contrato específico

        Args:
            contract_name: Nombre del contrato a validar
            data: Datos a validar
            version: Versión específica del contrato (opcional)
            context: Contexto adicional para validación

        Returns:
            Resultado de la validación
        """
        start_time = datetime.now()

        # Obtener contrato
        contract = self.registry.get_contract(contract_name)
        if not contract:
            return ValidationResult(
                valid=False,
                contract_name=contract_name,
                contract_version=version or "unknown",
                errors=[f"Contract {contract_name} not found"],
                validation_time=0.0,
                data_hash="",
            )

        target_version = version or contract.current_version

        # Verificar si la versión existe
        if target_version not in contract.versions:
            return ValidationResult(
                valid=False,
                contract_name=contract_name,
                contract_version=target_version,
                errors=[
                    f"Version {target_version} not found for contract {contract_name}"
                ],
                validation_time=0.0,
                data_hash="",
            )

        # Calcular hash de los datos para caching
        data_hash = self._calculate_data_hash(data)
        cache_key = f"{contract_name}:{target_version}:{data_hash}"

        # Verificar cache si está habilitado
        if self.enable_caching and cache_key in self.registry.validation_cache:
            cached_result = self.registry.validation_cache[cache_key]
            if (
                datetime.now() - cached_result.timestamp
            ).total_seconds() < self.cache_ttl_seconds:
                self.stats["cache_hits"] += 1
                return cached_result

        self.stats["cache_misses"] += 1

        # Realizar validación
        contract_version = contract.versions[target_version]

        # Verificar si la versión está deprecated
        warnings = []
        if contract_version.deprecated:
            warnings.append(f"Contract version {target_version} is deprecated")
            if contract_version.migration_guide:
                warnings.append(f"Migration guide: {contract_version.migration_guide}")

        # Validación con JSON Schema
        errors = []
        try:
            # Configurar validator según configuración del contrato
            validator_class = jsonschema.Draft7Validator

            # Modificar schema según configuración
            schema = contract_version.schema.copy()
            if not contract.allow_additional_properties:
                schema["additionalProperties"] = False

            validator = validator_class(schema)

            # Ejecutar validación
            json_errors = sorted(validator.iter_errors(data), key=lambda e: e.path)

            for error in json_errors:
                error_path = ".".join(str(p) for p in error.path)
                error_msg = f"Path '{error_path}': {error.message}"
                errors.append(error_msg)

        except Exception as e:
            errors.append(f"Schema validation error: {str(e)}")

        # Ejecutar validadores personalizados
        if contract.contract_type in self.custom_validators:
            for custom_validator in self.custom_validators[contract.contract_type]:
                try:
                    custom_errors = await self._run_custom_validator(
                        custom_validator, data, contract, context or {}
                    )
                    if custom_errors:
                        errors.extend(custom_errors)
                except Exception as e:
                    errors.append(f"Custom validation error: {str(e)}")

        # Crear resultado
        validation_time = (datetime.now() - start_time).total_seconds()
        is_valid = len(errors) == 0

        result = ValidationResult(
            valid=is_valid,
            contract_name=contract_name,
            contract_version=target_version,
            errors=errors,
            warnings=warnings,
            validation_time=validation_time,
            data_hash=data_hash,
        )

        # Actualizar estadísticas
        self.stats["validations_performed"] += 1
        if is_valid:
            self.stats["validations_passed"] += 1
        else:
            self.stats["validations_failed"] += 1

        # Guardar en cache
        if self.enable_caching:
            self.registry.validation_cache[cache_key] = result

        # Agregar a historial
        self.registry.validation_history.append(result)
        if len(self.registry.validation_history) > self.max_validation_history:
            self.registry.validation_history = self.registry.validation_history[
                -self.max_validation_history :
            ]

        # Enviar telemetría
        if self.telemetry:
            await self.telemetry.record_metric(
                "contract.validation",
                1.0,
                {
                    "contract_name": contract_name,
                    "contract_version": target_version,
                    "valid": str(is_valid),
                    "contract_type": contract.contract_type.value,
                },
            )

            await self.telemetry.record_metric(
                "contract.validation_time",
                validation_time,
                {"contract_name": contract_name, "contract_version": target_version},
            )

        logger.debug(
            f"Validated {contract_name} v{target_version}: {'PASS' if is_valid else 'FAIL'}"
        )

        return result

    async def _run_custom_validator(
        self,
        validator: Callable,
        data: Any,
        contract: Contract,
        context: Dict[str, Any],
    ) -> List[str]:
        """Ejecuta un validador personalizado"""
        if asyncio.iscoroutinefunction(validator):
            return await validator(data, contract, context)
        else:
            return validator(data, contract, context)

    def _calculate_data_hash(self, data: Any) -> str:
        """Calcula hash de los datos para caching"""
        try:
            # Convertir a JSON para normalizar
            json_str = json.dumps(data, sort_keys=True, default=str)
            return hashlib.md5(json_str.encode()).hexdigest()
        except:
            # Fallback para datos no serializables
            return hashlib.md5(str(data).encode()).hexdigest()

    async def validate_batch(
        self, validations: List[Dict[str, Any]]
    ) -> List[ValidationResult]:
        """
        Valida múltiples contratos en lote

        Args:
            validations: Lista de diccionarios con keys: contract_name, data, version (opcional)

        Returns:
            Lista de resultados de validación
        """
        tasks = []

        for validation_request in validations:
            contract_name = validation_request["contract_name"]
            data = validation_request["data"]
            version = validation_request.get("version")
            context = validation_request.get("context")

            task = asyncio.create_task(
                self.validate(contract_name, data, version, context)
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks)
        return results

    def get_validation_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas de validación"""
        total_validations = self.stats["validations_performed"]

        success_rate = 0.0
        if total_validations > 0:
            success_rate = (self.stats["validations_passed"] / total_validations) * 100

        cache_hit_rate = 0.0
        total_cache_requests = self.stats["cache_hits"] + self.stats["cache_misses"]
        if total_cache_requests > 0:
            cache_hit_rate = (self.stats["cache_hits"] / total_cache_requests) * 100

        return {
            **self.stats,
            "success_rate": success_rate,
            "cache_hit_rate": cache_hit_rate,
            "total_contracts": len(self.registry.contracts),
            "validation_history_size": len(self.registry.validation_history),
        }

    def get_contract_usage_stats(self) -> Dict[str, Dict[str, Any]]:
        """Obtiene estadísticas de uso por contrato"""
        stats = {}

        for result in self.registry.validation_history:
            contract_name = result.contract_name

            if contract_name not in stats:
                stats[contract_name] = {
                    "total_validations": 0,
                    "successful_validations": 0,
                    "failed_validations": 0,
                    "avg_validation_time": 0.0,
                    "versions_used": set(),
                    "latest_validation": None,
                }

            contract_stats = stats[contract_name]
            contract_stats["total_validations"] += 1

            if result.valid:
                contract_stats["successful_validations"] += 1
            else:
                contract_stats["failed_validations"] += 1

            contract_stats["versions_used"].add(result.contract_version)

            # Calcular tiempo promedio
            total_time = (
                contract_stats["avg_validation_time"]
                * (contract_stats["total_validations"] - 1)
                + result.validation_time
            )
            contract_stats["avg_validation_time"] = (
                total_time / contract_stats["total_validations"]
            )

            # Actualizar última validación
            if (
                contract_stats["latest_validation"] is None
                or result.timestamp > contract_stats["latest_validation"]
            ):
                contract_stats["latest_validation"] = result.timestamp

        # Convertir sets a listas para serialización
        for contract_stats in stats.values():
            contract_stats["versions_used"] = list(contract_stats["versions_used"])
            if contract_stats["latest_validation"]:
                contract_stats["latest_validation"] = contract_stats[
                    "latest_validation"
                ].isoformat()

        return stats

    def generate_compatibility_report(
        self, contract_name: str, old_version: str, new_version: str
    ) -> Dict[str, Any]:
        """Genera reporte de compatibilidad entre versiones de contrato"""

        contract = self.registry.get_contract(contract_name)
        if not contract:
            return {"error": f"Contract {contract_name} not found"}

        if old_version not in contract.versions or new_version not in contract.versions:
            return {"error": "One or both versions not found"}

        old_contract_version = contract.versions[old_version]
        new_contract_version = contract.versions[new_version]

        # Analizar cambios
        compatibility_issues = []

        # Verificar breaking changes explícitos
        breaking_changes = new_contract_version.breaking_changes
        if breaking_changes:
            compatibility_issues.extend(breaking_changes)

        # Analizar diferencias en schema
        old_schema = old_contract_version.schema
        new_schema = new_contract_version.schema

        schema_diff = self._analyze_schema_differences(old_schema, new_schema)

        return {
            "contract_name": contract_name,
            "old_version": old_version,
            "new_version": new_version,
            "compatible": len(compatibility_issues) == 0,
            "breaking_changes": breaking_changes,
            "schema_differences": schema_diff,
            "migration_guide": new_contract_version.migration_guide,
            "analysis_timestamp": datetime.now().isoformat(),
        }

    def _analyze_schema_differences(
        self, old_schema: Dict[str, Any], new_schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analiza diferencias entre dos schemas JSON"""
        differences = {
            "added_properties": [],
            "removed_properties": [],
            "type_changes": [],
            "constraint_changes": [],
        }

        # Analizar propiedades en el nivel raíz
        old_props = old_schema.get("properties", {})
        new_props = new_schema.get("properties", {})

        old_prop_names = set(old_props.keys())
        new_prop_names = set(new_props.keys())

        # Propiedades añadidas
        added_props = new_prop_names - old_prop_names
        differences["added_properties"] = list(added_props)

        # Propiedades eliminadas
        removed_props = old_prop_names - new_prop_names
        differences["removed_properties"] = list(removed_props)

        # Propiedades modificadas
        common_props = old_prop_names & new_prop_names
        for prop_name in common_props:
            old_prop = old_props[prop_name]
            new_prop = new_props[prop_name]

            # Verificar cambios de tipo
            old_type = old_prop.get("type")
            new_type = new_prop.get("type")

            if old_type != new_type:
                differences["type_changes"].append(
                    {"property": prop_name, "old_type": old_type, "new_type": new_type}
                )

        return differences

    async def load_contracts_from_file(self, file_path: str):
        """Carga contratos desde archivo YAML o JSON"""

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                if file_path.endswith(".yaml") or file_path.endswith(".yml"):
                    contracts_data = yaml.safe_load(f)
                else:
                    contracts_data = json.load(f)

            # Procesar cada contrato
            for contract_data in contracts_data.get("contracts", []):
                contract = Contract(**contract_data)
                self.registry.register_contract(contract)

            logger.info(
                f"Loaded {len(contracts_data.get('contracts', []))} contracts from {file_path}"
            )

        except Exception as e:
            logger.error(f"Failed to load contracts from {file_path}: {e}")
            raise

    async def export_contracts_to_file(self, file_path: str):
        """Exporta contratos a archivo YAML o JSON"""

        contracts_data = {
            "contracts": [
                contract.model_dump() for contract in self.registry.contracts.values()
            ]
        }

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                if file_path.endswith(".yaml") or file_path.endswith(".yml"):
                    yaml.dump(
                        contracts_data, f, default_flow_style=False, allow_unicode=True
                    )
                else:
                    json.dump(contracts_data, f, indent=2, default=str)

            logger.info(
                f"Exported {len(self.registry.contracts)} contracts to {file_path}"
            )

        except Exception as e:
            logger.error(f"Failed to export contracts to {file_path}: {e}")
            raise

    def clear_validation_cache(self):
        """Limpia el cache de validaciones"""
        self.registry.validation_cache.clear()
        logger.info("Validation cache cleared")

    def health_check(self) -> Dict[str, Any]:
        """Realiza health check del validador de contratos"""

        total_contracts = len(self.registry.contracts)
        deprecated_contracts = 0

        for contract in self.registry.contracts.values():
            current_version = contract.versions.get(contract.current_version)
            if current_version and current_version.deprecated:
                deprecated_contracts += 1

        recent_validations = [
            r
            for r in self.registry.validation_history
            if (datetime.now() - r.timestamp).total_seconds() < 300  # últimos 5 minutos
        ]

        return {
            "status": "healthy",
            "total_contracts": total_contracts,
            "deprecated_contracts": deprecated_contracts,
            "cache_enabled": self.enable_caching,
            "cache_size": len(self.registry.validation_cache),
            "validation_history_size": len(self.registry.validation_history),
            "recent_validations": len(recent_validations),
            "statistics": self.get_validation_statistics(),
        }

#!/usr/bin/env python3
"""
Validador para el archivo de rúbrica YAML.
Verifica consistencia de pesos, completitud y compatibilidad con ConjunctiveModel.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml


class RubricValidator:
    """Validador de rúbrica de evaluación PDT"""

    def __init__(self, rubric_path: str = "config/evaluation_rubric.yaml"):
        self.rubric_path = rubric_path
        self.rubric_data = None
        self.errors = []
        self.warnings = []

    def validate(self) -> Tuple[bool, List[str], List[str]]:
        """Ejecuta validación completa de la rúbrica"""
        try:
            self._load_rubric()
            self._validate_structure()
            self._validate_weights()
            self._validate_completeness()
            self._validate_conjunctive_config()
            self._validate_codes_uniqueness()

            return len(self.errors) == 0, self.errors, self.warnings

        except Exception as e:
            self.errors.append(f"Error crítico en validación: {e}")
            return False, self.errors, self.warnings

    def _load_rubric(self):
        """Carga el archivo YAML de rúbrica"""
        try:
            with open(self.rubric_path, "r", encoding="utf-8") as f:
                self.rubric_data = yaml.safe_load(f)
        except FileNotFoundError:
            raise Exception(f"Archivo de rúbrica no encontrado: {self.rubric_path}")
        except yaml.YAMLError as e:
            raise Exception(f"Error parsing YAML: {e}")

    def _validate_structure(self):
        """Valida la estructura básica del YAML"""
        required_sections = [
            "metadata",
            "thresholds",
            "dimensiones_evaluacion",
            "ponderacion_final",
            "validacion",
            "conjunctive_evaluation",
        ]

        for section in required_sections:
            if section not in self.rubric_data:
                self.errors.append(f"Sección requerida faltante: {section}")

        # Validar metadata
        if "metadata" in self.rubric_data:
            metadata = self.rubric_data["metadata"]
            required_metadata = ["version", "fecha_actualizacion", "descripcion"]
            for field in required_metadata:
                if field not in metadata:
                    self.warnings.append(f"Campo de metadata faltante: {field}")

    def _validate_weights(self):
        """Valida que todos los pesos sumen correctamente"""
        tolerancia = self.rubric_data.get("validacion", {}).get("tolerancia", 0.01)

        # 1. Validar ponderación final de dimensiones
        ponderacion_final = self.rubric_data.get("ponderacion_final", {})
        suma_dimensiones = sum(ponderacion_final.values())

        if abs(suma_dimensiones - 1.0) > tolerancia:
            self.errors.append(
                f"Suma de pesos de dimensiones = {suma_dimensiones:.4f}, debe ser 1.0 ± {tolerancia}"
            )

        # 2. Validar pesos dentro de cada dimensión
        dimensiones = self.rubric_data.get("dimensiones_evaluacion", {})

        for dim_id, dimension in dimensiones.items():
            self._validate_dimension_weights(dim_id, dimension, tolerancia)

    def _validate_dimension_weights(
        self, dim_id: str, dimension: Dict, tolerancia: float
    ):
        """Valida pesos dentro de una dimensión específica"""

        # Validar categorías si existen
        if "rubrica" in dimension and "categorias" in dimension["rubrica"]:
            categorias = dimension["rubrica"]["categorias"]
            suma_categorias = sum(cat.get("peso", 0) for cat in categorias.values())

            if abs(suma_categorias - 1.0) > tolerancia:
                self.errors.append(
                    f"Dimensión {dim_id}: suma de pesos de categorías = {suma_categorias:.4f}, debe ser 1.0"
                )

        # Validar preguntas si existen
        if "preguntas" in dimension:
            preguntas = dimension["preguntas"]
            suma_preguntas = sum(preg.get("peso", 0) for preg in preguntas.values())

            if suma_preguntas > 0 and abs(suma_preguntas - 1.0) > tolerancia:
                self.warnings.append(
                    f"Dimensión {dim_id}: suma de pesos de preguntas = {suma_preguntas:.4f}, podría necesitar ser 1.0"
                )

        # Validar subdimensiones si existen
        if "subdimensiones" in dimension:
            subdimensiones = dimension["subdimensiones"]
            suma_subdimensiones = sum(
                sub.get("peso", 0) for sub in subdimensiones.values()
            )

            if abs(suma_subdimensiones - 1.0) > tolerancia:
                self.errors.append(
                    f"Dimensión {dim_id}: suma de pesos de subdimensiones = {suma_subdimensiones:.4f}, debe ser 1.0"
                )

            # Validar criterios dentro de subdimensiones
            for sub_id, subdimension in subdimensiones.items():
                if "criterios" in subdimension:
                    criterios = subdimension["criterios"]
                    suma_criterios = sum(
                        crit.get("peso", 0) for crit in criterios.values()
                    )

                    if abs(suma_criterios - 1.0) > tolerancia:
                        self.warnings.append(
                            f"Subdimensión {dim_id}.{sub_id}: suma de pesos de criterios = {suma_criterios:.4f}"
                        )

    def _validate_completeness(self):
        """Valida que todos los elementos requeridos estén presentes"""
        dimensiones = self.rubric_data.get("dimensiones_evaluacion", {})

        for dim_id, dimension in dimensiones.items():
            # Verificar campos requeridos
            required_fields = ["nombre", "descripcion", "peso_dimension"]
            for field in required_fields:
                if field not in dimension:
                    self.errors.append(
                        f"Dimensión {dim_id}: campo requerido faltante '{field}'"
                    )

            # Verificar escala de respuesta
            if "escala_respuesta" not in dimension:
                self.warnings.append(
                    f"Dimensión {dim_id}: escala_respuesta no definida"
                )

            # Verificar rúbrica
            if "rubrica" not in dimension:
                self.errors.append(f"Dimensión {dim_id}: rúbrica faltante")
            else:
                self._validate_rubric_completeness(dim_id, dimension["rubrica"])

            # Verificar preguntas/criterios/eslabones
            content_keys = ["preguntas", "criterios", "eslabones", "subdimensiones"]
            if not any(key in dimension for key in content_keys):
                self.errors.append(
                    f"Dimensión {dim_id}: sin preguntas/criterios/eslabones definidos"
                )

    def _validate_rubric_completeness(self, dim_id: str, rubrica: Dict):
        """Valida completitud de rúbrica individual"""
        if "conversion_respuestas" not in rubrica:
            self.errors.append(
                f"Dimensión {dim_id}: conversion_respuestas faltante en rúbrica"
            )

    def _validate_conjunctive_config(self):
        """Valida configuración específica para ConjunctiveModel"""
        # Verificar thresholds
        thresholds = self.rubric_data.get("thresholds", {})
        required_thresholds = [
            "satisfied",
            "partial",
            "min_aggregate",
            "required_dimensions",
        ]

        for threshold in required_thresholds:
            if threshold not in thresholds:
                self.errors.append(f"Threshold requerido faltante: {threshold}")

        # Validar rangos de umbrales
        satisfied = thresholds.get("satisfied", 0.8)
        partial = thresholds.get("partial", 0.6)
        min_aggregate = thresholds.get("min_aggregate", 0.7)

        if satisfied <= partial:
            self.errors.append("Threshold 'satisfied' debe ser mayor que 'partial'")

        if partial < 0 or satisfied > 1.0:
            self.errors.append("Thresholds deben estar en rango [0, 1]")

        # Verificar dimensiones requeridas existen
        required_dimensions = thresholds.get("required_dimensions", [])
        available_dimensions = list(
            self.rubric_data.get("dimensiones_evaluacion", {}).keys()
        )
        available_dimensions.append("global")  # Dimensión global siempre disponible

        for req_dim in required_dimensions:
            if req_dim not in available_dimensions:
                self.errors.append(
                    f"Dimensión requerida '{req_dim}' no existe en dimensiones_evaluacion"
                )

        # Verificar configuración conjuntiva
        conjunctive = self.rubric_data.get("conjunctive_evaluation", {})
        if conjunctive.get("enable", False):
            if "requisitos_obligatorios" not in conjunctive:
                self.warnings.append(
                    "ConjunctiveModel habilitado pero sin requisitos_obligatorios definidos"
                )

    def _validate_codes_uniqueness(self):
        """Valida unicidad de códigos dentro de contextos apropiados"""
        all_codes = []

        # Recopilar todos los códigos
        dimensiones = self.rubric_data.get("dimensiones_evaluacion", {})

        for dim_id, dimension in dimensiones.items():
            if "codigo" in dimension:
                all_codes.append(("dimension", dim_id, dimension["codigo"]))

            # Códigos de preguntas
            if "preguntas" in dimension:
                for preg_id, pregunta in dimension["preguntas"].items():
                    if "codigo" in pregunta:
                        all_codes.append(
                            ("pregunta", f"{dim_id}.{preg_id}", pregunta["codigo"])
                        )

            # Códigos de subdimensiones y criterios
            if "subdimensiones" in dimension:
                for sub_id, subdimension in dimension["subdimensiones"].items():
                    if "codigo" in subdimension:
                        all_codes.append(
                            (
                                "subdimension",
                                f"{dim_id}.{sub_id}",
                                subdimension["codigo"],
                            )
                        )

                    if "criterios" in subdimension:
                        for crit_id, criterio in subdimension["criterios"].items():
                            if "codigo" in criterio:
                                all_codes.append(
                                    (
                                        "criterio",
                                        f"{dim_id}.{sub_id}.{crit_id}",
                                        criterio["codigo"],
                                    )
                                )

        # Verificar duplicados
        codes_seen = {}
        for code_type, context, code in all_codes:
            if code in codes_seen:
                self.warnings.append(
                    f"Código duplicado '{code}': {codes_seen[code]} y {context}"
                )
            else:
                codes_seen[code] = context

    def generate_report(self) -> str:
        """Genera reporte completo de validación"""
        report = []
        report.append("=" * 60)
        report.append("REPORTE DE VALIDACIÓN DE RÚBRICA PDT")
        report.append("=" * 60)
        report.append()

        if self.rubric_data:
            metadata = self.rubric_data.get("metadata", {})
            report.append(f"Versión: {metadata.get('version', 'N/A')}")
            report.append(f"Fecha: {metadata.get('fecha_actualizacion', 'N/A')}")
            report.append(f"Descripción: {metadata.get('descripcion', 'N/A')}")
            report.append()

        # Errores
        if self.errors:
            report.append("❌ ERRORES ENCONTRADOS:")
            for i, error in enumerate(self.errors, 1):
                report.append(f"  {i}. {error}")
            report.append()

        # Warnings
        if self.warnings:
            report.append("⚠️  ADVERTENCIAS:")
            for i, warning in enumerate(self.warnings, 1):
                report.append(f"  {i}. {warning}")
            report.append()

        # Estadísticas
        if self.rubric_data:
            report.append("📊 ESTADÍSTICAS:")
            dimensiones = self.rubric_data.get("dimensiones_evaluacion", {})
            report.append(f"  - Dimensiones definidas: {len(dimensiones)}")

            total_preguntas = sum(
                len(dim.get("preguntas", {})) for dim in dimensiones.values()
            )
            report.append(f"  - Total preguntas: {total_preguntas}")

            total_criterios = 0
            for dimension in dimensiones.values():
                if "subdimensiones" in dimension:
                    for subdim in dimension["subdimensiones"].values():
                        total_criterios += len(subdim.get("criterios", {}))
            report.append(f"  - Total criterios: {total_criterios}")

            # Configuración conjunctiva
            conjunctive = self.rubric_data.get("conjunctive_evaluation", {})
            report.append(
                f"  - ConjunctiveModel habilitado: {conjunctive.get('enable', False)}"
            )

            thresholds = self.rubric_data.get("thresholds", {})
            report.append(
                f"  - Threshold satisfied: {thresholds.get('satisfied', 'N/A')}"
            )
            report.append(f"  - Threshold partial: {thresholds.get('partial', 'N/A')}")
            report.append()

        # Resultado final
        report.append("=" * 60)
        if len(self.errors) == 0:
            report.append("✅ VALIDACIÓN EXITOSA")
            report.append(
                "La rúbrica cumple todos los requisitos de estructura y consistencia."
            )
        else:
            report.append("❌ VALIDACIÓN FALLIDA")
            report.append(
                f"Se encontraron {len(self.errors)} errores que deben corregirse."
            )

        if len(self.warnings) > 0:
            report.append(
                f"Se generaron {len(self.warnings)} advertencias para revisión."
            )

        report.append("=" * 60)

        return "\n".join(report)


def main():
    """Función principal para ejecutar validación"""
    validator = RubricValidator()

    print("Validando rúbrica de evaluación PDT...")
    print()

    is_valid, errors, warnings = validator.validate()

    print(validator.generate_report())

    return is_valid


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

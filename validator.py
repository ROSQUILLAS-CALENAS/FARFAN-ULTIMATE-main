"""
Validador de integridad documental para PDTs
"""

# # # from typing import List, Dict, Tuple, Any  # Module not found  # Module not found  # Module not found
# # # from enum import Enum  # Module not found  # Module not found  # Module not found
# # # from models import SectionBlock, SectionType, QualityIndicators  # Module not found  # Module not found  # Module not found


class ValidationResult(Enum):
    """Resultados de validación"""
    PASSED = "passed"
    WARNING = "warning" 
    FAILED = "failed"


class PDTValidator:
    """Validador de integridad para documentos PDT"""
    
    def __init__(self):
        # Secciones obligatorias según estándares DNP
        self.mandatory_sections = {
            SectionType.DIAGNOSTICO: {
                "required": True,
                "min_tokens": 1000,
                "description": "Diagnóstico situacional"
            },
            SectionType.PROGRAMAS: {
                "required": True,
                "min_tokens": 800,
                "description": "Programas y proyectos"
            },
            SectionType.PRESUPUESTO: {
                "required": True,
                "min_tokens": 500,
                "description": "Marco presupuestal"
            },
            SectionType.METAS: {
                "required": True,
                "min_tokens": 400,
                "description": "Metas e indicadores"
            },
            SectionType.SEGUIMIENTO: {
                "required": False,  # Recomendado pero no obligatorio
                "min_tokens": 300,
                "description": "Sistema de seguimiento"
            }
        }
        
        # Umbrales de calidad
        self.quality_thresholds = {
            "min_completeness": 0.8,  # 80% de secciones obligatorias
            "min_content_length": 1000,  # Tokens mínimos por sección
            "max_ocr_ratio": 0.4,  # Máximo 40% de contenido OCR
            "min_coherence": 0.6  # Score mínimo de coherencia
        }
        
        # Políticas de manejo de errores
        self.error_handling = {
            "missing_mandatory": "flag_and_continue",
            "insufficient_length": "warning", 
            "low_quality": "warning",
            "structural_issues": "flag_and_continue"
        }
    
    def check_mandatory_sections(self, blocks: List[SectionBlock]) -> Dict[str, Any]:
        """
        Verifica presencia de secciones obligatorias
        
        Args:
            blocks: Lista de bloques de sección detectados
        
        Returns:
            Diccionario con resultado de validación
        """
        result = {
            "validation_result": ValidationResult.PASSED,
            "missing_sections": [],
            "insufficient_sections": [],
            "present_sections": [],
            "warnings": [],
            "errors": []
        }
        
        # Mapear secciones encontradas
        found_sections = {block.section_type for block in blocks}
        
        # Verificar secciones obligatorias
        for section_type, config in self.mandatory_sections.items():
            if config["required"]:
                if section_type not in found_sections:
                    result["missing_sections"].append(section_type.value)
                    result["errors"].append(f"Sección obligatoria faltante: {config['description']}")
                    result["validation_result"] = ValidationResult.FAILED
                else:
                    result["present_sections"].append(section_type.value)
        
        # Verificar longitud de contenido
        for block in blocks:
            section_config = self.mandatory_sections.get(block.section_type)
            if section_config:
                token_count = len(block.text.split())
                min_tokens = section_config["min_tokens"]
                
                if token_count < min_tokens:
                    result["insufficient_sections"].append({
                        "section": block.section_type.value,
                        "current_tokens": token_count,
                        "required_tokens": min_tokens
                    })
                    result["warnings"].append(
                        f"Sección {block.section_type.value} tiene {token_count} tokens, "
                        f"requiere mínimo {min_tokens}"
                    )
                    if result["validation_result"] == ValidationResult.PASSED:
                        result["validation_result"] = ValidationResult.WARNING
        
        return result
    
    def calculate_quality_indicators(self, 
                                   blocks: List[SectionBlock],
                                   ocr_pages: int,
                                   total_pages: int,
                                   tables_found: int) -> QualityIndicators:
        """
        Calcula indicadores de calidad del documento
        
        Args:
            blocks: Bloques de sección
            ocr_pages: Número de páginas procesadas con OCR
            total_pages: Total de páginas del documento
            tables_found: Número de tablas encontradas
        
        Returns:
            Indicadores de calidad
        """
        # Completeness index
        mandatory_count = sum(1 for st, cfg in self.mandatory_sections.items() if cfg["required"])
        found_mandatory = sum(1 for block in blocks 
                            if block.section_type in self.mandatory_sections 
                            and self.mandatory_sections[block.section_type]["required"])
        
        completeness_index = found_mandatory / mandatory_count if mandatory_count > 0 else 0.0
        
        # OCR ratio
        ocr_ratio = ocr_pages / total_pages if total_pages > 0 else 0.0
        
        # Logical coherence hint (básico)
        logical_coherence_hint = self._calculate_coherence_hint(blocks)
        
        # Secciones presentes y faltantes
        found_sections = [block.section_type.value for block in blocks]
        mandatory_sections_list = [st.value for st, cfg in self.mandatory_sections.items() if cfg["required"]]
        missing_sections = [sec for sec in mandatory_sections_list if sec not in found_sections]
        
        return QualityIndicators(
            completeness_index=completeness_index,
            logical_coherence_hint=logical_coherence_hint,
            tables_found=tables_found,
            ocr_ratio=ocr_ratio,
            mandatory_sections_present=found_sections,
            missing_sections=missing_sections
        )
    
    def apply_error_policies(self, validation_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aplica políticas de manejo de errores
        
        Args:
            validation_result: Resultado de validación inicial
        
        Returns:
            Resultado con políticas aplicadas
        """
        processed_result = validation_result.copy()
        
        # Aplicar política para secciones faltantes
        if validation_result["missing_sections"]:
            if self.error_handling["missing_mandatory"] == "flag_and_continue":
                processed_result["continue_processing"] = True
                processed_result["flags"] = processed_result.get("flags", [])
                processed_result["flags"].append("missing_mandatory_sections")
                processed_result["confidence_reduction"] = 0.3  # Reducir confianza 30%
        
        # Aplicar política para contenido insuficiente
        if validation_result["insufficient_sections"]:
            if self.error_handling["insufficient_length"] == "warning":
                # Ya es warning, no cambiar resultado
                pass
        
        return processed_result
    
    def _calculate_coherence_hint(self, blocks: List[SectionBlock]) -> float:
        """
        Calcula hint básico de coherencia lógica
        
        Args:
            blocks: Bloques de sección
        
        Returns:
            Score de coherencia (0.0 a 1.0)
        """
        if not blocks:
            return 0.0
        
        coherence_score = 0.0
        factors = []
        
        # Factor 1: Orden lógico de secciones
        expected_order = [
            SectionType.DIAGNOSTICO,
            SectionType.PROGRAMAS,
            SectionType.PRESUPUESTO,
            SectionType.METAS,
            SectionType.SEGUIMIENTO
        ]
        
        section_positions = {}
        for i, block in enumerate(blocks):
            if block.section_type not in section_positions:
                section_positions[block.section_type] = i
        
        order_violations = 0
        for i in range(len(expected_order) - 1):
            current_type = expected_order[i]
            next_type = expected_order[i + 1]
            
            if (current_type in section_positions and 
                next_type in section_positions and
                section_positions[current_type] > section_positions[next_type]):
                order_violations += 1
        
        order_factor = max(0.0, 1.0 - (order_violations / max(1, len(expected_order) - 1)))
        factors.append(order_factor)
        
        # Factor 2: Distribución de contenido
        total_text = sum(len(block.text) for block in blocks)
        if total_text > 0:
            # Penalizar si una sección domina demasiado (>60%)
            max_section_ratio = max(len(block.text) / total_text for block in blocks)
            balance_factor = 1.0 if max_section_ratio < 0.6 else (1.0 - max_section_ratio)
            factors.append(balance_factor)
        
        # Factor 3: Confianza promedio de detección
        if blocks:
            avg_confidence = sum(block.confidence for block in blocks) / len(blocks)
            factors.append(avg_confidence)
        
        # Promedio de factores
        coherence_score = sum(factors) / len(factors) if factors else 0.0
        
        return min(1.0, max(0.0, coherence_score))
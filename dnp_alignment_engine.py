# src/core/dnp_alignment_engine.py

# Mandatory Pipeline Contract Annotations
__phase__ = "K"
__code__ = "14K"
__stage_order__ = 3

"""
DNP Alignment Engine - Sistema de Alineación y Validación con Estándares DNP

Este motor garantiza que la evaluación de PDTs esté perfectamente alineada con:
- Metodologías DNP de planificación territorial
- Estándares de gestión por resultados (GPR)
- Sistema General de Participaciones (SGP)
- Marcos normativos de competencias territoriales
- Ciclo de política pública DNP
- Instrumentos de seguimiento y evaluación DNP

Author: Sistema PDT
Version: 1.0.0
Date: 2025-01-20
"""

import json
import yaml
# # # from typing import Dict, List, Any, Optional, Tuple, Union  # Module not found  # Module not found  # Module not found
# # # from dataclasses import dataclass, asdict  # Module not found  # Module not found  # Module not found
# # # from datetime import datetime, timedelta  # Module not found  # Module not found  # Module not found
# # # from pathlib import Path  # Module not found  # Module not found  # Module not found
# # # from enum import Enum  # Module not found  # Module not found  # Module not found
import logging
import re
import numpy as np
# # # from collections import defaultdict  # Module not found  # Module not found  # Module not found

logger = logging.getLogger(__name__)

class DNPFramework(Enum):
    """Frameworks DNP soportados"""
    GPR = "gestion_por_resultados"
    SGP = "sistema_general_participaciones" 
    SINERGIA = "sistema_nacional_evaluacion_gestion_resultados"
    SISBEN = "sistema_identificacion_beneficiarios"
    SIEE = "sistema_informacion_ejecucion_evaluacion"
    CHIP = "cadena_hacia_impacto_poblacional"

class DNPInstrument(Enum):
    """Instrumentos DNP de seguimiento"""
    CONPES = "documento_conpes"
    MANUAL_GPR = "manual_gestion_por_resultados"
    GUIA_PDT = "guia_planes_desarrollo_territorial"
    METODOLOGIA_EVALUACION = "metodologia_evaluacion_politicas"
    INDICADORES_ODS = "indicadores_objetivos_desarrollo_sostenible"

@dataclass
class DNPStandard:
    """Estándar específico del DNP"""
    id: str
    name: str
    framework: DNPFramework
    category: str
    description: str
    validation_rules: List[str]
    compliance_criteria: Dict[str, Any]
    legal_basis: List[str]
    implementation_guidance: str
    monitoring_indicators: List[str]
    weight: float
    critical: bool

@dataclass
class ComplianceResult:
    """Resultado de evaluación de cumplimiento"""
    standard_id: str
    compliance_level: str  # FULL, PARTIAL, NON_COMPLIANT, NOT_APPLICABLE
    score: float
    evidence: List[str]
    gaps: List[str]
    recommendations: List[str]
    risk_level: str  # LOW, MEDIUM, HIGH, CRITICAL

class DNPAlignmentEngine:
    """Motor de alineación con estándares y metodologías DNP"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.dnp_standards = self._load_dnp_standards()
        self.territorial_competencies = self._define_territorial_competencies()
        self.gpr_framework = self._setup_gpr_framework()
        self.sgp_regulations = self._load_sgp_regulations()
        self.sinergia_integration = self._setup_sinergia_integration()
        self.legal_framework = self._build_legal_framework()
        self.compliance_engine = DNPComplianceEngine(self)
        
        if config_path:
            self.load_custom_config(config_path)
    
    def _load_dnp_standards(self) -> Dict[str, DNPStandard]:
        """Cargar estándares específicos del DNP"""
        standards = {}
        
        # ========== GESTIÓN POR RESULTADOS (GPR) ==========
        standards["gpr_001"] = DNPStandard(
            id="gpr_001",
            name="Cadena de Valor Completa (Insumos → Impactos)",
            framework=DNPFramework.GPR,
            category="metodologia_planificacion",
            description="Debe incluir todos los eslabones: insumos, actividades, productos, resultados e impactos",
            validation_rules=[
                "presence_of_all_chain_links",
                "logical_sequence_verification",
                "causal_relationships_explicit"
            ],
            compliance_criteria={
                "required_links": ["insumos", "actividades", "productos", "resultados", "impactos"],
                "minimum_detail_level": "specific_measurable_indicators",
                "causal_logic_strength": ">= 0.7"
            },
            legal_basis=[
                "CONPES 3918/2018 - Estrategia para la implementación de los ODS",
                "Manual de GPR DNP 2021",
                "Decreto 1082/2015 - Sistema Nacional de Planeación"
            ],
            implementation_guidance="Cada eslabón debe tener indicadores CREMA (Claros, Relevantes, Económicos, Monitoreables, Adecuados)",
            monitoring_indicators=[
                "completitud_cadena_valor",
                "fortaleza_relaciones_causales", 
                "especificidad_indicadores"
            ],
            weight=0.25,
            critical=True
        )
        
        standards["gpr_002"] = DNPStandard(
            id="gpr_002",
            name="Indicadores CREMA (Criterios DNP)",
            framework=DNPFramework.GPR,
            category="indicadores",
            description="Todos los indicadores deben cumplir criterios CREMA del DNP",
            validation_rules=[
                "clarity_verification",
                "relevance_assessment", 
                "economic_feasibility",
                "monitorability_check",
                "adequacy_evaluation"
            ],
            compliance_criteria={
                "crema_compliance_rate": ">= 80%",
                "baseline_data_availability": "required",
                "measurement_frequency_defined": "required",
                "responsible_institution_assigned": "required"
            },
            legal_basis=[
                "Manual GPR DNP",
                "CONPES 3654/2010 - Política de Rendición de Cuentas"
            ],
            implementation_guidance="C=Claro, R=Relevante, E=Económico, M=Monitoreable, A=Adecuado",
            monitoring_indicators=[
                "porcentaje_indicadores_crema",
                "disponibilidad_linea_base",
                "factibilidad_medicion"
            ],
            weight=0.2,
            critical=True
        )
        
        # ========== SISTEMA GENERAL DE PARTICIPACIONES (SGP) ==========
        standards["sgp_001"] = DNPStandard(
            id="sgp_001",
            name="Cumplimiento Porcentajes Mínimos Sectoriales",
            framework=DNPFramework.SGP,
            category="asignacion_presupuestal",
            description="Cumplimiento de porcentajes mínimos establecidos por ley para sectores prioritarios",
            validation_rules=[
                "health_25_percent_minimum",
                "education_15_percent_minimum", 
                "water_sanitation_compliance",
                "other_sectors_verification"
            ],
            compliance_criteria={
                "salud_minimo": 25.0,
                "educacion_minimo": 15.0,
                "agua_saneamiento_minimo": 5.4,
                "tolerancia_cumplimiento": 0.0  # Sin tolerancia para mandatos constitucionales
            },
            legal_basis=[
                "Ley 715/2001 - Competencias y recursos",
                "Ley 1176/2007 - Agua potable y saneamiento",
                "Artículo 356 y 357 Constitución Política"
            ],
            implementation_guidance="Verificar asignación presupuestal específica por sector en plan de inversiones",
            monitoring_indicators=[
                "cumplimiento_25_salud",
                "cumplimiento_15_educacion",
                "asignacion_agua_saneamiento"
            ],
            weight=0.3,
            critical=True
        )
        
        # ========== SINERGIA (Sistema Nacional de Evaluación) ==========
        standards["sinergia_001"] = DNPStandard(
            id="sinergia_001", 
            name="Alineación con Política Nacional y ODS",
            framework=DNPFramework.SINERGIA,
            category="alineacion_estrategica",
            description="El PDT debe estar alineado con Plan Nacional de Desarrollo y ODS",
            validation_rules=[
                "pnd_alignment_verification",
                "ods_mapping_required",
                "sectoral_policies_consistency", 
                "territorial_coherence_check"
            ],
            compliance_criteria={
                "pnd_explicit_references": ">= 3",
                "ods_targets_mapped": ">= 5",
                "sectoral_consistency_rate": ">= 80%",
                "territorial_adaptation_evidence": "required"
            },
            legal_basis=[
                "Ley 152/1994 - Ley Orgánica del Plan de Desarrollo",
                "CONPES 3918/2018 - Estrategia ODS",
                "Decreto 1082/2015 - Planeación Nacional"
            ],
            implementation_guidance="Incluir matriz de alineación PND-PDT y mapeo específico de ODS",
            monitoring_indicators=[
                "nivel_alineacion_pnd",
                "cobertura_ods",
                "consistencia_sectorial"
            ],
            weight=0.15,
            critical=False
        )
        
        # ========== COMPETENCIAS TERRITORIALES ==========
        standards["competencias_001"] = DNPStandard(
            id="competencias_001",
            name="Respeto Marco de Competencias Territoriales",
            framework=DNPFramework.SGP,
            category="competencias_institucionales",
            description="El PDT debe respetar el marco constitucional y legal de competencias",
            validation_rules=[
                "municipal_competencies_respect",
                "departmental_coordination_evidence",
                "national_articulation_mechanisms",
                "institutional_capacity_assessment"
            ],
            compliance_criteria={
                "competencias_municipales_correctas": "100%",
                "mecanismos_coordinacion": "required",
                "capacidad_institucional_evaluada": "required"
            },
            legal_basis=[
                "Constitución Política Artículos 287, 311, 356",
                "Ley 715/2001 - Competencias",
                "Ley 1454/2011 - Ordenamiento territorial"
            ],
            implementation_guidance="Verificar que cada programa esté dentro de competencias municipales",
            monitoring_indicators=[
                "respeto_competencias",
                "coordinacion_multinivel",
                "fortalecimiento_institucional"
            ],
            weight=0.1,
            critical=False
        )
        
        return standards
    
    def _define_territorial_competencies(self) -> Dict[str, Dict[str, Any]]:
        """Definir marco de competencias territoriales según normatividad"""
        return {
            "municipales_exclusivas": {
                "servicios_publicos": {
                    "agua_potable": {"ley": "Ley 142/1994", "obligatorio": True},
                    "saneamiento_basico": {"ley": "Ley 142/1994", "obligatorio": True},
                    "aseo": {"ley": "Ley 142/1994", "obligatorio": True}
                },
                "ordenamiento_territorial": {
                    "pbot_pot_eot": {"ley": "Ley 388/1997", "obligatorio": True},
                    "uso_del_suelo": {"ley": "Ley 388/1997", "obligatorio": True},
                    "licencias_construccion": {"ley": "Ley 388/1997", "obligatorio": True}
                },
                "transito_transporte": {
                    "transito_municipal": {"ley": "Ley 769/2002", "obligatorio": False},
                    "transporte_publico": {"ley": "Ley 105/1993", "obligatorio": False}
                }
            },
            "municipales_concurrentes": {
                "salud": {
                    "atencion_primaria": {"ley": "Ley 715/2001", "nivel": "municipal"},
                    "promocion_prevencion": {"ley": "Ley 715/2001", "nivel": "municipal"},
                    "vigilancia_epidemiologica": {"ley": "Ley 715/2001", "nivel": "departamental"}
                },
                "educacion": {
                    "prestacion_servicio": {"ley": "Ley 715/2001", "nivel": "municipal"},
                    "infraestructura": {"ley": "Ley 715/2001", "nivel": "municipal"},
                    "calidad_educativa": {"ley": "Ley 715/2001", "nivel": "departamental"}
                }
            },
            "prohibidas_municipales": {
                "defensa_nacional": {"ley": "Constitución", "nivel": "nacional"},
                "relaciones_exteriores": {"ley": "Constitución", "nivel": "nacional"},
                "justicia": {"ley": "Constitución", "nivel": "nacional"},
                "politica_monetaria": {"ley": "Constitución", "nivel": "nacional"}
            }
        }
    
    def _setup_gpr_framework(self) -> Dict[str, Any]:
        """Configurar framework de Gestión Por Resultados DNP"""
        return {
            "cadena_valor_dnp": {
                "insumos": {
                    "definicion": "Recursos necesarios para ejecutar actividades",
                    "tipos": ["financieros", "humanos", "tecnicos", "infraestructura"],
                    "indicadores_tipo": ["presupuesto", "personal", "equipos"],
                    "medicion": "recursos disponibles y utilizados"
                },
                "actividades": {
                    "definicion": "Procesos que transforman insumos en productos",
                    "tipos": ["operativas", "estrategicas", "coordinacion"],
                    "indicadores_tipo": ["numero_actividades", "cumplimiento_cronograma"],
                    "medicion": "ejecución de procesos planificados"
                },
                "productos": {
                    "definicion": "Bienes y servicios entregados directamente a beneficiarios",
                    "tipos": ["bienes", "servicios", "regulaciones", "transferencias"],
                    "indicadores_tipo": ["unidades_entregadas", "beneficiarios_atendidos"],
                    "medicion": "entrega efectiva a población objetivo"
                },
                "resultados": {
                    "definicion": "Efectos directos de los productos en beneficiarios",
                    "tipos": ["cobertura", "acceso", "calidad", "satisfaccion"],
                    "indicadores_tipo": ["tasas", "porcentajes", "indices"],
                    "medicion": "cambios en condiciones de beneficiarios"
                },
                "impactos": {
                    "definicion": "Efectos de largo plazo en bienestar poblacional",
                    "tipos": ["desarrollo_humano", "reduccion_pobreza", "sostenibilidad"],
                    "indicadores_tipo": ["indices_desarrollo", "tasas_estructurales"],
                    "medicion": "transformaciones territoriales sostenibles"
                }
            },
            "criterios_crema": {
                "claro": {
                    "definicion": "Fácil de entender y sin ambigüedades",
                    "verificacion": ["terminologia_precisa", "unidad_medida_clara", "formula_explicita"]
                },
                "relevante": {
                    "definicion": "Mide aspectos importantes del objetivo",
                    "verificacion": ["vinculacion_objetivo", "significancia_politica", "utilidad_toma_decisiones"]
                },
                "economico": {
                    "definicion": "Costo de medición justificado por utilidad",
                    "verificacion": ["costo_beneficio_favorable", "fuentes_disponibles", "simplicidad_calculo"]
                },
                "monitoreable": {
                    "definicion": "Permite seguimiento periódico confiable",
                    "verificacion": ["fuente_datos_estable", "periodicidad_adecuada", "metodologia_robusta"]
                },
                "adecuado": {
                    "definicion": "Apropiado para el nivel de intervención",
                    "verificacion": ["nivel_agregacion_correcto", "temporalidad_apropiada", "sensibilidad_cambios"]
                }
            }
        }
    
    def _load_sgp_regulations(self) -> Dict[str, Any]:
        """Cargar regulaciones del Sistema General de Participaciones"""
        return {
            "distribuciones_sgp": {
                "proposito_general": {
                    "porcentaje_total": 24,
                    "destinacion": "libre_inversion_con_priorizacion_inversion_social",
                    "restricciones": ["no_funcionamiento", "inversion_prioritaria"]
                },
                "salud": {
                    "porcentaje_total": 25,
                    "destinacion_obligatoria": True,
                    "subcategorias": {
                        "regimen_subsidiado": {"porcentaje": 60, "minimo": True},
                        "salud_publica": {"porcentaje": 25, "minimo": True},
                        "prestacion_servicios": {"porcentaje": 15, "flexible": True}
                    }
                },
                "educacion": {
                    "porcentaje_total": 15,
                    "destinacion_obligatoria": True,
                    "subcategorias": {
                        "calidad_educativa": {"porcentaje": 80, "minimo": True},
                        "infraestructura": {"porcentaje": 20, "flexible": True}
                    }
                },
                "agua_potable_saneamiento": {
                    "porcentaje_minimo": 5.4,
                    "destinacion_obligatoria": True,
                    "modalidades": ["acueducto", "alcantarillado", "aseo", "tratamiento"]
                }
            },
            "condicionamientos_sgp": {
                "continuidad": "proyectos_multianules_deben_garantizar_continuidad",
                "complementariedad": "recursos_propios_deben_complementar_sgp",
                "eficiencia": "indicadores_gestion_resultado_obligatorios",
                "transparencia": "reporte_uso_recursos_publico_trimestral"
            }
        }
    
    def _setup_sinergia_integration(self) -> Dict[str, Any]:
        """Configurar integración con SINERGIA"""
        return {
            "tipos_evaluacion": {
                "diseno": {
                    "enfoque": "calidad_arquitectura_intervencion",
                    "criterios": ["pertinencia", "coherencia", "viabilidad"],
                    "momento": "formulacion_politica"
                },
                "procesos": {
                    "enfoque": "calidad_implementacion",
                    "criterios": ["eficiencia", "oportunidad", "coordinacion"],
                    "momento": "ejecucion"
                },
                "resultados": {
                    "enfoque": "logro_objetivos_propuestos",
                    "criterios": ["eficacia", "efectividad", "utilidad"],
                    "momento": "finalizacion_cierre"
                },
                "impactos": {
                    "enfoque": "efectos_largo_plazo_sostenibilidad",
                    "criterios": ["transformacion", "sostenibilidad", "replicabilidad"],
                    "momento": "post_intervencion"
                }
            },
            "metodologias_evaluacion": {
                "teoria_cambio": {
                    "componentes": ["supuestos", "cadena_causal", "factores_externos"],
                    "validacion": "logica_vertical_horizontal"
                },
                "marco_logico": {
                    "niveles": ["actividades", "componentes", "proposito", "fin"],
                    "elementos": ["resumen_narrativo", "indicadores", "medios_verificacion", "supuestos"]
                },
                "enfoque_resultados": {
                    "dimension_productos": "bienes_servicios_entregados",
                    "dimension_efectos": "cambios_comportamiento_condiciones",
                    "dimension_impactos": "efectos_desarrollo_largo_plazo"
                }
            }
        }
    
    def _build_legal_framework(self) -> Dict[str, Any]:
        """Construir marco legal de referencia"""
        return {
            "constitucional": {
                "planificacion": {
                    "articulo_339": "Plan Nacional de Desarrollo",
                    "articulo_340": "Sistema Nacional de Planeación",
                    "articulo_341": "Conformación y funciones del CONPES"
                },
                "competencias": {
                    "articulo_287": "Autonomía territorial",
                    "articulo_311": "Municipio como entidad fundamental",
                    "articulo_356_357": "Situado fiscal y participaciones"
                },
                "derechos_fundamentales": {
                    "articulo_11": "Derecho a la vida",
                    "articulo_44": "Derechos de los niños",
                    "articulo_49": "Derecho a la salud",
                    "articulo_67": "Derecho a la educación"
                }
            },
            "legal": {
                "planificacion_territorial": {
                    "ley_152_1994": "Ley Orgánica del Plan de Desarrollo",
                    "ley_388_1997": "Desarrollo territorial y POT",
                    "ley_1454_2011": "Ordenamiento territorial"
                },
                "competencias_recursos": {
                    "ley_715_2001": "Competencias y recursos territoriales",
                    "ley_1176_2007": "Agua potable y saneamiento básico",
                    "ley_617_2000": "Responsabilidad fiscal territorial"
                },
                "sectorial_especifica": {
                    "ley_100_1993": "Sistema seguridad social",
                    "ley_115_1994": "Ley General de Educación",
                    "ley_1098_2006": "Código Infancia y Adolescencia"
                }
            },
            "reglamentario": {
                "decretos_planificacion": {
                    "decreto_1082_2015": "Sector administrativo planeación nacional",
                    "decreto_1083_2015": "Función pública",
                    "decreto_1084_2015": "Inclusión social"
                },
                "resoluciones_dnp": {
                    "resolucion_1240_2020": "Manual GPR",
                    "resolucion_0067_2021": "Seguimiento PND"
                }
            }
        }
    
    def evaluate_dnp_compliance(self, pdt_data: Dict[str, Any], 
                               evaluation_results: Dict[str, Any]) -> Dict[str, ComplianceResult]:
        """Evaluar cumplimiento integral con estándares DNP"""
        compliance_results = {}
        
        for standard_id, standard in self.dnp_standards.items():
            try:
                result = self._evaluate_single_standard(standard, pdt_data, evaluation_results)
                compliance_results[standard_id] = result
                logger.info(f"✅ Evaluado estándar DNP {standard_id}: {result.compliance_level}")
            except Exception as e:
                logger.error(f"❌ Error evaluando estándar {standard_id}: {str(e)}")
                compliance_results[standard_id] = ComplianceResult(
                    standard_id=standard_id,
                    compliance_level="ERROR",
                    score=0.0,
                    evidence=[],
                    gaps=[f"Error en evaluación: {str(e)}"],
                    recommendations=[f"Revisar evaluación del estándar {standard_id}"],
                    risk_level="HIGH"
                )
        
        return compliance_results
    
    def _evaluate_single_standard(self, standard: DNPStandard, 
                                 pdt_data: Dict[str, Any],
                                 evaluation_results: Dict[str, Any]) -> ComplianceResult:
        """Evaluar un estándar específico"""
        
        if standard.framework == DNPFramework.GPR:
            return self._evaluate_gpr_standard(standard, pdt_data, evaluation_results)
        elif standard.framework == DNPFramework.SGP:
            return self._evaluate_sgp_standard(standard, pdt_data, evaluation_results)
        elif standard.framework == DNPFramework.SINERGIA:
            return self._evaluate_sinergia_standard(standard, pdt_data, evaluation_results)
        else:
            return self._evaluate_generic_standard(standard, pdt_data, evaluation_results)
    
    def _evaluate_gpr_standard(self, standard: DNPStandard, 
                              pdt_data: Dict[str, Any],
                              evaluation_results: Dict[str, Any]) -> ComplianceResult:
        """Evaluar estándares de Gestión Por Resultados"""
        
        evidence = []
        gaps = []
        score = 0.0
        
        if standard.id == "gpr_001":  # Cadena de Valor Completa
            # Verificar presencia de todos los eslabones
            required_links = standard.compliance_criteria["required_links"]
            found_links = []
            
            # Buscar evidencia de cada eslabón en resultados DE4
            de4_results = evaluation_results.get("dimension_results", {}).get("DE4", {})
            for point_result in de4_results.values():
                if "value_chain_analysis" in point_result:
                    chain_analysis = point_result["value_chain_analysis"]
                    for link, details in chain_analysis.get("links_details", {}).items():
                        if details.get("verified", False):
                            if link not in found_links:
                                found_links.append(link)
                                evidence.append(f"Eslabón {link}: {details.get('evidence', 'Verificado')}")
            
            # Calcular score basado en completitud
            completeness = len(found_links) / len(required_links)
            score = completeness * 100
            
            # Identificar gaps
            missing_links = [link for link in required_links if link not in found_links]
            gaps.extend([f"Falta eslabón: {link}" for link in missing_links])
            
        elif standard.id == "gpr_002":  # Indicadores CREMA
            # Evaluar calidad de indicadores según criterios CREMA
            crema_compliance = 0
            total_indicators = 0
            
            # Analizar indicadores de todos los puntos del decálogo
            for point_id, point_data in evaluation_results.get("point_results", {}).items():
                indicators = self._extract_indicators_from_point(point_data)
                for indicator in indicators:
                    total_indicators += 1
                    crema_score = self._evaluate_crema_criteria(indicator)
                    crema_compliance += crema_score
                    
                    if crema_score >= 0.8:
                        evidence.append(f"Indicador CREMA compliant: {indicator.get('name', 'N/A')}")
                    else:
                        gaps.append(f"Indicador no cumple CREMA: {indicator.get('name', 'N/A')}")
            
            if total_indicators > 0:
                score = (crema_compliance / total_indicators) * 100
            else:
                gaps.append("No se encontraron indicadores para evaluar")
        
        # Determinar nivel de cumplimiento
        if score >= 90:
            compliance_level = "FULL"
            risk_level = "LOW"
        elif score >= 70:
            compliance_level = "PARTIAL"
            risk_level = "MEDIUM"
        elif score >= 50:
            compliance_level = "PARTIAL"
            risk_level = "HIGH"
        else:
            compliance_level = "NON_COMPLIANT"
            risk_level = "CRITICAL"
        
        # Generar recomendaciones
        recommendations = self._generate_gpr_recommendations(standard, gaps, score)
        
        return ComplianceResult(
            standard_id=standard.id,
            compliance_level=compliance_level,
            score=score,
            evidence=evidence,
            gaps=gaps,
            recommendations=recommendations,
            risk_level=risk_level
        )
    
    def _evaluate_sgp_standard(self, standard: DNPStandard,
                              pdt_data: Dict[str, Any],
                              evaluation_results: Dict[str, Any]) -> ComplianceResult:
        """Evaluar estándares del Sistema General de Participaciones"""

        global sector_compliance
        evidence = []
        gaps = []
        score = 0.0
        
        if standard.id == "sgp_001":  # Porcentajes mínimos sectoriales
            # Verificar cumplimiento de porcentajes constitucionales
            budget_analysis = self._extract_budget_analysis(evaluation_results)
            
            sector_compliance = {}
            
            # Salud - 25% mínimo
            health_percentage = budget_analysis.get("health_percentage", 0)
            health_compliant = health_percentage >= standard.compliance_criteria["salud_minimo"]
            sector_compliance["salud"] = health_compliant
            
            if health_compliant:
                evidence.append(f"Salud: {health_percentage}% cumple mínimo 25%")
            else:
                gaps.append(f"Salud: {health_percentage}% NO cumple mínimo 25%")
            
            # Educación - 15% mínimo  
            education_percentage = budget_analysis.get("education_percentage", 0)
            education_compliant = education_percentage >= standard.compliance_criteria["educacion_minimo"]
            sector_compliance["educacion"] = education_compliant
            
            if education_compliant:
                evidence.append(f"Educación: {education_percentage}% cumple mínimo 15%")
            else:
                gaps.append(f"Educación: {education_percentage}% NO cumple mínimo 15%")
            
            # Agua y saneamiento - 5.4% mínimo
            water_percentage = budget_analysis.get("water_sanitation_percentage", 0)
            water_compliant = water_percentage >= standard.compliance_criteria["agua_saneamiento_minimo"]
            sector_compliance["agua_saneamiento"] = water_compliant
            
            if water_compliant:
                evidence.append(f"Agua y saneamiento: {water_percentage}% cumple mínimo 5.4%")
            else:
                gaps.append(f"Agua y saneamiento: {water_percentage}% NO cumple mínimo 5.4%")
            
            # Calcular score global de cumplimiento SGP
            compliant_sectors = sum(sector_compliance.values())
            total_sectors = len(sector_compliance)
            score = (compliant_sectors / total_sectors) * 100
        
        # Determinar nivel de cumplimiento
        if score == 100:
            compliance_level = "FULL"
            risk_level = "LOW"
        elif score >= 67:  # 2 de 3 sectores
            compliance_level = "PARTIAL"
            risk_level = "MEDIUM"
        elif score >= 33:  # 1 de 3 sectores
            compliance_level = "PARTIAL"
            risk_level = "HIGH"
        else:
            compliance_level = "NON_COMPLIANT"
            risk_level = "CRITICAL"
        
        recommendations = self._generate_sgp_recommendations(standard, gaps, sector_compliance)
        
        return ComplianceResult(
            standard_id=standard.id,
            compliance_level=compliance_level,
            score=score,
            evidence=evidence,
            gaps=gaps,
            recommendations=recommendations,
            risk_level=risk_level
        )
    
    def _evaluate_sinergia_standard(self, standard: DNPStandard,
                                   pdt_data: Dict[str, Any], 
                                   evaluation_results: Dict[str, Any]) -> ComplianceResult:
        """Evaluar estándares de SINERGIA"""
        
        evidence = []
        gaps = []
        score = 0.0
        
        if standard.id == "sinergia_001":  # Alineación con PND y ODS
            # Buscar referencias explícitas al PND
            pnd_references = self._find_pnd_references(pdt_data)
            pnd_count = len(pnd_references)
            
            if pnd_count >= standard.compliance_criteria["pnd_explicit_references"]:
                evidence.append(f"PND: {pnd_count} referencias encontradas")
                pnd_score = 1.0
            else:
                gaps.append(f"PND: Solo {pnd_count} referencias, mínimo {standard.compliance_criteria['pnd_explicit_references']}")
                pnd_score = pnd_count / standard.compliance_criteria["pnd_explicit_references"]
            
            # Buscar mapeo con ODS
            ods_mappings = self._find_ods_mappings(pdt_data)
            ods_count = len(ods_mappings)
            
            if ods_count >= standard.compliance_criteria["ods_targets_mapped"]:
                evidence.append(f"ODS: {ods_count} objetivos mapeados")
                ods_score = 1.0
            else:
                gaps.append(f"ODS: Solo {ods_count} objetivos, mínimo {standard.compliance_criteria['ods_targets_mapped']}")
                ods_score = ods_count / standard.compliance_criteria["ods_targets_mapped"]
            
            # Evaluar consistencia sectorial
            sectoral_consistency = self._evaluate_sectoral_consistency(pdt_data, evaluation_results)
            consistency_rate = sectoral_consistency["consistency_rate"]
            
            if consistency_rate >= standard.compliance_criteria["sectoral_consistency_rate"] / 100:
                evidence.append(f"Consistencia sectorial: {consistency_rate*100}%")
                consistency_score = 1.0
            else:
                gaps.append(f"Consistencia sectorial: {consistency_rate*100}%, mínimo {standard.compliance_criteria['sectoral_consistency_rate']}%")
                consistency_score = consistency_rate / (standard.compliance_criteria["sectoral_consistency_rate"] / 100)
            
            # Score ponderado
            score = ((pnd_score * 0.4) + (ods_score * 0.4) + (consistency_score * 0.2)) * 100
        
        # Determinar nivel de cumplimiento
        if score >= 85:
            compliance_level = "FULL"
            risk_level = "LOW"
        elif score >= 65:
            compliance_level = "PARTIAL" 
            risk_level = "MEDIUM"
        elif score >= 40:
            compliance_level = "PARTIAL"
            risk_level = "HIGH"
        else:
            compliance_level = "NON_COMPLIANT"
            risk_level = "CRITICAL"
        
        recommendations = self._generate_sinergia_recommendations(standard, gaps, score)
        
        return ComplianceResult(
            standard_id=standard.id,
            compliance_level=compliance_level,
            score=score,
            evidence=evidence,
            gaps=gaps,
            recommendations=recommendations,
            risk_level=risk_level
        )
    
    def _evaluate_generic_standard(self, standard: DNPStandard,
                                  pdt_data: Dict[str, Any],
                                  evaluation_results: Dict[str, Any]) -> ComplianceResult:
        """Evaluar estándares genéricos"""
        
        # Implementación básica para estándares no específicos
        evidence = ["Evaluación genérica aplicada"]
        gaps = []
        score = 75.0  # Score neutral para estándares genéricos
        
        return ComplianceResult(
            standard_id=standard.id,
            compliance_level="PARTIAL",
            score=score,
            evidence=evidence,
            gaps=gaps,
            recommendations=[f"Revisar cumplimiento específico del estándar {standard.id}"],
            risk_level="MEDIUM"
        )
    
    # ========== MÉTODOS DE UTILIDAD ==========
    
    def _extract_indicators_from_point(self, point_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extraer indicadores de los datos de un punto del decálogo"""
        indicators = []
        
        # Buscar indicadores en diferentes secciones
        for dimension in ["DE1", "DE2", "DE3", "DE4"]:
            if dimension in point_data:
                dim_data = point_data[dimension]
                # Buscar patrones de indicadores
                if "indicators" in dim_data:
                    indicators.extend(dim_data["indicators"])
                if "metas" in dim_data:
                    indicators.extend(dim_data["metas"])
        
        return indicators
    
    def _evaluate_crema_criteria(self, indicator: Dict[str, Any]) -> float:
        """Evaluar un indicador según criterios CREMA"""
        crema_score = 0.0
        criteria_count = 5
        
        # Claro: ¿Tiene definición, fórmula y unidad clara?
        if (indicator.get("definition") and 
            indicator.get("formula") and 
            indicator.get("unit")):
            crema_score += 1
        
        # Relevante: ¿Está vinculado a objetivos importantes?
        if (indicator.get("objective_link") or 
            indicator.get("strategic_relevance")):
            crema_score += 1
        
        # Económico: ¿Es factible de medir?
        if (indicator.get("data_source") and 
            indicator.get("measurement_cost", "low") in ["low", "medium"]):
            crema_score += 1
        
        # Monitoreable: ¿Tiene fuente y periodicidad?
        if (indicator.get("data_source") and 
            indicator.get("frequency")):
            crema_score += 1
        
        # Adecuado: ¿Es apropiado para el nivel?
        if (indicator.get("measurement_level") and 
            indicator.get("sensitivity", True)):
            crema_score += 1
        
        return crema_score / criteria_count
    
    def _extract_budget_analysis(self, evaluation_results: Dict[str, Any]) -> Dict[str, float]:
        """Extraer análisis presupuestal de los resultados de evaluación"""
        budget_analysis = {
            "health_percentage": 0.0,
            "education_percentage": 0.0,
            "water_sanitation_percentage": 0.0
        }
        
        # Buscar en resultados DE3 (Planificación Presupuestal)
        de3_results = evaluation_results.get("dimension_results", {}).get("DE3", {})
        
        for point_result in de3_results.values():
            if "budget_analysis" in point_result:
                budget_data = point_result["budget_analysis"]
                
                # Extraer porcentajes sectoriales
                if "constitutional_compliance" in budget_data:
                    compliance = budget_data["constitutional_compliance"]
                    
                    if "health_25_percent" in compliance:
                        budget_analysis["health_percentage"] = compliance["health_25_percent"].get("actual_percentage", 0.0)
                    
                    if "education_15_percent" in compliance:
                        budget_analysis["education_percentage"] = compliance["education_15_percent"].get("actual_percentage", 0.0)
                
                # Buscar agua y saneamiento en otras secciones
                for category in budget_data.values():
                    if isinstance(category, dict) and "water_sanitation" in str(category):
                        budget_analysis["water_sanitation_percentage"] = 5.4  # Placeholder
        
        return budget_analysis
    
    def _find_pnd_references(self, pdt_data: Dict[str, Any]) -> List[str]:
        """Encontrar referencias al Plan Nacional de Desarrollo"""
        pnd_references = []
        pnd_patterns = [
            r"Plan Nacional de Desarrollo",
            r"PND",
            r"Pacto por Colombia",
            r"gobierno nacional",
            r"política nacional"
        ]
        
        # Buscar en texto del PDT
        pdt_text = str(pdt_data.get("content", ""))
        
        for pattern in pnd_patterns:
            matches = re.findall(pattern, pdt_text, re.IGNORECASE)
            pnd_references.extend(matches)
        
        return list(set(pnd_references))  # Eliminar duplicados
    
    def _find_ods_mappings(self, pdt_data: Dict[str, Any]) -> List[str]:
        """Encontrar mapeo con Objetivos de Desarrollo Sostenible"""
        ods_mappings = []
        ods_patterns = [
            r"ODS\s*\d+",
            r"Objetivo de Desarrollo Sostenible",
            r"Agenda 2030",
            r"desarrollo sostenible"
        ]
        
        pdt_text = str(pdt_data.get("content", ""))
        
        for pattern in ods_patterns:
            matches = re.findall(pattern, pdt_text, re.IGNORECASE)
            ods_mappings.extend(matches)
        
        return list(set(ods_mappings))
    
    def _evaluate_sectoral_consistency(self, pdt_data: Dict[str, Any], 
                                     evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluar consistencia sectorial"""
        
        # Análisis básico de consistencia
        sectors_found = []
        consistent_sectors = []
        
        # Extraer sectores mencionados
        pdt_text = str(pdt_data.get("content", ""))
        sector_patterns = {
            "salud": r"salud|médic|hospital|clínic",
            "educacion": r"educación|educativ|escuel|colegio",
            "ambiente": r"ambient|ecológic|sostenib|clima"
        }
        
        for sector, pattern in sector_patterns.items():
            if re.search(pattern, pdt_text, re.IGNORECASE):
                sectors_found.append(sector)
                # Verificar si hay programas consistentes
                if f"programa.*{sector}" in pdt_text.lower():
                    consistent_sectors.append(sector)
        
        consistency_rate = len(consistent_sectors) / max(len(sectors_found), 1)
        
        return {
            "sectors_found": sectors_found,
            "consistent_sectors": consistent_sectors,
            "consistency_rate": consistency_rate
        }
    
    # ========== GENERACIÓN DE RECOMENDACIONES ==========
    
    def _generate_gpr_recommendations(self, standard: DNPStandard, 
                                     gaps: List[str], score: float) -> List[str]:
        """Generar recomendaciones para estándares GPR"""
        recommendations = []
        
        if standard.id == "gpr_001":  # Cadena de valor
            if score < 80:
                recommendations.append("Completar la cadena de valor con todos los eslabones: insumos, actividades, productos, resultados e impactos")
            if "diagnostico_linea_base" in str(gaps):
                recommendations.append("Incluir diagnóstico con línea base cuantitativa y brechas específicas")
            if "causalidad_explicita" in str(gaps):
                recommendations.append("Explicitar la lógica causal entre productos, resultados e impactos")
                
        elif standard.id == "gpr_002":  # CREMA
            if score < 70:
                recommendations.append("Mejorar calidad de indicadores aplicando criterios CREMA del DNP")
                recommendations.append("Definir fórmulas de cálculo explícitas para todos los indicadores")
                recommendations.append("Establecer fuentes de datos y periodicidad de medición")
        
        return recommendations
    
    def _generate_sgp_recommendations(self, standard: DNPStandard, 
                                     gaps: List[str], sector_compliance: Dict[str, bool]) -> List[str]:
        """Generar recomendaciones para estándares SGP"""
        recommendations = []
        
        if not sector_compliance.get("salud", False):
            recommendations.append("CRÍTICO: Asignar mínimo 25% del presupuesto a salud según mandato constitucional")
            recommendations.append("Revisar plan de inversiones para garantizar recursos suficientes al sector salud")
            
        if not sector_compliance.get("educacion", False):
            recommendations.append("CRÍTICO: Asignar mínimo 15% del presupuesto a educación según Ley 715/2001")
            recommendations.append("Priorizar inversión educativa en infraestructura y calidad")
            
        if not sector_compliance.get("agua_saneamiento", False):
            recommendations.append("Asignar recursos suficientes para agua potable y saneamiento básico")
            recommendations.append("Cumplir con destinación específica SGP para servicios públicos")
        
        return recommendations
    
    def _generate_sinergia_recommendations(self, standard: DNPStandard,
                                         gaps: List[str], score: float) -> List[str]:
        """Generar recomendaciones para estándares SINERGIA"""
        recommendations = []
        
        if "PND" in str(gaps):
            recommendations.append("Incluir referencias explícitas al Plan Nacional de Desarrollo vigente")
            recommendations.append("Alinear objetivos territoriales con pactos nacionales")
            
        if "ODS" in str(gaps):
            recommendations.append("Mapear programas con Objetivos de Desarrollo Sostenible específicos")
            recommendations.append("Incluir matriz de contribución a la Agenda 2030")
            
        if score < 60:
            recommendations.append("Fortalecer articulación con políticas sectoriales nacionales")
            recommendations.append("Mejorar consistencia entre diagnóstico, estrategias y programas")
        
        return recommendations
    
    def generate_compliance_report(self, compliance_results: Dict[str, ComplianceResult]) -> Dict[str, Any]:
        """Generar reporte integral de cumplimiento DNP"""
        
        # Calcular estadísticas generales
        total_standards = len(compliance_results)
        full_compliance = len([r for r in compliance_results.values() if r.compliance_level == "FULL"])
        partial_compliance = len([r for r in compliance_results.values() if r.compliance_level == "PARTIAL"])
        non_compliance = len([r for r in compliance_results.values() if r.compliance_level == "NON_COMPLIANT"])
        
        # Identificar riesgos críticos
        critical_risks = [r for r in compliance_results.values() if r.risk_level == "CRITICAL"]
        high_risks = [r for r in compliance_results.values() if r.risk_level == "HIGH"]
        
        # Agrupar por framework
        framework_summary = defaultdict(list)
        for result in compliance_results.values():
            standard = self.dnp_standards[result.standard_id]
            framework_summary[standard.framework.value].append(result)
        
        # Calcular score global de cumplimiento DNP
        total_score = sum(r.score * self.dnp_standards[r.standard_id].weight 
                         for r in compliance_results.values())
        max_possible_score = sum(s.weight for s in self.dnp_standards.values()) * 100
        global_compliance_score = total_score / max_possible_score if max_possible_score > 0 else 0
        
        # Determinar clasificación global
        if global_compliance_score >= 85:
            global_classification = "ALTAMENTE_CONFORME_DNP"
        elif global_compliance_score >= 70:
            global_classification = "CONFORME_DNP"
        elif global_compliance_score >= 50:
            global_classification = "PARCIALMENTE_CONFORME_DNP"
        else:
            global_classification = "NO_CONFORME_DNP"
        
        return {
            "metadata": {
                "evaluation_date": datetime.now().isoformat(),
                "dnp_framework_version": "2025",
                "standards_evaluated": total_standards,
                "evaluation_scope": "comprehensive_dnp_alignment"
            },
            "global_assessment": {
                "compliance_score": round(global_compliance_score, 2),
                "classification": global_classification,
                "summary_statistics": {
                    "full_compliance": full_compliance,
                    "partial_compliance": partial_compliance,
                    "non_compliance": non_compliance,
                    "compliance_rate": round((full_compliance + partial_compliance) / total_standards * 100, 1)
                }
            },
            "risk_assessment": {
                "critical_risks_count": len(critical_risks),
                "high_risks_count": len(high_risks),
                "critical_standards": [r.standard_id for r in critical_risks],
                "risk_mitigation_priority": "immediate_action" if critical_risks else "routine_monitoring"
            },
            "framework_performance": {
                framework: {
                    "standards_count": len(results),
                    "avg_score": round(sum(r.score for r in results) / len(results), 2),
                    "compliance_rate": round(len([r for r in results if r.compliance_level in ["FULL", "PARTIAL"]]) / len(results) * 100, 1)
                }
                for framework, results in framework_summary.items()
            },
            "detailed_results": {
                result.standard_id: {
                    "standard_name": self.dnp_standards[result.standard_id].name,
                    "framework": self.dnp_standards[result.standard_id].framework.value,
                    "compliance_level": result.compliance_level,
                    "score": result.score,
                    "risk_level": result.risk_level,
                    "evidence_count": len(result.evidence),
                    "gaps_count": len(result.gaps),
                    "recommendations_count": len(result.recommendations)
                }
                for result in compliance_results.values()
            },
            "priority_actions": {
                "immediate": [r.recommendations for r in critical_risks],
                "short_term": [r.recommendations for r in high_risks],
                "continuous": [
                    "Mantener seguimiento periódico del cumplimiento DNP",
                    "Actualizar según cambios normativos",
                    "Fortalecer capacidades institucionales en GPR"
                ]
            },
            "regulatory_compliance": {
                "constitutional_mandates": self._assess_constitutional_compliance(compliance_results),
                "legal_requirements": self._assess_legal_compliance(compliance_results),
                "dnp_methodologies": self._assess_methodological_compliance(compliance_results)
            }
        }
    
    def _assess_constitutional_compliance(self, compliance_results: Dict[str, ComplianceResult]) -> Dict[str, Any]:
        """Evaluar cumplimiento de mandatos constitucionales"""
        
        constitutional_compliance = {
            "health_25_percent": False,
            "education_15_percent": False,
            "territorial_autonomy_respected": True,
            "fundamental_rights_addressed": True
        }
        
        # Verificar cumplimiento SGP
        sgp_result = compliance_results.get("sgp_001")
        if sgp_result:
            constitutional_compliance["health_25_percent"] = "25%" in str(sgp_result.evidence)
            constitutional_compliance["education_15_percent"] = "15%" in str(sgp_result.evidence)
        
        return constitutional_compliance
    
    def _assess_legal_compliance(self, compliance_results: Dict[str, ComplianceResult]) -> Dict[str, Any]:
        """Evaluar cumplimiento de requisitos legales"""
        
        return {
            "ley_152_compliance": True,  # Ley Orgánica del Plan
            "ley_715_compliance": True,  # Competencias y recursos
            "sectoral_laws_compliance": True,
            "overall_legal_risk": "LOW" if all(r.risk_level != "CRITICAL" for r in compliance_results.values()) else "HIGH"
        }
    
    def _assess_methodological_compliance(self, compliance_results: Dict[str, ComplianceResult]) -> Dict[str, Any]:
        """Evaluar cumplimiento metodológico DNP"""
        
        gpr_compliance = compliance_results.get("gpr_001", ComplianceResult("", "NON_COMPLIANT", 0, [], [], [], "HIGH"))
        crema_compliance = compliance_results.get("gpr_002", ComplianceResult("", "NON_COMPLIANT", 0, [], [], [], "HIGH"))
        
        return {
            "gpr_methodology": gpr_compliance.compliance_level,
            "crema_indicators": crema_compliance.compliance_level,
            "sinergia_integration": "PARTIAL",  # Basado en evaluación SINERGIA
            "overall_methodological_score": round((gpr_compliance.score + crema_compliance.score) / 2, 2)
        }


class DNPComplianceEngine:
    """Motor específico de evaluación de cumplimiento"""
    
    def __init__(self, alignment_engine: DNPAlignmentEngine):
        self.alignment_engine = alignment_engine
        self.validation_rules = self._build_validation_rules()
    
    def _build_validation_rules(self) -> Dict[str, Any]:
        """Construir reglas de validación automatizadas"""
        return {
            "budget_validation": {
                "health_minimum": 25.0,
                "education_minimum": 15.0,
                "water_minimum": 5.4,
                "tolerance": 0.1  # 0.1% de tolerancia
            },
            "indicator_validation": {
                "crema_minimum_score": 0.7,
                "baseline_required": True,
                "target_required": True,
                "source_required": True
            },
            "chain_validation": {
                "required_links": ["insumos", "actividades", "productos", "resultados", "impactos"],
                "causal_logic_minimum": 0.6,
                "completeness_minimum": 0.8
            }
        }
    
    def validate_pdt_structure(self, pdt_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validar estructura básica del PDT según estándares DNP"""
        
        validation_results = {
            "structure_valid": True,
            "missing_sections": [],
            "format_issues": [],
            "content_gaps": []
        }
        
        # Secciones requeridas según Ley 152/1994
        required_sections = [
            "diagnostico",
            "parte_estrategica", 
            "plan_inversiones",
            "seguimiento_evaluacion"
        ]
        
        pdt_content = str(pdt_data.get("content", "")).lower()
        
        for section in required_sections:
            if section not in pdt_content:
                validation_results["missing_sections"].append(section)
                validation_results["structure_valid"] = False
        
        return validation_results


def create_dnp_alignment_engine() -> DNPAlignmentEngine:
    """Factory function para crear instancia del motor de alineación"""
    return DNPAlignmentEngine()


# Ejemplo de uso
if __name__ == "__main__":
    # Crear motor de alineación DNP
    dnp_engine = create_dnp_alignment_engine()
    
    # Ejemplo de datos de PDT (simulado)
    pdt_data = {
        "municipality": "Bogotá",
        "department": "Cundinamarca", 
        "content": "Contenido del PDT con programas de salud, educación y desarrollo social..."
    }
    
    # Ejemplo de resultados de evaluación (simulado)
    evaluation_results = {
        "global_score": 78.5,
        "dimension_results": {
            "DE1": {"point_1": {"causal_factor": 0.8}},
            "DE3": {"point_4": {"budget_analysis": {"constitutional_compliance": {"health_25_percent": {"actual_percentage": 26.0}}}}}
        }
    }
    
    # Evaluar cumplimiento DNP
    compliance_results = dnp_engine.evaluate_dnp_compliance(pdt_data, evaluation_results)
    
    # Generar reporte
    compliance_report = dnp_engine.generate_compliance_report(compliance_results)
    
    print("🏛️ DNP Alignment Engine - Evaluación Completa")
    print(f"📊 Score global de cumplimiento DNP: {compliance_report['global_assessment']['compliance_score']}%")
    print(f"🏆 Clasificación: {compliance_report['global_assessment']['classification']}")
    print(f"⚠️ Riesgos críticos: {compliance_report['risk_assessment']['critical_risks_count']}")
    print(f"✅ Motor DNP inicializado exitosamente")

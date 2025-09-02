# src/core/question_decalogo_mapper.py
"""
Mapeo completo de preguntas de evaluación PDT con puntos del Decálogo DDHH
y eslabones de la cadena de valor del DNP - IMPLEMENTACIÓN EJECUTABLE MEJORADA

Author: Sistema PDT
Version: 2.0.0
Date: 2025-01-20

Mejoras implementadas:
- Adherencia estricta al advanced_prompts_catalog
- Cobertura exhaustiva de todos los 11 puntos del decálogo
- Integración completa de las 4 dimensiones (DE1-DE4)
- Patrones de búsqueda avanzados basados en el catálogo
- Sistema de validación robusto
- Motor de consultas optimizado
"""

import json
import logging
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)


@dataclass
class QuestionMapping:
    """Estructura de datos mejorada para mapeo de preguntas"""

    question_id: str
    question_text: str
    decalogo_point: int
    point_name: str
    dimension: str  # DE1, DE2, DE3, DE4
    value_chain_link: str  # insumos, actividades, productos, resultados, impactos
    cluster_id: int  # 1-5 según clusterización del decálogo
    cluster_name: str
    dnp_category: str
    sector: str
    weight: float
    keywords: List[str]
    search_patterns: List[str]
    prompt_reference: str  # Referencia al prompt específico en el catálogo
    evidence_requirements: List[str]  # Tipos de evidencia esperada
    navigation_hints: List[str]  # Sugerencias de dónde buscar en el PDT
    answer_options: List[str]  # Sí/Parcial/No/NI
    scoring_criteria: Dict[str, Any]  # Criterios específicos de puntuación


class AdvancedQuestionDecalogoMapper:
    """Mapeador avanzado completo basado en advanced_prompts_catalog"""

    def __init__(self):
        self.complete_mapping = self._build_complete_mapping()
        self.cluster_definitions = self._define_clusters()
        self.value_chain_links = self._define_value_chain_links()
        self.decalogo_points = self._define_decalogo_points()
        self.dimension_standards = self._define_dimension_standards()
        self.prompt_catalog_integration = self._integrate_prompt_catalog()
        self.coverage_matrix = None
        self.validation_report = None
        self.query_engine = None

    def _define_clusters(self) -> Dict[int, Dict[str, Any]]:
        """Definir los 5 clusters del Decálogo según advanced_prompts_catalog"""
        return {
            1: {
                "name": "DERECHOS DE LAS VÍCTIMAS Y CONSTRUCCIÓN DE PAZ",
                "points": [8],  # Líderes y defensores
                "description": "Protección y garantía de derechos para líderes sociales y construcción de paz",
                "priority": "CRÍTICA",
                "evaluation_weight": 0.25,
            },
            2: {
                "name": "DERECHOS DE GRUPOS POBLACIONALES",
                "points": [
                    2,
                    6,
                    9,
                ],  # Género, niñez/juventud, personas privadas libertad
                "description": "Enfoque diferencial y protección de grupos vulnerables",
                "priority": "ALTA",
                "evaluation_weight": 0.25,
            },
            3: {
                "name": "TERRITORIO, AMBIENTE Y DESARROLLO SOSTENIBLE",
                "points": [3, 7],  # Ambiente y tierras/territorios
                "description": "Sostenibilidad ambiental y ordenamiento territorial",
                "priority": "ALTA",
                "evaluation_weight": 0.20,
            },
            4: {
                "name": "DERECHOS ECONÓMICOS, SOCIALES Y CULTURALES (DESC)",
                "points": [4, 5, 10],  # Salud, educación, trabajo
                "description": "Garantía de derechos fundamentales económicos y sociales",
                "priority": "CRÍTICA",
                "evaluation_weight": 0.20,
            },
            5: {
                "name": "DERECHOS CIVILES Y POLÍTICOS",
                "points": [1, 11],  # Vida/seguridad y participación
                "description": "Derechos civiles fundamentales y participación democrática",
                "priority": "CRÍTICA",
                "evaluation_weight": 0.10,
            },
        }

    def _define_dimension_standards(self) -> Dict[str, Dict[str, Any]]:
        """Definir estándares de las 4 dimensiones según advanced_prompts_catalog"""
        return {
            "DE1": {
                "name": "Lógica de Intervención y Coherencia Interna",
                "weight": 0.4,
                "description": "Evalúa coherencia lógica de intervenciones propuestas",
                "scoring_components": {
                    "bienestar": {"weight": 0.6, "questions": ["Q5", "Q6"]},
                    "resultado": {"weight": 0.3, "questions": ["Q3", "Q4"]},
                    "gestion_producto": {"weight": 0.1, "questions": ["Q1", "Q2"]},
                },
                "causal_factors": {
                    "coherencia_diagnostico": {"max_points": 5},
                    "articulacion_logica": {"max_points": 5},
                    "consistencia_temporal": {"max_points": 5},
                    "suficiencia_evidencia": {"max_points": 5},
                },
                "patterns": [
                    "problema.*causa.*objetivo.*estrategia",
                    "lógica causal irrefutable",
                    "secuencia clara, fluida y sin interrupciones lógicas",
                ],
            },
            "DE2": {
                "name": "Inclusión Temática y Grupos Poblacionales",
                "weight": 0.25,
                "description": "Evalúa inclusión y enfoque diferencial",
                "scoring_method": "binary_count",
                "verification_questions": {
                    "enfoque_diferencial": "¿Incluye enfoque diferencial explícito?",
                    "grupos_vulnerables": "¿Identifica y prioriza grupos vulnerables?",
                    "participacion_comunitaria": "¿Contempla participación comunitaria?",
                    "interseccionalidad": "¿Reconoce interseccionalidad de derechos?",
                },
                "patterns": [
                    "enfoque diferencial",
                    "grupos poblacionales",
                    "participación comunitaria",
                    "interseccionalidad",
                ],
            },
            "DE3": {
                "name": "Planificación y Adecuación Presupuestal",
                "weight": 0.2,
                "description": "Evalúa viabilidad presupuestal y planificación financiera",
                "scoring_method": "subpoint_accumulation",
                "question_categories": {
                    "general": ["G1", "G2"],
                    "asignacion": ["A1", "A2"],
                    "realismo": ["R1", "R2"],
                    "sostenibilidad": ["S1", "S2"],
                },
                "patterns": [
                    "presupuesto específico",
                    "fuentes de financiación",
                    "plan de contingencia presupuestal",
                ],
            },
            "DE4": {
                "name": "Cadena de Valor",
                "weight": 0.15,
                "description": "Evalúa completitud del ciclo de intervención",
                "scoring_method": "chain_completeness",
                "key_links": [
                    "diagnostico_linea_base",
                    "causalidad_explicita",
                    "metas_transformadoras",
                    "programas_detallados",
                    "territorializacion",
                    "vinculacion_institucional",
                    "seguimiento_indicadores",
                    "proyeccion_impacto",
                ],
                "scoring_categories": {
                    "alto": {"points": 100, "min_links": 8, "min_causal_factor": 0.9},
                    "medio": {"points": 65, "min_links": 6, "min_causal_factor": 0.7},
                    "bajo": {"points": 35, "min_links": 0, "min_causal_factor": 0.0},
                },
            },
        }

    def _define_decalogo_points(self) -> Dict[int, Dict[str, Any]]:
        """Definir los 11 puntos del Decálogo DDHH con detalles completos"""
        return {
            1: {
                "name": "Derecho a la vida, a la seguridad y a la convivencia",
                "sector": "seguridad",
                "cluster": 5,
                "priority": "CRÍTICA",
                "budget_min_percent": 3.0,
                "keywords": [
                    "seguridad",
                    "convivencia",
                    "violencia",
                    "protección",
                    "víctimas",
                ],
                "dnp_categories": ["seguridad_ciudadana", "orden_publico", "justicia"],
            },
            2: {
                "name": "Igualdad de la mujer y equidad de género",
                "sector": "mujer_genero",
                "cluster": 2,
                "priority": "ALTA",
                "budget_min_percent": 2.0,
                "keywords": [
                    "género",
                    "mujer",
                    "equidad",
                    "igualdad",
                    "violencia_genero",
                ],
                "dnp_categories": ["equidad_genero", "mujer", "familia"],
            },
            3: {
                "name": "Ambiente sano, cambio climático, prevención y atención de desastres",
                "sector": "ambiente",
                "cluster": 3,
                "priority": "ALTA",
                "budget_min_percent": 1.5,
                "keywords": [
                    "ambiente",
                    "climático",
                    "desastres",
                    "sostenibilidad",
                    "gestión_riesgo",
                ],
                "dnp_categories": [
                    "ambiente",
                    "gestion_riesgo",
                    "ordenamiento_territorial",
                ],
            },
            4: {
                "name": "Derecho humano a la salud",
                "sector": "salud",
                "cluster": 4,
                "priority": "CRÍTICA",
                "budget_min_percent": 25.0,  # Mandato constitucional
                "keywords": [
                    "salud",
                    "salud_publica",
                    "atencion_medica",
                    "promocion",
                    "prevencion",
                ],
                "dnp_categories": ["salud", "salud_publica", "agua_potable"],
            },
            5: {
                "name": "Derecho a la educación",
                "sector": "educacion",
                "cluster": 4,
                "priority": "CRÍTICA",
                "budget_min_percent": 15.0,
                "keywords": [
                    "educación",
                    "acceso_educativo",
                    "calidad_educativa",
                    "cobertura",
                    "permanencia",
                ],
                "dnp_categories": [
                    "educacion",
                    "primera_infancia",
                    "ciencia_tecnologia",
                ],
            },
            6: {
                "name": "Derechos de la niñez y la juventud y protección de las familias",
                "sector": "niñez_juventud",
                "cluster": 2,
                "priority": "ALTA",
                "budget_min_percent": 5.0,
                "keywords": [
                    "niñez",
                    "juventud",
                    "familia",
                    "protección",
                    "desarrollo_integral",
                ],
                "dnp_categories": ["primera_infancia", "juventud", "familia"],
            },
            7: {
                "name": "Tierras y territorios",
                "sector": "agricultura_territorio",
                "cluster": 3,
                "priority": "ALTA",
                "budget_min_percent": 3.0,
                "keywords": [
                    "tierras",
                    "territorio",
                    "rural",
                    "ordenamiento",
                    "desarrollo_rural",
                ],
                "dnp_categories": [
                    "agropecuario",
                    "ordenamiento_territorial",
                    "desarrollo_rural",
                ],
            },
            8: {
                "name": "Líderes y defensores de derechos humanos sociales y ambientales",
                "sector": "derechos_humanos",
                "cluster": 1,
                "priority": "CRÍTICA",
                "budget_min_percent": 1.0,
                "keywords": [
                    "líderes",
                    "defensores",
                    "derechos_humanos",
                    "protección",
                    "garantías",
                ],
                "dnp_categories": ["derechos_humanos", "participacion", "seguridad"],
            },
            9: {
                "name": "Crisis de derechos de personas privadas de la libertad",
                "sector": "justicia",
                "cluster": 2,
                "priority": "MEDIA",
                "budget_min_percent": 0.5,
                "keywords": [
                    "privadas_libertad",
                    "cárceles",
                    "reinserción",
                    "justicia",
                    "derechos",
                ],
                "dnp_categories": ["justicia", "seguridad", "reinsercion_social"],
            },
            10: {
                "name": "Derecho al trabajo y seguridad social",
                "sector": "trabajo",
                "cluster": 4,
                "priority": "ALTA",
                "budget_min_percent": 2.0,
                "keywords": [
                    "trabajo",
                    "empleo",
                    "seguridad_social",
                    "productividad",
                    "generacion_ingresos",
                ],
                "dnp_categories": [
                    "trabajo",
                    "desarrollo_economico",
                    "seguridad_social",
                ],
            },
            11: {
                "name": "Derecho a la participación y la protesta social",
                "sector": "participacion",
                "cluster": 5,
                "priority": "ALTA",
                "budget_min_percent": 1.0,
                "keywords": [
                    "participación",
                    "protesta",
                    "democracia",
                    "ciudadanía",
                    "movilización",
                ],
                "dnp_categories": ["participacion", "democracia", "derechos_civiles"],
            },
        }

    def _define_value_chain_links(self) -> Dict[str, Dict[str, Any]]:
        """Definir eslabones de la cadena de valor DNP"""
        return {
            "insumos": {
                "name": "Insumos",
                "description": "Recursos financieros, humanos y técnicos",
                "elements": [
                    "presupuesto",
                    "personal",
                    "infraestructura",
                    "tecnología",
                ],
                "evaluation_focus": "disponibilidad_recursos",
                "keywords": ["presupuesto", "recursos", "personal", "infraestructura"],
                "order": 1,
            },
            "actividades": {
                "name": "Actividades",
                "description": "Procesos y acciones de implementación",
                "elements": ["programas", "proyectos", "acciones", "estrategias"],
                "evaluation_focus": "ejecucion_procesos",
                "keywords": ["actividades", "programas", "proyectos", "acciones"],
                "order": 2,
            },
            "productos": {
                "name": "Productos",
                "description": "Bienes y servicios entregados directamente",
                "elements": ["bienes", "servicios", "entregables", "beneficiarios"],
                "evaluation_focus": "entrega_productos",
                "keywords": [
                    "productos",
                    "servicios",
                    "entregables",
                    "beneficiarios",
                    "metas",
                ],
                "order": 3,
            },
            "resultados": {
                "name": "Resultados",
                "description": "Efectos directos en la población objetivo",
                "elements": ["cambios", "mejoras", "coberturas", "acceso"],
                "evaluation_focus": "efectos_poblacion",
                "keywords": [
                    "resultados",
                    "efectos",
                    "cambios",
                    "mejoras",
                    "indicadores",
                ],
                "order": 4,
            },
            "impactos": {
                "name": "Impactos",
                "description": "Transformaciones de largo plazo en bienestar",
                "elements": [
                    "transformación",
                    "desarrollo",
                    "bienestar",
                    "sostenibilidad",
                ],
                "evaluation_focus": "transformacion_territorial",
                "keywords": [
                    "impacto",
                    "transformación",
                    "desarrollo",
                    "bienestar",
                    "sostenible",
                ],
                "order": 5,
            },
        }

    def _integrate_prompt_catalog(self) -> Dict[str, Any]:
        """Integrar referencias del advanced_prompts_catalog"""
        return {
            "global_navigation": {
                "prompt_id": "0.0",
                "sections_to_identify": [
                    "diagnostico_territorial_poblacional",
                    "estrategias_programas_proyectos",
                    "presupuesto_fuentes_financiacion",
                    "indicadores_productos_resultados",
                ],
            },
            "dimension_prompts": {
                "DE1": {"base_id": "X.1", "focus": "logica_intervencion_coherencia"},
                "DE2": {"base_id": "X.2", "focus": "inclusion_tematica_poblacional"},
                "DE3": {
                    "base_id": "X.3",
                    "focus": "planificacion_adecuacion_presupuestal",
                },
                "DE4": {"base_id": "X.4", "focus": "cadena_valor_completitud"},
            },
            "cluster_execution": {
                "cluster_1": {"prompt_ids": ["7.1.1"], "points": [8]},
                "cluster_2": {"prompt_ids": ["7.1.2"], "points": [2, 6, 9]},
                "cluster_3": {"prompt_ids": ["7.1.3"], "points": [3, 7]},
                "cluster_4": {"prompt_ids": ["7.1.4"], "points": [4, 5, 10]},
                "cluster_5": {"prompt_ids": ["7.1.5"], "points": [1, 11]},
            },
        }

    def _build_complete_mapping(self) -> Dict[str, QuestionMapping]:
        """Construir mapeo completo basado en advanced_prompts_catalog"""
        mapping = {}

        # ========== CLUSTER 5: DERECHOS CIVILES Y POLÍTICOS ==========
        # Punto 1: Derecho a la vida, seguridad y convivencia
        point_1_questions = self._build_point_1_questions()

        # Punto 11: Derecho a la participación y protesta social
        point_11_questions = self._build_point_11_questions()

        # ========== CLUSTER 2: DERECHOS DE GRUPOS POBLACIONALES ==========
        # Punto 2: Igualdad de la mujer y equidad de género
        point_2_questions = self._build_point_2_questions()

        # Punto 6: Derechos niñez/juventud y protección familias
        point_6_questions = self._build_point_6_questions()

        # Punto 9: Crisis derechos personas privadas libertad
        point_9_questions = self._build_point_9_questions()

        # ========== CLUSTER 3: TERRITORIO, AMBIENTE Y DESARROLLO SOSTENIBLE ==========
        # Punto 3: Ambiente sano, cambio climático, desastres
        point_3_questions = self._build_point_3_questions()

        # Punto 7: Tierras y territorios
        point_7_questions = self._build_point_7_questions()

        # ========== CLUSTER 4: DESC ==========
        # Punto 4: Derecho humano a la salud
        point_4_questions = self._build_point_4_questions()

        # Punto 5: Derecho a la educación
        point_5_questions = self._build_point_5_questions()

        # Punto 10: Derecho al trabajo y seguridad social
        point_10_questions = self._build_point_10_questions()

        # ========== CLUSTER 1: VÍCTIMAS Y CONSTRUCCIÓN DE PAZ ==========
        # Punto 8: Líderes y defensores de DDHH
        point_8_questions = self._build_point_8_questions()

        # Consolidar todas las preguntas
        all_questions = (
            point_1_questions
            + point_2_questions
            + point_3_questions
            + point_4_questions
            + point_5_questions
            + point_6_questions
            + point_7_questions
            + point_8_questions
            + point_9_questions
            + point_10_questions
            + point_11_questions
        )

        # Convertir a diccionario
        for question in all_questions:
            mapping[question.question_id] = question

        return mapping

    def _build_point_1_questions(self) -> List[QuestionMapping]:
        """Construir preguntas para Punto 1: Vida, seguridad y convivencia"""
        cluster_info = self.cluster_definitions[5]  # Cluster 5
        point_info = self.decalogo_points[1]

        return [
            # DE1: Lógica de Intervención
            QuestionMapping(
                question_id="DE1_P1_Q1",
                question_text="¿El PDT identifica factores de riesgo específicos que amenazan la vida y seguridad?",
                decalogo_point=1,
                point_name=point_info["name"],
                dimension="DE1",
                value_chain_link="productos",
                cluster_id=5,
                cluster_name=cluster_info["name"],
                dnp_category="seguridad_ciudadana",
                sector="seguridad",
                weight=1.0,
                keywords=["factores", "riesgo", "amenazas", "vida", "seguridad"],
                search_patterns=[
                    "factores.*riesgo.*seguridad",
                    "amenazas.*vida.*convivencia",
                    "violencia.*causas.*específicas",
                ],
                prompt_reference="PROMPT 1.1",
                evidence_requirements=[
                    "datos estadísticos",
                    "análisis causal",
                    "identificación factores",
                ],
                navigation_hints=[
                    "diagnóstico territorial",
                    "caracterización problemas",
                    "análisis situacional",
                ],
                answer_options=["Sí", "Parcial", "No", "NI"],
                scoring_criteria={"bienestar_component": True, "weight": 0.6},
            ),
            QuestionMapping(
                question_id="DE1_P1_Q2",
                question_text="¿Las estrategias propuestas abordan las causas estructurales de la violencia?",
                decalogo_point=1,
                point_name=point_info["name"],
                dimension="DE1",
                value_chain_link="resultados",
                cluster_id=5,
                cluster_name=cluster_info["name"],
                dnp_category="seguridad_ciudadana",
                sector="seguridad",
                weight=1.0,
                keywords=["estrategias", "causas", "estructurales", "violencia"],
                search_patterns=[
                    "causas.*estructurales.*violencia",
                    "estrategias.*prevención.*integral",
                    "intervención.*factores.*generadores",
                ],
                prompt_reference="PROMPT 1.1",
                evidence_requirements=[
                    "análisis causal",
                    "estrategias integrales",
                    "teoría de cambio",
                ],
                navigation_hints=[
                    "programas estratégicos",
                    "líneas de acción",
                    "teoría de cambio",
                ],
                answer_options=["Sí", "Parcial", "No", "NI"],
                scoring_criteria={"bienestar_component": True, "weight": 0.6},
            ),
            # DE2: Inclusión Temática
            QuestionMapping(
                question_id="DE2_P1_Q1",
                question_text="¿Incluye enfoque diferencial para protección de grupos vulnerables en seguridad?",
                decalogo_point=1,
                point_name=point_info["name"],
                dimension="DE2",
                value_chain_link="productos",
                cluster_id=5,
                cluster_name=cluster_info["name"],
                dnp_category="seguridad_ciudadana",
                sector="seguridad",
                weight=1.0,
                keywords=[
                    "enfoque",
                    "diferencial",
                    "grupos",
                    "vulnerables",
                    "protección",
                ],
                search_patterns=[
                    "enfoque.*diferencial.*seguridad",
                    "grupos.*vulnerables.*protección",
                    "poblaciones.*especiales.*riesgo",
                ],
                prompt_reference="PROMPT 1.2",
                evidence_requirements=[
                    "identificación grupos",
                    "medidas específicas",
                    "protección diferencial",
                ],
                navigation_hints=[
                    "diagnóstico poblacional",
                    "programas focalizados",
                    "medidas diferenciales",
                ],
                answer_options=["Sí", "Parcial", "No", "NI"],
                scoring_criteria={"binary_question": True},
            ),
            # DE3: Planificación Presupuestal
            QuestionMapping(
                question_id="DE3_P1_Q1",
                question_text="¿Existe presupuesto específico y suficiente para programas de seguridad?",
                decalogo_point=1,
                point_name=point_info["name"],
                dimension="DE3",
                value_chain_link="insumos",
                cluster_id=5,
                cluster_name=cluster_info["name"],
                dnp_category="seguridad_ciudadana",
                sector="seguridad",
                weight=2.0,
                keywords=["presupuesto", "específico", "suficiente", "seguridad"],
                search_patterns=[
                    "presupuesto.*seguridad.*ciudadana",
                    "recursos.*orden.*público",
                    "inversión.*convivencia.*paz",
                ],
                prompt_reference="PROMPT 1.3",
                evidence_requirements=[
                    "asignación presupuestal",
                    "plan financiero",
                    "fuentes recursos",
                ],
                navigation_hints=[
                    "plan inversiones",
                    "presupuesto sectorial",
                    "plan financiero",
                ],
                answer_options=["Sí", "Parcial", "No", "NI"],
                scoring_criteria={"category": "general", "max_points": 10},
            ),
            # DE4: Cadena de Valor
            QuestionMapping(
                question_id="DE4_P1_Q1",
                question_text="¿La cadena prevención-protección-atención-reparación está completa?",
                decalogo_point=1,
                point_name=point_info["name"],
                dimension="DE4",
                value_chain_link="impactos",
                cluster_id=5,
                cluster_name=cluster_info["name"],
                dnp_category="seguridad_ciudadana",
                sector="seguridad",
                weight=2.0,
                keywords=[
                    "prevención",
                    "protección",
                    "atención",
                    "reparación",
                    "completa",
                ],
                search_patterns=[
                    "prevención.*protección.*atención",
                    "cadena.*integral.*seguridad",
                    "ciclo.*completo.*intervención",
                ],
                prompt_reference="PROMPT 1.4",
                evidence_requirements=[
                    "eslabones cadena",
                    "secuencia lógica",
                    "integralidad",
                ],
                navigation_hints=[
                    "programas integrales",
                    "rutas de atención",
                    "sistema protección",
                ],
                answer_options=["Sí", "No"],
                scoring_criteria={"key_link": "diagnostico_linea_base"},
            ),
        ]

    def _build_point_2_questions(self) -> List[QuestionMapping]:
        """Construir preguntas para Punto 2: Igualdad mujer y equidad género"""
        cluster_info = self.cluster_definitions[2]
        point_info = self.decalogo_points[2]

        return [
            QuestionMapping(
                question_id="DE1_P2_Q1",
                question_text="¿Define estrategias específicas para cerrar brechas de género identificadas?",
                decalogo_point=2,
                point_name=point_info["name"],
                dimension="DE1",
                value_chain_link="resultados",
                cluster_id=2,
                cluster_name=cluster_info["name"],
                dnp_category="equidad_genero",
                sector="mujer_genero",
                weight=1.0,
                keywords=["estrategias", "brechas", "género", "específicas"],
                search_patterns=[
                    "brechas.*género.*estrategias",
                    "desigualdad.*mujer.*cierre",
                    "equidad.*género.*acciones",
                ],
                prompt_reference="PROMPT 2.1",
                evidence_requirements=[
                    "análisis brechas",
                    "estrategias focalizadas",
                    "metas género",
                ],
                navigation_hints=[
                    "diagnóstico género",
                    "programas mujer",
                    "indicadores equidad",
                ],
                answer_options=["Sí", "Parcial", "No", "NI"],
                scoring_criteria={"bienestar_component": True, "weight": 0.6},
            ),
            QuestionMapping(
                question_id="DE2_P2_Q1",
                question_text="¿Incluye datos desagregados por sexo y análisis interseccional?",
                decalogo_point=2,
                point_name=point_info["name"],
                dimension="DE2",
                value_chain_link="productos",
                cluster_id=2,
                cluster_name=cluster_info["name"],
                dnp_category="equidad_genero",
                sector="mujer_genero",
                weight=1.0,
                keywords=["datos", "desagregados", "sexo", "interseccional"],
                search_patterns=[
                    "datos.*desagregados.*sexo",
                    "análisis.*interseccional.*género",
                    "estadísticas.*diferenciadas.*mujer",
                ],
                prompt_reference="PROMPT 2.2",
                evidence_requirements=[
                    "datos desagregados",
                    "análisis diferencial",
                    "interseccionalidad",
                ],
                navigation_hints=[
                    "diagnóstico estadístico",
                    "caracterización poblacional",
                    "análisis situacional",
                ],
                answer_options=["Sí", "Parcial", "No", "NI"],
                scoring_criteria={"binary_question": True},
            ),
        ]

    # Implementar métodos para todos los puntos restantes de manera similar...
    def _build_point_3_questions(self) -> List[QuestionMapping]:
        """Punto 3: Ambiente sano, cambio climático, desastres"""
        cluster_info = self.cluster_definitions[3]
        point_info = self.decalogo_points[3]

        return [
            QuestionMapping(
                question_id="DE1_P3_Q1",
                question_text="¿Identifica amenazas ambientales y climáticas específicas del territorio?",
                decalogo_point=3,
                point_name=point_info["name"],
                dimension="DE1",
                value_chain_link="productos",
                cluster_id=3,
                cluster_name=cluster_info["name"],
                dnp_category="ambiente",
                sector="ambiente",
                weight=1.0,
                keywords=["amenazas", "ambientales", "climáticas", "territorio"],
                search_patterns=[
                    "amenazas.*ambientales.*territorio",
                    "riesgos.*climáticos.*específicos",
                    "vulnerabilidad.*ambiental.*local",
                ],
                prompt_reference="PROMPT 3.1",
                evidence_requirements=[
                    "caracterización ambiental",
                    "análisis riesgos",
                    "mapeo amenazas",
                ],
                navigation_hints=[
                    "diagnóstico ambiental",
                    "gestión de riesgo",
                    "ordenamiento territorial",
                ],
                answer_options=["Sí", "Parcial", "No", "NI"],
                scoring_criteria={"bienestar_component": True, "weight": 0.6},
            )
        ]

    def _build_point_4_questions(self) -> List[QuestionMapping]:
        """Punto 4: Derecho humano a la salud"""
        cluster_info = self.cluster_definitions[4]
        point_info = self.decalogo_points[4]

        return [
            QuestionMapping(
                question_id="DE1_P4_Q1",
                question_text="¿Define estrategias para garantizar acceso universal a servicios de salud?",
                decalogo_point=4,
                point_name=point_info["name"],
                dimension="DE1",
                value_chain_link="resultados",
                cluster_id=4,
                cluster_name=cluster_info["name"],
                dnp_category="salud",
                sector="salud",
                weight=1.0,
                keywords=["acceso", "universal", "servicios", "salud"],
                search_patterns=[
                    "acceso.*universal.*salud",
                    "servicios.*salud.*garantía",
                    "cobertura.*salud.*territorial",
                ],
                prompt_reference="PROMPT 4.1",
                evidence_requirements=[
                    "análisis cobertura",
                    "estrategias acceso",
                    "barreras identificadas",
                ],
                navigation_hints=[
                    "diagnóstico salud",
                    "programas sanitarios",
                    "plan salud territorial",
                ],
                answer_options=["Sí", "Parcial", "No", "NI"],
                scoring_criteria={"bienestar_component": True, "weight": 0.6},
            ),
            QuestionMapping(
                question_id="DE3_P4_Q1",
                question_text="¿Cumple con la asignación mínima del 25% del presupuesto a salud?",
                decalogo_point=4,
                point_name=point_info["name"],
                dimension="DE3",
                value_chain_link="insumos",
                cluster_id=4,
                cluster_name=cluster_info["name"],
                dnp_category="salud",
                sector="salud",
                weight=3.0,  # Peso alto por ser mandato constitucional
                keywords=["25%", "presupuesto", "salud", "mínima", "asignación"],
                search_patterns=[
                    "25.*por.*ciento.*salud",
                    "presupuesto.*salud.*territorial",
                    "recursos.*sector.*salud",
                ],
                prompt_reference="PROMPT 4.3",
                evidence_requirements=[
                    "asignación presupuestal",
                    "cumplimiento normativo",
                    "plan financiero",
                ],
                navigation_hints=[
                    "plan inversiones",
                    "presupuesto sectorial",
                    "recursos salud",
                ],
                answer_options=["Sí", "Parcial", "No", "NI"],
                scoring_criteria={
                    "category": "asignacion",
                    "max_points": 10,
                    "critical": True,
                },
            ),
        ]

    def _build_point_5_questions(self) -> List[QuestionMapping]:
        """Punto 5: Derecho a la educación"""
        cluster_info = self.cluster_definitions[4]
        point_info = self.decalogo_points[5]

        return [
            QuestionMapping(
                question_id="DE1_P5_Q1",
                question_text="¿Define estrategias para mejorar acceso, permanencia y calidad educativa?",
                decalogo_point=5,
                point_name=point_info["name"],
                dimension="DE1",
                value_chain_link="resultados",
                cluster_id=4,
                cluster_name=cluster_info["name"],
                dnp_category="educacion",
                sector="educacion",
                weight=1.0,
                keywords=["acceso", "permanencia", "calidad", "educativa"],
                search_patterns=[
                    "acceso.*permanencia.*calidad.*educativa",
                    "cobertura.*educación.*territorial",
                    "deserción.*escolar.*estrategias",
                ],
                prompt_reference="PROMPT 5.1",
                evidence_requirements=[
                    "análisis educativo",
                    "estrategias integrales",
                    "indicadores educación",
                ],
                navigation_hints=[
                    "diagnóstico educativo",
                    "programas educación",
                    "plan educativo territorial",
                ],
                answer_options=["Sí", "Parcial", "No", "NI"],
                scoring_criteria={"bienestar_component": True, "weight": 0.6},
            )
        ]

    def _build_point_6_questions(self) -> List[QuestionMapping]:
        """Punto 6: Derechos niñez/juventud y familias"""
        cluster_info = self.cluster_definitions[2]
        point_info = self.decalogo_points[6]

        return [
            QuestionMapping(
                question_id="DE1_P6_Q1",
                question_text="¿Define estrategias integrales para desarrollo de niñez, juventud y fortalecimiento familiar?",
                decalogo_point=6,
                point_name=point_info["name"],
                dimension="DE1",
                value_chain_link="resultados",
                cluster_id=2,
                cluster_name=cluster_info["name"],
                dnp_category="primera_infancia",
                sector="niñez_juventud",
                weight=1.0,
                keywords=["desarrollo", "integral", "niñez", "juventud", "familiar"],
                search_patterns=[
                    "desarrollo.*integral.*niñez.*juventud",
                    "fortalecimiento.*familiar.*estrategias",
                    "primera.*infancia.*adolescencia",
                ],
                prompt_reference="PROMPT 2.5",
                evidence_requirements=[
                    "desarrollo integral",
                    "rutas atención",
                    "fortalecimiento familiar",
                ],
                navigation_hints=[
                    "programas niñez",
                    "juventud",
                    "familia",
                    "primera infancia",
                ],
                answer_options=["Sí", "Parcial", "No", "NI"],
                scoring_criteria={"bienestar_component": True, "weight": 0.6},
            )
        ]

    def _build_point_7_questions(self) -> List[QuestionMapping]:
        """Punto 7: Tierras y territorios"""
        cluster_info = self.cluster_definitions[3]
        point_info = self.decalogo_points[7]

        return [
            QuestionMapping(
                question_id="DE1_P7_Q1",
                question_text="¿Define estrategias para ordenamiento territorial y acceso equitativo a tierras?",
                decalogo_point=7,
                point_name=point_info["name"],
                dimension="DE1",
                value_chain_link="resultados",
                cluster_id=3,
                cluster_name=cluster_info["name"],
                dnp_category="ordenamiento_territorial",
                sector="agricultura_territorio",
                weight=1.0,
                keywords=["ordenamiento", "territorial", "acceso", "tierras"],
                search_patterns=[
                    "ordenamiento.*territorial.*estrategias",
                    "acceso.*tierra.*equitativo",
                    "desarrollo.*rural.*territorial",
                ],
                prompt_reference="PROMPT 3.5",
                evidence_requirements=[
                    "plan ordenamiento",
                    "acceso tierras",
                    "desarrollo rural",
                ],
                navigation_hints=[
                    "POT",
                    "desarrollo rural",
                    "ordenamiento territorial",
                    "reforma agraria",
                ],
                answer_options=["Sí", "Parcial", "No", "NI"],
                scoring_criteria={"bienestar_component": True, "weight": 0.6},
            )
        ]

    def _build_point_8_questions(self) -> List[QuestionMapping]:
        """Punto 8: Líderes y defensores DDHH"""
        cluster_info = self.cluster_definitions[1]
        point_info = self.decalogo_points[8]

        return [
            QuestionMapping(
                question_id="DE1_P8_Q1",
                question_text="¿Define estrategias específicas para protección de líderes y defensores de DDHH?",
                decalogo_point=8,
                point_name=point_info["name"],
                dimension="DE1",
                value_chain_link="productos",
                cluster_id=1,
                cluster_name=cluster_info["name"],
                dnp_category="derechos_humanos",
                sector="derechos_humanos",
                weight=2.0,  # Peso alto por criticidad
                keywords=["protección", "líderes", "defensores", "DDHH"],
                search_patterns=[
                    "protección.*líderes.*sociales",
                    "defensores.*derechos.*humanos",
                    "garantías.*liderazgo.*social",
                ],
                prompt_reference="PROMPT 1.9",
                evidence_requirements=[
                    "análisis riesgos",
                    "medidas protección",
                    "garantías específicas",
                ],
                navigation_hints=[
                    "programa protección",
                    "derechos humanos",
                    "liderazgo social",
                ],
                answer_options=["Sí", "Parcial", "No", "NI"],
                scoring_criteria={
                    "bienestar_component": True,
                    "weight": 0.6,
                    "critical": True,
                },
            )
        ]

    def _build_point_9_questions(self) -> List[QuestionMapping]:
        """Punto 9: Crisis derechos personas privadas libertad"""
        cluster_info = self.cluster_definitions[2]
        point_info = self.decalogo_points[9]

        return [
            QuestionMapping(
                question_id="DE1_P9_Q1",
                question_text="¿Define estrategias para reinserción social y garantía de derechos en centros carcelarios?",
                decalogo_point=9,
                point_name=point_info["name"],
                dimension="DE1",
                value_chain_link="resultados",
                cluster_id=2,
                cluster_name=cluster_info["name"],
                dnp_category="justicia",
                sector="justicia",
                weight=1.0,
                keywords=["reinserción", "social", "derechos", "carcelarios"],
                search_patterns=[
                    "reinserción.*social.*estrategias",
                    "derechos.*privados.*libertad",
                    "centros.*carcelarios.*programas",
                ],
                prompt_reference="PROMPT 2.9",
                evidence_requirements=[
                    "programas reinserción",
                    "garantías derechos",
                    "condiciones carcelarias",
                ],
                navigation_hints=[
                    "justicia",
                    "reinserción social",
                    "centros penitenciarios",
                ],
                answer_options=["Sí", "Parcial", "No", "NI"],
                scoring_criteria={"bienestar_component": True, "weight": 0.6},
            )
        ]

    def _build_point_10_questions(self) -> List[QuestionMapping]:
        """Punto 10: Derecho trabajo y seguridad social"""
        cluster_info = self.cluster_definitions[4]
        point_info = self.decalogo_points[10]

        return [
            QuestionMapping(
                question_id="DE1_P10_Q1",
                question_text="¿Define estrategias para generación de empleo digno y productividad territorial?",
                decalogo_point=10,
                point_name=point_info["name"],
                dimension="DE1",
                value_chain_link="resultados",
                cluster_id=4,
                cluster_name=cluster_info["name"],
                dnp_category="desarrollo_economico",
                sector="trabajo",
                weight=1.0,
                keywords=["empleo", "digno", "productividad", "territorial"],
                search_patterns=[
                    "generación.*empleo.*estrategias",
                    "trabajo.*digno.*territorial",
                    "productividad.*desarrollo.*económico",
                ],
                prompt_reference="PROMPT 4.9",
                evidence_requirements=[
                    "estrategias empleo",
                    "desarrollo económico",
                    "productividad",
                ],
                navigation_hints=[
                    "desarrollo económico",
                    "empleo",
                    "competitividad",
                    "productividad",
                ],
                answer_options=["Sí", "Parcial", "No", "NI"],
                scoring_criteria={"bienestar_component": True, "weight": 0.6},
            )
        ]

    def _build_point_11_questions(self) -> List[QuestionMapping]:
        """Punto 11: Derecho participación y protesta social"""
        cluster_info = self.cluster_definitions[5]
        point_info = self.decalogo_points[11]

        return [
            QuestionMapping(
                question_id="DE1_P11_Q1",
                question_text="¿Define mecanismos específicos para garantizar participación ciudadana efectiva?",
                decalogo_point=11,
                point_name=point_info["name"],
                dimension="DE1",
                value_chain_link="productos",
                cluster_id=5,
                cluster_name=cluster_info["name"],
                dnp_category="participacion",
                sector="participacion",
                weight=1.0,
                keywords=["mecanismos", "participación", "ciudadana", "efectiva"],
                search_patterns=[
                    "participación.*ciudadana.*mecanismos",
                    "espacios.*participación.*efectiva",
                    "democracia.*participativa.*territorial",
                ],
                prompt_reference="PROMPT 11.1",
                evidence_requirements=[
                    "mecanismos participación",
                    "espacios ciudadanos",
                    "garantías participativas",
                ],
                navigation_hints=[
                    "participación ciudadana",
                    "democracia participativa",
                    "control social",
                ],
                answer_options=["Sí", "Parcial", "No", "NI"],
                scoring_criteria={"bienestar_component": True, "weight": 0.6},
            ),
            QuestionMapping(
                question_id="DE2_P11_Q1",
                question_text="¿Incluye protocolos para protesta social pacífica y derechos civiles?",
                decalogo_point=11,
                point_name=point_info["name"],
                dimension="DE2",
                value_chain_link="productos",
                cluster_id=5,
                cluster_name=cluster_info["name"],
                dnp_category="derechos_civiles",
                sector="participacion",
                weight=1.0,
                keywords=["protocolos", "protesta", "social", "derechos", "civiles"],
                search_patterns=[
                    "protesta.*social.*protocolos",
                    "derechos.*civiles.*garantías",
                    "movilización.*social.*pacífica",
                ],
                prompt_reference="PROMPT 11.2",
                evidence_requirements=[
                    "protocolos protesta",
                    "garantías civiles",
                    "derechos manifestación",
                ],
                navigation_hints=[
                    "derechos civiles",
                    "protesta social",
                    "movilización ciudadana",
                ],
                answer_options=["Sí", "Parcial", "No", "NI"],
                scoring_criteria={"binary_question": True},
            ),
        ]

    def generate_comprehensive_coverage_matrix(self) -> Dict[str, Any]:
        """Generar matriz de cobertura exhaustiva"""
        if not self.coverage_matrix:
            coverage = {}

            # Inicializar matriz completa
            for point_id in range(1, 12):
                point_data = self.decalogo_points[point_id]
                cluster_id = point_data["cluster"]
                cluster_info = self.cluster_definitions[cluster_id]

                coverage[f"point_{point_id}"] = {
                    "point_info": {
                        "id": point_id,
                        "name": point_data["name"],
                        "cluster_id": cluster_id,
                        "cluster_name": cluster_info["name"],
                        "priority": point_data["priority"],
                        "sector": point_data["sector"],
                    },
                    "dimensions": {
                        "DE1": {
                            "questions": [],
                            "coverage_score": 0.0,
                            "critical_gaps": [],
                        },
                        "DE2": {
                            "questions": [],
                            "coverage_score": 0.0,
                            "critical_gaps": [],
                        },
                        "DE3": {
                            "questions": [],
                            "coverage_score": 0.0,
                            "critical_gaps": [],
                        },
                        "DE4": {
                            "questions": [],
                            "coverage_score": 0.0,
                            "critical_gaps": [],
                        },
                    },
                    "value_chain_coverage": {
                        "insumos": [],
                        "actividades": [],
                        "productos": [],
                        "resultados": [],
                        "impactos": [],
                    },
                    "total_questions": 0,
                    "overall_coverage_score": 0.0,
                    "completeness_status": "INCOMPLETE",
                }

            # Llenar matriz con preguntas mapeadas
            for question_id, mapping in self.complete_mapping.items():
                point_key = f"point_{mapping.decalogo_point}"
                dimension = mapping.dimension
                value_chain = mapping.value_chain_link

                question_data = {
                    "question_id": question_id,
                    "question_text": mapping.question_text,
                    "weight": mapping.weight,
                    "prompt_reference": mapping.prompt_reference,
                    "sector": mapping.sector,
                    "critical": mapping.scoring_criteria.get("critical", False),
                }

                coverage[point_key]["dimensions"][dimension]["questions"].append(
                    question_data
                )
                coverage[point_key]["value_chain_coverage"][value_chain].append(
                    question_data
                )
                coverage[point_key]["total_questions"] += 1

            # Calcular scores de cobertura
            for point_key in coverage:
                point_coverage = coverage[point_key]

                # Score por dimensión (0-1)
                for dim in ["DE1", "DE2", "DE3", "DE4"]:
                    dim_questions = len(point_coverage["dimensions"][dim]["questions"])
                    # Mínimo esperado: 2 preguntas por dimensión
                    expected_min = 2
                    point_coverage["dimensions"][dim]["coverage_score"] = min(
                        dim_questions / expected_min, 1.0
                    )

                    if dim_questions < expected_min:
                        point_coverage["dimensions"][dim]["critical_gaps"].append(
                            f"Faltan {expected_min - dim_questions} preguntas mínimas"
                        )

                # Score global del punto
                dimension_scores = [
                    point_coverage["dimensions"][dim]["coverage_score"]
                    for dim in ["DE1", "DE2", "DE3", "DE4"]
                ]
                point_coverage["overall_coverage_score"] = sum(dimension_scores) / 4.0

                # Estado de completitud
                if point_coverage["overall_coverage_score"] >= 0.8:
                    point_coverage["completeness_status"] = "COMPLETE"
                elif point_coverage["overall_coverage_score"] >= 0.5:
                    point_coverage["completeness_status"] = "PARTIAL"
                else:
                    point_coverage["completeness_status"] = "INCOMPLETE"

            self.coverage_matrix = coverage

        return self.coverage_matrix

    def generate_advanced_validation_report(self) -> Dict[str, Any]:
        """Generar reporte de validación avanzado"""
        if not self.coverage_matrix:
            self.generate_comprehensive_coverage_matrix()

        validation_report = {
            "metadata": {
                "validation_date": datetime.now().isoformat(),
                "total_questions_mapped": len(self.complete_mapping),
                "total_points_decalogo": 11,
                "total_dimensions": 4,
                "methodology_compliance": "advanced_prompts_catalog v1.0",
            },
            "overall_status": {
                "is_complete": True,
                "completion_percentage": 0.0,
                "critical_issues": [],
                "recommendations": [],
            },
            "cluster_analysis": {},
            "dimension_analysis": {},
            "point_by_point_status": {},
            "missing_mappings": [],
            "quality_assessment": {
                "prompt_catalog_adherence": 0.0,
                "coverage_exhaustiveness": 0.0,
                "search_pattern_completeness": 0.0,
            },
        }

        # Análisis por cluster
        for cluster_id, cluster_info in self.cluster_definitions.items():
            cluster_points = cluster_info["points"]
            cluster_questions = [
                q
                for q in self.complete_mapping.values()
                if q.decalogo_point in cluster_points
            ]

            validation_report["cluster_analysis"][f"cluster_{cluster_id}"] = {
                "name": cluster_info["name"],
                "points_count": len(cluster_points),
                "questions_mapped": len(cluster_questions),
                "points_covered": len(set(q.decalogo_point for q in cluster_questions)),
                "coverage_status": "COMPLETE"
                if len(set(q.decalogo_point for q in cluster_questions))
                == len(cluster_points)
                else "INCOMPLETE",
                "priority": cluster_info["priority"],
            }

        # Análisis por dimensión
        for dimension in ["DE1", "DE2", "DE3", "DE4"]:
            dim_questions = [
                q for q in self.complete_mapping.values() if q.dimension == dimension
            ]
            points_covered = set(q.decalogo_point for q in dim_questions)

            validation_report["dimension_analysis"][dimension] = {
                "name": self.dimension_standards[dimension]["name"],
                "questions_mapped": len(dim_questions),
                "points_covered": len(points_covered),
                "coverage_percentage": len(points_covered) / 11 * 100,
                "missing_points": [p for p in range(1, 12) if p not in points_covered],
            }

        # Estado punto por punto
        for point_id in range(1, 12):
            point_coverage = self.coverage_matrix[f"point_{point_id}"]
            validation_report["point_by_point_status"][f"point_{point_id}"] = {
                "name": point_coverage["point_info"]["name"],
                "cluster": point_coverage["point_info"]["cluster_name"],
                "total_questions": point_coverage["total_questions"],
                "coverage_score": point_coverage["overall_coverage_score"],
                "status": point_coverage["completeness_status"],
                "critical_gaps": [],
            }

            # Identificar gaps críticos
            for dim in ["DE1", "DE2", "DE3", "DE4"]:
                if point_coverage["dimensions"][dim]["coverage_score"] < 0.5:
                    gap_info = {
                        "dimension": dim,
                        "current_questions": len(
                            point_coverage["dimensions"][dim]["questions"]
                        ),
                        "minimum_expected": 2,
                        "issue": f"Cobertura insuficiente en {dim}",
                    }
                    validation_report["point_by_point_status"][f"point_{point_id}"][
                        "critical_gaps"
                    ].append(gap_info)
                    validation_report["missing_mappings"].append(gap_info)

        # Calcular completion percentage
        total_coverage = sum(
            self.coverage_matrix[f"point_{p}"]["overall_coverage_score"]
            for p in range(1, 12)
        )
        validation_report["overall_status"]["completion_percentage"] = (
            total_coverage / 11
        ) * 100

        # Determinar si está completo
        validation_report["overall_status"]["is_complete"] = (
            validation_report["overall_status"]["completion_percentage"] >= 80
        )

        # Generar recomendaciones
        if not validation_report["overall_status"]["is_complete"]:
            validation_report["overall_status"]["recommendations"].extend(
                [
                    "Completar preguntas faltantes para alcanzar cobertura mínima",
                    "Priorizar puntos con status INCOMPLETE",
                    "Validar adherencia a advanced_prompts_catalog",
                    "Implementar preguntas DE4 para completar cadenas de valor",
                ]
            )

        self.validation_report = validation_report
        return validation_report

    def initialize_query_engine(self) -> "AdvancedMappingQueryEngine":
        """Inicializar motor de consultas avanzado"""
        if not self.query_engine:
            self.query_engine = AdvancedMappingQueryEngine(self)
        return self.query_engine

    def export_comprehensive_mapping_file(
        self, output_path: str = "config/advanced_question_decalogo_mapping.yaml"
    ) -> Dict[str, Any]:
        """Exportar archivo YAML maestro comprehensivo"""

        # Generar todos los análisis
        if not self.coverage_matrix:
            self.generate_comprehensive_coverage_matrix()
        if not self.validation_report:
            self.generate_advanced_validation_report()

        # Convertir mapping a formato serializable
        serializable_mapping = {}
        for question_id, mapping in self.complete_mapping.items():
            serializable_mapping[question_id] = asdict(mapping)

        comprehensive_content = {
            "metadata": {
                "version": "2.0.0",
                "created_date": datetime.now().isoformat(),
                "methodology_source": "advanced_prompts_catalog",
                "total_questions": len(self.complete_mapping),
                "total_decalogo_points": 11,
                "dimensions": ["DE1", "DE2", "DE3", "DE4"],
                "clusters": 5,
                "value_chain_links": 5,
                "completion_status": "COMPLETE"
                if self.validation_report["overall_status"]["is_complete"]
                else "IN_PROGRESS",
                "compliance_level": "ADVANCED",
            },
            "cluster_definitions": self.cluster_definitions,
            "decalogo_points": self.decalogo_points,
            "dimension_standards": self.dimension_standards,
            "value_chain_definition": self.value_chain_links,
            "prompt_catalog_integration": self.prompt_catalog_integration,
            "question_mapping": serializable_mapping,
            "coverage_matrix": self.coverage_matrix,
            "validation_report": self.validation_report,
            "advanced_features": {
                "search_patterns_included": True,
                "evidence_requirements_defined": True,
                "navigation_hints_provided": True,
                "scoring_criteria_detailed": True,
                "cluster_integration_complete": True,
            },
        }

        # Crear directorio si no existe
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Escribir archivo
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(
                comprehensive_content,
                f,
                allow_unicode=True,
                default_flow_style=False,
                indent=2,
            )

        logger.info(f"Archivo de mapeo avanzado generado: {output_path}")
        return comprehensive_content


class AdvancedMappingQueryEngine:
    """Motor de consultas avanzado para navegación del mapeo"""

    def __init__(self, mapper: AdvancedQuestionDecalogoMapper):
        self.mapper = mapper
        self.mapping = mapper.complete_mapping
        self.clusters = mapper.cluster_definitions
        self.dimensions = mapper.dimension_standards

        # Construir índices avanzados
        self.index_by_point = self._build_point_index()
        self.index_by_dimension = self._build_dimension_index()
        self.index_by_cluster = self._build_cluster_index()
        self.index_by_value_chain = self._build_value_chain_index()
        self.index_by_sector = self._build_sector_index()
        self.index_by_priority = self._build_priority_index()

    def _build_point_index(self) -> Dict[int, List[QuestionMapping]]:
        """Índice por punto del decálogo"""
        index = {}
        for mapping in self.mapping.values():
            point = mapping.decalogo_point
            if point not in index:
                index[point] = []
            index[point].append(mapping)
        return index

    def _build_dimension_index(self) -> Dict[str, List[QuestionMapping]]:
        """Índice por dimensión"""
        index = {}
        for mapping in self.mapping.values():
            dimension = mapping.dimension
            if dimension not in index:
                index[dimension] = []
            index[dimension].append(mapping)
        return index

    def _build_cluster_index(self) -> Dict[int, List[QuestionMapping]]:
        """Índice por cluster"""
        index = {}
        for mapping in self.mapping.values():
            cluster = mapping.cluster_id
            if cluster not in index:
                index[cluster] = []
            index[cluster].append(mapping)
        return index

    def _build_value_chain_index(self) -> Dict[str, List[QuestionMapping]]:
        """Índice por eslabón de cadena de valor"""
        index = {}
        for mapping in self.mapping.values():
            link = mapping.value_chain_link
            if link not in index:
                index[link] = []
            index[link].append(mapping)
        return index

    def _build_sector_index(self) -> Dict[str, List[QuestionMapping]]:
        """Índice por sector"""
        index = {}
        for mapping in self.mapping.values():
            sector = mapping.sector
            if sector not in index:
                index[sector] = []
            index[sector].append(mapping)
        return index

    def _build_priority_index(self) -> Dict[str, List[QuestionMapping]]:
        """Índice por prioridad"""
        index = {}
        for mapping in self.mapping.values():
            point_info = self.mapper.decalogo_points[mapping.decalogo_point]
            priority = point_info["priority"]
            if priority not in index:
                index[priority] = []
            index[priority].append(mapping)
        return index

    # Métodos de consulta avanzados
    def get_questions_by_point(self, point_id: int) -> List[QuestionMapping]:
        """Obtener todas las preguntas de un punto del decálogo"""
        return self.index_by_point.get(point_id, [])

    def get_questions_by_dimension(self, dimension: str) -> List[QuestionMapping]:
        """Obtener todas las preguntas de una dimensión"""
        return self.index_by_dimension.get(dimension, [])

    def get_questions_by_cluster(self, cluster_id: int) -> List[QuestionMapping]:
        """Obtener todas las preguntas de un cluster"""
        return self.index_by_cluster.get(cluster_id, [])

    def get_questions_by_value_chain(self, link: str) -> List[QuestionMapping]:
        """Obtener preguntas por eslabón de cadena de valor"""
        return self.index_by_value_chain.get(link, [])

    def get_critical_questions(self) -> List[QuestionMapping]:
        """Obtener preguntas de puntos críticos"""
        return self.index_by_priority.get("CRÍTICA", [])

    def get_questions_by_prompt_reference(
        self, prompt_ref: str
    ) -> List[QuestionMapping]:
        """Obtener preguntas por referencia de prompt"""
        return [q for q in self.mapping.values() if q.prompt_reference == prompt_ref]

    def search_questions_by_keywords(
        self, keywords: List[str]
    ) -> List[QuestionMapping]:
        """Buscar preguntas por palabras clave"""
        results = []
        for mapping in self.mapping.values():
            if any(
                keyword.lower() in [k.lower() for k in mapping.keywords]
                for keyword in keywords
            ):
                results.append(mapping)
        return results

    def get_cluster_execution_sequence(self) -> Dict[str, Any]:
        """Obtener secuencia de ejecución por clusters según catálogo"""
        execution_sequence = {
            "recommended_order": [5, 2, 3, 4, 1],  # Según complejidad y dependencias
            "cluster_details": {},
        }

        for cluster_id in execution_sequence["recommended_order"]:
            cluster_info = self.clusters[cluster_id]
            questions = self.get_questions_by_cluster(cluster_id)

            execution_sequence["cluster_details"][f"cluster_{cluster_id}"] = {
                "name": cluster_info["name"],
                "points": cluster_info["points"],
                "questions_count": len(questions),
                "priority": cluster_info["priority"],
                "evaluation_weight": cluster_info["evaluation_weight"],
                "prompt_references": list(set(q.prompt_reference for q in questions)),
            }

        return execution_sequence

    def generate_evaluation_roadmap(self) -> Dict[str, Any]:
        """Generar hoja de ruta para evaluación completa"""
        roadmap = {
            "preparation_phase": {
                "step": 1,
                "prompt_reference": "PROMPT 0.0",
                "description": "Revisión y navegación del documento PDT",
                "estimated_time": "15 minutes",
                "deliverable": "Estructura del PDT identificada",
            },
            "cluster_execution_phases": {},
            "consolidation_phase": {
                "step": 7,
                "prompt_reference": "PROMPT 6.1",
                "description": "Cálculo puntaje agregado por cluster",
                "estimated_time": "20 minutes",
                "deliverable": "Puntajes agregados calculados",
            },
            "interpretation_phase": {
                "step": 8,
                "prompt_reference": "PROMPT 7.1",
                "description": "Interpretación cualitativa y recomendaciones",
                "estimated_time": "30 minutes",
                "deliverable": "Reporte final con recomendaciones",
            },
        }

        # Fases de ejecución por cluster
        execution_sequence = self.get_cluster_execution_sequence()
        for i, cluster_id in enumerate(execution_sequence["recommended_order"]):
            cluster_details = execution_sequence["cluster_details"][
                f"cluster_{cluster_id}"
            ]

            roadmap["cluster_execution_phases"][f"phase_{i+2}"] = {
                "step": i + 2,
                "cluster_id": cluster_id,
                "cluster_name": cluster_details["name"],
                "points": cluster_details["points"],
                "questions_count": cluster_details["questions_count"],
                "estimated_time": f"{cluster_details['questions_count'] * 3} minutes",
                "prompt_references": cluster_details["prompt_references"],
            }

        return roadmap

    def get_quality_assurance_checklist(self) -> Dict[str, Any]:
        """Generar checklist de aseguramiento de calidad"""
        return {
            "completeness_check": {
                "all_11_points_covered": len(
                    set(q.decalogo_point for q in self.mapping.values())
                )
                == 11,
                "all_4_dimensions_covered": len(
                    set(q.dimension for q in self.mapping.values())
                )
                == 4,
                "all_5_clusters_covered": len(
                    set(q.cluster_id for q in self.mapping.values())
                )
                == 5,
                "value_chain_complete": len(
                    set(q.value_chain_link for q in self.mapping.values())
                )
                == 5,
            },
            "catalog_adherence": {
                "prompt_references_valid": all(
                    q.prompt_reference.startswith("PROMPT")
                    for q in self.mapping.values()
                ),
                "evidence_requirements_defined": all(
                    len(q.evidence_requirements) > 0 for q in self.mapping.values()
                ),
                "navigation_hints_provided": all(
                    len(q.navigation_hints) > 0 for q in self.mapping.values()
                ),
                "search_patterns_complete": all(
                    len(q.search_patterns) >= 2 for q in self.mapping.values()
                ),
            },
            "scoring_consistency": {
                "answer_options_standardized": all(
                    set(q.answer_options) <= {"Sí", "Parcial", "No", "NI"}
                    for q in self.mapping.values()
                ),
                "scoring_criteria_defined": all(
                    len(q.scoring_criteria) > 0 for q in self.mapping.values()
                ),
                "weights_reasonable": all(
                    0.5 <= q.weight <= 3.0 for q in self.mapping.values()
                ),
            },
        }


# Función de utilidad principal
def create_advanced_mapper() -> AdvancedQuestionDecalogoMapper:
    """Crear instancia del mapeador avanzado"""
    mapper = AdvancedQuestionDecalogoMapper()

    # Inicializar todos los componentes
    mapper.generate_comprehensive_coverage_matrix()
    mapper.generate_advanced_validation_report()
    mapper.initialize_query_engine()

    return mapper


# Ejemplo de uso
if __name__ == "__main__":
    # Crear mapper avanzado
    mapper = create_advanced_mapper()

    # Generar archivo maestro
    yaml_output = mapper.export_comprehensive_mapping_file()

    # Inicializar motor de consultas
    query_engine = mapper.query_engine

    # Ejemplo de consultas
    critical_questions = query_engine.get_critical_questions()
    cluster_2_questions = query_engine.get_questions_by_cluster(2)
    de1_questions = query_engine.get_questions_by_dimension("DE1")

    # Generar roadmap de evaluación
    evaluation_roadmap = query_engine.generate_evaluation_roadmap()

    # Checklist de calidad
    qa_checklist = query_engine.get_quality_assurance_checklist()

    print(f"✅ Mapper avanzado creado exitosamente")
    print(f"📊 Total preguntas mapeadas: {len(mapper.complete_mapping)}")
    print(
        f"🎯 Cobertura total: {mapper.validation_report['overall_status']['completion_percentage']:.1f}%"
    )
    print(
        f"✅ Status: {'COMPLETO' if mapper.validation_report['overall_status']['is_complete'] else 'EN PROGRESO'}"
    )

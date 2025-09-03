"""
Motor de Recomendaciones Inteligentes (IntelligentRecommendationEngine)
Genera recomendaciones personalizadas y accionables basadas en análisis de brechas
"""

import logging
import numpy as np
# # # from typing import Dict, List, Any, Optional, Tuple  # Module not found  # Module not found  # Module not found
# # # from dataclasses import asdict  # Module not found  # Module not found  # Module not found
# # # from datetime import datetime  # Module not found  # Module not found  # Module not found
import json
# # # from pathlib import Path  # Module not found  # Module not found  # Module not found

# # # from models import (  # Module not found  # Module not found  # Module not found
    AdaptiveScoringResults, IntelligentRecommendations, RecommendationItem,
    PDTContext, ComplianceStatus
)

logger = logging.getLogger(__name__)


class IntelligentRecommendationEngine:
    """
    Motor de Recomendaciones Inteligentes que genera consejos accionables
    basándose en:
    - Análisis de brechas entre puntuaciones reales y predichas
    - Importancia de características del AdaptiveScoringEngine
    - Contexto municipal específico
    - Patrones de mejores prácticas
    """
    
    def __init__(self, recommendations_db_path: str = "data/recommendations"):
        self.recommendations_db_path = Path(recommendations_db_path)
        self.recommendations_db_path.mkdir(parents=True, exist_ok=True)
        
        # Base de conocimientos de recomendaciones
        self.recommendation_templates = self._initialize_recommendation_templates()
        
        # Patrones de mejores prácticas por contexto
        self.best_practices_patterns = self._load_best_practices()
        
        # Matriz de impacto de recomendaciones
        self.impact_matrix = self._initialize_impact_matrix()
        
        # Configuración de priorización
        self.priority_weights = {
            'gap_magnitude': 0.30,
            'feature_importance': 0.25,
            'implementation_feasibility': 0.20,
            'expected_impact': 0.15,
            'context_relevance': 0.10
        }
        
        # Umbrales de confianza
        self.confidence_thresholds = {
            'high': 0.75,
            'medium': 0.50,
            'low': 0.25
        }
    
    def generate_recommendations(self, 
                               adaptive_results: AdaptiveScoringResults,
                               pdt_context: PDTContext) -> IntelligentRecommendations:
        """
        Genera recomendaciones inteligentes basadas en análisis de brechas
        """
        logger.info(f"Generating intelligent recommendations for {pdt_context.municipality_name}")
        
        # 1. Análisis de brechas
        gap_analysis = self._analyze_performance_gaps(adaptive_results)
        
        # 2. Identificar áreas de mejora prioritarias
        priority_areas = self._identify_priority_areas(gap_analysis, adaptive_results.feature_importance)
        
        # 3. Generar recomendaciones específicas
        recommendations = []
        
        for area in priority_areas:
            area_recommendations = self._generate_area_recommendations(
                area, gap_analysis, pdt_context, adaptive_results
            )
            recommendations.extend(area_recommendations)
        
        # 4. Priorizar y filtrar recomendaciones
        prioritized_recommendations = self._prioritize_recommendations(
            recommendations, pdt_context, adaptive_results
        )
        
        # 5. Calcular impacto esperado agregado
        total_impact = sum(rec.expected_impact for rec in prioritized_recommendations)
        high_priority_count = sum(1 for rec in prioritized_recommendations 
                                 if rec.priority == 'HIGH')
        
        # 6. Crear objeto de recomendaciones
        # Create summary for strongly-typed response  
# # #         from models import Priority, DifficultyLevel  # Module not found  # Module not found  # Module not found
        summary_by_priority = {}
        summary_by_category = {}
        summary_by_difficulty = {}
        
        for rec in prioritized_recommendations:
            # Count by priority
            if isinstance(rec.priority, str):
                priority_key = Priority(rec.priority)
            else:
                priority_key = rec.priority
            summary_by_priority[priority_key] = summary_by_priority.get(priority_key, 0) + 1
            
            # Count by category
            summary_by_category[rec.category] = summary_by_category.get(rec.category, 0) + 1
            
            # Count by difficulty
            if isinstance(rec.implementation_difficulty, str):
                difficulty_key = DifficultyLevel(rec.implementation_difficulty)
            else:
                difficulty_key = rec.implementation_difficulty
            summary_by_difficulty[difficulty_key] = summary_by_difficulty.get(difficulty_key, 0) + 1
        
# # #         from schemas.api_models import RecommendationSummary  # Module not found  # Module not found  # Module not found
        summary = RecommendationSummary(
            total_recommendations=len(prioritized_recommendations),
            by_priority=summary_by_priority,
            by_category=summary_by_category,
            by_difficulty=summary_by_difficulty,
            expected_total_impact=total_impact,
            average_confidence=sum(rec.confidence for rec in prioritized_recommendations) / 
                             max(1, len(prioritized_recommendations))
        )
        
        intelligent_recommendations = IntelligentRecommendations(
            municipality_id=pdt_context.municipality_code,
            municipality_name=pdt_context.municipality_name,
            summary=summary,
            recommendations=prioritized_recommendations,
            gap_analysis=gap_analysis,
            feature_importance=adaptive_results.feature_importance,
            generation_timestamp=datetime.now()
        )
        
        logger.info(f"Generated {len(prioritized_recommendations)} recommendations "
                   f"({high_priority_count} high priority)")
        
        return intelligent_recommendations
    
    def _analyze_performance_gaps(self, adaptive_results: AdaptiveScoringResults) -> Dict[str, float]:
        """Analiza brechas entre puntuaciones reales y predichas"""
        gap_analysis = {}
        
        # Brecha global
        global_gap = abs(adaptive_results.predicted_global_score - adaptive_results.global_score)
        gap_analysis['global'] = global_gap
        
        # Brechas por dimensión
        for dim_id, dim_score in adaptive_results.dimension_scores.items():
            gap = abs(dim_score.predicted_score - dim_score.raw_score)
            gap_analysis[f'dimension_{dim_id}'] = gap
        
        # Brechas por punto del Decálogo
        for point_id, point_score in adaptive_results.decalogo_scores.items():
            gap = abs(point_score.predicted_score - point_score.raw_score)
            gap_analysis[f'decalogo_{point_id}'] = gap
        
        # Identificar brechas de satisfacibilidad
        compliance_gaps = 0
        for dim_score in adaptive_results.dimension_scores.values():
            if dim_score.compliance_status == ComplianceStatus.NO_CUMPLE:
                compliance_gaps += 1
        
        for point_score in adaptive_results.decalogo_scores.values():
            if point_score.compliance_status == ComplianceStatus.NO_CUMPLE:
                compliance_gaps += 1
        
        gap_analysis['compliance_deficit'] = compliance_gaps / (
            len(adaptive_results.dimension_scores) + len(adaptive_results.decalogo_scores)
        )
        
        return gap_analysis
    
    def _identify_priority_areas(self, gap_analysis: Dict[str, float], 
                               feature_importance: Dict[str, float]) -> List[Dict[str, Any]]:
        """Identifica áreas de mejora prioritarias"""
        priority_areas = []
        
        # Ordenar brechas por magnitud
        sorted_gaps = sorted(gap_analysis.items(), key=lambda x: x[1], reverse=True)
        
        for gap_type, gap_magnitude in sorted_gaps[:10]:  # Top 10 brechas
            if gap_magnitude < 0.1:  # Umbral mínimo de brecha
                continue
            
            area = {
                'type': gap_type,
                'gap_magnitude': gap_magnitude,
                'priority_score': gap_magnitude
            }
            
            # Ajustar prioridad basada en importancia de características
            if gap_type in feature_importance:
                importance = feature_importance[gap_type]
                area['priority_score'] *= (1.0 + importance)
            
            priority_areas.append(area)
        
        # Ordenar por puntuación de prioridad
        priority_areas.sort(key=lambda x: x['priority_score'], reverse=True)
        
        return priority_areas[:5]  # Top 5 áreas prioritarias
    
    def _generate_area_recommendations(self, area: Dict[str, Any], 
                                     gap_analysis: Dict[str, float],
                                     pdt_context: PDTContext,
                                     adaptive_results: AdaptiveScoringResults) -> List[RecommendationItem]:
        """Genera recomendaciones específicas para un área de mejora"""
        recommendations = []
        area_type = area['type']
        gap_magnitude = area['gap_magnitude']
        
        # Seleccionar plantillas de recomendación apropiadas
        if area_type.startswith('dimension_'):
            dimension = area_type.split('_')[1]
            templates = self.recommendation_templates.get(f'dimension_{dimension}', [])
        elif area_type.startswith('decalogo_'):
            point = area_type.split('_')[1]
            templates = self.recommendation_templates.get(f'decalogo_{point}', [])
        else:
            templates = self.recommendation_templates.get('general', [])
        
        # Generar recomendaciones basadas en plantillas
        for i, template in enumerate(templates[:3]):  # Máximo 3 recomendaciones por área
            rec = self._instantiate_recommendation_template(
                template, area, pdt_context, adaptive_results, i
            )
            if rec:
                recommendations.append(rec)
        
        # Agregar recomendaciones contextuales específicas
        contextual_recs = self._generate_contextual_recommendations(
            area_type, pdt_context, gap_magnitude
        )
        recommendations.extend(contextual_recs)
        
        return recommendations
    
    def _instantiate_recommendation_template(self, template: Dict[str, Any], 
                                           area: Dict[str, Any],
                                           pdt_context: PDTContext,
                                           adaptive_results: AdaptiveScoringResults,
                                           index: int) -> Optional[RecommendationItem]:
        """Instancia una plantilla de recomendación con datos específicos"""
        try:
            # Calcular impacto esperado
            base_impact = template.get('base_impact', 0.1)
            gap_multiplier = min(2.0, area['gap_magnitude'] * 10)  # Amplificador de brecha
            expected_impact = base_impact * gap_multiplier
            
            # Determinar prioridad
            priority_score = area['priority_score']
            if priority_score > 0.5:
                priority = 'HIGH'
            elif priority_score > 0.25:
                priority = 'MEDIUM'
            else:
                priority = 'LOW'
            
            # Calcular confianza
            confidence = min(0.9, template.get('base_confidence', 0.7) * 
                           adaptive_results.model_confidence)
            
            # Personalizar descripción
            description = template['description'].format(
                municipality=pdt_context.municipality_name,
                population=pdt_context.population,
                gap=area['gap_magnitude']
            )
            
            # Generar pasos de acción contextualizados
            action_steps = []
            for step_template in template.get('action_steps', []):
                step = step_template.format(
                    municipality=pdt_context.municipality_name
                )
                action_steps.append(step)
            
            recommendation = RecommendationItem(
                recommendation_id=f"{area['type']}_{index}_{int(datetime.now().timestamp())}",
                category=template['category'],
                priority=priority,
                confidence=confidence,
                title=template['title'],
                description=description,
                expected_impact=expected_impact,
                implementation_difficulty=template['implementation_difficulty'],
                related_dimensions=template.get('related_dimensions', []),
                evidence_support=template.get('evidence_support', []),
                action_steps=action_steps
            )
            
            return recommendation
            
        except Exception as e:
            logger.warning(f"Failed to instantiate recommendation template: {str(e)}")
            return None
    
    def _generate_contextual_recommendations(self, area_type: str, 
                                           pdt_context: PDTContext, 
                                           gap_magnitude: float) -> List[RecommendationItem]:
        """Genera recomendaciones contextuales basadas en características municipales"""
        contextual_recs = []
        
        # Recomendaciones basadas en tamaño de municipio
        if pdt_context.population < 50000:
            if area_type.startswith('dimension_DE1'):
                contextual_recs.append(self._create_small_municipality_rec(
                    "participacion_ciudadana", pdt_context, gap_magnitude
                ))
        elif pdt_context.population > 200000:
            if area_type.startswith('dimension_DE2'):
                contextual_recs.append(self._create_large_municipality_rec(
                    "gestion_urbana", pdt_context, gap_magnitude
                ))
        
        # Recomendaciones basadas en índices de desarrollo
        if pdt_context.education_index < 0.6:
            contextual_recs.append(self._create_education_improvement_rec(
                pdt_context, gap_magnitude
            ))
        
        if pdt_context.poverty_index > 0.4:
            contextual_recs.append(self._create_poverty_reduction_rec(
                pdt_context, gap_magnitude
            ))
        
        return contextual_recs
    
    def _prioritize_recommendations(self, recommendations: List[RecommendationItem],
                                  pdt_context: PDTContext,
                                  adaptive_results: AdaptiveScoringResults) -> List[RecommendationItem]:
        """Prioriza y filtra recomendaciones basándose en múltiples criterios"""
        # Calcular puntuación de prioridad para cada recomendación
        for rec in recommendations:
            priority_score = 0.0
            
            # Factor de impacto esperado
            priority_score += rec.expected_impact * self.priority_weights['expected_impact']
            
            # Factor de confianza
            priority_score += rec.confidence * self.priority_weights['feature_importance']
            
            # Factor de factibilidad de implementación
            difficulty_score = {'EASY': 1.0, 'MEDIUM': 0.7, 'HARD': 0.3}.get(
                rec.implementation_difficulty, 0.5
            )
            priority_score += difficulty_score * self.priority_weights['implementation_feasibility']
            
            # Factor de relevancia contextual
            context_score = self._calculate_context_relevance(rec, pdt_context)
            priority_score += context_score * self.priority_weights['context_relevance']
            
            # Asignar puntuación calculada
            rec._priority_score = priority_score
        
        # Ordenar por puntuación de prioridad
        recommendations.sort(key=lambda x: getattr(x, '_priority_score', 0), reverse=True)
        
        # Filtrar por calidad mínima
        min_score = 0.3
        filtered_recs = [rec for rec in recommendations 
                        if getattr(rec, '_priority_score', 0) >= min_score]
        
        # Limitar número total de recomendaciones
        return filtered_recs[:15]
    
    def _calculate_context_relevance(self, recommendation: RecommendationItem,
                                   pdt_context: PDTContext) -> float:
        """Calcula la relevancia contextual de una recomendación"""
        relevance_score = 0.5  # Score base
        
        # Ajustar basado en categoría y contexto municipal
        category = recommendation.category
        
        if category == 'participacion' and pdt_context.governance_index < 0.6:
            relevance_score += 0.3
        elif category == 'desarrollo_economico' and pdt_context.gdp_per_capita < 5000000:
            relevance_score += 0.2
        elif category == 'medio_ambiente' and pdt_context.environmental_index < 0.5:
            relevance_score += 0.25
        elif category == 'infraestructura' and pdt_context.infrastructure_index < 0.6:
            relevance_score += 0.2
        
        # Ajustar basado en urbanización
        if pdt_context.urbanization_rate > 0.8 and category in ['transporte', 'servicios']:
            relevance_score += 0.15
        elif pdt_context.urbanization_rate < 0.4 and category in ['rural', 'agricultura']:
            relevance_score += 0.15
        
        return min(1.0, relevance_score)
    
    def _create_small_municipality_rec(self, rec_type: str, context: PDTContext,
                                     gap: float) -> RecommendationItem:
        """Crea recomendación específica para municipios pequeños"""
        return RecommendationItem(
            recommendation_id=f"small_mun_{rec_type}_{int(datetime.now().timestamp())}",
            category="participacion",
            priority="MEDIUM",
            confidence=0.7,
            title="Fortalecimiento de la Participación Ciudadana en Municipios Pequeños",
            description=f"Implementar mecanismos de participación ciudadana adaptados "
                       f"al tamaño y características de {context.municipality_name}",
            expected_impact=gap * 0.3,
            implementation_difficulty="EASY",
            related_dimensions=["DE1"],
            action_steps=[
                "Crear comités vecinales por sectores",
                "Implementar audiencias públicas mensuales",
                "Desarrollar canales digitales de participación"
            ]
        )
    
    def _create_large_municipality_rec(self, rec_type: str, context: PDTContext,
                                     gap: float) -> RecommendationItem:
        """Crea recomendación específica para municipios grandes"""
        return RecommendationItem(
            recommendation_id=f"large_mun_{rec_type}_{int(datetime.now().timestamp())}",
            category="gestion_urbana",
            priority="HIGH",
            confidence=0.8,
            title="Optimización de la Gestión Urbana Integrada",
            description=f"Desarrollar sistemas integrados de gestión urbana para "
                       f"mejorar la eficiencia en {context.municipality_name}",
            expected_impact=gap * 0.4,
            implementation_difficulty="MEDIUM",
            related_dimensions=["DE2", "DE3"],
            action_steps=[
                "Implementar sistema de información geográfica municipal",
                "Crear centro de operaciones urbanas",
                "Desarrollar indicadores de seguimiento territorial"
            ]
        )
    
    def _create_education_improvement_rec(self, context: PDTContext,
                                        gap: float) -> RecommendationItem:
        """Crea recomendación para mejora educativa"""
        return RecommendationItem(
            recommendation_id=f"education_{int(datetime.now().timestamp())}",
            category="educacion",
            priority="HIGH",
            confidence=0.85,
            title="Fortalecimiento del Sistema Educativo Municipal",
            description=f"Desarrollar estrategias integrales para mejorar "
                       f"la calidad educativa en {context.municipality_name}",
            expected_impact=gap * 0.5,
            implementation_difficulty="MEDIUM",
            related_dimensions=["DE1", "DE4"],
            action_steps=[
                "Crear programa de formación docente continua",
                "Mejorar infraestructura educativa existente",
                "Implementar tecnologías educativas digitales"
            ]
        )
    
    def _create_poverty_reduction_rec(self, context: PDTContext,
                                    gap: float) -> RecommendationItem:
        """Crea recomendación para reducción de pobreza"""
        return RecommendationItem(
            recommendation_id=f"poverty_{int(datetime.now().timestamp())}",
            category="desarrollo_social",
            priority="HIGH",
            confidence=0.9,
            title="Programa Integral de Reducción de Pobreza",
            description=f"Implementar estrategias multidimensionales para "
                       f"la reducción de la pobreza en {context.municipality_name}",
            expected_impact=gap * 0.6,
            implementation_difficulty="HARD",
            related_dimensions=["DE1", "DE2", "DE4"],
            action_steps=[
                "Crear programa de generación de ingresos",
                "Mejorar acceso a servicios básicos",
                "Desarrollar programas de inclusión social"
            ]
        )
    
    def _initialize_recommendation_templates(self) -> Dict[str, List[Dict[str, Any]]]:
        """Inicializa plantillas de recomendaciones por área"""
        return {
            'dimension_DE1': [
                {
                    'title': 'Fortalecimiento de la Participación Ciudadana',
                    'category': 'participacion',
                    'description': 'Mejorar los mecanismos de participación ciudadana en {municipality}',
                    'base_impact': 0.15,
                    'base_confidence': 0.8,
                    'implementation_difficulty': 'MEDIUM',
                    'related_dimensions': ['DE1'],
                    'action_steps': [
                        'Crear espacios de diálogo ciudadano',
                        'Implementar presupuesto participativo',
                        'Fortalecer organizaciones comunitarias'
                    ]
                }
            ],
            'dimension_DE2': [
                {
                    'title': 'Optimización de la Gestión Territorial',
                    'category': 'gestion_territorial',
                    'description': 'Mejorar la planificación y gestión del territorio en {municipality}',
                    'base_impact': 0.18,
                    'base_confidence': 0.75,
                    'implementation_difficulty': 'MEDIUM',
                    'related_dimensions': ['DE2'],
                    'action_steps': [
                        'Actualizar el Plan de Ordenamiento Territorial',
                        'Implementar sistema de información territorial',
                        'Fortalecer capacidades técnicas municipales'
                    ]
                }
            ],
            'dimension_DE3': [
                {
                    'title': 'Desarrollo de Capacidades Institucionales',
                    'category': 'fortalecimiento_institucional',
                    'description': 'Fortalecer las capacidades institucionales del municipio {municipality}',
                    'base_impact': 0.20,
                    'base_confidence': 0.85,
                    'implementation_difficulty': 'HARD',
                    'related_dimensions': ['DE3'],
                    'action_steps': [
                        'Implementar sistema de gestión por procesos',
                        'Capacitar funcionarios municipales',
                        'Modernizar sistemas de información'
                    ]
                }
            ],
            'dimension_DE4': [
                {
                    'title': 'Fortalecimiento Financiero Municipal',
                    'category': 'gestion_financiera',
                    'description': 'Mejorar la gestión financiera y sostenibilidad fiscal en {municipality}',
                    'base_impact': 0.16,
                    'base_confidence': 0.78,
                    'implementation_difficulty': 'MEDIUM',
                    'related_dimensions': ['DE4'],
                    'action_steps': [
                        'Optimizar el recaudo tributario',
                        'Implementar presupuesto por resultados',
                        'Diversificar fuentes de financiación'
                    ]
                }
            ],
            'general': [
                {
                    'title': 'Mejora General del PDT',
                    'category': 'planificacion',
                    'description': 'Fortalecer la planificación territorial integral en {municipality}',
                    'base_impact': 0.12,
                    'base_confidence': 0.70,
                    'implementation_difficulty': 'MEDIUM',
                    'related_dimensions': ['DE1', 'DE2', 'DE3', 'DE4'],
                    'action_steps': [
                        'Revisar y actualizar diagnósticos',
                        'Mejorar articulación entre programas',
                        'Fortalecer seguimiento y evaluación'
                    ]
                }
            ]
        }
    
    def _load_best_practices(self) -> Dict[str, Any]:
        """Carga patrones de mejores prácticas"""
        # En una implementación real, esto cargaría desde una base de datos
        return {
            'high_performance_municipalities': [],
            'success_patterns': {},
            'context_specific_practices': {}
        }
    
    def _initialize_impact_matrix(self) -> Dict[str, Dict[str, float]]:
        """Inicializa matriz de impacto de recomendaciones"""
        return {
            'participacion': {'DE1': 0.8, 'DE2': 0.3, 'DE3': 0.4, 'DE4': 0.2},
            'gestion_territorial': {'DE1': 0.2, 'DE2': 0.9, 'DE3': 0.5, 'DE4': 0.3},
            'fortalecimiento_institucional': {'DE1': 0.4, 'DE2': 0.6, 'DE3': 0.9, 'DE4': 0.7},
            'gestion_financiera': {'DE1': 0.2, 'DE2': 0.3, 'DE3': 0.5, 'DE4': 0.9}
        }
    
    def get_recommendation_analytics(self, 
                                   recommendations: IntelligentRecommendations) -> Dict[str, Any]:
        """Genera analíticas de las recomendaciones generadas"""
        analytics = {
            'total_recommendations': recommendations.total_recommendations,
            'high_priority_ratio': recommendations.high_priority_count / recommendations.total_recommendations,
            'expected_impact_distribution': {},
            'category_distribution': {},
            'difficulty_distribution': {},
            'dimension_coverage': {}
        }
        
        # Distribuciones
        for rec in recommendations.recommendations:
            # Por categoría
            category = rec.category
            analytics['category_distribution'][category] = analytics['category_distribution'].get(category, 0) + 1
            
            # Por dificultad
            difficulty = rec.implementation_difficulty
            analytics['difficulty_distribution'][difficulty] = analytics['difficulty_distribution'].get(difficulty, 0) + 1
            
            # Por dimensiones relacionadas
            for dim in rec.related_dimensions:
                analytics['dimension_coverage'][dim] = analytics['dimension_coverage'].get(dim, 0) + 1
        
        return analytics
"""
Sistema de Enriquecimiento Analítico para Validación Normativa
Genera insights avanzados y alimenta sistemas de análisis downstream
"""

# # # from typing import Dict, List, Any, Optional, Tuple, Set  # Module not found  # Module not found  # Module not found
# # # from dataclasses import dataclass, field  # Module not found  # Module not found  # Module not found
# # # from enum import Enum  # Module not found  # Module not found  # Module not found
# # # from datetime import datetime, timedelta  # Module not found  # Module not found  # Module not found
import json
import logging
import numpy as np
# # # from collections import defaultdict, Counter  # Module not found  # Module not found  # Module not found

# # # from normative_validator import (  # Module not found  # Module not found  # Module not found

# Mandatory Pipeline Contract Annotations
__phase__ = "T"
__code__ = "60T"
__stage_order__ = 9

    NormativeValidationResult, 
    ValidationFinding, 
    ComplianceStatus,
    NormativeLevel
)
# # # from models import SectionBlock, SectionType, QualityIndicators  # Module not found  # Module not found  # Module not found


class AnalyticsDimension(Enum):
    """Dimensiones analíticas disponibles"""
    TEMPORAL = "temporal"
    SECTORIAL = "sectorial"
    TERRITORIAL = "territorial"
    NORMATIVE_LEVEL = "normative_level"
    RISK_PROFILE = "risk_profile"
    COMPLIANCE_PATTERN = "compliance_pattern"


class InsightType(Enum):
    """Tipos de insights generados"""
    TREND = "trend"
    ANOMALY = "anomaly"
    CORRELATION = "correlation"
    PREDICTION = "prediction"
    RECOMMENDATION = "recommendation"
    BENCHMARK = "benchmark"


@dataclass
class AnalyticalInsight:
    """Insight analítico generado"""
    insight_id: str
    type: InsightType
    dimension: AnalyticsDimension
    title: str
    description: str
    value: Any
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    generated_at: datetime = field(default_factory=datetime.now)


@dataclass
class ComplianceMetrics:
    """Métricas de cumplimiento normativo"""
    overall_score: float
    by_level: Dict[str, float]
    by_section: Dict[str, float]
    risk_distribution: Dict[str, int]
    trend_indicators: Dict[str, float]


@dataclass
class PredictiveModel:
    """Modelo predictivo para riesgos normativos"""
    model_id: str
    features: List[str]
    predictions: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    model_performance: Dict[str, float]


class NormativeAnalyticsEngine:
    """Motor de análisis avanzado para validación normativa"""
    
    def __init__(self):
        self.historical_data: List[NormativeValidationResult] = []
        self.analytical_models = {}
        self.insight_generators = {}
        self.logger = logging.getLogger(__name__)
        
        # Inicializar generadores de insights
        self._initialize_insight_generators()
    
    def add_validation_result(self, result: NormativeValidationResult):
        """Agrega resultado de validación al histórico"""
        self.historical_data.append(result)
        
        # Mantener solo los últimos 1000 resultados para optimizar memoria
        if len(self.historical_data) > 1000:
            self.historical_data = self.historical_data[-1000:]
    
    async def generate_analytical_insights(self, 
                                         current_result: NormativeValidationResult,
                                         context: Dict[str, Any] = None) -> List[AnalyticalInsight]:
        """
        Genera insights analíticos basados en resultado actual e histórico
        
        Args:
            current_result: Resultado de validación actual
            context: Contexto adicional (metadatos del PDT, etc.)
        
        Returns:
            Lista de insights generados
        """
        
        insights = []
        context = context or {}
        
        # Agregar resultado actual al histórico
        self.add_validation_result(current_result)
        
        # Generar insights por dimensión
        for dimension in AnalyticsDimension:
            generator = self.insight_generators.get(dimension)
            if generator:
                try:
                    dimension_insights = await generator(current_result, context)
                    insights.extend(dimension_insights)
                except Exception as e:
                    self.logger.warning(f"Error generating insights for {dimension.value}: {e}")
        
        return insights
    
    def _initialize_insight_generators(self):
        """Inicializa generadores de insights por dimensión"""
        
        self.insight_generators = {
            AnalyticsDimension.TEMPORAL: self._generate_temporal_insights,
            AnalyticsDimension.SECTORIAL: self._generate_sectorial_insights,
            AnalyticsDimension.TERRITORIAL: self._generate_territorial_insights,
            AnalyticsDimension.NORMATIVE_LEVEL: self._generate_normative_level_insights,
            AnalyticsDimension.RISK_PROFILE: self._generate_risk_profile_insights,
            AnalyticsDimension.COMPLIANCE_PATTERN: self._generate_compliance_pattern_insights
        }
    
    async def _generate_temporal_insights(self, 
                                        current_result: NormativeValidationResult,
                                        context: Dict[str, Any]) -> List[AnalyticalInsight]:
        """Genera insights temporales"""
        
        insights = []
        
        if len(self.historical_data) < 2:
            return insights
        
        # Análisis de tendencia de cumplimiento
        recent_scores = [r.overall_compliance_score for r in self.historical_data[-30:]]
        
        if len(recent_scores) >= 5:
            # Calcular tendencia
            x = np.arange(len(recent_scores))
            trend = np.polyfit(x, recent_scores, 1)[0]
            
            if abs(trend) > 0.01:  # Cambio significativo
                trend_direction = "mejorando" if trend > 0 else "empeorando"
                
                insights.append(AnalyticalInsight(
                    insight_id=f"temporal_trend_{datetime.now().strftime('%Y%m%d')}",
                    type=InsightType.TREND,
                    dimension=AnalyticsDimension.TEMPORAL,
                    title="Tendencia de Cumplimiento Normativo",
                    description=f"El cumplimiento normativo está {trend_direction} con una tasa de {abs(trend):.3f} por documento",
                    value={
                        "trend_slope": trend,
                        "direction": trend_direction,
                        "recent_average": np.mean(recent_scores),
                        "data_points": len(recent_scores)
                    },
                    confidence=min(0.9, len(recent_scores) / 30.0)
                ))
        
        # Detección de anomalías temporales
        if len(recent_scores) >= 10:
            mean_score = np.mean(recent_scores[:-1])
            std_score = np.std(recent_scores[:-1])
            current_score = current_result.overall_compliance_score
            
            z_score = abs((current_score - mean_score) / std_score) if std_score > 0 else 0
            
            if z_score > 2:  # Anomalía significativa
                insights.append(AnalyticalInsight(
                    insight_id=f"temporal_anomaly_{current_result.pdt_id}",
                    type=InsightType.ANOMALY,
                    dimension=AnalyticsDimension.TEMPORAL,
                    title="Anomalía en Cumplimiento Normativo",
                    description=f"El score de cumplimiento ({current_score:.2f}) es significativamente diferente al patrón histórico (μ={mean_score:.2f}, σ={std_score:.2f})",
                    value={
                        "z_score": z_score,
                        "current_score": current_score,
                        "historical_mean": mean_score,
                        "historical_std": std_score,
                        "anomaly_type": "positive" if current_score > mean_score else "negative"
                    },
                    confidence=min(0.95, z_score / 3.0)
                ))
        
        return insights
    
    async def _generate_sectorial_insights(self, 
                                         current_result: NormativeValidationResult,
                                         context: Dict[str, Any]) -> List[AnalyticalInsight]:
        """Genera insights sectoriales"""
        
        insights = []
        
        # Análisis por tipo de sección más problemática
        section_issues = defaultdict(int)
        
        for finding in current_result.findings:
            if finding.status == ComplianceStatus.NON_COMPLIANT:
                # Inferir sección basada en el mensaje o check_id
                section = self._infer_section_from_finding(finding)
                if section:
                    section_issues[section] += 1
        
        if section_issues:
            most_problematic = max(section_issues.items(), key=lambda x: x[1])
            
            insights.append(AnalyticalInsight(
                insight_id=f"sectorial_analysis_{current_result.pdt_id}",
                type=InsightType.BENCHMARK,
                dimension=AnalyticsDimension.SECTORIAL,
                title="Análisis Sectorial de Cumplimiento",
                description=f"La sección '{most_problematic[0]}' presenta la mayor cantidad de incumplimientos ({most_problematic[1]})",
                value={
                    "problematic_sections": dict(section_issues),
                    "most_problematic_section": most_problematic[0],
                    "issues_count": most_problematic[1],
                    "total_sections_with_issues": len(section_issues)
                },
                confidence=0.8
            ))
        
        return insights
    
    async def _generate_territorial_insights(self, 
                                           current_result: NormativeValidationResult,
                                           context: Dict[str, Any]) -> List[AnalyticalInsight]:
        """Genera insights territoriales"""
        
        insights = []
        
        # Análisis por tipo de territorio (si está disponible en contexto)
        territory_type = context.get('territory_type', 'unknown')
        territory_size = context.get('population', 0)
        
        if territory_type != 'unknown':
            # Comparar con territorios similares
            similar_results = [
                r for r in self.historical_data
                if self._get_territory_type_from_metadata(r) == territory_type
            ]
            
            if len(similar_results) >= 5:
                similar_scores = [r.overall_compliance_score for r in similar_results]
                avg_similar = np.mean(similar_scores)
                current_score = current_result.overall_compliance_score
                
                performance = "superior" if current_score > avg_similar + 0.1 else \
                            "inferior" if current_score < avg_similar - 0.1 else "similar"
                
                insights.append(AnalyticalInsight(
                    insight_id=f"territorial_benchmark_{current_result.pdt_id}",
                    type=InsightType.BENCHMARK,
                    dimension=AnalyticsDimension.TERRITORIAL,
                    title="Comparación Territorial",
                    description=f"El desempeño es {performance} al promedio de territorios {territory_type} ({avg_similar:.2f})",
                    value={
                        "territory_type": territory_type,
                        "current_score": current_score,
                        "benchmark_score": avg_similar,
                        "comparison_sample_size": len(similar_results),
                        "performance_category": performance
                    },
                    confidence=min(0.9, len(similar_results) / 20.0)
                ))
        
        return insights
    
    async def _generate_normative_level_insights(self, 
                                               current_result: NormativeValidationResult,
                                               context: Dict[str, Any]) -> List[AnalyticalInsight]:
        """Genera insights por nivel normativo"""
        
        insights = []
        
        # Análisis de cumplimiento por nivel normativo
        level_performance = {}
        
        for level_name, level_data in current_result.summary_by_level.items():
            if level_data.get('total_checks', 0) > 0:
                level_performance[level_name] = level_data.get('compliance_rate', 0.0)
        
        if level_performance:
            # Identificar nivel más problemático
            worst_level = min(level_performance.items(), key=lambda x: x[1])
            best_level = max(level_performance.items(), key=lambda x: x[1])
            
            if worst_level[1] < 0.7:  # Umbral de problema
                insights.append(AnalyticalInsight(
                    insight_id=f"normative_level_analysis_{current_result.pdt_id}",
                    type=InsightType.RECOMMENDATION,
                    dimension=AnalyticsDimension.NORMATIVE_LEVEL,
                    title="Recomendación por Nivel Normativo",
                    description=f"Priorizar cumplimiento en nivel {worst_level[0]} (tasa actual: {worst_level[1]:.1%})",
                    value={
                        "level_performance": level_performance,
                        "worst_performing_level": worst_level[0],
                        "worst_performance_rate": worst_level[1],
                        "best_performing_level": best_level[0],
                        "improvement_needed": 0.8 - worst_level[1]  # Target 80%
                    },
                    confidence=0.85
                ))
        
        return insights
    
    async def _generate_risk_profile_insights(self, 
                                            current_result: NormativeValidationResult,
                                            context: Dict[str, Any]) -> List[AnalyticalInsight]:
        """Genera insights de perfil de riesgo"""
        
        insights = []
        
        # Calcular distribución de riesgos
        risk_distribution = Counter([f.risk_level for f in current_result.findings])
        total_findings = len(current_result.findings)
        
        if total_findings > 0:
            critical_ratio = risk_distribution.get('critical', 0) / total_findings
            high_ratio = risk_distribution.get('high', 0) / total_findings
            
            # Clasificar perfil de riesgo
            if critical_ratio > 0.1:
                risk_profile = "crítico"
                risk_score = 0.9
            elif high_ratio > 0.3:
                risk_profile = "alto"
                risk_score = 0.7
            elif high_ratio > 0.1:
                risk_profile = "moderado"
                risk_score = 0.5
            else:
                risk_profile = "bajo"
                risk_score = 0.2
            
            insights.append(AnalyticalInsight(
                insight_id=f"risk_profile_{current_result.pdt_id}",
                type=InsightType.PREDICTION,
                dimension=AnalyticsDimension.RISK_PROFILE,
                title="Perfil de Riesgo Normativo",
                description=f"El PDT presenta un perfil de riesgo {risk_profile} (score: {risk_score:.1f})",
                value={
                    "risk_profile": risk_profile,
                    "risk_score": risk_score,
                    "risk_distribution": dict(risk_distribution),
                    "critical_ratio": critical_ratio,
                    "high_ratio": high_ratio,
                    "total_findings": total_findings
                },
                confidence=0.8
            ))
            
            # Predicción de tiempo de resolución
            estimated_resolution_days = self._estimate_resolution_time(risk_distribution, context)
            
            if estimated_resolution_days:
                insights.append(AnalyticalInsight(
                    insight_id=f"resolution_prediction_{current_result.pdt_id}",
                    type=InsightType.PREDICTION,
                    dimension=AnalyticsDimension.RISK_PROFILE,
                    title="Predicción de Tiempo de Resolución",
                    description=f"Tiempo estimado para resolución de incumplimientos: {estimated_resolution_days} días",
                    value={
                        "estimated_days": estimated_resolution_days,
                        "confidence_interval": (estimated_resolution_days * 0.7, estimated_resolution_days * 1.3),
                        "factors_considered": ["risk_level", "finding_complexity", "historical_patterns"]
                    },
                    confidence=0.6
                ))
        
        return insights
    
    async def _generate_compliance_pattern_insights(self, 
                                                  current_result: NormativeValidationResult,
                                                  context: Dict[str, Any]) -> List[AnalyticalInsight]:
        """Genera insights de patrones de cumplimiento"""
        
        insights = []
        
        # Análisis de patrones recurrentes
        if len(self.historical_data) >= 10:
            # Identificar checks que fallan frecuentemente
            check_failure_rates = defaultdict(list)
            
            for result in self.historical_data[-20:]:  # Últimos 20 resultados
                failed_checks = {f.check_id for f in result.findings 
                               if f.status == ComplianceStatus.NON_COMPLIANT}
                
                all_checks = {f.check_id for f in result.findings}
                
                for check_id in all_checks:
                    check_failure_rates[check_id].append(1 if check_id in failed_checks else 0)
            
            # Identificar checks con alta tasa de falla
            high_failure_checks = []
            for check_id, failures in check_failure_rates.items():
                if len(failures) >= 5:  # Mínimo de datos
                    failure_rate = np.mean(failures)
                    if failure_rate > 0.4:  # >40% de falla
                        high_failure_checks.append((check_id, failure_rate))
            
            if high_failure_checks:
                # Ordenar por tasa de falla
                high_failure_checks.sort(key=lambda x: x[1], reverse=True)
                worst_check = high_failure_checks[0]
                
                insights.append(AnalyticalInsight(
                    insight_id=f"compliance_pattern_{datetime.now().strftime('%Y%m%d')}",
                    type=InsightType.CORRELATION,
                    dimension=AnalyticsDimension.COMPLIANCE_PATTERN,
                    title="Patrón de Incumplimientos Recurrentes",
                    description=f"Check '{worst_check[0]}' falla en {worst_check[1]:.1%} de los casos",
                    value={
                        "high_failure_checks": high_failure_checks,
                        "worst_performing_check": worst_check[0],
                        "worst_failure_rate": worst_check[1],
                        "analysis_sample_size": len(self.historical_data[-20:])
                    },
                    confidence=min(0.9, len(self.historical_data) / 50.0)
                ))
        
        return insights
    
    def _infer_section_from_finding(self, finding: ValidationFinding) -> Optional[str]:
        """Infiere la sección del documento basada en el finding"""
        
        check_id = finding.check_id.lower()
        message = finding.message.lower()
        
        if 'estructura' in check_id or 'diagnostico' in message:
            return 'diagnostico'
        elif 'participacion' in check_id or 'ciudadan' in message:
            return 'participacion'
        elif 'articulacion' in check_id or 'nacional' in message:
            return 'articulacion'
        elif 'presupuesto' in message or 'inversion' in message:
            return 'presupuesto'
        elif 'meta' in message or 'indicador' in message:
            return 'metas'
        
        return None
    
    def _get_territory_type_from_metadata(self, result: NormativeValidationResult) -> str:
        """Extrae tipo de territorio de metadatos"""
        # Lógica simplificada - se puede expandir
        return "municipal"  # Default
    
    def _estimate_resolution_time(self, risk_distribution: Counter, context: Dict[str, Any]) -> Optional[int]:
        """Estima tiempo de resolución basado en distribución de riesgos"""
        
        # Modelo simple basado en riesgos
        base_days = 5
        
        critical_count = risk_distribution.get('critical', 0)
        high_count = risk_distribution.get('high', 0)
        medium_count = risk_distribution.get('medium', 0)
        
        estimated_days = (base_days + 
                         critical_count * 10 +  # 10 días por riesgo crítico
                         high_count * 5 +       # 5 días por riesgo alto
                         medium_count * 2)      # 2 días por riesgo medio
        
        # Ajustar por capacidad organizacional (si está disponible)
        org_capacity = context.get('organizational_capacity', 1.0)
        estimated_days = int(estimated_days / org_capacity)
        
        return max(1, min(365, estimated_days))  # Entre 1 día y 1 año


class CompliancePredictor:
    """Predictor de cumplimiento normativo"""
    
    def __init__(self):
        self.models = {}
        self.feature_extractors = {}
        self.logger = logging.getLogger(__name__)
    
    def train_prediction_model(self, historical_data: List[NormativeValidationResult]):
        """Entrena modelo predictivo con datos históricos"""
        
        if len(historical_data) < 10:
            self.logger.warning("Insufficient data for model training")
            return
        
        # Extraer características y targets
        features = []
        targets = []
        
        for result in historical_data:
            feature_vector = self._extract_features(result)
            target_value = result.overall_compliance_score
            
            features.append(feature_vector)
            targets.append(target_value)
        
        # Modelo simple de regresión lineal (placeholder)
        # En implementación real usaría sklearn o similar
        features_array = np.array(features)
        targets_array = np.array(targets)
        
        # Calcular correlaciones simples
        correlations = {}
        for i, feature_name in enumerate(self._get_feature_names()):
            if features_array.shape[0] > 1:
                correlation = np.corrcoef(features_array[:, i], targets_array)[0, 1]
                if not np.isnan(correlation):
                    correlations[feature_name] = correlation
        
        self.models['simple_predictor'] = {
            'correlations': correlations,
            'mean_target': np.mean(targets_array),
            'std_target': np.std(targets_array)
        }
        
        self.logger.info("Prediction model trained successfully")
    
    def predict_compliance(self, result: NormativeValidationResult) -> Dict[str, Any]:
        """Predice score de cumplimiento futuro"""
        
        if 'simple_predictor' not in self.models:
            return {"error": "Model not trained"}
        
        model = self.models['simple_predictor']
        features = self._extract_features(result)
        feature_names = self._get_feature_names()
        
        # Predicción simple basada en correlaciones
        predicted_score = model['mean_target']
        confidence = 0.5
        
        if model['correlations'] and len(features) == len(feature_names):
            weighted_prediction = 0
            total_weight = 0
            
            for i, (feature_name, feature_value) in enumerate(zip(feature_names, features)):
                correlation = model['correlations'].get(feature_name, 0)
                weight = abs(correlation)
                
                # Contribución normalizada
                contribution = correlation * (feature_value - 0.5) * model['std_target']
                weighted_prediction += weight * contribution
                total_weight += weight
            
            if total_weight > 0:
                predicted_score += weighted_prediction / total_weight
                confidence = min(0.9, total_weight / len(feature_names))
        
        predicted_score = max(0.0, min(1.0, predicted_score))
        
        return {
            "predicted_score": predicted_score,
            "confidence": confidence,
            "current_score": result.overall_compliance_score,
            "prediction_change": predicted_score - result.overall_compliance_score,
            "model_version": "simple_predictor_v1.0"
        }
    
    def _extract_features(self, result: NormativeValidationResult) -> List[float]:
        """Extrae características numéricas del resultado"""
        
        features = []
        
        # Feature 1: Score actual
        features.append(result.overall_compliance_score)
        
        # Feature 2: Ratio de findings críticos
        critical_findings = len([f for f in result.findings if f.risk_level == "critical"])
        total_findings = len(result.findings)
        features.append(critical_findings / max(1, total_findings))
        
        # Feature 3: Ratio de non-compliant
        non_compliant = len([f for f in result.findings if f.status == ComplianceStatus.NON_COMPLIANT])
        features.append(non_compliant / max(1, total_findings))
        
        # Feature 4: Diversidad de niveles normativos
        levels_covered = len(result.summary_by_level)
        features.append(levels_covered / 5.0)  # Normalizar por max 5 niveles
        
        # Feature 5: Promedio de confianza en findings
        if result.findings:
            avg_confidence = np.mean([f.confidence for f in result.findings])
            features.append(avg_confidence)
        else:
            features.append(0.5)
        
        return features
    
    def _get_feature_names(self) -> List[str]:
        """Obtiene nombres de las características"""
        return [
            "current_compliance_score",
            "critical_findings_ratio",
            "non_compliant_ratio", 
            "normative_levels_diversity",
            "average_finding_confidence"
        ]


# Función principal de integración
async def enhance_analytics(validation_result: NormativeValidationResult,
                          context: Dict[str, Any] = None,
                          analytics_engine: Optional[NormativeAnalyticsEngine] = None) -> Dict[str, Any]:
    """
    Función de integración para enriquecimiento analítico
    
    Args:
        validation_result: Resultado de validación normativa
        context: Contexto adicional para análisis
        analytics_engine: Motor analítico (se crea uno nuevo si no se proporciona)
    
    Returns:
        Diccionario con insights y enriquecimientos analíticos
    """
    
    if analytics_engine is None:
        analytics_engine = NormativeAnalyticsEngine()
    
    try:
        # Generar insights analíticos
        insights = await analytics_engine.generate_analytical_insights(validation_result, context)
        
        # Crear predictor y generar predicción si hay suficientes datos
        predictor = CompliancePredictor()
        prediction = None
        
        if len(analytics_engine.historical_data) >= 10:
            predictor.train_prediction_model(analytics_engine.historical_data)
            prediction = predictor.predict_compliance(validation_result)
        
        # Generar métricas de cumplimiento
        compliance_metrics = _calculate_compliance_metrics(validation_result, analytics_engine.historical_data)
        
        return {
            "analytical_insights": insights,
            "predictive_analysis": prediction,
            "compliance_metrics": compliance_metrics,
            "enhancement_successful": True,
            "analytics_metadata": {
                "insights_count": len(insights),
                "historical_data_points": len(analytics_engine.historical_data),
                "analysis_timestamp": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        logging.error(f"Error in analytics enhancement: {e}")
        return {
            "analytical_insights": [],
            "predictive_analysis": None,
            "compliance_metrics": None,
            "enhancement_successful": False,
            "error_message": str(e)
        }


def _calculate_compliance_metrics(validation_result: NormativeValidationResult,
                                historical_data: List[NormativeValidationResult]) -> ComplianceMetrics:
    """Calcula métricas de cumplimiento detalladas"""
    
    # Métricas actuales
    overall_score = validation_result.overall_compliance_score
    by_level = {level: data.get('compliance_rate', 0.0) 
               for level, data in validation_result.summary_by_level.items()}
    
    risk_distribution = Counter([f.risk_level for f in validation_result.findings])
    
    # Métricas de tendencia si hay datos históricos
    trend_indicators = {}
    if len(historical_data) >= 5:
        recent_scores = [r.overall_compliance_score for r in historical_data[-10:]]
        if len(recent_scores) >= 2:
            trend_indicators["recent_average"] = np.mean(recent_scores)
            trend_indicators["improvement_rate"] = (recent_scores[-1] - recent_scores[0]) / len(recent_scores)
            trend_indicators["volatility"] = np.std(recent_scores)
    
    # Por sección (simplificado)
    by_section = {}
    
    return ComplianceMetrics(
        overall_score=overall_score,
        by_level=by_level,
        by_section=by_section,
        risk_distribution=dict(risk_distribution),
        trend_indicators=trend_indicators
    )
"""
"""
Analizador adaptativo que ajusta el comportamiento del sistema
en tiempo real basado en métricas y machine learning
"""

import asyncio
import json
import logging
import pickle
import statistics
# # # from collections import deque  # Module not found  # Module not found  # Module not found
# # # from dataclasses import dataclass, field  # Module not found  # Module not found  # Module not found
# # # from datetime import datetime, timedelta  # Module not found  # Module not found  # Module not found
# # # from enum import Enum  # Module not found  # Module not found  # Module not found
# # # from functools import total_ordering  # Module not found  # Module not found  # Module not found
# # # from pathlib import Path  # Module not found  # Module not found  # Module not found
# # # from typing import Any, Callable, Dict, List, Optional, Tuple  # Module not found  # Module not found  # Module not found
# # # from uuid import uuid4  # Module not found  # Module not found  # Module not found
import hashlib

import numpy as np
# # # from pydantic import BaseModel, Field  # Module not found  # Module not found  # Module not found

# Import audit logger for execution tracing
try:
# # #     from canonical_flow.analysis.audit_logger import get_audit_logger  # Module not found  # Module not found  # Module not found
except ImportError:
    # Fallback when audit logger is not available
    get_audit_logger = None

logger = logging.getLogger(__name__)


@total_ordering
@dataclass
class ProcessingResult:
    """Standardized response object for all analysis_nlp components."""
    
    status: str
    execution_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    artifact_paths: List[str] = field(default_factory=list)
    component_type: str = "adaptive_analyzer"
    stage: str = "analysis_nlp"
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __lt__(self, other):
        if not isinstance(other, ProcessingResult):
            return NotImplemented
        return self.timestamp < other.timestamp
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary with deterministic ordering."""
        return {
            "artifact_paths": sorted(self.artifact_paths),
            "component_type": self.component_type,
            "errors": sorted(self.errors),
            "execution_id": self.execution_id,
            "metadata": dict(sorted(self.metadata.items())),
            "stage": self.stage,
            "status": self.status,
            "timestamp": self.timestamp.isoformat(),
            "warnings": sorted(self.warnings)
        }


class AnalysisMode(str, Enum):
    """Modos de análisis adaptativo"""

    REACTIVE = "reactive"  # Solo reaccionar a problemas
    PREDICTIVE = "predictive"  # Predecir y prevenir problemas
    PROACTIVE = "proactive"  # Optimizar continuamente
    LEARNING = "learning"  # Aprender patrones y mejorar


class SystemState(str, Enum):
    """Estados del sistema identificados por el analizador"""

    OPTIMAL = "optimal"
    STABLE = "stable"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    RECOVERING = "recovering"
    UNKNOWN = "unknown"


class AdaptationAction(str, Enum):
    """Acciones de adaptación disponibles"""

    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    ADJUST_TIMEOUT = "adjust_timeout"
    ENABLE_CIRCUIT_BREAKER = "enable_circuit_breaker"
    ADJUST_BACKPRESSURE = "adjust_backpressure"
    REROUTE_TRAFFIC = "reroute_traffic"
    ENABLE_CACHING = "enable_caching"
    ADJUST_PARALLELISM = "adjust_parallelism"
    TRIGGER_MAINTENANCE = "trigger_maintenance"
    NO_ACTION = "no_action"


class MetricTrend(BaseModel):
    """Tendencia de una métrica específica"""

    metric_name: str
    current_value: float
    trend_direction: str  # "increasing", "decreasing", "stable"
    trend_strength: float  # 0.0 a 1.0
    prediction_next_hour: float
    confidence: float  # 0.0 a 1.0
    anomaly_score: float  # 0.0 a 1.0


class AdaptationRecommendation(BaseModel):
    """Recomendación de adaptación del sistema"""

    action: AdaptationAction
    component: str
    reason: str
    confidence: float
    expected_impact: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    priority: int = 0  # 0=low, 1=medium, 2=high, 3=critical
    estimated_execution_time: float = 0.0
    rollback_procedure: Optional[str] = None


class SystemAnalysis(BaseModel):
    """Análisis completo del estado del sistema"""

    timestamp: datetime = Field(default_factory=datetime.now)
    overall_state: SystemState
    confidence: float
    component_states: Dict[str, SystemState] = Field(default_factory=dict)
    metric_trends: List[MetricTrend] = Field(default_factory=list)
    anomalies_detected: List[str] = Field(default_factory=list)
    recommendations: List[AdaptationRecommendation] = Field(default_factory=list)
    predicted_issues: List[Dict[str, Any]] = Field(default_factory=list)
    health_score: float = 0.0  # 0.0 a 100.0


class AdaptiveModel:
    """Modelo adaptativo básico para análisis de tendencias"""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.data_points: deque = deque(maxlen=window_size)
        self.weights: Optional[np.ndarray] = None
        self.last_update = datetime.now()

    def add_data_point(self, value: float, timestamp: datetime = None):
        """Añade punto de datos al modelo"""
        if timestamp is None:
            timestamp = datetime.now()

        self.data_points.append({"value": value, "timestamp": timestamp})

        # Reentrenar modelo si tenemos suficientes datos
        if len(self.data_points) >= 10:
            self._train_model()

    def _train_model(self):
        """Entrena modelo simple de regresión lineal"""
        if len(self.data_points) < 3:
            return

        # Preparar datos
        values = np.array([dp["value"] for dp in self.data_points])
        timestamps = np.array([dp["timestamp"].timestamp() for dp in self.data_points])

        # Normalizar timestamps
        t_min = timestamps.min()
        x = (timestamps - t_min) / 3600.0  # horas desde inicio

        # Regresión lineal simple
        try:
            A = np.vstack([x, np.ones(len(x))]).T
            self.weights = np.linalg.lstsq(A, values, rcond=None)[0]
        except:
            self.weights = None

    def predict_trend(self, hours_ahead: float = 1.0) -> Tuple[float, float, str]:
        """
        Predice tendencia futura

        Returns:
            (predicted_value, confidence, trend_direction)
        """
        if self.weights is None or len(self.data_points) < 3:
            current_value = self.data_points[-1]["value"] if self.data_points else 0.0
            return current_value, 0.0, "unknown"

        # Calcular predicción
        current_time = datetime.now().timestamp()
        t_min = self.data_points[0]["timestamp"].timestamp()
        current_x = (current_time - t_min) / 3600.0
        future_x = current_x + hours_ahead

        predicted_value = self.weights[0] * future_x + self.weights[1]

        # Calcular confianza basada en R²
        values = np.array([dp["value"] for dp in self.data_points])
        x = np.array(
            [(dp["timestamp"].timestamp() - t_min) / 3600.0 for dp in self.data_points]
        )

        y_pred = self.weights[0] * x + self.weights[1]
        ss_res = np.sum((values - y_pred) ** 2)
        ss_tot = np.sum((values - np.mean(values)) ** 2)

        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        confidence = max(0.0, min(1.0, r_squared))

        # Determinar dirección de tendencia
        trend_slope = self.weights[0]
        if abs(trend_slope) < 0.01:  # threshold for "stable"
            trend_direction = "stable"
        elif trend_slope > 0:
            trend_direction = "increasing"
        else:
            trend_direction = "decreasing"

        return predicted_value, confidence, trend_direction

    def detect_anomaly(
        self, value: float, threshold: float = 2.0
    ) -> Tuple[bool, float]:
        """
        Detecta anomalía usando desviación estándar

        Returns:
            (is_anomaly, anomaly_score)
        """
        if len(self.data_points) < 10:
            return False, 0.0

        values = [dp["value"] for dp in self.data_points]
        mean_val = statistics.mean(values)
        std_val = statistics.stdev(values) if len(values) > 1 else 0.0

        if std_val == 0:
            return False, 0.0

        z_score = abs((value - mean_val) / std_val)
        anomaly_score = min(1.0, z_score / threshold)

        return z_score > threshold, anomaly_score


class AdaptiveAnalyzer:
    """
    Analizador principal que monitorea métricas del sistema
    y recomienda adaptaciones automáticas
    """

    def __init__(self, mode: AnalysisMode = AnalysisMode.PREDICTIVE):
        self.mode = mode
        self.models: Dict[str, AdaptiveModel] = {}
        self.thresholds: Dict[str, Dict[str, float]] = {}
        self.adaptation_handlers: Dict[str, Callable] = {}
        self.history: List[SystemAnalysis] = []
        self.last_analysis: Optional[SystemAnalysis] = None

        # Configuración
        self.analysis_interval = 30.0  # segundos
        self.metric_retention_hours = 24
        self.anomaly_threshold = 2.5
        self.min_confidence_for_action = 0.7

        # Estado interno
        self._running = False
        self._analysis_task: Optional[asyncio.Task] = None

        # Configuración por defecto de thresholds
        self._setup_default_thresholds()

    def _setup_default_thresholds(self):
        """Configura thresholds por defecto para métricas comunes"""
        self.thresholds = {
            "cpu_usage": {"warning": 70.0, "critical": 85.0, "optimal": 50.0},
            "memory_usage": {"warning": 75.0, "critical": 90.0, "optimal": 60.0},
            "response_time": {"warning": 5.0, "critical": 10.0, "optimal": 1.0},
            "error_rate": {"warning": 0.05, "critical": 0.10, "optimal": 0.01},
            "queue_size": {"warning": 1000, "critical": 5000, "optimal": 100},
            "throughput": {
                "warning": 50,  # requests/sec
                "critical": 10,
                "optimal": 100,
            },
        }

    def register_adaptation_handler(self, action: str, handler: Callable):
        """Registra handler para ejecutar adaptaciones"""
        self.adaptation_handlers[action] = handler
        logger.info(f"Registered adaptation handler: {action}")

    def update_thresholds(self, metric_name: str, thresholds: Dict[str, float]):
        """Actualiza thresholds para una métrica específica"""
        self.thresholds[metric_name] = thresholds
        logger.info(f"Updated thresholds for {metric_name}: {thresholds}")

    async def record_metric(
        self, metric_name: str, value: float, component: str = "system"
    ):
        """Registra una métrica para análisis"""

        # Crear modelo si no existe
        model_key = f"{component}.{metric_name}"
        if model_key not in self.models:
            self.models[model_key] = AdaptiveModel()

        # Añadir punto de datos
        self.models[model_key].add_data_point(value)

        # Si estamos en modo reactivo, analizar inmediatamente valores críticos
        if self.mode == AnalysisMode.REACTIVE:
            await self._check_immediate_action(metric_name, value, component)

    async def _check_immediate_action(
        self, metric_name: str, value: float, component: str
    ):
        """Verifica si se necesita acción inmediata en modo reactivo"""

        if metric_name in self.thresholds:
            thresholds = self.thresholds[metric_name]

            if "critical" in thresholds and value >= thresholds["critical"]:
                # Generar recomendación de emergencia
                recommendation = self._generate_emergency_recommendation(
                    metric_name, value, component
                )

                if recommendation:
                    await self._execute_recommendation(recommendation)

    def _generate_emergency_recommendation(
        self, metric_name: str, value: float, component: str
    ) -> Optional[AdaptationRecommendation]:
        """Genera recomendación de emergencia para valor crítico"""

        action_map = {
            "cpu_usage": AdaptationAction.SCALE_UP,
            "memory_usage": AdaptationAction.SCALE_UP,
            "response_time": AdaptationAction.ENABLE_CIRCUIT_BREAKER,
            "error_rate": AdaptationAction.ENABLE_CIRCUIT_BREAKER,
            "queue_size": AdaptationAction.ADJUST_BACKPRESSURE,
        }

        if metric_name in action_map:
            return AdaptationRecommendation(
                action=action_map[metric_name],
                component=component,
                reason=f"Critical {metric_name}: {value}",
                confidence=1.0,
                expected_impact="Prevent system failure",
                priority=3,
                parameters={"emergency": True, "metric_value": value},
            )

        return None

    async def start_continuous_analysis(self):
        """Inicia análisis continuo del sistema"""
        if self._running:
            return

        self._running = True
        self._analysis_task = asyncio.create_task(self._analysis_loop())
        logger.info(f"Started continuous analysis in {self.mode.value} mode")

    async def stop_continuous_analysis(self):
        """Detiene análisis continuo"""
        if not self._running:
            return

        self._running = False
        if self._analysis_task:
            self._analysis_task.cancel()
            try:
                await self._analysis_task
            except asyncio.CancelledError:
                pass

        logger.info("Stopped continuous analysis")

    async def _analysis_loop(self):
        """Loop principal de análisis continuo"""
        while self._running:
            try:
                analysis = await self.analyze_system()
                self.history.append(analysis)
                self.last_analysis = analysis

                # Mantener solo últimas 24 horas de análisis
                cutoff_time = datetime.now() - timedelta(hours=24)
                self.history = [a for a in self.history if a.timestamp >= cutoff_time]

                # Ejecutar recomendaciones automáticamente si es apropiado
                if self.mode in [AnalysisMode.PROACTIVE, AnalysisMode.PREDICTIVE]:
                    await self._execute_high_confidence_recommendations(analysis)

                await asyncio.sleep(self.analysis_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in analysis loop: {e}")
                await asyncio.sleep(self.analysis_interval)

    async def analyze_system(self) -> SystemAnalysis:
        """Realiza análisis completo del estado del sistema"""
        # Audit logging for component execution
        audit_logger = get_audit_logger() if get_audit_logger else None
        input_data = {
            "models_count": len(self.models),
            "analysis_mode": self.mode.value,
            "thresholds_configured": len(self.thresholds)
        }
        
        if audit_logger:
            with audit_logger.audit_component_execution("13A", input_data) as audit_ctx:
                result = await self._analyze_system_internal()
                audit_ctx.set_output({
                    "overall_state": result.overall_state.value,
                    "health_score": result.health_score,
                    "recommendations_count": len(result.recommendations),
                    "anomalies_detected": len(result.anomalies_detected)
                })
                return result
        else:
            return await self._analyze_system_internal()

    async def _analyze_system_internal(self) -> SystemAnalysis:
        """Internal implementation of system analysis."""
        analysis = SystemAnalysis(
            timestamp=datetime.now(), overall_state=SystemState.UNKNOWN, confidence=0.0
        )

        # Analizar tendencias de métricas
        metric_trends = []
        component_states = {}
        anomalies = []
        health_scores = []

        for model_key, model in self.models.items():
            if len(model.data_points) < 3:
                continue

            component, metric_name = model_key.split(".", 1)

            # Predecir tendencia
            predicted_value, confidence, trend_direction = model.predict_trend()
            current_value = model.data_points[-1]["value"]

            # Detectar anomalías
            is_anomaly, anomaly_score = model.detect_anomaly(current_value)

            if is_anomaly:
                anomalies.append(f"Anomaly in {model_key}: {current_value}")

            # Crear trend
            trend = MetricTrend(
                metric_name=metric_name,
                current_value=current_value,
                trend_direction=trend_direction,
                trend_strength=confidence,
                prediction_next_hour=predicted_value,
                confidence=confidence,
                anomaly_score=anomaly_score,
            )
            metric_trends.append(trend)

            # Calcular estado del componente
            component_state = self._calculate_component_state(
                metric_name, current_value, predicted_value
            )

            if component not in component_states:
                component_states[component] = component_state
            else:
                # Tomar el peor estado
                if self._state_severity(component_state) > self._state_severity(
                    component_states[component]
                ):
                    component_states[component] = component_state

            # Contribuir al health score
            health_score = self._calculate_health_score(metric_name, current_value)
            health_scores.append(health_score)

        # Consolidar estado general
        analysis.metric_trends = metric_trends
        analysis.component_states = component_states
        analysis.anomalies_detected = anomalies

        # Calcular estado general del sistema
        if component_states:
            worst_state = max(component_states.values(), key=self._state_severity)
            analysis.overall_state = worst_state

            # Calcular confianza basada en número de componentes analizados
            analysis.confidence = min(1.0, len(component_states) / 5.0)

        # Calcular health score general
        if health_scores:
            analysis.health_score = statistics.mean(health_scores)

        # Generar recomendaciones
        analysis.recommendations = self._generate_recommendations(analysis)

        # Predecir problemas futuros
        analysis.predicted_issues = self._predict_future_issues(analysis)

        logger.debug(
            f"System analysis completed: {analysis.overall_state.value} (confidence: {analysis.confidence:.2f})"
        )

        return analysis

    def _state_severity(self, state: SystemState) -> int:
        """Retorna severidad numérica del estado"""
        severity_map = {
            SystemState.OPTIMAL: 0,
            SystemState.STABLE: 1,
            SystemState.RECOVERING: 2,
            SystemState.DEGRADED: 3,
            SystemState.CRITICAL: 4,
            SystemState.UNKNOWN: 2,
        }
        return severity_map.get(state, 2)

    def _calculate_component_state(
        self, metric_name: str, current_value: float, predicted_value: float
    ) -> SystemState:
        """Calcula estado de componente basado en métrica"""

        if metric_name not in self.thresholds:
            return SystemState.UNKNOWN

        thresholds = self.thresholds[metric_name]

        # Verificar estado actual
        if "critical" in thresholds and current_value >= thresholds["critical"]:
            return SystemState.CRITICAL
        elif "warning" in thresholds and current_value >= thresholds["warning"]:
            return SystemState.DEGRADED
        elif "optimal" in thresholds and current_value <= thresholds["optimal"]:
            return SystemState.OPTIMAL

        # Si la predicción indica problema futuro
        if "critical" in thresholds and predicted_value >= thresholds["critical"]:
            return SystemState.DEGRADED

        return SystemState.STABLE

    def _calculate_health_score(self, metric_name: str, value: float) -> float:
        """Calcula score de salud para una métrica (0-100)"""

        if metric_name not in self.thresholds:
            return 50.0  # neutral

        thresholds = self.thresholds[metric_name]

        if "optimal" in thresholds and "critical" in thresholds:
            optimal = thresholds["optimal"]
            critical = thresholds["critical"]

            if value <= optimal:
                return 100.0
            elif value >= critical:
                return 0.0
            else:
                # Interpolación lineal entre optimal y critical
                range_size = critical - optimal
                distance_from_optimal = value - optimal
                score = 100.0 - (distance_from_optimal / range_size) * 100.0
                return max(0.0, min(100.0, score))

        return 50.0

    def _generate_recommendations(
        self, analysis: SystemAnalysis
    ) -> List[AdaptationRecommendation]:
        """Genera recomendaciones de adaptación basadas en análisis"""

        recommendations = []

        # Analizar cada componente
        for component, state in analysis.component_states.items():
            if state == SystemState.CRITICAL:
                recommendations.extend(
                    self._generate_critical_recommendations(component, analysis)
                )
            elif state == SystemState.DEGRADED:
                recommendations.extend(
                    self._generate_degraded_recommendations(component, analysis)
                )
            elif state == SystemState.OPTIMAL and self.mode == AnalysisMode.PROACTIVE:
                recommendations.extend(
                    self._generate_optimization_recommendations(component, analysis)
                )

        # Ordenar por prioridad
        recommendations.sort(key=lambda r: r.priority, reverse=True)

        return recommendations

    def _generate_critical_recommendations(
        self, component: str, analysis: SystemAnalysis
    ) -> List[AdaptationRecommendation]:
        """Genera recomendaciones para componentes en estado crítico"""

        recommendations = []

        # Buscar métricas críticas para este componente
        for trend in analysis.metric_trends:
            if trend.anomaly_score > 0.8:  # Alta anomalía
                if trend.metric_name in ["cpu_usage", "memory_usage"]:
                    recommendations.append(
                        AdaptationRecommendation(
                            action=AdaptationAction.SCALE_UP,
                            component=component,
                            reason=f"High {trend.metric_name}: {trend.current_value:.2f}",
                            confidence=trend.confidence,
                            expected_impact="Reduce resource pressure",
                            priority=3,
                            parameters={"target_instances": 2},
                        )
                    )

                elif trend.metric_name == "error_rate":
                    recommendations.append(
                        AdaptationRecommendation(
                            action=AdaptationAction.ENABLE_CIRCUIT_BREAKER,
                            component=component,
                            reason=f"High error rate: {trend.current_value:.3f}",
                            confidence=trend.confidence,
                            expected_impact="Prevent cascade failures",
                            priority=3,
                        )
                    )

                elif trend.metric_name == "response_time":
                    recommendations.append(
                        AdaptationRecommendation(
                            action=AdaptationAction.ADJUST_TIMEOUT,
                            component=component,
                            reason=f"High response time: {trend.current_value:.2f}s",
                            confidence=trend.confidence,
                            expected_impact="Prevent timeout errors",
                            priority=2,
                            parameters={"new_timeout": trend.current_value * 1.5},
                        )
                    )

        return recommendations

    def _generate_degraded_recommendations(
        self, component: str, analysis: SystemAnalysis
    ) -> List[AdaptationRecommendation]:
        """Genera recomendaciones para componentes degradados"""

        recommendations = []

        for trend in analysis.metric_trends:
            if trend.trend_direction == "increasing" and trend.confidence > 0.6:
                if trend.metric_name == "queue_size":
                    recommendations.append(
                        AdaptationRecommendation(
                            action=AdaptationAction.ADJUST_BACKPRESSURE,
                            component=component,
                            reason=f"Growing queue size: {trend.current_value}",
                            confidence=trend.confidence,
                            expected_impact="Prevent queue overflow",
                            priority=2,
                        )
                    )

                elif trend.metric_name == "response_time":
                    recommendations.append(
                        AdaptationRecommendation(
                            action=AdaptationAction.ENABLE_CACHING,
                            component=component,
                            reason="Response time trending upward",
                            confidence=trend.confidence,
                            expected_impact="Reduce response times",
                            priority=1,
                        )
                    )

        return recommendations

    def _generate_optimization_recommendations(
        self, component: str, analysis: SystemAnalysis
    ) -> List[AdaptationRecommendation]:
        """Genera recomendaciones de optimización para componentes óptimos"""

        recommendations = []

        # Solo en modo proactivo
        for trend in analysis.metric_trends:
            if trend.current_value < self.thresholds.get(trend.metric_name, {}).get(
                "optimal", float("inf")
            ):
                if trend.metric_name == "cpu_usage" and trend.current_value < 30:
                    recommendations.append(
                        AdaptationRecommendation(
                            action=AdaptationAction.SCALE_DOWN,
                            component=component,
                            reason="Low CPU utilization, can optimize resources",
                            confidence=trend.confidence,
                            expected_impact="Cost reduction without performance impact",
                            priority=0,
                        )
                    )

        return recommendations

    def _predict_future_issues(self, analysis: SystemAnalysis) -> List[Dict[str, Any]]:
        """Predice problemas futuros basado en tendencias"""

        predicted_issues = []

        for trend in analysis.metric_trends:
            if trend.trend_direction == "increasing" and trend.confidence > 0.7:
                # Estimar tiempo hasta threshold crítico
                if trend.metric_name in self.thresholds:
                    critical_threshold = self.thresholds[trend.metric_name].get(
                        "critical"
                    )

                    if (
                        critical_threshold
                        and trend.prediction_next_hour > critical_threshold
                    ):
                        predicted_issues.append(
                            {
                                "metric": trend.metric_name,
                                "issue": f"Will exceed critical threshold",
                                "predicted_time": "within 1 hour",
                                "confidence": trend.confidence,
                                "current_value": trend.current_value,
                                "predicted_value": trend.prediction_next_hour,
                                "threshold": critical_threshold,
                            }
                        )

        return predicted_issues

    async def _execute_high_confidence_recommendations(self, analysis: SystemAnalysis):
        """Ejecuta automáticamente recomendaciones de alta confianza"""

        for recommendation in analysis.recommendations:
            if (
                recommendation.confidence >= self.min_confidence_for_action
                and recommendation.priority >= 2
            ):  # medium o high priority
                await self._execute_recommendation(recommendation)

    async def _execute_recommendation(self, recommendation: AdaptationRecommendation):
        """Ejecuta una recomendación específica"""

        action_key = recommendation.action.value

        if action_key in self.adaptation_handlers:
            try:
                handler = self.adaptation_handlers[action_key]
                await handler(recommendation)

                logger.info(
                    f"Executed adaptation: {recommendation.action.value} for {recommendation.component}"
                )

            except Exception as e:
                logger.error(f"Failed to execute adaptation {action_key}: {e}")
        else:
            logger.warning(f"No handler registered for adaptation action: {action_key}")

    def get_system_summary(self) -> Dict[str, Any]:
        """Obtiene resumen del estado actual del sistema"""

        if not self.last_analysis:
            return {"status": "no_analysis_available"}

        analysis = self.last_analysis

        # Contar recomendaciones por prioridad
        rec_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        for rec in analysis.recommendations:
            rec_counts[rec.priority] += 1

        return {
            "timestamp": analysis.timestamp.isoformat(),
            "overall_state": analysis.overall_state.value,
            "health_score": analysis.health_score,
            "confidence": analysis.confidence,
            "components_count": len(analysis.component_states),
            "metrics_tracked": len(analysis.metric_trends),
            "anomalies_detected": len(analysis.anomalies_detected),
            "recommendations": {
                "critical": rec_counts[3],
                "high": rec_counts[2],
                "medium": rec_counts[1],
                "low": rec_counts[0],
            },
            "predicted_issues": len(analysis.predicted_issues),
            "analysis_mode": self.mode.value,
        }

    def get_detailed_metrics(self) -> Dict[str, Any]:
        """Obtiene métricas detalladas de todos los modelos"""

        detailed_metrics = {}

        for model_key, model in self.models.items():
            if len(model.data_points) > 0:
                current_value = model.data_points[-1]["value"]
                predicted_value, confidence, trend_direction = model.predict_trend()
                is_anomaly, anomaly_score = model.detect_anomaly(current_value)

                detailed_metrics[model_key] = {
                    "current_value": current_value,
                    "predicted_value": predicted_value,
                    "trend_direction": trend_direction,
                    "confidence": confidence,
                    "anomaly_detected": is_anomaly,
                    "anomaly_score": anomaly_score,
                    "data_points": len(model.data_points),
                }

        return detailed_metrics

    def _generate_stable_id(self, data: Any) -> str:
        """Generate stable, deterministic ID based on input data."""
        if isinstance(data, dict):
            # Sort keys for consistent ordering
            sorted_data = json.dumps(data, sort_keys=True, default=str)
        else:
            sorted_data = str(data)
        
        # Include component identifier for uniqueness
        content = f"{self.__class__.__name__}:{sorted_data}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _save_analysis_artifact(self, analysis: SystemAnalysis, execution_id: str) -> str:
        """Save analysis results to canonical_flow/analysis/ with _adaptive.json naming."""
        try:
            # Ensure output directory exists
            output_dir = Path("canonical_flow/analysis")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate deterministic filename
            timestamp_str = analysis.timestamp.strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp_str}_{execution_id}_adaptive.json"
            artifact_path = output_dir / filename
            
            # Prepare artifact data with deterministic structure
            artifact_data = {
                "analysis_id": execution_id,
                "timestamp": analysis.timestamp.isoformat(),
                "overall_state": analysis.overall_state.value,
                "confidence": analysis.confidence,
                "health_score": analysis.health_score,
                "component_states": dict(sorted(analysis.component_states.items())),
                "metric_trends": [
                    {
                        "metric_name": trend.metric_name,
                        "current_value": trend.current_value,
                        "trend_direction": trend.trend_direction,
                        "trend_strength": trend.trend_strength,
                        "prediction_next_hour": trend.prediction_next_hour,
                        "confidence": trend.confidence,
                        "anomaly_score": trend.anomaly_score
                    }
                    for trend in sorted(analysis.metric_trends, key=lambda x: x.metric_name)
                ],
                "anomalies_detected": sorted(analysis.anomalies_detected),
                "recommendations": [
                    {
                        "action": rec.action.value,
                        "component": rec.component,
                        "reason": rec.reason,
                        "confidence": rec.confidence,
                        "expected_impact": rec.expected_impact,
                        "priority": rec.priority,
                        "parameters": dict(sorted(rec.parameters.items()))
                    }
                    for rec in sorted(analysis.recommendations, key=lambda x: (x.priority, x.component))
                ],
                "predicted_issues": analysis.predicted_issues,
                "mode": self.mode.value,
                "component_type": "adaptive_analyzer",
                "stage": "analysis_nlp"
            }
            
            # Write artifact with deterministic JSON formatting
            with open(artifact_path, 'w', encoding='utf-8') as f:
                json.dump(artifact_data, f, indent=2, sort_keys=True, ensure_ascii=False)
            
            logger.info(f"Analysis artifact saved to: {artifact_path}")
            return str(artifact_path)
            
        except Exception as e:
            logger.error(f"Failed to save analysis artifact: {e}")
            raise

    def process(self, data: Optional[Dict[str, Any]] = None, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Standardized process function that accepts input parameters following established schema,
        performs strategy selection and component orchestration for adaptive analysis workflow.
        
        Args:
            data: Input data containing metrics and configuration parameters
            context: Processing context with metadata and execution parameters
            
        Returns:
            Standardized response object containing execution status, artifact paths, and error details
        """
        # Generate deterministic execution ID
        input_hash = self._generate_stable_id({"data": data, "context": context})
        execution_id = f"adaptive_{input_hash}"
        
        result = ProcessingResult(
            status="processing",
            execution_id=execution_id,
            component_type="adaptive_analyzer",
            stage="analysis_nlp"
        )
        
        try:
# # #             # Extract configuration from inputs  # Module not found  # Module not found  # Module not found
            config = {}
            if data:
                config.update(data)
            if context:
                config.update(context)
            
            # Strategy selection based on input parameters
            mode = config.get("analysis_mode", "predictive")
            if mode in [m.value for m in AnalysisMode]:
                self.mode = AnalysisMode(mode)
                logger.info(f"Analysis mode set to: {mode}")
            
            # Update configuration if provided
            if "thresholds" in config:
                for metric_name, thresholds in config["thresholds"].items():
                    self.update_thresholds(metric_name, thresholds)
                    
            # Component orchestration - process metrics if provided
            if "metrics" in config:
                metrics_data = config["metrics"]
                for metric_name, metric_info in metrics_data.items():
                    component = metric_info.get("component", "system")
                    value = metric_info.get("value")
                    if value is not None:
                        # Run metric recording synchronously
                        import asyncio
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            loop.run_until_complete(self.record_metric(metric_name, value, component))
                        finally:
                            loop.close()
            
            # Perform system analysis
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                analysis = loop.run_until_complete(self.analyze_system())
            finally:
                loop.close()
                
            # Save analysis artifact
            artifact_path = self._save_analysis_artifact(analysis, execution_id)
            result.artifact_paths.append(artifact_path)
            
            # Update result metadata
            result.metadata.update({
                "overall_state": analysis.overall_state.value,
                "confidence": analysis.confidence,
                "health_score": analysis.health_score,
                "recommendations_count": len(analysis.recommendations),
                "anomalies_count": len(analysis.anomalies_detected),
                "components_analyzed": len(analysis.component_states),
                "metrics_processed": len(analysis.metric_trends)
            })
            
            # Execute high-confidence recommendations if in proactive mode
            if self.mode in [AnalysisMode.PROACTIVE, AnalysisMode.PREDICTIVE]:
                high_confidence_recs = [
                    rec for rec in analysis.recommendations 
                    if rec.confidence >= self.min_confidence_for_action
                ]
                
                if high_confidence_recs:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        for rec in high_confidence_recs[:3]:  # Limit to top 3
                            loop.run_until_complete(self._execute_recommendation(rec))
                    except Exception as exec_error:
                        result.warnings.append(f"Recommendation execution error: {exec_error}")
                    finally:
                        loop.close()
            
            result.status = "success"
            logger.info(f"Adaptive analysis completed successfully: {execution_id}")
            
        except Exception as e:
            result.status = "error"
            result.errors.append(f"Processing failed: {str(e)}")
            logger.error(f"Adaptive analysis failed: {e}", exc_info=True)
            
            # Ensure we still have basic metadata even on failure
            if not result.metadata:
                result.metadata = {
                    "analysis_mode": self.mode.value,
                    "error_type": type(e).__name__
                }
        
        return result.to_dict()

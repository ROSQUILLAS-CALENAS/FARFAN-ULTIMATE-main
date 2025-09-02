"""
Colector de telemetría con OpenTelemetry para observabilidad completa
incluye métricas, trazas, logs y linaje de datos
"""

import asyncio
import json
import logging
import time
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

from pydantic import BaseModel, Field

# OpenTelemetry imports
try:
    from opentelemetry import metrics, trace
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
        OTLPMetricExporter,
    )
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.asyncio import AsyncioInstrumentor
    from opentelemetry.instrumentation.requests import RequestsInstrumentor
    from opentelemetry.propagate import set_global_textmap
    from opentelemetry.propagators.b3 import B3MultiFormat
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import Span, TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False

    # Mock classes for fallback
    class TracerProvider:
        pass

    class MeterProvider:
        pass

    class Span:
        pass


logger = logging.getLogger(__name__)


class TelemetryLevel(str, Enum):
    """Niveles de telemetría"""

    DISABLED = "disabled"
    BASIC = "basic"  # Solo métricas básicas
    STANDARD = "standard"  # Métricas + trazas principales
    DETAILED = "detailed"  # Todo + linaje de datos
    DEBUG = "debug"  # Todo + debug información


class DataLineageEvent(BaseModel):
    """Evento de linaje de datos"""

    event_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: datetime = Field(default_factory=datetime.now)
    operation: str
    input_datasets: List[str] = Field(default_factory=list)
    output_datasets: List[str] = Field(default_factory=list)
    transformation: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    quality_metrics: Dict[str, float] = Field(default_factory=dict)
    execution_context: Dict[str, str] = Field(default_factory=dict)


class MetricDefinition(BaseModel):
    """Definición de una métrica personalizada"""

    name: str
    description: str
    unit: str
    metric_type: str  # "counter", "gauge", "histogram"
    labels: List[str] = Field(default_factory=list)
    retention_days: int = 30


class TelemetryConfig(BaseModel):
    """Configuración del sistema de telemetría"""

    level: TelemetryLevel = TelemetryLevel.STANDARD
    service_name: str = "pdt-processing-engine"
    service_version: str = "1.0.0"

    # OpenTelemetry configuration
    otlp_endpoint: Optional[str] = None
    otlp_headers: Dict[str, str] = Field(default_factory=dict)

    # Sampling configuration
    trace_sampling_rate: float = 1.0
    metric_export_interval: float = 30.0

    # Data lineage configuration
    enable_data_lineage: bool = True
    lineage_retention_days: int = 90

    # Custom metrics
    custom_metrics: List[MetricDefinition] = Field(default_factory=list)

    # Resource attributes
    resource_attributes: Dict[str, str] = Field(default_factory=dict)


class TelemetryCollector:
    """
    Colector principal de telemetría que maneja métricas, trazas,
    logs y linaje de datos usando OpenTelemetry
    """

    def __init__(self, config: TelemetryConfig):
        self.config = config
        self.tracer_provider: Optional[TracerProvider] = None
        self.meter_provider: Optional[MeterProvider] = None
        self.tracer = None
        self.meter = None

        # Métricas personalizadas
        self.custom_metrics: Dict[str, Any] = {}

        # Historial de eventos
        self.data_lineage_events: List[DataLineageEvent] = []
        self.metric_history: Dict[str, List[Dict[str, Any]]] = {}

        # Estado interno
        self.initialized = False
        self._cleanup_task: Optional[asyncio.Task] = None

        # Inicializar si OpenTelemetry está disponible
        if OTEL_AVAILABLE and config.level != TelemetryLevel.DISABLED:
            self._initialize_opentelemetry()
        else:
            logger.warning("OpenTelemetry not available or telemetry disabled")

    def _initialize_opentelemetry(self):
        """Inicializa OpenTelemetry con la configuración especificada"""

        try:
            # Resource attributes
            resource_attributes = {
                "service.name": self.config.service_name,
                "service.version": self.config.service_version,
                **self.config.resource_attributes,
            }
            resource = Resource.create(resource_attributes)

            # Initialize tracer provider
            self.tracer_provider = TracerProvider(resource=resource)

            # Configure span exporter if endpoint provided
            if self.config.otlp_endpoint:
                otlp_exporter = OTLPSpanExporter(
                    endpoint=self.config.otlp_endpoint, headers=self.config.otlp_headers
                )
                span_processor = BatchSpanProcessor(otlp_exporter)
                self.tracer_provider.add_span_processor(span_processor)

            # Set global tracer provider
            trace.set_tracer_provider(self.tracer_provider)
            self.tracer = trace.get_tracer(__name__)

            # Initialize meter provider
            metric_readers = []
            if self.config.otlp_endpoint:
                otlp_metric_exporter = OTLPMetricExporter(
                    endpoint=self.config.otlp_endpoint, headers=self.config.otlp_headers
                )
                metric_reader = PeriodicExportingMetricReader(
                    exporter=otlp_metric_exporter,
                    export_interval_millis=int(
                        self.config.metric_export_interval * 1000
                    ),
                )
                metric_readers.append(metric_reader)

            self.meter_provider = MeterProvider(
                resource=resource, metric_readers=metric_readers
            )
            metrics.set_meter_provider(self.meter_provider)
            self.meter = metrics.get_meter(__name__)

            # Initialize custom metrics
            self._initialize_custom_metrics()

            # Set up propagators
            set_global_textmap(B3MultiFormat())

            # Instrument common libraries
            RequestsInstrumentor().instrument()
            AsyncioInstrumentor().instrument()

            self.initialized = True
            logger.info(f"OpenTelemetry initialized for {self.config.service_name}")

        except Exception as e:
            logger.error(f"Failed to initialize OpenTelemetry: {e}")
            self.initialized = False

    def _initialize_custom_metrics(self):
        """Inicializa métricas personalizadas definidas en configuración"""

        if not self.meter:
            return

        for metric_def in self.config.custom_metrics:
            try:
                if metric_def.metric_type == "counter":
                    metric = self.meter.create_counter(
                        name=metric_def.name,
                        description=metric_def.description,
                        unit=metric_def.unit,
                    )
                elif metric_def.metric_type == "gauge":
                    metric = self.meter.create_up_down_counter(
                        name=metric_def.name,
                        description=metric_def.description,
                        unit=metric_def.unit,
                    )
                elif metric_def.metric_type == "histogram":
                    metric = self.meter.create_histogram(
                        name=metric_def.name,
                        description=metric_def.description,
                        unit=metric_def.unit,
                    )
                else:
                    logger.warning(f"Unknown metric type: {metric_def.metric_type}")
                    continue

                self.custom_metrics[metric_def.name] = metric
                logger.debug(f"Initialized custom metric: {metric_def.name}")

            except Exception as e:
                logger.error(f"Failed to initialize metric {metric_def.name}: {e}")

    async def start_span(
        self, name: str, attributes: Dict[str, Any] = None
    ) -> Optional[Span]:
        """
        Inicia un nuevo span de OpenTelemetry

        Args:
            name: Nombre del span
            attributes: Atributos adicionales

        Returns:
            Span object o None si no está disponible
        """
        if not self.initialized or not self.tracer:
            return None

        span = self.tracer.start_span(name)

        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, str(value))

        return span

    async def record_metric(
        self, name: str, value: float, labels: Dict[str, str] = None
    ):
        """
        Registra una métrica

        Args:
            name: Nombre de la métrica
            value: Valor de la métrica
            labels: Labels adicionales
        """
        timestamp = datetime.now()

        # Almacenar en historial local
        if name not in self.metric_history:
            self.metric_history[name] = []

        self.metric_history[name].append(
            {"timestamp": timestamp, "value": value, "labels": labels or {}}
        )

        # Mantener solo últimos datos según configuración
        retention_limit = timestamp - timedelta(days=30)  # default retention
        self.metric_history[name] = [
            m for m in self.metric_history[name] if m["timestamp"] >= retention_limit
        ]

        # Enviar a OpenTelemetry si está disponible
        if self.initialized and name in self.custom_metrics:
            try:
                metric = self.custom_metrics[name]

                # Convertir labels a formato OpenTelemetry
                otel_labels = {}
                if labels:
                    otel_labels = {k: str(v) for k, v in labels.items()}

                # Registrar métrica según tipo
                if hasattr(metric, "add"):
                    metric.add(value, otel_labels)
                elif hasattr(metric, "record"):
                    metric.record(value, otel_labels)

            except Exception as e:
                logger.error(f"Failed to record metric {name}: {e}")

    async def record_data_lineage(self, event: DataLineageEvent):
        """
        Registra evento de linaje de datos

        Args:
            event: Evento de linaje de datos
        """
        if self.config.level not in [TelemetryLevel.DETAILED, TelemetryLevel.DEBUG]:
            return

        # Almacenar evento
        self.data_lineage_events.append(event)

        # Mantener solo eventos dentro del período de retención
        retention_limit = datetime.now() - timedelta(
            days=self.config.lineage_retention_days
        )
        self.data_lineage_events = [
            e for e in self.data_lineage_events if e.timestamp >= retention_limit
        ]

        # Crear span para linaje si está disponible
        if self.initialized and self.tracer:
            with self.tracer.start_as_current_span("data_lineage") as span:
                span.set_attribute("lineage.operation", event.operation)
                span.set_attribute("lineage.input_datasets", str(event.input_datasets))
                span.set_attribute(
                    "lineage.output_datasets", str(event.output_datasets)
                )
                span.set_attribute("lineage.transformation", event.transformation)

                # Añadir métricas de calidad como atributos
                for metric_name, metric_value in event.quality_metrics.items():
                    span.set_attribute(f"lineage.quality.{metric_name}", metric_value)

        logger.debug(f"Recorded data lineage event: {event.operation}")

    async def trace_function(
        self, func_name: str, func: Callable, *args, **kwargs
    ) -> Any:
        """
        Ejecuta función con tracing automático

        Args:
            func_name: Nombre para el span
            func: Función a ejecutar
            *args, **kwargs: Argumentos para la función

        Returns:
            Resultado de la función
        """
        if not self.initialized or not self.tracer:
            # Ejecutar sin tracing
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)

        with self.tracer.start_as_current_span(func_name) as span:
            try:
                start_time = time.time()

                # Añadir atributos básicos
                span.set_attribute("function.name", func_name)
                span.set_attribute("function.args_count", len(args))
                span.set_attribute("function.kwargs_count", len(kwargs))

                # Ejecutar función
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)

                # Registrar métricas de ejecución
                execution_time = time.time() - start_time
                span.set_attribute("function.execution_time", execution_time)
                span.set_attribute("function.success", True)

                return result

            except Exception as e:
                # Registrar error
                span.set_attribute("function.success", False)
                span.set_attribute("function.error", str(e))
                span.record_exception(e)

                # Re-lanzar excepción
                raise e

    async def record_workflow_started(self, workflow_id: str, workflow_name: str):
        """Registra inicio de workflow"""
        await self.record_metric(
            "workflow.started",
            1,
            {"workflow_id": workflow_id, "workflow_name": workflow_name},
        )

    async def record_workflow_completed(self, workflow_id: str, duration: timedelta):
        """Registra finalización exitosa de workflow"""
        await self.record_metric("workflow.completed", 1, {"workflow_id": workflow_id})

        await self.record_metric(
            "workflow.duration", duration.total_seconds(), {"workflow_id": workflow_id}
        )

    async def record_workflow_failed(self, workflow_id: str, error: str):
        """Registra fallo de workflow"""
        await self.record_metric(
            "workflow.failed",
            1,
            {"workflow_id": workflow_id, "error_type": type(error).__name__},
        )

    async def record_event_processed(
        self, event_type: str, handler_name: str, processing_time: float
    ):
        """Registra procesamiento de evento"""
        await self.record_metric(
            "event.processed", 1, {"event_type": event_type, "handler": handler_name}
        )

        await self.record_metric(
            "event.processing_time",
            processing_time,
            {"event_type": event_type, "handler": handler_name},
        )

    async def record_event_failed(self, event_type: str, handler_name: str, error: str):
        """Registra fallo en procesamiento de evento"""
        await self.record_metric(
            "event.failed",
            1,
            {
                "event_type": event_type,
                "handler": handler_name,
                "error_type": type(error).__name__,
            },
        )

    async def record_circuit_breaker_opened(self, circuit_name: str):
        """Registra apertura de circuit breaker"""
        await self.record_metric(
            "circuit_breaker.opened", 1, {"circuit_name": circuit_name}
        )

    async def record_circuit_breaker_closed(self, circuit_name: str):
        """Registra cierre de circuit breaker"""
        await self.record_metric(
            "circuit_breaker.closed", 1, {"circuit_name": circuit_name}
        )

    async def record_backpressure_metrics(self, metrics: Dict[str, Any]):
        """Registra métricas de backpressure"""
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                await self.record_metric(f"backpressure.{metric_name}", value)

    async def record_migration_phase_change(
        self, component: str, from_phase: str, to_phase: str
    ):
        """Registra cambio de fase en migración"""
        await self.record_metric(
            "migration.phase_change",
            1,
            {"component": component, "from_phase": from_phase, "to_phase": to_phase},
        )

    async def record_migration_rollback(
        self, component: str, from_phase: str, reason: str
    ):
        """Registra rollback de migración"""
        await self.record_metric(
            "migration.rollback",
            1,
            {"component": component, "from_phase": from_phase, "reason": reason},
        )

    async def record_traffic_percentage_change(
        self, component: str, old_percentage: float, new_percentage: float
    ):
        """Registra cambio de porcentaje de tráfico"""
        await self.record_metric(
            "migration.traffic_percentage", new_percentage, {"component": component}
        )

    async def record_adaptive_action(
        self, action: str, component: str, reason: str, confidence: float
    ):
        """Registra acción adaptativa ejecutada"""
        await self.record_metric(
            "adaptive.action_executed",
            1,
            {"action": action, "component": component, "reason": reason},
        )

        await self.record_metric(
            "adaptive.action_confidence",
            confidence,
            {"action": action, "component": component},
        )

    def get_metric_history(
        self, metric_name: str, since: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Obtiene historial de una métrica

        Args:
            metric_name: Nombre de la métrica
            since: Fecha desde la que obtener datos

        Returns:
            Lista de puntos de datos
        """
        if metric_name not in self.metric_history:
            return []

        history = self.metric_history[metric_name]

        if since:
            history = [h for h in history if h["timestamp"] >= since]

        return history

    def get_data_lineage(
        self, dataset_name: str = None, operation: str = None
    ) -> List[DataLineageEvent]:
        """
        Obtiene eventos de linaje de datos

        Args:
            dataset_name: Filtrar por dataset
            operation: Filtrar por operación

        Returns:
            Lista de eventos de linaje
        """
        events = self.data_lineage_events.copy()

        if dataset_name:
            events = [
                e
                for e in events
                if dataset_name in e.input_datasets or dataset_name in e.output_datasets
            ]

        if operation:
            events = [e for e in events if e.operation == operation]

        return events

    async def get_telemetry_summary(self) -> Dict[str, Any]:
        """Obtiene resumen del estado de telemetría"""

        # Contar métricas por tipo
        metric_counts = {}
        for metric_name, history in self.metric_history.items():
            if history:
                metric_type = "custom"  # Podríamos inferir tipo desde configuración
                metric_counts[metric_type] = metric_counts.get(metric_type, 0) + 1

        # Estadísticas de linaje
        lineage_operations = {}
        for event in self.data_lineage_events:
            lineage_operations[event.operation] = (
                lineage_operations.get(event.operation, 0) + 1
            )

        return {
            "telemetry_level": self.config.level.value,
            "initialized": self.initialized,
            "service_name": self.config.service_name,
            "service_version": self.config.service_version,
            "metrics": {
                "total_metrics": len(self.metric_history),
                "custom_metrics": len(self.custom_metrics),
                "metrics_by_type": metric_counts,
            },
            "data_lineage": {
                "total_events": len(self.data_lineage_events),
                "operations": lineage_operations,
                "retention_days": self.config.lineage_retention_days,
            },
            "configuration": {
                "otlp_endpoint": self.config.otlp_endpoint is not None,
                "trace_sampling_rate": self.config.trace_sampling_rate,
                "metric_export_interval": self.config.metric_export_interval,
            },
        }

    async def start_cleanup_task(self):
        """Inicia tarea de limpieza periódica"""
        if self._cleanup_task is not None:
            return

        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("Started telemetry cleanup task")

    async def stop_cleanup_task(self):
        """Detiene tarea de limpieza"""
        if self._cleanup_task is not None:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None

        logger.info("Stopped telemetry cleanup task")

    async def _cleanup_loop(self):
        """Loop de limpieza periódica de datos antiguos"""
        while True:
            try:
                await asyncio.sleep(3600)  # Ejecutar cada hora
                await self._cleanup_old_data()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in telemetry cleanup: {e}")

    async def _cleanup_old_data(self):
        """Limpia datos antiguos basado en políticas de retención"""
        now = datetime.now()

        # Limpiar historial de métricas
        for metric_name in list(self.metric_history.keys()):
            # Usar retención por defecto de 30 días
            retention_limit = now - timedelta(days=30)

            old_count = len(self.metric_history[metric_name])
            self.metric_history[metric_name] = [
                m
                for m in self.metric_history[metric_name]
                if m["timestamp"] >= retention_limit
            ]
            new_count = len(self.metric_history[metric_name])

            if old_count != new_count:
                logger.debug(
                    f"Cleaned up {old_count - new_count} old entries for metric {metric_name}"
                )

        # Limpiar eventos de linaje
        retention_limit = now - timedelta(days=self.config.lineage_retention_days)
        old_lineage_count = len(self.data_lineage_events)
        self.data_lineage_events = [
            e for e in self.data_lineage_events if e.timestamp >= retention_limit
        ]
        new_lineage_count = len(self.data_lineage_events)

        if old_lineage_count != new_lineage_count:
            logger.debug(
                f"Cleaned up {old_lineage_count - new_lineage_count} old lineage events"
            )

    async def shutdown(self):
        """Cierre limpio del sistema de telemetría"""
        logger.info("Shutting down telemetry collector...")

        # Parar tareas de limpieza
        await self.stop_cleanup_task()

        # Flush pending telemetry data
        if self.tracer_provider and hasattr(self.tracer_provider, "force_flush"):
            self.tracer_provider.force_flush()

        if self.meter_provider and hasattr(self.meter_provider, "force_flush"):
            self.meter_provider.force_flush()

        logger.info("Telemetry collector shutdown completed")


def create_telemetry_collector(
    service_name: str = "pdt-processing-engine",
    level: TelemetryLevel = TelemetryLevel.STANDARD,
    otlp_endpoint: Optional[str] = None,
) -> TelemetryCollector:
    """
    Factory function para crear colector de telemetría

    Args:
        service_name: Nombre del servicio
        level: Nivel de telemetría deseado
        otlp_endpoint: Endpoint OTLP opcional

    Returns:
        Instancia configurada de TelemetryCollector
    """

    config = TelemetryConfig(
        service_name=service_name, level=level, otlp_endpoint=otlp_endpoint
    )

    # Añadir métricas estándar del sistema
    standard_metrics = [
        MetricDefinition(
            name="workflow.started",
            description="Number of workflows started",
            unit="1",
            metric_type="counter",
            labels=["workflow_id", "workflow_name"],
        ),
        MetricDefinition(
            name="workflow.completed",
            description="Number of workflows completed",
            unit="1",
            metric_type="counter",
            labels=["workflow_id"],
        ),
        MetricDefinition(
            name="workflow.failed",
            description="Number of workflows failed",
            unit="1",
            metric_type="counter",
            labels=["workflow_id", "error_type"],
        ),
        MetricDefinition(
            name="workflow.duration",
            description="Workflow execution duration",
            unit="s",
            metric_type="histogram",
            labels=["workflow_id"],
        ),
        MetricDefinition(
            name="event.processed",
            description="Number of events processed",
            unit="1",
            metric_type="counter",
            labels=["event_type", "handler"],
        ),
        MetricDefinition(
            name="event.failed",
            description="Number of events failed",
            unit="1",
            metric_type="counter",
            labels=["event_type", "handler", "error_type"],
        ),
        MetricDefinition(
            name="circuit_breaker.opened",
            description="Number of circuit breaker opens",
            unit="1",
            metric_type="counter",
            labels=["circuit_name"],
        ),
        MetricDefinition(
            name="adaptive.action_executed",
            description="Number of adaptive actions executed",
            unit="1",
            metric_type="counter",
            labels=["action", "component", "reason"],
        ),
    ]

    config.custom_metrics = standard_metrics

    return TelemetryCollector(config)

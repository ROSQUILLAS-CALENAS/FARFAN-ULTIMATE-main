"""
Prometheus-based metrics collection and monitoring system
"""

import asyncio
import logging
import threading
import time
# # # from concurrent.futures import ThreadPoolExecutor  # Module not found  # Module not found  # Module not found
# # # from dataclasses import dataclass, field  # Module not found  # Module not found  # Module not found
# # # from datetime import datetime, timedelta  # Module not found  # Module not found  # Module not found
# # # from typing import Any, Dict, List, Optional, Union  # Module not found  # Module not found  # Module not found

import psutil

try:
# # #     from prometheus_async.aio import MetricsHTTPServer  # Module not found  # Module not found  # Module not found
# # #     from prometheus_client import (  # Module not found  # Module not found  # Module not found
        CONTENT_TYPE_LATEST,
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
    )

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    Counter = Histogram = Gauge = CollectorRegistry = None

import redis.asyncio as redis
import ray


@dataclass
class SystemMetrics:
    """System-level metrics snapshot"""

    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, float] = field(default_factory=dict)
    process_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "cpu_usage": self.cpu_usage,
            "memory_usage": self.memory_usage,
            "disk_usage": self.disk_usage,
            "network_io": self.network_io,
            "process_count": self.process_count,
        }


@dataclass
class ProcessingMetrics:
    """Processing pipeline metrics"""

    timestamp: datetime
    throughput: float  # documents per second
    latency_p50: float  # milliseconds
    latency_p95: float
    latency_p99: float
    error_rate: float  # percentage
    active_workers: int
    queue_depth: int
    processing_time_avg: float
    retry_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "throughput": self.throughput,
            "latency_p50": self.latency_p50,
            "latency_p95": self.latency_p95,
            "latency_p99": self.latency_p99,
            "error_rate": self.error_rate,
            "active_workers": self.active_workers,
            "queue_depth": self.queue_depth,
            "processing_time_avg": self.processing_time_avg,
            "retry_count": self.retry_count,
        }


@dataclass
class WorkloadMetrics:
    """Workload characteristics and patterns"""

    timestamp: datetime
    document_size_avg: float  # MB
    document_complexity: float  # 0-1 scale
    ocr_ratio: float  # percentage requiring OCR
    table_density: float  # tables per page
    language_distribution: Dict[str, float]
    processing_urgency: float = 1.0  # priority multiplier

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "document_size_avg": self.document_size_avg,
            "document_complexity": self.document_complexity,
            "ocr_ratio": self.ocr_ratio,
            "table_density": self.table_density,
            "language_distribution": self.language_distribution,
            "processing_urgency": self.processing_urgency,
        }


class MetricsCollector:
    """
    Comprehensive metrics collection system using Prometheus and Redis.
    Monitors system resources, processing performance, and workload patterns.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        prometheus_port: int = 8000,
        collection_interval: float = 5.0,
        enable_prometheus: bool = True,
    ):
        self.logger = logging.getLogger(__name__)
        self.redis_url = redis_url
        self.prometheus_port = prometheus_port
        self.collection_interval = collection_interval
        self.enable_prometheus = enable_prometheus

        # Redis connection
        self.redis: Optional[redis.Redis] = None

        # Metrics storage
        self.metrics_history: Dict[str, List[Dict]] = {
            "system": [],
            "processing": [],
            "workload": [],
        }
        self.max_history = 1000  # Keep last 1000 measurements

        # Processing state tracking
        self.processing_times: List[float] = []
        self.processing_errors: List[datetime] = []
        self.processing_success: List[datetime] = []

        # Thread-safe locks
        self._metrics_lock = threading.RLock()
        self._collection_active = False

        # Initialize Prometheus if available
        if PROMETHEUS_AVAILABLE and enable_prometheus:
            self._init_prometheus()
        else:
            self.logger.warning("Prometheus not available or disabled")

    def _init_prometheus(self):
        """Initialize Prometheus metrics"""
        self.registry = CollectorRegistry()

        # System metrics
        self.cpu_usage_gauge = Gauge(
            "system_cpu_usage_percent",
            "System CPU usage percentage",
            registry=self.registry,
        )
        self.memory_usage_gauge = Gauge(
            "system_memory_usage_percent",
            "System memory usage percentage",
            registry=self.registry,
        )
        self.disk_usage_gauge = Gauge(
            "system_disk_usage_percent",
            "System disk usage percentage",
            registry=self.registry,
        )

        # Processing metrics
        self.throughput_gauge = Gauge(
            "processing_throughput_docs_per_sec",
            "Document processing throughput",
            registry=self.registry,
        )
        self.latency_histogram = Histogram(
            "processing_latency_seconds",
            "Document processing latency",
            buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
            registry=self.registry,
        )
        self.error_counter = Counter(
            "processing_errors_total",
            "Total processing errors",
            ["error_type"],
            registry=self.registry,
        )
        self.active_workers_gauge = Gauge(
            "processing_active_workers",
            "Number of active processing workers",
            registry=self.registry,
        )
        self.queue_depth_gauge = Gauge(
            "processing_queue_depth", "Processing queue depth", registry=self.registry
        )

        # Workload metrics
        self.document_size_gauge = Gauge(
            "workload_document_size_mb",
            "Average document size in MB",
            registry=self.registry,
        )
        self.ocr_ratio_gauge = Gauge(
            "workload_ocr_ratio",
            "Ratio of documents requiring OCR",
            registry=self.registry,
        )

    async def initialize(self):
        """Initialize Redis connection and start collection"""
        try:
            self.redis = redis.from_url(self.redis_url)
            await self.redis.ping()
            self.logger.info("Connected to Redis for metrics storage")
        except Exception as e:
            self.logger.warning(f"Failed to connect to Redis: {e}")
            self.redis = None

        # Start metrics collection
        await self.start_collection()

    async def start_collection(self):
        """Start periodic metrics collection"""
        if self._collection_active:
            return

        self._collection_active = True
        asyncio.create_task(self._collection_loop())
        self.logger.info("Started metrics collection")

    async def stop_collection(self):
        """Stop metrics collection"""
        self._collection_active = False
        if self.redis:
            await self.redis.close()

    async def _collection_loop(self):
        """Main metrics collection loop"""
        while self._collection_active:
            try:
                # Collect all metrics
                system_metrics = await self._collect_system_metrics()
                processing_metrics = await self._collect_processing_metrics()
                workload_metrics = await self._collect_workload_metrics()

                # Update Prometheus if available
                if PROMETHEUS_AVAILABLE and self.enable_prometheus:
                    self._update_prometheus_metrics(
                        system_metrics, processing_metrics, workload_metrics
                    )

                # Store in history with thread safety
                with self._metrics_lock:
                    self._store_metrics(
                        system_metrics, processing_metrics, workload_metrics
                    )

                # Store in Redis if available
                if self.redis:
                    await self._store_redis_metrics(
                        system_metrics, processing_metrics, workload_metrics
                    )

            except Exception as e:
                self.logger.error(f"Error in metrics collection: {e}")

            await asyncio.sleep(self.collection_interval)

    async def _collect_system_metrics(self) -> SystemMetrics:
        """Collect system-level metrics"""
        cpu_usage = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        # Network I/O
        net_io = psutil.net_io_counters()
        network_io = {
            "bytes_sent": net_io.bytes_sent,
            "bytes_recv": net_io.bytes_recv,
            "packets_sent": net_io.packets_sent,
            "packets_recv": net_io.packets_recv,
        }

        return SystemMetrics(
            timestamp=datetime.now(),
            cpu_usage=cpu_usage,
            memory_usage=memory.percent,
            disk_usage=disk.percent,
            network_io=network_io,
            process_count=len(psutil.pids()),
        )

    async def _collect_processing_metrics(self) -> ProcessingMetrics:
        """Collect processing pipeline metrics"""
        now = datetime.now()

        # Calculate throughput based on recent success events
        recent_window = now - timedelta(seconds=60)  # 1 minute window
        recent_successes = [ts for ts in self.processing_success if ts > recent_window]
        throughput = len(recent_successes) / 60.0  # per second

        # Calculate latency percentiles
        with self._metrics_lock:
            processing_times = self.processing_times.copy()

        if processing_times:
            processing_times.sort()
            n = len(processing_times)
            latency_p50 = processing_times[int(n * 0.5)] * 1000  # ms
            latency_p95 = processing_times[int(n * 0.95)] * 1000
            latency_p99 = processing_times[int(n * 0.99)] * 1000
            processing_time_avg = sum(processing_times) / n * 1000
        else:
            latency_p50 = latency_p95 = latency_p99 = processing_time_avg = 0

        # Calculate error rate
        recent_errors = [ts for ts in self.processing_errors if ts > recent_window]
        total_recent = len(recent_successes) + len(recent_errors)
        error_rate = (len(recent_errors) / max(1, total_recent)) * 100

        # Ray cluster info if available
        try:
            if ray.is_initialized():
                ray_nodes = ray.nodes()
                active_workers = sum(1 for node in ray_nodes if node["Alive"])
            else:
                active_workers = 1
        except:
            active_workers = 1

        return ProcessingMetrics(
            timestamp=now,
            throughput=throughput,
            latency_p50=latency_p50,
            latency_p95=latency_p95,
            latency_p99=latency_p99,
            error_rate=error_rate,
            active_workers=active_workers,
            queue_depth=0,  # Would need integration with actual queue
            processing_time_avg=processing_time_avg,
            retry_count=len(recent_errors),
        )

    async def _collect_workload_metrics(self) -> WorkloadMetrics:
        """Collect workload characteristics"""
        # This would integrate with actual document processing pipeline
        # For now, return default values
        return WorkloadMetrics(
            timestamp=datetime.now(),
            document_size_avg=2.5,  # MB
            document_complexity=0.7,
            ocr_ratio=0.3,
            table_density=0.15,
            language_distribution={"es": 0.9, "en": 0.1},
        )

    def _update_prometheus_metrics(
        self,
        system: SystemMetrics,
        processing: ProcessingMetrics,
        workload: WorkloadMetrics,
    ):
        """Update Prometheus metrics"""
        # System metrics
        self.cpu_usage_gauge.set(system.cpu_usage)
        self.memory_usage_gauge.set(system.memory_usage)
        self.disk_usage_gauge.set(system.disk_usage)

        # Processing metrics
        self.throughput_gauge.set(processing.throughput)
        self.active_workers_gauge.set(processing.active_workers)
        self.queue_depth_gauge.set(processing.queue_depth)

        # Workload metrics
        self.document_size_gauge.set(workload.document_size_avg)
        self.ocr_ratio_gauge.set(workload.ocr_ratio)

    def _store_metrics(
        self,
        system: SystemMetrics,
        processing: ProcessingMetrics,
        workload: WorkloadMetrics,
    ):
        """Store metrics in local history with size limit"""
        self.metrics_history["system"].append(system.to_dict())
        self.metrics_history["processing"].append(processing.to_dict())
        self.metrics_history["workload"].append(workload.to_dict())

        # Trim history to max size
        for category in self.metrics_history:
            if len(self.metrics_history[category]) > self.max_history:
                self.metrics_history[category] = self.metrics_history[category][
                    -self.max_history :
                ]

    async def _store_redis_metrics(
        self,
        system: SystemMetrics,
        processing: ProcessingMetrics,
        workload: WorkloadMetrics,
    ):
        """Store metrics in Redis for persistence"""
        try:
            timestamp = int(time.time())

            # Store with expiration (24 hours)
            await self.redis.setex(
                f"metrics:system:{timestamp}", 86400, str(system.to_dict())
            )
            await self.redis.setex(
                f"metrics:processing:{timestamp}", 86400, str(processing.to_dict())
            )
            await self.redis.setex(
                f"metrics:workload:{timestamp}", 86400, str(workload.to_dict())
            )

        except Exception as e:
            self.logger.warning(f"Failed to store metrics in Redis: {e}")

    def record_processing_start(self, document_id: str):
        """Record start of document processing"""
        self._processing_start_times = getattr(self, "_processing_start_times", {})
        self._processing_start_times[document_id] = time.time()

    def record_processing_success(self, document_id: str):
        """Record successful document processing"""
        start_time = getattr(self, "_processing_start_times", {}).get(document_id)
        if start_time:
            processing_time = time.time() - start_time
            with self._metrics_lock:
                self.processing_times.append(processing_time)
                self.processing_success.append(datetime.now())

                # Trim lists to prevent memory growth
                if len(self.processing_times) > 1000:
                    self.processing_times = self.processing_times[-500:]
                if len(self.processing_success) > 1000:
                    self.processing_success = self.processing_success[-500:]

            # Update Prometheus histogram
            if PROMETHEUS_AVAILABLE and self.enable_prometheus:
                self.latency_histogram.observe(processing_time)

    def record_processing_error(self, document_id: str, error_type: str):
        """Record processing error"""
        with self._metrics_lock:
            self.processing_errors.append(datetime.now())

            # Trim error list
            if len(self.processing_errors) > 1000:
                self.processing_errors = self.processing_errors[-500:]

        # Update Prometheus counter
        if PROMETHEUS_AVAILABLE and self.enable_prometheus:
            self.error_counter.labels(error_type=error_type).inc()

    def get_recent_metrics(self, category: str, minutes: int = 5) -> List[Dict]:
        """Get recent metrics for a category"""
        cutoff = datetime.now() - timedelta(minutes=minutes)

        with self._metrics_lock:
            metrics = self.metrics_history.get(category, [])
            return [
                m for m in metrics if datetime.fromisoformat(m["timestamp"]) > cutoff
            ]

    async def get_prometheus_metrics(self) -> str:
        """Get Prometheus metrics for scraping"""
        if not PROMETHEUS_AVAILABLE or not self.enable_prometheus:
            return ""

        return generate_latest(self.registry).decode("utf-8")

    async def health_check(self) -> Dict[str, Any]:
        """Health check for metrics system"""
        health_status = {
            "metrics_collector": "healthy",
            "collection_active": self._collection_active,
            "redis_connected": self.redis is not None,
            "prometheus_enabled": PROMETHEUS_AVAILABLE and self.enable_prometheus,
            "metrics_count": {
                category: len(metrics)
                for category, metrics in self.metrics_history.items()
            },
        }

        if self.redis:
            try:
                await self.redis.ping()
                health_status["redis_status"] = "connected"
            except:
                health_status["redis_status"] = "disconnected"
                health_status["redis_connected"] = False

        return health_status

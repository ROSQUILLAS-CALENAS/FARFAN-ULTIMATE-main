"""
Industrial-grade adaptive controller for Ray worker pools and processing window management.

This module provides a production-ready system for dynamically scaling distributed computing
resources based on workload demands and performance metrics.
"""

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Union
from concurrent.futures import ThreadPoolExecutor
import json

try:
    import ray
    from ray.exceptions import RaySystemError
    from ray.util.state import list_nodes

    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False


# Configuration and Data Models
class ScalingAction(Enum):
    """Scaling action types."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    MAINTAIN = "maintain"
    EMERGENCY_SCALE_DOWN = "emergency_scale_down"


@dataclass
class ScalingDecision:
    """Represents a scaling decision from the decision engine."""
    action: ScalingAction
    target_workers: int
    target_window_size: int
    target_frequency: float
    confidence: float
    reasons: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class WorkerMetrics:
    """Worker performance metrics."""
    worker_id: str
    cpu_usage: float
    memory_usage: float
    task_count: int
    error_count: int
    uptime: float
    last_heartbeat: datetime


@dataclass
class WindowPerformance:
    """Processing window performance data."""
    window_id: str
    timestamp: datetime
    window_size: int
    processing_time: float
    throughput: float
    error_rate: float
    success_count: int
    error_count: int


class ControllerState(Enum):
    """Controller operational states."""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    SCALING = "scaling"
    ERROR = "error"
    SHUTDOWN = "shutdown"


class RayWorkerPool:
    """
    Industrial-grade Ray worker pool with dynamic scaling capabilities.

    Features:
    - Graceful scaling with health monitoring
    - Worker lifecycle management
    - Performance tracking and metrics
    - Fault tolerance and recovery
    """

    def __init__(
            self,
            worker_class: Optional[str] = None,
            initial_workers: int = 2,
            min_workers: int = 1,
            max_workers: int = 50,
            worker_timeout: float = 30.0,
            health_check_interval: float = 60.0,
    ):
        if not RAY_AVAILABLE:
            raise ImportError("Ray is required but not available")

        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Configuration
        self.worker_class = worker_class or "DefaultWorker"
        self.min_workers = max(1, min_workers)
        self.max_workers = max(self.min_workers, max_workers)
        self.worker_timeout = worker_timeout
        self.health_check_interval = health_check_interval

        # State management
        self.active_workers: Dict[str, ray.ObjectRef] = {}
        self.worker_metrics: Dict[str, WorkerMetrics] = {}
        self.worker_creation_times: Dict[str, datetime] = {}
        self.failed_workers: Set[str] = set()

        # Scaling management
        self.target_workers = max(initial_workers, self.min_workers)
        self.scaling_history: List[Dict[str, Any]] = []
        self.last_scale_time = datetime.now()

        # Thread safety
        self._scaling_lock = threading.RLock()
        self._health_lock = threading.RLock()
        self._initialized = False
        self._shutdown_event = threading.Event()

        # Background tasks
        self._health_task: Optional[asyncio.Task] = None
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="ray-pool")

    async def initialize(self) -> bool:
        """Initialize Ray cluster and worker pool."""
        try:
            if not ray.is_initialized():
                ray.init(
                    ignore_reinit_error=True,
                    include_dashboard=False,
                    log_to_driver=False,
                    num_cpus=None,  # Auto-detect
                    object_store_memory=None,  # Auto-detect
                )

            await self._create_initial_workers()
            self._start_health_monitoring()
            self._initialized = True

            self.logger.info(
                f"Ray worker pool initialized with {len(self.active_workers)} workers"
            )
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize Ray worker pool: {e}")
            return False

    async def _create_initial_workers(self):
        """Create initial set of workers."""
        for i in range(self.target_workers):
            try:
                worker_id = f"worker_{i}_{int(time.time())}"
                worker_ref = await self._create_worker(worker_id)
                if worker_ref:
                    self.active_workers[worker_id] = worker_ref
                    self.worker_creation_times[worker_id] = datetime.now()

            except Exception as e:
                self.logger.error(f"Failed to create initial worker {i}: {e}")

    async def _create_worker(self, worker_id: str) -> Optional[ray.ObjectRef]:
        """Create a single worker with error handling."""
        try:
            @ray.remote
            class WorkerActor:
                def __init__(self, worker_id: str):
                    self.worker_id = worker_id
                    self.start_time = time.time()
                    self.task_count = 0
                    self.error_count = 0

                def get_status(self):
                    return {
                        "worker_id": self.worker_id,
                        "uptime": time.time() - self.start_time,
                        "task_count": self.task_count,
                        "error_count": self.error_count,
                        "timestamp": time.time()
                    }

                def process_task(self, task_data):
                    try:
                        self.task_count += 1
                        # Simulate work
                        time.sleep(0.1)
                        return {"status": "success", "worker_id": self.worker_id}
                    except Exception as e:
                        self.error_count += 1
                        raise e

                def shutdown(self):
                    return f"Worker {self.worker_id} shutting down"

            worker = WorkerActor.remote(worker_id)
            return worker

        except Exception as e:
            self.logger.error(f"Failed to create worker {worker_id}: {e}")
            return None

    def _start_health_monitoring(self):
        """Start background health monitoring."""
        if self._health_task is None or self._health_task.done():
            self._health_task = asyncio.create_task(self._health_monitor_loop())

    async def _health_monitor_loop(self):
        """Background health monitoring loop."""
        while not self._shutdown_event.is_set() and self._initialized:
            try:
                await self._check_worker_health()
                await asyncio.sleep(self.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Health monitor error: {e}")
                await asyncio.sleep(10)  # Back off on error

    async def _check_worker_health(self):
        """Check health of all workers."""
        unhealthy_workers = []

        with self._health_lock:
            for worker_id, worker_ref in list(self.active_workers.items()):
                try:
                    # Check if worker is still responsive
                    status_future = worker_ref.get_status.remote()
                    status = await asyncio.wait_for(
                        self._ray_get_async(status_future),
                        timeout=self.worker_timeout
                    )

                    # Update metrics
                    self.worker_metrics[worker_id] = WorkerMetrics(
                        worker_id=worker_id,
                        cpu_usage=0.0,  # Would get from actual monitoring
                        memory_usage=0.0,  # Would get from actual monitoring
                        task_count=status.get("task_count", 0),
                        error_count=status.get("error_count", 0),
                        uptime=status.get("uptime", 0),
                        last_heartbeat=datetime.now()
                    )

                except (asyncio.TimeoutError, RaySystemError, Exception) as e:
                    self.logger.warning(f"Worker {worker_id} health check failed: {e}")
                    unhealthy_workers.append(worker_id)

        # Remove unhealthy workers
        for worker_id in unhealthy_workers:
            await self._remove_unhealthy_worker(worker_id)

    async def _ray_get_async(self, object_ref):
        """Async wrapper for ray.get()."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, ray.get, object_ref)

    async def _remove_unhealthy_worker(self, worker_id: str):
        """Remove an unhealthy worker."""
        try:
            if worker_id in self.active_workers:
                worker_ref = self.active_workers.pop(worker_id)
                ray.cancel(worker_ref, force=True)

            self.failed_workers.add(worker_id)
            self.worker_metrics.pop(worker_id, None)
            self.worker_creation_times.pop(worker_id, None)

            self.logger.warning(f"Removed unhealthy worker: {worker_id}")

        except Exception as e:
            self.logger.error(f"Failed to remove unhealthy worker {worker_id}: {e}")

    async def scale_workers(self, target_count: int, reason: str = "manual") -> bool:
        """Scale worker pool to target count."""
        if not self._initialized:
            self.logger.error("Worker pool not initialized")
            return False

        target_count = max(self.min_workers, min(target_count, self.max_workers))
        current_count = len(self.active_workers)

        if target_count == current_count:
            return True

        with self._scaling_lock:
            try:
                if target_count > current_count:
                    success = await self._scale_up(target_count - current_count, reason)
                else:
                    success = await self._scale_down(current_count - target_count, reason)

                if success:
                    self.target_workers = target_count
                    self._record_scaling_event(current_count, target_count, reason)

                return success

            except Exception as e:
                self.logger.error(f"Scaling operation failed: {e}")
                return False

    async def _scale_up(self, count: int, reason: str) -> bool:
        """Scale up by adding workers."""
        created_count = 0

        for i in range(count):
            worker_id = f"worker_{len(self.active_workers) + i}_{int(time.time())}"
            worker_ref = await self._create_worker(worker_id)

            if worker_ref:
                self.active_workers[worker_id] = worker_ref
                self.worker_creation_times[worker_id] = datetime.now()
                created_count += 1
            else:
                self.logger.error(f"Failed to create worker {worker_id}")

        self.logger.info(f"Scaled up by {created_count}/{count} workers ({reason})")
        return created_count > 0

    async def _scale_down(self, count: int, reason: str) -> bool:
        """Scale down by removing workers gracefully."""
        if count >= len(self.active_workers):
            count = len(self.active_workers) - self.min_workers

        # Select workers to remove (oldest first)
        workers_by_age = sorted(
            self.active_workers.items(),
            key=lambda x: self.worker_creation_times.get(x[0], datetime.now())
        )

        removed_count = 0
        for worker_id, worker_ref in workers_by_age[-count:]:
            try:
                # Graceful shutdown
                shutdown_future = worker_ref.shutdown.remote()
                await asyncio.wait_for(
                    self._ray_get_async(shutdown_future),
                    timeout=self.worker_timeout
                )

                ray.cancel(worker_ref, force=False)
                del self.active_workers[worker_id]
                self.worker_creation_times.pop(worker_id, None)
                self.worker_metrics.pop(worker_id, None)
                removed_count += 1

            except Exception as e:
                self.logger.error(f"Failed to gracefully remove worker {worker_id}: {e}")
                # Force removal
                ray.cancel(worker_ref, force=True)
                self.active_workers.pop(worker_id, None)
                removed_count += 1

        self.logger.info(f"Scaled down by {removed_count}/{count} workers ({reason})")
        return removed_count > 0

    def _record_scaling_event(self, old_count: int, new_count: int, reason: str):
        """Record scaling event for analysis."""
        event = {
            "timestamp": datetime.now().isoformat(),
            "old_count": old_count,
            "new_count": new_count,
            "change": new_count - old_count,
            "reason": reason,
        }

        self.scaling_history.append(event)

        # Maintain rolling history
        if len(self.scaling_history) > 1000:
            self.scaling_history = self.scaling_history[-500:]

    def get_worker_count(self) -> int:
        """Get current active worker count."""
        return len(self.active_workers)

    def get_worker_stats(self) -> Dict[str, Any]:
        """Get comprehensive worker statistics."""
        now = datetime.now()

        # Calculate uptime statistics
        uptimes = []
        for worker_id, created_time in self.worker_creation_times.items():
            if worker_id in self.active_workers:
                uptime = (now - created_time).total_seconds()
                uptimes.append(uptime)

        # Calculate task statistics
        total_tasks = sum(m.task_count for m in self.worker_metrics.values())
        total_errors = sum(m.error_count for m in self.worker_metrics.values())

        return {
            "active_workers": len(self.active_workers),
            "target_workers": self.target_workers,
            "min_workers": self.min_workers,
            "max_workers": self.max_workers,
            "failed_workers": len(self.failed_workers),
            "average_uptime": sum(uptimes) / len(uptimes) if uptimes else 0,
            "total_tasks_processed": total_tasks,
            "total_errors": total_errors,
            "error_rate": (total_errors / max(total_tasks, 1)) * 100,
            "recent_scaling_events": len([
                e for e in self.scaling_history
                if (now - datetime.fromisoformat(e["timestamp"])).total_seconds() < 3600
            ]),
        }

    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check."""
        health = {
            "ray_initialized": ray.is_initialized() if RAY_AVAILABLE else False,
            "workers_healthy": True,
            "cluster_healthy": True,
            "worker_count": len(self.active_workers),
            "failed_workers": len(self.failed_workers),
            "last_health_check": datetime.now().isoformat(),
        }

        try:
            if ray.is_initialized():
                # Check cluster health
                nodes = list_nodes()
                alive_nodes = [n for n in nodes if n.state == "ALIVE"]

                health.update({
                    "total_nodes": len(nodes),
                    "alive_nodes": len(alive_nodes),
                    "cluster_healthy": len(alive_nodes) > 0,
                })

            # Check worker health
            unhealthy_count = 0
            for worker_id, metrics in self.worker_metrics.items():
                if (datetime.now() - metrics.last_heartbeat).total_seconds() > 120:
                    unhealthy_count += 1

            health["workers_healthy"] = unhealthy_count == 0
            health["unhealthy_worker_count"] = unhealthy_count

        except Exception as e:
            health["error"] = str(e)
            health["cluster_healthy"] = False

        return health

    async def shutdown(self):
        """Graceful shutdown of worker pool."""
        self.logger.info("Initiating worker pool shutdown...")
        self._shutdown_event.set()

        try:
            # Cancel health monitoring
            if self._health_task and not self._health_task.done():
                self._health_task.cancel()
                try:
                    await self._health_task
                except asyncio.CancelledError:
                    pass

            # Shutdown all workers
            with self._scaling_lock:
                await self._scale_down(len(self.active_workers), "shutdown")

            # Shutdown executor
            self._executor.shutdown(wait=True)

            # Shutdown Ray if we initialized it
            if ray.is_initialized():
                ray.shutdown()

            self.logger.info("Worker pool shutdown complete")

        except Exception as e:
            self.logger.error(f"Error during worker pool shutdown: {e}")


class ProcessingWindowManager:
    """
    Advanced processing window management with adaptive sizing and performance optimization.

    Features:
    - Dynamic window sizing based on performance
    - Frequency adjustment for optimal throughput
    - Performance tracking and analysis
    - Congestion control
    """

    def __init__(
            self,
            initial_window_size: int = 10,
            min_window_size: int = 1,
            max_window_size: int = 100,
            initial_frequency: float = 1.0,
            performance_window: int = 50,
    ):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Window configuration
        self.current_window_size = max(min_window_size, initial_window_size)
        self.min_window_size = min_window_size
        self.max_window_size = max_window_size

        # Frequency management
        self.current_frequency = initial_frequency
        self.target_frequency = initial_frequency
        self.min_frequency = 0.1
        self.max_frequency = 10.0

        # Performance tracking
        self.performance_window = performance_window
        self.window_performance: List[WindowPerformance] = []
        self.adjustment_history: List[Dict[str, Any]] = []

        # Active window tracking
        self.active_windows: Dict[str, Dict[str, Any]] = {}
        self.window_counter = 0

        # Thread safety
        self._window_lock = threading.RLock()

    def adjust_window_size(self, new_size: int, reason: str = "adaptive") -> bool:
        """Adjust processing window size with validation."""
        new_size = max(self.min_window_size, min(new_size, self.max_window_size))

        if new_size == self.current_window_size:
            return True

        with self._window_lock:
            old_size = self.current_window_size
            self.current_window_size = new_size

            self._record_adjustment("window_size", old_size, new_size, reason)
            self.logger.info(f"Window size adjusted: {old_size} -> {new_size} ({reason})")

        return True

    def adjust_frequency(self, new_frequency: float, reason: str = "adaptive") -> bool:
        """Adjust processing frequency with validation."""
        new_frequency = max(self.min_frequency, min(new_frequency, self.max_frequency))

        if abs(new_frequency - self.current_frequency) < 0.01:
            return True

        with self._window_lock:
            old_frequency = self.current_frequency
            self.current_frequency = new_frequency
            self.target_frequency = new_frequency

            self._record_adjustment("frequency", old_frequency, new_frequency, reason)
            self.logger.info(
                f"Frequency adjusted: {old_frequency:.2f}Hz -> {new_frequency:.2f}Hz ({reason})"
            )

        return True

    def _record_adjustment(self, adj_type: str, old_value: Union[int, float],
                           new_value: Union[int, float], reason: str):
        """Record parameter adjustment."""
        adjustment = {
            "timestamp": datetime.now().isoformat(),
            "type": adj_type,
            "old_value": old_value,
            "new_value": new_value,
            "reason": reason,
        }

        self.adjustment_history.append(adjustment)

        # Maintain rolling history
        if len(self.adjustment_history) > 500:
            self.adjustment_history = self.adjustment_history[-250:]

    def create_processing_window(self, documents: List[str]) -> str:
        """Create new processing window and return window ID."""
        with self._window_lock:
            self.window_counter += 1
            window_id = f"window_{self.window_counter}_{int(time.time())}"

            window = {
                "id": window_id,
                "documents": documents,
                "size": len(documents),
                "created_at": datetime.now(),
                "status": "created",
                "start_time": None,
                "end_time": None,
                "success_count": 0,
                "error_count": 0,
            }

            self.active_windows[window_id] = window
            return window_id

    def start_window_processing(self, window_id: str) -> bool:
        """Mark window as started processing."""
        with self._window_lock:
            if window_id in self.active_windows:
                self.active_windows[window_id]["start_time"] = datetime.now()
                self.active_windows[window_id]["status"] = "processing"
                return True
        return False

    def complete_window_processing(self, window_id: str, success_count: int,
                                   error_count: int) -> bool:
        """Complete window processing and record performance metrics."""
        with self._window_lock:
            if window_id not in self.active_windows:
                return False

            window = self.active_windows[window_id]
            end_time = datetime.now()
            window["end_time"] = end_time
            window["status"] = "completed"
            window["success_count"] = success_count
            window["error_count"] = error_count

            # Calculate performance metrics
            if window["start_time"]:
                processing_time = (end_time - window["start_time"]).total_seconds()
                throughput = success_count / max(processing_time, 0.001)
                error_rate = (error_count / max(window["size"], 1)) * 100

                performance = WindowPerformance(
                    window_id=window_id,
                    timestamp=end_time,
                    window_size=window["size"],
                    processing_time=processing_time,
                    throughput=throughput,
                    error_rate=error_rate,
                    success_count=success_count,
                    error_count=error_count,
                )

                self.window_performance.append(performance)

                # Maintain performance history
                if len(self.window_performance) > self.performance_window * 2:
                    self.window_performance = self.window_performance[-self.performance_window:]

            # Clean up completed window
            del self.active_windows[window_id]
            return True

    def get_optimal_window_size(self) -> int:
        """Calculate optimal window size based on recent performance."""
        if len(self.window_performance) < 5:
            return self.current_window_size

        # Analyze recent performance by window size
        size_performance = {}
        recent_windows = self.window_performance[-20:]

        for perf in recent_windows:
            size = perf.window_size
            if size not in size_performance:
                size_performance[size] = []
            size_performance[size].append(perf)

        # Find optimal size based on efficiency score
        best_size = self.current_window_size
        best_score = 0

        for size, performances in size_performance.items():
            if len(performances) < 2:  # Need multiple samples
                continue

            avg_throughput = sum(p.throughput for p in performances) / len(performances)
            avg_error_rate = sum(p.error_rate for p in performances) / len(performances)
            avg_processing_time = sum(p.processing_time for p in performances) / len(performances)

            # Efficiency score: balance throughput, error rate, and processing time
            error_penalty = max(0, avg_error_rate - 5) * 0.1  # Penalty for >5% error rate
            time_penalty = max(0, avg_processing_time - 10) * 0.01  # Penalty for >10s processing

            score = avg_throughput * (1 - error_penalty - time_penalty)

            if score > best_score:
                best_score = score
                best_size = size

        return best_size

    def get_recommended_frequency(self) -> float:
        """Calculate recommended processing frequency."""
        if len(self.window_performance) < 3:
            return self.current_frequency

        recent_windows = self.window_performance[-10:]
        avg_processing_time = sum(p.processing_time for p in recent_windows) / len(recent_windows)
        avg_error_rate = sum(p.error_rate for p in recent_windows) / len(recent_windows)

        # Adjust frequency based on performance
        if avg_error_rate > 10:  # High error rate - slow down
            recommended_frequency = self.current_frequency * 0.8
        elif avg_error_rate < 2 and avg_processing_time < 5:  # Good performance - speed up
            recommended_frequency = self.current_frequency * 1.2
        else:
            recommended_frequency = self.current_frequency

        return max(self.min_frequency, min(recommended_frequency, self.max_frequency))

    def get_window_stats(self) -> Dict[str, Any]:
        """Get comprehensive window processing statistics."""
        recent_performance = self.window_performance[-20:]

        stats = {
            "current_window_size": self.current_window_size,
            "current_frequency": self.current_frequency,
            "active_windows": len(self.active_windows),
            "total_windows_processed": len(self.window_performance),
            "recent_adjustments": len([
                adj for adj in self.adjustment_history
                if (datetime.now() - datetime.fromisoformat(adj["timestamp"])).total_seconds() < 3600
            ]),
        }

        if recent_performance:
            stats.update({
                "avg_throughput": sum(p.throughput for p in recent_performance) / len(recent_performance),
                "avg_error_rate": sum(p.error_rate for p in recent_performance) / len(recent_performance),
                "avg_processing_time": sum(p.processing_time for p in recent_performance) / len(recent_performance),
                "optimal_window_size": self.get_optimal_window_size(),
                "recommended_frequency": self.get_recommended_frequency(),
            })

        return stats


class AdaptiveController:
    """
    Industrial-grade adaptive controller for distributed processing systems.

    Coordinates Ray worker scaling and processing window management based on
    real-time performance metrics and workload demands.

    Features:
    - Automatic scaling based on performance metrics
    - Congestion control and backpressure handling
    - Health monitoring and fault recovery
    - Performance optimization
    - Emergency scaling capabilities
    """

    def __init__(
            self,
            min_workers: int = 2,
            max_workers: int = 50,
            min_window_size: int = 5,
            max_window_size: int = 100,
            adjustment_cooldown: float = 30.0,
            emergency_threshold: float = 0.9,
    ):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Component initialization
        self.worker_pool = RayWorkerPool(
            min_workers=min_workers,
            max_workers=max_workers,
        )

        self.window_manager = ProcessingWindowManager(
            min_window_size=min_window_size,
            max_window_size=max_window_size,
        )

        # Control parameters
        self.adjustment_cooldown = adjustment_cooldown
        self.emergency_threshold = emergency_threshold
        self.last_adjustment = datetime.now() - timedelta(seconds=adjustment_cooldown)

        # State management
        self.state = ControllerState.INITIALIZING
        self.is_active = False
        self._control_lock = threading.RLock()
        self._shutdown_event = threading.Event()

        # Callbacks and monitoring
        self.scaling_callbacks: List[Callable] = []
        self._monitoring_task: Optional[asyncio.Task] = None

        # Performance metrics
        self.performance_history: List[Dict[str, Any]] = []

    async def initialize(self) -> bool:
        """Initialize controller and all components."""
        try:
            self.state = ControllerState.INITIALIZING

            # Initialize worker pool
            if not await self.worker_pool.initialize():
                raise RuntimeError("Failed to initialize worker pool")

            # Start monitoring
            self._start_monitoring()

            self.state = ControllerState.ACTIVE
            self.is_active = True

            self.logger.info("Adaptive controller initialized successfully")
            return True

        except Exception as e:
            self.state = ControllerState.ERROR
            self.logger.error(f"Failed to initialize adaptive controller: {e}")
            return False

    def _start_monitoring(self):
        """Start background monitoring tasks."""
        if self._monitoring_task is None or self._monitoring_task.done():
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())

    async def _monitoring_loop(self):
        """Background monitoring and optimization loop."""
        while not self._shutdown_event.is_set() and self.is_active:
            try:
                await self._perform_health_checks()
                await self._optimize_performance()
                await self._check_emergency_conditions()
                await asyncio.sleep(30)  # Monitor every 30 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(60)  # Back off on error

    async def _perform_health_checks(self):
        """Perform comprehensive health checks."""
        try:
            worker_health = await self.worker_pool.health_check()
            window_stats = self.window_manager.get_window_stats()

            # Record performance metrics
            metrics = {
                "timestamp": datetime.now().isoformat(),
                "worker_count": worker_health["worker_count"],
                "workers_healthy": worker_health["workers_healthy"],
                "cluster_healthy": worker_health["cluster_healthy"],
                "window_size": window_stats["current_window_size"],
                "frequency": window_stats["current_frequency"],
                "avg_throughput": window_stats.get("avg_throughput", 0),
                "avg_error_rate": window_stats.get("avg_error_rate", 0),
            }

            self.performance_history.append(metrics)

            # Maintain rolling history
            if len(self.performance_history) > 1000:
                self.performance_history = self.performance_history[-500:]

        except Exception as e:
            self.logger.error(f"Health check failed: {e}")

    async def _optimize_performance(self):
        """Automatic performance optimization."""
        try:
            # Check if we're in cooldown
            if (datetime.now() - self.last_adjustment).total_seconds() < self.adjustment_cooldown:
                return

            window_stats = self.window_manager.get_window_stats()
            worker_stats = self.worker_pool.get_worker_stats()

            # Generate optimization decision
            decision = self._generate_optimization_decision(window_stats, worker_stats)

            if decision and decision.action != ScalingAction.MAINTAIN:
                await self.apply_scaling_decision(decision)

        except Exception as e:
            self.logger.error(f"Performance optimization failed: {e}")

    def _generate_optimization_decision(self, window_stats: Dict, worker_stats: Dict) -> Optional[ScalingDecision]:
        """Generate optimization decision based on current metrics."""
        current_workers = worker_stats["active_workers"]
        current_window_size = window_stats["current_window_size"]
        current_frequency = window_stats["current_frequency"]

        avg_throughput = window_stats.get("avg_throughput", 0)
        avg_error_rate = window_stats.get("avg_error_rate", 0)
        worker_error_rate = worker_stats.get("error_rate", 0)

        # Decision logic
        action = ScalingAction.MAINTAIN
        reasons = []
        confidence = 0.5

        # Scale up conditions
        if avg_error_rate > 15 or worker_error_rate > 10:
            if current_workers < self.worker_pool.max_workers:
                action = ScalingAction.SCALE_UP
                reasons.append("high_error_rate")
                confidence += 0.3

        elif avg_throughput > 0 and len(self.performance_history) > 5:
            recent_throughput = [m["avg_throughput"] for m in self.performance_history[-5:]]
            if all(t > 0 for t in recent_throughput) and avg_throughput < min(recent_throughput) * 0.8:
                if current_workers < self.worker_pool.max_workers:
                    action = ScalingAction.SCALE_UP
                    reasons.append("throughput_degradation")
                    confidence += 0.2

        # Scale down conditions
        elif avg_error_rate < 2 and worker_error_rate < 2:
            if current_workers > self.worker_pool.min_workers:
                if avg_throughput > 0 and len(self.window_manager.active_windows) == 0:
                    action = ScalingAction.SCALE_DOWN
                    reasons.append("low_utilization")
                    confidence += 0.2

        # Calculate targets
        if action == ScalingAction.SCALE_UP:
            target_workers = min(current_workers + max(1, current_workers // 4), self.worker_pool.max_workers)
        elif action == ScalingAction.SCALE_DOWN:
            target_workers = max(current_workers - 1, self.worker_pool.min_workers)
        else:
            target_workers = current_workers

        # Window size optimization
        optimal_window_size = window_stats.get("optimal_window_size", current_window_size)
        if abs(optimal_window_size - current_window_size) > 2:
            if action == ScalingAction.MAINTAIN:
                action = ScalingAction.SCALE_UP if optimal_window_size > current_window_size else ScalingAction.SCALE_DOWN
            reasons.append("window_optimization")
            confidence += 0.1

        # Frequency optimization
        recommended_frequency = window_stats.get("recommended_frequency", current_frequency)

        if confidence > 0.6 and action != ScalingAction.MAINTAIN:
            return ScalingDecision(
                action=action,
                target_workers=target_workers,
                target_window_size=optimal_window_size,
                target_frequency=recommended_frequency,
                confidence=confidence,
                reasons=reasons,
            )

        return None

    async def _check_emergency_conditions(self):
        """Check for emergency conditions requiring immediate action."""
        try:
            worker_health = await self.worker_pool.health_check()

            # Emergency conditions
            emergency_conditions = [
                not worker_health["cluster_healthy"],
                worker_health["worker_count"] == 0,
                worker_health.get("error_rate", 0) > 50,
                len(self.worker_pool.failed_workers) > self.worker_pool.max_workers * 0.5,
            ]

            if any(emergency_conditions):
                self.logger.warning("Emergency conditions detected, initiating emergency scale down")
                await self.emergency_scale_down()

        except Exception as e:
            self.logger.error(f"Emergency condition check failed: {e}")

    async def apply_scaling_decision(self, decision: ScalingDecision) -> bool:
        """Apply scaling decision from decision engine or internal optimization."""
        if not self.is_active:
            self.logger.warning("Controller not active, ignoring scaling decision")
            return False

        # Check cooldown period
        now = datetime.now()
        if (now - self.last_adjustment).total_seconds() < self.adjustment_cooldown:
            if decision.action != ScalingAction.EMERGENCY_SCALE_DOWN:
                self.logger.debug("Adjustment cooldown active, deferring scaling decision")
                return False

        # Only apply decisions with sufficient confidence
        if decision.confidence < 0.5 and decision.action != ScalingAction.EMERGENCY_SCALE_DOWN:
            self.logger.debug(f"Decision confidence too low ({decision.confidence:.2f}), skipping")
            return False

        success = True
        self.state = ControllerState.SCALING

        with self._control_lock:
            try:
                # Apply worker scaling
                current_workers = self.worker_pool.get_worker_count()
                if decision.target_workers != current_workers:
                    reason = f"{decision.action.value}: {', '.join(decision.reasons)}"
                    success &= await self.worker_pool.scale_workers(decision.target_workers, reason)

                # Apply window size adjustment
                if decision.target_window_size != self.window_manager.current_window_size:
                    reason = f"{decision.action.value}: {', '.join(decision.reasons)}"
                    success &= self.window_manager.adjust_window_size(decision.target_window_size, reason)

                # Apply frequency adjustment
                if abs(decision.target_frequency - self.window_manager.current_frequency) > 0.1:
                    reason = f"{decision.action.value}: {', '.join(decision.reasons)}"
                    success &= self.window_manager.adjust_frequency(decision.target_frequency, reason)

                if success:
                    self.last_adjustment = now
                    self.logger.info(f"Successfully applied scaling decision: {decision.action.value}")

                    # Notify callbacks
                    await self._notify_callbacks(decision)

                self.state = ControllerState.ACTIVE
                return success

            except Exception as e:
                self.state = ControllerState.ERROR
                self.logger.error(f"Failed to apply scaling decision: {e}")
                return False

    async def _notify_callbacks(self, decision: ScalingDecision):
        """Notify all registered callbacks of scaling events."""
        for callback in self.scaling_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(decision)
                else:
                    callback(decision)
            except Exception as e:
                self.logger.error(f"Callback notification failed: {e}")

    def add_scaling_callback(self, callback: Callable):
        """Add callback to be notified of scaling events."""
        self.scaling_callbacks.append(callback)

    def remove_scaling_callback(self, callback: Callable):
        """Remove scaling callback."""
        try:
            self.scaling_callbacks.remove(callback)
        except ValueError:
            pass

    async def process_batch(self, documents: List[str]) -> Dict[str, Any]:
        """Process a batch of documents using the managed window system."""
        if not self.is_active:
            raise RuntimeError("Controller not active")

        # Create processing window
        window_id = self.window_manager.create_processing_window(documents)

        try:
            # Start processing
            self.window_manager.start_window_processing(window_id)

            # Simulate processing (in real implementation, distribute to workers)
            success_count = 0
            error_count = 0

            # Here you would actually distribute work to Ray workers
            # For demo purposes, we'll simulate the work
            await asyncio.sleep(len(documents) * 0.1)  # Simulate processing time

            success_count = max(0, len(documents) - len(documents) // 10)  # 90% success rate
            error_count = len(documents) - success_count

            # Complete processing
            self.window_manager.complete_window_processing(window_id, success_count, error_count)

            return {
                "window_id": window_id,
                "total_documents": len(documents),
                "success_count": success_count,
                "error_count": error_count,
                "success_rate": (success_count / len(documents)) * 100 if documents else 0,
            }

        except Exception as e:
            self.logger.error(f"Batch processing failed for window {window_id}: {e}")
            # Mark as failed
            self.window_manager.complete_window_processing(window_id, 0, len(documents))
            raise

    def get_current_state(self) -> Dict[str, Any]:
        """Get comprehensive current controller state."""
        return {
            "controller_state": self.state.value,
            "active": self.is_active,
            "worker_pool": self.worker_pool.get_worker_stats(),
            "window_manager": self.window_manager.get_window_stats(),
            "last_adjustment": self.last_adjustment.isoformat(),
            "cooldown_remaining": max(
                0,
                self.adjustment_cooldown - (datetime.now() - self.last_adjustment).total_seconds(),
            ),
            "performance_history_length": len(self.performance_history),
            "callbacks_registered": len(self.scaling_callbacks),
        }

    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check of all components."""
        health = {
            "controller_active": self.is_active,
            "controller_state": self.state.value,
            "timestamp": datetime.now().isoformat(),
        }

        try:
            # Worker pool health
            health["worker_pool"] = await self.worker_pool.health_check()

            # Window manager health
            window_stats = self.window_manager.get_window_stats()
            health["window_manager"] = {
                "active_windows": window_stats["active_windows"],
                "total_processed": window_stats["total_windows_processed"],
                "current_performance": {
                    "throughput": window_stats.get("avg_throughput", 0),
                    "error_rate": window_stats.get("avg_error_rate", 0),
                },
            }

            # Overall system health
            worker_healthy = health["worker_pool"]["workers_healthy"]
            cluster_healthy = health["worker_pool"]["cluster_healthy"]

            health["overall_healthy"] = (
                    self.is_active and
                    worker_healthy and
                    cluster_healthy and
                    self.state in [ControllerState.ACTIVE, ControllerState.SCALING]
            )

        except Exception as e:
            health["error"] = str(e)
            health["overall_healthy"] = False

        return health

    async def emergency_scale_down(self) -> bool:
        """Emergency scale down to minimal resources."""
        self.logger.warning("Executing emergency scale down")

        emergency_decision = ScalingDecision(
            action=ScalingAction.EMERGENCY_SCALE_DOWN,
            target_workers=self.worker_pool.min_workers,
            target_window_size=self.window_manager.min_window_size,
            target_frequency=0.5,  # Reduce frequency
            confidence=1.0,
            reasons=["emergency_conditions"],
        )

        try:
            # Bypass cooldown for emergency
            old_cooldown = self.adjustment_cooldown
            self.adjustment_cooldown = 0

            success = await self.apply_scaling_decision(emergency_decision)

            # Restore cooldown
            self.adjustment_cooldown = old_cooldown

            if success:
                self.logger.info("Emergency scale down completed successfully")
            else:
                self.logger.error("Emergency scale down failed")

            return success

        except Exception as e:
            self.logger.error(f"Emergency scale down failed: {e}")
            return False

    async def force_scale_to(self, workers: int, window_size: int, frequency: float, reason: str = "manual") -> bool:
        """Force scaling to specific parameters (bypasses cooldown and confidence checks)."""
        self.logger.info(f"Force scaling to {workers} workers, {window_size} window size, {frequency}Hz")

        decision = ScalingDecision(
            action=ScalingAction.SCALE_UP if workers > self.worker_pool.get_worker_count() else ScalingAction.SCALE_DOWN,
            target_workers=workers,
            target_window_size=window_size,
            target_frequency=frequency,
            confidence=1.0,
            reasons=[reason],
        )

        # Temporarily bypass cooldown
        old_adjustment_time = self.last_adjustment
        self.last_adjustment = datetime.now() - timedelta(seconds=self.adjustment_cooldown + 1)

        try:
            success = await self.apply_scaling_decision(decision)
            return success
        finally:
            if not success:
                self.last_adjustment = old_adjustment_time

    def export_metrics(self, filepath: Optional[str] = None) -> Dict[str, Any]:
        """Export performance metrics and history."""
        metrics_data = {
            "export_timestamp": datetime.now().isoformat(),
            "controller_config": {
                "min_workers": self.worker_pool.min_workers,
                "max_workers": self.worker_pool.max_workers,
                "min_window_size": self.window_manager.min_window_size,
                "max_window_size": self.window_manager.max_window_size,
                "adjustment_cooldown": self.adjustment_cooldown,
            },
            "performance_history": self.performance_history,
            "worker_scaling_history": self.worker_pool.scaling_history,
            "window_adjustment_history": self.window_manager.adjustment_history,
            "window_performance": [
                {
                    "window_id": p.window_id,
                    "timestamp": p.timestamp.isoformat(),
                    "window_size": p.window_size,
                    "processing_time": p.processing_time,
                    "throughput": p.throughput,
                    "error_rate": p.error_rate,
                    "success_count": p.success_count,
                    "error_count": p.error_count,
                }
                for p in self.window_manager.window_performance
            ],
        }

        if filepath:
            try:
                with open(filepath, 'w') as f:
                    json.dump(metrics_data, f, indent=2, default=str)
                self.logger.info(f"Metrics exported to {filepath}")
            except Exception as e:
                self.logger.error(f"Failed to export metrics to {filepath}: {e}")

        return metrics_data

    async def shutdown(self):
        """Graceful shutdown of entire controller system."""
        self.logger.info("Initiating adaptive controller shutdown...")

        self.is_active = False
        self.state = ControllerState.SHUTDOWN
        self._shutdown_event.set()

        try:
            # Cancel monitoring
            if self._monitoring_task and not self._monitoring_task.done():
                self._monitoring_task.cancel()
                try:
                    await self._monitoring_task
                except asyncio.CancelledError:
                    pass

            # Shutdown worker pool
            await self.worker_pool.shutdown()

            # Clear callbacks
            self.scaling_callbacks.clear()

            self.logger.info("Adaptive controller shutdown complete")

        except Exception as e:
            self.logger.error(f"Error during controller shutdown: {e}")


# Factory and utility functions
def create_adaptive_controller(
        min_workers: int = 2,
        max_workers: int = 20,
        min_window_size: int = 5,
        max_window_size: int = 50,
        **kwargs
) -> AdaptiveController:
    """Factory function to create and configure an adaptive controller."""
    return AdaptiveController(
        min_workers=min_workers,
        max_workers=max_workers,
        min_window_size=min_window_size,
        max_window_size=max_window_size,
        **kwargs
    )


async def run_adaptive_system_example():
    """Example usage of the adaptive controller system."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create controller
    controller = create_adaptive_controller(
        min_workers=2,
        max_workers=10,
        min_window_size=5,
        max_window_size=30,
        adjustment_cooldown=15.0,
    )

    # Add callback for scaling events
    async def scaling_callback(decision: ScalingDecision):
        print(f"Scaling event: {decision.action.value} -> {decision.target_workers} workers")

    controller.add_scaling_callback(scaling_callback)

    try:
        # Initialize system
        if not await controller.initialize():
            raise RuntimeError("Failed to initialize controller")

        print("Adaptive controller system initialized")

        # Simulate workload
        for i in range(5):
            documents = [f"doc_{j}" for j in range(10 + i * 5)]  # Increasing workload

            result = await controller.process_batch(documents)
            print(f"Batch {i + 1}: {result}")

            # Check health
            health = await controller.health_check()
            print(f"System health: {health['overall_healthy']}")

            await asyncio.sleep(2)

        # Get final state
        state = controller.get_current_state()
        print(f"Final state: {state}")

        # Export metrics
        metrics = controller.export_metrics()
        print(f"Exported {len(metrics['performance_history'])} performance records")

    finally:
        # Shutdown
        await controller.shutdown()
        print("System shutdown complete")


if __name__ == "__main__":
    # Run example if Ray is available
    if RAY_AVAILABLE:
        asyncio.run(run_adaptive_system_example())
    else:
        print("Ray not available. Install with: pip install ray")
        print("This module provides industrial-grade adaptive scaling for Ray-based systems.")
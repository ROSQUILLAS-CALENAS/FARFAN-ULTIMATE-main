"""
Adaptive controller for Ray worker pools and processing window management
"""

import asyncio
import logging
import threading
import time
# # # from datetime import datetime, timedelta  # Module not found  # Module not found  # Module not found
# # # from typing import Any, Callable, Dict, List, Optional  # Module not found  # Module not found  # Module not found

import ray
# # # from ray.exceptions import RaySystemError  # Module not found  # Module not found  # Module not found
# # # from ray.util.state import list_nodes  # Module not found  # Module not found  # Module not found

# # # from .decision_engine import ScalingAction, ScalingDecision  # Module not found  # Module not found  # Module not found


class RayWorkerPool:
    """Ray-based worker pool management with dynamic scaling"""

    def __init__(
        self,
        worker_class: str = "ray.actor.Actor",
        initial_workers: int = 1,
        min_workers: int = 1,
        max_workers: int = 50,
    ):
        self.logger = logging.getLogger(__name__)
        self.worker_class = worker_class
        self.min_workers = min_workers
        self.max_workers = max_workers

        # Worker management
        self.active_workers: List[ray.ObjectRef] = []
        self.worker_tasks: Dict[ray.ObjectRef, str] = {}
        self.worker_creation_times: Dict[ray.ObjectRef, datetime] = {}

        # Scaling history
        self.scaling_history: List[Dict[str, Any]] = []

        # Thread safety
        self._scaling_lock = threading.RLock()
        self._is_initialized = False

        # Initialize with minimum workers
        self.target_workers = max(initial_workers, min_workers)

    async def initialize(self):
        """Initialize Ray cluster and create initial workers"""
        try:
            if not ray.is_initialized():
                ray.init(
                    ignore_reinit_error=True,
                    include_dashboard=False,
                    log_to_driver=False,
                )

            await self._scale_to_target(self.target_workers, "initialization")
            self._is_initialized = True
            self.logger.info(
                f"Initialized Ray worker pool with {len(self.active_workers)} workers"
            )

        except Exception as e:
            self.logger.error(f"Failed to initialize Ray worker pool: {e}")
            raise

    async def scale_workers(self, target_count: int, reason: str = "manual") -> bool:
        """Scale worker pool to target count"""
        if not self._is_initialized:
            await self.initialize()

        target_count = max(self.min_workers, min(target_count, self.max_workers))

        if target_count == len(self.active_workers):
            self.logger.debug(f"Worker count already at target: {target_count}")
            return True

        with self._scaling_lock:
            return await self._scale_to_target(target_count, reason)

    async def _scale_to_target(self, target_count: int, reason: str) -> bool:
        """Internal scaling implementation"""
        current_count = len(self.active_workers)

        try:
            if target_count > current_count:
                # Scale up
                workers_to_add = target_count - current_count
                await self._add_workers(workers_to_add)
                self.logger.info(
                    f"Scaled up by {workers_to_add} workers (reason: {reason})"
                )

            elif target_count < current_count:
                # Scale down
                workers_to_remove = current_count - target_count
                await self._remove_workers(workers_to_remove)
                self.logger.info(
                    f"Scaled down by {workers_to_remove} workers (reason: {reason})"
                )

            # Record scaling event
            self._record_scaling_event(current_count, target_count, reason)
            self.target_workers = target_count

            return True

        except Exception as e:
            self.logger.error(f"Failed to scale workers to {target_count}: {e}")
            return False

    async def _add_workers(self, count: int):
        """Add new workers to the pool"""
        for i in range(count):
            try:
                # Create worker actor (simplified - would need actual worker class)
                worker = ray.remote(self._create_worker).remote()

                self.active_workers.append(worker)
                self.worker_creation_times[worker] = datetime.now()
                self.worker_tasks[worker] = f"worker_{len(self.active_workers)}"

            except Exception as e:
                self.logger.error(f"Failed to create worker {i}: {e}")

    async def _remove_workers(self, count: int):
# # #         """Remove workers from the pool (graceful shutdown)"""  # Module not found  # Module not found  # Module not found
        workers_to_remove = self.active_workers[-count:]

        for worker in workers_to_remove:
            try:
                # Graceful shutdown (would call actual cleanup method)
                ray.cancel(worker, force=True)

                self.active_workers.remove(worker)
                self.worker_tasks.pop(worker, None)
                self.worker_creation_times.pop(worker, None)

            except Exception as e:
                self.logger.error(f"Failed to remove worker: {e}")

    def _create_worker(self) -> Any:
        """Create a basic worker (placeholder - implement actual worker logic)"""
        return {
            "id": f"worker_{time.time()}",
            "created_at": datetime.now(),
            "status": "active",
        }

    def _record_scaling_event(self, old_count: int, new_count: int, reason: str):
        """Record scaling event for analysis"""
        event = {
            "timestamp": datetime.now(),
            "old_count": old_count,
            "new_count": new_count,
            "change": new_count - old_count,
            "reason": reason,
        }

        self.scaling_history.append(event)

        # Keep only recent history
        if len(self.scaling_history) > 100:
            self.scaling_history = self.scaling_history[-100:]

    def get_worker_count(self) -> int:
        """Get current active worker count"""
        return len(self.active_workers)

    def get_worker_stats(self) -> Dict[str, Any]:
        """Get worker pool statistics"""
        now = datetime.now()

        uptime_stats = []
        for worker, created_time in self.worker_creation_times.items():
            uptime = (now - created_time).total_seconds()
            uptime_stats.append(uptime)

        return {
            "active_workers": len(self.active_workers),
            "target_workers": self.target_workers,
            "min_workers": self.min_workers,
            "max_workers": self.max_workers,
            "average_uptime": sum(uptime_stats) / len(uptime_stats)
            if uptime_stats
            else 0,
            "recent_scaling_events": len(
                [
                    e
                    for e in self.scaling_history
                    if (now - e["timestamp"]).total_seconds() < 3600  # Last hour
                ]
            ),
        }

    async def health_check(self) -> Dict[str, Any]:
        """Check health of Ray cluster and workers"""
        health = {
            "ray_initialized": ray.is_initialized(),
            "workers_healthy": True,
            "cluster_healthy": True,
            "worker_count": len(self.active_workers),
        }

        try:
            if ray.is_initialized():
                # Check cluster nodes
                nodes = list_nodes()
                healthy_nodes = [n for n in nodes if n.state == "ALIVE"]
                health["total_nodes"] = len(nodes)
                health["healthy_nodes"] = len(healthy_nodes)
                health["cluster_healthy"] = len(healthy_nodes) > 0

            # Check individual workers (simplified)
            unhealthy_workers = []
            for worker in self.active_workers:
                try:
                    # Would check worker status with actual worker class
                    pass
                except Exception as e:
                    unhealthy_workers.append(str(worker))

            health["workers_healthy"] = len(unhealthy_workers) == 0
            health["unhealthy_workers"] = unhealthy_workers

        except Exception as e:
            health["error"] = str(e)
            health["cluster_healthy"] = False

        return health

    async def shutdown(self):
        """Shutdown all workers and Ray cluster"""
        try:
            with self._scaling_lock:
                # Remove all workers
                await self._remove_workers(len(self.active_workers))

            if ray.is_initialized():
                ray.shutdown()

            self.logger.info("Ray worker pool shutdown complete")

        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")


class ProcessingWindowManager:
    """Manages processing window sizes and frequency adjustments"""

    def __init__(
        self,
        initial_window_size: int = 10,
        min_window_size: int = 1,
        max_window_size: int = 100,
        initial_frequency: float = 1.0,
    ):
        self.logger = logging.getLogger(__name__)

        # Window management
        self.current_window_size = initial_window_size
        self.min_window_size = min_window_size
        self.max_window_size = max_window_size

        # Frequency management
        self.current_frequency = initial_frequency  # Hz
        self.target_frequency = initial_frequency

        # Adjustment history
        self.adjustment_history: List[Dict[str, Any]] = []

        # Active tasks and windows
        self.active_windows: Dict[str, Dict[str, Any]] = {}
        self.window_performance: List[Dict[str, Any]] = []

    def adjust_window_size(self, new_size: int, reason: str = "adaptive") -> bool:
        """Adjust processing window size"""
        new_size = max(self.min_window_size, min(new_size, self.max_window_size))

        if new_size == self.current_window_size:
            return True

        old_size = self.current_window_size
        self.current_window_size = new_size

        # Record adjustment
        adjustment = {
            "timestamp": datetime.now(),
            "type": "window_size",
            "old_value": old_size,
            "new_value": new_size,
            "reason": reason,
        }

        self.adjustment_history.append(adjustment)
        self.logger.info(f"Adjusted window size: {old_size} -> {new_size} ({reason})")

        return True

    def adjust_frequency(self, new_frequency: float, reason: str = "adaptive") -> bool:
        """Adjust processing frequency"""
        if new_frequency <= 0:
            return False

        old_frequency = self.current_frequency
        self.current_frequency = new_frequency
        self.target_frequency = new_frequency

        # Record adjustment
        adjustment = {
            "timestamp": datetime.now(),
            "type": "frequency",
            "old_value": old_frequency,
            "new_value": new_frequency,
            "reason": reason,
        }

        self.adjustment_history.append(adjustment)
        self.logger.info(
            f"Adjusted frequency: {old_frequency:.2f}Hz -> {new_frequency:.2f}Hz ({reason})"
        )

        return True

    def create_processing_window(
        self, window_id: str, documents: List[str]
    ) -> Dict[str, Any]:
        """Create new processing window"""
        window = {
            "id": window_id,
            "documents": documents,
            "size": len(documents),
            "created_at": datetime.now(),
            "status": "active",
            "start_time": None,
            "end_time": None,
        }

        self.active_windows[window_id] = window
        return window

    def start_window_processing(self, window_id: str) -> bool:
        """Mark window as started processing"""
        if window_id in self.active_windows:
            self.active_windows[window_id]["start_time"] = datetime.now()
            self.active_windows[window_id]["status"] = "processing"
            return True
        return False

    def complete_window_processing(
        self, window_id: str, success_count: int, error_count: int
    ) -> bool:
        """Mark window as completed and record performance"""
        if window_id not in self.active_windows:
            return False

        window = self.active_windows[window_id]
        window["end_time"] = datetime.now()
        window["status"] = "completed"
        window["success_count"] = success_count
        window["error_count"] = error_count

        # Calculate performance metrics
        if window["start_time"]:
            processing_time = (
                window["end_time"] - window["start_time"]
            ).total_seconds()
            throughput = success_count / max(processing_time, 0.1)
            error_rate = error_count / max(window["size"], 1) * 100

            performance = {
                "window_id": window_id,
                "timestamp": window["end_time"],
                "window_size": window["size"],
                "processing_time": processing_time,
                "throughput": throughput,
                "error_rate": error_rate,
                "success_count": success_count,
                "error_count": error_count,
            }

            self.window_performance.append(performance)

            # Keep only recent performance data
            if len(self.window_performance) > 100:
                self.window_performance = self.window_performance[-100:]

        # Clean up completed window
        del self.active_windows[window_id]
        return True

    def get_optimal_window_size(self) -> int:
        """Calculate optimal window size based on recent performance"""
        if len(self.window_performance) < 5:
            return self.current_window_size

        # Find window size with best throughput/error_rate ratio
        performance_by_size = {}

        for perf in self.window_performance[-20:]:  # Recent 20 windows
            size = perf["window_size"]
            if size not in performance_by_size:
                performance_by_size[size] = []
            performance_by_size[size].append(perf)

        best_size = self.current_window_size
        best_score = 0

        for size, performances in performance_by_size.items():
            if len(performances) < 2:  # Need multiple samples
                continue

            avg_throughput = sum(p["throughput"] for p in performances) / len(
                performances
            )
            avg_error_rate = sum(p["error_rate"] for p in performances) / len(
                performances
            )

            # Score combines throughput and low error rate
            score = avg_throughput * (100 - avg_error_rate) / 100

            if score > best_score:
                best_score = score
                best_size = size

        return best_size

    def get_window_stats(self) -> Dict[str, Any]:
        """Get window processing statistics"""
        recent_performance = self.window_performance[-20:]

        stats = {
            "current_window_size": self.current_window_size,
            "current_frequency": self.current_frequency,
            "active_windows": len(self.active_windows),
            "recent_adjustments": len(
                [
                    adj
                    for adj in self.adjustment_history
                    if (datetime.now() - adj["timestamp"]).total_seconds() < 3600
                ]
            ),
        }

        if recent_performance:
            stats.update(
                {
                    "avg_throughput": sum(p["throughput"] for p in recent_performance)
                    / len(recent_performance),
                    "avg_error_rate": sum(p["error_rate"] for p in recent_performance)
                    / len(recent_performance),
                    "avg_processing_time": sum(
                        p["processing_time"] for p in recent_performance
                    )
                    / len(recent_performance),
                }
            )

        return stats


class AdaptiveController:
    """
    Main adaptive controller that coordinates Ray worker scaling and processing
    window adjustments based on decision engine recommendations.
    """

    def __init__(
        self,
        min_workers: int = 1,
        max_workers: int = 50,
        min_window_size: int = 1,
        max_window_size: int = 100,
        adjustment_cooldown: float = 30.0,  # seconds
    ):
        self.logger = logging.getLogger(__name__)

        # Component initialization
        self.worker_pool = RayWorkerPool(
            min_workers=min_workers, max_workers=max_workers
        )
        self.window_manager = ProcessingWindowManager(
            min_window_size=min_window_size, max_window_size=max_window_size
        )

        # Control parameters
        self.adjustment_cooldown = adjustment_cooldown
        self.last_adjustment = datetime.now() - timedelta(seconds=adjustment_cooldown)

        # Callbacks for system integration
        self.scaling_callbacks: List[Callable] = []

        # Control state
        self.is_active = False
        self._control_lock = threading.RLock()

    async def initialize(self):
        """Initialize controller components"""
        try:
            await self.worker_pool.initialize()
            self.is_active = True
            self.logger.info("Adaptive controller initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize adaptive controller: {e}")
            raise

    async def apply_scaling_decision(self, decision: ScalingDecision) -> bool:
# # #         """Apply scaling decision from decision engine"""  # Module not found  # Module not found  # Module not found
        if not self.is_active:
            self.logger.warning("Controller not active, ignoring scaling decision")
            return False

        # Check cooldown period
        now = datetime.now()
        if (now - self.last_adjustment).total_seconds() < self.adjustment_cooldown:
            self.logger.debug("Adjustment cooldown active, deferring scaling decision")
            return False

        # Only apply non-maintain decisions with sufficient confidence
        if decision.action == ScalingAction.MAINTAIN:
            return True

        if decision.confidence < 0.5:
            self.logger.debug(
                f"Decision confidence too low ({decision.confidence:.2f}), skipping"
            )
            return False

        success = True

        with self._control_lock:
            try:
                # Apply worker scaling
                current_workers = self.worker_pool.get_worker_count()
                if decision.target_workers != current_workers:
                    success &= await self.worker_pool.scale_workers(
                        decision.target_workers,
                        f"{decision.action.value}: {', '.join(decision.reasons)}",
                    )

                # Apply window size adjustment
                if (
                    decision.target_window_size
                    != self.window_manager.current_window_size
                ):
                    success &= self.window_manager.adjust_window_size(
                        decision.target_window_size,
                        f"{decision.action.value}: {', '.join(decision.reasons)}",
                    )

                # Apply frequency adjustment
                if (
                    abs(
                        decision.target_frequency
                        - self.window_manager.current_frequency
                    )
                    > 0.1
                ):
                    success &= self.window_manager.adjust_frequency(
                        decision.target_frequency,
                        f"{decision.action.value}: {', '.join(decision.reasons)}",
                    )

                if success:
                    self.last_adjustment = now
                    self.logger.info(
                        f"Successfully applied scaling decision: {decision.action.value}"
                    )

                    # Notify callbacks
                    for callback in self.scaling_callbacks:
                        try:
                            await callback(decision)
                        except Exception as e:
                            self.logger.error(f"Callback error: {e}")

                return success

            except Exception as e:
                self.logger.error(f"Failed to apply scaling decision: {e}")
                return False

    def add_scaling_callback(self, callback: Callable):
        """Add callback to be notified of scaling events"""
        self.scaling_callbacks.append(callback)

    def get_current_state(self) -> Dict[str, Any]:
        """Get current controller state"""
        return {
            "active": self.is_active,
            "worker_pool": self.worker_pool.get_worker_stats(),
            "window_manager": self.window_manager.get_window_stats(),
            "last_adjustment": self.last_adjustment.isoformat(),
            "cooldown_remaining": max(
                0,
                self.adjustment_cooldown
                - (datetime.now() - self.last_adjustment).total_seconds(),
            ),
        }

    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check"""
        health = {
            "controller_active": self.is_active,
            "worker_pool_health": await self.worker_pool.health_check(),
            "window_manager_health": {
                "active_windows": len(self.window_manager.active_windows),
                "recent_performance_data": len(self.window_manager.window_performance),
            },
        }

        return health

    async def shutdown(self):
        """Graceful shutdown of controller"""
        self.is_active = False

        try:
            await self.worker_pool.shutdown()
            self.logger.info("Adaptive controller shutdown complete")
        except Exception as e:
            self.logger.error(f"Error during controller shutdown: {e}")

    async def emergency_scale_down(self) -> bool:
        """Emergency scale down to minimal resources"""
        self.logger.warning("Executing emergency scale down")

        try:
            # Scale to minimum workers
            success = await self.worker_pool.scale_workers(
                self.worker_pool.min_workers, "emergency_scale_down"
            )

            # Reduce window size and frequency
            self.window_manager.adjust_window_size(
                self.window_manager.min_window_size, "emergency_scale_down"
            )

            self.window_manager.adjust_frequency(
                0.5, "emergency_scale_down"  # Half frequency
            )

            return success

        except Exception as e:
            self.logger.error(f"Emergency scale down failed: {e}")
            return False

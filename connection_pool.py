# # # from __future__ import annotations  # Module not found  # Module not found  # Module not found


# Mandatory Pipeline Contract Annotations
__phase__ = "O"
__code__ = "109O"
__stage_order__ = 7

"""
Production-grade, thread-safe, metrics-instrumented connection manager.

Key properties:
- Configurable via environment variables or by passing a ConnectionManagerConfig on first access.
  Env keys: DATABASE_URL, DB_POOL_SIZE, DB_MAX_OVERFLOW, DB_POOL_TIMEOUT,
            DB_ACQUIRE_TIMEOUT, HTTP_*, TP_* (captured generically).
- Front-gate bounded semaphore provides deterministic queue-length and wait-time
  measurements for capacity planning (M/M/c style analysis).
- Prefer using db_connection() context manager to ensure proper accounting and release.
- Safe under high concurrency: locks protect counters; bounded semaphore controls admission.

This manager optionally uses SQLAlchemy Engine if available; otherwise it yields
lightweight DummyConnection objects that no-op. This makes it safe in test and
non-DB environments while still providing admission control metrics.
"""

import os
import threading
import time
# # # from contextlib import contextmanager  # Module not found  # Module not found  # Module not found
# # # from dataclasses import dataclass, field  # Module not found  # Module not found  # Module not found
# # # from typing import Any, Callable, Dict, Optional  # Module not found  # Module not found  # Module not found

# Optional dependencies
try:  # SQLAlchemy is optional; we handle absence gracefully
    import sqlalchemy  # type: ignore
# # #     from sqlalchemy import create_engine  # type: ignore  # Module not found  # Module not found  # Module not found
# # #     from sqlalchemy.engine import Engine  # type: ignore  # Module not found  # Module not found  # Module not found
except Exception:  # pragma: no cover
    sqlalchemy = None  # type: ignore
    create_engine = None  # type: ignore
    Engine = Any  # type: ignore

try:  # Optional Prometheus client
# # #     from prometheus_client import Counter, Gauge  # type: ignore  # Module not found  # Module not found  # Module not found
except Exception:  # pragma: no cover
    Gauge = None  # type: ignore
    Counter = None  # type: ignore


@dataclass
class ConnectionManagerConfig:
    database_url: Optional[str] = None
    pool_size: int = 5
    max_overflow: int = 10
    pool_timeout: float = 30.0  # seconds for DB pool wait
    acquire_timeout: float = 10.0  # seconds for front-gate wait
    front_gate_capacity: Optional[int] = None  # default: pool_size + max_overflow
    pool_pre_ping: bool = True
    echo: bool = False
    # Pass-through dicts for HTTP_* and TP_* environment families
    http_config: Dict[str, Any] = field(default_factory=dict)
    tp_config: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_env(env: Optional[Dict[str, str]] = None) -> "ConnectionManagerConfig":
        e = env or os.environ

        def _int(name: str, default: int) -> int:
            try:
                return int(e.get(name, default))
            except Exception:
                return default

        def _float(name: str, default: float) -> float:
            try:
                return float(e.get(name, default))
            except Exception:
                return default

        http_cfg: Dict[str, Any] = {k: v for k, v in e.items() if k.startswith("HTTP_")}
        tp_cfg: Dict[str, Any] = {k: v for k, v in e.items() if k.startswith("TP_")}
        return ConnectionManagerConfig(
            database_url=e.get("DATABASE_URL") or e.get("DB_URL"),
            pool_size=_int("DB_POOL_SIZE", 5),
            max_overflow=_int("DB_MAX_OVERFLOW", 10),
            pool_timeout=_float("DB_POOL_TIMEOUT", 30.0),
            acquire_timeout=_float("DB_ACQUIRE_TIMEOUT", 10.0),
            front_gate_capacity=_int("DB_FRONT_GATE_CAPACITY", -1),
            pool_pre_ping=(e.get("DB_POOL_PRE_PING", "1") == "1"),
            echo=(e.get("SQL_ECHO", "0") == "1"),
            http_config=http_cfg,
            tp_config=tp_cfg,
        )


class DummyConnection:
    """A lightweight connection placeholder when no DB engine is available."""

    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


class ConnectionManager:
    def __init__(self, cfg: ConnectionManagerConfig) -> None:
        self.cfg = cfg
        # Front-gate admission capacity defaults to pool_size + max_overflow
        capacity = (
            cfg.front_gate_capacity
            if (cfg.front_gate_capacity and cfg.front_gate_capacity > 0)
            else (cfg.pool_size + cfg.max_overflow)
        )
        self._gate = threading.BoundedSemaphore(value=max(1, capacity))
        # Locks and counters
        self._lock = threading.Lock()
        self._in_use = 0
        self._total_acquires = 0
        self._total_releases = 0
        self._timeouts = 0
        self._current_waiters = 0
        # Wait-time stats (EMA + simple aggregates)
        self._wait_count = 0
        self._wait_total = 0.0
        self._wait_ema = 0.0
        self._wait_ema_alpha = 0.2
        self._wait_p95 = 0.0  # approximate via max of last-N bounded tracking
        self._last_waits: list[float] = []
        self._last_waits_max = 256
        # Optional Prometheus
        self._prom: Dict[str, Any] = {}
        self._init_prometheus()
        # SQLAlchemy engine (optional)
        self._engine: Optional[Engine] = None
        self._init_engine()

    # ------------- Initialization helpers -------------
    def _init_prometheus(self) -> None:
        if Gauge is None or Counter is None:
            return
        try:
            self._prom["queue_length"] = Gauge(
                "conn_front_gate_queue_length", "Current front-gate waiters"
            )
            self._prom["in_use"] = Gauge(
                "conn_in_use", "Connections in use (front-gate admitted)"
            )
            self._prom["acquires_total"] = Counter(
                "conn_acquires_total", "Total successful admissions"
            )
            self._prom["timeouts_total"] = Counter(
                "conn_timeouts_total", "Total admission timeouts"
            )
            self._prom["wait_time_seconds"] = Gauge(
                "conn_wait_time_seconds_ema", "Admission wait time EMA (seconds)"
            )
        except Exception:
            self._prom.clear()

    def _init_engine(self) -> None:
        if not self.cfg.database_url or create_engine is None:
            self._engine = None
            return
        try:
            self._engine = create_engine(
                self.cfg.database_url,
                pool_size=self.cfg.pool_size,
                max_overflow=self.cfg.max_overflow,
                pool_timeout=self.cfg.pool_timeout,
                pool_pre_ping=self.cfg.pool_pre_ping,
                echo=self.cfg.echo,
                future=True,
            )
        except Exception:
            # Fallback to no engine; metrics still work
            self._engine = None

    # ------------- Public API -------------
    @contextmanager
    def db_connection(self):
        """
        Admission-controlled DB connection scope. Always prefer using this context manager
        to ensure proper accounting and release, even if the underlying engine is absent.
        """
        start = time.perf_counter()
        # Account waiter deterministically before attempting acquire
        with self._lock:
            self._current_waiters += 1
            self._update_prom_queue()
        acquired = self._gate.acquire(timeout=self.cfg.acquire_timeout)
        wait = time.perf_counter() - start
        # Decrement waiter count as we either admitted or timed out
        with self._lock:
            self._current_waiters = max(0, self._current_waiters - 1)
            self._update_prom_queue()
        if not acquired:
            with self._lock:
                self._timeouts += 1
                if self._prom.get("timeouts_total"):
                    try:
                        self._prom["timeouts_total"].inc()
                    except Exception:
                        pass
            # Surface a timeout error
            raise TimeoutError(
                f"Connection front-gate acquire timed out after {self.cfg.acquire_timeout}s"
            )

        # Update admission metrics
        with self._lock:
            self._in_use += 1
            self._total_acquires += 1
            # Wait-time stats
            self._wait_count += 1
            self._wait_total += wait
            self._wait_ema = (
                (1 - self._wait_ema_alpha) * self._wait_ema
                + self._wait_ema_alpha * wait
                if self._wait_count > 1
                else wait
            )
            self._last_waits.append(wait)
            if len(self._last_waits) > self._last_waits_max:
                self._last_waits.pop(0)
            self._wait_p95 = (
                sorted(self._last_waits)[int(0.95 * (len(self._last_waits) - 1))]
                if self._last_waits
                else 0.0
            )
            # Prometheus
            if self._prom.get("acquires_total"):
                try:
                    self._prom["acquires_total"].inc()
                except Exception:
                    pass
            if self._prom.get("in_use"):
                try:
                    self._prom["in_use"].set(self._in_use)
                except Exception:
                    pass
            if self._prom.get("wait_time_seconds"):
                try:
                    self._prom["wait_time_seconds"].set(self._wait_ema)
                except Exception:
                    pass

        # Provide an actual DB connection if engine is available
        conn = None
        try:
            if self._engine is not None:
                # SQLAlchemy connection (pool_timeout applies at engine level)
                conn = self._engine.connect()
            else:
                conn = DummyConnection()
            yield conn
        finally:
            # Ensure connection closed and front-gate released
            try:
                if conn is not None:
                    try:
                        conn.close()  # type: ignore[attr-defined]
                    except Exception:
                        pass
            finally:
                with self._lock:
                    self._in_use = max(0, self._in_use - 1)
                    self._total_releases += 1
                    if self._prom.get("in_use"):
                        try:
                            self._prom["in_use"].set(self._in_use)
                        except Exception:
                            pass
                try:
                    self._gate.release()
                except Exception:
                    # Should not happen with BoundedSemaphore if balanced
                    pass

    def current_metrics(self) -> Dict[str, Any]:
        """Return a snapshot of the manager's counters and wait stats."""
        with self._lock:
            avg_wait = (
                (self._wait_total / self._wait_count) if self._wait_count else 0.0
            )
            metrics = {
                "pool": {
                    "configured_pool_size": self.cfg.pool_size,
                    "configured_max_overflow": self.cfg.max_overflow,
                    "front_gate_capacity": (
                        self.cfg.front_gate_capacity
                        if (
                            self.cfg.front_gate_capacity
                            and self.cfg.front_gate_capacity > 0
                        )
                        else (self.cfg.pool_size + self.cfg.max_overflow)
                    ),
                },
                "counts": {
                    "current_waiters": self._current_waiters,
                    "in_use": self._in_use,
                    "total_acquires": self._total_acquires,
                    "total_releases": self._total_releases,
                    "timeouts": self._timeouts,
                },
                "wait_times": {
                    "ema_seconds": self._wait_ema,
                    "avg_seconds": avg_wait,
                    "p95_seconds": self._wait_p95,
                },
                "engine": {
                    "available": bool(self._engine is not None),
                    "url_present": bool(self.cfg.database_url is not None),
                },
                "http": self.cfg.http_config,
                "threadpool": self.cfg.tp_config,
            }
        return metrics

    # Convenience helpers
    def engine(self) -> Optional[Any]:
        return self._engine

    def _update_prom_queue(self) -> None:
        if self._prom.get("queue_length"):
            try:
                self._prom["queue_length"].set(self._current_waiters)
            except Exception:
                pass


# ------------- Singleton access -------------
_singleton_lock = threading.Lock()
_singleton_instance: Optional[ConnectionManager] = None


def get_connection_manager(
    config: Optional[ConnectionManagerConfig] = None,
) -> ConnectionManager:
    """
    Return a singleton ConnectionManager. On first access, you can pass a
    ConnectionManagerConfig to override environment-based defaults.
    Subsequent calls ignore config and return the existing instance.
    """
    global _singleton_instance
    if _singleton_instance is not None:
        return _singleton_instance
    with _singleton_lock:
        if _singleton_instance is None:
            cfg = config or ConnectionManagerConfig.from_env()
            _singleton_instance = ConnectionManager(cfg)
        return _singleton_instance


__all__ = [
    "ConnectionManagerConfig",
    "ConnectionManager",
    "get_connection_manager",
]

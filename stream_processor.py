"""
Stream processing component using Kafka for real-time window adjustment and backpressure detection
"""

import asyncio
import json
import logging
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Set

try:
    from kafka import KafkaConsumer, KafkaProducer
    from kafka.errors import KafkaError

    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False
    KafkaProducer = KafkaConsumer = None


@dataclass
class StreamEvent:
    """Base class for stream processing events"""

    event_id: str
    timestamp: datetime
    event_type: str
    source: str
    data: Dict[str, Any]

    def to_json(self) -> str:
        """Serialize to JSON string"""
        data_copy = asdict(self)
        data_copy["timestamp"] = self.timestamp.isoformat()
        return json.dumps(data_copy)

    @classmethod
    def from_json(cls, json_str: str) -> "StreamEvent":
        """Deserialize from JSON string"""
        data = json.loads(json_str)
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


@dataclass
class ProcessingEvent(StreamEvent):
    """Processing pipeline events"""

    document_id: str
    stage: str  # started, completed, failed
    processing_time: Optional[float] = None
    error_message: Optional[str] = None

    def __post_init__(self):
        self.event_type = "processing"


@dataclass
class MetricsEvent(StreamEvent):
    """Metrics update events"""

    metrics_type: str  # system, processing, workload
    values: Dict[str, float]

    def __post_init__(self):
        self.event_type = "metrics"


@dataclass
class ScalingEvent(StreamEvent):
    """Scaling action events"""

    action: str
    old_workers: int
    new_workers: int
    old_window_size: int
    new_window_size: int
    reasons: List[str]

    def __post_init__(self):
        self.event_type = "scaling"


class BackpressureDetector:
    """Detects backpressure conditions in the processing pipeline"""

    def __init__(
        self,
        queue_depth_threshold: int = 50,
        latency_threshold: float = 5000,  # ms
        error_rate_threshold: float = 10.0,  # percentage
        detection_window: int = 60,  # seconds
    ):
        self.queue_depth_threshold = queue_depth_threshold
        self.latency_threshold = latency_threshold
        self.error_rate_threshold = error_rate_threshold
        self.detection_window = detection_window

        # Tracking data
        self.recent_events: deque = deque(maxlen=1000)
        self.queue_depth_history: deque = deque(maxlen=100)
        self.latency_history: deque = deque(maxlen=100)
        self.error_count_window: deque = deque(maxlen=100)

        # Backpressure state
        self.is_backpressure_detected = False
        self.backpressure_start_time: Optional[datetime] = None
        self.backpressure_severity = 0.0  # 0-1 scale

    def update_metrics(self, metrics: Dict[str, Any]):
        """Update backpressure detection with new metrics"""
        now = datetime.now()

        # Extract relevant metrics
        queue_depth = metrics.get("queue_depth", 0)
        latency_p95 = metrics.get("latency_p95", 0)
        error_rate = metrics.get("error_rate", 0)

        # Update history
        self.queue_depth_history.append((now, queue_depth))
        self.latency_history.append((now, latency_p95))
        self.error_count_window.append((now, error_rate))

        # Detect backpressure conditions
        backpressure_indicators = []

        # Queue depth check
        if queue_depth > self.queue_depth_threshold:
            backpressure_indicators.append(f"High queue depth: {queue_depth}")

        # Latency check
        if latency_p95 > self.latency_threshold:
            backpressure_indicators.append(f"High latency: {latency_p95:.0f}ms")

        # Error rate check
        if error_rate > self.error_rate_threshold:
            backpressure_indicators.append(f"High error rate: {error_rate:.1f}%")

        # Trend analysis
        if self._is_trending_up("queue_depth"):
            backpressure_indicators.append("Queue depth trending up")

        if self._is_trending_up("latency"):
            backpressure_indicators.append("Latency trending up")

        # Update backpressure state
        if backpressure_indicators:
            if not self.is_backpressure_detected:
                self.is_backpressure_detected = True
                self.backpressure_start_time = now

            # Calculate severity based on multiple factors
            severity = 0.0
            if queue_depth > self.queue_depth_threshold:
                severity += min(0.4, queue_depth / self.queue_depth_threshold * 0.4)
            if latency_p95 > self.latency_threshold:
                severity += min(0.3, latency_p95 / self.latency_threshold * 0.3)
            if error_rate > self.error_rate_threshold:
                severity += min(0.3, error_rate / self.error_rate_threshold * 0.3)

            self.backpressure_severity = min(1.0, severity)

        else:
            if self.is_backpressure_detected:
                # Check if backpressure has been resolved for sufficient time
                if (
                    self.backpressure_start_time
                    and (now - self.backpressure_start_time).total_seconds() > 30
                ):
                    self.is_backpressure_detected = False
                    self.backpressure_start_time = None
                    self.backpressure_severity = 0.0

    def _is_trending_up(self, metric_type: str) -> bool:
        """Check if metric is trending upward"""
        if metric_type == "queue_depth":
            history = list(self.queue_depth_history)
        elif metric_type == "latency":
            history = list(self.latency_history)
        else:
            return False

        if len(history) < 5:
            return False

        # Simple trend detection: compare recent average to older average
        recent_values = [v for _, v in history[-5:]]
        older_values = [v for _, v in history[-10:-5]]

        if not older_values:
            return False

        recent_avg = sum(recent_values) / len(recent_values)
        older_avg = sum(older_values) / len(older_values)

        return recent_avg > older_avg * 1.2  # 20% increase threshold

    def get_backpressure_status(self) -> Dict[str, Any]:
        """Get current backpressure status"""
        return {
            "detected": self.is_backpressure_detected,
            "severity": self.backpressure_severity,
            "duration": (
                (datetime.now() - self.backpressure_start_time).total_seconds()
                if self.backpressure_start_time
                else 0
            ),
            "current_queue_depth": (
                self.queue_depth_history[-1][1] if self.queue_depth_history else 0
            ),
            "current_latency": (
                self.latency_history[-1][1] if self.latency_history else 0
            ),
        }


class StreamProcessor:
    """
    Kafka-based stream processor for real-time window adjustment and backpressure detection
    """

    def __init__(
        self,
        kafka_bootstrap_servers: str = "localhost:9092",
        consumer_group: str = "adaptive_feedback",
        topics: Optional[Dict[str, str]] = None,
        enable_kafka: bool = True,
    ):
        self.logger = logging.getLogger(__name__)
        self.kafka_servers = kafka_bootstrap_servers
        self.consumer_group = consumer_group
        self.enable_kafka = enable_kafka and KAFKA_AVAILABLE

        # Topic configuration
        self.topics = topics or {
            "metrics": "pdt.metrics",
            "processing": "pdt.processing",
            "scaling": "pdt.scaling",
            "alerts": "pdt.alerts",
        }

        # Kafka components
        self.producer: Optional[KafkaProducer] = None
        self.consumers: Dict[str, KafkaConsumer] = {}

        # Stream processing components
        self.backpressure_detector = BackpressureDetector()

        # Event handling
        self.event_handlers: Dict[str, List[Callable]] = {
            "metrics": [],
            "processing": [],
            "scaling": [],
            "backpressure": [],
        }

        # Processing state
        self.is_running = False
        self.consumer_tasks: List[asyncio.Task] = []
        self._processing_lock = threading.RLock()

        # In-memory fallback (when Kafka unavailable)
        self.event_buffer: deque = deque(maxlen=10000)

        if not self.enable_kafka:
            self.logger.warning("Kafka not available, using in-memory processing")

    async def initialize(self):
        """Initialize stream processing components"""
        if self.enable_kafka:
            try:
                await self._initialize_kafka()
                self.logger.info("Stream processor initialized with Kafka")
            except Exception as e:
                self.logger.error(f"Failed to initialize Kafka: {e}")
                self.enable_kafka = False
                self.logger.info("Falling back to in-memory stream processing")
        else:
            self.logger.info("Stream processor initialized in memory-only mode")

        # Start processing loop
        await self.start_processing()

    async def _initialize_kafka(self):
        """Initialize Kafka producer and consumers"""
        # Initialize producer
        self.producer = KafkaProducer(
            bootstrap_servers=self.kafka_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            key_serializer=lambda k: str(k).encode("utf-8"),
            acks="all",
            retries=3,
            max_block_ms=5000,
        )

        # Initialize consumers for each topic
        for topic_type, topic_name in self.topics.items():
            consumer = KafkaConsumer(
                topic_name,
                bootstrap_servers=self.kafka_servers,
                group_id=self.consumer_group,
                value_deserializer=lambda m: json.loads(m.decode("utf-8")),
                auto_offset_reset="latest",
                enable_auto_commit=True,
                consumer_timeout_ms=1000,
            )
            self.consumers[topic_type] = consumer

    async def start_processing(self):
        """Start stream processing tasks"""
        if self.is_running:
            return

        self.is_running = True

        if self.enable_kafka:
            # Start consumer tasks for each topic
            for topic_type, consumer in self.consumers.items():
                task = asyncio.create_task(self._consume_topic(topic_type, consumer))
                self.consumer_tasks.append(task)
        else:
            # Start in-memory processing task
            task = asyncio.create_task(self._process_memory_events())
            self.consumer_tasks.append(task)

        # Start backpressure monitoring
        backpressure_task = asyncio.create_task(self._monitor_backpressure())
        self.consumer_tasks.append(backpressure_task)

        self.logger.info("Stream processing started")

    async def _consume_topic(self, topic_type: str, consumer: KafkaConsumer):
        """Consume messages from a specific Kafka topic"""
        while self.is_running:
            try:
                message_batch = consumer.poll(timeout_ms=1000)

                for topic_partition, messages in message_batch.items():
                    for message in messages:
                        try:
                            await self._process_stream_event(
                                topic_type, message.value, message.timestamp
                            )
                        except Exception as e:
                            self.logger.error(f"Error processing message: {e}")

            except Exception as e:
                self.logger.error(f"Error consuming from topic {topic_type}: {e}")
                await asyncio.sleep(5)  # Back off on errors

    async def _process_memory_events(self):
        """Process events from in-memory buffer"""
        while self.is_running:
            try:
                if self.event_buffer:
                    with self._processing_lock:
                        events_to_process = list(self.event_buffer)
                        self.event_buffer.clear()

                    for event_data in events_to_process:
                        await self._process_stream_event(
                            event_data["topic_type"],
                            event_data["data"],
                            event_data["timestamp"],
                        )
                else:
                    await asyncio.sleep(0.1)  # Short sleep when no events

            except Exception as e:
                self.logger.error(f"Error in memory event processing: {e}")

    async def _process_stream_event(
        self, topic_type: str, event_data: Dict, timestamp: float
    ):
        """Process a single stream event"""
        try:
            # Update backpressure detector with metrics
            if topic_type == "metrics":
                self.backpressure_detector.update_metrics(event_data)

            # Call registered handlers
            handlers = self.event_handlers.get(topic_type, [])
            for handler in handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(event_data)
                    else:
                        handler(event_data)
                except Exception as e:
                    self.logger.error(f"Handler error for {topic_type}: {e}")

        except Exception as e:
            self.logger.error(f"Error processing stream event: {e}")

    async def _monitor_backpressure(self):
        """Monitor and respond to backpressure conditions"""
        while self.is_running:
            try:
                backpressure_status = (
                    self.backpressure_detector.get_backpressure_status()
                )

                # Trigger backpressure handlers if conditions detected
                if backpressure_status["detected"]:
                    handlers = self.event_handlers.get("backpressure", [])
                    for handler in handlers:
                        try:
                            if asyncio.iscoroutinefunction(handler):
                                await handler(backpressure_status)
                            else:
                                handler(backpressure_status)
                        except Exception as e:
                            self.logger.error(f"Backpressure handler error: {e}")

                await asyncio.sleep(5)  # Check every 5 seconds

            except Exception as e:
                self.logger.error(f"Error in backpressure monitoring: {e}")

    async def publish_event(self, topic_type: str, event: StreamEvent) -> bool:
        """Publish event to stream"""
        try:
            event_data = {
                "event_id": event.event_id,
                "timestamp": event.timestamp.isoformat(),
                "event_type": event.event_type,
                "source": event.source,
                "data": event.data,
            }

            if self.enable_kafka and self.producer:
                # Send to Kafka
                topic_name = self.topics.get(topic_type, f"pdt.{topic_type}")
                future = self.producer.send(
                    topic_name, value=event_data, key=event.event_id
                )

                # Don't block on send, but log errors
                def on_send_error(excp):
                    self.logger.error(f"Failed to send event to {topic_name}: {excp}")

                future.add_errback(on_send_error)
                return True

            else:
                # Store in memory buffer
                with self._processing_lock:
                    self.event_buffer.append(
                        {
                            "topic_type": topic_type,
                            "data": event_data,
                            "timestamp": time.time(),
                        }
                    )
                return True

        except Exception as e:
            self.logger.error(f"Failed to publish event: {e}")
            return False

    def add_event_handler(self, topic_type: str, handler: Callable):
        """Add event handler for specific topic type"""
        if topic_type not in self.event_handlers:
            self.event_handlers[topic_type] = []
        self.event_handlers[topic_type].append(handler)

    def remove_event_handler(self, topic_type: str, handler: Callable):
        """Remove event handler"""
        if topic_type in self.event_handlers:
            try:
                self.event_handlers[topic_type].remove(handler)
            except ValueError:
                pass

    async def publish_metrics_event(self, metrics: Dict[str, Any]) -> bool:
        """Convenience method to publish metrics event"""
        event = MetricsEvent(
            event_id=f"metrics_{int(time.time())}",
            timestamp=datetime.now(),
            source="metrics_collector",
            data=metrics,
            metrics_type="combined",
            values={},  # Would be populated with specific metric values
        )
        return await self.publish_event("metrics", event)

    async def publish_processing_event(
        self,
        document_id: str,
        stage: str,
        processing_time: Optional[float] = None,
        error: Optional[str] = None,
    ) -> bool:
        """Convenience method to publish processing event"""
        event = ProcessingEvent(
            event_id=f"processing_{document_id}_{int(time.time())}",
            timestamp=datetime.now(),
            source="processing_pipeline",
            data={},
            document_id=document_id,
            stage=stage,
            processing_time=processing_time,
            error_message=error,
        )
        return await self.publish_event("processing", event)

    async def publish_scaling_event(
        self,
        action: str,
        old_workers: int,
        new_workers: int,
        old_window: int,
        new_window: int,
        reasons: List[str],
    ) -> bool:
        """Convenience method to publish scaling event"""
        event = ScalingEvent(
            event_id=f"scaling_{int(time.time())}",
            timestamp=datetime.now(),
            source="adaptive_controller",
            data={},
            action=action,
            old_workers=old_workers,
            new_workers=new_workers,
            old_window_size=old_window,
            new_window_size=new_window,
            reasons=reasons,
        )
        return await self.publish_event("scaling", event)

    def get_backpressure_status(self) -> Dict[str, Any]:
        """Get current backpressure detection status"""
        return self.backpressure_detector.get_backpressure_status()

    async def health_check(self) -> Dict[str, Any]:
        """Health check for stream processor"""
        health = {
            "stream_processor_active": self.is_running,
            "kafka_enabled": self.enable_kafka,
            "backpressure_detected": self.backpressure_detector.is_backpressure_detected,
            "consumer_tasks_running": len(
                [t for t in self.consumer_tasks if not t.done()]
            ),
            "event_handlers_registered": {
                topic: len(handlers) for topic, handlers in self.event_handlers.items()
            },
        }

        if self.enable_kafka:
            health["kafka_producer_ready"] = self.producer is not None
            health["kafka_consumers"] = len(self.consumers)
        else:
            health["memory_buffer_size"] = len(self.event_buffer)

        return health

    async def shutdown(self):
        """Shutdown stream processor"""
        self.is_running = False

        # Cancel consumer tasks
        for task in self.consumer_tasks:
            task.cancel()

        # Wait for tasks to complete
        if self.consumer_tasks:
            try:
                await asyncio.gather(*self.consumer_tasks, return_exceptions=True)
            except Exception as e:
                self.logger.error(f"Error during task shutdown: {e}")

        # Close Kafka connections
        if self.enable_kafka:
            if self.producer:
                self.producer.close()

            for consumer in self.consumers.values():
                consumer.close()

        self.logger.info("Stream processor shutdown complete")

"""
Main feedback loop coordinator that integrates all adaptive system components
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
import threading
import json

from metrics_collector import MetricsCollector, SystemMetrics, ProcessingMetrics, WorkloadMetrics
from decision_engine import DecisionEngine, ScalingDecision, ScalingAction, ThresholdConfiguration
from adaptive_controller import AdaptiveController
from stream_processor import StreamProcessor, ProcessingEvent


class FeedbackLoopConfiguration:
    """Configuration for the adaptive feedback loop"""
    
    def __init__(
        self,
        # Loop timing
        feedback_interval: float = 10.0,  # seconds
        metrics_collection_interval: float = 5.0,
        decision_cooldown: float = 30.0,
        
        # Thresholds
        thresholds: Optional[ThresholdConfiguration] = None,
        
        # System bounds
        min_workers: int = 1,
        max_workers: int = 50,
        min_window_size: int = 1,
        max_window_size: int = 100,
        
        # Integration settings
        enable_stream_processing: bool = True,
        enable_prometheus: bool = True,
        kafka_servers: str = "localhost:9092",
        redis_url: str = "redis://localhost:6379",
        
        # Safety settings
        enable_emergency_scaling: bool = True,
        max_scaling_events_per_hour: int = 10
    ):
        self.feedback_interval = feedback_interval
        self.metrics_collection_interval = metrics_collection_interval
        self.decision_cooldown = decision_cooldown
        
        self.thresholds = thresholds or ThresholdConfiguration()
        
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.min_window_size = min_window_size
        self.max_window_size = max_window_size
        
        self.enable_stream_processing = enable_stream_processing
        self.enable_prometheus = enable_prometheus
        self.kafka_servers = kafka_servers
        self.redis_url = redis_url
        
        self.enable_emergency_scaling = enable_emergency_scaling
        self.max_scaling_events_per_hour = max_scaling_events_per_hour
        
    def validate(self) -> bool:
        """Validate configuration parameters"""
        return (
            self.feedback_interval > 0 and
            self.metrics_collection_interval > 0 and
            self.decision_cooldown > 0 and
            self.min_workers <= self.max_workers and
            self.min_window_size <= self.max_window_size and
            self.thresholds.validate()
        )


class FeedbackLoop:
    """
    Main adaptive feedback control system that coordinates all components to provide
    real-time performance optimization through automatic scaling and adjustment.
    """
    
    def __init__(self, config: FeedbackLoopConfiguration):
        if not config.validate():
            raise ValueError("Invalid feedback loop configuration")
            
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Core components
        self.metrics_collector = MetricsCollector(
            redis_url=config.redis_url,
            collection_interval=config.metrics_collection_interval,
            enable_prometheus=config.enable_prometheus
        )
        
        self.decision_engine = DecisionEngine(
            thresholds=config.thresholds
        )
        
        self.adaptive_controller = AdaptiveController(
            min_workers=config.min_workers,
            max_workers=config.max_workers,
            min_window_size=config.min_window_size,
            max_window_size=config.max_window_size,
            adjustment_cooldown=config.decision_cooldown
        )
        
        # Stream processing (optional)
        self.stream_processor = None
        if config.enable_stream_processing:
            self.stream_processor = StreamProcessor(
                kafka_bootstrap_servers=config.kafka_servers
            )
            
        # Feedback loop state
        self.is_running = False
        self.feedback_task: Optional[asyncio.Task] = None
        self.last_feedback_cycle = datetime.now()
        
        # Performance tracking
        self.cycle_metrics: List[Dict[str, Any]] = []
        self.scaling_events_history: List[Dict[str, Any]] = []
        
        # Safety mechanisms
        self.recent_scaling_events = 0
        self.scaling_events_reset_time = datetime.now()
        
        # Integration callbacks
        self.processing_callbacks: List[Callable] = []
        self.scaling_callbacks: List[Callable] = []
        
        # Thread safety
        self._state_lock = threading.RLock()
        
    async def initialize(self):
        """Initialize all feedback loop components"""
        try:
            self.logger.info("Initializing adaptive feedback control system...")
            
            # Initialize core components
            await self.metrics_collector.initialize()
            await self.adaptive_controller.initialize()
            
            # Initialize stream processor if enabled
            if self.stream_processor:
                await self.stream_processor.initialize()
                self._setup_stream_handlers()
                
            # Register callbacks
            self.adaptive_controller.add_scaling_callback(self._on_scaling_event)
            
            self.logger.info("Adaptive feedback control system initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize feedback loop: {e}")
            raise
            
    def _setup_stream_handlers(self):
        """Setup stream processing event handlers"""
        if not self.stream_processor:
            return
            
        # Metrics events
        self.stream_processor.add_event_handler(
            'metrics', 
            self._handle_stream_metrics
        )
        
        # Processing events  
        self.stream_processor.add_event_handler(
            'processing',
            self._handle_stream_processing
        )
        
        # Backpressure events
        self.stream_processor.add_event_handler(
            'backpressure',
            self._handle_backpressure
        )
        
    async def _handle_stream_metrics(self, event_data: Dict[str, Any]):
        """Handle metrics events from stream"""
        # Forward to decision engine for real-time analysis
        self.decision_engine.update_metrics(event_data)
        
    async def _handle_stream_processing(self, event_data: Dict[str, Any]):
        """Handle processing events from stream"""
        # Record processing events in metrics collector
        document_id = event_data.get('document_id', 'unknown')
        stage = event_data.get('stage', 'unknown')
        
        if stage == 'started':
            self.metrics_collector.record_processing_start(document_id)
        elif stage == 'completed':
            self.metrics_collector.record_processing_success(document_id)
        elif stage == 'failed':
            error_type = event_data.get('error_message', 'unknown_error')
            self.metrics_collector.record_processing_error(document_id, error_type)
            
    async def _handle_backpressure(self, backpressure_status: Dict[str, Any]):
        """Handle backpressure detection events"""
        if backpressure_status['detected'] and backpressure_status['severity'] > 0.7:
            self.logger.warning(f"High backpressure detected: {backpressure_status}")
            
            # Trigger immediate feedback cycle for emergency response
            if self.config.enable_emergency_scaling:
                await self._emergency_response(backpressure_status)
                
    async def _emergency_response(self, backpressure_status: Dict[str, Any]):
        """Handle emergency backpressure conditions"""
        self.logger.warning("Executing emergency response to backpressure")
        
        try:
            # Force an immediate feedback cycle with emergency parameters
            await self._execute_feedback_cycle(emergency=True)
            
        except Exception as e:
            self.logger.error(f"Emergency response failed: {e}")
            
    async def start(self):
        """Start the adaptive feedback loop"""
        if self.is_running:
            self.logger.warning("Feedback loop already running")
            return
            
        with self._state_lock:
            self.is_running = True
            self.feedback_task = asyncio.create_task(self._feedback_loop())
            
        self.logger.info(f"Adaptive feedback loop started (interval: {self.config.feedback_interval}s)")
        
    async def stop(self):
        """Stop the feedback loop gracefully"""
        with self._state_lock:
            if not self.is_running:
                return
                
            self.is_running = False
            
        if self.feedback_task:
            self.feedback_task.cancel()
            try:
                await self.feedback_task
            except asyncio.CancelledError:
                pass
                
        self.logger.info("Adaptive feedback loop stopped")
        
    async def _feedback_loop(self):
        """Main feedback loop implementation"""
        while self.is_running:
            try:
                cycle_start = datetime.now()
                
                # Execute feedback cycle
                await self._execute_feedback_cycle()
                
                # Track cycle performance
                cycle_duration = (datetime.now() - cycle_start).total_seconds()
                self._record_cycle_metrics(cycle_duration)
                
                self.last_feedback_cycle = datetime.now()
                
                # Wait for next cycle
                await asyncio.sleep(self.config.feedback_interval)
                
            except Exception as e:
                self.logger.error(f"Error in feedback loop: {e}")
                # Continue running even if individual cycles fail
                await asyncio.sleep(self.config.feedback_interval)
                
    async def _execute_feedback_cycle(self, emergency: bool = False):
        """Execute a single feedback cycle"""
        try:
            # 1. Collect current metrics from all sources
            metrics = await self._collect_current_metrics()
            
            # 2. Update decision engine with latest metrics
            self.decision_engine.update_metrics(metrics)
            
            # 3. Analyze performance patterns and generate decision
            decision = self.decision_engine.generate_scaling_decision()
            
            # 4. Apply decision if confidence threshold met
            confidence_threshold = 0.3 if emergency else 0.5
            if decision.confidence >= confidence_threshold:
                # Check scaling rate limits (unless emergency)
                if emergency or self._can_execute_scaling():
                    success = await self.adaptive_controller.apply_scaling_decision(decision)
                    
                    if success:
                        # Update decision engine with new state
                        controller_state = self.adaptive_controller.get_current_state()
                        self.decision_engine.update_system_state(
                            controller_state['worker_pool']['active_workers'],
                            controller_state['window_manager']['current_window_size'],
                            controller_state['window_manager']['current_frequency']
                        )
                        
                        # Record scaling event
                        await self._record_scaling_event(decision, success)
                        
                        # Publish scaling event to stream
                        if self.stream_processor:
                            await self.stream_processor.publish_scaling_event(
                                decision.action.value,
                                decision.target_workers,
                                decision.target_workers,  # Would track actual old values
                                decision.target_window_size,
                                decision.target_window_size,  # Would track actual old values
                                decision.reasons
                            )
                else:
                    self.logger.debug("Scaling rate limited, deferring decision")
            else:
                self.logger.debug(f"Decision confidence too low: {decision.confidence:.2f}")
                
            # 5. Publish metrics to stream if available
            if self.stream_processor:
                await self.stream_processor.publish_metrics_event(metrics)
                
        except Exception as e:
            self.logger.error(f"Error in feedback cycle: {e}")
            raise
            
    async def _collect_current_metrics(self) -> Dict[str, Any]:
        """Collect current metrics from all sources"""
        metrics = {}
        
        try:
            # Get recent metrics from collector
            system_metrics = self.metrics_collector.get_recent_metrics('system', minutes=1)
            processing_metrics = self.metrics_collector.get_recent_metrics('processing', minutes=1)
            workload_metrics = self.metrics_collector.get_recent_metrics('workload', minutes=1)
            
            # Use most recent values
            if system_metrics:
                metrics['system'] = system_metrics[-1]
            if processing_metrics:
                metrics['processing'] = processing_metrics[-1]
            if workload_metrics:
                metrics['workload'] = workload_metrics[-1]
                
        except Exception as e:
            self.logger.error(f"Error collecting metrics: {e}")
            # Return empty metrics rather than fail completely
            
        return metrics
        
    def _can_execute_scaling(self) -> bool:
        """Check if scaling action can be executed based on rate limits"""
        now = datetime.now()
        
        # Reset counter if hour has passed
        if (now - self.scaling_events_reset_time).total_seconds() >= 3600:
            self.recent_scaling_events = 0
            self.scaling_events_reset_time = now
            
        # Check rate limit
        return self.recent_scaling_events < self.config.max_scaling_events_per_hour
        
    def _record_cycle_metrics(self, duration: float):
        """Record feedback cycle performance metrics"""
        cycle_metric = {
            'timestamp': datetime.now(),
            'duration': duration,
            'success': True
        }
        
        self.cycle_metrics.append(cycle_metric)
        
        # Keep only recent metrics
        if len(self.cycle_metrics) > 100:
            self.cycle_metrics = self.cycle_metrics[-100:]
            
    async def _record_scaling_event(self, decision: ScalingDecision, success: bool):
        """Record scaling event for tracking"""
        scaling_event = {
            'timestamp': datetime.now(),
            'decision': decision.to_dict(),
            'success': success
        }
        
        self.scaling_events_history.append(scaling_event)
        self.recent_scaling_events += 1
        
        # Keep only recent history
        if len(self.scaling_events_history) > 50:
            self.scaling_events_history = self.scaling_events_history[-50:]
            
    async def _on_scaling_event(self, decision: ScalingDecision):
        """Handle scaling event callback"""
        # Notify registered callbacks
        for callback in self.scaling_callbacks:
            try:
                await callback(decision)
            except Exception as e:
                self.logger.error(f"Scaling callback error: {e}")
                
    # Public interface methods
    
    def add_processing_callback(self, callback: Callable):
        """Add callback for processing events"""
        self.processing_callbacks.append(callback)
        
    def add_scaling_callback(self, callback: Callable):
        """Add callback for scaling events"""
        self.scaling_callbacks.append(callback)
        
    async def record_processing_event(self, document_id: str, stage: str, processing_time: Optional[float] = None, error: Optional[str] = None):
        """Record processing event in the feedback system"""
        # Record in metrics collector
        if stage == 'started':
            self.metrics_collector.record_processing_start(document_id)
        elif stage == 'completed':
            self.metrics_collector.record_processing_success(document_id)
        elif stage == 'failed':
            self.metrics_collector.record_processing_error(document_id, error or 'unknown')
            
        # Publish to stream if available
        if self.stream_processor:
            await self.stream_processor.publish_processing_event(
                document_id, stage, processing_time, error
            )
            
        # Notify callbacks
        for callback in self.processing_callbacks:
            try:
                await callback(document_id, stage, processing_time, error)
            except Exception as e:
                self.logger.error(f"Processing callback error: {e}")
                
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        status = {
            'feedback_loop': {
                'running': self.is_running,
                'last_cycle': self.last_feedback_cycle.isoformat() if self.last_feedback_cycle else None,
                'recent_cycles': len(self.cycle_metrics),
                'recent_scaling_events': self.recent_scaling_events,
                'average_cycle_time': (
                    sum(m['duration'] for m in self.cycle_metrics[-10:]) / 
                    len(self.cycle_metrics[-10:])
                    if self.cycle_metrics else 0
                )
            },
            'components': {}
        }
        
        # Component health checks
        try:
            status['components']['metrics_collector'] = await self.metrics_collector.health_check()
            status['components']['adaptive_controller'] = await self.adaptive_controller.health_check()
            
            if self.stream_processor:
                status['components']['stream_processor'] = await self.stream_processor.health_check()
                
            status['components']['decision_engine'] = self.decision_engine.get_performance_summary()
            
        except Exception as e:
            status['error'] = str(e)
            
        return status
        
    async def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of key system metrics"""
        try:
            # Get current metrics
            current_metrics = await self._collect_current_metrics()
            
            # Get decision engine analysis
            analysis = self.decision_engine.analyze_performance_patterns()
            
            # Get backpressure status
            backpressure_status = None
            if self.stream_processor:
                backpressure_status = self.stream_processor.get_backpressure_status()
                
            return {
                'current_metrics': current_metrics,
                'performance_analysis': analysis,
                'backpressure_status': backpressure_status,
                'controller_state': self.adaptive_controller.get_current_state()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting metrics summary: {e}")
            return {'error': str(e)}
            
    async def trigger_manual_scaling(self, workers: Optional[int] = None, window_size: Optional[int] = None) -> bool:
        """Manually trigger scaling action"""
        try:
            success = True
            
            if workers is not None:
                success &= await self.adaptive_controller.worker_pool.scale_workers(
                    workers, "manual_trigger"
                )
                
            if window_size is not None:
                success &= self.adaptive_controller.window_manager.adjust_window_size(
                    window_size, "manual_trigger"
                )
                
            return success
            
        except Exception as e:
            self.logger.error(f"Manual scaling failed: {e}")
            return False
            
    async def shutdown(self):
        """Graceful shutdown of entire feedback system"""
        try:
            self.logger.info("Shutting down adaptive feedback system...")
            
            # Stop feedback loop
            await self.stop()
            
            # Shutdown components
            await self.metrics_collector.stop_collection()
            await self.adaptive_controller.shutdown()
            
            if self.stream_processor:
                await self.stream_processor.shutdown()
                
            self.logger.info("Adaptive feedback system shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
            raise
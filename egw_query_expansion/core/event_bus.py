"""
Synchronous Event Bus System for EGW Query Expansion Pipeline

This module implements a synchronous event-driven architecture that decouples
orchestrators from validators while maintaining deterministic execution order.
Provides typed events for pipeline stage transitions and validator verdicts.
"""

import hashlib
import logging
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Type, Union
from uuid import uuid4

logger = logging.getLogger(__name__)


class EventPriority(Enum):
    """Priority levels for event processing"""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4


class EventStatus(Enum):
    """Event processing status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class EventMetadata:
    """Metadata associated with events"""
    timestamp: float = field(default_factory=time.time)
    source_id: str = ""
    correlation_id: str = field(default_factory=lambda: str(uuid4()))
    priority: EventPriority = EventPriority.NORMAL
    retry_count: int = 0
    max_retries: int = 3
    timeout_ms: int = 5000


class BaseEvent(ABC):
    """Base class for all events in the system"""
    
    def __init__(self, data: Any = None, metadata: Optional[EventMetadata] = None):
        self.data = data
        self.metadata = metadata or EventMetadata()
        self.event_id = str(uuid4())
        self.status = EventStatus.PENDING
        
    @property
    @abstractmethod
    def event_type(self) -> str:
        """Return the event type identifier"""
        pass
    
    def get_hash(self) -> str:
        """Generate deterministic hash for event deduplication"""
        content = f"{self.event_type}_{self.data}_{self.metadata.source_id}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]


class EventHandler(ABC):
    """Base class for event handlers"""
    
    @abstractmethod
    def can_handle(self, event: BaseEvent) -> bool:
        """Check if this handler can process the given event"""
        pass
    
    @abstractmethod
    def handle(self, event: BaseEvent) -> Optional[BaseEvent]:
        """Handle the event and optionally return a response event"""
        pass
    
    @property
    @abstractmethod
    def handler_id(self) -> str:
        """Return unique handler identifier"""
        pass


@dataclass
class EventProcessingResult:
    """Result of event processing"""
    success: bool
    response_events: List[BaseEvent] = field(default_factory=list)
    error_message: Optional[str] = None
    processing_time_ms: float = 0.0


class SynchronousEventBus:
    """
    Synchronous event bus that maintains execution order while decoupling components
    """
    
    def __init__(self, max_queue_size: int = 10000):
        self.handlers: Dict[str, List[EventHandler]] = defaultdict(list)
        self.event_queue: deque = deque(maxlen=max_queue_size)
        self.processing_history: List[BaseEvent] = []
        self.active_subscriptions: Dict[str, Set[str]] = defaultdict(set)
        self.event_filters: Dict[str, Callable[[BaseEvent], bool]] = {}
        self.processing_lock = False
        
    def subscribe(self, event_type: str, handler: EventHandler, 
                 event_filter: Optional[Callable[[BaseEvent], bool]] = None):
        """
        Subscribe handler to specific event type
        
        Args:
            event_type: Type of event to subscribe to
            handler: Handler to process the events
            event_filter: Optional filter function for event filtering
        """
        self.handlers[event_type].append(handler)
        self.active_subscriptions[event_type].add(handler.handler_id)
        
        if event_filter:
            filter_key = f"{event_type}_{handler.handler_id}"
            self.event_filters[filter_key] = event_filter
            
        logger.debug(f"Handler {handler.handler_id} subscribed to {event_type}")
    
    def unsubscribe(self, event_type: str, handler: EventHandler):
        """Unsubscribe handler from event type"""
        if handler in self.handlers[event_type]:
            self.handlers[event_type].remove(handler)
            self.active_subscriptions[event_type].discard(handler.handler_id)
            
            filter_key = f"{event_type}_{handler.handler_id}"
            self.event_filters.pop(filter_key, None)
            
        logger.debug(f"Handler {handler.handler_id} unsubscribed from {event_type}")
    
    def publish(self, event: BaseEvent) -> EventProcessingResult:
        """
        Publish event for synchronous processing
        
        Args:
            event: Event to publish
            
        Returns:
            EventProcessingResult with processing outcome
        """
        start_time = time.time()
        
        try:
            # Add to queue and process immediately (synchronous)
            self.event_queue.append(event)
            
            # Prevent recursive processing
            if self.processing_lock:
                return EventProcessingResult(
                    success=False,
                    error_message="Event bus is already processing events"
                )
            
            self.processing_lock = True
            result = self._process_event(event)
            
            # Add to history
            self.processing_history.append(event)
            
            # Maintain history size
            if len(self.processing_history) > 1000:
                self.processing_history = self.processing_history[-500:]
                
            processing_time = (time.time() - start_time) * 1000
            result.processing_time_ms = processing_time
            
            return result
            
        finally:
            self.processing_lock = False
    
    def _process_event(self, event: BaseEvent) -> EventProcessingResult:
        """Process event synchronously with all applicable handlers"""
        event.status = EventStatus.PROCESSING
        response_events = []
        errors = []
        
        handlers = self.handlers.get(event.event_type, [])
        
        if not handlers:
            logger.warning(f"No handlers registered for event type: {event.event_type}")
            event.status = EventStatus.COMPLETED
            return EventProcessingResult(success=True)
        
        # Process with all handlers in registration order
        for handler in handlers:
            try:
                # Apply event filter if exists
                filter_key = f"{event.event_type}_{handler.handler_id}"
                if filter_key in self.event_filters:
                    if not self.event_filters[filter_key](event):
                        continue
                
                # Check if handler can process this event
                if not handler.can_handle(event):
                    continue
                    
                # Process the event
                response = handler.handle(event)
                
                if response:
                    response_events.append(response)
                    
            except Exception as e:
                error_msg = f"Handler {handler.handler_id} failed: {str(e)}"
                errors.append(error_msg)
                logger.error(error_msg, exc_info=True)
        
        # Determine overall success
        success = len(errors) == 0
        event.status = EventStatus.COMPLETED if success else EventStatus.FAILED
        
        return EventProcessingResult(
            success=success,
            response_events=response_events,
            error_message="; ".join(errors) if errors else None
        )
    
    def publish_batch(self, events: List[BaseEvent]) -> List[EventProcessingResult]:
        """
        Process multiple events in order
        
        Args:
            events: List of events to process
            
        Returns:
            List of processing results
        """
        results = []
        for event in events:
            result = self.publish(event)
            results.append(result)
            
            # Stop on first failure if required
            if not result.success and event.metadata.priority == EventPriority.CRITICAL:
                break
                
        return results
    
    def get_subscribers(self, event_type: str) -> List[str]:
        """Get list of handler IDs subscribed to event type"""
        return list(self.active_subscriptions.get(event_type, set()))
    
    def get_event_history(self, event_type: Optional[str] = None, 
                         limit: int = 100) -> List[BaseEvent]:
        """
        Get event processing history
        
        Args:
            event_type: Optional filter by event type
            limit: Maximum number of events to return
            
        Returns:
            List of historical events
        """
        if event_type:
            filtered = [e for e in self.processing_history if e.event_type == event_type]
        else:
            filtered = self.processing_history
            
        return filtered[-limit:]
    
    def clear_history(self):
        """Clear event processing history"""
        self.processing_history.clear()
        logger.info("Event bus history cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get event bus statistics"""
        event_type_counts = defaultdict(int)
        for event in self.processing_history:
            event_type_counts[event.event_type] += 1
            
        return {
            "total_events_processed": len(self.processing_history),
            "event_types": dict(event_type_counts),
            "active_subscriptions": {
                event_type: len(handlers) 
                for event_type, handlers in self.active_subscriptions.items()
            },
            "queue_size": len(self.event_queue),
            "handlers_registered": sum(len(handlers) for handlers in self.handlers.values())
        }


# Global event bus instance
_global_event_bus: Optional[SynchronousEventBus] = None


def get_event_bus() -> SynchronousEventBus:
    """Get global event bus instance"""
    global _global_event_bus
    if _global_event_bus is None:
        _global_event_bus = SynchronousEventBus()
    return _global_event_bus


def set_event_bus(bus: SynchronousEventBus):
    """Set global event bus instance"""
    global _global_event_bus
    _global_event_bus = bus


class EventBusManager:
    """Manager for multiple event bus instances"""
    
    def __init__(self):
        self.buses: Dict[str, SynchronousEventBus] = {}
        
    def create_bus(self, name: str, max_queue_size: int = 10000) -> SynchronousEventBus:
        """Create named event bus"""
        bus = SynchronousEventBus(max_queue_size)
        self.buses[name] = bus
        return bus
        
    def get_bus(self, name: str) -> Optional[SynchronousEventBus]:
        """Get named event bus"""
        return self.buses.get(name)
        
    def remove_bus(self, name: str) -> bool:
        """Remove named event bus"""
        if name in self.buses:
            del self.buses[name]
            return True
        return False
        
    def list_buses(self) -> List[str]:
        """List all bus names"""
        return list(self.buses.keys())
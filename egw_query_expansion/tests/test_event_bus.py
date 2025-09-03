"""
Tests for the synchronous event bus system
"""

import pytest
import time
from typing import Any, Optional
from unittest.mock import Mock

from egw_query_expansion.core.event_bus import (
    BaseEvent, EventHandler, SynchronousEventBus, EventPriority, 
    EventStatus, EventMetadata, EventProcessingResult
)


class TestEvent(BaseEvent):
    """Test event implementation"""
    
    def __init__(self, data: Any = None, event_type_name: str = "test"):
        super().__init__(data)
        self._event_type = event_type_name
    
    @property
    def event_type(self) -> str:
        return self._event_type


class TestHandler(EventHandler):
    """Test event handler implementation"""
    
    def __init__(self, handler_id: str, can_handle_types: list = None):
        self.handler_id_value = handler_id
        self.can_handle_types = can_handle_types or ["test"]
        self.handled_events = []
        self.should_fail = False
        self.response_event = None
    
    def can_handle(self, event: BaseEvent) -> bool:
        return event.event_type in self.can_handle_types
    
    def handle(self, event: BaseEvent) -> Optional[BaseEvent]:
        if self.should_fail:
            raise Exception("Handler configured to fail")
        
        self.handled_events.append(event)
        return self.response_event
    
    @property
    def handler_id(self) -> str:
        return self.handler_id_value


class TestSynchronousEventBus:
    """Test cases for SynchronousEventBus"""
    
    def test_bus_creation(self):
        """Test event bus creation"""
        bus = SynchronousEventBus()
        assert bus is not None
        assert len(bus.handlers) == 0
        assert len(bus.event_queue) == 0
    
    def test_handler_subscription(self):
        """Test handler subscription to event types"""
        bus = SynchronousEventBus()
        handler = TestHandler("test_handler")
        
        bus.subscribe("test", handler)
        
        assert "test" in bus.handlers
        assert handler in bus.handlers["test"]
        assert "test_handler" in bus.active_subscriptions["test"]
    
    def test_handler_unsubscription(self):
        """Test handler unsubscription from event types"""
        bus = SynchronousEventBus()
        handler = TestHandler("test_handler")
        
        bus.subscribe("test", handler)
        bus.unsubscribe("test", handler)
        
        assert handler not in bus.handlers["test"]
        assert "test_handler" not in bus.active_subscriptions["test"]
    
    def test_event_publishing_and_processing(self):
        """Test event publishing and synchronous processing"""
        bus = SynchronousEventBus()
        handler = TestHandler("test_handler")
        
        bus.subscribe("test", handler)
        
        event = TestEvent("test_data")
        result = bus.publish(event)
        
        assert result.success
        assert len(handler.handled_events) == 1
        assert handler.handled_events[0] == event
        assert event.status == EventStatus.COMPLETED
    
    def test_multiple_handlers_for_same_event(self):
        """Test multiple handlers processing the same event"""
        bus = SynchronousEventBus()
        handler1 = TestHandler("handler1")
        handler2 = TestHandler("handler2")
        
        bus.subscribe("test", handler1)
        bus.subscribe("test", handler2)
        
        event = TestEvent("test_data")
        result = bus.publish(event)
        
        assert result.success
        assert len(handler1.handled_events) == 1
        assert len(handler2.handled_events) == 1
    
    def test_handler_failure_handling(self):
        """Test handling of handler failures"""
        bus = SynchronousEventBus()
        handler = TestHandler("failing_handler")
        handler.should_fail = True
        
        bus.subscribe("test", handler)
        
        event = TestEvent("test_data")
        result = bus.publish(event)
        
        assert not result.success
        assert result.error_message is not None
        assert "Handler failing_handler failed" in result.error_message
    
    def test_event_filtering(self):
        """Test event filtering functionality"""
        bus = SynchronousEventBus()
        handler = TestHandler("test_handler")
        
        # Create filter that only allows events with specific data
        def event_filter(event: BaseEvent) -> bool:
            return event.data == "allowed_data"
        
        bus.subscribe("test", handler, event_filter)
        
        # Event that should be filtered out
        filtered_event = TestEvent("blocked_data")
        result1 = bus.publish(filtered_event)
        
        # Event that should pass through
        allowed_event = TestEvent("allowed_data")
        result2 = bus.publish(allowed_event)
        
        assert result1.success  # Processing succeeds but handler doesn't run
        assert result2.success
        assert len(handler.handled_events) == 1  # Only allowed event handled
        assert handler.handled_events[0].data == "allowed_data"
    
    def test_no_handlers_for_event_type(self):
        """Test behavior when no handlers are registered for event type"""
        bus = SynchronousEventBus()
        
        event = TestEvent("test_data", "unhandled_type")
        result = bus.publish(event)
        
        assert result.success  # Should succeed even with no handlers
        assert event.status == EventStatus.COMPLETED
    
    def test_response_event_handling(self):
        """Test handling of response events from handlers"""
        bus = SynchronousEventBus()
        handler = TestHandler("test_handler")
        
        response_event = TestEvent("response_data", "response")
        handler.response_event = response_event
        
        bus.subscribe("test", handler)
        
        event = TestEvent("test_data")
        result = bus.publish(event)
        
        assert result.success
        assert len(result.response_events) == 1
        assert result.response_events[0] == response_event
    
    def test_batch_event_processing(self):
        """Test batch processing of multiple events"""
        bus = SynchronousEventBus()
        handler = TestHandler("test_handler")
        
        bus.subscribe("test", handler)
        
        events = [
            TestEvent("data1"),
            TestEvent("data2"),
            TestEvent("data3")
        ]
        
        results = bus.publish_batch(events)
        
        assert len(results) == 3
        assert all(result.success for result in results)
        assert len(handler.handled_events) == 3
    
    def test_event_history_tracking(self):
        """Test event history tracking"""
        bus = SynchronousEventBus()
        handler = TestHandler("test_handler")
        
        bus.subscribe("test", handler)
        
        events = [TestEvent(f"data{i}") for i in range(5)]
        
        for event in events:
            bus.publish(event)
        
        history = bus.get_event_history()
        assert len(history) == 5
        
        # Test filtering by event type
        test_history = bus.get_event_history("test")
        assert len(test_history) == 5
        
        # Test limit
        limited_history = bus.get_event_history(limit=3)
        assert len(limited_history) == 3
    
    def test_event_bus_statistics(self):
        """Test event bus statistics"""
        bus = SynchronousEventBus()
        handler = TestHandler("test_handler")
        
        bus.subscribe("test", handler)
        bus.subscribe("other", TestHandler("other_handler"))
        
        # Process some events
        for i in range(3):
            bus.publish(TestEvent(f"data{i}"))
        
        stats = bus.get_stats()
        
        assert stats["total_events_processed"] == 3
        assert stats["event_types"]["test"] == 3
        assert stats["handlers_registered"] == 2
        assert "test" in stats["active_subscriptions"]
    
    def test_event_metadata_handling(self):
        """Test event metadata processing"""
        bus = SynchronousEventBus()
        handler = TestHandler("test_handler")
        
        bus.subscribe("test", handler)
        
        metadata = EventMetadata(
            source_id="test_source",
            priority=EventPriority.HIGH
        )
        
        event = TestEvent("test_data")
        event.metadata = metadata
        
        result = bus.publish(event)
        
        assert result.success
        assert event.metadata.source_id == "test_source"
        assert event.metadata.priority == EventPriority.HIGH
    
    def test_processing_time_tracking(self):
        """Test processing time tracking"""
        bus = SynchronousEventBus()
        handler = TestHandler("slow_handler")
        
        # Add delay to handler
        original_handle = handler.handle
        def slow_handle(event):
            time.sleep(0.01)  # 10ms delay
            return original_handle(event)
        handler.handle = slow_handle
        
        bus.subscribe("test", handler)
        
        event = TestEvent("test_data")
        result = bus.publish(event)
        
        assert result.success
        assert result.processing_time_ms > 0
    
    def test_event_deduplication(self):
        """Test event deduplication by hash"""
        bus = SynchronousEventBus()
        handler = TestHandler("test_handler")
        
        bus.subscribe("test", handler)
        
        event1 = TestEvent("same_data")
        event2 = TestEvent("same_data")
        
        # Should have same hash
        assert event1.get_hash() == event2.get_hash()
        
        bus.publish(event1)
        bus.publish(event2)
        
        # Both should be processed (deduplication would be handled at application level)
        assert len(handler.handled_events) == 2


class TestBaseEvent:
    """Test cases for BaseEvent"""
    
    def test_event_creation(self):
        """Test basic event creation"""
        event = TestEvent("test_data")
        
        assert event.data == "test_data"
        assert event.status == EventStatus.PENDING
        assert event.event_id is not None
        assert event.metadata is not None
    
    def test_event_hash_generation(self):
        """Test event hash generation for deduplication"""
        event1 = TestEvent("data")
        event2 = TestEvent("data")
        event3 = TestEvent("different_data")
        
        # Same data should produce same hash
        assert event1.get_hash() == event2.get_hash()
        
        # Different data should produce different hash
        assert event1.get_hash() != event3.get_hash()


class TestEventMetadata:
    """Test cases for EventMetadata"""
    
    def test_metadata_creation(self):
        """Test metadata creation with defaults"""
        metadata = EventMetadata()
        
        assert metadata.timestamp > 0
        assert metadata.priority == EventPriority.NORMAL
        assert metadata.retry_count == 0
        assert metadata.max_retries == 3
    
    def test_metadata_with_custom_values(self):
        """Test metadata creation with custom values"""
        metadata = EventMetadata(
            source_id="custom_source",
            priority=EventPriority.CRITICAL,
            max_retries=5
        )
        
        assert metadata.source_id == "custom_source"
        assert metadata.priority == EventPriority.CRITICAL
        assert metadata.max_retries == 5


if __name__ == "__main__":
    pytest.main([__file__])
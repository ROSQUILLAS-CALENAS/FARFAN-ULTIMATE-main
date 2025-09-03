"""
Synchronous Event Bus System for Pipeline Architecture
=====================================================

Implements a synchronous, typed event bus that supports event publishing,
subscription, and delivery with built-in error handling and event ordering.
"""

import logging
import threading
from collections import defaultdict, deque
from typing import Dict, List, Callable, Any, Optional, Set, Type
from datetime import datetime
from uuid import uuid4
import traceback

from event_schemas import BaseEvent, EventType, EventRegistry


logger = logging.getLogger(__name__)


class EventSubscription:
    """Represents a subscription to specific event types"""
    
    def __init__(self, subscriber_name: str, handler: Callable[[BaseEvent], None],
                 event_types: Set[EventType], priority: int = 0):
        self.subscription_id = str(uuid4())
        self.subscriber_name = subscriber_name
        self.handler = handler
        self.event_types = event_types
        self.priority = priority  # Higher priority = executed first
        self.created_at = datetime.utcnow()
        self.active = True
        self.error_count = 0
        self.last_error_at: Optional[datetime] = None
        
    def matches_event(self, event_type: EventType) -> bool:
        """Check if this subscription matches the given event type"""
        return self.active and event_type in self.event_types
    
    def handle_event(self, event: BaseEvent) -> bool:
        """Handle an event with error tracking"""
        try:
            self.handler(event)
            return True
        except Exception as e:
            self.error_count += 1
            self.last_error_at = datetime.utcnow()
            logger.error(f"Error in event handler {self.subscriber_name}: {e}")
            logger.debug(f"Handler error traceback: {traceback.format_exc()}")
            return False


class EventDeliveryStats:
    """Statistics for event delivery and processing"""
    
    def __init__(self):
        self.events_published = 0
        self.events_delivered = 0
        self.events_failed = 0
        self.handler_errors = 0
        self.subscription_count = 0
        self.event_type_counts: Dict[EventType, int] = defaultdict(int)
        self.last_reset = datetime.utcnow()
    
    def reset(self):
        """Reset all statistics"""
        self.__init__()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary"""
        return {
            'events_published': self.events_published,
            'events_delivered': self.events_delivered,
            'events_failed': self.events_failed,
            'handler_errors': self.handler_errors,
            'subscription_count': self.subscription_count,
            'event_type_counts': dict(self.event_type_counts),
            'last_reset': self.last_reset.isoformat(),
            'success_rate': (self.events_delivered / max(1, self.events_published)) * 100,
            'error_rate': (self.handler_errors / max(1, self.events_delivered)) * 100
        }


class EventBus:
    """
    Synchronous event bus for pipeline architecture.
    Provides typed event publishing and subscription with ordered delivery.
    """
    
    def __init__(self, max_event_history: int = 1000, 
                 max_handler_errors: int = 10,
                 enable_event_logging: bool = True):
        self._subscriptions: Dict[str, EventSubscription] = {}
        self._event_type_subscriptions: Dict[EventType, List[str]] = defaultdict(list)
        self._event_history: deque = deque(maxlen=max_event_history)
        self._stats = EventDeliveryStats()
        self._lock = threading.RLock()  # Allow reentrant locking for nested calls
        self._max_handler_errors = max_handler_errors
        self._enable_event_logging = enable_event_logging
        
        logger.info("EventBus initialized")
    
    def subscribe(self, 
                  subscriber_name: str,
                  event_types: List[EventType] | EventType,
                  handler: Callable[[BaseEvent], None],
                  priority: int = 0) -> str:
        """
        Subscribe to specific event types.
        
        Args:
            subscriber_name: Name of the subscriber
            event_types: Event type(s) to subscribe to
            handler: Function to handle events
            priority: Priority for handler execution (higher = first)
            
        Returns:
            Subscription ID
        """
        if isinstance(event_types, EventType):
            event_types = [event_types]
        
        event_type_set = set(event_types)
        
        with self._lock:
            subscription = EventSubscription(
                subscriber_name=subscriber_name,
                handler=handler,
                event_types=event_type_set,
                priority=priority
            )
            
            self._subscriptions[subscription.subscription_id] = subscription
            
            # Update event type mappings
            for event_type in event_type_set:
                self._event_type_subscriptions[event_type].append(subscription.subscription_id)
                # Sort by priority (descending)
                self._event_type_subscriptions[event_type].sort(
                    key=lambda sub_id: self._subscriptions[sub_id].priority,
                    reverse=True
                )
            
            self._stats.subscription_count = len(self._subscriptions)
            
            logger.info(f"Subscribed {subscriber_name} to {len(event_type_set)} event types")
            return subscription.subscription_id
    
    def unsubscribe(self, subscription_id: str) -> bool:
        """
        Unsubscribe using subscription ID.
        
        Args:
            subscription_id: ID returned from subscribe()
            
        Returns:
            True if subscription was removed
        """
        with self._lock:
            if subscription_id not in self._subscriptions:
                return False
            
            subscription = self._subscriptions[subscription_id]
            
            # Remove from event type mappings
            for event_type in subscription.event_types:
                if subscription_id in self._event_type_subscriptions[event_type]:
                    self._event_type_subscriptions[event_type].remove(subscription_id)
            
            # Remove subscription
            del self._subscriptions[subscription_id]
            self._stats.subscription_count = len(self._subscriptions)
            
            logger.info(f"Unsubscribed {subscription.subscriber_name}")
            return True
    
    def publish(self, event: BaseEvent) -> Dict[str, Any]:
        """
        Publish an event synchronously to all subscribers.
        
        Args:
            event: Event to publish
            
        Returns:
            Delivery result with stats
        """
        if not event.validate_data():
            error_msg = f"Event validation failed for {event.__class__.__name__}"
            logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'handlers_called': 0,
                'handlers_succeeded': 0,
                'handlers_failed': 0
            }
        
        delivery_result = {
            'success': True,
            'event_id': event.event_id,
            'event_type': event.event_type.value,
            'handlers_called': 0,
            'handlers_succeeded': 0,
            'handlers_failed': 0,
            'delivery_time': datetime.utcnow(),
            'errors': []
        }
        
        with self._lock:
            self._stats.events_published += 1
            self._stats.event_type_counts[event.event_type] += 1
            
            # Store event in history if logging enabled
            if self._enable_event_logging:
                self._event_history.append({
                    'event_id': event.event_id,
                    'event_type': event.event_type.value,
                    'source': event.source,
                    'timestamp': event.timestamp,
                    'correlation_id': event.correlation_id
                })
            
            # Get subscribers for this event type
            subscription_ids = self._event_type_subscriptions.get(event.event_type, [])
            
            if not subscription_ids:
                logger.debug(f"No subscribers for event type {event.event_type.value}")
                return delivery_result
            
            # Deliver to all subscribers in priority order
            for sub_id in subscription_ids:
                subscription = self._subscriptions.get(sub_id)
                if not subscription or not subscription.active:
                    continue
                
                delivery_result['handlers_called'] += 1
                
                # Check if this subscription has too many errors
                if subscription.error_count >= self._max_handler_errors:
                    logger.warning(
                        f"Subscription {subscription.subscriber_name} disabled due to "
                        f"{subscription.error_count} errors"
                    )
                    subscription.active = False
                    continue
                
                # Deliver event
                success = subscription.handle_event(event)
                
                if success:
                    delivery_result['handlers_succeeded'] += 1
                    self._stats.events_delivered += 1
                else:
                    delivery_result['handlers_failed'] += 1
                    self._stats.handler_errors += 1
                    delivery_result['errors'].append({
                        'subscriber': subscription.subscriber_name,
                        'error_count': subscription.error_count,
                        'last_error_at': subscription.last_error_at.isoformat() if subscription.last_error_at else None
                    })
            
            # Update overall success flag
            delivery_result['success'] = delivery_result['handlers_failed'] == 0
            
            if delivery_result['handlers_failed'] > 0:
                self._stats.events_failed += 1
        
        logger.debug(f"Published {event.event_type.value} to {delivery_result['handlers_called']} handlers")
        return delivery_result
    
    def publish_typed_event(self, event_type: EventType, 
                          source: str,
                          data: Any = None,
                          correlation_id: str = None,
                          **metadata) -> Dict[str, Any]:
        """
        Convenience method to publish a typed event.
        
        Args:
            event_type: Type of event to create and publish
            source: Source component name
            data: Event data payload
            correlation_id: Optional correlation ID
            **metadata: Additional metadata
            
        Returns:
            Delivery result
        """
        try:
            event = EventRegistry.create_event(
                event_type=event_type,
                source=source,
                data=data,
                correlation_id=correlation_id,
                metadata=metadata
            )
            return self.publish(event)
        except Exception as e:
            logger.error(f"Failed to create and publish event {event_type.value}: {e}")
            return {
                'success': False,
                'error': str(e),
                'handlers_called': 0,
                'handlers_succeeded': 0,
                'handlers_failed': 0
            }
    
    def get_subscription_info(self, subscription_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific subscription"""
        with self._lock:
            subscription = self._subscriptions.get(subscription_id)
            if not subscription:
                return None
            
            return {
                'subscription_id': subscription.subscription_id,
                'subscriber_name': subscription.subscriber_name,
                'event_types': [et.value for et in subscription.event_types],
                'priority': subscription.priority,
                'active': subscription.active,
                'error_count': subscription.error_count,
                'last_error_at': subscription.last_error_at.isoformat() if subscription.last_error_at else None,
                'created_at': subscription.created_at.isoformat()
            }
    
    def list_subscriptions(self) -> List[Dict[str, Any]]:
        """List all current subscriptions"""
        with self._lock:
            return [
                self.get_subscription_info(sub_id) 
                for sub_id in self._subscriptions.keys()
            ]
    
    def get_event_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent event history"""
        with self._lock:
            history_list = list(self._event_history)
            return history_list[-limit:] if limit < len(history_list) else history_list
    
    def get_stats(self) -> Dict[str, Any]:
        """Get event bus statistics"""
        with self._lock:
            stats = self._stats.to_dict()
            stats['active_subscriptions'] = sum(
                1 for sub in self._subscriptions.values() if sub.active
            )
            stats['inactive_subscriptions'] = sum(
                1 for sub in self._subscriptions.values() if not sub.active
            )
            return stats
    
    def reset_stats(self):
        """Reset event delivery statistics"""
        with self._lock:
            self._stats.reset()
            logger.info("Event bus statistics reset")
    
    def clear_history(self):
        """Clear event history"""
        with self._lock:
            self._event_history.clear()
            logger.info("Event history cleared")
    
    def reactivate_subscription(self, subscription_id: str) -> bool:
        """Reactivate a disabled subscription and reset error count"""
        with self._lock:
            subscription = self._subscriptions.get(subscription_id)
            if not subscription:
                return False
            
            subscription.active = True
            subscription.error_count = 0
            subscription.last_error_at = None
            
            logger.info(f"Reactivated subscription {subscription.subscriber_name}")
            return True
    
    def shutdown(self):
        """Shutdown the event bus and cleanup resources"""
        with self._lock:
            logger.info("Shutting down event bus")
            self._subscriptions.clear()
            self._event_type_subscriptions.clear()
            self._event_history.clear()
            self._stats.reset()


# ============================================================================
# CONVENIENCE DECORATORS AND FUNCTIONS
# ============================================================================

def event_handler(*event_types: EventType, priority: int = 0):
    """
    Decorator to mark functions as event handlers.
    
    Usage:
        @event_handler(EventType.STAGE_STARTED, EventType.STAGE_COMPLETED)
        def my_handler(event: BaseEvent):
            print(f"Received event: {event.event_type}")
    """
    def decorator(func):
        func._event_types = list(event_types)
        func._event_priority = priority
        return func
    return decorator


def subscribe_decorated_handlers(event_bus: EventBus, 
                               handler_object: Any, 
                               subscriber_name: str = None) -> List[str]:
    """
    Subscribe all decorated event handler methods on an object.
    
    Args:
        event_bus: EventBus instance
        handler_object: Object containing decorated handler methods
        subscriber_name: Name for the subscriber (defaults to class name)
        
    Returns:
        List of subscription IDs
    """
    if subscriber_name is None:
        subscriber_name = handler_object.__class__.__name__
    
    subscription_ids = []
    
    for attr_name in dir(handler_object):
        attr = getattr(handler_object, attr_name)
        if (callable(attr) and 
            hasattr(attr, '_event_types') and 
            hasattr(attr, '_event_priority')):
            
            sub_id = event_bus.subscribe(
                subscriber_name=f"{subscriber_name}.{attr_name}",
                event_types=attr._event_types,
                handler=attr,
                priority=attr._event_priority
            )
            subscription_ids.append(sub_id)
    
    logger.info(f"Subscribed {len(subscription_ids)} handlers for {subscriber_name}")
    return subscription_ids
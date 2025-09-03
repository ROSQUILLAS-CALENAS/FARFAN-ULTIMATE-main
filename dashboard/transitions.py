"""
Transition state management for smooth UI animations and queued backend updates.
Manages update queuing during active transitions and smooth application afterward.
"""

import threading
import time
# # # from typing import Dict, Any, List, Tuple, Optional, Set  # Module not found  # Module not found  # Module not found
# # # from dataclasses import dataclass  # Module not found  # Module not found  # Module not found
# # # from datetime import datetime, timedelta  # Module not found  # Module not found  # Module not found
import logging
# # # from queue import Queue  # Module not found  # Module not found  # Module not found
# # # from enum import Enum  # Module not found  # Module not found  # Module not found


class TransitionState(Enum):
    """States for UI transitions."""
    IDLE = "idle"
    STARTING = "starting"
    ACTIVE = "active"
    COMPLETING = "completing"
    COMPLETED = "completed"


@dataclass
class QueuedUpdate:
    """Represents a queued update during transitions."""
    layer: str
    updates: Dict[str, Any]
    timestamp: datetime
    priority: int = 0
    source: str = "unknown"


@dataclass
class ActiveTransition:
    """Represents an active UI transition."""
    transition_id: str
    transition_type: str
    state: TransitionState
    start_time: datetime
    duration_ms: int
    progress: float = 0.0
    metadata: Dict[str, Any] = None


class TransitionManager:
    """Manages UI transitions and queues backend updates during animations."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Active transitions tracking
        self._active_transitions: Dict[str, ActiveTransition] = {}
        self._transition_counter = 0
        self._lock = threading.RLock()
        
        # Update queuing
        self._update_queue: Queue[QueuedUpdate] = Queue()
        self._queued_count = 0
        self._max_queue_size = 1000
        
        # Transition types that should queue updates
        self._blocking_transition_types = {
            'radial_menu_expand',
            'radial_menu_collapse', 
            'overlay_fade_in',
            'overlay_fade_out',
            'focus_mode_transition',
            'viewport_zoom',
            'viewport_pan',
            'sidebar_toggle'
        }
        
        # Performance monitoring
        self._performance_stats = {
            'total_transitions': 0,
            'queued_updates_total': 0,
            'avg_transition_duration': 0.0,
            'max_queue_size_reached': 0
        }
    
    def start(self, transition_type: str, duration_ms: int = 300, 
             metadata: Optional[Dict[str, Any]] = None) -> str:
        """Start a new UI transition."""
        with self._lock:
            self._transition_counter += 1
            transition_id = f"{transition_type}_{self._transition_counter}_{int(time.time() * 1000)}"
            
            transition = ActiveTransition(
                transition_id=transition_id,
                transition_type=transition_type,
                state=TransitionState.STARTING,
                start_time=datetime.utcnow(),
                duration_ms=duration_ms,
                metadata=metadata or {}
            )
            
            self._active_transitions[transition_id] = transition
            
            # Update performance stats
            self._performance_stats['total_transitions'] += 1
            
            self.logger.debug(f"Started transition: {transition_id} ({transition_type})")
            return transition_id
    
    def update_progress(self, transition_id: str, progress: float):
        """Update progress of an active transition."""
        with self._lock:
            if transition_id in self._active_transitions:
                transition = self._active_transitions[transition_id]
                transition.progress = max(0.0, min(1.0, progress))
                
                # Update state based on progress
                if progress > 0.0 and transition.state == TransitionState.STARTING:
                    transition.state = TransitionState.ACTIVE
                elif progress >= 1.0 and transition.state == TransitionState.ACTIVE:
                    transition.state = TransitionState.COMPLETING
    
    def complete(self, transition_id: str) -> List[Tuple[str, Dict[str, Any]]]:
        """Complete a transition and return queued updates."""
        with self._lock:
            if transition_id not in self._active_transitions:
                return []
            
            transition = self._active_transitions[transition_id]
            transition.state = TransitionState.COMPLETED
            
            # Calculate actual duration for stats
            actual_duration = (datetime.utcnow() - transition.start_time).total_seconds() * 1000
            self._update_performance_stats(actual_duration)
            
            # Get queued updates for this transition period
            queued_updates = self._extract_queued_updates()
            
            # Remove completed transition
            del self._active_transitions[transition_id]
            
            self.logger.debug(f"Completed transition: {transition_id}, "
                            f"applying {len(queued_updates)} queued updates")
            
            return queued_updates
    
    def queue_update(self, layer: str, updates: Dict[str, Any], 
                    priority: int = 0, source: str = "backend") -> bool:
        """Queue an update during active transitions."""
        with self._lock:
            # Check if any blocking transitions are active
            should_queue = self._should_queue_updates()
            
            if not should_queue:
                return False  # No need to queue
            
            # Check queue size limit
            if self._queued_count >= self._max_queue_size:
                self.logger.warning("Update queue is full, dropping oldest updates")
                self._drop_oldest_updates()
            
            # Add to queue
            queued_update = QueuedUpdate(
                layer=layer,
                updates=updates,
                timestamp=datetime.utcnow(),
                priority=priority,
                source=source
            )
            
            self._update_queue.put(queued_update)
            self._queued_count += 1
            
            # Update stats
            self._performance_stats['queued_updates_total'] += 1
            self._performance_stats['max_queue_size_reached'] = max(
                self._performance_stats['max_queue_size_reached'],
                self._queued_count
            )
            
            self.logger.debug(f"Queued {layer} update (queue size: {self._queued_count})")
            return True
    
    def _should_queue_updates(self) -> bool:
        """Determine if updates should be queued based on active transitions."""
        for transition in self._active_transitions.values():
            if (transition.transition_type in self._blocking_transition_types and 
                transition.state in [TransitionState.STARTING, TransitionState.ACTIVE]):
                return True
        return False
    
    def _extract_queued_updates(self) -> List[Tuple[str, Dict[str, Any]]]:
        """Extract and sort queued updates."""
        updates = []
        
        # Extract all queued updates
        while not self._update_queue.empty():
            try:
                queued_update = self._update_queue.get_nowait()
                updates.append((queued_update.layer, queued_update.updates))
                self._queued_count -= 1
            except:
                break
        
        # Sort by priority and timestamp (higher priority first, then older first)
        updates.sort(key=lambda x: (-x[1].get('priority', 0), x[1].get('timestamp', datetime.min)))
        
        return [(layer, update_data) for layer, update_data in updates]
    
    def _drop_oldest_updates(self, count: int = 10):
        """Drop oldest updates to make room in queue."""
        dropped = 0
        temp_updates = []
        
        # Extract all updates
        while not self._update_queue.empty() and dropped < count:
            try:
                self._update_queue.get_nowait()
                self._queued_count -= 1
                dropped += 1
            except:
                break
        
# # #         self.logger.warning(f"Dropped {dropped} oldest updates from queue")  # Module not found  # Module not found  # Module not found
    
    def _update_performance_stats(self, duration_ms: float):
        """Update performance statistics."""
        current_avg = self._performance_stats['avg_transition_duration']
        total_transitions = self._performance_stats['total_transitions']
        
        # Calculate running average
        self._performance_stats['avg_transition_duration'] = (
            (current_avg * (total_transitions - 1) + duration_ms) / total_transitions
        )
    
    def is_active(self) -> bool:
        """Check if any blocking transitions are active."""
        with self._lock:
            return self._should_queue_updates()
    
    def get_active_transitions(self) -> List[Dict[str, Any]]:
        """Get information about active transitions."""
        with self._lock:
            transitions = []
            for transition in self._active_transitions.values():
                transitions.append({
                    'id': transition.transition_id,
                    'type': transition.transition_type,
                    'state': transition.state.value,
                    'progress': transition.progress,
                    'duration_ms': transition.duration_ms,
                    'elapsed_ms': (datetime.utcnow() - transition.start_time).total_seconds() * 1000,
                    'metadata': transition.metadata
                })
            return transitions
    
    def get_queued_count(self) -> int:
        """Get number of queued updates."""
        return self._queued_count
    
    def cancel_transition(self, transition_id: str) -> bool:
        """Cancel an active transition."""
        with self._lock:
            if transition_id in self._active_transitions:
                transition = self._active_transitions[transition_id]
                transition.state = TransitionState.COMPLETED
                del self._active_transitions[transition_id]
                
                self.logger.debug(f"Cancelled transition: {transition_id}")
                return True
            return False
    
    def cancel_all_transitions(self) -> int:
        """Cancel all active transitions."""
        with self._lock:
            cancelled_count = len(self._active_transitions)
            self._active_transitions.clear()
            
            self.logger.info(f"Cancelled {cancelled_count} active transitions")
            return cancelled_count
    
    def flush_queue(self) -> List[Tuple[str, Dict[str, Any]]]:
        """Flush all queued updates immediately."""
        with self._lock:
            updates = self._extract_queued_updates()
            self.logger.info(f"Flushed {len(updates)} queued updates")
            return updates
    
    def configure_blocking_types(self, transition_types: Set[str]):
        """Configure which transition types should block updates."""
        with self._lock:
            self._blocking_transition_types = set(transition_types)
            self.logger.info(f"Updated blocking transition types: {transition_types}")
    
    def add_blocking_type(self, transition_type: str):
        """Add a transition type that should block updates."""
        with self._lock:
            self._blocking_transition_types.add(transition_type)
    
    def remove_blocking_type(self, transition_type: str):
# # #         """Remove a transition type from blocking updates."""  # Module not found  # Module not found  # Module not found
        with self._lock:
            self._blocking_transition_types.discard(transition_type)
    
    def set_max_queue_size(self, max_size: int):
        """Set maximum queue size."""
        with self._lock:
            self._max_queue_size = max(10, max_size)  # Minimum of 10
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        with self._lock:
            return self._performance_stats.copy()
    
    def reset_performance_stats(self):
        """Reset performance statistics."""
        with self._lock:
            self._performance_stats = {
                'total_transitions': 0,
                'queued_updates_total': 0,
                'avg_transition_duration': 0.0,
                'max_queue_size_reached': 0
            }
    
    def cleanup_completed_transitions(self, max_age_seconds: float = 300):
        """Clean up old completed transitions."""
        with self._lock:
            current_time = datetime.utcnow()
            to_remove = []
            
            for transition_id, transition in self._active_transitions.items():
                age = (current_time - transition.start_time).total_seconds()
                if transition.state == TransitionState.COMPLETED and age > max_age_seconds:
                    to_remove.append(transition_id)
            
            for transition_id in to_remove:
                del self._active_transitions[transition_id]
            
            if to_remove:
                self.logger.debug(f"Cleaned up {len(to_remove)} completed transitions")
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get transition manager diagnostics."""
        with self._lock:
            return {
                'active_transitions_count': len(self._active_transitions),
                'queued_updates_count': self._queued_count,
                'max_queue_size': self._max_queue_size,
                'blocking_transition_types': list(self._blocking_transition_types),
                'is_blocking_active': self.is_active(),
                'performance_stats': self._performance_stats.copy(),
                'active_transitions': self.get_active_transitions()
            }
    
    # Context manager support for transitions
    
    def __enter__(self):
        """Enter context manager."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager - cleanup any remaining transitions."""
        self.cancel_all_transitions()
        self.flush_queue()


class TransitionContext:
    """Context manager for handling transitions with automatic cleanup."""
    
    def __init__(self, transition_manager: TransitionManager, 
                 transition_type: str, duration_ms: int = 300, 
                 metadata: Optional[Dict[str, Any]] = None):
        self.transition_manager = transition_manager
        self.transition_type = transition_type
        self.duration_ms = duration_ms
        self.metadata = metadata
        self.transition_id = None
    
    def __enter__(self):
        """Start transition."""
        self.transition_id = self.transition_manager.start(
            self.transition_type, 
            self.duration_ms, 
            self.metadata
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Complete transition and apply queued updates."""
        if self.transition_id:
            return self.transition_manager.complete(self.transition_id)
        return []
    
    def update_progress(self, progress: float):
        """Update transition progress."""
        if self.transition_id:
            self.transition_manager.update_progress(self.transition_id, progress)
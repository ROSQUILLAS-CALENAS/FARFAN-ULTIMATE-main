"""
Centralized store implementation for dashboard state management.
"""

import asyncio
import threading
# # # from typing import Dict, Any, Optional, Callable, List  # Module not found  # Module not found  # Module not found
# # # from dataclasses import dataclass  # Module not found  # Module not found  # Module not found
# # # from datetime import datetime  # Module not found  # Module not found  # Module not found
import logging

# # # from .ui_state import UIStateManager  # Module not found  # Module not found  # Module not found
# # # from .backend_state import BackendStateManager  # Module not found  # Module not found  # Module not found
# # # from .persistence import StatePersistence  # Module not found  # Module not found  # Module not found
# # # from .transitions import TransitionManager  # Module not found  # Module not found  # Module not found


@dataclass
class StateLayer:
    """Represents a state layer with metadata."""
    data: Dict[str, Any]
    timestamp: datetime
    version: int
    source: str


class DashboardStore:
    """Centralized store for dashboard state management with separate UI and backend layers."""
    
    def __init__(self, persistence_key: str = "dashboard_state"):
        self.logger = logging.getLogger(__name__)
        
        # State managers
        self.ui_state = UIStateManager()
        self.backend_state = BackendStateManager()
        self.persistence = StatePersistence(persistence_key)
        self.transitions = TransitionManager()
        
        # Internal state
        self._subscribers: Dict[str, List[Callable]] = {
            'ui': [],
            'backend': [],
            'global': []
        }
        self._lock = threading.RLock()
        self._debounce_timers: Dict[str, threading.Timer] = {}
        
# # #         # Initialize from persisted state  # Module not found  # Module not found  # Module not found
        self._initialize_from_persistence()
        
    def _initialize_from_persistence(self):
# # #         """Initialize store from persisted state."""  # Module not found  # Module not found  # Module not found
        try:
            persisted_state = self.persistence.load_state()
            if persisted_state:
                # Restore UI state
                if 'ui' in persisted_state:
                    self.ui_state.restore_state(persisted_state['ui'])
                
                # Backend state is not persisted (ephemeral)
# # #                 self.logger.info("Store initialized from persisted state")  # Module not found  # Module not found  # Module not found
        except Exception as e:
            self.logger.warning(f"Failed to load persisted state: {e}")
    
    def subscribe(self, layer: str, callback: Callable[[Dict[str, Any]], None]) -> str:
        """Subscribe to state changes for a specific layer."""
        with self._lock:
            subscription_id = f"{layer}_{len(self._subscribers[layer])}"
            if layer not in self._subscribers:
                self._subscribers[layer] = []
            self._subscribers[layer].append(callback)
            return subscription_id
    
    def unsubscribe(self, layer: str, subscription_id: str):
# # #         """Unsubscribe from state changes."""  # Module not found  # Module not found  # Module not found
        with self._lock:
            if layer in self._subscribers:
                # Remove by subscription_id logic would go here
                pass
    
    def get_state(self, layer: Optional[str] = None) -> Dict[str, Any]:
        """Get current state for specified layer or all layers."""
        with self._lock:
            if layer == 'ui':
                return self.ui_state.get_state()
            elif layer == 'backend':
                return self.backend_state.get_state()
            else:
                return {
                    'ui': self.ui_state.get_state(),
                    'backend': self.backend_state.get_state(),
                    'meta': {
                        'last_sync': datetime.utcnow().isoformat(),
                        'ui_version': self.ui_state.get_version(),
                        'backend_version': self.backend_state.get_version()
                    }
                }
    
    def update_ui_state(self, updates: Dict[str, Any], debounce_ms: int = 100) -> bool:
        """Update UI state with optional debouncing."""
        return self._debounced_update('ui', updates, debounce_ms)
    
    def update_backend_state(self, updates: Dict[str, Any], optimistic: bool = True) -> bool:
        """Update backend state with optional optimistic updates."""
        if optimistic:
            return self._optimistic_backend_update(updates)
        else:
            return self._direct_backend_update(updates)
    
    def _debounced_update(self, layer: str, updates: Dict[str, Any], debounce_ms: int) -> bool:
        """Apply debounced update to prevent rapid state changes during interactions."""
        with self._lock:
            # Cancel existing debounce timer
            timer_key = f"{layer}_debounce"
            if timer_key in self._debounce_timers:
                self._debounce_timers[timer_key].cancel()
            
            # Set new debounce timer
            timer = threading.Timer(
                debounce_ms / 1000.0,
                self._apply_update,
                args=(layer, updates)
            )
            self._debounce_timers[timer_key] = timer
            timer.start()
            
            return True
    
    def _apply_update(self, layer: str, updates: Dict[str, Any]):
        """Apply the actual update after debounce period."""
        try:
            with self._lock:
                if layer == 'ui':
                    success = self.ui_state.update(updates)
                elif layer == 'backend':
                    success = self.backend_state.update(updates)
                else:
                    success = False
                
                if success:
                    # Notify subscribers
                    self._notify_subscribers(layer, updates)
                    
                    # Persist UI state changes
                    if layer == 'ui':
                        self._persist_state()
                        
                    # Clean up debounce timer
                    timer_key = f"{layer}_debounce"
                    if timer_key in self._debounce_timers:
                        del self._debounce_timers[timer_key]
        except Exception as e:
            self.logger.error(f"Failed to apply {layer} update: {e}")
    
    def _optimistic_backend_update(self, updates: Dict[str, Any]) -> bool:
        """Apply optimistic backend update with rollback capability."""
        with self._lock:
            # Store rollback state
            rollback_state = self.backend_state.get_state()
            
            try:
                # Apply optimistic update
                success = self.backend_state.update(updates, optimistic=True)
                if success:
                    self._notify_subscribers('backend', updates)
                    
                    # Store rollback info for potential conflicts
                    self.backend_state.store_rollback_point(rollback_state)
                    
                return success
            except Exception as e:
                self.logger.error(f"Optimistic update failed: {e}")
                # Rollback on failure
                self.backend_state.restore_state(rollback_state)
                return False
    
    def _direct_backend_update(self, updates: Dict[str, Any]) -> bool:
        """Apply direct backend update without optimistic behavior."""
        with self._lock:
            try:
                success = self.backend_state.update(updates, optimistic=False)
                if success:
                    self._notify_subscribers('backend', updates)
                return success
            except Exception as e:
                self.logger.error(f"Direct backend update failed: {e}")
                return False
    
    def handle_backend_conflict(self, server_state: Dict[str, Any]) -> bool:
        """Handle backend state conflicts by rolling back optimistic updates."""
        with self._lock:
            try:
                # Rollback to last known good state
                success = self.backend_state.handle_conflict(server_state)
                if success:
                    self._notify_subscribers('backend', server_state)
                return success
            except Exception as e:
                self.logger.error(f"Conflict resolution failed: {e}")
                return False
    
    def queue_backend_update(self, updates: Dict[str, Any]) -> bool:
        """Queue backend update during active UI transitions."""
        if self.transitions.is_active():
            return self.transitions.queue_update('backend', updates)
        else:
            return self.update_backend_state(updates)
    
    def start_transition(self, transition_type: str, duration_ms: int = 300) -> str:
        """Start a UI transition that will queue backend updates."""
        return self.transitions.start(transition_type, duration_ms)
    
    def complete_transition(self, transition_type: str):
        """Complete a UI transition and apply queued backend updates."""
        queued_updates = self.transitions.complete(transition_type)
        
        # Apply queued updates
        for layer, updates in queued_updates:
            if layer == 'backend':
                self._direct_backend_update(updates)
    
    def _notify_subscribers(self, layer: str, updates: Dict[str, Any]):
        """Notify all subscribers of state changes."""
        # Notify layer-specific subscribers
        for callback in self._subscribers.get(layer, []):
            try:
                callback(updates)
            except Exception as e:
                self.logger.error(f"Subscriber callback failed: {e}")
        
        # Notify global subscribers
        for callback in self._subscribers.get('global', []):
            try:
                callback({'layer': layer, 'updates': updates})
            except Exception as e:
                self.logger.error(f"Global subscriber callback failed: {e}")
    
    def _persist_state(self):
        """Persist current UI state to storage."""
        try:
            state_to_persist = {
                'ui': self.ui_state.get_persistent_state(),
                'timestamp': datetime.utcnow().isoformat()
            }
            self.persistence.save_state(state_to_persist)
        except Exception as e:
            self.logger.error(f"Failed to persist state: {e}")
    
    def reset_to_defaults(self):
        """Reset store to default state."""
        with self._lock:
            self.ui_state.reset()
            self.backend_state.reset()
            self.persistence.clear_state()
            self._notify_subscribers('global', {'action': 'reset'})
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get store diagnostics for debugging."""
        with self._lock:
            return {
                'ui_state_version': self.ui_state.get_version(),
                'backend_state_version': self.backend_state.get_version(),
                'active_transitions': self.transitions.get_active_transitions(),
                'subscriber_counts': {
                    layer: len(callbacks) 
                    for layer, callbacks in self._subscribers.items()
                },
                'queued_updates': self.transitions.get_queued_count(),
                'debounce_timers_active': len(self._debounce_timers)
            }
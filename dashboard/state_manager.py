"""
Main state management orchestrator that coordinates all dashboard state components.
Provides a high-level API for dashboard state operations.
"""

import asyncio
import threading
# # # from typing import Dict, Any, Optional, Callable, List, Tuple  # Module not found  # Module not found  # Module not found
# # # from datetime import datetime  # Module not found  # Module not found  # Module not found
import logging

# # # from .store import DashboardStore  # Module not found  # Module not found  # Module not found
# # # from .transitions import TransitionContext  # Module not found  # Module not found  # Module not found


class StateManager:
    """High-level state management orchestrator for dashboard operations."""
    
    def __init__(self, persistence_key: str = "dashboard_state", 
                 auto_save_interval: float = 30.0):
        self.logger = logging.getLogger(__name__)
        
        # Core store
        self.store = DashboardStore(persistence_key)
        
        # Auto-save configuration
        self.auto_save_interval = auto_save_interval
        self._auto_save_timer: Optional[threading.Timer] = None
        self._should_auto_save = True
        
        # Event callbacks
        self._event_callbacks: Dict[str, List[Callable]] = {
            'state_changed': [],
            'transition_started': [],
            'transition_completed': [],
            'conflict_resolved': [],
            'error_occurred': []
        }
        
        # Subscribe to store changes
        self.store.subscribe('global', self._handle_store_change)
        
        # Start auto-save
        self._start_auto_save()
        
    def _start_auto_save(self):
        """Start automatic saving of state."""
        if self._should_auto_save and self.auto_save_interval > 0:
            self._auto_save_timer = threading.Timer(
                self.auto_save_interval,
                self._auto_save
            )
            self._auto_save_timer.daemon = True
            self._auto_save_timer.start()
    
    def _auto_save(self):
        """Perform automatic save."""
        try:
            self.store._persist_state()
            self.logger.debug("Auto-save completed")
        except Exception as e:
            self.logger.error(f"Auto-save failed: {e}")
        finally:
            # Schedule next auto-save
            if self._should_auto_save:
                self._start_auto_save()
    
    def _handle_store_change(self, change_data: Dict[str, Any]):
# # #         """Handle changes from the store."""  # Module not found  # Module not found  # Module not found
        self._emit_event('state_changed', change_data)
    
    def _emit_event(self, event_type: str, data: Dict[str, Any]):
        """Emit event to registered callbacks."""
        for callback in self._event_callbacks.get(event_type, []):
            try:
                callback(data)
            except Exception as e:
                self.logger.error(f"Event callback failed: {e}")
    
    # Event subscription
    
    def on(self, event_type: str, callback: Callable[[Dict[str, Any]], None]) -> str:
        """Subscribe to state manager events."""
        if event_type not in self._event_callbacks:
            self._event_callbacks[event_type] = []
        
        self._event_callbacks[event_type].append(callback)
        return f"{event_type}_{len(self._event_callbacks[event_type]) - 1}"
    
    def off(self, event_type: str, callback_or_id):
# # #         """Unsubscribe from state manager events."""  # Module not found  # Module not found  # Module not found
        if event_type in self._event_callbacks:
            if callable(callback_or_id):
                # Remove by callback function
                try:
                    self._event_callbacks[event_type].remove(callback_or_id)
                except ValueError:
                    pass
            # Note: ID-based removal would require more tracking
    
    # State operations
    
    def get_state(self, layer: Optional[str] = None) -> Dict[str, Any]:
        """Get current state."""
        return self.store.get_state(layer)
    
    def update_ui_state(self, updates: Dict[str, Any], debounce_ms: int = 100) -> bool:
        """Update UI state with debouncing."""
        success = self.store.update_ui_state(updates, debounce_ms)
        if success:
            self._emit_event('state_changed', {
                'layer': 'ui',
                'updates': updates,
                'timestamp': datetime.utcnow().isoformat()
            })
        return success
    
    def update_backend_state(self, updates: Dict[str, Any], optimistic: bool = True) -> bool:
        """Update backend state."""
        success = self.store.update_backend_state(updates, optimistic)
        if success:
            self._emit_event('state_changed', {
                'layer': 'backend',
                'updates': updates,
                'optimistic': optimistic,
                'timestamp': datetime.utcnow().isoformat()
            })
        return success
    
    def handle_backend_conflict(self, server_state: Dict[str, Any]) -> bool:
        """Handle backend state conflicts."""
        success = self.store.handle_backend_conflict(server_state)
        if success:
            self._emit_event('conflict_resolved', {
                'server_state': server_state,
                'timestamp': datetime.utcnow().isoformat()
            })
        return success
    
    # Transition management
    
    def create_transition_context(self, transition_type: str, duration_ms: int = 300,
                                metadata: Optional[Dict[str, Any]] = None) -> TransitionContext:
        """Create a transition context manager."""
        context = TransitionContext(
            self.store.transitions, 
            transition_type, 
            duration_ms, 
            metadata
        )
        
        # Add event emission
        original_enter = context.__enter__
        original_exit = context.__exit__
        
        def enhanced_enter():
            result = original_enter()
            self._emit_event('transition_started', {
                'transition_type': transition_type,
                'duration_ms': duration_ms,
                'transition_id': context.transition_id,
                'timestamp': datetime.utcnow().isoformat()
            })
            return result
        
        def enhanced_exit(exc_type, exc_val, exc_tb):
            queued_updates = original_exit(exc_type, exc_val, exc_tb)
            self._emit_event('transition_completed', {
                'transition_type': transition_type,
                'transition_id': context.transition_id,
                'queued_updates_count': len(queued_updates) if queued_updates else 0,
                'timestamp': datetime.utcnow().isoformat()
            })
            return queued_updates
        
        context.__enter__ = enhanced_enter
        context.__exit__ = enhanced_exit
        
        return context
    
    def start_transition(self, transition_type: str, duration_ms: int = 300) -> str:
        """Start a UI transition."""
        transition_id = self.store.start_transition(transition_type, duration_ms)
        self._emit_event('transition_started', {
            'transition_type': transition_type,
            'duration_ms': duration_ms,
            'transition_id': transition_id,
            'timestamp': datetime.utcnow().isoformat()
        })
        return transition_id
    
    def complete_transition(self, transition_id: str):
        """Complete a UI transition."""
        self.store.complete_transition(transition_id)
        self._emit_event('transition_completed', {
            'transition_id': transition_id,
            'timestamp': datetime.utcnow().isoformat()
        })
    
    # High-level UI operations
    
    def show_radial_menu(self, menu_id: str, center: Tuple[float, float], 
                        items: Dict[str, Any], animated: bool = True) -> bool:
        """Show radial menu with optional animation."""
        if animated:
            with self.create_transition_context('radial_menu_expand', 300):
                return self.store.ui_state.set_radial_menu(menu_id, center, items, True)
        else:
            return self.store.ui_state.set_radial_menu(menu_id, center, items, True)
    
    def hide_radial_menu(self, menu_id: str, animated: bool = True) -> bool:
        """Hide radial menu with optional animation."""
        if animated:
            with self.create_transition_context('radial_menu_collapse', 300):
                return self.update_ui_state({
                    'radial_menus': {
                        menu_id: {'visible': False}
                    }
                })
        else:
            return self.update_ui_state({
                'radial_menus': {
                    menu_id: {'visible': False}
                }
            })
    
    def toggle_focus_mode(self, component: Optional[str] = None, 
                         level: str = 'enhanced', animated: bool = True) -> bool:
        """Toggle focus mode with optional animation."""
        if animated:
            with self.create_transition_context('focus_mode_transition', 500):
                return self.store.ui_state.toggle_focus_mode(component, level)
        else:
            return self.store.ui_state.toggle_focus_mode(component, level)
    
    def show_overlay(self, overlay_id: str, overlay_type: str, content: Dict[str, Any],
                    position: Optional[Tuple[float, float]] = None, animated: bool = True) -> bool:
        """Show overlay with optional animation."""
        if animated:
            with self.create_transition_context('overlay_fade_in', 250):
                return self.store.ui_state.show_overlay(overlay_id, overlay_type, content, position)
        else:
            return self.store.ui_state.show_overlay(overlay_id, overlay_type, content, position)
    
    def hide_overlay(self, overlay_id: str, animated: bool = True) -> bool:
        """Hide overlay with optional animation."""
        if animated:
            with self.create_transition_context('overlay_fade_out', 250):
                return self.store.ui_state.hide_overlay(overlay_id)
        else:
            return self.store.ui_state.hide_overlay(overlay_id)
    
    def update_viewport(self, zoom: Optional[float] = None, 
                       pan_x: Optional[float] = None, pan_y: Optional[float] = None,
                       rotation: Optional[float] = None, animated: bool = True) -> bool:
        """Update viewport with optional animation."""
        if animated and (zoom is not None or pan_x is not None or pan_y is not None):
            transition_type = 'viewport_zoom' if zoom is not None else 'viewport_pan'
            with self.create_transition_context(transition_type, 300):
                return self.store.ui_state.set_viewport(zoom, pan_x, pan_y, rotation)
        else:
            return self.store.ui_state.set_viewport(zoom, pan_x, pan_y, rotation)
    
    # Backend data operations
    
    def update_analysis_progress(self, stage: str, progress: float, 
                               status: str = "running", optimistic: bool = True) -> bool:
        """Update analysis progress."""
        return self.store.backend_state.update_analysis_progress(stage, progress, status, optimistic)
    
    def add_analysis_result(self, result_id: str, result_data: Dict[str, Any],
                          confidence: float = 1.0, optimistic: bool = True) -> bool:
        """Add analysis result."""
        return self.store.backend_state.add_analysis_result(result_id, result_data, confidence, optimistic)
    
    def update_real_time_metrics(self, metrics: Dict[str, float], 
                               optimistic: bool = False) -> bool:
        """Update real-time metrics."""
        return self.store.backend_state.update_real_time_metrics(metrics, optimistic)
    
    def add_alert(self, alert_id: str, alert_type: str, message: str,
                 severity: str = "info", optimistic: bool = True) -> bool:
        """Add system alert."""
        return self.store.backend_state.add_alert(alert_id, alert_type, message, severity, optimistic)
    
    def update_system_health(self, health_status: str, 
                           resource_info: Dict[str, Any], optimistic: bool = False) -> bool:
        """Update system health."""
        return self.store.backend_state.update_system_health(health_status, resource_info, optimistic)
    
    # Persistence operations
    
    def save_state(self) -> bool:
        """Manually save state."""
        try:
            self.store._persist_state()
            return True
        except Exception as e:
            self.logger.error(f"Manual save failed: {e}")
            self._emit_event('error_occurred', {
                'error_type': 'save_failed',
                'error_message': str(e),
                'timestamp': datetime.utcnow().isoformat()
            })
            return False
    
    def reset_state(self):
        """Reset to default state."""
        self.store.reset_to_defaults()
        self._emit_event('state_changed', {
            'action': 'reset',
            'timestamp': datetime.utcnow().isoformat()
        })
    
    def export_state(self, export_path: str) -> bool:
        """Export current state to file."""
        return self.store.persistence.export_state(export_path)
    
    def import_state(self, import_path: str) -> bool:
# # #         """Import state from file."""  # Module not found  # Module not found  # Module not found
        imported_state = self.store.persistence.import_state(import_path)
        if imported_state:
            # Apply imported UI state
            self.store.ui_state.restore_state(imported_state.get('ui', {}))
            self._emit_event('state_changed', {
                'action': 'imported',
                'source': import_path,
                'timestamp': datetime.utcnow().isoformat()
            })
            return True
        return False
    
    # Diagnostics and monitoring
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get comprehensive diagnostics."""
        return {
            'store': self.store.get_diagnostics(),
            'ui_state': {
                'version': self.store.ui_state.get_version(),
                'change_callbacks': len(self.store.ui_state._change_callbacks)
            },
            'backend_state': self.store.backend_state.get_diagnostics(),
            'persistence': self.store.persistence.get_diagnostics(),
            'transitions': self.store.transitions.get_diagnostics(),
            'state_manager': {
                'auto_save_interval': self.auto_save_interval,
                'auto_save_active': self._should_auto_save,
                'event_callback_counts': {
                    event_type: len(callbacks)
                    for event_type, callbacks in self._event_callbacks.items()
                }
            }
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        return {
            'transitions': self.store.transitions.get_performance_stats(),
            'backend_state': {
                'optimistic_updates': len(self.store.backend_state.get_optimistic_updates()),
                'rollback_points': len(self.store.backend_state.get_rollback_points())
            },
            'timestamp': datetime.utcnow().isoformat()
        }
    
    # Cleanup and shutdown
    
    def shutdown(self):
        """Shutdown state manager and cleanup resources."""
        self._should_auto_save = False
        
        if self._auto_save_timer:
            self._auto_save_timer.cancel()
        
        # Final save
        try:
            self.store._persist_state()
        except Exception as e:
            self.logger.error(f"Final save failed: {e}")
        
        # Cleanup transitions
        self.store.transitions.cancel_all_transitions()
        self.store.transitions.flush_queue()
        
        self.logger.info("State manager shutdown complete")
    
    # Context manager support
    
    def __enter__(self):
        """Enter context manager."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        self.shutdown()


# Convenience functions for common patterns

def create_radial_menu_transition(state_manager: StateManager, menu_id: str, 
                                center: Tuple[float, float], items: Dict[str, Any]):
    """Create a context manager for radial menu transitions."""
    class RadialMenuTransition:
        def __enter__(self):
            return state_manager.show_radial_menu(menu_id, center, items, animated=True)
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            return state_manager.hide_radial_menu(menu_id, animated=True)
    
    return RadialMenuTransition()


def batch_backend_updates(state_manager: StateManager):
    """Create a context manager for batching backend updates."""
    class BatchedUpdates:
        def __init__(self):
            self.updates = []
        
        def __enter__(self):
            return self
        
        def add_update(self, updates: Dict[str, Any], optimistic: bool = True):
            self.updates.append((updates, optimistic))
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            # Apply all updates
            for updates, optimistic in self.updates:
                state_manager.update_backend_state(updates, optimistic)
    
    return BatchedUpdates()
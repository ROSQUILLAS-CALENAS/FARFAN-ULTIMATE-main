"""
Backend state management for real-time analysis pipeline data.
Handles optimistic updates with rollback capabilities.
"""

# # # from typing import Dict, Any, Optional, List, Tuple  # Module not found  # Module not found  # Module not found
# # # from dataclasses import dataclass, asdict  # Module not found  # Module not found  # Module not found
# # # from datetime import datetime, timedelta  # Module not found  # Module not found  # Module not found
import threading
import json
import logging



# Mandatory Pipeline Contract Annotations
__phase__ = "O"
__code__ = "121O"
__stage_order__ = 7

@dataclass
class DataUpdate:
# # #     """Represents a single data update from the backend."""  # Module not found  # Module not found  # Module not found
    id: str
    data: Any
    timestamp: datetime
    version: int
    source: str
    confidence: float = 1.0


@dataclass
class RollbackPoint:
    """Represents a state rollback point."""
    state: Dict[str, Any]
    timestamp: datetime
    version: int
    reason: str


class BackendStateManager:
    """Manages backend data state with optimistic updates and rollback."""
    
    def __init__(self, rollback_limit: int = 10):
        self.logger = logging.getLogger(__name__)
        
        self._state = {
            'analysis_pipeline': {
                'status': 'idle',
                'progress': 0.0,
                'current_stage': None,
                'results': {},
                'errors': [],
                'warnings': []
            },
            'real_time_data': {
                'metrics': {},
                'alerts': [],
                'performance': {},
                'connections': {}
            },
            'processed_data': {
                'documents': {},
                'embeddings': {},
                'classifications': {},
                'aggregations': {}
            },
            'system_status': {
                'health': 'healthy',
                'resources': {},
                'services': {},
                'last_heartbeat': None
            }
        }
        
        self._version = 0
        self._lock = threading.RLock()
        
        # Optimistic update tracking
        self._optimistic_updates: Dict[str, DataUpdate] = {}
        self._pending_confirmations: Dict[str, datetime] = {}
        self._confirmation_timeout = timedelta(seconds=30)
        
        # Rollback management
        self._rollback_points: List[RollbackPoint] = []
        self._rollback_limit = rollback_limit
        
        # Conflict resolution
        self._conflict_handlers: Dict[str, callable] = {}
        
    def get_state(self) -> Dict[str, Any]:
        """Get current backend state."""
        with self._lock:
            return self._deep_copy_state(self._state)
    
    def update(self, updates: Dict[str, Any], optimistic: bool = False) -> bool:
        """Update backend state with optional optimistic behavior."""
        with self._lock:
            try:
                if optimistic:
                    return self._apply_optimistic_update(updates)
                else:
                    return self._apply_confirmed_update(updates)
            except Exception as e:
                self.logger.error(f"Backend state update failed: {e}")
                return False
    
    def _apply_optimistic_update(self, updates: Dict[str, Any]) -> bool:
        """Apply optimistic update that can be rolled back."""
        update_id = f"opt_{self._version}_{datetime.utcnow().timestamp()}"
        
        # Store current state for potential rollback
        rollback_point = RollbackPoint(
            state=self._deep_copy_state(self._state),
            timestamp=datetime.utcnow(),
            version=self._version,
            reason=f"Optimistic update {update_id}"
        )
        
        # Apply updates
        self._apply_state_updates(updates)
        self._version += 1
        
        # Track optimistic update
        data_update = DataUpdate(
            id=update_id,
            data=updates,
            timestamp=datetime.utcnow(),
            version=self._version,
            source="optimistic",
            confidence=0.8  # Lower confidence for optimistic updates
        )
        
        self._optimistic_updates[update_id] = data_update
        self._pending_confirmations[update_id] = datetime.utcnow()
        
        # Store rollback point
        self._add_rollback_point(rollback_point)
        
        return True
    
    def _apply_confirmed_update(self, updates: Dict[str, Any]) -> bool:
# # #         """Apply confirmed update from backend."""  # Module not found  # Module not found  # Module not found
        # Apply updates
        self._apply_state_updates(updates)
        self._version += 1
        
        # Clear any matching optimistic updates
        self._confirm_optimistic_updates(updates)
        
        return True
    
    def _apply_state_updates(self, updates: Dict[str, Any]):
        """Apply updates to internal state."""
        for key, value in updates.items():
            if key in self._state:
                if isinstance(self._state[key], dict) and isinstance(value, dict):
                    self._merge_dict_updates(self._state[key], value)
                else:
                    self._state[key] = value
    
    def _merge_dict_updates(self, target: Dict[str, Any], updates: Dict[str, Any]):
        """Recursively merge dictionary updates."""
        for key, value in updates.items():
            if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                self._merge_dict_updates(target[key], value)
            else:
                target[key] = value
    
    def _confirm_optimistic_updates(self, confirmed_data: Dict[str, Any]):
        """Confirm optimistic updates that match confirmed data."""
        to_remove = []
        
        for update_id, optimistic_update in self._optimistic_updates.items():
            if self._updates_match(optimistic_update.data, confirmed_data):
                to_remove.append(update_id)
        
        # Remove confirmed updates
        for update_id in to_remove:
            self._optimistic_updates.pop(update_id, None)
            self._pending_confirmations.pop(update_id, None)
    
    def _updates_match(self, optimistic_data: Dict[str, Any], confirmed_data: Dict[str, Any]) -> bool:
        """Check if optimistic and confirmed updates match."""
        # Simple matching logic - can be made more sophisticated
        for key in optimistic_data:
            if key in confirmed_data:
                if optimistic_data[key] != confirmed_data[key]:
                    return False
        return True
    
    def handle_conflict(self, server_state: Dict[str, Any]) -> bool:
        """Handle conflicts between optimistic updates and server state."""
        with self._lock:
            try:
                # Find the most recent rollback point before conflicts
                rollback_target = self._find_pre_conflict_state()
                
                if rollback_target:
                    # Rollback to safe state
                    self._state = self._deep_copy_state(rollback_target.state)
                    self._version = rollback_target.version
                    
                    # Clear conflicting optimistic updates
                    self._clear_optimistic_updates()
                    
                    # Apply server state
                    self._apply_state_updates(server_state)
                    self._version += 1
                    
                    self.logger.info(f"Rolled back to version {rollback_target.version} due to conflict")
                    return True
                else:
                    # No rollback point available, force accept server state
                    self._state = self._deep_copy_state(server_state)
                    self._version += 1
                    self._clear_optimistic_updates()
                    
                    self.logger.warning("No rollback point available, accepting server state")
                    return True
                    
            except Exception as e:
                self.logger.error(f"Conflict resolution failed: {e}")
                return False
    
    def _find_pre_conflict_state(self) -> Optional[RollbackPoint]:
        """Find the most recent rollback point before optimistic updates."""
        if not self._rollback_points:
            return None
        
        # Return the most recent rollback point
        return self._rollback_points[-1] if self._rollback_points else None
    
    def _clear_optimistic_updates(self):
        """Clear all optimistic updates."""
        self._optimistic_updates.clear()
        self._pending_confirmations.clear()
    
    def store_rollback_point(self, state: Dict[str, Any], reason: str = "manual"):
        """Manually store a rollback point."""
        rollback_point = RollbackPoint(
            state=self._deep_copy_state(state),
            timestamp=datetime.utcnow(),
            version=self._version,
            reason=reason
        )
        self._add_rollback_point(rollback_point)
    
    def _add_rollback_point(self, rollback_point: RollbackPoint):
        """Add rollback point and maintain limit."""
        self._rollback_points.append(rollback_point)
        
        # Maintain rollback point limit
        if len(self._rollback_points) > self._rollback_limit:
            self._rollback_points.pop(0)
    
    def cleanup_expired_updates(self):
        """Clean up expired optimistic updates."""
        with self._lock:
            current_time = datetime.utcnow()
            expired_updates = []
            
            for update_id, timestamp in self._pending_confirmations.items():
                if current_time - timestamp > self._confirmation_timeout:
                    expired_updates.append(update_id)
            
            # Remove expired updates
            for update_id in expired_updates:
                self._optimistic_updates.pop(update_id, None)
                self._pending_confirmations.pop(update_id, None)
                self.logger.warning(f"Optimistic update {update_id} expired")
    
    def restore_state(self, state: Dict[str, Any]):
# # #         """Restore state from external source."""  # Module not found  # Module not found  # Module not found
        with self._lock:
            self._state = self._deep_copy_state(state)
            self._version += 1
            self._clear_optimistic_updates()
    
    def reset(self):
        """Reset to default state."""
        with self._lock:
            self.__init__(self._rollback_limit)
    
    def get_version(self) -> int:
        """Get current state version."""
        return self._version
    
    def _deep_copy_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Create deep copy of state."""
        import copy
        return copy.deepcopy(state)
    
    # Data-specific update methods
    
    def update_analysis_progress(self, stage: str, progress: float, 
                               status: str = "running", optimistic: bool = True) -> bool:
        """Update analysis pipeline progress."""
        updates = {
            'analysis_pipeline': {
                'current_stage': stage,
                'progress': max(0.0, min(1.0, progress)),
                'status': status,
                'last_update': datetime.utcnow().isoformat()
            }
        }
        return self.update(updates, optimistic=optimistic)
    
    def add_analysis_result(self, result_id: str, result_data: Dict[str, Any],
                          confidence: float = 1.0, optimistic: bool = True) -> bool:
        """Add new analysis result."""
        updates = {
            'analysis_pipeline': {
                'results': {
                    result_id: {
                        'data': result_data,
                        'confidence': confidence,
                        'timestamp': datetime.utcnow().isoformat()
                    }
                }
            }
        }
        return self.update(updates, optimistic=optimistic)
    
    def update_real_time_metrics(self, metrics: Dict[str, float], 
                               optimistic: bool = False) -> bool:
        """Update real-time performance metrics."""
        updates = {
            'real_time_data': {
                'metrics': metrics,
                'last_update': datetime.utcnow().isoformat()
            }
        }
        return self.update(updates, optimistic=optimistic)
    
    def add_alert(self, alert_id: str, alert_type: str, message: str,
                 severity: str = "info", optimistic: bool = True) -> bool:
        """Add new system alert."""
        alert = {
            'id': alert_id,
            'type': alert_type,
            'message': message,
            'severity': severity,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Add to alerts list
        current_alerts = self._state.get('real_time_data', {}).get('alerts', [])
        updated_alerts = current_alerts + [alert]
        
        # Keep only recent alerts (last 100)
        if len(updated_alerts) > 100:
            updated_alerts = updated_alerts[-100:]
        
        updates = {
            'real_time_data': {
                'alerts': updated_alerts
            }
        }
        return self.update(updates, optimistic=optimistic)
    
    def update_system_health(self, health_status: str, 
                           resource_info: Dict[str, Any], optimistic: bool = False) -> bool:
        """Update system health status."""
        updates = {
            'system_status': {
                'health': health_status,
                'resources': resource_info,
                'last_heartbeat': datetime.utcnow().isoformat()
            }
        }
        return self.update(updates, optimistic=optimistic)
    
    def get_optimistic_updates(self) -> List[DataUpdate]:
        """Get list of pending optimistic updates."""
        with self._lock:
            return list(self._optimistic_updates.values())
    
    def get_rollback_points(self) -> List[RollbackPoint]:
        """Get available rollback points."""
        with self._lock:
            return self._rollback_points.copy()
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostic information."""
        with self._lock:
            return {
                'version': self._version,
                'optimistic_updates_count': len(self._optimistic_updates),
                'pending_confirmations': len(self._pending_confirmations),
                'rollback_points': len(self._rollback_points),
                'oldest_optimistic_update': min(
                    (update.timestamp for update in self._optimistic_updates.values()),
                    default=None
                ),
                'state_size_estimate': len(json.dumps(self._state, default=str))
            }
"""
UI state management for dashboard interactions.
Handles radial menu positions, focus mode toggles, overlay visibility.
"""

from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import threading
import math


@dataclass
class RadialMenuConfig:
    """Configuration for radial menu position and state."""
    center_x: float
    center_y: float
    radius: float
    rotation: float
    items: Dict[str, Any]
    visible: bool = False
    expanded: bool = False


@dataclass
class OverlayState:
    """State for dashboard overlays."""
    visible: bool
    type: str
    content: Dict[str, Any]
    position: Optional[Tuple[float, float]] = None
    opacity: float = 1.0


class UIStateManager:
    """Manages UI interaction state with persistence and validation."""
    
    def __init__(self):
        self._state = {
            'radial_menus': {},  # Dict[str, RadialMenuConfig]
            'focus_mode': {
                'enabled': False,
                'component': None,
                'level': 'normal'  # normal, enhanced, minimal
            },
            'overlays': {},  # Dict[str, OverlayState]
            'viewport': {
                'zoom': 1.0,
                'pan_x': 0.0,
                'pan_y': 0.0,
                'rotation': 0.0
            },
            'interactions': {
                'active_drag': None,
                'hover_target': None,
                'selected_items': set(),
                'keyboard_focus': None
            },
            'layout': {
                'sidebar_width': 300,
                'sidebar_collapsed': False,
                'grid_snap': True,
                'grid_size': 20
            }
        }
        
        self._version = 0
        self._lock = threading.RLock()
        self._change_callbacks = []
        
    def get_state(self) -> Dict[str, Any]:
        """Get current UI state."""
        with self._lock:
            # Convert sets to lists for serialization
            state_copy = self._deep_copy_state(self._state)
            state_copy['interactions']['selected_items'] = list(
                state_copy['interactions']['selected_items']
            )
            return state_copy
    
    def get_persistent_state(self) -> Dict[str, Any]:
        """Get state that should be persisted across sessions."""
        with self._lock:
            return {
                'radial_menus': self._state['radial_menus'],
                'focus_mode': {
                    'level': self._state['focus_mode']['level']
                },
                'layout': self._state['layout'],
                'viewport': self._state['viewport']
            }
    
    def update(self, updates: Dict[str, Any]) -> bool:
        """Update UI state with validation."""
        with self._lock:
            try:
                validated_updates = self._validate_updates(updates)
                if not validated_updates:
                    return False
                
                old_state = self._deep_copy_state(self._state)
                
                # Apply updates
                self._apply_updates(validated_updates)
                self._version += 1
                
                # Notify change callbacks
                self._notify_change_callbacks(old_state, validated_updates)
                
                return True
                
            except Exception as e:
                # Rollback on error
                return False
    
    def _validate_updates(self, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Validate update structure and values."""
        validated = {}
        
        # Validate radial menu updates
        if 'radial_menus' in updates:
            validated['radial_menus'] = self._validate_radial_menus(
                updates['radial_menus']
            )
        
        # Validate focus mode updates
        if 'focus_mode' in updates:
            validated['focus_mode'] = self._validate_focus_mode(
                updates['focus_mode']
            )
        
        # Validate overlay updates
        if 'overlays' in updates:
            validated['overlays'] = self._validate_overlays(
                updates['overlays']
            )
        
        # Validate viewport updates
        if 'viewport' in updates:
            validated['viewport'] = self._validate_viewport(
                updates['viewport']
            )
        
        # Validate interaction updates
        if 'interactions' in updates:
            validated['interactions'] = self._validate_interactions(
                updates['interactions']
            )
        
        # Validate layout updates
        if 'layout' in updates:
            validated['layout'] = self._validate_layout(
                updates['layout']
            )
        
        return validated
    
    def _validate_radial_menus(self, menus: Dict[str, Any]) -> Dict[str, Any]:
        """Validate radial menu configurations."""
        validated = {}
        
        for menu_id, config in menus.items():
            if isinstance(config, dict):
                validated[menu_id] = {
                    'center_x': self._clamp_float(config.get('center_x', 0), -9999, 9999),
                    'center_y': self._clamp_float(config.get('center_y', 0), -9999, 9999),
                    'radius': self._clamp_float(config.get('radius', 100), 10, 500),
                    'rotation': self._normalize_angle(config.get('rotation', 0)),
                    'items': config.get('items', {}),
                    'visible': bool(config.get('visible', False)),
                    'expanded': bool(config.get('expanded', False))
                }
        
        return validated
    
    def _validate_focus_mode(self, focus_config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate focus mode configuration."""
        valid_levels = {'normal', 'enhanced', 'minimal'}
        
        return {
            'enabled': bool(focus_config.get('enabled', False)),
            'component': focus_config.get('component'),
            'level': focus_config.get('level', 'normal') if focus_config.get('level') in valid_levels else 'normal'
        }
    
    def _validate_overlays(self, overlays: Dict[str, Any]) -> Dict[str, Any]:
        """Validate overlay configurations."""
        validated = {}
        
        for overlay_id, config in overlays.items():
            if isinstance(config, dict):
                validated[overlay_id] = {
                    'visible': bool(config.get('visible', False)),
                    'type': str(config.get('type', 'info')),
                    'content': config.get('content', {}),
                    'position': self._validate_position(config.get('position')),
                    'opacity': self._clamp_float(config.get('opacity', 1.0), 0.0, 1.0)
                }
        
        return validated
    
    def _validate_viewport(self, viewport: Dict[str, Any]) -> Dict[str, Any]:
        """Validate viewport configuration."""
        return {
            'zoom': self._clamp_float(viewport.get('zoom', 1.0), 0.1, 10.0),
            'pan_x': self._clamp_float(viewport.get('pan_x', 0), -10000, 10000),
            'pan_y': self._clamp_float(viewport.get('pan_y', 0), -10000, 10000),
            'rotation': self._normalize_angle(viewport.get('rotation', 0))
        }
    
    def _validate_interactions(self, interactions: Dict[str, Any]) -> Dict[str, Any]:
        """Validate interaction state."""
        validated = {}
        
        if 'active_drag' in interactions:
            validated['active_drag'] = interactions['active_drag']
        
        if 'hover_target' in interactions:
            validated['hover_target'] = interactions['hover_target']
        
        if 'selected_items' in interactions:
            items = interactions['selected_items']
            if isinstance(items, (list, set)):
                validated['selected_items'] = set(items)
        
        if 'keyboard_focus' in interactions:
            validated['keyboard_focus'] = interactions['keyboard_focus']
        
        return validated
    
    def _validate_layout(self, layout: Dict[str, Any]) -> Dict[str, Any]:
        """Validate layout configuration."""
        return {
            'sidebar_width': self._clamp_int(layout.get('sidebar_width', 300), 100, 800),
            'sidebar_collapsed': bool(layout.get('sidebar_collapsed', False)),
            'grid_snap': bool(layout.get('grid_snap', True)),
            'grid_size': self._clamp_int(layout.get('grid_size', 20), 5, 100)
        }
    
    def _validate_position(self, position) -> Optional[Tuple[float, float]]:
        """Validate position tuple."""
        if isinstance(position, (list, tuple)) and len(position) == 2:
            try:
                return (float(position[0]), float(position[1]))
            except (ValueError, TypeError):
                pass
        return None
    
    def _clamp_float(self, value, min_val: float, max_val: float) -> float:
        """Clamp float value to range."""
        try:
            return max(min_val, min(max_val, float(value)))
        except (ValueError, TypeError):
            return min_val
    
    def _clamp_int(self, value, min_val: int, max_val: int) -> int:
        """Clamp integer value to range."""
        try:
            return max(min_val, min(max_val, int(value)))
        except (ValueError, TypeError):
            return min_val
    
    def _normalize_angle(self, angle) -> float:
        """Normalize angle to 0-360 range."""
        try:
            angle = float(angle) % 360
            return angle if angle >= 0 else angle + 360
        except (ValueError, TypeError):
            return 0.0
    
    def _apply_updates(self, updates: Dict[str, Any]):
        """Apply validated updates to state."""
        for key, value in updates.items():
            if key in self._state:
                if isinstance(self._state[key], dict) and isinstance(value, dict):
                    self._state[key].update(value)
                else:
                    self._state[key] = value
    
    def _deep_copy_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Create deep copy of state."""
        import copy
        return copy.deepcopy(state)
    
    def _notify_change_callbacks(self, old_state: Dict[str, Any], updates: Dict[str, Any]):
        """Notify registered change callbacks."""
        for callback in self._change_callbacks:
            try:
                callback(old_state, updates)
            except Exception:
                pass  # Ignore callback errors
    
    def restore_state(self, state: Dict[str, Any]):
        """Restore state from persistence."""
        with self._lock:
            validated_state = self._validate_updates(state)
            if validated_state:
                self._apply_updates(validated_state)
                self._version += 1
    
    def reset(self):
        """Reset to default state."""
        with self._lock:
            self.__init__()
    
    def get_version(self) -> int:
        """Get current state version."""
        return self._version
    
    def add_change_callback(self, callback):
        """Add callback for state changes."""
        self._change_callbacks.append(callback)
    
    def remove_change_callback(self, callback):
        """Remove state change callback."""
        if callback in self._change_callbacks:
            self._change_callbacks.remove(callback)
    
    # Convenience methods for common operations
    
    def set_radial_menu(self, menu_id: str, center: Tuple[float, float], 
                       items: Dict[str, Any], visible: bool = True) -> bool:
        """Set radial menu configuration."""
        return self.update({
            'radial_menus': {
                menu_id: {
                    'center_x': center[0],
                    'center_y': center[1],
                    'radius': 100,
                    'rotation': 0,
                    'items': items,
                    'visible': visible,
                    'expanded': False
                }
            }
        })
    
    def toggle_focus_mode(self, component: Optional[str] = None, 
                         level: str = 'enhanced') -> bool:
        """Toggle focus mode."""
        current = self._state['focus_mode']['enabled']
        return self.update({
            'focus_mode': {
                'enabled': not current,
                'component': component,
                'level': level
            }
        })
    
    def show_overlay(self, overlay_id: str, overlay_type: str, 
                    content: Dict[str, Any], position: Optional[Tuple[float, float]] = None) -> bool:
        """Show overlay with content."""
        return self.update({
            'overlays': {
                overlay_id: {
                    'visible': True,
                    'type': overlay_type,
                    'content': content,
                    'position': position,
                    'opacity': 1.0
                }
            }
        })
    
    def hide_overlay(self, overlay_id: str) -> bool:
        """Hide specific overlay."""
        return self.update({
            'overlays': {
                overlay_id: {
                    'visible': False
                }
            }
        })
    
    def set_viewport(self, zoom: Optional[float] = None, pan_x: Optional[float] = None,
                    pan_y: Optional[float] = None, rotation: Optional[float] = None) -> bool:
        """Update viewport settings."""
        viewport_updates = {}
        
        if zoom is not None:
            viewport_updates['zoom'] = zoom
        if pan_x is not None:
            viewport_updates['pan_x'] = pan_x
        if pan_y is not None:
            viewport_updates['pan_y'] = pan_y
        if rotation is not None:
            viewport_updates['rotation'] = rotation
        
        return self.update({'viewport': viewport_updates}) if viewport_updates else True
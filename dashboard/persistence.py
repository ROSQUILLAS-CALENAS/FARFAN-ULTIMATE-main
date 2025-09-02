"""
State persistence mechanisms for dashboard configurations.
Preserves radial menu configurations and focus mode preferences across sessions.
"""

import json
import os
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging
import threading
from pathlib import Path


class StatePersistence:
    """Handles persistence of dashboard state to local storage."""
    
    def __init__(self, storage_key: str = "dashboard_state", 
                 storage_dir: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.storage_key = storage_key
        
        # Set up storage directory
        if storage_dir is None:
            # Use user's data directory
            if os.name == 'nt':  # Windows
                base_dir = os.path.expanduser("~/AppData/Local")
            else:  # Unix-like
                base_dir = os.path.expanduser("~/.local/share")
            
            storage_dir = os.path.join(base_dir, "dashboard_state")
        
        self.storage_dir = Path(storage_dir)
        self.storage_file = self.storage_dir / f"{storage_key}.json"
        
        # Ensure storage directory exists
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self._lock = threading.Lock()
        
        # Backup settings
        self.max_backups = 5
        self.backup_dir = self.storage_dir / "backups"
        self.backup_dir.mkdir(exist_ok=True)
    
    def save_state(self, state: Dict[str, Any]) -> bool:
        """Save state to persistent storage."""
        with self._lock:
            try:
                # Create backup of existing state
                if self.storage_file.exists():
                    self._create_backup()
                
                # Prepare state for serialization
                serializable_state = self._prepare_for_serialization(state)
                
                # Add metadata
                full_state = {
                    'version': '1.0',
                    'saved_at': datetime.utcnow().isoformat(),
                    'state': serializable_state
                }
                
                # Write to temporary file first
                temp_file = self.storage_file.with_suffix('.tmp')
                with open(temp_file, 'w', encoding='utf-8') as f:
                    json.dump(full_state, f, indent=2, default=self._json_serializer)
                
                # Atomic rename
                temp_file.replace(self.storage_file)
                
                self.logger.debug(f"State saved to {self.storage_file}")
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to save state: {e}")
                return False
    
    def load_state(self) -> Optional[Dict[str, Any]]:
        """Load state from persistent storage."""
        with self._lock:
            try:
                if not self.storage_file.exists():
                    self.logger.debug("No state file exists")
                    return None
                
                with open(self.storage_file, 'r', encoding='utf-8') as f:
                    full_state = json.load(f)
                
                # Validate structure
                if not isinstance(full_state, dict) or 'state' not in full_state:
                    self.logger.warning("Invalid state file format")
                    return None
                
                # Extract actual state
                state = full_state['state']
                
                # Deserialize special types
                deserialized_state = self._prepare_after_deserialization(state)
                
                self.logger.debug(f"State loaded from {self.storage_file}")
                return deserialized_state
                
            except Exception as e:
                self.logger.error(f"Failed to load state: {e}")
                # Try to load from backup
                return self._load_from_backup()
    
    def clear_state(self) -> bool:
        """Clear persisted state."""
        with self._lock:
            try:
                if self.storage_file.exists():
                    # Create backup before clearing
                    self._create_backup()
                    self.storage_file.unlink()
                    self.logger.info("State cleared")
                return True
            except Exception as e:
                self.logger.error(f"Failed to clear state: {e}")
                return False
    
    def _create_backup(self):
        """Create backup of current state file."""
        try:
            if not self.storage_file.exists():
                return
            
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            backup_file = self.backup_dir / f"{self.storage_key}_{timestamp}.json"
            
            # Copy current state to backup
            backup_file.write_bytes(self.storage_file.read_bytes())
            
            # Clean up old backups
            self._cleanup_old_backups()
            
            self.logger.debug(f"Backup created: {backup_file}")
            
        except Exception as e:
            self.logger.warning(f"Failed to create backup: {e}")
    
    def _cleanup_old_backups(self):
        """Remove old backup files, keeping only the most recent ones."""
        try:
            backup_files = list(self.backup_dir.glob(f"{self.storage_key}_*.json"))
            
            # Sort by modification time (newest first)
            backup_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
            
            # Remove old backups
            for backup_file in backup_files[self.max_backups:]:
                backup_file.unlink()
                self.logger.debug(f"Removed old backup: {backup_file}")
                
        except Exception as e:
            self.logger.warning(f"Failed to cleanup old backups: {e}")
    
    def _load_from_backup(self) -> Optional[Dict[str, Any]]:
        """Attempt to load state from most recent backup."""
        try:
            backup_files = list(self.backup_dir.glob(f"{self.storage_key}_*.json"))
            
            if not backup_files:
                return None
            
            # Get most recent backup
            most_recent = max(backup_files, key=lambda f: f.stat().st_mtime)
            
            with open(most_recent, 'r', encoding='utf-8') as f:
                full_state = json.load(f)
            
            if isinstance(full_state, dict) and 'state' in full_state:
                state = self._prepare_after_deserialization(full_state['state'])
                self.logger.info(f"State recovered from backup: {most_recent}")
                return state
                
        except Exception as e:
            self.logger.error(f"Failed to load from backup: {e}")
        
        return None
    
    def _prepare_for_serialization(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare state for JSON serialization."""
        serializable = {}
        
        for key, value in state.items():
            if key == 'interactions':
                # Handle special case of selected_items set
                interactions = value.copy()
                if 'selected_items' in interactions and isinstance(interactions['selected_items'], set):
                    interactions['selected_items'] = list(interactions['selected_items'])
                serializable[key] = interactions
            else:
                serializable[key] = self._serialize_value(value)
        
        return serializable
    
    def _prepare_after_deserialization(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare state after JSON deserialization."""
        deserialized = {}
        
        for key, value in state.items():
            if key == 'interactions':
                # Handle special case of selected_items list -> set
                interactions = value.copy()
                if 'selected_items' in interactions and isinstance(interactions['selected_items'], list):
                    interactions['selected_items'] = set(interactions['selected_items'])
                deserialized[key] = interactions
            else:
                deserialized[key] = self._deserialize_value(value)
        
        return deserialized
    
    def _serialize_value(self, value):
        """Serialize individual values."""
        if isinstance(value, set):
            return {'__type__': 'set', 'value': list(value)}
        elif isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self._serialize_value(item) for item in value]
        else:
            return value
    
    def _deserialize_value(self, value):
        """Deserialize individual values."""
        if isinstance(value, dict):
            if value.get('__type__') == 'set':
                return set(value['value'])
            else:
                return {k: self._deserialize_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self._deserialize_value(item) for item in value]
        else:
            return value
    
    def _json_serializer(self, obj):
        """Custom JSON serializer for special types."""
        if isinstance(obj, set):
            return list(obj)
        elif hasattr(obj, 'isoformat'):  # datetime objects
            return obj.isoformat()
        else:
            return str(obj)
    
    def get_backup_files(self) -> List[Dict[str, Any]]:
        """Get list of available backup files with metadata."""
        backup_files = []
        
        try:
            for backup_file in self.backup_dir.glob(f"{self.storage_key}_*.json"):
                stat = backup_file.stat()
                backup_files.append({
                    'filename': backup_file.name,
                    'path': str(backup_file),
                    'size': stat.st_size,
                    'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    'created': datetime.fromtimestamp(stat.st_ctime).isoformat()
                })
        except Exception as e:
            self.logger.error(f"Failed to get backup files: {e}")
        
        # Sort by modification time (newest first)
        backup_files.sort(key=lambda f: f['modified'], reverse=True)
        return backup_files
    
    def restore_from_backup(self, backup_filename: str) -> Optional[Dict[str, Any]]:
        """Restore state from specific backup file."""
        try:
            backup_file = self.backup_dir / backup_filename
            
            if not backup_file.exists():
                self.logger.error(f"Backup file not found: {backup_filename}")
                return None
            
            with open(backup_file, 'r', encoding='utf-8') as f:
                full_state = json.load(f)
            
            if isinstance(full_state, dict) and 'state' in full_state:
                state = self._prepare_after_deserialization(full_state['state'])
                self.logger.info(f"State restored from backup: {backup_filename}")
                return state
            else:
                self.logger.error(f"Invalid backup file format: {backup_filename}")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to restore from backup {backup_filename}: {e}")
            return None
    
    def export_state(self, export_path: str) -> bool:
        """Export current state to external file."""
        try:
            current_state = self.load_state()
            if current_state is None:
                self.logger.warning("No state to export")
                return False
            
            export_file = Path(export_path)
            export_file.parent.mkdir(parents=True, exist_ok=True)
            
            export_data = {
                'version': '1.0',
                'exported_at': datetime.utcnow().isoformat(),
                'source': self.storage_key,
                'state': current_state
            }
            
            with open(export_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, default=self._json_serializer)
            
            self.logger.info(f"State exported to: {export_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export state: {e}")
            return False
    
    def import_state(self, import_path: str) -> Optional[Dict[str, Any]]:
        """Import state from external file."""
        try:
            import_file = Path(import_path)
            
            if not import_file.exists():
                self.logger.error(f"Import file not found: {import_path}")
                return None
            
            with open(import_file, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            if isinstance(import_data, dict) and 'state' in import_data:
                state = self._prepare_after_deserialization(import_data['state'])
                self.logger.info(f"State imported from: {import_path}")
                return state
            else:
                self.logger.error(f"Invalid import file format: {import_path}")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to import state: {e}")
            return None
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get persistence diagnostics."""
        diagnostics = {
            'storage_file': str(self.storage_file),
            'storage_exists': self.storage_file.exists(),
            'storage_dir': str(self.storage_dir),
            'backup_dir': str(self.backup_dir),
            'backup_count': len(list(self.backup_dir.glob(f"{self.storage_key}_*.json"))),
            'max_backups': self.max_backups
        }
        
        try:
            if self.storage_file.exists():
                stat = self.storage_file.stat()
                diagnostics.update({
                    'storage_size': stat.st_size,
                    'last_modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
                })
        except Exception as e:
            diagnostics['storage_error'] = str(e)
        
        return diagnostics
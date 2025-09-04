"""
Integration Layer for Component Registry Synchronization

Provides bidirectional synchronization between component registry and 
canonical_flow index.json file, ensuring metadata consistency.
"""

import json
import sqlite3
import logging
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
from enum import Enum

__phase__ = "O"
__code__ = "141O"
__stage_order__ = 7

logger = logging.getLogger(__name__)


class SyncDirection(Enum):
    REGISTRY_TO_INDEX = "registry_to_index"
    INDEX_TO_REGISTRY = "index_to_registry"
    BIDIRECTIONAL = "bidirectional"


class LifecycleState(Enum):
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    EXPERIMENTAL = "experimental"
    MAINTENANCE = "maintenance"
    ARCHIVED = "archived"


@dataclass
class ComponentMetadata:
    """Component metadata structure"""
    code: str
    stage: str
    alias_path: str
    original_path: str
    owner: str = ""
    lifecycle_state: LifecycleState = LifecycleState.ACTIVE
    evidence_score: float = 0.0
    registration_date: datetime = None
    last_modified: datetime = None
    governance_waivers: List[str] = None
    
    def __post_init__(self):
        if self.registration_date is None:
            self.registration_date = datetime.now()
        if self.last_modified is None:
            self.last_modified = datetime.now()
        if self.governance_waivers is None:
            self.governance_waivers = []


class ComponentRegistry:
    """SQL-based component registry"""
    
    def __init__(self, db_path: str = "component_registry.db"):
        self.db_path = db_path
        self.connection = None
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize SQLite database schema"""
        try:
            self.connection = sqlite3.connect(self.db_path)
            self.connection.execute("""
                CREATE TABLE IF NOT EXISTS components (
                    code TEXT PRIMARY KEY,
                    stage TEXT NOT NULL,
                    alias_path TEXT NOT NULL,
                    original_path TEXT NOT NULL,
                    owner TEXT,
                    lifecycle_state TEXT DEFAULT 'active',
                    evidence_score REAL DEFAULT 0.0,
                    registration_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_modified TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    governance_waivers TEXT
                )
            """)
            
            self.connection.execute("""
                CREATE TABLE IF NOT EXISTS sync_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    operation TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    component_code TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    success BOOLEAN,
                    error_message TEXT
                )
            """)
            
            self.connection.commit()
            logger.info(f"Initialized component registry database: {self.db_path}")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    def register_component(self, metadata: ComponentMetadata) -> bool:
        """Register a component in the registry"""
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO components (
                    code, stage, alias_path, original_path, owner,
                    lifecycle_state, evidence_score, registration_date,
                    last_modified, governance_waivers
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metadata.code,
                metadata.stage,
                metadata.alias_path,
                metadata.original_path,
                metadata.owner,
                metadata.lifecycle_state.value,
                metadata.evidence_score,
                metadata.registration_date.isoformat(),
                metadata.last_modified.isoformat(),
                json.dumps(metadata.governance_waivers)
            ))
            
            self.connection.commit()
            logger.info(f"Registered component {metadata.code} in registry")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register component {metadata.code}: {e}")
            return False
    
    def get_component(self, code: str) -> Optional[ComponentMetadata]:
        """Get component metadata by code"""
        try:
            cursor = self.connection.cursor()
            cursor.execute("SELECT * FROM components WHERE code = ?", (code,))
            row = cursor.fetchone()
            
            if row:
                return ComponentMetadata(
                    code=row[0],
                    stage=row[1],
                    alias_path=row[2],
                    original_path=row[3],
                    owner=row[4] or "",
                    lifecycle_state=LifecycleState(row[5]),
                    evidence_score=row[6] or 0.0,
                    registration_date=datetime.fromisoformat(row[7]),
                    last_modified=datetime.fromisoformat(row[8]),
                    governance_waivers=json.loads(row[9] or "[]")
                )
            return None
            
        except Exception as e:
            logger.error(f"Failed to get component {code}: {e}")
            return None
    
    def list_components(self) -> List[ComponentMetadata]:
        """List all registered components"""
        try:
            cursor = self.connection.cursor()
            cursor.execute("SELECT * FROM components ORDER BY code")
            rows = cursor.fetchall()
            
            components = []
            for row in rows:
                components.append(ComponentMetadata(
                    code=row[0],
                    stage=row[1],
                    alias_path=row[2],
                    original_path=row[3],
                    owner=row[4] or "",
                    lifecycle_state=LifecycleState(row[5]),
                    evidence_score=row[6] or 0.0,
                    registration_date=datetime.fromisoformat(row[7]),
                    last_modified=datetime.fromisoformat(row[8]),
                    governance_waivers=json.loads(row[9] or "[]")
                ))
            
            return components
            
        except Exception as e:
            logger.error(f"Failed to list components: {e}")
            return []
    
    def update_lifecycle_state(self, code: str, state: LifecycleState, waivers: List[str] = None) -> bool:
        """Update component lifecycle state and waivers"""
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                UPDATE components 
                SET lifecycle_state = ?, governance_waivers = ?, last_modified = ?
                WHERE code = ?
            """, (
                state.value,
                json.dumps(waivers or []),
                datetime.now().isoformat(),
                code
            ))
            
            self.connection.commit()
            return cursor.rowcount > 0
            
        except Exception as e:
            logger.error(f"Failed to update lifecycle state for {code}: {e}")
            return False
    
    def log_sync_operation(self, operation: str, direction: str, component_code: str = None, 
                          success: bool = True, error_message: str = None):
        """Log synchronization operations"""
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                INSERT INTO sync_history (operation, direction, component_code, success, error_message)
                VALUES (?, ?, ?, ?, ?)
            """, (operation, direction, component_code, success, error_message))
            
            self.connection.commit()
            
        except Exception as e:
            logger.error(f"Failed to log sync operation: {e}")
    
    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()


class IntegrationLayer:
    """
    Integration layer that synchronizes component registry with canonical_flow index.json
    
    Provides bidirectional sync methods that update registry entries when index.json 
    changes and vice versa, ensuring component metadata consistency.
    """
    
    def __init__(self, index_path: str = "canonical_flow/index.json", 
                 registry_path: str = "component_registry.db"):
        self.index_path = Path(index_path)
        self.registry = ComponentRegistry(registry_path)
        self.last_index_hash = None
        
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate hash of file contents for change detection"""
        try:
            if file_path.exists():
                content = file_path.read_text()
                return hashlib.md5(content.encode()).hexdigest()
            return ""
        except Exception as e:
            logger.error(f"Failed to calculate hash for {file_path}: {e}")
            return ""
    
    def _load_index_data(self) -> List[Dict[str, Any]]:
        """Load data from index.json file"""
        try:
            if self.index_path.exists():
                with open(self.index_path, 'r') as f:
                    return json.load(f)
            return []
        except Exception as e:
            logger.error(f"Failed to load index data: {e}")
            return []
    
    def _save_index_data(self, data: List[Dict[str, Any]]) -> bool:
        """Save data to index.json file"""
        try:
            with open(self.index_path, 'w') as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Failed to save index data: {e}")
            return False
    
    def sync_index_to_registry(self) -> Dict[str, Any]:
        """Sync changes from index.json to registry"""
        result = {
            "direction": "index_to_registry",
            "timestamp": datetime.now().isoformat(),
            "components_updated": 0,
            "components_added": 0,
            "errors": []
        }
        
        try:
            index_data = self._load_index_data()
            
            for entry in index_data:
                code = entry.get("code")
                if not code:
                    continue
                
                # Check if component exists in registry
                existing = self.registry.get_component(code)
                
                # Create metadata from index entry
                metadata = ComponentMetadata(
                    code=code,
                    stage=entry.get("stage", ""),
                    alias_path=entry.get("alias_path", ""),
                    original_path=entry.get("original_path", ""),
                    owner=entry.get("owner", ""),
                    lifecycle_state=LifecycleState(entry.get("lifecycle_state", "active")),
                    evidence_score=entry.get("evidence_score", 0.0),
                    governance_waivers=entry.get("governance_waivers", [])
                )
                
                if existing:
                    # Update existing component if different
                    if (existing.alias_path != metadata.alias_path or 
                        existing.original_path != metadata.original_path or
                        existing.stage != metadata.stage):
                        
                        metadata.registration_date = existing.registration_date
                        if self.registry.register_component(metadata):
                            result["components_updated"] += 1
                            self.registry.log_sync_operation("update", "index_to_registry", code)
                else:
                    # Add new component
                    if self.registry.register_component(metadata):
                        result["components_added"] += 1
                        self.registry.log_sync_operation("add", "index_to_registry", code)
            
            logger.info(f"Synced {result['components_added']} new and {result['components_updated']} updated components from index")
            
        except Exception as e:
            error_msg = f"Failed to sync index to registry: {e}"
            logger.error(error_msg)
            result["errors"].append(error_msg)
            self.registry.log_sync_operation("sync", "index_to_registry", success=False, error_message=str(e))
        
        return result
    
    def sync_registry_to_index(self) -> Dict[str, Any]:
        """Sync changes from registry to index.json"""
        result = {
            "direction": "registry_to_index",
            "timestamp": datetime.now().isoformat(),
            "components_exported": 0,
            "errors": []
        }
        
        try:
            # Get all components from registry
            components = self.registry.list_components()
            
            # Convert to index format
            index_data = []
            for comp in components:
                entry = {
                    "code": comp.code,
                    "stage": comp.stage,
                    "alias_path": comp.alias_path,
                    "original_path": comp.original_path,
                    "owner": comp.owner,
                    "lifecycle_state": comp.lifecycle_state.value,
                    "evidence_score": comp.evidence_score,
                    "registration_date": comp.registration_date.isoformat(),
                    "last_modified": comp.last_modified.isoformat()
                }
                
                # Include governance waivers if any
                if comp.governance_waivers:
                    entry["governance_waivers"] = comp.governance_waivers
                
                index_data.append(entry)
            
            # Sort by code for consistency
            index_data.sort(key=lambda x: x["code"])
            
            # Save to index.json
            if self._save_index_data(index_data):
                result["components_exported"] = len(index_data)
                self.registry.log_sync_operation("export", "registry_to_index")
                logger.info(f"Exported {len(index_data)} components to index")
            else:
                raise Exception("Failed to save index data")
            
        except Exception as e:
            error_msg = f"Failed to sync registry to index: {e}"
            logger.error(error_msg)
            result["errors"].append(error_msg)
            self.registry.log_sync_operation("sync", "registry_to_index", success=False, error_message=str(e))
        
        return result
    
    def detect_changes(self) -> Dict[str, Any]:
        """Detect changes in index.json file"""
        current_hash = self._calculate_file_hash(self.index_path)
        
        changes = {
            "index_changed": False,
            "current_hash": current_hash,
            "previous_hash": self.last_index_hash
        }
        
        if self.last_index_hash and self.last_index_hash != current_hash:
            changes["index_changed"] = True
        
        self.last_index_hash = current_hash
        return changes
    
    def bidirectional_sync(self) -> Dict[str, Any]:
        """Perform bidirectional synchronization"""
        result = {
            "direction": "bidirectional",
            "timestamp": datetime.now().isoformat(),
            "index_to_registry": None,
            "registry_to_index": None,
            "changes_detected": False,
            "errors": []
        }
        
        try:
            # Detect changes
            changes = self.detect_changes()
            result["changes_detected"] = changes["index_changed"]
            
            # If index changed, sync to registry first
            if changes["index_changed"]:
                result["index_to_registry"] = self.sync_index_to_registry()
                logger.info("Performed index → registry sync due to detected changes")
            
            # Then sync registry back to index (to ensure consistency)
            result["registry_to_index"] = self.sync_registry_to_index()
            
            self.registry.log_sync_operation("bidirectional_sync", "bidirectional")
            
        except Exception as e:
            error_msg = f"Failed bidirectional sync: {e}"
            logger.error(error_msg)
            result["errors"].append(error_msg)
            self.registry.log_sync_operation("bidirectional_sync", "bidirectional", 
                                           success=False, error_message=str(e))
        
        return result
    
    def get_inconsistencies(self) -> Dict[str, Any]:
        """Identify inconsistencies between registry and index"""
        inconsistencies = {
            "timestamp": datetime.now().isoformat(),
            "registry_only": [],
            "index_only": [],
            "metadata_mismatches": []
        }
        
        try:
            # Load both data sources
            registry_components = {comp.code: comp for comp in self.registry.list_components()}
            index_data = self._load_index_data()
            index_components = {entry["code"]: entry for entry in index_data}
            
            # Find components only in registry
            registry_codes = set(registry_components.keys())
            index_codes = set(index_components.keys())
            
            inconsistencies["registry_only"] = list(registry_codes - index_codes)
            inconsistencies["index_only"] = list(index_codes - registry_codes)
            
            # Check metadata mismatches for common components
            common_codes = registry_codes & index_codes
            for code in common_codes:
                registry_comp = registry_components[code]
                index_comp = index_components[code]
                
                mismatches = []
                
                if registry_comp.stage != index_comp.get("stage", ""):
                    mismatches.append(f"stage: registry='{registry_comp.stage}' vs index='{index_comp.get('stage')}'")
                
                if registry_comp.alias_path != index_comp.get("alias_path", ""):
                    mismatches.append(f"alias_path: registry='{registry_comp.alias_path}' vs index='{index_comp.get('alias_path')}'")
                
                if registry_comp.original_path != index_comp.get("original_path", ""):
                    mismatches.append(f"original_path: registry='{registry_comp.original_path}' vs index='{index_comp.get('original_path')}'")
                
                if mismatches:
                    inconsistencies["metadata_mismatches"].append({
                        "code": code,
                        "mismatches": mismatches
                    })
            
        except Exception as e:
            logger.error(f"Failed to detect inconsistencies: {e}")
            inconsistencies["error"] = str(e)
        
        return inconsistencies
    
    def close(self):
        """Close integration layer resources"""
        if self.registry:
            self.registry.close()
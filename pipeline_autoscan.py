#!/usr/bin/env python3
"""
Autoscan reconciliation system for pipeline components.
Compares index.json against filesystem reality and maintains synchronization.
"""

import json
import os
import sys
import hashlib
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import subprocess
from datetime import datetime, timezone


@dataclass
class ComponentChange:
    """Represents a detected change in a pipeline component."""
    component_name: str
    change_type: str  # 'added', 'removed', 'modified', 'moved'
    old_path: Optional[str] = None
    new_path: Optional[str] = None
    old_hash: Optional[str] = None
    new_hash: Optional[str] = None
    metadata: Dict[str, Any] = None


@dataclass 
class ScanResult:
    """Result of filesystem scan operation."""
    components_found: int
    changes_detected: List[ComponentChange]
    orphaned_files: List[str]
    missing_files: List[str]
    scan_timestamp: str


class PipelineAutoscan:
    """Autoscan reconciliation system for pipeline components."""
    
    def __init__(self, index_path: str = "pipeline_index.json", 
                 canonical_root: str = "canonical_flow"):
        self.index_path = Path(index_path)
        self.canonical_root = Path(canonical_root)
        self.repo_root = Path.cwd()
        
        # Phase mappings for auto-detection
        self.phase_prefixes = {
            'I_': 'ingestion_preparation',
            'X_': 'context_construction', 
            'K_': 'knowledge_extraction',
            'A_': 'analysis_nlp',
            'L_': 'classification_evaluation',
            'O_': 'orchestration_control',
            'R_': 'search_retrieval',
            'S_': 'synthesis_output',
            'G_': 'aggregation_reporting', 
            'T_': 'integration_storage'
        }
    
    def load_index(self) -> Dict[str, Any]:
        """Load the current pipeline index."""
        if not self.index_path.exists():
            return {
                "version": "1.0.0",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "metadata": {
                    "description": "Canonical pipeline component specification index",
                    "total_components": 0,
                    "phases": list(self.phase_prefixes.values()),
                    "maintainer": "automated_index_system"
                },
                "components": []
            }
        
        with open(self.index_path, 'r') as f:
            return json.load(f)
    
    def save_index(self, index_data: Dict[str, Any]) -> None:
        """Save the updated pipeline index."""
        index_data["generated_at"] = datetime.now(timezone.utc).isoformat()
        index_data["metadata"]["total_components"] = len(index_data["components"])
        
        with open(self.index_path, 'w') as f:
            json.dump(index_data, f, indent=2, sort_keys=False)
    
    def get_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file contents."""
        if not file_path.exists() or not file_path.is_file():
            return ""
        
        hasher = hashlib.sha256()
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception:
            return ""
    
    def scan_filesystem_components(self) -> Dict[str, Dict[str, Any]]:
        """Scan filesystem for pipeline components."""
        components = {}
        
        # Scan canonical flow directory structure
        if self.canonical_root.exists():
            for phase_dir in self.canonical_root.iterdir():
                if not phase_dir.is_dir() or phase_dir.name.startswith('.'):
                    continue
                
                # Determine phase from directory name
                phase = self._detect_phase_from_dirname(phase_dir.name)
                if not phase:
                    continue
                
                # Scan Python files in phase directory
                for py_file in phase_dir.glob("*.py"):
                    if py_file.name == "__init__.py":
                        continue
                    
                    component_name = py_file.stem
                    file_hash = self.get_file_hash(py_file)
                    
                    # Extract component code from filename if present
                    code = self._extract_component_code(py_file.name)
                    
                    try:
                        rel_path = str(py_file.relative_to(self.repo_root))
                    except ValueError:
                        # Handle cases where file is outside repo root
                        rel_path = str(py_file)
                    
                    components[component_name] = {
                        "name": component_name,
                        "code": code,
                        "phase": phase,
                        "canonical_path": rel_path,
                        "file_hash": file_hash,
                        "enabled": True,
                        "last_modified": py_file.stat().st_mtime
                    }
        
        # Scan root level Python files that might be components
        for py_file in self.repo_root.glob("*.py"):
            if py_file.name.startswith(("test_", "__", "setup")):
                continue
            
            component_name = py_file.stem
            if component_name not in components:
                file_hash = self.get_file_hash(py_file)
                
                try:
                    rel_path = str(py_file.relative_to(self.repo_root))
                except ValueError:
                    # Handle cases where file is outside repo root
                    rel_path = str(py_file)
                
                components[component_name] = {
                    "name": component_name,
                    "code": None,
                    "phase": "unclassified",
                    "canonical_path": rel_path,
                    "original_path": rel_path,
                    "file_hash": file_hash,
                    "enabled": False,  # Unclassified components disabled by default
                    "last_modified": py_file.stat().st_mtime,
                    "description": "Unclassified component - needs manual review"
                }
        
        return components
    
    def _detect_phase_from_dirname(self, dirname: str) -> Optional[str]:
        """Detect phase from directory name."""
        for prefix, phase in self.phase_prefixes.items():
            if dirname.startswith(prefix):
                return phase
        return None
    
    def _extract_component_code(self, filename: str) -> Optional[str]:
        """Extract component code from filename (e.g., 01I from 01I_component.py)."""
        parts = filename.split('_', 1)
        if len(parts) > 1 and len(parts[0]) >= 2:
            code_part = parts[0]
            # Check if it matches pattern like 01I, 02X, etc.
            if (len(code_part) >= 3 and 
                code_part[:-1].isdigit() and 
                code_part[-1].isalpha()):
                return code_part
        return None
    
    def detect_changes(self, index_data: Dict[str, Any], 
                      filesystem_components: Dict[str, Dict[str, Any]]) -> List[ComponentChange]:
        """Detect changes between index and filesystem."""
        changes = []
        
        # Create lookup maps
        index_components = {comp["name"]: comp for comp in index_data.get("components", [])}
        index_paths = {comp["canonical_path"]: comp for comp in index_data.get("components", [])}
        
        # Check for new components
        for name, fs_comp in filesystem_components.items():
            if name not in index_components:
                changes.append(ComponentChange(
                    component_name=name,
                    change_type="added",
                    new_path=fs_comp["canonical_path"],
                    new_hash=fs_comp["file_hash"],
                    metadata=fs_comp
                ))
        
        # Check for removed/modified components
        for name, index_comp in index_components.items():
            if name not in filesystem_components:
                changes.append(ComponentChange(
                    component_name=name,
                    change_type="removed",
                    old_path=index_comp["canonical_path"],
                    old_hash=index_comp.get("file_hash", "")
                ))
            else:
                fs_comp = filesystem_components[name]
                
                # Check for modifications
                old_hash = index_comp.get("file_hash", "")
                new_hash = fs_comp["file_hash"]
                
                if old_hash != new_hash:
                    changes.append(ComponentChange(
                        component_name=name,
                        change_type="modified",
                        old_path=index_comp["canonical_path"],
                        new_path=fs_comp["canonical_path"],
                        old_hash=old_hash,
                        new_hash=new_hash
                    ))
                
                # Check for moves
                if index_comp["canonical_path"] != fs_comp["canonical_path"]:
                    changes.append(ComponentChange(
                        component_name=name,
                        change_type="moved",
                        old_path=index_comp["canonical_path"],
                        new_path=fs_comp["canonical_path"]
                    ))
        
        return changes
    
    def update_index_with_changes(self, index_data: Dict[str, Any],
                                 filesystem_components: Dict[str, Dict[str, Any]],
                                 changes: List[ComponentChange]) -> Dict[str, Any]:
        """Update index data with detected changes."""
        # Create new components list based on filesystem scan
        updated_components = []
        
        for name, fs_comp in filesystem_components.items():
            # Find existing component in index
            existing = None
            for comp in index_data.get("components", []):
                if comp["name"] == name:
                    existing = comp
                    break
            
            if existing:
                # Update existing component with filesystem data
                updated_comp = existing.copy()
                updated_comp.update({
                    "canonical_path": fs_comp["canonical_path"],
                    "file_hash": fs_comp["file_hash"],
                    "last_modified": fs_comp["last_modified"]
                })
                
                # Update phase if it changed
                if fs_comp["phase"] != "unclassified":
                    updated_comp["phase"] = fs_comp["phase"]
                
                # Update code if detected
                if fs_comp["code"]:
                    updated_comp["code"] = fs_comp["code"]
                
            else:
                # New component
                updated_comp = {
                    "name": name,
                    "code": fs_comp["code"],
                    "phase": fs_comp["phase"],
                    "dependencies": [],
                    "canonical_path": fs_comp["canonical_path"],
                    "original_path": fs_comp.get("original_path", fs_comp["canonical_path"]),
                    "description": fs_comp.get("description", f"Auto-detected {fs_comp['phase']} component"),
                    "enabled": fs_comp["enabled"],
                    "file_hash": fs_comp["file_hash"],
                    "last_modified": fs_comp["last_modified"]
                }
            
            updated_components.append(updated_comp)
        
        # Sort components by phase order and then by code
        phase_order = list(self.phase_prefixes.values()) + ["unclassified"]
        
        def sort_key(comp):
            phase_idx = phase_order.index(comp.get("phase", "unclassified"))
            code = comp.get("code", "ZZZ")  # Put components without codes at end
            return (phase_idx, code)
        
        updated_components.sort(key=sort_key)
        
        # Update index data
        index_data["components"] = updated_components
        index_data["metadata"]["total_components"] = len(updated_components)
        
        return index_data
    
    def run_autoscan(self, update_index: bool = True) -> ScanResult:
        """Run complete autoscan process."""
        print("🔍 Starting pipeline component autoscan...")
        
        # Load current index
        index_data = self.load_index()
        print(f"📋 Loaded index with {len(index_data.get('components', []))} components")
        
        # Scan filesystem
        print("🗂️  Scanning filesystem for components...")
        filesystem_components = self.scan_filesystem_components()
        print(f"📁 Found {len(filesystem_components)} filesystem components")
        
        # Detect changes
        changes = self.detect_changes(index_data, filesystem_components)
        print(f"🔄 Detected {len(changes)} changes")
        
        # Report changes
        for change in changes:
            emoji = {"added": "✅", "removed": "❌", "modified": "📝", "moved": "📦"}[change.change_type]
            print(f"  {emoji} {change.change_type.upper()}: {change.component_name}")
        
        # Update index if requested
        if update_index and changes:
            print("💾 Updating index...")
            updated_index = self.update_index_with_changes(index_data, filesystem_components, changes)
            self.save_index(updated_index)
            print("✅ Index updated successfully")
        
        # Find orphaned and missing files
        orphaned_files = []
        missing_files = []
        
        index_paths = {comp["canonical_path"] for comp in index_data.get("components", [])}
        fs_paths = {comp["canonical_path"] for comp in filesystem_components.values()}
        
        orphaned_files = list(fs_paths - index_paths)
        missing_files = list(index_paths - fs_paths)
        
        return ScanResult(
            components_found=len(filesystem_components),
            changes_detected=changes,
            orphaned_files=orphaned_files,
            missing_files=missing_files,
            scan_timestamp=datetime.now(timezone.utc).isoformat()
        )
    
    def validate_index_integrity(self) -> Tuple[bool, List[str]]:
        """Validate index integrity against filesystem."""
        errors = []
        index_data = self.load_index()
        
        for component in index_data.get("components", []):
            canonical_path = Path(component["canonical_path"])
            
            # Check if file exists
            if not canonical_path.exists():
                errors.append(f"Missing file: {canonical_path}")
                continue
            
            # Validate hash if present
            stored_hash = component.get("file_hash")
            if stored_hash:
                actual_hash = self.get_file_hash(canonical_path)
                if stored_hash != actual_hash:
                    errors.append(f"Hash mismatch: {canonical_path}")
            
            # Validate dependencies exist
            for dep_code in component.get("dependencies", []):
                dep_exists = any(
                    c.get("code") == dep_code 
                    for c in index_data["components"]
                )
                if not dep_exists:
                    errors.append(f"Missing dependency {dep_code} for {component['name']}")
        
        return len(errors) == 0, errors


def main():
    """Main CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Pipeline component autoscan system")
    parser.add_argument("--index", default="pipeline_index.json", help="Index file path")
    parser.add_argument("--canonical-root", default="canonical_flow", help="Canonical flow root directory")
    parser.add_argument("--no-update", action="store_true", help="Don't update index, just report changes")
    parser.add_argument("--validate", action="store_true", help="Validate index integrity")
    parser.add_argument("--output", help="Output scan results to JSON file")
    
    args = parser.parse_args()
    
    scanner = PipelineAutoscan(args.index, args.canonical_root)
    
    if args.validate:
        print("🔍 Validating index integrity...")
        is_valid, errors = scanner.validate_index_integrity()
        
        if is_valid:
            print("✅ Index integrity validation passed")
            return 0
        else:
            print("❌ Index integrity validation failed:")
            for error in errors:
                print(f"  • {error}")
            return 1
    
    # Run autoscan
    result = scanner.run_autoscan(update_index=not args.no_update)
    
    # Output results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(asdict(result), f, indent=2, default=str)
    
    # Exit with status code based on changes detected
    return 1 if result.changes_detected else 0


if __name__ == "__main__":
    sys.exit(main())
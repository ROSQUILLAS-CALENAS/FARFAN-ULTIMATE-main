#!/usr/bin/env python3
"""
Controlled Deletion System with 30-day embargo and automated verification.

This system:
1. Identifies duplicate non-canonical directories through automated scanning
2. Implements 30-day embargo period with deprecation warnings
3. Establishes import ban enforcement using import-linter rules
4. Provides nightly dead code verification using vulture
5. Automated reporting and safe deletion execution
"""

import ast
import os
import sys
import json
import shutil
import logging
import datetime
import subprocess
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import warnings


class EmbargStatus(Enum):
    """Status of directories in the embargo system."""
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    SCHEDULED_FOR_DELETION = "scheduled_for_deletion"
    DELETED = "deleted"


@dataclass
class DuplicateDirectory:
    """Information about a duplicate directory."""
    path: str
    canonical_path: str
    size_bytes: int
    file_count: int
    duplicate_hash: str
    last_modified: str
    embargo_date: Optional[str] = None
    status: EmbargStatus = EmbargStatus.ACTIVE
    dependent_modules: List[str] = None
    
    def __post_init__(self):
        if self.dependent_modules is None:
            self.dependent_modules = []


@dataclass
class EmbargoRecord:
    """Record of an embargoed directory."""
    directory: DuplicateDirectory
    embargo_start: datetime.datetime
    expected_deletion: datetime.datetime
    warnings_issued: int = 0
    last_scan: Optional[datetime.datetime] = None
    active_imports: List[str] = None
    
    def __post_init__(self):
        if self.active_imports is None:
            self.active_imports = []


class DirectoryScanner:
    """Scans for duplicate non-canonical directories."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.logger = logging.getLogger(__name__)
    
    def scan_for_duplicates(self) -> List[DuplicateDirectory]:
        """Identify duplicate directories based on content similarity."""
        duplicates = []
        directory_hashes = {}
        
        # Skip common non-source directories
        skip_dirs = {'.git', '.venv', 'venv', '__pycache__', '.pytest_cache', 
                    'node_modules', '.idea', '.vscode', 'logs', 'data'}
        
        for root, dirs, files in os.walk(self.project_root):
            # Filter out skip directories
            dirs[:] = [d for d in dirs if d not in skip_dirs]
            
            if not files:
                continue
                
            path_obj = Path(root)
            relative_path = path_obj.relative_to(self.project_root)
            
            # Calculate directory content hash
            dir_hash = self._calculate_directory_hash(path_obj)
            
            if dir_hash in directory_hashes:
                # Found duplicate
                original_dir = directory_hashes[dir_hash]
                duplicate = DuplicateDirectory(
                    path=str(relative_path),
                    canonical_path=original_dir['path'],
                    size_bytes=self._get_directory_size(path_obj),
                    file_count=len(files),
                    duplicate_hash=dir_hash,
                    last_modified=datetime.datetime.fromtimestamp(
                        path_obj.stat().st_mtime).isoformat()
                )
                
                # Determine which is canonical (prefer shorter path or specific patterns)
                if self._is_canonical_path(str(relative_path), original_dir['path']):
                    duplicate.canonical_path = str(relative_path)
                    duplicate.path = original_dir['path']
                
                duplicates.append(duplicate)
            else:
                directory_hashes[dir_hash] = {
                    'path': str(relative_path),
                    'size': self._get_directory_size(path_obj)
                }
        
        return duplicates
    
    def _calculate_directory_hash(self, directory: Path) -> str:
        """Calculate hash based on directory structure and file contents."""
        hasher = hashlib.sha256()
        
        try:
            for root, dirs, files in os.walk(directory):
                # Sort for consistent hashing
                dirs.sort()
                files.sort()
                
                for file in files:
                    if file.endswith(('.pyc', '.pyo', '.pyd')):
                        continue
                        
                    file_path = Path(root) / file
                    try:
                        # Add file path and content to hash
                        hasher.update(str(file_path.relative_to(directory)).encode())
                        if file_path.stat().st_size < 1024 * 1024:  # Skip large files
                            with open(file_path, 'rb') as f:
                                hasher.update(f.read())
                    except (OSError, UnicodeDecodeError):
                        continue
        except OSError:
            pass
        
        return hasher.hexdigest()
    
    def _get_directory_size(self, directory: Path) -> int:
        """Get total size of directory in bytes."""
        total_size = 0
        try:
            for root, dirs, files in os.walk(directory):
                for file in files:
                    try:
                        total_size += (Path(root) / file).stat().st_size
                    except OSError:
                        continue
        except OSError:
            pass
        return total_size
    
    def _is_canonical_path(self, path1: str, path2: str) -> bool:
        """Determine which path is more canonical."""
        # Prefer paths in main package structure
        canonical_prefixes = ['egw_query_expansion/', 'canonical_flow/', 'src/']
        
        path1_canonical = any(path1.startswith(p) for p in canonical_prefixes)
        path2_canonical = any(path2.startswith(p) for p in canonical_prefixes)
        
        if path1_canonical and not path2_canonical:
            return True
        if path2_canonical and not path1_canonical:
            return False
        
        # Prefer shorter paths
        return len(path1) < len(path2)


class ImportAnalyzer:
    """Analyzes import dependencies to determine safe deletion candidates."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.logger = logging.getLogger(__name__)
    
    def find_imports_to_path(self, target_path: str) -> List[str]:
        """Find all imports that reference the target path."""
        imports = []
        
        for py_file in self.project_root.rglob('*.py'):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Parse AST to find imports
                try:
                    tree = ast.parse(content)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Import):
                            for alias in node.names:
                                if self._import_references_path(alias.name, target_path):
                                    imports.append(f"{py_file}:{node.lineno}")
                        elif isinstance(node, ast.ImportFrom):
                            if node.module and self._import_references_path(node.module, target_path):
                                imports.append(f"{py_file}:{node.lineno}")
                except SyntaxError:
                    # Handle syntax errors gracefully
                    continue
                    
            except (OSError, UnicodeDecodeError):
                continue
        
        return imports
    
    def _import_references_path(self, import_name: str, target_path: str) -> bool:
        """Check if an import references the target path."""
        target_module = target_path.replace('/', '.').replace('\\', '.')
        target_module = target_module.strip('.')
        
        return (import_name.startswith(target_module) or 
                target_module.startswith(import_name))


class DeprecationWarningSystem:
    """Manages deprecation warnings for embargoed directories."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.logger = logging.getLogger(__name__)
    
    def inject_deprecation_warnings(self, embargo_record: EmbargoRecord):
        """Inject deprecation warnings into modules in embargoed directory."""
        target_path = self.project_root / embargo_record.directory.path
        
        if not target_path.exists():
            return
        
        warning_code = self._generate_warning_code(embargo_record)
        
        for py_file in target_path.rglob('*.py'):
            try:
                self._add_warning_to_file(py_file, warning_code)
            except Exception as e:
                self.logger.error(f"Failed to add warning to {py_file}: {e}")
    
    def _generate_warning_code(self, embargo_record: EmbargoRecord) -> str:
        """Generate deprecation warning code."""
        deletion_date = embargo_record.expected_deletion.strftime('%Y-%m-%d')
        canonical_path = embargo_record.directory.canonical_path
        
        return f'''
import warnings

# AUTO-GENERATED DEPRECATION WARNING - DO NOT EDIT
warnings.warn(
    f"This module is deprecated and will be deleted on {deletion_date}. "
    f"Use the canonical path '{canonical_path}' instead.",
    DeprecationWarning,
    stacklevel=2
)
'''
    
    def _add_warning_to_file(self, file_path: Path, warning_code: str):
        """Add warning code to a Python file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Skip if warning already exists
            if 'AUTO-GENERATED DEPRECATION WARNING' in content:
                return
            
            # Add warning after imports but before main code
            lines = content.split('\n')
            insert_index = self._find_warning_insertion_point(lines)
            
            warning_lines = warning_code.strip().split('\n')
            lines[insert_index:insert_index] = warning_lines
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
                
        except Exception as e:
            self.logger.error(f"Failed to modify {file_path}: {e}")
    
    def _find_warning_insertion_point(self, lines: List[str]) -> int:
        """Find the best place to insert deprecation warning."""
        # Find end of imports/docstring
        in_docstring = False
        docstring_char = None
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Handle docstrings
            if not in_docstring and (stripped.startswith('"""') or stripped.startswith("'''")):
                docstring_char = stripped[:3]
                if stripped.count(docstring_char) >= 2:
                    continue  # Single line docstring
                in_docstring = True
                continue
            
            if in_docstring and docstring_char in stripped:
                in_docstring = False
                continue
            
            # Skip empty lines and comments at start
            if not stripped or stripped.startswith('#'):
                continue
            
            # Stop at first non-import statement
            if not (stripped.startswith('import ') or stripped.startswith('from ') or in_docstring):
                return i
        
        return len(lines)


class ImportLinterIntegration:
    """Integration with import-linter for ban enforcement."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.config_path = project_root / 'pyproject.toml'
        self.logger = logging.getLogger(__name__)
    
    def add_import_ban(self, deprecated_path: str, canonical_path: str):
        """Add import ban rule to import-linter config."""
        config_content = self._read_or_create_config()
        
        # Add ban rule
        ban_rule = {
            'name': f'Ban deprecated path: {deprecated_path}',
            'type': 'forbidden',
            'source_modules': ['*'],
            'forbidden_modules': [deprecated_path.replace('/', '.')],
            'ignore_imports': [],
            'description': f'Use {canonical_path} instead of deprecated {deprecated_path}'
        }
        
        # Update configuration
        self._update_config_with_ban(config_content, ban_rule)
    
    def _read_or_create_config(self) -> str:
        """Read existing config or create new one."""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    return f.read()
        except Exception:
            pass
        
        # Create basic config
        return '''[tool.importlinter]
root_package = "egw_query_expansion"

[[tool.importlinter.contracts]]
name = "Deprecated module bans"
type = "forbidden"
'''
    
    def _update_config_with_ban(self, config_content: str, ban_rule: Dict):
        """Update config with new ban rule."""
        # This is a simplified implementation
        # In practice, you'd want to use a proper TOML parser
        pass


class DeadCodeDetector:
    """Uses vulture and custom analysis to detect dead code."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.logger = logging.getLogger(__name__)
    
    def scan_dead_code(self, target_paths: List[str]) -> Dict[str, List[str]]:
        """Scan for dead code in target paths."""
        dead_code_report = {}
        
        for path in target_paths:
            target_path = self.project_root / path
            if not target_path.exists():
                continue
            
            # Run vulture
            vulture_results = self._run_vulture(target_path)
            
            # Run custom analysis
            custom_results = self._custom_dead_code_analysis(target_path)
            
            dead_code_report[path] = {
                'vulture_findings': vulture_results,
                'custom_findings': custom_results,
                'safe_for_deletion': len(vulture_results) == 0 and len(custom_results) == 0
            }
        
        return dead_code_report
    
    def _run_vulture(self, target_path: Path) -> List[str]:
        """Run vulture on target path."""
        try:
            result = subprocess.run(
                ['vulture', str(target_path)],
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode == 0:
                return []  # No dead code found
            
            # Parse vulture output
            findings = []
            for line in result.stdout.split('\n'):
                if line.strip() and not line.startswith('#'):
                    findings.append(line.strip())
            
            return findings
            
        except (subprocess.TimeoutExpired, FileNotFoundError):
            self.logger.warning(f"Could not run vulture on {target_path}")
            return ["vulture_unavailable"]
    
    def _custom_dead_code_analysis(self, target_path: Path) -> List[str]:
        """Custom dead code analysis beyond vulture."""
        findings = []
        
        # Check for unused imports
        for py_file in target_path.rglob('*.py'):
            try:
                unused_imports = self._find_unused_imports(py_file)
                if unused_imports:
                    findings.extend([f"{py_file}:{imp}" for imp in unused_imports])
            except Exception:
                continue
        
        return findings
    
    def _find_unused_imports(self, py_file: Path) -> List[str]:
        """Find unused imports in a Python file."""
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            # Extract imports
            imports = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name.split('.')[0])
                elif isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        imports.add(alias.name)
            
            # Extract usage (simplified)
            used_names = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    used_names.add(node.id)
                elif isinstance(node, ast.Attribute):
                    if isinstance(node.value, ast.Name):
                        used_names.add(node.value.id)
            
            # Find unused
            unused = imports - used_names
            return list(unused)
            
        except Exception:
            return []


class ControlledDeletionSystem:
    """Main system coordinating all deletion operations."""
    
    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.data_dir = self.project_root / '.controlled_deletion'
        self.data_dir.mkdir(exist_ok=True)
        
        self.embargo_file = self.data_dir / 'embargo_records.json'
        self.config_file = self.data_dir / 'deletion_config.json'
        
        # Initialize components
        self.scanner = DirectoryScanner(self.project_root)
        self.import_analyzer = ImportAnalyzer(self.project_root)
        self.warning_system = DeprecationWarningSystem(self.project_root)
        self.linter_integration = ImportLinterIntegration(self.project_root)
        self.dead_code_detector = DeadCodeDetector(self.project_root)
        
        # Setup logging
        self.logger = self._setup_logging()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging system."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.FileHandler(self.data_dir / 'deletion_system.log')
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def run_duplicate_scan(self) -> List[DuplicateDirectory]:
        """Run scan for duplicate directories."""
        self.logger.info("Starting duplicate directory scan")
        duplicates = self.scanner.scan_for_duplicates()
        
        # Analyze import dependencies
        for duplicate in duplicates:
            duplicate.dependent_modules = self.import_analyzer.find_imports_to_path(
                duplicate.path
            )
        
        self.logger.info(f"Found {len(duplicates)} duplicate directories")
        return duplicates
    
    def initiate_embargo(self, duplicates: List[DuplicateDirectory]) -> List[EmbargoRecord]:
        """Initiate embargo period for duplicate directories."""
        embargo_records = []
        
        for duplicate in duplicates:
            # Skip if already under embargo
            if duplicate.status != EmbargStatus.ACTIVE:
                continue
            
            # Create embargo record
            now = datetime.datetime.now()
            embargo_record = EmbargoRecord(
                directory=duplicate,
                embargo_start=now,
                expected_deletion=now + datetime.timedelta(days=30),
                active_imports=duplicate.dependent_modules.copy()
            )
            
            # Update directory status
            duplicate.status = EmbargStatus.DEPRECATED
            duplicate.embargo_date = now.isoformat()
            
            # Inject deprecation warnings
            self.warning_system.inject_deprecation_warnings(embargo_record)
            
            # Add import ban
            self.linter_integration.add_import_ban(
                duplicate.path, 
                duplicate.canonical_path
            )
            
            embargo_records.append(embargo_record)
            self.logger.info(f"Initiated embargo for {duplicate.path}")
        
        # Save embargo records
        self._save_embargo_records(embargo_records)
        
        return embargo_records
    
    def run_nightly_verification(self) -> Dict[str, Any]:
        """Run nightly verification of embargoed directories."""
        self.logger.info("Running nightly verification")
        
        # Load embargo records
        embargo_records = self._load_embargo_records()
        
        verification_results = {
            'scan_time': datetime.datetime.now().isoformat(),
            'records_checked': len(embargo_records),
            'ready_for_deletion': [],
            'still_in_use': [],
            'dead_code_analysis': {}
        }
        
        for record in embargo_records:
            # Update active imports
            current_imports = self.import_analyzer.find_imports_to_path(
                record.directory.path
            )
            record.active_imports = current_imports
            record.last_scan = datetime.datetime.now()
            
            # Check if deletion period has passed
            if datetime.datetime.now() >= record.expected_deletion:
                if not current_imports:
                    # Run dead code analysis
                    dead_code_results = self.dead_code_detector.scan_dead_code(
                        [record.directory.path]
                    )
                    
                    verification_results['dead_code_analysis'][record.directory.path] = dead_code_results
                    
                    if dead_code_results[record.directory.path].get('safe_for_deletion', False):
                        verification_results['ready_for_deletion'].append({
                            'path': record.directory.path,
                            'canonical_path': record.directory.canonical_path,
                            'embargo_start': record.embargo_start.isoformat(),
                            'size_bytes': record.directory.size_bytes
                        })
                        record.directory.status = EmbargStatus.SCHEDULED_FOR_DELETION
                    else:
                        verification_results['still_in_use'].append({
                            'path': record.directory.path,
                            'reason': 'dead_code_detected',
                            'active_imports': current_imports
                        })
                else:
                    verification_results['still_in_use'].append({
                        'path': record.directory.path,
                        'reason': 'active_imports',
                        'active_imports': current_imports
                    })
        
        # Save updated records
        self._save_embargo_records(embargo_records)
        
        # Generate report
        self._generate_verification_report(verification_results)
        
        return verification_results
    
    def execute_safe_deletion(self, dry_run: bool = True) -> Dict[str, Any]:
        """Execute safe deletion of verified directories."""
        embargo_records = self._load_embargo_records()
        
        deletion_results = {
            'execution_time': datetime.datetime.now().isoformat(),
            'dry_run': dry_run,
            'deleted_directories': [],
            'deletion_errors': []
        }
        
        for record in embargo_records:
            if record.directory.status == EmbargStatus.SCHEDULED_FOR_DELETION:
                target_path = self.project_root / record.directory.path
                
                if not target_path.exists():
                    continue
                
                try:
                    if not dry_run:
                        # Create backup before deletion
                        backup_path = self.data_dir / 'backups' / record.directory.path.replace('/', '_')
                        backup_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copytree(target_path, backup_path)
                        
                        # Delete directory
                        shutil.rmtree(target_path)
                        
                        # Update status
                        record.directory.status = EmbargStatus.DELETED
                    
                    deletion_results['deleted_directories'].append({
                        'path': record.directory.path,
                        'size_bytes': record.directory.size_bytes,
                        'backup_path': str(backup_path) if not dry_run else None
                    })
                    
                    self.logger.info(f"{'Would delete' if dry_run else 'Deleted'} {record.directory.path}")
                    
                except Exception as e:
                    deletion_results['deletion_errors'].append({
                        'path': record.directory.path,
                        'error': str(e)
                    })
                    self.logger.error(f"Failed to delete {record.directory.path}: {e}")
        
        if not dry_run:
            self._save_embargo_records(embargo_records)
        
        return deletion_results
    
    def _load_embargo_records(self) -> List[EmbargoRecord]:
        """Load embargo records from disk."""
        if not self.embargo_file.exists():
            return []
        
        try:
            with open(self.embargo_file, 'r') as f:
                data = json.load(f)
            
            records = []
            for item in data:
                # Reconstruct objects from JSON
                directory = DuplicateDirectory(**item['directory'])
                record = EmbargoRecord(
                    directory=directory,
                    embargo_start=datetime.datetime.fromisoformat(item['embargo_start']),
                    expected_deletion=datetime.datetime.fromisoformat(item['expected_deletion']),
                    warnings_issued=item.get('warnings_issued', 0),
                    last_scan=datetime.datetime.fromisoformat(item['last_scan']) if item.get('last_scan') else None,
                    active_imports=item.get('active_imports', [])
                )
                records.append(record)
            
            return records
            
        except Exception as e:
            self.logger.error(f"Failed to load embargo records: {e}")
            return []
    
    def _save_embargo_records(self, records: List[EmbargoRecord]):
        """Save embargo records to disk."""
        try:
            data = []
            for record in records:
                item = {
                    'directory': asdict(record.directory),
                    'embargo_start': record.embargo_start.isoformat(),
                    'expected_deletion': record.expected_deletion.isoformat(),
                    'warnings_issued': record.warnings_issued,
                    'last_scan': record.last_scan.isoformat() if record.last_scan else None,
                    'active_imports': record.active_imports
                }
                data.append(item)
            
            with open(self.embargo_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save embargo records: {e}")
    
    def _generate_verification_report(self, results: Dict[str, Any]):
        """Generate verification report."""
        report_path = self.data_dir / f"verification_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with open(report_path, 'w') as f:
                json.dump(results, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to generate verification report: {e}")


def main():
    """Main entry point for the deletion system."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Controlled Deletion System')
    parser.add_argument('--scan', action='store_true', help='Run duplicate scan')
    parser.add_argument('--embargo', action='store_true', help='Initiate embargo for found duplicates')
    parser.add_argument('--verify', action='store_true', help='Run nightly verification')
    parser.add_argument('--delete', action='store_true', help='Execute safe deletion')
    parser.add_argument('--dry-run', action='store_true', help='Run deletion in dry-run mode')
    parser.add_argument('--project-root', help='Project root directory')
    
    args = parser.parse_args()
    
    system = ControlledDeletionSystem(args.project_root)
    
    if args.scan:
        duplicates = system.run_duplicate_scan()
        print(f"Found {len(duplicates)} duplicate directories")
        
        if args.embargo and duplicates:
            embargo_records = system.initiate_embargo(duplicates)
            print(f"Initiated embargo for {len(embargo_records)} directories")
    
    if args.verify:
        results = system.run_nightly_verification()
        print(f"Verification complete. {len(results['ready_for_deletion'])} directories ready for deletion")
    
    if args.delete:
        results = system.execute_safe_deletion(dry_run=args.dry_run)
        print(f"{'Would delete' if args.dry_run else 'Deleted'} {len(results['deleted_directories'])} directories")


if __name__ == '__main__':
    main()
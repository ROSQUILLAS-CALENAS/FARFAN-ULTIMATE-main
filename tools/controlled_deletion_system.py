#!/usr/bin/env python3
"""
Controlled Deletion System for Non-Canonical Directories

This system implements a staged removal process for duplicate non-canonical directories
identified in the canonical flow audit. It provides:

1. Embargo mechanism with configurable grace periods (default 30 days)
2. Deprecation warnings for modules in marked directories
3. Import bans through static analysis
4. Nightly automated verification for dead code detection
5. Safe removal after embargo period expires
"""

import ast
import datetime
import json
import logging
import os
import subprocess
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
import hashlib
import re
import importlib
import importlib.util
from collections import defaultdict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class EmbargoRecord:
    """Record of a directory under embargo for deletion"""
    directory: str
    embargo_date: datetime.datetime
    grace_period_days: int
    reason: str
    status: str = "embargoed"  # embargoed, warning, ready_for_deletion, deleted
    external_references: List[str] = field(default_factory=list)
    last_scan: Optional[datetime.datetime] = None
    metadata: Dict = field(default_factory=dict)
    
    @property
    def expiry_date(self) -> datetime.datetime:
        """Calculate when the embargo expires"""
        return self.embargo_date + datetime.timedelta(days=self.grace_period_days)
    
    @property
    def is_expired(self) -> bool:
        """Check if embargo period has expired"""
        return datetime.datetime.now() >= self.expiry_date
    
    @property
    def days_remaining(self) -> int:
        """Days remaining in embargo period"""
        delta = self.expiry_date - datetime.datetime.now()
        return max(0, delta.days)


class DeprecationWarningInjector:
    """Injects deprecation warnings into Python modules"""
    
    def __init__(self):
        self.warning_template = '''
import warnings
warnings.warn(
    "This module is deprecated and scheduled for removal. "
    "It is part of a non-canonical directory structure that will be deleted on {expiry_date}. "
    "Please migrate to the canonical equivalent: {canonical_path}",
    DeprecationWarning,
    stacklevel=2
)
'''
    
    def inject_warning(self, file_path: Path, expiry_date: datetime.datetime, canonical_path: str):
        """Inject deprecation warning at the top of a Python file"""
        if not file_path.suffix == '.py':
            return False
            
        try:
            content = file_path.read_text(encoding='utf-8')
            
            # Check if warning already exists
            if 'This module is deprecated and scheduled for removal' in content:
                return True
                
            # Find insertion point (after shebang and encoding declarations)
            lines = content.split('\n')
            insert_index = 0
            
            for i, line in enumerate(lines):
                if line.startswith('#!') or line.startswith('# -*- coding:') or line.startswith('# coding:'):
                    insert_index = i + 1
                elif line.strip() == '' and i < 5:  # Allow a few empty lines at top
                    continue
                else:
                    break
                    
            # Insert warning
            warning_code = self.warning_template.format(
                expiry_date=expiry_date.strftime('%Y-%m-%d'),
                canonical_path=canonical_path
            )
            
            new_lines = (
                lines[:insert_index] + 
                [warning_code] + 
                lines[insert_index:]
            )
            
            file_path.write_text('\n'.join(new_lines), encoding='utf-8')
            logger.info(f"Injected deprecation warning in {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to inject warning in {file_path}: {e}")
            return False


class ImportAnalyzer(ast.NodeVisitor):
    """AST visitor to analyze imports in Python files"""
    
    def __init__(self):
        self.imports = set()
        self.from_imports = set()
        
    def visit_Import(self, node):
        for alias in node.names:
            self.imports.add(alias.name)
        self.generic_visit(node)
        
    def visit_ImportFrom(self, node):
        if node.module:
            for alias in node.names:
                full_name = f"{node.module}.{alias.name}"
                self.from_imports.add(full_name)
                self.from_imports.add(node.module)
        self.generic_visit(node)


class StaticAnalysisEngine:
    """Static analysis engine for detecting imports and dead code"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.embargoed_modules = set()
        
    def analyze_file_imports(self, file_path: Path) -> Tuple[Set[str], Set[str]]:
        """Analyze imports in a Python file"""
        try:
            if file_path.suffix != '.py':
                return set(), set()
                
            content = file_path.read_text(encoding='utf-8')
            tree = ast.parse(content, filename=str(file_path))
            
            analyzer = ImportAnalyzer()
            analyzer.visit(tree)
            
            return analyzer.imports, analyzer.from_imports
            
        except Exception as e:
            logger.warning(f"Failed to analyze {file_path}: {e}")
            return set(), set()
    
    def find_external_references(self, embargoed_dir: str) -> List[str]:
        """Find external references to modules in embargoed directory"""
        references = []
        embargoed_path = Path(embargoed_dir)
        
        # Get all Python modules in embargoed directory
        embargoed_modules = set()
        for py_file in embargoed_path.rglob('*.py'):
            relative_path = py_file.relative_to(self.project_root)
            module_name = str(relative_path).replace('/', '.').replace('\\', '.').replace('.py', '')
            embargoed_modules.add(module_name)
        
        # Scan all Python files in project for references
        for py_file in self.project_root.rglob('*.py'):
            # Skip files in embargoed directory
            try:
                py_file.relative_to(embargoed_path)
                continue  # This file is in embargoed directory
            except ValueError:
                pass  # This file is outside embargoed directory
            
            imports, from_imports = self.analyze_file_imports(py_file)
            
            # Check for references to embargoed modules
            for module in embargoed_modules:
                if (module in imports or 
                    any(module in imp for imp in from_imports) or
                    any(imp.startswith(module + '.') for imp in from_imports)):
                    references.append(f"{py_file}:{module}")
        
        return references
    
    def scan_with_vulture(self, directory: Path) -> Dict[str, List[str]]:
        """Use vulture to find dead code in directory"""
        try:
            cmd = [sys.executable, '-m', 'vulture', str(directory)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            dead_code = {
                'unused_functions': [],
                'unused_classes': [],
                'unused_variables': [],
                'unused_imports': []
            }
            
            for line in result.stdout.split('\n'):
                if 'unused function' in line:
                    dead_code['unused_functions'].append(line.strip())
                elif 'unused class' in line:
                    dead_code['unused_classes'].append(line.strip())
                elif 'unused variable' in line:
                    dead_code['unused_variables'].append(line.strip())
                elif 'unused import' in line:
                    dead_code['unused_imports'].append(line.strip())
            
            return dead_code
            
        except Exception as e:
            logger.warning(f"Vulture scan failed for {directory}: {e}")
            return {}


class ControlledDeletionManager:
    """Main manager for the controlled deletion system"""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.project_root = Path.cwd()
        if self.project_root.name == 'tools':
            self.project_root = self.project_root.parent
        self.config_path = config_path or self.project_root / 'tools' / 'deletion_config.json'
        self.embargo_db_path = self.project_root / 'tools' / 'embargo_registry.json'
        self.reports_dir = self.project_root / 'tools' / 'deletion_reports'
        
        self.config = self._load_config()
        self.embargo_registry: Dict[str, EmbargoRecord] = self._load_embargo_registry()
        
        self.warning_injector = DeprecationWarningInjector()
        self.static_analyzer = StaticAnalysisEngine(self.project_root)
        
        # Create reports directory
        self.reports_dir.mkdir(exist_ok=True)
    
    def _load_config(self) -> Dict:
        """Load configuration from file"""
        default_config = {
            "default_grace_period_days": 30,
            "enable_vulture": True,
            "enable_import_linter": True,
            "canonical_mapping": {
                # Maps non-canonical paths to canonical equivalents
            },
            "excluded_patterns": [
                "venv/*",
                "__pycache__/*",
                "*.pyc",
                ".git/*"
            ],
            "notification_emails": [],
            "ci_integration": {
                "fail_on_deprecated_imports": True,
                "generate_reports": True
            }
        }
        
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    loaded_config = json.load(f)
                    default_config.update(loaded_config)
            except Exception as e:
                logger.error(f"Failed to load config: {e}")
        
        return default_config
    
    def _save_config(self):
        """Save configuration to file"""
        try:
            self.config_path.parent.mkdir(exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
    
    def _load_embargo_registry(self) -> Dict[str, EmbargoRecord]:
        """Load embargo registry from file"""
        if not self.embargo_db_path.exists():
            return {}
        
        try:
            with open(self.embargo_db_path, 'r') as f:
                data = json.load(f)
            
            registry = {}
            for path, record_data in data.items():
                # Convert datetime strings back to datetime objects
                record_data['embargo_date'] = datetime.datetime.fromisoformat(record_data['embargo_date'])
                if record_data['last_scan']:
                    record_data['last_scan'] = datetime.datetime.fromisoformat(record_data['last_scan'])
                
                registry[path] = EmbargoRecord(**record_data)
            
            return registry
            
        except Exception as e:
            logger.error(f"Failed to load embargo registry: {e}")
            return {}
    
    def _save_embargo_registry(self):
        """Save embargo registry to file"""
        try:
            self.embargo_db_path.parent.mkdir(exist_ok=True)
            
            # Convert to serializable format
            data = {}
            for path, record in self.embargo_registry.items():
                record_data = {
                    'directory': record.directory,
                    'embargo_date': record.embargo_date.isoformat(),
                    'grace_period_days': record.grace_period_days,
                    'reason': record.reason,
                    'status': record.status,
                    'external_references': record.external_references,
                    'last_scan': record.last_scan.isoformat() if record.last_scan else None,
                    'metadata': record.metadata
                }
                data[path] = record_data
            
            with open(self.embargo_db_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save embargo registry: {e}")
    
    def embargo_directory(self, directory: str, reason: str, grace_period_days: Optional[int] = None) -> bool:
        """Place a directory under embargo for deletion"""
        if not Path(directory).exists():
            logger.error(f"Directory {directory} does not exist")
            return False
        
        grace_period = grace_period_days or self.config['default_grace_period_days']
        
        record = EmbargoRecord(
            directory=directory,
            embargo_date=datetime.datetime.now(),
            grace_period_days=grace_period,
            reason=reason,
            status="embargoed"
        )
        
        self.embargo_registry[directory] = record
        self._save_embargo_registry()
        
        # Inject deprecation warnings
        self._inject_warnings(directory, record)
        
        # Generate initial report
        self._generate_embargo_report(record)
        
        logger.info(f"Directory {directory} placed under embargo until {record.expiry_date}")
        return True
    
    def _inject_warnings(self, directory: str, record: EmbargoRecord):
        """Inject deprecation warnings into all Python files in directory"""
        dir_path = Path(directory)
        canonical_path = self.config['canonical_mapping'].get(directory, "canonical_flow/")
        
        for py_file in dir_path.rglob('*.py'):
            if py_file.name != '__init__.py':  # Skip __init__.py files
                self.warning_injector.inject_warning(
                    py_file, 
                    record.expiry_date,
                    canonical_path
                )
    
    def scan_embargoed_directories(self) -> Dict[str, Dict]:
        """Scan all embargoed directories for external references and dead code"""
        scan_results = {}
        
        for directory, record in self.embargo_registry.items():
            if record.status in ['embargoed', 'warning']:
                logger.info(f"Scanning {directory}")
                
                # Find external references
                external_refs = self.static_analyzer.find_external_references(directory)
                record.external_references = external_refs
                record.last_scan = datetime.datetime.now()
                
                # Scan for dead code with vulture
                dead_code = {}
                if self.config.get('enable_vulture', True):
                    dead_code = self.static_analyzer.scan_with_vulture(Path(directory))
                
                # Update status based on scan results
                if not external_refs and record.is_expired:
                    record.status = "ready_for_deletion"
                elif external_refs:
                    record.status = "warning"
                
                scan_results[directory] = {
                    'external_references': external_refs,
                    'dead_code': dead_code,
                    'status': record.status,
                    'days_remaining': record.days_remaining
                }
        
        self._save_embargo_registry()
        return scan_results
    
    def _generate_embargo_report(self, record: EmbargoRecord):
        """Generate detailed report for an embargoed directory"""
        report_path = self.reports_dir / f"{record.directory.replace('/', '_')}_embargo_report.json"
        
        report = {
            'directory': record.directory,
            'embargo_date': record.embargo_date.isoformat(),
            'expiry_date': record.expiry_date.isoformat(),
            'grace_period_days': record.grace_period_days,
            'days_remaining': record.days_remaining,
            'reason': record.reason,
            'status': record.status,
            'external_references': record.external_references,
            'last_scan': record.last_scan.isoformat() if record.last_scan else None,
            'metadata': record.metadata,
            'generated_at': datetime.datetime.now().isoformat()
        }
        
        try:
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Report generated: {report_path}")
        except Exception as e:
            logger.error(f"Failed to generate report: {e}")
    
    def generate_summary_report(self) -> Dict:
        """Generate summary report of all embargoed directories"""
        summary = {
            'total_embargoed': len(self.embargo_registry),
            'ready_for_deletion': 0,
            'with_external_refs': 0,
            'expired_embargoes': 0,
            'directories': [],
            'generated_at': datetime.datetime.now().isoformat()
        }
        
        for directory, record in self.embargo_registry.items():
            if record.status == 'ready_for_deletion':
                summary['ready_for_deletion'] += 1
            if record.external_references:
                summary['with_external_refs'] += 1
            if record.is_expired:
                summary['expired_embargoes'] += 1
            
            summary['directories'].append({
                'directory': directory,
                'status': record.status,
                'days_remaining': record.days_remaining,
                'external_refs_count': len(record.external_references),
                'expiry_date': record.expiry_date.isoformat()
            })
        
        # Save summary report
        summary_path = self.reports_dir / 'embargo_summary.json'
        try:
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            logger.info(f"Summary report generated: {summary_path}")
        except Exception as e:
            logger.error(f"Failed to generate summary report: {e}")
        
        return summary
    
    def check_deprecated_imports(self) -> List[str]:
        """Check for imports of deprecated modules (for CI integration)"""
        violations = []
        embargoed_paths = set(self.embargo_registry.keys())
        
        for py_file in self.project_root.rglob('*.py'):
            # Skip files in embargoed directories
            file_in_embargo = any(
                str(py_file).startswith(embargo_path) 
                for embargo_path in embargoed_paths
            )
            if file_in_embargo:
                continue
            
            imports, from_imports = self.static_analyzer.analyze_file_imports(py_file)
            all_imports = imports | from_imports
            
            for embargo_path in embargoed_paths:
                embargo_module = embargo_path.replace('/', '.').replace('\\', '.')
                
                for imp in all_imports:
                    if imp.startswith(embargo_module):
                        violations.append(f"{py_file}:{imp}")
        
        return violations
    
    def safe_delete_ready_directories(self, dry_run: bool = True) -> List[str]:
        """Safely delete directories that are ready for deletion"""
        deleted = []
        
        for directory, record in list(self.embargo_registry.items()):
            if record.status == 'ready_for_deletion' and not record.external_references:
                if dry_run:
                    logger.info(f"Would delete {directory}")
                    deleted.append(directory)
                else:
                    try:
                        import shutil
                        shutil.rmtree(directory)
                        record.status = 'deleted'
                        record.metadata['deletion_date'] = datetime.datetime.now().isoformat()
                        logger.info(f"Deleted {directory}")
                        deleted.append(directory)
                    except Exception as e:
                        logger.error(f"Failed to delete {directory}: {e}")
        
        if not dry_run:
            self._save_embargo_registry()
        
        return deleted
    
    def run_nightly_scan(self):
        """Run the nightly automated verification scan"""
        logger.info("Starting nightly embargo scan")
        
        # Scan all embargoed directories
        scan_results = self.scan_embargoed_directories()
        
        # Generate reports
        summary = self.generate_summary_report()
        
        # Generate individual reports
        for record in self.embargo_registry.values():
            self._generate_embargo_report(record)
        
        # Check for CI violations
        violations = self.check_deprecated_imports()
        if violations:
            logger.warning(f"Found {len(violations)} deprecated import violations")
        
        logger.info(f"Nightly scan completed. {summary['ready_for_deletion']} directories ready for deletion")
        
        return {
            'scan_results': scan_results,
            'summary': summary,
            'violations': violations
        }


def main():
    """CLI interface for the controlled deletion system"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Controlled Deletion System")
    parser.add_argument('command', choices=[
        'embargo', 'scan', 'report', 'check-imports', 'delete', 'nightly'
    ])
    parser.add_argument('--directory', help="Directory to embargo")
    parser.add_argument('--reason', help="Reason for embargo")
    parser.add_argument('--grace-period', type=int, help="Grace period in days")
    parser.add_argument('--dry-run', action='store_true', help="Dry run mode")
    
    args = parser.parse_args()
    
    manager = ControlledDeletionManager()
    
    if args.command == 'embargo':
        if not args.directory or not args.reason:
            print("--directory and --reason are required for embargo command")
            sys.exit(1)
        success = manager.embargo_directory(args.directory, args.reason, args.grace_period)
        sys.exit(0 if success else 1)
    
    elif args.command == 'scan':
        results = manager.scan_embargoed_directories()
        print(json.dumps(results, indent=2))
    
    elif args.command == 'report':
        summary = manager.generate_summary_report()
        print(json.dumps(summary, indent=2))
    
    elif args.command == 'check-imports':
        violations = manager.check_deprecated_imports()
        if violations:
            print("Deprecated import violations found:")
            for violation in violations:
                print(f"  {violation}")
            sys.exit(1)
        else:
            print("No deprecated import violations found")
    
    elif args.command == 'delete':
        deleted = manager.safe_delete_ready_directories(dry_run=args.dry_run)
        if args.dry_run:
            print("Dry run - would delete:")
        else:
            print("Deleted directories:")
        for directory in deleted:
            print(f"  {directory}")
    
    elif args.command == 'nightly':
        results = manager.run_nightly_scan()
        print("Nightly scan completed")
        print(f"Ready for deletion: {results['summary']['ready_for_deletion']}")
        print(f"Import violations: {len(results['violations'])}")


if __name__ == '__main__':
    main()
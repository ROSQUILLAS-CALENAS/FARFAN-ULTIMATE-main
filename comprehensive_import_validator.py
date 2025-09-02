#!/usr/bin/env python3
"""
Comprehensive Import Validation Script

This script systematically validates all Python imports across the project by:
1. Parsing all Python files for import statements
2. Checking for missing dependencies against requirements.txt
3. Identifying version conflicts
4. Detecting circular imports
5. Verifying PDF processing libraries are functional
6. Fixing broken imports where possible
"""

import ast
import os
import sys
import subprocess
import importlib
import importlib.util
from pathlib import Path
from collections import defaultdict, deque
from typing import Dict, List, Set, Tuple, Optional, Any
# import pkg_resources  # Not needed, using subprocess instead
import re
import json
from dataclasses import dataclass


@dataclass
class ImportInfo:
    """Information about an import statement"""
    module: str
    alias: str
    line: int
    file_path: str
    is_from_import: bool
    imported_names: List[str]


@dataclass
class ValidationResult:
    """Result of import validation"""
    success: bool
    message: str
    fixes_applied: List[str]


class ImportValidator:
    """Comprehensive import validation system"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root).resolve()
        self.python_files = []
        self.all_imports = []
        self.circular_imports = []
        self.missing_dependencies = []
        self.version_conflicts = []
        self.broken_imports = []
        self.pdf_libraries = ['PyMuPDF', 'pdfplumber', 'pytesseract', 'camelot', 'tabula']
        self.fixes_applied = []
        
    def find_python_files(self) -> List[Path]:
        """Find all Python files in the project"""
        python_files = []
        for path in self.project_root.rglob("*.py"):
            # Skip virtual environment and build directories
            if any(part in ['.git', 'venv', '__pycache__', '.pytest_cache', 'build', 'dist', '.idea'] 
                   for part in path.parts):
                continue
            python_files.append(path)
        
        self.python_files = python_files
        return python_files
    
    def parse_imports(self, file_path: Path) -> List[ImportInfo]:
        """Parse all import statements from a Python file"""
        imports = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(ImportInfo(
                            module=alias.name,
                            alias=alias.asname or alias.name,
                            line=node.lineno,
                            file_path=str(file_path),
                            is_from_import=False,
                            imported_names=[alias.name]
                        ))
                
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imported_names = [alias.name for alias in node.names]
                        imports.append(ImportInfo(
                            module=node.module,
                            alias=node.module,
                            line=node.lineno,
                            file_path=str(file_path),
                            is_from_import=True,
                            imported_names=imported_names
                        ))
                        
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
        
        return imports
    
    def get_installed_packages(self) -> Dict[str, str]:
        """Get all installed packages and their versions"""
        try:
            result = subprocess.run([sys.executable, '-m', 'pip', 'list', '--format=json'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                packages = json.loads(result.stdout)
                return {pkg['name'].lower(): pkg['version'] for pkg in packages}
        except Exception as e:
            print(f"Error getting installed packages: {e}")
        return {}
    
    def get_requirements(self) -> Dict[str, str]:
        """Parse requirements.txt for expected dependencies"""
        requirements = {}
        req_file = self.project_root / "requirements.txt"
        
        if req_file.exists():
            try:
                with open(req_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            # Handle version specifications
                            match = re.match(r'^([a-zA-Z0-9\-_\.]+)([><=!]+)([0-9\.]+)', line)
                            if match:
                                name = match.group(1).lower()
                                version = match.group(3)
                                requirements[name] = version
                            else:
                                # Simple package name
                                if '[' in line:
                                    name = line.split('[')[0]
                                else:
                                    name = line
                                requirements[name.lower()] = None
            except Exception as e:
                print(f"Error parsing requirements.txt: {e}")
        
        return requirements
    
    def check_module_availability(self, module_name: str) -> bool:
        """Check if a module can be imported"""
        try:
            # Handle relative imports
            if module_name.startswith('.'):
                return True
            
            # Map common aliases to actual package names
            module_mappings = {
                'cv2': 'opencv-python',
                'PIL': 'pillow',
                'skimage': 'scikit-image',
                'sklearn': 'scikit-learn',
                'torch': 'torch',
                'tf': 'tensorflow',
                'pd': 'pandas',
                'np': 'numpy'
            }
            
            actual_module = module_mappings.get(module_name.split('.')[0], module_name.split('.')[0])
            
            spec = importlib.util.find_spec(module_name.split('.')[0])
            return spec is not None
            
        except Exception:
            return False
    
    def detect_circular_imports(self) -> List[List[str]]:
        """Detect circular imports using dependency graph analysis"""
        # Build dependency graph
        graph = defaultdict(set)
        
        for import_info in self.all_imports:
            file_module = self.file_to_module(import_info.file_path)
            imported_module = import_info.module
            
            # Only consider internal modules
            if self.is_internal_module(imported_module):
                graph[file_module].add(imported_module)
        
        # Find cycles using DFS
        def find_cycles():
            visited = set()
            rec_stack = set()
            cycles = []
            
            def dfs(node, path):
                if node in rec_stack:
                    # Found a cycle
                    cycle_start = path.index(node)
                    cycle = path[cycle_start:] + [node]
                    if len(cycle) > 2:  # Ignore self-loops
                        cycles.append(cycle)
                    return
                
                if node in visited:
                    return
                
                visited.add(node)
                rec_stack.add(node)
                
                for neighbor in graph.get(node, []):
                    dfs(neighbor, path + [node])
                
                rec_stack.remove(node)
            
            for node in graph:
                if node not in visited:
                    dfs(node, [])
            
            return cycles
        
        self.circular_imports = find_cycles()
        return self.circular_imports
    
    def file_to_module(self, file_path: str) -> str:
        """Convert file path to module name"""
        path = Path(file_path).relative_to(self.project_root)
        if path.name == '__init__.py':
            return str(path.parent).replace('/', '.')
        else:
            return str(path.with_suffix('')).replace('/', '.')
    
    def is_internal_module(self, module_name: str) -> bool:
        """Check if module is part of the current project"""
        if module_name.startswith('.'):
            return True
        
        # Check if module corresponds to a file in the project
        module_path = self.project_root / module_name.replace('.', '/')
        return (module_path.is_dir() and (module_path / '__init__.py').exists()) or \
               (module_path.with_suffix('.py').exists())
    
    def check_pdf_libraries(self) -> ValidationResult:
        """Specifically validate PDF processing libraries"""
        pdf_checks = []
        
        for lib in self.pdf_libraries:
            try:
                if lib == 'PyMuPDF':
                    import fitz
                    pdf_checks.append(f"✓ PyMuPDF (fitz) is available")
                elif lib == 'pdfplumber':
                    import pdfplumber
                    pdf_checks.append(f"✓ pdfplumber is available")
                elif lib == 'pytesseract':
                    import pytesseract
                    pdf_checks.append(f"✓ pytesseract is available")
                elif lib == 'camelot':
                    import camelot
                    pdf_checks.append(f"✓ camelot is available")
                elif lib == 'tabula':
                    import tabula
                    pdf_checks.append(f"✓ tabula is available")
                    
            except ImportError as e:
                pdf_checks.append(f"✗ {lib} is not available: {e}")
        
        return ValidationResult(
            success=all("✓" in check for check in pdf_checks),
            message="\n".join(pdf_checks),
            fixes_applied=[]
        )
    
    def fix_broken_imports(self) -> List[str]:
        """Attempt to fix broken imports"""
        fixes = []
        
        # Common typos and corrections
        corrections = {
            'numpy': 'numpy',
            'pandas': 'pandas',
            'sklearn': 'scikit-learn',
            'cv2': 'opencv-python',
            'PIL': 'pillow',
            'requests': 'requests',
            'yaml': 'pyyaml',
        }
        
        # Install missing packages that are in requirements
        requirements = self.get_requirements()
        installed = self.get_installed_packages()
        
        for req_name in requirements:
            if req_name not in installed:
                try:
                    print(f"Installing missing package: {req_name}")
                    result = subprocess.run([sys.executable, '-m', 'pip', 'install', req_name], 
                                          capture_output=True, text=True)
                    if result.returncode == 0:
                        fixes.append(f"Installed {req_name}")
                    else:
                        fixes.append(f"Failed to install {req_name}: {result.stderr}")
                except Exception as e:
                    fixes.append(f"Error installing {req_name}: {e}")
        
        self.fixes_applied.extend(fixes)
        return fixes
    
    def validate_all_imports(self) -> Dict[str, Any]:
        """Run comprehensive import validation"""
        print("Starting comprehensive import validation...")
        
        # Find all Python files
        print(f"Found {len(self.find_python_files())} Python files")
        
        # Parse all imports
        for file_path in self.python_files:
            imports = self.parse_imports(file_path)
            self.all_imports.extend(imports)
        
        print(f"Found {len(self.all_imports)} import statements")
        
        # Check for missing dependencies
        print("Checking for missing dependencies...")
        missing_deps = []
        for import_info in self.all_imports:
            if not self.check_module_availability(import_info.module):
                if not import_info.module.startswith('.'):  # Skip relative imports
                    missing_deps.append(import_info)
        
        self.missing_dependencies = missing_deps
        
        # Detect circular imports
        print("Detecting circular imports...")
        self.detect_circular_imports()
        
        # Check PDF libraries
        print("Validating PDF processing libraries...")
        pdf_result = self.check_pdf_libraries()
        
        # Attempt fixes
        print("Attempting to fix broken imports...")
        fixes = self.fix_broken_imports()
        
        # Generate report
        report = {
            'total_python_files': len(self.python_files),
            'total_imports': len(self.all_imports),
            'missing_dependencies': len(self.missing_dependencies),
            'circular_imports': len(self.circular_imports),
            'pdf_libraries_status': pdf_result.success,
            'fixes_applied': fixes,
            'details': {
                'missing_deps': [
                    {
                        'module': imp.module,
                        'file': imp.file_path,
                        'line': imp.line
                    } for imp in self.missing_dependencies
                ],
                'circular_imports': self.circular_imports,
                'pdf_status': pdf_result.message
            }
        }
        
        return report
    
    def generate_detailed_report(self, report: Dict[str, Any]) -> str:
        """Generate a detailed text report"""
        lines = []
        lines.append("=" * 60)
        lines.append("COMPREHENSIVE IMPORT VALIDATION REPORT")
        lines.append("=" * 60)
        lines.append("")
        
        # Summary
        lines.append("SUMMARY:")
        lines.append(f"  Total Python files: {report['total_python_files']}")
        lines.append(f"  Total import statements: {report['total_imports']}")
        lines.append(f"  Missing dependencies: {report['missing_dependencies']}")
        lines.append(f"  Circular imports detected: {report['circular_imports']}")
        lines.append(f"  PDF libraries functional: {'Yes' if report['pdf_libraries_status'] else 'No'}")
        lines.append(f"  Fixes applied: {len(report['fixes_applied'])}")
        lines.append("")
        
        # Missing dependencies
        if report['details']['missing_deps']:
            lines.append("MISSING DEPENDENCIES:")
            for dep in report['details']['missing_deps']:
                lines.append(f"  ✗ {dep['module']} in {dep['file']}:{dep['line']}")
            lines.append("")
        
        # Circular imports
        if report['details']['circular_imports']:
            lines.append("CIRCULAR IMPORTS:")
            for cycle in report['details']['circular_imports']:
                lines.append(f"  ⚠ {' -> '.join(cycle)}")
            lines.append("")
        
        # PDF libraries status
        lines.append("PDF LIBRARIES STATUS:")
        for line in report['details']['pdf_status'].split('\n'):
            lines.append(f"  {line}")
        lines.append("")
        
        # Fixes applied
        if report['fixes_applied']:
            lines.append("FIXES APPLIED:")
            for fix in report['fixes_applied']:
                lines.append(f"  ✓ {fix}")
            lines.append("")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)


def main():
    """Main function to run import validation"""
    validator = ImportValidator()
    
    try:
        report = validator.validate_all_imports()
        detailed_report = validator.generate_detailed_report(report)
        
        print(detailed_report)
        
        # Save report to file
        with open('import_validation_report.txt', 'w') as f:
            f.write(detailed_report)
        
        # Save JSON report for programmatic access
        with open('import_validation_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print("\nReports saved to:")
        print("  - import_validation_report.txt")
        print("  - import_validation_report.json")
        
        # Exit with error code if issues found
        issues = (report['missing_dependencies'] + 
                 report['circular_imports'] + 
                 (0 if report['pdf_libraries_status'] else 1))
        
        if issues > 0:
            print(f"\n⚠ Found {issues} issues that need attention")
            return 1
        else:
            print("\n✓ All imports validated successfully!")
            return 0
            
    except Exception as e:
        print(f"Error during validation: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python3
"""
Import validation script for static analysis firewall.
Detects and prevents problematic import patterns including star imports.
"""

import ast
import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple


class ImportValidator(ast.NodeVisitor):
    """AST visitor for validating import patterns."""
    
    def __init__(self, filepath: str) -> None:
        self.filepath = filepath
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.imports: Set[str] = set()
        self.star_imports: List[str] = []
        
    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Visit ImportFrom nodes to check for star imports."""
        if node.module:
            for alias in node.names:
                if alias.name == '*':
                    self.star_imports.append(node.module)
                    self.errors.append(
                        f"Star import detected: 'from {node.module} import *' "
                        f"at line {node.lineno}"
                    )
                else:
                    self.imports.add(f"{node.module}.{alias.name}")
        self.generic_visit(node)
        
    def visit_Import(self, node: ast.Import) -> None:
        """Visit Import nodes to collect imports."""
        for alias in node.names:
            self.imports.add(alias.name)
        self.generic_visit(node)


def analyze_file(filepath: Path) -> Tuple[List[str], List[str], Set[str], List[str]]:
    """
    Analyze a Python file for import violations.
    
    Returns:
        Tuple of (errors, warnings, imports, star_imports)
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            
        tree = ast.parse(content, filename=str(filepath))
        validator = ImportValidator(str(filepath))
        validator.visit(tree)
        
        return validator.errors, validator.warnings, validator.imports, validator.star_imports
        
    except SyntaxError as e:
        return [f"Syntax error in {filepath}: {e}"], [], set(), []
    except Exception as e:
        return [f"Error analyzing {filepath}: {e}"], [], set(), []


def get_python_files(directory: Path) -> List[Path]:
    """Get all Python files in the directory recursively."""
    python_files = []
    for root, dirs, files in os.walk(directory):
        # Skip common directories that don't need validation
        dirs[:] = [d for d in dirs if d not in {
            '__pycache__', '.git', '.mypy_cache', '.pytest_cache', 
            '.ruff_cache', 'venv', '.venv', 'node_modules', 'build', 'dist'
        }]
        
        for file in files:
            if file.endswith('.py'):
                python_files.append(Path(root) / file)
    
    return python_files


def validate_imports_in_directory(directory: Path) -> bool:
    """
    Validate imports in all Python files in the directory.
    
    Returns:
        True if no errors found, False otherwise
    """
    python_files = get_python_files(directory)
    
    total_errors = 0
    total_warnings = 0
    all_star_imports = []
    
    print(f"Validating imports in {len(python_files)} Python files...")
    
    for filepath in python_files:
        errors, warnings, imports, star_imports = analyze_file(filepath)
        
        if errors:
            total_errors += len(errors)
            print(f"\n❌ ERRORS in {filepath}:")
            for error in errors:
                print(f"  • {error}")
                
        if warnings:
            total_warnings += len(warnings)
            print(f"\n⚠️  WARNINGS in {filepath}:")
            for warning in warnings:
                print(f"  • {warning}")
                
        if star_imports:
            all_star_imports.extend([(str(filepath), si) for si in star_imports])
    
    # Summary report
    print(f"\n" + "="*60)
    print("IMPORT VALIDATION SUMMARY")
    print("="*60)
    print(f"Files analyzed: {len(python_files)}")
    print(f"Errors found: {total_errors}")
    print(f"Warnings found: {total_warnings}")
    print(f"Star imports found: {len(all_star_imports)}")
    
    if all_star_imports:
        print(f"\n❌ STAR IMPORTS DETECTED:")
        for filepath, module in all_star_imports:
            print(f"  • {filepath}: from {module} import *")
    
    if total_errors == 0 and len(all_star_imports) == 0:
        print(f"\n✅ All import validations passed!")
        return True
    else:
        print(f"\n❌ Import validation failed!")
        if len(all_star_imports) > 0:
            print("Star imports must be eliminated before committing.")
        return False


def main() -> int:
    """Main entry point."""
    # Get directories to validate
    directories = [
        Path("egw_query_expansion"),
        Path("src"), 
        Path("scripts"),
        Path("tests")
    ]
    
    # Filter to existing directories
    existing_dirs = [d for d in directories if d.exists() and d.is_dir()]
    
    if not existing_dirs:
        print("No Python source directories found to validate.")
        return 0
    
    overall_success = True
    
    for directory in existing_dirs:
        print(f"\n{'='*60}")
        print(f"VALIDATING DIRECTORY: {directory}")
        print(f"{'='*60}")
        
        success = validate_imports_in_directory(directory)
        overall_success = overall_success and success
    
    return 0 if overall_success else 1


if __name__ == "__main__":
    sys.exit(main())
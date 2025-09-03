#!/usr/bin/env python3
"""
Static analysis tool to detect and reject star imports.
Part of the comprehensive static analysis configuration system.
"""

import ast
import sys
from pathlib import Path
from typing import List, Optional, Tuple


class StarImportDetector(ast.NodeVisitor):
    """AST visitor to detect star imports (from ... import *)."""
    
    def __init__(self) -> None:
        self.star_imports: List[Tuple[int, str]] = []
        
    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Check for star imports in ImportFrom nodes."""
        if any(alias.name == '*' for alias in node.names):
            module_name = node.module or "unknown"
            self.star_imports.append((node.lineno, module_name))
        self.generic_visit(node)


def check_file_for_star_imports(file_path: Path) -> List[Tuple[int, str]]:
    """
    Check a single Python file for star imports.
    
    Args:
        file_path: Path to the Python file to check
        
    Returns:
        List of (line_number, module_name) tuples for star imports found
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        tree = ast.parse(content, filename=str(file_path))
        detector = StarImportDetector()
        detector.visit(tree)
        
        return detector.star_imports
        
    except (SyntaxError, UnicodeDecodeError) as e:
        print(f"Warning: Could not parse {file_path}: {e}", file=sys.stderr)
        return []


def generate_remediation_suggestion(file_path: Path, line_num: int, module_name: str) -> str:
    """
    Generate a remediation suggestion for a star import.
    
    Args:
        file_path: Path to the file with the star import
        line_num: Line number of the star import
        module_name: Name of the module being star imported
        
    Returns:
        Formatted remediation suggestion
    """
    suggestions = {
        "typing": "Consider importing specific types: from typing import List, Dict, Optional, Union",
        "numpy": "Use explicit imports: import numpy as np",
        "pandas": "Use explicit imports: import pandas as pd",
        "matplotlib.pyplot": "Use standard convention: import matplotlib.pyplot as plt",
        "torch": "Use explicit imports: import torch or import torch.nn as nn",
        "transformers": "Import specific classes: from transformers import AutoModel, AutoTokenizer",
    }
    
    specific_suggestion = suggestions.get(module_name, 
        f"Replace 'from {module_name} import *' with explicit imports of needed symbols")
    
    return f"""
REMEDIATION SUGGESTION for {file_path}:{line_num}:
- Issue: Star import from '{module_name}' pollutes namespace and hides dependencies
- Fix: {specific_suggestion}
- Example: from {module_name} import symbol1, symbol2
- Benefits: Explicit dependencies, better IDE support, no namespace pollution
"""


def main() -> int:
    """
    Main function to check files for star imports.
    
    Returns:
        Exit code: 0 if no star imports found, 1 if found
    """
    if len(sys.argv) < 2:
        print("Usage: check_star_imports.py <file1.py> [file2.py ...]", file=sys.stderr)
        return 1
        
    total_violations = 0
    
    for file_arg in sys.argv[1:]:
        file_path = Path(file_arg)
        
        if not file_path.exists():
            print(f"Warning: File {file_path} does not exist", file=sys.stderr)
            continue
            
        if not file_path.suffix == '.py':
            continue
            
        star_imports = check_file_for_star_imports(file_path)
        
        if star_imports:
            total_violations += len(star_imports)
            print(f"\n❌ STAR IMPORTS DETECTED in {file_path}:")
            
            for line_num, module_name in star_imports:
                print(f"  Line {line_num}: from {module_name} import *")
                print(generate_remediation_suggestion(file_path, line_num, module_name))
                
    if total_violations > 0:
        print(f"\n🚨 TOTAL VIOLATIONS: {total_violations} star import(s) found")
        print("❌ PRE-COMMIT HOOK FAILED: Please fix star imports before committing")
        return 1
    else:
        print("✅ No star imports detected. Good job!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
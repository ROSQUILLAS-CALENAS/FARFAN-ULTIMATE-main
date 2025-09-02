#!/usr/bin/env python3
"""
Run contract validation utility with minimal dependencies
"""

import ast
import sys
from pathlib import Path
import re
from typing import List, Dict, Set, Optional

# Add project root to path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

def find_contract_files() -> List[Path]:
    """Find all contract-related files."""
    patterns = [
        "**/test_*_contract.py",
        "**/test_*contract*.py", 
        "**/*contract*.py",
        "**/test_routing*.py",
        "**/test_snapshot*.py",
    ]
    
    files = set()
    for pattern in patterns:
        files.update(project_root.glob(pattern))
    
    # Filter out __pycache__ and .git
    return [f for f in files if "__pycache__" not in str(f) and ".git" not in str(f)]

def analyze_imports(file_path: Path) -> Dict[str, List[str]]:
    """Analyze imports in a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        imports = {"from": [], "import": [], "issues": []}
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module:
                    imports["from"].append((node.lineno, node.module))
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imports["import"].append((node.lineno, alias.name))
        
        # Check for problematic patterns
        lines = content.split('\n')
        for i, line in enumerate(lines, 1):
            # Check for canonical_flow imports
            if "canonical_flow" in line and "import" in line:
                imports["issues"].append(f"Line {i}: Non-canonical import: {line.strip()}")
            
            # Check for missing sys.path setup
            if "from egw_query_expansion" in line or "from snapshot_manager" in line:
                # Look for sys.path setup in nearby lines
                setup_found = False
                for j in range(max(0, i-10), min(len(lines), i+5)):
                    if "sys.path" in lines[j] or "project_root" in lines[j]:
                        setup_found = True
                        break
                
                if not setup_found:
                    imports["issues"].append(f"Line {i}: Missing path setup for import: {line.strip()}")
        
        return imports
    except Exception as e:
        return {"from": [], "import": [], "issues": [f"Parse error: {e}"]}

def main():
    print("🔍 Contract Files Validation Report")
    print("=" * 50)
    
    contract_files = find_contract_files()
    print(f"Found {len(contract_files)} contract files\n")
    
    all_issues = []
    
    for file_path in sorted(contract_files):
        rel_path = file_path.relative_to(project_root)
        print(f"📁 {rel_path}")
        
        imports = analyze_imports(file_path)
        
        # Show from imports
        if imports["from"]:
            print("  From imports:")
            for line_no, module in imports["from"]:
                print(f"    Line {line_no}: from {module}")
        
        # Show regular imports  
        if imports["import"]:
            print("  Import statements:")
            for line_no, module in imports["import"]:
                print(f"    Line {line_no}: import {module}")
        
        # Show issues
        if imports["issues"]:
            print("  ⚠️  Issues found:")
            for issue in imports["issues"]:
                print(f"    {issue}")
                all_issues.append(f"{rel_path}: {issue}")
        else:
            print("  ✅ No issues found")
        
        print()
    
    print("=" * 50)
    print(f"📊 Summary: {len(all_issues)} total issues found")
    
    if all_issues:
        print("\n🔧 Issues to fix:")
        for issue in all_issues:
            print(f"  - {issue}")
        
        print("\n💡 Recommended fixes:")
        print("  1. Add canonical path setup to files missing it:")
        print("     ```python")
        print("     import sys")
        print("     from pathlib import Path")
        print("     project_root = Path(__file__).resolve().parents[1]")
        print("     if str(project_root) not in sys.path:")
        print("         sys.path.insert(0, str(project_root))")
        print("     ```")
        print("  2. Replace canonical_flow imports with direct canonical imports")
        print("  3. Use egw_query_expansion.core.* for core modules")
    else:
        print("🎉 All contract files look good!")
    
    return 0 if not all_issues else 1

if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python3
"""
Static analysis tool to detect runtime circular import patterns.
Part of the comprehensive static analysis configuration system.
"""

import ast
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


class ImportAnalyzer(ast.NodeVisitor):
    """AST visitor to extract import information from Python files."""
    
    def __init__(self, file_path: Path) -> None:
        self.file_path = file_path
        self.imports: List[str] = []
        self.relative_imports: List[str] = []
        
    def visit_Import(self, node: ast.Import) -> None:
        """Extract regular imports."""
        for alias in node.names:
            self.imports.append(alias.name.split('.')[0])
        self.generic_visit(node)
        
    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Extract from imports."""
        if node.module:
            if node.level > 0:  # Relative import
                self.relative_imports.append(node.module.split('.')[0])
            else:  # Absolute import
                self.imports.append(node.module.split('.')[0])
        self.generic_visit(node)


def build_dependency_graph(file_paths: List[Path], package_prefix: str = "egw_query_expansion") -> Dict[str, Set[str]]:
    """
    Build a dependency graph from Python files.
    
    Args:
        file_paths: List of Python file paths to analyze
        package_prefix: Package prefix to focus on for circular import detection
        
    Returns:
        Dictionary mapping module names to their dependencies
    """
    graph: Dict[str, Set[str]] = defaultdict(set)
    
    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            tree = ast.parse(content, filename=str(file_path))
            analyzer = ImportAnalyzer(file_path)
            analyzer.visit(tree)
            
            # Convert file path to module name
            module_name = str(file_path.relative_to(file_path.parents[1])).replace('/', '.').replace('\\', '.').rstrip('.py')
            if module_name.endswith('.__init__'):
                module_name = module_name[:-9]
                
            # Focus on internal package imports
            for imp in analyzer.imports + analyzer.relative_imports:
                if imp.startswith(package_prefix):
                    graph[module_name].add(imp)
                    
        except (SyntaxError, UnicodeDecodeError, ValueError) as e:
            print(f"Warning: Could not parse {file_path}: {e}", file=sys.stderr)
            continue
            
    return graph


def find_circular_dependencies(graph: Dict[str, Set[str]]) -> List[List[str]]:
    """
    Find circular dependencies in the import graph using DFS.
    
    Args:
        graph: Dependency graph
        
    Returns:
        List of circular dependency cycles
    """
    visited = set()
    rec_stack = set()
    cycles = []
    
    def dfs(node: str, path: List[str]) -> None:
        if node in rec_stack:
            # Found a cycle
            cycle_start = path.index(node)
            cycles.append(path[cycle_start:] + [node])
            return
            
        if node in visited:
            return
            
        visited.add(node)
        rec_stack.add(node)
        path.append(node)
        
        for neighbor in graph.get(node, set()):
            dfs(neighbor, path.copy())
            
        rec_stack.remove(node)
        
    for node in graph:
        if node not in visited:
            dfs(node, [])
            
    return cycles


def generate_circular_import_remediation(cycle: List[str]) -> str:
    """
    Generate remediation suggestions for a circular import.
    
    Args:
        cycle: List of modules in the circular dependency
        
    Returns:
        Formatted remediation suggestion
    """
    return f"""
CIRCULAR IMPORT REMEDIATION for cycle: {' → '.join(cycle)}

COMMON SOLUTIONS:
1. **Move shared code to a common module**:
   - Create a separate module for shared functions/classes
   - Import from the common module instead of each other

2. **Use local imports**:
   - Move imports inside functions where they're used
   - Delays import until runtime when needed

3. **Refactor architecture**:
   - Consider if the circular dependency indicates poor separation of concerns
   - Split modules based on responsibility, not convenience

4. **Use TYPE_CHECKING imports**:
   ```python
   from typing import TYPE_CHECKING
   if TYPE_CHECKING:
       from .other_module import SomeClass
   ```

EXAMPLE REFACTORING:
Before:
  # module_a.py
  from .module_b import function_b
  
  # module_b.py  
  from .module_a import function_a

After:
  # common.py
  def shared_function(): ...
  
  # module_a.py
  from .common import shared_function
  
  # module_b.py
  from .common import shared_function
"""


def main() -> int:
    """
    Main function to check files for circular imports.
    
    Returns:
        Exit code: 0 if no circular imports found, 1 if found
    """
    if len(sys.argv) < 2:
        print("Usage: check_circular_imports.py <file1.py> [file2.py ...]", file=sys.stderr)
        return 1
        
    file_paths = []
    for file_arg in sys.argv[1:]:
        file_path = Path(file_arg)
        if file_path.exists() and file_path.suffix == '.py':
            file_paths.append(file_path)
            
    if not file_paths:
        print("No valid Python files provided", file=sys.stderr)
        return 1
        
    # Build dependency graph
    graph = build_dependency_graph(file_paths)
    
    # Find circular dependencies
    cycles = find_circular_dependencies(graph)
    
    if cycles:
        print(f"\n❌ CIRCULAR IMPORTS DETECTED ({len(cycles)} cycle(s)):")
        
        for i, cycle in enumerate(cycles, 1):
            print(f"\nCycle {i}: {' → '.join(cycle)}")
            print(generate_circular_import_remediation(cycle))
            
        print(f"\n🚨 TOTAL VIOLATIONS: {len(cycles)} circular import cycle(s) found")
        print("❌ PRE-COMMIT HOOK FAILED: Please fix circular imports before committing")
        return 1
    else:
        print("✅ No circular imports detected. Good job!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python3
"""
Circular import detection script for static analysis firewall.
Analyzes import dependency graph to detect and prevent circular dependencies.
"""

import ast
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional


class ImportGraphAnalyzer(ast.NodeVisitor):
    """AST visitor for building import dependency graph."""
    
    def __init__(self, filepath: str, package_root: Path) -> None:
        self.filepath = Path(filepath)
        self.package_root = package_root
        self.imports: Set[str] = set()
        
    def visit_Import(self, node: ast.Import) -> None:
        """Visit Import nodes."""
        for alias in node.names:
            self.imports.add(alias.name)
        self.generic_visit(node)
        
    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Visit ImportFrom nodes."""
        if node.module and not node.module.startswith('.'):
            # Only process absolute imports
            self.imports.add(node.module)
        elif node.module and node.module.startswith('.'):
            # Handle relative imports
            relative_module = self._resolve_relative_import(node.module, node.level)
            if relative_module:
                self.imports.add(relative_module)
        self.generic_visit(node)
        
    def _resolve_relative_import(self, module: str, level: int) -> Optional[str]:
        """Resolve relative import to absolute module name."""
        try:
            # Get the package path relative to package root
            rel_path = self.filepath.relative_to(self.package_root)
            
            # Remove the .py extension
            if rel_path.suffix == '.py':
                rel_path = rel_path.with_suffix('')
            
            # Convert path to module name
            parts = list(rel_path.parts)
            if parts[-1] == '__init__':
                parts = parts[:-1]
            
            # Go up 'level' directories
            for _ in range(level):
                if parts:
                    parts.pop()
            
            # Add the relative module
            if module.startswith('.'):
                module = module[1:]  # Remove leading dot
            if module:
                parts.append(module)
            
            return '.'.join(parts) if parts else None
        except (ValueError, IndexError):
            return None


def build_dependency_graph(directory: Path) -> Dict[str, Set[str]]:
    """
    Build import dependency graph for Python files in directory.
    
    Returns:
        Dictionary mapping module names to their dependencies
    """
    graph = defaultdict(set)
    python_files = []
    
    # Find all Python files
    for root, dirs, files in os.walk(directory):
        # Skip common directories
        dirs[:] = [d for d in dirs if d not in {
            '__pycache__', '.git', '.mypy_cache', '.pytest_cache',
            '.ruff_cache', 'venv', '.venv', 'node_modules', 'build', 'dist'
        }]
        
        for file in files:
            if file.endswith('.py'):
                python_files.append(Path(root) / file)
    
    # Analyze each file
    for filepath in python_files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content, filename=str(filepath))
            analyzer = ImportGraphAnalyzer(str(filepath), directory)
            analyzer.visit(tree)
            
            # Convert file path to module name
            rel_path = filepath.relative_to(directory)
            if rel_path.suffix == '.py':
                rel_path = rel_path.with_suffix('')
            
            parts = list(rel_path.parts)
            if parts[-1] == '__init__':
                parts = parts[:-1]
                
            module_name = '.'.join(parts) if parts else ''
            
            if module_name:
                # Filter imports to only include internal modules
                internal_imports = {
                    imp for imp in analyzer.imports
                    if imp.startswith(directory.name) or not '.' in imp.split('.')[0]
                }
                graph[module_name] = internal_imports
                
        except (SyntaxError, UnicodeDecodeError, ValueError) as e:
            print(f"Warning: Could not analyze {filepath}: {e}")
            continue
    
    return dict(graph)


def detect_cycles(graph: Dict[str, Set[str]]) -> List[List[str]]:
    """
    Detect cycles in dependency graph using DFS.
    
    Returns:
        List of cycles, where each cycle is a list of module names
    """
    WHITE, GRAY, BLACK = 0, 1, 2
    color = defaultdict(int)
    parent = {}
    cycles = []
    
    def dfs_visit(node: str, path: List[str]) -> None:
        color[node] = GRAY
        path.append(node)
        
        for neighbor in graph.get(node, set()):
            if color[neighbor] == WHITE:
                parent[neighbor] = node
                dfs_visit(neighbor, path.copy())
            elif color[neighbor] == GRAY:
                # Back edge found - cycle detected
                cycle_start = path.index(neighbor)
                cycle = path[cycle_start:] + [neighbor]
                cycles.append(cycle)
        
        color[node] = BLACK
        if path and path[-1] == node:
            path.pop()
    
    # Run DFS from all unvisited nodes
    for node in graph:
        if color[node] == WHITE:
            dfs_visit(node, [])
    
    return cycles


def analyze_circular_imports(directory: Path) -> Tuple[Dict[str, Set[str]], List[List[str]]]:
    """
    Analyze directory for circular imports.
    
    Returns:
        Tuple of (dependency_graph, cycles)
    """
    print(f"Building dependency graph for {directory}...")
    graph = build_dependency_graph(directory)
    
    print(f"Detecting cycles in {len(graph)} modules...")
    cycles = detect_cycles(graph)
    
    return graph, cycles


def main() -> int:
    """Main entry point."""
    # Get directories to analyze
    directories = [
        Path("egw_query_expansion"),
        Path("src"),
        Path("scripts"),
        Path("tests")
    ]
    
    # Filter to existing directories
    existing_dirs = [d for d in directories if d.exists() and d.is_dir()]
    
    if not existing_dirs:
        print("No Python source directories found to analyze.")
        return 0
    
    total_cycles = 0
    overall_success = True
    
    for directory in existing_dirs:
        print(f"\n{'='*60}")
        print(f"ANALYZING DIRECTORY: {directory}")
        print(f"{'='*60}")
        
        try:
            graph, cycles = analyze_circular_imports(directory)
            
            if cycles:
                total_cycles += len(cycles)
                overall_success = False
                
                print(f"\n❌ CIRCULAR IMPORTS DETECTED ({len(cycles)} cycles):")
                for i, cycle in enumerate(cycles, 1):
                    print(f"\n  Cycle {i}: {' → '.join(cycle)}")
            else:
                print(f"\n✅ No circular imports detected in {directory}")
                
            # Print dependency statistics
            total_modules = len(graph)
            total_dependencies = sum(len(deps) for deps in graph.values())
            avg_dependencies = total_dependencies / total_modules if total_modules > 0 else 0
            
            print(f"\n📊 DEPENDENCY STATISTICS:")
            print(f"  • Total modules: {total_modules}")
            print(f"  • Total dependencies: {total_dependencies}")
            print(f"  • Average dependencies per module: {avg_dependencies:.1f}")
            
        except Exception as e:
            print(f"❌ Error analyzing {directory}: {e}")
            overall_success = False
    
    # Overall summary
    print(f"\n" + "="*60)
    print("CIRCULAR IMPORT ANALYSIS SUMMARY")
    print("="*60)
    print(f"Directories analyzed: {len(existing_dirs)}")
    print(f"Total cycles found: {total_cycles}")
    
    if overall_success:
        print(f"\n✅ No circular imports detected!")
        return 0
    else:
        print(f"\n❌ Circular imports detected! Fix before committing.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
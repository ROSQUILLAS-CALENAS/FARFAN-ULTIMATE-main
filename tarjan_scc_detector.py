#!/usr/bin/env python3
"""
Tarjan's Strongly Connected Components (SCC) Algorithm for Import Dependency Analysis

This module implements Tarjan's algorithm to detect strongly connected components 
in the project's import dependency graph, identifying circular dependencies 
and generating comprehensive cycle detection reports.
"""

import ast
import os
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
import time
import json


class ImportVisitor(ast.NodeVisitor):
    """AST visitor to extract import statements from Python files."""
    
    def __init__(self, module_path: str, project_root: str):
        self.module_path = module_path
        self.project_root = project_root
        self.imports = set()
        
    def visit_Import(self, node):
        """Handle 'import module' statements."""
        for alias in node.names:
            self.imports.add(alias.name.split('.')[0])
        self.generic_visit(node)
        
    def visit_ImportFrom(self, node):
        """Handle 'from module import ...' statements."""
        if node.module:
            self.imports.add(node.module.split('.')[0])
        self.generic_visit(node)


class TarjanSCCDetector:
    """
    Implementation of Tarjan's algorithm for strongly connected components detection
    in the import dependency graph of a Python project.
    """
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root).resolve()
        self.dependency_graph = defaultdict(set)
        self.modules = set()
        
        # Tarjan's algorithm state
        self.index_counter = 0
        self.index = {}
        self.lowlinks = {}
        self.on_stack = {}
        self.stack = []
        self.sccs = []
        
        # Analysis metrics
        self.analysis_start_time = None
        self.analysis_end_time = None
        self.total_files_scanned = 0
        self.total_imports_found = 0
        
    def find_python_files(self) -> List[Path]:
        """Find all Python files in the project directory."""
        python_files = []
        
        # Skip common non-source directories
        skip_dirs = {
            '.git', '__pycache__', '.pytest_cache', 'node_modules', 
            'venv', '.venv', 'env', '.env', 'build', 'dist', 
            '.mypy_cache', '.tox', 'htmlcov', '.coverage'
        }
        
        for root, dirs, files in os.walk(self.project_root):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if d not in skip_dirs]
            
            for file in files:
                if file.endswith('.py'):
                    python_files.append(Path(root) / file)
                    
        return python_files
    
    def extract_imports_from_file(self, file_path: Path) -> Set[str]:
        """Extract import dependencies from a single Python file."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            tree = ast.parse(content, filename=str(file_path))
            visitor = ImportVisitor(str(file_path), str(self.project_root))
            visitor.visit(tree)
            
            # Filter to only include internal project modules
            internal_imports = set()
            for imp in visitor.imports:
                if self.is_internal_module(imp):
                    internal_imports.add(imp)
                    
            return internal_imports
            
        except (SyntaxError, UnicodeDecodeError, FileNotFoundError) as e:
            print(f"Warning: Could not parse {file_path}: {e}")
            return set()
    
    def is_internal_module(self, module_name: str) -> bool:
        """Check if a module is internal to the project."""
        # Check if there's a corresponding Python file or package
        possible_paths = [
            self.project_root / f"{module_name}.py",
            self.project_root / module_name / "__init__.py"
        ]
        
        return any(path.exists() for path in possible_paths)
    
    def build_dependency_graph(self):
        """Build the dependency graph by scanning all Python files."""
        print("Building dependency graph...")
        self.analysis_start_time = time.time()
        
        python_files = self.find_python_files()
        self.total_files_scanned = len(python_files)
        
        for file_path in python_files:
            # Convert file path to module name
            rel_path = file_path.relative_to(self.project_root)
            
            if rel_path.name == '__init__.py':
                module_name = str(rel_path.parent).replace(os.sep, '.')
            else:
                module_name = str(rel_path.with_suffix('')).replace(os.sep, '.')
                
            # Handle root-level files
            if module_name.startswith('.'):
                module_name = module_name[1:]
                
            self.modules.add(module_name)
            
            # Extract imports from this file
            imports = self.extract_imports_from_file(file_path)
            self.total_imports_found += len(imports)
            
            for imported_module in imports:
                self.dependency_graph[module_name].add(imported_module)
                
        print(f"Scanned {self.total_files_scanned} files, found {self.total_imports_found} import relationships")
        print(f"Identified {len(self.modules)} internal modules")
    
    def tarjan_scc(self):
        """
        Run Tarjan's algorithm to find strongly connected components.
        """
        print("Running Tarjan's algorithm...")
        
        self.index_counter = 0
        self.index = {}
        self.lowlinks = {}
        self.on_stack = {}
        self.stack = []
        self.sccs = []
        
        for module in self.modules:
            if module not in self.index:
                self._strongconnect(module)
                
        self.analysis_end_time = time.time()
        print(f"Found {len(self.sccs)} strongly connected components")
        
    def _strongconnect(self, module: str):
        """
        Recursive helper function for Tarjan's algorithm.
        """
        # Set the depth index for this module
        self.index[module] = self.index_counter
        self.lowlinks[module] = self.index_counter
        self.index_counter += 1
        self.stack.append(module)
        self.on_stack[module] = True
        
        # Consider successors of module
        for successor in self.dependency_graph.get(module, set()):
            if successor not in self.modules:
                continue  # Skip external dependencies
                
            if successor not in self.index:
                # Successor has not yet been visited; recurse on it
                self._strongconnect(successor)
                self.lowlinks[module] = min(self.lowlinks[module], self.lowlinks[successor])
            elif self.on_stack.get(successor, False):
                # Successor is in stack and hence in the current SCC
                self.lowlinks[module] = min(self.lowlinks[module], self.index[successor])
                
        # If module is a root node, pop the stack and create an SCC
        if self.lowlinks[module] == self.index[module]:
            scc = []
            while True:
                w = self.stack.pop()
                self.on_stack[w] = False
                scc.append(w)
                if w == module:
                    break
            self.sccs.append(scc)
    
    def analyze_cycles(self) -> List[Dict[str, Any]]:
        """
        Analyze the detected SCCs to identify cycles and their characteristics.
        """
        cycle_analysis = []
        
        for i, scc in enumerate(self.sccs):
            if len(scc) > 1:  # Only multi-node SCCs represent cycles
                analysis = {
                    'scc_id': i,
                    'modules': scc,
                    'cycle_size': len(scc),
                    'internal_edges': self._count_internal_edges(scc),
                    'external_dependencies': self._get_external_dependencies(scc),
                    'cycle_paths': self._find_cycle_paths(scc)
                }
                cycle_analysis.append(analysis)
                
        return sorted(cycle_analysis, key=lambda x: x['cycle_size'], reverse=True)
    
    def _count_internal_edges(self, scc: List[str]) -> int:
        """Count edges within the SCC."""
        scc_set = set(scc)
        count = 0
        for module in scc:
            for dep in self.dependency_graph.get(module, set()):
                if dep in scc_set:
                    count += 1
        return count
    
    def _get_external_dependencies(self, scc: List[str]) -> Dict[str, Set[str]]:
        """Get external dependencies for each module in the SCC."""
        scc_set = set(scc)
        external_deps = {}
        
        for module in scc:
            external_deps[module] = set()
            for dep in self.dependency_graph.get(module, set()):
                if dep not in scc_set and dep in self.modules:
                    external_deps[module].add(dep)
                    
        return {k: list(v) for k, v in external_deps.items()}
    
    def _find_cycle_paths(self, scc: List[str], max_paths: int = 5) -> List[List[str]]:
        """Find actual cycle paths within the SCC."""
        if len(scc) <= 1:
            return []
            
        scc_set = set(scc)
        paths = []
        
        # For each module in SCC, try to find a path back to itself
        for start_module in scc[:max_paths]:  # Limit to avoid exponential explosion
            path = self._find_path_to_self(start_module, scc_set)
            if path and len(path) > 1:
                paths.append(path)
                
        return paths
    
    def _find_path_to_self(self, start: str, scc_set: Set[str], max_depth: int = 10) -> Optional[List[str]]:
        """Find a path from start back to start within the SCC."""
        visited = set()
        path = []
        
        def dfs(current: str, target: str, depth: int) -> bool:
            if depth > max_depth:
                return False
                
            if current in visited:
                return False
                
            visited.add(current)
            path.append(current)
            
            if current == target and len(path) > 1:
                return True
                
            for dep in self.dependency_graph.get(current, set()):
                if dep in scc_set:
                    if dfs(dep, target, depth + 1):
                        return True
                        
            path.pop()
            visited.remove(current)
            return False
        
        if dfs(start, start, 0):
            return path[:]
        return None
    
    def generate_comprehensive_report(self) -> str:
        """Generate a comprehensive analysis report in Markdown format."""
        cycle_analysis = self.analyze_cycles()
        
        report = []
        report.append("# DAG Health Report - Dependency Cycle Analysis")
        report.append("")
        report.append(f"**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"**Analysis Duration**: {self.analysis_end_time - self.analysis_start_time:.2f} seconds")
        report.append("")
        
        # Executive Summary
        report.append("## Executive Summary")
        report.append("")
        report.append(f"- **Total Files Scanned**: {self.total_files_scanned}")
        report.append(f"- **Total Modules Analyzed**: {len(self.modules)}")
        report.append(f"- **Total Import Relationships**: {self.total_imports_found}")
        report.append(f"- **Strongly Connected Components Found**: {len(self.sccs)}")
        report.append(f"- **Dependency Cycles Detected**: {len(cycle_analysis)}")
        report.append("")
        
        if len(cycle_analysis) == 0:
            report.append("✅ **No dependency cycles detected! Your project has a clean DAG structure.**")
        else:
            report.append(f"⚠️  **{len(cycle_analysis)} dependency cycles detected that require attention.**")
        report.append("")
        
        # Detailed Analysis
        if cycle_analysis:
            report.append("## Detailed Cycle Analysis")
            report.append("")
            
            for i, analysis in enumerate(cycle_analysis, 1):
                report.append(f"### Cycle #{i} (SCC ID: {analysis['scc_id']})")
                report.append("")
                report.append(f"**Modules Involved ({analysis['cycle_size']} modules):**")
                for module in analysis['modules']:
                    report.append(f"- `{module}`")
                report.append("")
                
                report.append(f"**Internal Dependencies**: {analysis['internal_edges']}")
                report.append("")
                
                # External dependencies
                if analysis['external_dependencies']:
                    report.append("**External Dependencies:**")
                    for module, deps in analysis['external_dependencies'].items():
                        if deps:
                            report.append(f"- `{module}` depends on: {', '.join(f'`{dep}`' for dep in deps)}")
                    report.append("")
                
                # Cycle paths
                if analysis['cycle_paths']:
                    report.append("**Detected Cycle Paths:**")
                    for j, path in enumerate(analysis['cycle_paths'], 1):
                        path_str = ' → '.join(f'`{module}`' for module in path)
                        report.append(f"{j}. {path_str}")
                    report.append("")
                
                report.append("---")
                report.append("")
        
        # All SCCs (including single-node)
        report.append("## All Strongly Connected Components")
        report.append("")
        
        single_node_sccs = [scc for scc in self.sccs if len(scc) == 1]
        multi_node_sccs = [scc for scc in self.sccs if len(scc) > 1]
        
        report.append(f"**Single-node SCCs (no cycles)**: {len(single_node_sccs)}")
        report.append(f"**Multi-node SCCs (cycles)**: {len(multi_node_sccs)}")
        report.append("")
        
        if multi_node_sccs:
            report.append("### Multi-node SCCs (Dependency Cycles)")
            for i, scc in enumerate(multi_node_sccs):
                modules_str = ', '.join(f'`{m}`' for m in scc)
                report.append(f"{i+1}. **Size {len(scc)}**: {modules_str}")
        
        report.append("")
        report.append("### Single-node SCCs (No Cycles)")
        report.append("<details>")
        report.append("<summary>Click to expand list of modules without cycles</summary>")
        report.append("")
        
        for scc in single_node_sccs:
            report.append(f"- `{scc[0]}`")
        
        report.append("")
        report.append("</details>")
        report.append("")
        
        # Recommendations
        report.append("## Recommendations")
        report.append("")
        
        if cycle_analysis:
            report.append("### Cycle Breaking Strategies")
            report.append("")
            report.append("1. **Dependency Injection**: Replace direct imports with dependency injection patterns")
            report.append("2. **Interface Segregation**: Extract interfaces to break direct dependencies")
            report.append("3. **Module Splitting**: Split large modules that are central to multiple cycles")
            report.append("4. **Lazy Loading**: Use dynamic imports to break compile-time cycles")
            report.append("5. **Observer Pattern**: Replace bidirectional dependencies with event-driven patterns")
            report.append("")
            
            report.append("### Priority Order for Cycle Breaking")
            report.append("")
            
            for i, analysis in enumerate(cycle_analysis[:10], 1):  # Top 10
                report.append(f"{i}. **Cycle #{i}** ({analysis['cycle_size']} modules, {analysis['internal_edges']} edges)")
                report.append(f"   - Largest impact: breaking this cycle affects {analysis['cycle_size']} modules")
                report.append("")
        
        # Module Statistics
        report.append("## Module Dependency Statistics")
        report.append("")
        
        # Calculate module statistics
        in_degrees = defaultdict(int)
        out_degrees = defaultdict(int)
        
        for module, deps in self.dependency_graph.items():
            out_degrees[module] = len(deps)
            for dep in deps:
                if dep in self.modules:
                    in_degrees[dep] += 1
        
        # Top modules by dependencies
        top_importers = sorted(out_degrees.items(), key=lambda x: x[1], reverse=True)[:10]
        top_imported = sorted(in_degrees.items(), key=lambda x: x[1], reverse=True)[:10]
        
        report.append("### Top Modules by Outgoing Dependencies (Most Importing)")
        for module, count in top_importers:
            report.append(f"- `{module}`: {count} dependencies")
        report.append("")
        
        report.append("### Top Modules by Incoming Dependencies (Most Imported)")
        for module, count in top_imported:
            report.append(f"- `{module}`: {count} dependents")
        report.append("")
        
        # Technical Details
        report.append("## Technical Details")
        report.append("")
        report.append("### Algorithm")
        report.append("- **Method**: Tarjan's Strongly Connected Components Algorithm")
        report.append("- **Time Complexity**: O(V + E) where V = modules, E = import relationships")
        report.append("- **Space Complexity**: O(V)")
        report.append("")
        
        report.append("### Definitions")
        report.append("- **Strongly Connected Component (SCC)**: A maximal set of modules where every module is reachable from every other module in the set")
        report.append("- **Dependency Cycle**: An SCC with more than one module, indicating circular dependencies")
        report.append("- **DAG**: Directed Acyclic Graph - the ideal structure for module dependencies")
        report.append("")
        
        return "\n".join(report)
    
    def run_analysis(self) -> str:
        """Run the complete dependency cycle analysis."""
        print("Starting Tarjan SCC Analysis...")
        
        # Build dependency graph
        self.build_dependency_graph()
        
        # Run Tarjan's algorithm
        self.tarjan_scc()
        
        # Generate report
        report = self.generate_comprehensive_report()
        
        print("Analysis complete!")
        return report
    
    def save_report(self, output_path: str = "G_reporting/DAG_Health.md"):
        """Run analysis and save the report to file."""
        # Ensure output directory exists
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Run analysis and generate report
        report = self.run_analysis()
        
        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
            
        print(f"Report saved to: {output_file}")
        return str(output_file)


def main():
    """Main entry point for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Tarjan's SCC Detector for Python Import Dependencies"
    )
    parser.add_argument(
        "--project-root", 
        default=".", 
        help="Root directory of the project to analyze"
    )
    parser.add_argument(
        "--output", 
        default="G_reporting/DAG_Health.md",
        help="Output file path for the analysis report"
    )
    parser.add_argument(
        "--limit-sccs",
        type=int,
        default=300,
        help="Maximum number of SCCs to analyze in detail"
    )
    
    args = parser.parse_args()
    
    # Create detector and run analysis
    detector = TarjanSCCDetector(args.project_root)
    detector.save_report(args.output)


if __name__ == "__main__":
    main()
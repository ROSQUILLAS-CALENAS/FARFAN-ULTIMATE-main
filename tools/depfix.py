#!/usr/bin/env python3
"""
Comprehensive Dependency Analysis Tool
=====================================

Implements three core algorithms for dependency cycle detection and resolution:
1. Kahn's algorithm for topological sorting
2. Tarjan's algorithm for strongly connected component detection  
3. Eades' greedy feedback arc set algorithm for minimal edge cuts

Analyzes Python import dependencies, detects circular dependencies,
and generates concrete proposals for breaking cycles.
"""

import ast
import argparse
import json
import os
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ImportStatement:
    """Represents a Python import statement with metadata."""
    module: str
    line_number: int
    import_type: str  # 'import', 'from_import'
    alias: Optional[str] = None
    specific_items: List[str] = None

    def __post_init__(self):
        if self.specific_items is None:
            self.specific_items = []


@dataclass
class DependencyEdge:
    """Represents a dependency edge between modules."""
    from_module: str
    to_module: str
    import_statements: List[ImportStatement]
    weight: int = 1  # For feedback arc set algorithm


@dataclass
class StronglyConnectedComponent:
    """Represents a strongly connected component (cycle)."""
    modules: List[str]
    edges: List[DependencyEdge]
    cycle_length: int


@dataclass
class CycleBreakingProposal:
    """Represents a proposal for breaking a dependency cycle."""
    scc: StronglyConnectedComponent
    edges_to_remove: List[DependencyEdge]
    stub_files_to_create: List[str]
    import_statements_to_modify: List[Tuple[str, ImportStatement]]


class ImportExtractor(ast.NodeVisitor):
    """AST visitor to extract import statements from Python files."""
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.imports: List[ImportStatement] = []
    
    def visit_Import(self, node: ast.Import):
        """Visit import statements."""
        for alias in node.names:
            self.imports.append(ImportStatement(
                module=alias.name,
                line_number=node.lineno,
                import_type='import',
                alias=alias.asname
            ))
    
    def visit_ImportFrom(self, node: ast.ImportFrom):
        """Visit from-import statements."""
        if node.module:
            specific_items = []
            for alias in node.names:
                specific_items.append(alias.name)
            
            self.imports.append(ImportStatement(
                module=node.module,
                line_number=node.lineno,
                import_type='from_import',
                specific_items=specific_items
            ))


class DependencyGraph:
    """Represents the dependency graph of Python modules."""
    
    def __init__(self):
        self.nodes: Set[str] = set()
        self.edges: List[DependencyEdge] = []
        self.adjacency_list: Dict[str, List[str]] = defaultdict(list)
        self.edge_map: Dict[Tuple[str, str], DependencyEdge] = {}
    
    def add_edge(self, from_module: str, to_module: str, import_stmt: ImportStatement):
        """Add a dependency edge to the graph."""
        self.nodes.add(from_module)
        self.nodes.add(to_module)
        
        edge_key = (from_module, to_module)
        if edge_key in self.edge_map:
            # Add to existing edge
            self.edge_map[edge_key].import_statements.append(import_stmt)
            self.edge_map[edge_key].weight += 1
        else:
            # Create new edge
            edge = DependencyEdge(from_module, to_module, [import_stmt], 1)
            self.edges.append(edge)
            self.edge_map[edge_key] = edge
            self.adjacency_list[from_module].append(to_module)
    
    def get_in_degree(self) -> Dict[str, int]:
        """Calculate in-degree for each node."""
        in_degree = {node: 0 for node in self.nodes}
        for edge in self.edges:
            in_degree[edge.to_module] += 1
        return in_degree
    
    def get_reverse_graph(self) -> Dict[str, List[str]]:
        """Get reverse adjacency list for Tarjan's algorithm."""
        reverse_graph = defaultdict(list)
        for from_node, to_nodes in self.adjacency_list.items():
            for to_node in to_nodes:
                reverse_graph[to_node].append(from_node)
        return reverse_graph


class KahnsAlgorithm:
    """Implementation of Kahn's algorithm for topological sorting."""
    
    @staticmethod
    def topological_sort(graph: DependencyGraph) -> Tuple[List[str], bool]:
        """
        Perform topological sort using Kahn's algorithm.
        Returns (sorted_nodes, is_dag) where is_dag indicates if graph is acyclic.
        """
        in_degree = graph.get_in_degree()
        queue = deque([node for node, degree in in_degree.items() if degree == 0])
        result = []
        
        while queue:
            current = queue.popleft()
            result.append(current)
            
            # Reduce in-degree of adjacent nodes
            for neighbor in graph.adjacency_list[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # Check if all nodes were processed (DAG) or cycles exist
        is_dag = len(result) == len(graph.nodes)
        return result, is_dag


class TarjansAlgorithm:
    """Implementation of Tarjan's algorithm for strongly connected components."""
    
    def __init__(self):
        self.index_counter = 0
        self.stack = []
        self.lowlinks = {}
        self.index = {}
        self.on_stack = {}
        self.strongly_connected_components = []
    
    def strongly_connected_components_tarjan(self, graph: DependencyGraph) -> List[StronglyConnectedComponent]:
        """Find all strongly connected components using Tarjan's algorithm."""
        self.index_counter = 0
        self.stack = []
        self.lowlinks = {}
        self.index = {}
        self.on_stack = {}
        self.strongly_connected_components = []
        
        for node in graph.nodes:
            if node not in self.index:
                self._strong_connect(node, graph)
        
        # Convert to StronglyConnectedComponent objects
        sccs = []
        for scc_nodes in self.strongly_connected_components:
            if len(scc_nodes) > 1:  # Only interested in cycles
                scc_edges = self._get_scc_edges(scc_nodes, graph)
                sccs.append(StronglyConnectedComponent(
                    modules=scc_nodes,
                    edges=scc_edges,
                    cycle_length=len(scc_nodes)
                ))
        
        return sccs
    
    def _strong_connect(self, node: str, graph: DependencyGraph):
        """Recursive helper for Tarjan's algorithm."""
        self.index[node] = self.index_counter
        self.lowlinks[node] = self.index_counter
        self.index_counter += 1
        self.stack.append(node)
        self.on_stack[node] = True
        
        for successor in graph.adjacency_list[node]:
            if successor not in self.index:
                self._strong_connect(successor, graph)
                self.lowlinks[node] = min(self.lowlinks[node], self.lowlinks[successor])
            elif self.on_stack.get(successor, False):
                self.lowlinks[node] = min(self.lowlinks[node], self.index[successor])
        
        # If node is a root node, pop the stack and create an SCC
        if self.lowlinks[node] == self.index[node]:
            component = []
            while True:
                successor = self.stack.pop()
                self.on_stack[successor] = False
                component.append(successor)
                if successor == node:
                    break
            self.strongly_connected_components.append(component)
    
    def _get_scc_edges(self, scc_nodes: List[str], graph: DependencyGraph) -> List[DependencyEdge]:
        """Get all edges within a strongly connected component."""
        scc_node_set = set(scc_nodes)
        scc_edges = []
        
        for edge in graph.edges:
            if edge.from_module in scc_node_set and edge.to_module in scc_node_set:
                scc_edges.append(edge)
        
        return scc_edges


class EadesGreedyFeedbackArcSet:
    """Implementation of Eades' greedy algorithm for feedback arc set."""
    
    @staticmethod
    def find_feedback_arc_set(scc: StronglyConnectedComponent) -> List[DependencyEdge]:
        """
        Find minimal feedback arc set using Eades' greedy algorithm.
        This is a simplified version that prioritizes edges with lower weight.
        """
        if len(scc.modules) <= 1:
            return []
        
        # Create a copy of the SCC for manipulation
        remaining_edges = scc.edges.copy()
        feedback_edges = []
        
        # Build temporary graph
        temp_graph = DependencyGraph()
        for edge in remaining_edges:
            temp_graph.add_edge(edge.from_module, edge.to_module, edge.import_statements[0])
        
        # Greedy approach: remove edges until no cycles remain
        while True:
            # Check if current graph is acyclic using Kahn's algorithm
            _, is_dag = KahnsAlgorithm.topological_sort(temp_graph)
            if is_dag:
                break
            
            # Find edge with minimum weight to remove
            min_weight_edge = min(remaining_edges, key=lambda e: e.weight)
            feedback_edges.append(min_weight_edge)
            remaining_edges.remove(min_weight_edge)
            
            # Rebuild temporary graph without this edge
            temp_graph = DependencyGraph()
            for edge in remaining_edges:
                temp_graph.add_edge(edge.from_module, edge.to_module, edge.import_statements[0])
        
        return feedback_edges


class DependencyAnalyzer:
    """Main class for analyzing Python module dependencies."""
    
    def __init__(self, root_paths: List[str], exclude_patterns: List[str] = None):
        self.root_paths = [Path(p) for p in root_paths]
        self.exclude_patterns = exclude_patterns or ['__pycache__', '.git', 'venv', '.venv']
        self.graph = DependencyGraph()
        self.module_files: Dict[str, Path] = {}
    
    def analyze(self) -> Dict[str, Any]:
        """Perform complete dependency analysis."""
        logger.info("Starting dependency analysis...")
        
        # Step 1: Extract dependencies
        self._extract_dependencies()
        
        # Step 2: Detect cycles using Tarjan's algorithm
        tarjans = TarjansAlgorithm()
        sccs = tarjans.strongly_connected_components_tarjan(self.graph)
        
        # Step 3: For each SCC, find minimal feedback arc set
        cycle_breaking_proposals = []
        for scc in sccs:
            feedback_edges = EadesGreedyFeedbackArcSet.find_feedback_arc_set(scc)
            proposal = self._generate_cycle_breaking_proposal(scc, feedback_edges)
            cycle_breaking_proposals.append(proposal)
        
        # Step 4: Generate topological order (if possible)
        topo_order, is_dag = KahnsAlgorithm.topological_sort(self.graph)
        
        return {
            'total_modules': len(self.graph.nodes),
            'total_dependencies': len(self.graph.edges),
            'is_dag': is_dag,
            'cycles_detected': len(sccs),
            'topological_order': topo_order if is_dag else None,
            'strongly_connected_components': sccs,
            'cycle_breaking_proposals': cycle_breaking_proposals
        }
    
    def _extract_dependencies(self):
        """Extract dependencies from Python files."""
        logger.info("Extracting dependencies from Python files...")
        
        python_files = []
        for root_path in self.root_paths:
            python_files.extend(self._find_python_files(root_path))
        
        logger.info(f"Found {len(python_files)} Python files")
        
        for file_path in python_files:
            module_name = self._get_module_name(file_path)
            self.module_files[module_name] = file_path
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content, filename=str(file_path))
                extractor = ImportExtractor(str(file_path))
                extractor.visit(tree)
                
                for import_stmt in extractor.imports:
                    # Only consider internal imports
                    if self._is_internal_import(import_stmt.module):
                        self.graph.add_edge(module_name, import_stmt.module, import_stmt)
                        
            except Exception as e:
                logger.warning(f"Error parsing {file_path}: {e}")
    
    def _find_python_files(self, root_path: Path) -> List[Path]:
        """Recursively find all Python files."""
        python_files = []
        
        for item in root_path.rglob('*.py'):
            if not any(pattern in str(item) for pattern in self.exclude_patterns):
                python_files.append(item)
        
        return python_files
    
    def _get_module_name(self, file_path: Path) -> str:
        """Convert file path to module name."""
        # Convert path to module notation
        relative_path = file_path
        for root_path in self.root_paths:
            try:
                relative_path = file_path.relative_to(root_path)
                break
            except ValueError:
                continue
        
        # Convert to module name
        parts = list(relative_path.parts)
        if parts[-1] == '__init__.py':
            parts = parts[:-1]
        elif parts[-1].endswith('.py'):
            parts[-1] = parts[-1][:-3]
        
        return '.'.join(parts)
    
    def _is_internal_import(self, module_name: str) -> bool:
        """Check if import is internal to the analyzed codebase."""
        # Check if this module exists in our module files
        if module_name in self.module_files:
            return True
        
        # Check if it starts with any of our analyzed paths
        for root_path in self.root_paths:
            root_name = root_path.name
            if module_name.startswith(root_name) or module_name.startswith('.'):
                return True
        
        # Check if we can find a file corresponding to this import
        for root_path in self.root_paths:
            # Try to resolve the import path
            potential_paths = [
                root_path / f"{module_name.replace('.', '/')}.py",
                root_path / f"{module_name.replace('.', '/')}" / "__init__.py"
            ]
            for path in potential_paths:
                if path.exists():
                    return True
        
        return False
    
    def _generate_cycle_breaking_proposal(self, scc: StronglyConnectedComponent, 
                                        feedback_edges: List[DependencyEdge]) -> CycleBreakingProposal:
        """Generate concrete proposal for breaking dependency cycle."""
        stub_files = []
        import_modifications = []
        
        for edge in feedback_edges:
            # Propose creating a stub file for the target module
            stub_file = f"{edge.to_module}_stub.py"
            stub_files.append(stub_file)
            
            # Identify specific import statements to modify
            for import_stmt in edge.import_statements:
                import_modifications.append((edge.from_module, import_stmt))
        
        return CycleBreakingProposal(
            scc=scc,
            edges_to_remove=feedback_edges,
            stub_files_to_create=stub_files,
            import_statements_to_modify=import_modifications
        )


class OutputFormatter:
    """Formats analysis results for different output types."""
    
    @staticmethod
    def format_human_readable(analysis_result: Dict[str, Any]) -> str:
        """Format results in human-readable format."""
        output = []
        output.append("=" * 60)
        output.append("DEPENDENCY ANALYSIS REPORT")
        output.append("=" * 60)
        output.append(f"Total Modules: {analysis_result['total_modules']}")
        output.append(f"Total Dependencies: {analysis_result['total_dependencies']}")
        output.append(f"Is DAG: {analysis_result['is_dag']}")
        output.append(f"Cycles Detected: {analysis_result['cycles_detected']}")
        output.append("")
        
        if analysis_result['cycles_detected'] > 0:
            output.append("STRONGLY CONNECTED COMPONENTS (CYCLES):")
            output.append("-" * 40)
            for i, scc in enumerate(analysis_result['strongly_connected_components'], 1):
                output.append(f"Cycle #{i} - {scc.cycle_length} modules:")
                output.append(f"  Modules: {', '.join(scc.modules)}")
                output.append(f"  Edges: {len(scc.edges)}")
                output.append("")
            
            output.append("CYCLE BREAKING PROPOSALS:")
            output.append("-" * 40)
            for i, proposal in enumerate(analysis_result['cycle_breaking_proposals'], 1):
                output.append(f"Proposal #{i}:")
                output.append(f"  Edges to remove: {len(proposal.edges_to_remove)}")
                for edge in proposal.edges_to_remove:
                    output.append(f"    {edge.from_module} -> {edge.to_module}")
                output.append(f"  Stub files to create: {', '.join(proposal.stub_files_to_create)}")
                output.append(f"  Import statements to modify: {len(proposal.import_statements_to_modify)}")
                output.append("")
        
        if analysis_result['topological_order']:
            output.append("TOPOLOGICAL ORDER:")
            output.append("-" * 20)
            for i, module in enumerate(analysis_result['topological_order'], 1):
                output.append(f"{i:2d}. {module}")
        
        return '\n'.join(output)
    
    @staticmethod
    def format_json(analysis_result: Dict[str, Any]) -> str:
        """Format results as JSON."""
        # Convert dataclasses to dictionaries for JSON serialization
        json_result = analysis_result.copy()
        
        # Convert SCCs
        json_result['strongly_connected_components'] = [
            asdict(scc) for scc in analysis_result['strongly_connected_components']
        ]
        
        # Convert proposals
        json_result['cycle_breaking_proposals'] = [
            asdict(proposal) for proposal in analysis_result['cycle_breaking_proposals']
        ]
        
        return json.dumps(json_result, indent=2, default=str)


def main():
    """Main entry point for the dependency analysis tool."""
    parser = argparse.ArgumentParser(
        description='Comprehensive dependency analysis tool for Python projects'
    )
    parser.add_argument(
        'paths', nargs='+',
        help='Root paths to analyze (can specify multiple paths)'
    )
    parser.add_argument(
        '--output', choices=['human', 'json', 'both'], default='human',
        help='Output format (default: human)'
    )
    parser.add_argument(
        '--output-file', type=str,
        help='Output file path (default: stdout)'
    )
    parser.add_argument(
        '--exclude', nargs='*', default=['__pycache__', '.git', 'venv', '.venv'],
        help='Patterns to exclude from analysis'
    )
    parser.add_argument(
        '--verbose', action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate paths
    for path in args.paths:
        if not os.path.exists(path):
            print(f"Error: Path {path} does not exist", file=sys.stderr)
            return 1
    
    try:
        # Perform analysis
        analyzer = DependencyAnalyzer(args.paths, args.exclude)
        result = analyzer.analyze()
        
        # Format output
        if args.output in ['human', 'both']:
            human_output = OutputFormatter.format_human_readable(result)
            if args.output_file and args.output == 'human':
                with open(args.output_file, 'w') as f:
                    f.write(human_output)
            else:
                print(human_output)
        
        if args.output in ['json', 'both']:
            json_output = OutputFormatter.format_json(result)
            if args.output_file:
                json_file = args.output_file
                if args.output == 'both':
                    json_file = args.output_file.replace('.txt', '.json').replace('.md', '.json')
                    if json_file == args.output_file:
                        json_file = args.output_file + '.json'
                with open(json_file, 'w') as f:
                    f.write(json_output)
            else:
                if args.output == 'both':
                    print("\n" + "="*60)
                    print("JSON OUTPUT:")
                    print("="*60)
                print(json_output)
        
        # Exit with appropriate code
        return 0 if result['cycles_detected'] == 0 else 1
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
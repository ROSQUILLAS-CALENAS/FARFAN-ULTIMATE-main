"""
Import graph analysis tests for detecting circular dependencies and phase violations.

These tests use graph analysis techniques to detect complex dependency patterns
that might not be caught by simple import analysis.
"""

import networkx as nx
import pytest
from pathlib import Path
from typing import Dict, List, Set, Tuple
from .test_phase_enforcement import ImportAnalyzer, PhaseViolationError


class ImportGraphAnalyzer:
    """Analyzes import dependencies using graph theory."""
    
    def __init__(self, root_path: str = "canonical_flow"):
        self.import_analyzer = ImportAnalyzer(root_path)
        self.graph = nx.DiGraph()
        self.phase_order = self.import_analyzer.phase_order
        self.phase_index = self.import_analyzer.phase_index
    
    def build_import_graph(self) -> nx.DiGraph:
        """Build a directed graph of module dependencies."""
        self.graph.clear()
        
        for phase in self.phase_order:
            phase_path = Path(self.import_analyzer.root_path) / phase
            if not phase_path.exists():
                continue
            
            for py_file in phase_path.glob("**/*.py"):
                if py_file.name == "__init__.py":
                    continue
                
                module_name = self._file_to_module_name(py_file)
                self.graph.add_node(module_name, phase=phase)
                
                imports = self.import_analyzer.get_imports_from_file(py_file)
                
                for import_name in imports:
                    if import_name.startswith("canonical_flow."):
                        self.graph.add_edge(module_name, import_name)
        
        return self.graph
    
    def _file_to_module_name(self, file_path: Path) -> str:
        """Convert file path to module name."""
        relative_path = file_path.relative_to(self.import_analyzer.root_path.parent)
        module_parts = relative_path.with_suffix('').parts
        return '.'.join(module_parts)
    
    def find_circular_dependencies(self) -> List[List[str]]:
        """Find circular dependencies in the import graph."""
        if not self.graph:
            self.build_import_graph()
        
        try:
            cycles = list(nx.simple_cycles(self.graph))
            return cycles
        except nx.NetworkXError:
            return []
    
    def find_phase_violations_via_graph(self) -> Dict[str, List[Tuple[str, str]]]:
        """Find phase violations using graph analysis."""
        if not self.graph:
            self.build_import_graph()
        
        violations = {}
        
        for edge in self.graph.edges():
            source_module, target_module = edge
            
            source_phase = self.graph.nodes.get(source_module, {}).get('phase')
            target_phase_name = self.import_analyzer.get_phase_from_module(target_module)
            
            if source_phase and target_phase_name:
                source_idx = self.phase_index.get(source_phase, -1)
                target_idx = self.phase_index.get(target_phase_name, -1)
                
                # Violation: importing from earlier phase
                if source_idx > target_idx and source_idx != -1 and target_idx != -1:
                    if source_phase not in violations:
                        violations[source_phase] = []
                    violations[source_phase].append((source_module, target_module))
        
        return violations
    
    def analyze_phase_connectivity(self) -> Dict[str, Dict[str, int]]:
        """Analyze connectivity patterns between phases."""
        if not self.graph:
            self.build_import_graph()
        
        connectivity = {}
        
        for source_phase in self.phase_order:
            connectivity[source_phase] = {}
            for target_phase in self.phase_order:
                connectivity[source_phase][target_phase] = 0
        
        for edge in self.graph.edges():
            source_module, target_module = edge
            
            source_phase = self.graph.nodes.get(source_module, {}).get('phase')
            target_phase_name = self.import_analyzer.get_phase_from_module(target_module)
            
            if source_phase and target_phase_name and target_phase_name in self.phase_index:
                connectivity[source_phase][target_phase_name] += 1
        
        return connectivity
    
    def get_strongly_connected_components(self) -> List[List[str]]:
        """Find strongly connected components (potential circular dependencies)."""
        if not self.graph:
            self.build_import_graph()
        
        return list(nx.strongly_connected_components(self.graph))


@pytest.fixture
def graph_analyzer():
    """Create an ImportGraphAnalyzer instance."""
    return ImportGraphAnalyzer()


@pytest.mark.architecture
@pytest.mark.phase_enforcement
class TestImportGraphAnalysis:
    """Architecture fitness functions using graph analysis."""
    
    def test_no_circular_dependencies(self, graph_analyzer):
        """Test that there are no circular dependencies in the import graph."""
        cycles = graph_analyzer.find_circular_dependencies()
        
        if cycles:
            error_msg = f"Found {len(cycles)} circular dependencies:\n"
            for i, cycle in enumerate(cycles[:5]):  # Show first 5 cycles
                error_msg += f"  Cycle {i+1}: {' → '.join(cycle)} → {cycle[0]}\n"
            if len(cycles) > 5:
                error_msg += f"  ... and {len(cycles) - 5} more cycles\n"
            
            raise PhaseViolationError(error_msg)
    
    def test_phase_violations_via_graph_analysis(self, graph_analyzer):
        """Test for phase violations using graph-based analysis."""
        violations = graph_analyzer.find_phase_violations_via_graph()
        
        if violations:
            error_msg = "Phase violations detected via graph analysis:\n"
            for phase, phase_violations in violations.items():
                error_msg += f"\n{phase}:\n"
                for source_module, target_module in phase_violations:
                    error_msg += f"  {source_module} → {target_module}\n"
            
            raise PhaseViolationError(error_msg)
    
    def test_phase_connectivity_matrix(self, graph_analyzer):
        """Test phase connectivity follows expected patterns."""
        connectivity = graph_analyzer.analyze_phase_connectivity()
        
        violations = []
        
        for source_phase, targets in connectivity.items():
            source_idx = graph_analyzer.phase_index.get(source_phase, -1)
            
            for target_phase, count in targets.items():
                target_idx = graph_analyzer.phase_index.get(target_phase, -1)
                
                # Check for backward dependencies
                if count > 0 and source_idx > target_idx and source_idx != -1 and target_idx != -1:
                    violations.append(f"{source_phase} → {target_phase} ({count} dependencies)")
        
        if violations:
            error_msg = "Phase connectivity violations:\n"
            for violation in violations:
                error_msg += f"  {violation}\n"
            
            raise PhaseViolationError(error_msg)
    
    def test_strongly_connected_components_analysis(self, graph_analyzer):
        """Test for strongly connected components that might indicate design issues."""
        components = graph_analyzer.get_strongly_connected_components()
        
        # Filter out single-node components (these are normal)
        multi_node_components = [comp for comp in components if len(comp) > 1]
        
        if multi_node_components:
            error_msg = f"Found {len(multi_node_components)} strongly connected components:\n"
            for i, component in enumerate(multi_node_components):
                error_msg += f"  Component {i+1}: {list(component)}\n"
            
            # This might be a warning rather than an error in some cases
            pytest.fail(error_msg)
    
    def test_acyclic_phase_flow(self, graph_analyzer):
        """Test that the overall phase flow is acyclic."""
        # Build a phase-level graph
        phase_graph = nx.DiGraph()
        
        connectivity = graph_analyzer.analyze_phase_connectivity()
        
        for source_phase, targets in connectivity.items():
            for target_phase, count in targets.items():
                if count > 0:
                    phase_graph.add_edge(source_phase, target_phase, weight=count)
        
        # Check for cycles at the phase level
        try:
            cycles = list(nx.simple_cycles(phase_graph))
            if cycles:
                error_msg = f"Found {len(cycles)} phase-level cycles:\n"
                for cycle in cycles:
                    error_msg += f"  {' → '.join(cycle)} → {cycle[0]}\n"
                
                raise PhaseViolationError(error_msg)
        except nx.NetworkXError:
            pass  # No cycles found
    
    def test_forward_dependency_flow(self, graph_analyzer):
        """Test that dependencies generally flow forward through phases."""
        connectivity = graph_analyzer.analyze_phase_connectivity()
        
        forward_deps = 0
        backward_deps = 0
        same_phase_deps = 0
        
        for source_phase, targets in connectivity.items():
            source_idx = graph_analyzer.phase_index.get(source_phase, -1)
            
            for target_phase, count in targets.items():
                target_idx = graph_analyzer.phase_index.get(target_phase, -1)
                
                if count > 0 and source_idx != -1 and target_idx != -1:
                    if source_idx < target_idx:
                        forward_deps += count
                    elif source_idx > target_idx:
                        backward_deps += count
                    else:
                        same_phase_deps += count
        
        total_deps = forward_deps + backward_deps + same_phase_deps
        
        if total_deps > 0:
            backward_ratio = backward_deps / total_deps
            
            # Allow some tolerance for backward dependencies (e.g., shared utilities)
            if backward_ratio > 0.1:  # More than 10% backward dependencies
                pytest.fail(
                    f"Too many backward dependencies: {backward_deps}/{total_deps} "
                    f"({backward_ratio:.1%}). Forward: {forward_deps}, "
                    f"Same phase: {same_phase_deps}"
                )


if __name__ == "__main__":
    # Allow running this file directly for debugging
    analyzer = ImportGraphAnalyzer()
    analyzer.build_import_graph()
    
    print(f"Import graph has {analyzer.graph.number_of_nodes()} nodes and {analyzer.graph.number_of_edges()} edges")
    
    cycles = analyzer.find_circular_dependencies()
    print(f"Found {len(cycles)} circular dependencies")
    
    violations = analyzer.find_phase_violations_via_graph()
    total_violations = sum(len(v) for v in violations.values())
    print(f"Found {total_violations} phase violations")
    
    connectivity = analyzer.analyze_phase_connectivity()
    print("\nPhase connectivity matrix:")
    for source_phase, targets in connectivity.items():
        for target_phase, count in targets.items():
            if count > 0:
                print(f"  {source_phase} → {target_phase}: {count}")
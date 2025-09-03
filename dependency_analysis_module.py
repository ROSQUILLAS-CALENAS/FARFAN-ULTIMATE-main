#!/usr/bin/env python3
"""
Dependency Analysis Module for Pipeline Components

Scans all 336 canonical pipeline components to extract interdependencies 
by parsing import statements, function calls, and data flow patterns.
Generates both Mermaid DAG diagrams and JSON dependency graphs with
canonical phase lane assignments and cycle detection validation.
"""

import ast
import json
import os
import re
import sys
# # # from collections import defaultdict, deque  # Module not found  # Module not found  # Module not found
# # # from dataclasses import dataclass, asdict  # Module not found  # Module not found  # Module not found
# # # from pathlib import Path  # Module not found  # Module not found  # Module not found
# # # from typing import Dict, List, Set, Tuple, Optional, Any  # Module not found  # Module not found  # Module not found
# # # from datetime import datetime  # Module not found  # Module not found  # Module not found


@dataclass
class ComponentInfo:
    """Information about a pipeline component."""
    path: str
    name: str
    canonical_phase: str
    imports: List[str]
    function_calls: List[str]
    data_inputs: List[str]
    data_outputs: List[str]
    dependencies: List[str]
    dependents: List[str]


@dataclass
class DependencyEdge:
    """Represents a dependency relationship between components."""
    source: str
    target: str
    edge_type: str  # 'import', 'function_call', 'data_flow'
    confidence: float
    metadata: Dict[str, Any]


class DependencyAnalyzer:
    """Main class for analyzing pipeline component dependencies."""
    
    # Canonical phase mapping based on directory structure
    CANONICAL_PHASES = {
        'I_ingestion_preparation': 'I_ingestion',
        'X_context_construction': 'X_context',
        'K_knowledge_extraction': 'K_knowledge',
        'A_analysis_nlp': 'A_analysis',
        'L_classification_evaluation': 'L_classification',
        'O_orchestration_control': 'O_orchestration',
        'R_search_retrieval': 'R_retrieval',
        'S_synthesis_output': 'S_synthesis',
        'T_integration_storage': 'T_storage',
        'G_aggregation_reporting': 'G_reporting',
        'calibration': 'calibration',
        'mathematical_enhancers': 'mathematical',
        'analysis': 'analysis_core',
        'evaluation': 'evaluation_core',
        'knowledge': 'knowledge_core',
        'ingestion': 'ingestion_core'
    }
    
    def __init__(self, root_dir: str = "."):
        self.root_dir = Path(root_dir)
        self.components: Dict[str, ComponentInfo] = {}
        self.dependency_graph: Dict[str, List[DependencyEdge]] = defaultdict(list)
        self.phase_components: Dict[str, List[str]] = defaultdict(list)
        self.cycles_detected: List[List[str]] = []
        
    def scan_components(self) -> None:
        """Scan all Python files to identify pipeline components."""
        print("Scanning components...")
        
        # Load canonical order if available
        canonical_manifest = self.root_dir / "canonical_order_manifest.json"
        canonical_order = []
        if canonical_manifest.exists():
            with open(canonical_manifest, 'r') as f:
                data = json.load(f)
                canonical_order = data.get('sequence', [])
        
        # Find all Python files
        python_files = list(self.root_dir.glob("**/*.py"))
        
        for py_file in python_files:
            try:
                if self._should_skip_file(py_file):
                    continue
                    
                component_info = self._analyze_component(py_file)
                if component_info:
                    self.components[component_info.name] = component_info
                    self.phase_components[component_info.canonical_phase].append(component_info.name)
                    
            except Exception as e:
                print(f"Error analyzing {py_file}: {e}")
                
        print(f"Found {len(self.components)} components across {len(self.phase_components)} phases")
        
    def _should_skip_file(self, py_file: Path) -> bool:
        """Determine if a file should be skipped during analysis."""
        skip_patterns = [
            "__pycache__",
            ".git",
            "venv",
            "env",
            ".pytest_cache",
            "test_",
            "_test.py",
            ".egg-info"
        ]
        
        file_str = str(py_file)
        return any(pattern in file_str for pattern in skip_patterns)
        
    def _analyze_component(self, py_file: Path) -> Optional[ComponentInfo]:
        """Analyze a single Python file to extract component information."""
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Parse AST
            tree = ast.parse(content)
            
            # Extract component metadata
            relative_path = str(py_file.relative_to(self.root_dir))
            component_name = py_file.stem
            canonical_phase = self._determine_canonical_phase(relative_path)
            
            # Extract imports
            imports = self._extract_imports(tree)
            
            # Extract function calls
            function_calls = self._extract_function_calls(tree)
            
            # Extract data flow patterns
            data_inputs, data_outputs = self._extract_data_flow(tree, content)
            
            return ComponentInfo(
                path=relative_path,
                name=component_name,
                canonical_phase=canonical_phase,
                imports=imports,
                function_calls=function_calls,
                data_inputs=data_inputs,
                data_outputs=data_outputs,
                dependencies=[],
                dependents=[]
            )
            
        except Exception as e:
            print(f"Failed to analyze {py_file}: {e}")
            return None
            
    def _determine_canonical_phase(self, file_path: str) -> str:
        """Determine the canonical phase for a component based on its path."""
        for phase_dir, phase_name in self.CANONICAL_PHASES.items():
            if phase_dir in file_path:
                return phase_name
                
        # Default phase classification based on path structure
        parts = file_path.split('/')
        if len(parts) > 1:
            if 'canonical_flow' in parts:
                idx = parts.index('canonical_flow')
                if idx + 1 < len(parts):
                    next_part = parts[idx + 1]
                    return self.CANONICAL_PHASES.get(next_part, 'unknown')
            
        # Fallback classification
        if 'ingestion' in file_path or 'loader' in file_path:
            return 'I_ingestion'
        elif 'context' in file_path:
            return 'X_context'
        elif 'knowledge' in file_path or 'embedding' in file_path:
            return 'K_knowledge'
        elif 'analysis' in file_path or 'nlp' in file_path:
            return 'A_analysis'
        elif 'classification' in file_path or 'scoring' in file_path:
            return 'L_classification'
        elif 'orchestration' in file_path or 'orchestrator' in file_path:
            return 'O_orchestration'
        elif 'retrieval' in file_path or 'search' in file_path:
            return 'R_retrieval'
        elif 'synthesis' in file_path or 'output' in file_path:
            return 'S_synthesis'
        elif 'storage' in file_path or 'integration' in file_path:
            return 'T_storage'
        elif 'aggregation' in file_path or 'reporting' in file_path:
            return 'G_reporting'
        else:
            return 'unknown'
    
    def _extract_imports(self, tree: ast.AST) -> List[str]:
# # #         """Extract import statements from AST."""  # Module not found  # Module not found  # Module not found
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    base_module = node.module
                    for alias in node.names:
                        imports.append(f"{base_module}.{alias.name}")
                        
        return imports
    
    def _extract_function_calls(self, tree: ast.AST) -> List[str]:
# # #         """Extract function calls from AST."""  # Module not found  # Module not found  # Module not found
        function_calls = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    function_calls.append(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    if isinstance(node.func.value, ast.Name):
                        function_calls.append(f"{node.func.value.id}.{node.func.attr}")
                        
        return list(set(function_calls))
    
    def _extract_data_flow(self, tree: ast.AST, content: str) -> Tuple[List[str], List[str]]:
# # #         """Extract data input/output patterns from code."""  # Module not found  # Module not found  # Module not found
        inputs = []
        outputs = []
        
        # Look for common patterns in comments and docstrings
        input_patterns = [
            r'input[s]?[:\s]*([^,\n]+)',
            r'expects?[:\s]*([^,\n]+)',
            r'takes?[:\s]*([^,\n]+)',
            r'receives?[:\s]*([^,\n]+)'
        ]
        
        output_patterns = [
            r'output[s]?[:\s]*([^,\n]+)',
            r'returns?[:\s]*([^,\n]+)',
            r'produces?[:\s]*([^,\n]+)',
            r'generates?[:\s]*([^,\n]+)'
        ]
        
        for pattern in input_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                inputs.append(match.group(1).strip())
                
        for pattern in output_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                outputs.append(match.group(1).strip())
                
        return inputs, outputs
    
    def build_dependency_graph(self) -> None:
        """Build the dependency graph by analyzing relationships between components."""
        print("Building dependency graph...")
        
        for component_name, component in self.components.items():
            # Analyze import-based dependencies
            self._analyze_import_dependencies(component)
            
            # Analyze function call dependencies
            self._analyze_function_call_dependencies(component)
            
            # Analyze data flow dependencies
            self._analyze_data_flow_dependencies(component)
            
        # Update bidirectional relationships
        self._update_dependency_relationships()
        
    def _analyze_import_dependencies(self, component: ComponentInfo) -> None:
        """Analyze dependencies based on import statements."""
        for import_stmt in component.imports:
            # Check if import corresponds to another component
            potential_deps = self._resolve_import_to_components(import_stmt)
            
            for dep_name in potential_deps:
                if dep_name in self.components and dep_name != component.name:
                    edge = DependencyEdge(
                        source=dep_name,
                        target=component.name,
                        edge_type='import',
                        confidence=0.9,
                        metadata={'import_statement': import_stmt}
                    )
                    self.dependency_graph[dep_name].append(edge)
    
    def _analyze_function_call_dependencies(self, component: ComponentInfo) -> None:
        """Analyze dependencies based on function calls."""
        for func_call in component.function_calls:
            # Look for function calls that might reference other components
            potential_deps = self._resolve_function_call_to_components(func_call)
            
            for dep_name in potential_deps:
                if dep_name in self.components and dep_name != component.name:
                    edge = DependencyEdge(
                        source=dep_name,
                        target=component.name,
                        edge_type='function_call',
                        confidence=0.7,
                        metadata={'function_call': func_call}
                    )
                    self.dependency_graph[dep_name].append(edge)
    
    def _analyze_data_flow_dependencies(self, component: ComponentInfo) -> None:
        """Analyze dependencies based on data flow patterns."""
        # Match data outputs of other components with inputs of this component
        for input_pattern in component.data_inputs:
            for other_name, other_component in self.components.items():
                if other_name != component.name:
                    for output_pattern in other_component.data_outputs:
                        if self._patterns_match(input_pattern, output_pattern):
                            edge = DependencyEdge(
                                source=other_name,
                                target=component.name,
                                edge_type='data_flow',
                                confidence=0.6,
                                metadata={
                                    'input_pattern': input_pattern,
                                    'output_pattern': output_pattern
                                }
                            )
                            self.dependency_graph[other_name].append(edge)
    
    def _resolve_import_to_components(self, import_stmt: str) -> List[str]:
        """Resolve an import statement to potential component names."""
        components = []
        
# # #         # Extract potential component names from import  # Module not found  # Module not found  # Module not found
        parts = import_stmt.split('.')
        for part in parts:
            if part in self.components:
                components.append(part)
                
        # Check for relative imports within canonical_flow
        if 'canonical_flow' in import_stmt:
            for comp_name in self.components:
                if comp_name in import_stmt:
                    components.append(comp_name)
                    
        return components
    
    def _resolve_function_call_to_components(self, func_call: str) -> List[str]:
        """Resolve a function call to potential component names."""
        components = []
        
        # Extract potential component references
        parts = func_call.split('.')
        for part in parts:
            if part in self.components:
                components.append(part)
                
        return components
    
    def _patterns_match(self, input_pattern: str, output_pattern: str) -> bool:
        """Check if input and output patterns match."""
        # Simple keyword matching - could be enhanced with NLP
        input_words = set(input_pattern.lower().split())
        output_words = set(output_pattern.lower().split())
        
        common_words = input_words & output_words
        return len(common_words) > 0
    
    def _update_dependency_relationships(self) -> None:
        """Update bidirectional dependency relationships in components."""
        for source_name, edges in self.dependency_graph.items():
            if source_name not in self.components:
                continue
                
            for edge in edges:
                target_name = edge.target
                
                # Skip if target component doesn't exist
                if target_name not in self.components:
                    continue
                
                # Update source component's dependents
                if target_name not in self.components[source_name].dependents:
                    self.components[source_name].dependents.append(target_name)
                    
                # Update target component's dependencies
                if source_name not in self.components[target_name].dependencies:
                    self.components[target_name].dependencies.append(source_name)
    
    def detect_cycles(self) -> None:
        """Detect cycles in the dependency graph to ensure DAG properties."""
        print("Detecting cycles...")
        
        # Use DFS-based cycle detection
        visited = set()
        rec_stack = set()
        self.cycles_detected = []
        
        def dfs_cycle_detection(node: str, path: List[str]) -> bool:
            if node not in self.components:
                return False
                
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            # Visit all dependent nodes
            for dependent in self.components[node].dependents:
                if dependent not in self.components:
                    continue
                    
                if dependent not in visited:
                    if dfs_cycle_detection(dependent, path.copy()):
                        return True
                elif dependent in rec_stack:
                    # Found a cycle
                    try:
                        cycle_start = path.index(dependent)
                        cycle = path[cycle_start:] + [dependent]
                        self.cycles_detected.append(cycle)
                        return True
                    except ValueError:
                        # Handle case where dependent is not in current path
                        cycle = path + [dependent]
                        self.cycles_detected.append(cycle)
                        return True
            
            rec_stack.remove(node)
            return False
        
        for component_name in self.components:
            if component_name not in visited:
                dfs_cycle_detection(component_name, [])
        
        if self.cycles_detected:
            print(f"WARNING: Found {len(self.cycles_detected)} cycles in the dependency graph!")
        else:
            print("✓ No cycles detected - DAG properties maintained")
    
    def generate_mermaid_diagram(self) -> str:
        """Generate a Mermaid DAG diagram."""
        print("Generating Mermaid diagram...")
        
        mermaid = ["graph TD"]
        
        # Add phase subgraphs
        phase_colors = {
            'I_ingestion': '#e1f5fe',
            'X_context': '#f3e5f5',
            'K_knowledge': '#e8f5e8',
            'A_analysis': '#fff3e0',
            'L_classification': '#fce4ec',
            'O_orchestration': '#e0f2f1',
            'R_retrieval': '#f1f8e9',
            'S_synthesis': '#fff8e1',
            'T_storage': '#e8eaf6',
            'G_reporting': '#fafafa',
            'mathematical': '#e3f2fd',
            'calibration': '#f9fbe7',
            'unknown': '#ffebee'
        }
        
        for phase, components in self.phase_components.items():
            if components:
                color = phase_colors.get(phase, '#f5f5f5')
                mermaid.append(f"    subgraph {phase}[{phase.upper()}]")
                mermaid.append(f"        style {phase} fill:{color}")
                
                for comp_name in components:
                    comp_id = comp_name.replace('-', '_').replace('.', '_')
                    mermaid.append(f"        {comp_id}[\"{comp_name}\"]")
                
                mermaid.append("    end")
        
        # Add dependency edges
        mermaid.append("")
        for source_name, edges in self.dependency_graph.items():
            source_id = source_name.replace('-', '_').replace('.', '_')
            
            for edge in edges:
                target_id = edge.target.replace('-', '_').replace('.', '_')
                
                # Different arrow styles for different edge types
                if edge.edge_type == 'import':
                    arrow = "-->|import|"
                elif edge.edge_type == 'function_call':
                    arrow = "-.->|call|"
                elif edge.edge_type == 'data_flow':
                    arrow = "==>|data|"
                else:
                    arrow = "-->"
                
                mermaid.append(f"    {source_id} {arrow} {target_id}")
        
        # Add cycle warnings as comments (limit to first 20 for readability)
        if self.cycles_detected:
            mermaid.append("")
            mermaid.append("    %% CYCLES DETECTED (showing first 20):")
            for i, cycle in enumerate(self.cycles_detected[:20]):
                cycle_str = " -> ".join(cycle)
                mermaid.append(f"    %% Cycle {i+1}: {cycle_str}")
            
            if len(self.cycles_detected) > 20:
                remaining = len(self.cycles_detected) - 20
                mermaid.append(f"    %% ... and {remaining} more cycles")
        
        return "\n".join(mermaid)
    
    def generate_json_structure(self) -> Dict[str, Any]:
        """Generate JSON dependency graph structure."""
        print("Generating JSON structure...")
        
        # Convert components to serializable format
        components_dict = {}
        for name, component in self.components.items():
            components_dict[name] = asdict(component)
        
        # Convert dependency edges to serializable format
        edges_dict = {}
        for source, edges in self.dependency_graph.items():
            edges_dict[source] = [asdict(edge) for edge in edges]
        
        # Phase statistics
        phase_stats = {}
        for phase, components in self.phase_components.items():
            phase_stats[phase] = {
                'component_count': len(components),
                'components': components
            }
        
        # Build final structure
        json_structure = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'total_components': len(self.components),
                'total_phases': len(self.phase_components),
                'total_edges': sum(len(edges) for edges in self.dependency_graph.values()),
                'cycles_detected': len(self.cycles_detected),
                'dag_valid': len(self.cycles_detected) == 0
            },
            'canonical_phases': {
                phase: {
                    'lane_name': phase,
                    'components': components,
                    'component_count': len(components)
                }
                for phase, components in self.phase_components.items()
            },
            'components': components_dict,
            'dependency_edges': edges_dict,
            'cycles': self.cycles_detected,
            'phase_statistics': phase_stats
        }
        
        return json_structure
    
    def save_outputs(self) -> None:
        """Save both Mermaid diagram and JSON structure to files."""
        print("Saving outputs...")
        
        # Generate outputs
        mermaid_content = self.generate_mermaid_diagram()
        json_structure = self.generate_json_structure()
        
        # Save Mermaid diagram
        mermaid_file = self.root_dir / "pipeline_dependency_diagram.mmd"
        with open(mermaid_file, 'w', encoding='utf-8') as f:
            f.write(mermaid_content)
        print(f"✓ Mermaid diagram saved to: {mermaid_file}")
        
        # Save JSON structure
        json_file = self.root_dir / "dependencies.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_structure, f, indent=2, ensure_ascii=False)
        print(f"✓ JSON dependency graph saved to: {json_file}")
        
        # Generate summary report
        self._generate_summary_report(json_structure)
    
    def _generate_summary_report(self, json_structure: Dict[str, Any]) -> None:
        """Generate a human-readable summary report."""
        report_lines = [
            "# Pipeline Dependency Analysis Report",
            f"Generated: {json_structure['metadata']['generated_at']}",
            "",
            "## Summary Statistics",
            f"- Total Components: {json_structure['metadata']['total_components']}",
            f"- Total Phases: {json_structure['metadata']['total_phases']}",
            f"- Total Dependencies: {json_structure['metadata']['total_edges']}",
            f"- Cycles Detected: {json_structure['metadata']['cycles_detected']}",
            f"- DAG Valid: {json_structure['metadata']['dag_valid']}",
            "",
            "## Phase Distribution"
        ]
        
        for phase, info in json_structure['canonical_phases'].items():
            report_lines.append(f"- **{phase}**: {info['component_count']} components")
        
        if self.cycles_detected:
            report_lines.extend([
                "",
                "## ⚠️  Detected Cycles",
                "The following cycles were detected in the dependency graph:"
            ])
            
            for i, cycle in enumerate(self.cycles_detected):
                cycle_str = " → ".join(cycle)
                report_lines.append(f"{i+1}. {cycle_str}")
        
        report_content = "\n".join(report_lines)
        
        report_file = self.root_dir / "dependency_analysis_report.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        print(f"✓ Summary report saved to: {report_file}")
    
    def run_full_analysis(self) -> None:
        """Run the complete dependency analysis pipeline."""
        print("=" * 60)
        print("PIPELINE DEPENDENCY ANALYSIS")
        print("=" * 60)
        
        try:
            # Step 1: Scan components
            self.scan_components()
            
            # Step 2: Build dependency graph
            self.build_dependency_graph()
            
            # Step 3: Detect cycles
            self.detect_cycles()
            
            # Step 4: Save outputs
            self.save_outputs()
            
            print("=" * 60)
            print("✓ ANALYSIS COMPLETE")
            print("=" * 60)
            
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main entry point for the dependency analysis module."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze pipeline component dependencies")
    parser.add_argument("--root", "-r", default=".", help="Root directory to analyze")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG)
    
    analyzer = DependencyAnalyzer(root_dir=args.root)
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()
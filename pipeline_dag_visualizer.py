#!/usr/bin/env python3
"""
DAG visualization system for pipeline components.
Generates PNG and SVG visualizations of the pipeline dependency graph.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional
import subprocess
from dataclasses import dataclass
from collections import defaultdict, deque


@dataclass 
class DAGNode:
    """Represents a node in the DAG visualization."""
    name: str
    code: str
    phase: str
    dependencies: List[str]
    description: str
    enabled: bool
    level: int = 0  # For layering
    x: float = 0.0  # Layout coordinates
    y: float = 0.0


@dataclass
class DAGEdge:
    """Represents an edge in the DAG visualization."""
    from_node: str
    to_node: str
    from_code: str
    to_code: str


class PipelineDAGVisualizer:
    """DAG visualization system for pipeline components."""
    
    def __init__(self, index_path: str = "pipeline_index.json"):
        self.index_path = Path(index_path)
        self.phase_colors = {
            'ingestion_preparation': '#FF6B6B',      # Red
            'context_construction': '#4ECDC4',       # Teal
            'knowledge_extraction': '#45B7D1',       # Blue
            'analysis_nlp': '#96CEB4',               # Green
            'classification_evaluation': '#FFEAA7',   # Yellow
            'orchestration_control': '#DDA0DD',       # Plum
            'search_retrieval': '#98D8C8',           # Mint
            'synthesis_output': '#F7DC6F',           # Light yellow
            'aggregation_reporting': '#BB8FCE',      # Light purple
            'integration_storage': '#85C1E9',        # Light blue
            'unclassified': '#BDC3C7'                # Gray
        }
        
        self.phase_order = [
            'ingestion_preparation',
            'context_construction', 
            'knowledge_extraction',
            'analysis_nlp',
            'classification_evaluation',
            'orchestration_control',
            'search_retrieval',
            'synthesis_output',
            'aggregation_reporting',
            'integration_storage',
            'unclassified'
        ]
    
    def load_index(self) -> Dict[str, Any]:
        """Load pipeline index."""
        if not self.index_path.exists():
            raise FileNotFoundError(f"Index file not found: {self.index_path}")
        
        with open(self.index_path, 'r') as f:
            return json.load(f)
    
    def build_dag(self, index_data: Dict[str, Any]) -> Tuple[List[DAGNode], List[DAGEdge]]:
        """Build DAG nodes and edges from index data."""
        nodes = []
        edges = []
        
        # Create code to component mapping
        code_to_comp = {}
        name_to_comp = {}
        
        for comp in index_data.get("components", []):
            code = comp.get("code", comp["name"])
            code_to_comp[code] = comp
            name_to_comp[comp["name"]] = comp
        
        # Create nodes
        for comp in index_data["components"]:
            node = DAGNode(
                name=comp["name"],
                code=comp.get("code", comp["name"]),
                phase=comp.get("phase", "unclassified"),
                dependencies=comp.get("dependencies", []),
                description=comp.get("description", ""),
                enabled=comp.get("enabled", True)
            )
            nodes.append(node)
        
        # Create edges from dependencies
        for node in nodes:
            for dep_code in node.dependencies:
                if dep_code in code_to_comp:
                    edge = DAGEdge(
                        from_node=code_to_comp[dep_code]["name"],
                        to_node=node.name,
                        from_code=dep_code,
                        to_code=node.code
                    )
                    edges.append(edge)
        
        return nodes, edges
    
    def calculate_layout(self, nodes: List[DAGNode], edges: List[DAGEdge]) -> None:
        """Calculate node positions for visualization layout."""
        # Group nodes by phase
        phase_groups = defaultdict(list)
        for node in nodes:
            phase_groups[node.phase].append(node)
        
        # Calculate levels (topological ordering)
        self._calculate_topological_levels(nodes, edges)
        
        # Position nodes by phase and level
        y_offset = 0
        phase_height = 150
        
        for phase in self.phase_order:
            if phase not in phase_groups:
                continue
            
            phase_nodes = phase_groups[phase]
            
            # Sort nodes in phase by level, then by code
            phase_nodes.sort(key=lambda n: (n.level, n.code))
            
            # Position nodes horizontally within phase
            x_spacing = 200
            x_start = 50
            
            for i, node in enumerate(phase_nodes):
                node.x = x_start + (i * x_spacing)
                node.y = y_offset + 75
            
            y_offset += phase_height
    
    def _calculate_topological_levels(self, nodes: List[DAGNode], edges: List[DAGEdge]) -> None:
        """Calculate topological levels for nodes."""
        # Build adjacency lists
        predecessors = defaultdict(set)
        successors = defaultdict(set)
        
        for edge in edges:
            predecessors[edge.to_node].add(edge.from_node)
            successors[edge.from_node].add(edge.to_node)
        
        # Find entry points (nodes with no dependencies)
        entry_points = []
        for node in nodes:
            if not predecessors[node.name]:
                entry_points.append(node.name)
                node.level = 0
        
        # BFS to assign levels
        queue = deque(entry_points)
        processed = set()
        
        while queue:
            current = queue.popleft()
            if current in processed:
                continue
            
            processed.add(current)
            current_node = next(n for n in nodes if n.name == current)
            
            # Process successors
            for successor in successors[current]:
                successor_node = next(n for n in nodes if n.name == successor)
                successor_node.level = max(successor_node.level, current_node.level + 1)
                
                # Add to queue if all predecessors are processed
                if all(pred in processed for pred in predecessors[successor]):
                    queue.append(successor)
    
    def generate_graphviz_dot(self, nodes: List[DAGNode], edges: List[DAGEdge]) -> str:
        """Generate Graphviz DOT format for the DAG."""
        dot_lines = [
            "digraph PipelineDAG {",
            "  rankdir=TB;",
            "  node [shape=box, style=filled];",
            "  edge [color=gray50];",
            ""
        ]
        
        # Add subgraphs for phases
        for phase in self.phase_order:
            phase_nodes = [n for n in nodes if n.phase == phase]
            if not phase_nodes:
                continue
            
            color = self.phase_colors[phase]
            phase_label = phase.replace('_', ' ').title()
            
            dot_lines.extend([
                f"  subgraph cluster_{phase} {{",
                f"    label=\"{phase_label}\";",
                f"    style=filled;",
                f"    fillcolor=\"{color}30\";",
                f"    color=\"{color}\";",
                ""
            ])
            
            # Add nodes in this phase
            for node in phase_nodes:
                status_style = "" if node.enabled else ", style=\"filled,dashed\""
                tooltip = node.description.replace('"', '\\"')
                
                dot_lines.append(
                    f"    \"{node.name}\" [label=\"{node.code}\\n{node.name}\", "
                    f"fillcolor=\"{color}\", tooltip=\"{tooltip}\"{status_style}];"
                )
            
            dot_lines.extend([
                "  }",
                ""
            ])
        
        # Add edges
        dot_lines.append("  // Dependencies")
        for edge in edges:
            dot_lines.append(f"  \"{edge.from_node}\" -> \"{edge.to_node}\";")
        
        dot_lines.append("}")
        
        return "\n".join(dot_lines)
    
    def generate_mermaid_diagram(self, nodes: List[DAGNode], edges: List[DAGEdge]) -> str:
        """Generate Mermaid diagram format for the DAG."""
        mermaid_lines = [
            "graph TD",
            ""
        ]
        
        # Add nodes with styling
        for node in nodes:
            style_class = node.phase.replace('_', '')
            status_marker = "❌" if not node.enabled else ""
            label = f"{node.code}<br/>{node.name}{status_marker}"
            
            mermaid_lines.append(f"  {node.code}[\"{label}\"]")
        
        mermaid_lines.append("")
        
        # Add edges
        for edge in edges:
            mermaid_lines.append(f"  {edge.from_code} --> {edge.to_code}")
        
        mermaid_lines.append("")
        
        # Add styling
        mermaid_lines.extend([
            "  %% Styling",
            "  classDef ingestionpreparation fill:#FF6B6B,stroke:#333,stroke-width:2px",
            "  classDef contextconstruction fill:#4ECDC4,stroke:#333,stroke-width:2px", 
            "  classDef knowledgeextraction fill:#45B7D1,stroke:#333,stroke-width:2px",
            "  classDef analysisnlp fill:#96CEB4,stroke:#333,stroke-width:2px",
            "  classDef classificationevaluation fill:#FFEAA7,stroke:#333,stroke-width:2px",
            "  classDef orchestrationcontrol fill:#DDA0DD,stroke:#333,stroke-width:2px",
            "  classDef searchretrieval fill:#98D8C8,stroke:#333,stroke-width:2px",
            "  classDef synthesisoutput fill:#F7DC6F,stroke:#333,stroke-width:2px",
            "  classDef aggregationreporting fill:#BB8FCE,stroke:#333,stroke-width:2px",
            "  classDef integrationstorage fill:#85C1E9,stroke:#333,stroke-width:2px",
            "  classDef unclassified fill:#BDC3C7,stroke:#333,stroke-width:2px",
            ""
        ])
        
        # Apply classes to nodes
        phase_to_nodes = defaultdict(list)
        for node in nodes:
            phase_class = node.phase.replace('_', '')
            phase_to_nodes[phase_class].append(node.code)
        
        for phase_class, node_codes in phase_to_nodes.items():
            node_list = ",".join(node_codes)
            mermaid_lines.append(f"  class {node_list} {phase_class}")
        
        return "\n".join(mermaid_lines)
    
    def render_png(self, dot_content: str, output_path: str) -> bool:
        """Render PNG using Graphviz."""
        try:
            result = subprocess.run(
                ["dot", "-Tpng", "-o", output_path],
                input=dot_content,
                text=True,
                capture_output=True,
                check=True
            )
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error rendering PNG: {e.stderr}")
            return False
        except FileNotFoundError:
            print("Error: Graphviz 'dot' command not found. Please install Graphviz.")
            return False
    
    def render_svg(self, dot_content: str, output_path: str) -> bool:
        """Render SVG using Graphviz."""
        try:
            result = subprocess.run(
                ["dot", "-Tsvg", "-o", output_path],
                input=dot_content,
                text=True,
                capture_output=True,
                check=True
            )
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error rendering SVG: {e.stderr}")
            return False
        except FileNotFoundError:
            print("Error: Graphviz 'dot' command not found. Please install Graphviz.")
            return False
    
    def generate_all_formats(self, output_prefix: str = "pipeline_dag") -> Dict[str, bool]:
        """Generate all visualization formats."""
        print("📊 Generating pipeline DAG visualizations...")
        
        # Load index and build DAG
        index_data = self.load_index()
        nodes, edges = self.build_dag(index_data)
        
        print(f"📈 Built DAG with {len(nodes)} nodes and {len(edges)} edges")
        
        # Calculate layout
        self.calculate_layout(nodes, edges)
        
        results = {}
        
        # Generate Graphviz DOT
        dot_content = self.generate_graphviz_dot(nodes, edges)
        dot_path = f"{output_prefix}.dot"
        
        with open(dot_path, 'w') as f:
            f.write(dot_content)
        
        print(f"💾 Generated DOT file: {dot_path}")
        results['dot'] = True
        
        # Generate Mermaid
        mermaid_content = self.generate_mermaid_diagram(nodes, edges)
        mermaid_path = f"{output_prefix}.mmd"
        
        with open(mermaid_path, 'w') as f:
            f.write(mermaid_content)
        
        print(f"💾 Generated Mermaid file: {mermaid_path}")
        results['mermaid'] = True
        
        # Render PNG
        png_path = f"{output_prefix}.png"
        results['png'] = self.render_png(dot_content, png_path)
        
        if results['png']:
            print(f"🖼️  Generated PNG: {png_path}")
        
        # Render SVG
        svg_path = f"{output_prefix}.svg"
        results['svg'] = self.render_svg(dot_content, svg_path)
        
        if results['svg']:
            print(f"🖼️  Generated SVG: {svg_path}")
        
        return results
    
    def validate_dag(self) -> Tuple[bool, List[str]]:
        """Validate DAG for cycles and orphaned nodes."""
        errors = []
        
        try:
            index_data = self.load_index()
            nodes, edges = self.build_dag(index_data)
            
            # Check for cycles using DFS
            if self._has_cycles(nodes, edges):
                errors.append("DAG contains cycles (circular dependencies)")
            
            # Check for orphaned dependencies
            component_codes = {n.code for n in nodes}
            
            for node in nodes:
                for dep_code in node.dependencies:
                    if dep_code not in component_codes:
                        errors.append(f"Component {node.name} has orphaned dependency: {dep_code}")
            
            # Check for unreachable nodes
            entry_points = [n for n in nodes if not n.dependencies]
            if not entry_points:
                errors.append("No entry points found (all components have dependencies)")
            
            reachable = self._find_reachable_nodes(nodes, edges, entry_points)
            unreachable = [n.name for n in nodes if n.name not in reachable]
            
            if unreachable:
                errors.append(f"Unreachable components: {', '.join(unreachable)}")
            
        except Exception as e:
            errors.append(f"DAG validation error: {e}")
        
        return len(errors) == 0, errors
    
    def _has_cycles(self, nodes: List[DAGNode], edges: List[DAGEdge]) -> bool:
        """Check for cycles using DFS."""
        # Build adjacency list
        graph = defaultdict(list)
        for edge in edges:
            graph[edge.from_node].append(edge.to_node)
        
        # DFS with colors: white=0, gray=1, black=2
        colors = {node.name: 0 for node in nodes}
        
        def dfs(node_name: str) -> bool:
            colors[node_name] = 1  # Gray
            
            for neighbor in graph[node_name]:
                if colors[neighbor] == 1:  # Back edge found
                    return True
                if colors[neighbor] == 0 and dfs(neighbor):
                    return True
            
            colors[node_name] = 2  # Black
            return False
        
        for node in nodes:
            if colors[node.name] == 0:
                if dfs(node.name):
                    return True
        
        return False
    
    def _find_reachable_nodes(self, nodes: List[DAGNode], edges: List[DAGEdge], 
                             entry_points: List[DAGNode]) -> Set[str]:
        """Find all reachable nodes from entry points."""
        # Build adjacency list
        graph = defaultdict(list)
        for edge in edges:
            graph[edge.from_node].append(edge.to_node)
        
        reachable = set()
        queue = deque([ep.name for ep in entry_points])
        
        while queue:
            current = queue.popleft()
            if current in reachable:
                continue
            
            reachable.add(current)
            queue.extend(graph[current])
        
        return reachable


def main():
    """Main CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Pipeline DAG visualization system")
    parser.add_argument("--index", default="pipeline_index.json", help="Index file path")
    parser.add_argument("--output", default="pipeline_dag", help="Output file prefix")
    parser.add_argument("--validate", action="store_true", help="Validate DAG structure")
    parser.add_argument("--format", choices=["png", "svg", "dot", "mermaid", "all"], 
                       default="all", help="Output format")
    
    args = parser.parse_args()
    
    visualizer = PipelineDAGVisualizer(args.index)
    
    if args.validate:
        print("🔍 Validating DAG structure...")
        is_valid, errors = visualizer.validate_dag()
        
        if is_valid:
            print("✅ DAG validation passed")
        else:
            print("❌ DAG validation failed:")
            for error in errors:
                print(f"  • {error}")
            return 1
    
    if args.format == "all":
        results = visualizer.generate_all_formats(args.output)
        
        failed_formats = [fmt for fmt, success in results.items() if not success]
        if failed_formats:
            print(f"⚠️  Failed to generate: {', '.join(failed_formats)}")
            return 1
        
        print("✅ All visualizations generated successfully")
        
    else:
        # Generate specific format
        index_data = visualizer.load_index()
        nodes, edges = visualizer.build_dag(index_data)
        
        if args.format == "dot":
            content = visualizer.generate_graphviz_dot(nodes, edges)
            output_path = f"{args.output}.dot"
            with open(output_path, 'w') as f:
                f.write(content)
            print(f"💾 Generated: {output_path}")
        
        elif args.format == "mermaid":
            content = visualizer.generate_mermaid_diagram(nodes, edges)
            output_path = f"{args.output}.mmd"
            with open(output_path, 'w') as f:
                f.write(content)
            print(f"💾 Generated: {output_path}")
        
        elif args.format == "png":
            dot_content = visualizer.generate_graphviz_dot(nodes, edges)
            output_path = f"{args.output}.png"
            if visualizer.render_png(dot_content, output_path):
                print(f"🖼️  Generated: {output_path}")
            else:
                return 1
        
        elif args.format == "svg":
            dot_content = visualizer.generate_graphviz_dot(nodes, edges)
            output_path = f"{args.output}.svg"
            if visualizer.render_svg(dot_content, output_path):
                print(f"🖼️  Generated: {output_path}")
            else:
                return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
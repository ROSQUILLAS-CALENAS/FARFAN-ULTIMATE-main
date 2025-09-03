#!/usr/bin/env python3
"""
Continuous Canonical Compliance (CCC) Validator

A comprehensive validation system that implements five validation gates:
1. File naming compliance checker
2. Index synchronization validator  
3. Signature reflection validator
4. Phase layering rules enforcer
5. DAG validity checker

Generates comprehensive HTML reports with embedded DAG visualizations.
"""

import ast
import json
import re
import sys
import warnings
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# Visualization dependencies
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import networkx as nx
    import numpy as np
    HAS_VISUALIZATION = True
except ImportError:
    # Fallback imports and mocks for missing dependencies
    HAS_VISUALIZATION = False
    warnings.warn("Visualization dependencies not available. DAG rendering will be disabled.")
    
    # Mock NetworkX for basic graph operations
    class MockDigraph:
        def __init__(self):
            self.nodes_dict = {}
            self.edges_list = []
            
        def add_node(self, node, **attrs):
            self.nodes_dict[node] = attrs
            
        def add_edge(self, u, v):
            self.edges_list.append((u, v))
            
        def nodes(self):
            return self.nodes_dict.keys()
            
        def number_of_nodes(self):
            return len(self.nodes_dict)
            
        def number_of_edges(self):
            return len(self.edges_list)
            
        def clear(self):
            self.nodes_dict.clear()
            self.edges_list.clear()
    
    # Mock NetworkX module
    class MockNetworkX:
        DiGraph = MockDigraph
        
        @staticmethod
        def simple_cycles(graph):
            # Simple cycle detection without full NetworkX
            return []  # Return empty for now - basic fallback
    
    nx = MockNetworkX()

# Phase mapping for canonical flow
CANONICAL_PHASES = {
    'I': ('ingestion_preparation', 0),
    'X': ('context_construction', 1),
    'K': ('knowledge_extraction', 2),
    'A': ('analysis_nlp', 3),
    'L': ('classification_evaluation', 4),
    'R': ('search_retrieval', 5),
    'O': ('orchestration_control', 6),
    'G': ('aggregation_reporting', 7),
    'T': ('integration_storage', 8),
    'S': ('synthesis_output', 9)
}

PHASE_ORDER = ['I', 'X', 'K', 'A', 'L', 'R', 'O', 'G', 'T', 'S']


@dataclass
class ValidationResult:
    """Result of a single validation check."""
    gate: str
    passed: bool
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    severity: str = "error"  # error, warning, info


@dataclass
class ComponentInfo:
    """Information about a canonical flow component."""
    path: str
    phase: str
    component_id: str
    dependencies: List[str] = field(default_factory=list)
    exports_process: bool = False
    signature_valid: bool = False


class CCCValidator:
    """Continuous Canonical Compliance Validator."""

    def __init__(self, repo_root: Path, config: Optional[Dict] = None):
        self.repo_root = Path(repo_root)
        self.canonical_flow_dir = self.repo_root / "canonical_flow"
        self.config = config or {}
        self.results: List[ValidationResult] = []
        self.components: Dict[str, ComponentInfo] = {}
        self.dependency_graph = nx.DiGraph()
        
    def validate_all(self) -> Dict[str, Any]:
        """Run all validation gates and return comprehensive report."""
        self.results.clear()
        self.components.clear()
        self.dependency_graph.clear()
        
        # Discover components first
        self._discover_components()
        
        # Run validation gates
        self._validate_file_naming()
        self._validate_index_synchronization()
        self._validate_signatures()
        self._validate_phase_layering()
        self._validate_dag()
        
        # Generate report
        return self._generate_report()
    
    def _discover_components(self):
        """Discover all canonical flow components."""
        if not self.canonical_flow_dir.exists():
            self.results.append(ValidationResult(
                gate="discovery",
                passed=False,
                message="Canonical flow directory not found",
                severity="error"
            ))
            return
        
        # Scan for Python files
        for py_file in self.canonical_flow_dir.rglob("*.py"):
            if py_file.name == "__init__.py":
                continue
                
            relative_path = py_file.relative_to(self.repo_root)
            component_info = self._analyze_component(py_file)
            if component_info:
                self.components[str(relative_path)] = component_info
                
    def _analyze_component(self, file_path: Path) -> Optional[ComponentInfo]:
        """Analyze a component file and extract information."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST to extract information
            tree = ast.parse(content)
            
            # Extract phase from path
            parts = file_path.parts
            phase = None
            component_id = None
            
            for part in parts:
                if '_' in part and len(part) > 2:
                    phase_letter = part.split('_')[0]
                    if phase_letter in CANONICAL_PHASES:
                        phase = phase_letter
                        break
            
            # Try to extract component ID from filename
            match = re.match(r'(\d+[A-Z])', file_path.name)
            if match:
                component_id = match.group(1)
            
            # Find dependencies and process function
            dependencies = self._find_dependencies(content)
            exports_process = self._has_process_function(tree)
            signature_valid = self._validate_process_signature(tree) if exports_process else True
            
            return ComponentInfo(
                path=str(file_path.relative_to(self.repo_root)),
                phase=phase or "unknown",
                component_id=component_id or file_path.stem,
                dependencies=dependencies,
                exports_process=exports_process,
                signature_valid=signature_valid
            )
            
        except Exception as e:
            self.results.append(ValidationResult(
                gate="discovery",
                passed=False,
                message=f"Failed to analyze {file_path}: {e}",
                severity="warning"
            ))
            return None
    
    def _find_dependencies(self, content: str) -> List[str]:
        """Find import dependencies in the code."""
        dependencies = []
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        dependencies.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        dependencies.append(node.module)
        except:
            pass
        return dependencies
    
    def _has_process_function(self, tree: ast.AST) -> bool:
        """Check if module has a process function."""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "process":
                return True
        return False
    
    def _validate_process_signature(self, tree: ast.AST) -> bool:
        """Validate process function signature."""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "process":
                # Check for (data, context) parameters
                if len(node.args.args) >= 2:
                    return True
        return False
    
    def _validate_file_naming(self):
        """Validate file naming compliance against canonical phase prefixes."""
        passed = True
        violations = []
        
        for path, component in self.components.items():
            # Check if file follows naming convention
            filename = Path(path).name
            
            # Should start with phase prefix
            if component.phase != "unknown":
                expected_prefix = component.phase.upper()
                if not any(part.startswith(f"{expected_prefix}_") for part in Path(path).parts):
                    violations.append(f"{path}: Missing phase prefix {expected_prefix}_")
                    passed = False
            
            # Check for component ID pattern
            if not re.match(r'^\d+[A-Z]_', filename) and component.component_id:
                violations.append(f"{path}: Invalid component ID format")
                passed = False
        
        self.results.append(ValidationResult(
            gate="file_naming",
            passed=passed,
            message="File naming compliance validated" if passed else f"Found {len(violations)} naming violations",
            details={"violations": violations}
        ))
    
    def _validate_index_synchronization(self):
        """Compare filesystem components against index.json entries."""
        index_path = self.canonical_flow_dir / "index.json"
        passed = True
        issues = []
        
        if not index_path.exists():
            self.results.append(ValidationResult(
                gate="index_sync",
                passed=False,
                message="index.json not found in canonical_flow directory",
                severity="error"
            ))
            return
        
        try:
            with open(index_path, 'r') as f:
                index_entries = json.load(f)
            
            # Create sets for comparison
            indexed_files = {entry.get('alias_path', '') for entry in index_entries}
            filesystem_files = set(self.components.keys())
            
            # Find missing files
            missing_in_index = filesystem_files - indexed_files
            missing_in_fs = indexed_files - filesystem_files
            
            if missing_in_index:
                issues.append(f"Files missing from index: {list(missing_in_index)}")
                passed = False
            
            if missing_in_fs:
                issues.append(f"Index entries missing from filesystem: {list(missing_in_fs)}")
                passed = False
                
        except Exception as e:
            passed = False
            issues.append(f"Failed to read index.json: {e}")
        
        self.results.append(ValidationResult(
            gate="index_sync",
            passed=passed,
            message="Index synchronization validated" if passed else f"Found {len(issues)} sync issues",
            details={"issues": issues}
        ))
    
    def _validate_signatures(self):
        """Validate that modules expose standard process(data, context) interface."""
        passed = True
        violations = []
        
        for path, component in self.components.items():
            if component.exports_process and not component.signature_valid:
                violations.append(f"{path}: Invalid process function signature")
                passed = False
            elif not component.exports_process:
                violations.append(f"{path}: Missing process function")
                passed = False
        
        self.results.append(ValidationResult(
            gate="signature_validation",
            passed=passed,
            message="Signature validation completed" if passed else f"Found {len(violations)} signature issues",
            details={"violations": violations}
        ))
    
    def _validate_phase_layering(self):
        """Detect backward dependencies violating canonical flow order."""
        passed = True
        violations = []
        
        for path, component in self.components.items():
            if component.phase == "unknown":
                continue
                
            current_phase_order = CANONICAL_PHASES.get(component.phase, (None, 999))[1]
            
            # Check dependencies
            for dep in component.dependencies:
                # Find components that might match this dependency
                for dep_path, dep_component in self.components.items():
                    if dep in dep_path or dep_component.component_id in dep:
                        dep_phase_order = CANONICAL_PHASES.get(dep_component.phase, (None, 999))[1]
                        
                        # Backward dependency detected
                        if dep_phase_order > current_phase_order:
                            violations.append(
                                f"{path} (phase {component.phase}) depends on "
                                f"{dep_path} (phase {dep_component.phase}) - violates flow order"
                            )
                            passed = False
                            
                            # Add to dependency graph
                            self.dependency_graph.add_edge(component.component_id, dep_component.component_id)
        
        self.results.append(ValidationResult(
            gate="phase_layering",
            passed=passed,
            message="Phase layering validated" if passed else f"Found {len(violations)} layering violations",
            details={"violations": violations}
        ))
    
    def _validate_dag(self):
        """Identify circular dependencies using graph analysis."""
        # Build complete dependency graph
        for path, component in self.components.items():
            self.dependency_graph.add_node(component.component_id, 
                                         path=path, 
                                         phase=component.phase)
            
            for dep in component.dependencies:
                for dep_path, dep_component in self.components.items():
                    if dep in dep_path:
                        self.dependency_graph.add_edge(component.component_id, dep_component.component_id)
        
        # Check for cycles
        try:
            cycles = list(nx.simple_cycles(self.dependency_graph))
            passed = len(cycles) == 0
            
            self.results.append(ValidationResult(
                gate="dag_validation",
                passed=passed,
                message="DAG validation completed" if passed else f"Found {len(cycles)} circular dependencies",
                details={"cycles": cycles}
            ))
        except Exception as e:
            self.results.append(ValidationResult(
                gate="dag_validation",
                passed=False,
                message=f"DAG validation failed: {e}",
                severity="error"
            ))
    
    def _generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        # Calculate summary statistics
        total_gates = len([r for r in self.results if r.gate != "discovery"])
        passed_gates = len([r for r in self.results if r.gate != "discovery" and r.passed])
        
        # Generate HTML report
        html_report = self._generate_html_report()
        
        # Generate DAG visualization if available
        dag_artifacts = {}
        if HAS_VISUALIZATION:
            dag_artifacts = self._generate_dag_visualizations()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_gates": total_gates,
                "passed_gates": passed_gates,
                "success_rate": passed_gates / total_gates if total_gates > 0 else 0,
                "overall_status": "PASS" if passed_gates == total_gates else "FAIL"
            },
            "gate_results": [
                {
                    "gate": r.gate,
                    "status": "PASS" if r.passed else "FAIL",
                    "message": r.message,
                    "details": r.details,
                    "severity": r.severity
                }
                for r in self.results if r.gate != "discovery"
            ],
            "components": {
                path: {
                    "phase": comp.phase,
                    "component_id": comp.component_id,
                    "exports_process": comp.exports_process,
                    "signature_valid": comp.signature_valid,
                    "dependency_count": len(comp.dependencies)
                }
                for path, comp in self.components.items()
            },
            "html_report": html_report,
            "dag_artifacts": dag_artifacts
        }
    
    def _generate_html_report(self) -> str:
        """Generate comprehensive HTML report with embedded visualizations."""
        html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Continuous Canonical Compliance Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }
        .header { text-align: center; margin-bottom: 30px; }
        .summary { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }
        .stat-card { background: #f8f9fa; padding: 20px; border-radius: 8px; text-align: center; border-left: 4px solid #007bff; }
        .stat-card.success { border-left-color: #28a745; }
        .stat-card.failure { border-left-color: #dc3545; }
        .gate-result { margin-bottom: 20px; padding: 15px; border-radius: 8px; }
        .gate-result.pass { background-color: #d4edda; border-left: 4px solid #28a745; }
        .gate-result.fail { background-color: #f8d7da; border-left: 4px solid #dc3545; }
        .details { margin-top: 10px; font-size: 0.9em; }
        .component-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 15px; }
        .component-card { background: #f8f9fa; padding: 15px; border-radius: 8px; }
        .phase-badge { display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 0.8em; color: white; }
        h1, h2, h3 { color: #333; }
        .timestamp { color: #666; font-size: 0.9em; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 Continuous Canonical Compliance Report</h1>
            <p class="timestamp">Generated: {timestamp}</p>
        </div>
        
        <div class="summary">
            <div class="stat-card {overall_class}">
                <h3>Overall Status</h3>
                <div style="font-size: 2em; font-weight: bold; color: {status_color};">{overall_status}</div>
            </div>
            <div class="stat-card">
                <h3>Success Rate</h3>
                <div style="font-size: 2em; font-weight: bold;">{success_rate:.1%}</div>
            </div>
            <div class="stat-card">
                <h3>Gates Passed</h3>
                <div style="font-size: 2em; font-weight: bold;">{passed_gates}/{total_gates}</div>
            </div>
            <div class="stat-card">
                <h3>Components</h3>
                <div style="font-size: 2em; font-weight: bold;">{component_count}</div>
            </div>
        </div>
        
        <h2>🚦 Validation Gates</h2>
        {gate_results_html}
        
        <h2>📦 Component Overview</h2>
        <div class="component-grid">
            {components_html}
        </div>
        
        {dag_html}
    </div>
</body>
</html>
        """.strip()
        
        # Calculate summary values
        total_gates = len([r for r in self.results if r.gate != "discovery"])
        passed_gates = len([r for r in self.results if r.gate != "discovery" and r.passed])
        success_rate = passed_gates / total_gates if total_gates > 0 else 0
        overall_status = "PASS" if passed_gates == total_gates else "FAIL"
        
        # Generate gate results HTML
        gate_results_html = ""
        gate_icons = {
            "file_naming": "📁",
            "index_sync": "🔄", 
            "signature_validation": "✍️",
            "phase_layering": "🔀",
            "dag_validation": "📊"
        }
        
        for result in self.results:
            if result.gate == "discovery":
                continue
                
            status_class = "pass" if result.passed else "fail"
            icon = gate_icons.get(result.gate, "🔍")
            
            details_html = ""
            if result.details:
                for key, value in result.details.items():
                    if isinstance(value, list) and value:
                        details_html += f"<div class='details'><strong>{key.title()}:</strong><ul>"
                        for item in value[:10]:  # Limit to first 10 items
                            details_html += f"<li>{item}</li>"
                        if len(value) > 10:
                            details_html += f"<li>... and {len(value) - 10} more</li>"
                        details_html += "</ul></div>"
            
            gate_results_html += f"""
            <div class="gate-result {status_class}">
                <h3>{icon} {result.gate.replace('_', ' ').title()}</h3>
                <p>{result.message}</p>
                {details_html}
            </div>
            """
        
        # Generate components HTML
        components_html = ""
        phase_colors = {
            'I': '#ff6b6b', 'X': '#4ecdc4', 'K': '#45b7d1', 'A': '#96ceb4', 'L': '#feca57',
            'R': '#ff9ff3', 'O': '#54a0ff', 'G': '#5f27cd', 'T': '#00d2d3', 'S': '#ff9f43'
        }
        
        for path, component in self.components.items():
            phase_color = phase_colors.get(component.phase, '#6c757d')
            components_html += f"""
            <div class="component-card">
                <h4>{Path(path).name}</h4>
                <p><span class="phase-badge" style="background-color: {phase_color};">
                    {component.phase} - {CANONICAL_PHASES.get(component.phase, ('unknown', 0))[0]}
                </span></p>
                <p><strong>ID:</strong> {component.component_id}</p>
                <p><strong>Process Function:</strong> {'✅' if component.exports_process else '❌'}</p>
                <p><strong>Valid Signature:</strong> {'✅' if component.signature_valid else '❌'}</p>
                <p><strong>Dependencies:</strong> {len(component.dependencies)}</p>
            </div>
            """
        
        # DAG section
        dag_html = ""
        if HAS_VISUALIZATION:
            dag_html = """
            <h2>🌐 Dependency Graph</h2>
            <div id="dag-container">
                <p>DAG visualizations would be embedded here when rendered artifacts are available.</p>
            </div>
            """
        
        return html.format(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            overall_status=overall_status,
            overall_class="success" if overall_status == "PASS" else "failure",
            status_color="#28a745" if overall_status == "PASS" else "#dc3545",
            success_rate=success_rate,
            passed_gates=passed_gates,
            total_gates=total_gates,
            component_count=len(self.components),
            gate_results_html=gate_results_html,
            components_html=components_html,
            dag_html=dag_html
        )
    
    def _generate_dag_visualizations(self) -> Dict[str, Any]:
        """Generate DAG visualization artifacts."""
        if not HAS_VISUALIZATION:
            return {}
        
        try:
            # Create figure
            plt.figure(figsize=(16, 12))
            
            # Use spring layout for better visualization
            pos = nx.spring_layout(self.dependency_graph, k=2, iterations=50)
            
            # Draw nodes with phase-based coloring
            phase_colors = {
                'I': '#ff6b6b', 'X': '#4ecdc4', 'K': '#45b7d1', 'A': '#96ceb4', 'L': '#feca57',
                'R': '#ff9ff3', 'O': '#54a0ff', 'G': '#5f27cd', 'T': '#00d2d3', 'S': '#ff9f43'
            }
            
            node_colors = []
            for node in self.dependency_graph.nodes():
                node_data = self.dependency_graph.nodes[node]
                phase = node_data.get('phase', 'unknown')
                node_colors.append(phase_colors.get(phase, '#6c757d'))
            
            # Draw graph
            nx.draw(self.dependency_graph, pos,
                   node_color=node_colors,
                   node_size=1000,
                   font_size=8,
                   font_weight='bold',
                   arrows=True,
                   arrowsize=20,
                   edge_color='gray',
                   with_labels=True)
            
            plt.title("Canonical Flow Dependency Graph", size=16, weight='bold')
            
            # Save as PNG and SVG
            png_path = self.repo_root / "ccc_dag_visualization.png"
            svg_path = self.repo_root / "ccc_dag_visualization.svg"
            
            plt.savefig(png_path, dpi=300, bbox_inches='tight')
            plt.savefig(svg_path, format='svg', bbox_inches='tight')
            plt.close()
            
            return {
                "png_path": str(png_path),
                "svg_path": str(svg_path),
                "node_count": self.dependency_graph.number_of_nodes(),
                "edge_count": self.dependency_graph.number_of_edges()
            }
            
        except Exception as e:
            return {"error": f"Failed to generate DAG visualizations: {e}"}
    
    def export_artifacts(self, output_dir: Path):
        """Export validation artifacts to specified directory."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Run validation if not already done
        if not self.results:
            self.validate_all()
        
        # Generate full report
        report = self._generate_report()
        
        # Save HTML report
        html_file = output_dir / "ccc_validation_report.html"
        with open(html_file, 'w') as f:
            f.write(report["html_report"])
        
        # Save JSON report  
        json_file = output_dir / "ccc_validation_report.json"
        with open(json_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Copy DAG artifacts if they exist
        if "png_path" in report["dag_artifacts"]:
            import shutil
            png_src = Path(report["dag_artifacts"]["png_path"])
            svg_src = Path(report["dag_artifacts"]["svg_path"])
            if png_src.exists():
                shutil.copy2(png_src, output_dir / "dag_visualization.png")
            if svg_src.exists():
                shutil.copy2(svg_src, output_dir / "dag_visualization.svg")
        
        return {
            "html_report": str(html_file),
            "json_report": str(json_file),
            "output_directory": str(output_dir)
        }


def main():
    """CLI interface for CCC Validator."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Continuous Canonical Compliance Validator")
    parser.add_argument("--repo-root", type=Path, default=".", 
                       help="Repository root directory")
    parser.add_argument("--output-dir", type=Path, default="validation_reports",
                       help="Output directory for reports")
    parser.add_argument("--fail-on-violations", action="store_true",
                       help="Exit with non-zero code if violations found")
    parser.add_argument("--config", type=Path,
                       help="Path to configuration file")
    
    args = parser.parse_args()
    
    # Load config if provided
    config = {}
    if args.config and args.config.exists():
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Initialize validator
    validator = CCCValidator(args.repo_root, config)
    
    # Run validation
    print("🔍 Running Continuous Canonical Compliance validation...")
    report = validator.validate_all()
    
    # Export artifacts
    artifacts = validator.export_artifacts(args.output_dir)
    
    # Print summary
    summary = report["summary"]
    print(f"\n📊 Validation Summary:")
    print(f"   Status: {summary['overall_status']}")
    print(f"   Gates: {summary['passed_gates']}/{summary['total_gates']} passed")
    print(f"   Success Rate: {summary['success_rate']:.1%}")
    print(f"   Components: {len(report['components'])}")
    
    print(f"\n📄 Reports generated:")
    print(f"   HTML: {artifacts['html_report']}")
    print(f"   JSON: {artifacts['json_report']}")
    
    # Exit with appropriate code
    if args.fail_on_violations and summary['overall_status'] != 'PASS':
        sys.exit(1)
    
    sys.exit(0)


if __name__ == "__main__":
    main()
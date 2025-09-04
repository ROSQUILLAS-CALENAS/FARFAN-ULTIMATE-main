#!/usr/bin/env python3
"""
Pipeline Index System - Comprehensive single source of truth for pipeline components.

This system implements:
1. Autoscan system for automatic pipeline component discovery
2. DAG visualization with PNG/SVG generation
3. Validation logic for index.json vs filesystem reconciliation
4. Git hooks for automatic updates on commits
"""

import json
import os
import subprocess
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
from collections import defaultdict
import ast

# Optional visualization imports
try:
    import graphviz
    GRAPHVIZ_AVAILABLE = True
except ImportError:
    GRAPHVIZ_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ComponentInfo:
    """Information about a pipeline component"""
    code: str
    stage: str
    alias_path: str
    original_path: str
    file_hash: str = ""
    dependencies: List[str] = None
    imports: List[str] = None
    exports: List[str] = None
    last_modified: str = ""
    size_bytes: int = 0
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.imports is None:
            self.imports = []
        if self.exports is None:
            self.exports = []

class PipelineIndexSystem:
    """Main class for managing the pipeline index system"""
    
    # Stage definitions and their order
    STAGE_ORDER = [
        "ingestion_preparation",
        "context_construction", 
        "knowledge_extraction",
        "analysis_nlp",
        "classification_evaluation",
        "orchestration_control",
        "search_retrieval",
        "synthesis_output",
        "aggregation_reporting",
        "integration_storage"
    ]
    
    STAGE_PREFIXES = {
        "ingestion_preparation": "I",
        "context_construction": "X",
        "knowledge_extraction": "K", 
        "analysis_nlp": "A",
        "classification_evaluation": "L",
        "orchestration_control": "O",
        "search_retrieval": "R",
        "synthesis_output": "S",
        "aggregation_reporting": "G",
        "integration_storage": "T"
    }
    
    def __init__(self, root_dir: Path = None, index_path: Path = None):
        self.root_dir = root_dir or Path(".")
        self.index_path = index_path or Path("canonical_flow/index.json")
        self.canonical_dir = Path("canonical_flow")
        
    def calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of a file"""
        if not file_path.exists():
            return ""
        
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def extract_imports_exports(self, file_path: Path) -> Tuple[List[str], List[str]]:
        """Extract imports and exports from a Python file using AST"""
        if not file_path.exists() or not file_path.suffix == '.py':
            return [], []
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            imports = []
            exports = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        for alias in node.names:
                            imports.append(f"{node.module}.{alias.name}")
                elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    if not node.name.startswith('_'):  # Public functions/classes
                        exports.append(node.name)
                        
            return imports, exports
            
        except Exception as e:
            logger.warning(f"Could not parse {file_path}: {e}")
            return [], []
    
    def scan_filesystem_components(self) -> List[ComponentInfo]:
        """Scan the filesystem to discover all pipeline components"""
        components = []
        component_counter = defaultdict(int)
        
        # Scan known directories for Python files
        scan_dirs = [
            self.root_dir,
            self.root_dir / "canonical_flow",
            self.root_dir / "retrieval_engine", 
            self.root_dir / "semantic_reranking",
            self.root_dir / "analysis_nlp",
            self.root_dir / "G_aggregation_reporting"
        ]
        
        for scan_dir in scan_dirs:
            if not scan_dir.exists():
                continue
                
            for py_file in scan_dir.rglob("*.py"):
                if "__pycache__" in str(py_file) or py_file.name.startswith("test_"):
                    continue
                    
                # Determine stage based on directory or filename patterns
                stage = self._determine_stage(py_file)
                if not stage:
                    continue
                    
                component_counter[stage] += 1
                prefix = self.STAGE_PREFIXES.get(stage, "U")
                code = f"{component_counter[stage]:02d}{prefix}"
                
                # Calculate relative paths
                try:
                    original_path = str(py_file.relative_to(self.root_dir))
                except ValueError:
                    original_path = str(py_file)
                
                alias_path = f"canonical_flow/{prefix}_{stage}/{code}_{py_file.name}"
                
                # Extract metadata
                file_hash = self.calculate_file_hash(py_file)
                imports, exports = self.extract_imports_exports(py_file)
                
                stat = py_file.stat()
                last_modified = datetime.fromtimestamp(stat.st_mtime).isoformat()
                
                component = ComponentInfo(
                    code=code,
                    stage=stage,
                    alias_path=alias_path,
                    original_path=original_path,
                    file_hash=file_hash,
                    imports=imports,
                    exports=exports,
                    last_modified=last_modified,
                    size_bytes=stat.st_size
                )
                
                components.append(component)
        
        return sorted(components, key=lambda x: x.code)
    
    def _determine_stage(self, file_path: Path) -> Optional[str]:
        """Determine the stage of a component based on its path and name"""
        path_str = str(file_path).lower()
        name = file_path.name.lower()
        
        # Stage mapping based on keywords and paths
        stage_keywords = {
            "ingestion_preparation": ["pdf_reader", "loader", "extractor", "normative"],
            "context_construction": ["context", "immutable", "lineage"],
            "knowledge_extraction": ["knowledge", "graph", "causal", "embedding"],
            "analysis_nlp": ["analyzer", "nlp", "evidence", "mapping", "processor"],
            "classification_evaluation": ["scoring", "adaptive", "conformal", "risk"],
            "orchestration_control": ["orchestrator", "router", "controller", "validator", "decision"],
            "search_retrieval": ["retrieval", "index", "search", "rerank", "hybrid"],
            "synthesis_output": ["synthesizer", "formatter", "answer"],
            "aggregation_reporting": ["aggregator", "compiler", "report"],
            "integration_storage": ["metrics", "analytics", "feedback", "optimization"]
        }
        
        # Check directory first
        for stage, keywords in stage_keywords.items():
            stage_prefix = self.STAGE_PREFIXES[stage]
            if f"{stage_prefix}_{stage}" in path_str or stage in path_str:
                return stage
                
        # Check filename keywords
        for stage, keywords in stage_keywords.items():
            if any(keyword in name for keyword in keywords):
                return stage
                
        return None
    
    def load_current_index(self) -> List[ComponentInfo]:
        """Load the current index.json file"""
        if not self.index_path.exists():
            return []
            
        try:
            with open(self.index_path, 'r') as f:
                data = json.load(f)
            
            components = []
            for item in data:
                component = ComponentInfo(**item)
                components.append(component)
                
            return components
            
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            return []
    
    def save_index(self, components: List[ComponentInfo]):
        """Save components to index.json"""
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = [asdict(comp) for comp in components]
        
        with open(self.index_path, 'w') as f:
            json.dump(data, f, indent=2, sort_keys=True)
            
        logger.info(f"Saved {len(components)} components to {self.index_path}")
    
    def reconcile_index(self) -> Tuple[List[ComponentInfo], Dict[str, Any]]:
        """Reconcile filesystem state with index.json"""
        fs_components = self.scan_filesystem_components()
        index_components = self.load_current_index()
        
        # Create lookup dicts
        fs_dict = {comp.original_path: comp for comp in fs_components}
        index_dict = {comp.original_path: comp for comp in index_components}
        
        reconciliation_report = {
            "timestamp": datetime.now().isoformat(),
            "added": [],
            "modified": [],
            "deleted": [],
            "unchanged": [],
            "hash_mismatches": []
        }
        
        # Find added, modified, unchanged
        for path, fs_comp in fs_dict.items():
            if path not in index_dict:
                reconciliation_report["added"].append(path)
            else:
                index_comp = index_dict[path]
                if fs_comp.file_hash != index_comp.file_hash:
                    reconciliation_report["modified"].append(path)
                    reconciliation_report["hash_mismatches"].append({
                        "path": path,
                        "fs_hash": fs_comp.file_hash,
                        "index_hash": index_comp.file_hash
                    })
                else:
                    reconciliation_report["unchanged"].append(path)
        
        # Find deleted
        for path in index_dict:
            if path not in fs_dict:
                reconciliation_report["deleted"].append(path)
        
        return fs_components, reconciliation_report
    
    def calculate_dependencies(self, components: List[ComponentInfo]) -> Dict[str, List[str]]:
        """Calculate component dependencies based on imports"""
        component_map = {comp.original_path: comp for comp in components}
        dependencies = defaultdict(list)
        
        for comp in components:
            for import_name in comp.imports:
                # Try to match imports to other components
                for other_comp in components:
                    if comp.code == other_comp.code:
                        continue
                        
                    # Check if import matches filename or exports
                    other_name = Path(other_comp.original_path).stem
                    if (import_name.endswith(other_name) or 
                        any(exp in import_name for exp in other_comp.exports)):
                        dependencies[comp.code].append(other_comp.code)
                        break
        
        return dict(dependencies)
    
    def generate_dag_visualization(self, components: List[ComponentInfo], 
                                 output_dir: Path = Path("pipeline_visualizations")):
        """Generate DAG visualizations in PNG and SVG formats"""
        output_dir.mkdir(exist_ok=True)
        
        # Calculate dependencies
        dependencies = self.calculate_dependencies(components)
        
        # Update components with dependencies
        for comp in components:
            comp.dependencies = dependencies.get(comp.code, [])
        
        # Generate Graphviz DAG if available
        if GRAPHVIZ_AVAILABLE:
            self._generate_graphviz_dag(components, dependencies, output_dir)
        else:
            logger.warning("Graphviz not available - skipping Graphviz visualizations")
        
        # Generate NetworkX/Matplotlib DAG if available
        if NETWORKX_AVAILABLE and MATPLOTLIB_AVAILABLE:
            self._generate_networkx_dag(components, dependencies, output_dir)
        else:
            logger.warning("NetworkX/Matplotlib not available - skipping NetworkX visualizations")
        
        # Generate simple text-based visualization as fallback
        self._generate_text_dag(components, dependencies, output_dir)
        
        logger.info(f"Generated available visualizations in {output_dir}")
    
    def _generate_graphviz_dag(self, components: List[ComponentInfo], 
                             dependencies: Dict[str, List[str]], 
                             output_dir: Path):
        """Generate DAG using Graphviz"""
        try:
            dot = graphviz.Digraph(comment='Pipeline DAG')
            dot.attr(rankdir='TB', size='12,16')
            
            # Color map for stages
            stage_colors = {
                "ingestion_preparation": "#FF6B6B",
                "context_construction": "#4ECDC4", 
                "knowledge_extraction": "#45B7D1",
                "analysis_nlp": "#96CEB4",
                "classification_evaluation": "#FFEAA7",
                "orchestration_control": "#DDA0DD",
                "search_retrieval": "#98D8C8",
                "synthesis_output": "#F7DC6F",
                "aggregation_reporting": "#BB8FCE",
                "integration_storage": "#85C1E9"
            }
            
            # Add nodes grouped by stage
            with dot.subgraph() as s:
                s.attr(rank='same')
                for stage in self.STAGE_ORDER:
                    stage_components = [c for c in components if c.stage == stage]
                    if stage_components:
                        for comp in stage_components:
                            color = stage_colors.get(stage, "#CCCCCC")
                            s.node(comp.code, 
                                  f"{comp.code}\n{Path(comp.original_path).stem}",
                                  fillcolor=color, style='filled')
            
            # Add edges for dependencies
            for comp_code, deps in dependencies.items():
                for dep_code in deps:
                    dot.edge(dep_code, comp_code)
            
            # Render outputs
            dot.render(output_dir / 'pipeline_dag', format='png', cleanup=True)
            dot.render(output_dir / 'pipeline_dag', format='svg', cleanup=True)
            
        except Exception as e:
            logger.error(f"Error generating Graphviz DAG: {e}")
    
    def _generate_networkx_dag(self, components: List[ComponentInfo],
                             dependencies: Dict[str, List[str]],
                             output_dir: Path):
        """Generate DAG using NetworkX and Matplotlib"""
        try:
            G = nx.DiGraph()
            
            # Add nodes with stage information
            for comp in components:
                G.add_node(comp.code, stage=comp.stage, path=comp.original_path)
            
            # Add edges
            for comp_code, deps in dependencies.items():
                for dep_code in deps:
                    if G.has_node(dep_code) and G.has_node(comp_code):
                        G.add_edge(dep_code, comp_code)
            
            # Create layout
            plt.figure(figsize=(16, 12))
            
            # Position nodes by stage
            pos = {}
            stage_positions = {}
            x_offset = 0
            
            for i, stage in enumerate(self.STAGE_ORDER):
                stage_nodes = [n for n in G.nodes() 
                             if G.nodes[n].get('stage') == stage]
                if stage_nodes:
                    y_positions = [j * 2 for j in range(len(stage_nodes))]
                    y_center = sum(y_positions) / len(y_positions) if y_positions else 0
                    
                    for j, node in enumerate(stage_nodes):
                        pos[node] = (i * 3, y_positions[j] - y_center)
                    
                    stage_positions[stage] = (i * 3, -len(stage_nodes))
            
            # Draw the graph
            nx.draw(G, pos, with_labels=True, node_color='lightblue',
                   node_size=1500, font_size=8, font_weight='bold',
                   arrows=True, edge_color='gray', arrowsize=20)
            
            plt.title("Pipeline Component Dependencies", fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            # Save outputs
            plt.savefig(output_dir / 'pipeline_dag_networkx.png', dpi=300, bbox_inches='tight')
            plt.savefig(output_dir / 'pipeline_dag_networkx.svg', format='svg', bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error generating NetworkX DAG: {e}")
    
    def _generate_text_dag(self, components: List[ComponentInfo],
                          dependencies: Dict[str, List[str]],
                          output_dir: Path):
        """Generate text-based DAG visualization as fallback"""
        try:
            dag_text = ["Pipeline Component Dependencies", "=" * 40, ""]
            
            # Group by stage
            by_stage = defaultdict(list)
            for comp in components:
                by_stage[comp.stage].append(comp)
            
            for stage in self.STAGE_ORDER:
                if stage not in by_stage:
                    continue
                    
                dag_text.append(f"\n{self.STAGE_PREFIXES[stage]} - {stage.replace('_', ' ').title()}:")
                dag_text.append("-" * (len(stage) + 10))
                
                for comp in sorted(by_stage[stage], key=lambda x: x.code):
                    deps = dependencies.get(comp.code, [])
                    if deps:
                        dep_str = " <- " + ", ".join(deps)
                    else:
                        dep_str = ""
                        
                    dag_text.append(f"  {comp.code}: {Path(comp.original_path).stem}{dep_str}")
            
            # Add dependency summary
            dag_text.append("\n\nDependency Summary:")
            dag_text.append("-" * 20)
            total_deps = sum(len(deps) for deps in dependencies.values())
            dag_text.append(f"Total Dependencies: {total_deps}")
            dag_text.append(f"Components with Dependencies: {len([d for d in dependencies.values() if d])}")
            
            # Save text DAG
            with open(output_dir / 'pipeline_dag.txt', 'w') as f:
                f.write('\n'.join(dag_text))
            
            logger.info("Generated text-based DAG visualization")
            
        except Exception as e:
            logger.error(f"Error generating text DAG: {e}")
    
    def validate_index_consistency(self, components: List[ComponentInfo] = None) -> Dict[str, Any]:
        """Validate that index.json matches filesystem state"""
        if components is None:
            components, reconciliation = self.reconcile_index()
        else:
            _, reconciliation = self.reconcile_index()
        
        validation_report = {
            "timestamp": datetime.now().isoformat(),
            "valid": True,
            "errors": [],
            "warnings": [],
            "reconciliation": reconciliation
        }
        
        # Check for critical mismatches
        if reconciliation["added"]:
            validation_report["valid"] = False
            validation_report["errors"].append(
                f"New components found that are not in index: {reconciliation['added']}")
        
        if reconciliation["deleted"]:
            validation_report["valid"] = False
            validation_report["errors"].append(
                f"Components in index but missing from filesystem: {reconciliation['deleted']}")
        
        if reconciliation["modified"]:
            validation_report["warnings"].append(
                f"Modified components detected: {reconciliation['modified']}")
        
        # Validate stage sequence
        stage_violations = self._validate_stage_sequence(components)
        if stage_violations:
            validation_report["valid"] = False
            validation_report["errors"].extend(stage_violations)
        
        # Validate component codes are sequential
        code_violations = self._validate_component_codes(components)
        if code_violations:
            validation_report["warnings"].extend(code_violations)
        
        return validation_report
    
    def _validate_stage_sequence(self, components: List[ComponentInfo]) -> List[str]:
        """Validate that components follow correct stage sequence"""
        violations = []
        dependencies = self.calculate_dependencies(components)
        
        stage_order_map = {stage: i for i, stage in enumerate(self.STAGE_ORDER)}
        
        for comp in components:
            comp_stage_order = stage_order_map.get(comp.stage, -1)
            
            for dep_code in comp.dependencies:
                dep_comp = next((c for c in components if c.code == dep_code), None)
                if dep_comp:
                    dep_stage_order = stage_order_map.get(dep_comp.stage, -1)
                    
                    if dep_stage_order > comp_stage_order:
                        violations.append(
                            f"Stage sequence violation: {comp.code} ({comp.stage}) "
                            f"depends on {dep_code} ({dep_comp.stage})")
        
        return violations
    
    def _validate_component_codes(self, components: List[ComponentInfo]) -> List[str]:
        """Validate component code sequencing"""
        violations = []
        
        # Group by stage and check sequence
        by_stage = defaultdict(list)
        for comp in components:
            by_stage[comp.stage].append(comp)
        
        for stage, stage_comps in by_stage.items():
            stage_comps.sort(key=lambda x: x.code)
            expected_num = 1
            
            for comp in stage_comps:
                code_num = int(comp.code[:2])
                if code_num != expected_num:
                    violations.append(
                        f"Code sequence gap in {stage}: expected {expected_num:02d}, "
                        f"found {code_num:02d} for {comp.code}")
                expected_num = code_num + 1
        
        return violations
    
    def setup_git_hooks(self):
        """Set up git hooks for automatic index reconciliation"""
        git_dir = self.root_dir / ".git"
        if not git_dir.exists():
            logger.warning("Not a git repository - cannot set up hooks")
            return
        
        hooks_dir = git_dir / "hooks"
        hooks_dir.mkdir(exist_ok=True)
        
        # Create pre-commit hook
        pre_commit_hook = hooks_dir / "pre-commit"
        pre_commit_content = f"""#!/bin/bash
# Auto-generated pipeline index reconciliation hook

echo "Running pipeline index reconciliation..."
python3 pipeline_index_system.py --reconcile --validate

if [ $? -ne 0 ]; then
    echo "Pipeline index validation failed - commit aborted"
    exit 1
fi

echo "Pipeline index validation passed"
"""
        
        with open(pre_commit_hook, 'w') as f:
            f.write(pre_commit_content)
        
        pre_commit_hook.chmod(0o755)
        
        # Create post-commit hook for visualization update
        post_commit_hook = hooks_dir / "post-commit"
        post_commit_content = f"""#!/bin/bash
# Auto-generated pipeline visualization update hook

echo "Updating pipeline visualizations..."
python3 pipeline_index_system.py --visualize --quiet
"""
        
        with open(post_commit_hook, 'w') as f:
            f.write(post_commit_content)
            
        post_commit_hook.chmod(0o755)
        
        logger.info("Git hooks installed successfully")

def main():
    """Main CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Pipeline Index System")
    parser.add_argument("--scan", action="store_true", 
                       help="Scan filesystem for components")
    parser.add_argument("--reconcile", action="store_true",
                       help="Reconcile index with filesystem")
    parser.add_argument("--validate", action="store_true",
                       help="Validate index consistency") 
    parser.add_argument("--visualize", action="store_true",
                       help="Generate DAG visualizations")
    parser.add_argument("--setup-hooks", action="store_true",
                       help="Setup git hooks")
    parser.add_argument("--quiet", action="store_true",
                       help="Reduce output verbosity")
    
    args = parser.parse_args()
    
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    
    system = PipelineIndexSystem()
    
    if args.setup_hooks:
        system.setup_git_hooks()
        
    if args.scan or args.reconcile:
        components, reconciliation = system.reconcile_index()
        
        if not args.quiet:
            print(f"Reconciliation Report:")
            print(f"  Added: {len(reconciliation['added'])}")
            print(f"  Modified: {len(reconciliation['modified'])}")
            print(f"  Deleted: {len(reconciliation['deleted'])}")
            print(f"  Unchanged: {len(reconciliation['unchanged'])}")
        
        if args.reconcile:
            system.save_index(components)
    
    if args.validate:
        validation = system.validate_index_consistency()
        
        if not args.quiet:
            print(f"Validation Result: {'PASS' if validation['valid'] else 'FAIL'}")
            
            if validation['errors']:
                print("Errors:")
                for error in validation['errors']:
                    print(f"  - {error}")
                    
            if validation['warnings']:
                print("Warnings:")
                for warning in validation['warnings']:
                    print(f"  - {warning}")
        
        # Exit with error code if validation failed
        if not validation['valid']:
            exit(1)
    
    if args.visualize:
        components = system.load_current_index()
        if not components:
            components, _ = system.reconcile_index()
            system.save_index(components)
        
        system.generate_dag_visualization(components)

if __name__ == "__main__":
    main()
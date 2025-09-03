#!/usr/bin/env python3
"""
Dependency Doctor - Comprehensive Pipeline Architecture Validator

A CLI tool that performs comprehensive pre-commit validation by scanning the codebase for:
- Backward dependencies violating I→X→K→A→L→R→O→G→T→S phase ordering
- Missing mandatory annotations (__phase__, __code__, __stage_order__)
- Signature drift from standard process(data, context) -> Dict[str, Any] interface
- Circular dependency hotspots requiring adapter/bridge patterns

Usage:
    python tools/dep_doctor.py [--auto-fix] [--verbose] [--path PATH]
    
Integration with pre-commit:
    Add to .pre-commit-config.yaml:
    - repo: local
      hooks:
        - id: dep-doctor
          name: Dependency Doctor
          entry: python tools/dep_doctor.py
          language: system
          pass_filenames: false
"""

import argparse
import ast
import os
import sys
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
import subprocess
import json

# Import adapter templates if available
try:
    from .adapter_bridge_templates import (
        generate_adapter, generate_bridge, generate_port_interface, 
        generate_dependency_container
    )
    TEMPLATES_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    TEMPLATES_AVAILABLE = False

# Phase ordering definition I→X→K→A→L→R→O→G→T→S
CANONICAL_PHASE_ORDER = [
    'I',  # Ingestion & Preparation
    'X',  # Context Construction  
    'K',  # Knowledge Extraction
    'A',  # Analysis & NLP
    'L',  # Classification & Evaluation
    'R',  # Search & Retrieval
    'O',  # Orchestration & Control
    'G',  # Aggregation & Reporting
    'T',  # Integration & Storage
    'S'   # Synthesis & Output
]

PHASE_NAMES = {
    'I': 'Ingestion & Preparation',
    'X': 'Context Construction',
    'K': 'Knowledge Extraction', 
    'A': 'Analysis & NLP',
    'L': 'Classification & Evaluation',
    'R': 'Search & Retrieval',
    'O': 'Orchestration & Control',
    'G': 'Aggregation & Reporting',
    'T': 'Integration & Storage',
    'S': 'Synthesis & Output'
}

class DependencyViolation:
    def __init__(self, violation_type: str, file_path: str, details: str, line_number: int = 0):
        self.violation_type = violation_type
        self.file_path = file_path
        self.details = details
        self.line_number = line_number
        
    def __str__(self):
        return f"{self.violation_type}: {self.file_path}:{self.line_number} - {self.details}"

class ArchitectureAnalyzer:
    def __init__(self, root_path: str = "."):
        self.root_path = Path(root_path)
        self.violations: List[DependencyViolation] = []
        self.module_phases: Dict[str, str] = {}
        self.module_codes: Dict[str, str] = {}
        self.module_stage_orders: Dict[str, int] = {}
        self.dependencies: Dict[str, Set[str]] = defaultdict(set)
        
    def scan_codebase(self) -> None:
        """Scan the entire codebase for violations."""
        print("🔍 Scanning codebase for architecture violations...")
        
        for py_file in self.root_path.rglob("*.py"):
            if self._should_skip_file(py_file):
                continue
                
            try:
                self._analyze_file(py_file)
            except Exception as e:
                print(f"⚠️ Warning: Could not analyze {py_file}: {e}")
        
        self._validate_phase_ordering()
        self._detect_circular_dependencies()
    
    def _should_skip_file(self, file_path: Path) -> bool:
        """Determine if file should be skipped in analysis."""
        skip_patterns = [
            "venv/", "__pycache__/", ".git/", "node_modules/",
            "test_", "_test.py", "tests/", ".pytest_cache/",
            "build/", "dist/", ".tox/"
        ]
        
        path_str = str(file_path)
        return any(pattern in path_str for pattern in skip_patterns)
    
    def _analyze_file(self, file_path: Path) -> None:
        """Analyze a single Python file for violations."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            tree = ast.parse(content, filename=str(file_path))
            
            # Extract annotations and dependencies
            self._extract_annotations(file_path, tree, content)
            self._extract_dependencies(file_path, tree)
            self._validate_process_signature(file_path, tree, content)
            
        except SyntaxError as e:
            self.violations.append(
                DependencyViolation(
                    "SYNTAX_ERROR",
                    str(file_path),
                    f"Syntax error: {e}",
                    e.lineno or 0
                )
            )
    
    def _extract_annotations(self, file_path: Path, tree: ast.AST, content: str) -> None:
        """Extract mandatory annotations from module."""
        rel_path = str(file_path.relative_to(self.root_path))
        
        # Check for canonical flow modules
        if "canonical_flow" not in str(file_path):
            return
            
        # Extract phase, code, and stage_order annotations
        phase = self._find_annotation(tree, content, "__phase__")
        code = self._find_annotation(tree, content, "__code__")
        stage_order = self._find_annotation(tree, content, "__stage_order__")
        
        # Store found annotations
        if phase:
            self.module_phases[rel_path] = phase
        else:
            self.violations.append(
                DependencyViolation(
                    "MISSING_PHASE_ANNOTATION",
                    rel_path,
                    "Missing mandatory __phase__ annotation"
                )
            )
        
        if code:
            self.module_codes[rel_path] = code
        else:
            self.violations.append(
                DependencyViolation(
                    "MISSING_CODE_ANNOTATION", 
                    rel_path,
                    "Missing mandatory __code__ annotation"
                )
            )
        
        if stage_order:
            try:
                self.module_stage_orders[rel_path] = int(stage_order)
            except ValueError:
                self.violations.append(
                    DependencyViolation(
                        "INVALID_STAGE_ORDER",
                        rel_path,
                        f"Invalid __stage_order__ value: {stage_order}"
                    )
                )
        else:
            self.violations.append(
                DependencyViolation(
                    "MISSING_STAGE_ORDER_ANNOTATION",
                    rel_path, 
                    "Missing mandatory __stage_order__ annotation"
                )
            )
    
    def _find_annotation(self, tree: ast.AST, content: str, annotation_name: str) -> Optional[str]:
        """Find annotation value in module."""
        # Try AST approach first
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == annotation_name:
                        if isinstance(node.value, ast.Constant):
                            return str(node.value.value)
                        elif isinstance(node.value, ast.Str):  # Python < 3.8
                            return node.value.s
        
        # Fallback to regex
        pattern = rf'{annotation_name}\s*=\s*["\']([^"\']+)["\']'
        match = re.search(pattern, content)
        return match.group(1) if match else None
    
    def _extract_dependencies(self, file_path: Path, tree: ast.AST) -> None:
        """Extract import dependencies from module."""
        rel_path = str(file_path.relative_to(self.root_path))
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if self._is_internal_import(alias.name):
                        self.dependencies[rel_path].add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module and self._is_internal_import(node.module):
                    self.dependencies[rel_path].add(node.module)
    
    def _is_internal_import(self, import_name: str) -> bool:
        """Check if import is internal to the project."""
        internal_prefixes = [
            'canonical_flow', 'src', 'analysis_nlp', 'retrieval_engine',
            'G_aggregation_reporting', 'semantic_reranking'
        ]
        return any(import_name.startswith(prefix) for prefix in internal_prefixes)
    
    def _validate_process_signature(self, file_path: Path, tree: ast.AST, content: str) -> None:
        """Validate process function signature compliance."""
        rel_path = str(file_path.relative_to(self.root_path))
        
        # Look for process function definitions
        process_functions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "process":
                process_functions.append(node)
        
        if not process_functions:
            # Check if this is a module that should have a process function
            if self._should_have_process_function(file_path):
                self.violations.append(
                    DependencyViolation(
                        "MISSING_PROCESS_FUNCTION",
                        rel_path,
                        "Module missing required process(data, context) -> Dict[str, Any] function"
                    )
                )
            return
        
        # Validate each process function signature
        for func in process_functions:
            self._validate_single_process_signature(rel_path, func)
    
    def _should_have_process_function(self, file_path: Path) -> bool:
        """Determine if module should have a process function."""
        # Most canonical flow modules should have process functions
        path_str = str(file_path)
        return ("canonical_flow" in path_str and 
                not any(skip in path_str for skip in ["__init__.py", "test_", "schemas"]))
    
    def _validate_single_process_signature(self, rel_path: str, func: ast.FunctionDef) -> None:
        """Validate a single process function signature."""
        args = func.args
        
        # Expected: def process(data, context=None) or process(self, data, context=None)
        min_args = 2 if args.args and args.args[0].arg == 'self' else 2
        max_args = 3 if args.args and args.args[0].arg == 'self' else 2
        
        actual_args = len(args.args)
        
        if actual_args < min_args:
            self.violations.append(
                DependencyViolation(
                    "INVALID_PROCESS_SIGNATURE",
                    rel_path,
                    f"process() function has {actual_args} args, expected at least {min_args}: (data, context)"
                )
            )
        
        # Check return type annotation if present
        if func.returns:
            return_annotation = ast.unparse(func.returns) if hasattr(ast, 'unparse') else str(func.returns)
            if "Dict[str, Any]" not in return_annotation and "dict" not in return_annotation.lower():
                self.violations.append(
                    DependencyViolation(
                        "INVALID_PROCESS_RETURN_TYPE",
                        rel_path,
                        f"process() should return Dict[str, Any], found: {return_annotation}"
                    )
                )
    
    def _validate_phase_ordering(self) -> None:
        """Validate phase ordering dependencies I→X→K→A→L→R→O→G→T→S."""
        phase_indices = {phase: i for i, phase in enumerate(CANONICAL_PHASE_ORDER)}
        
        for module, deps in self.dependencies.items():
            if module not in self.module_phases:
                continue
                
            module_phase = self.module_phases[module]
            if module_phase not in phase_indices:
                continue
                
            module_phase_idx = phase_indices[module_phase]
            
            for dep in deps:
                if dep not in self.module_phases:
                    continue
                    
                dep_phase = self.module_phases[dep]
                if dep_phase not in phase_indices:
                    continue
                    
                dep_phase_idx = phase_indices[dep_phase]
                
                # Check for backward dependency violation
                if dep_phase_idx > module_phase_idx:
                    self.violations.append(
                        DependencyViolation(
                            "BACKWARD_DEPENDENCY_VIOLATION",
                            module,
                            f"Phase {module_phase} module depends on later phase {dep_phase} module {dep}"
                        )
                    )
    
    def _detect_circular_dependencies(self) -> None:
        """Detect circular dependencies that need adapter patterns."""
        # Build dependency graph
        graph = defaultdict(set)
        for module, deps in self.dependencies.items():
            for dep in deps:
                graph[module].add(dep)
        
        # Find strongly connected components (cycles)
        visited = set()
        rec_stack = set()
        
        def dfs(node, path):
            if node in rec_stack:
                # Found cycle
                cycle_start = path.index(node)
                cycle = path[cycle_start:] + [node]
                self.violations.append(
                    DependencyViolation(
                        "CIRCULAR_DEPENDENCY",
                        " -> ".join(cycle),
                        "Circular dependency detected, consider adapter/bridge pattern"
                    )
                )
                return
                
            if node in visited:
                return
                
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in graph[node]:
                dfs(neighbor, path + [node])
            
            rec_stack.remove(node)
        
        for module in graph:
            if module not in visited:
                dfs(module, [])
    
    def generate_remediation_suggestions(self) -> Dict[str, List[str]]:
        """Generate actionable remediation suggestions."""
        suggestions = defaultdict(list)
        
        for violation in self.violations:
            vtype = violation.violation_type
            
            if vtype == "MISSING_PHASE_ANNOTATION":
                suggestions[violation.file_path].append(
                    "Add __phase__ = 'X' annotation at module level (replace X with appropriate phase)"
                )
                
            elif vtype == "MISSING_CODE_ANNOTATION":
                suggestions[violation.file_path].append(
                    "Add __code__ = 'XXY' annotation at module level (e.g. '05I' for stage 5 ingestion)"
                )
                
            elif vtype == "MISSING_STAGE_ORDER_ANNOTATION":
                suggestions[violation.file_path].append(
                    "Add __stage_order__ = N annotation at module level (integer stage order)"
                )
                
            elif vtype == "MISSING_PROCESS_FUNCTION":
                suggestions[violation.file_path].append(
                    "Add standard process(data, context=None) -> Dict[str, Any] function"
                )
                
            elif vtype == "INVALID_PROCESS_SIGNATURE":
                suggestions[violation.file_path].append(
                    "Update process function signature to: def process(data, context=None) -> Dict[str, Any]"
                )
                
            elif vtype == "BACKWARD_DEPENDENCY_VIOLATION":
                # Extract phases from violation details
                details = violation.details
                phase_match = re.search(r'Phase (\w+) module depends on later phase (\w+)', details)
                if phase_match and TEMPLATES_AVAILABLE:
                    source_phase, target_phase = phase_match.groups()
                    suggestions[violation.file_path].extend([
                        f"Break backward dependency: {details}",
                        f"Consider creating adapter: {source_phase}To{target_phase}Adapter",
                        f"Or use bridge pattern: {source_phase}{target_phase}Bridge",
                        "Use dependency injection container to manage lifecycle"
                    ])
                else:
                    suggestions[violation.file_path].append(
                        f"Break backward dependency: {violation.details}. Consider dependency injection or adapter pattern."
                    )
                
            elif vtype == "CIRCULAR_DEPENDENCY":
                cycle_modules = violation.file_path.split(" -> ")
                cycle_suggestion = "Break circular dependency using:"
                if TEMPLATES_AVAILABLE:
                    cycle_suggestion += "\n  • Adapter/Bridge pattern (use tools/adapter_bridge_templates.py)"
                    cycle_suggestion += "\n  • Port interfaces for hot nodes"
                    cycle_suggestion += "\n  • Dependency injection container"
                else:
                    cycle_suggestion += " adapter/bridge pattern or dependency injection"
                    
                for module in cycle_modules:
                    suggestions[module].append(cycle_suggestion)
        
        return dict(suggestions)
    
    def auto_fix_violations(self) -> int:
        """Automatically fix simple violations where possible."""
        fixes_applied = 0
        
        for violation in self.violations:
            if violation.violation_type in ["MISSING_PHASE_ANNOTATION", "MISSING_CODE_ANNOTATION", "MISSING_STAGE_ORDER_ANNOTATION"]:
                if self._apply_annotation_fix(violation):
                    fixes_applied += 1
            elif violation.violation_type == "MISSING_PROCESS_FUNCTION":
                if self._apply_process_function_fix(violation):
                    fixes_applied += 1
            elif violation.violation_type == "CIRCULAR_DEPENDENCY" and TEMPLATES_AVAILABLE:
                if self._generate_adapter_scaffolding(violation):
                    fixes_applied += 1
        
        return fixes_applied
    
    def _apply_annotation_fix(self, violation: DependencyViolation) -> bool:
        """Apply automatic fix for missing annotations."""
        try:
            file_path = Path(violation.file_path)
            if not file_path.exists():
                return False
                
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Infer values from file path/name
            rel_path = str(file_path)
            
            # Find insertion point (after imports, before first class/function)
            insert_line = self._find_annotation_insertion_point(lines)
            
            if violation.violation_type == "MISSING_PHASE_ANNOTATION":
                phase = self._infer_phase_from_path(rel_path)
                lines.insert(insert_line, f'__phase__ = "{phase}"\n')
                
            elif violation.violation_type == "MISSING_CODE_ANNOTATION":
                code = self._infer_code_from_path(rel_path)
                lines.insert(insert_line, f'__code__ = "{code}"\n')
                
            elif violation.violation_type == "MISSING_STAGE_ORDER_ANNOTATION":
                stage_order = self._infer_stage_order_from_path(rel_path)
                lines.insert(insert_line, f'__stage_order__ = {stage_order}\n')
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)
                
            return True
            
        except Exception as e:
            print(f"⚠️ Could not auto-fix {violation.file_path}: {e}")
            return False
    
    def _apply_process_function_fix(self, violation: DependencyViolation) -> bool:
        """Apply automatic fix for missing process function."""
        try:
            file_path = Path(violation.file_path)
            if not file_path.exists():
                return False
                
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Add basic process function template
            process_template = '''

def process(data=None, context=None) -> Dict[str, Any]:
    """
    Standard process function interface.
    
    Args:
        data: Input data to process
        context: Optional context dictionary
        
    Returns:
        Dict[str, Any]: Processing results
    """
    # TODO: Implement processing logic
    return {
        "status": "success",
        "data": data,
        "processed_at": str(datetime.now())
    }
'''
            
            # Add imports if needed
            imports_to_add = []
            if "from typing import" not in content and "Dict[str, Any]" not in content:
                imports_to_add.append("from typing import Dict, Any, Optional")
            if "import datetime" not in content and "from datetime" not in content:
                imports_to_add.append("from datetime import datetime")
            
            if imports_to_add:
                # Find insertion point for imports
                lines = content.split('\n')
                import_insert_line = self._find_import_insertion_point(lines)
                for imp in imports_to_add:
                    lines.insert(import_insert_line, imp)
                    import_insert_line += 1
                content = '\n'.join(lines)
            
            content += process_template
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
                
            return True
            
        except Exception as e:
            print(f"⚠️ Could not auto-fix process function in {violation.file_path}: {e}")
            return False
    
    def _find_annotation_insertion_point(self, lines: List[str]) -> int:
        """Find appropriate line to insert module-level annotations."""
        # Look for end of imports and docstrings
        in_docstring = False
        docstring_quotes = None
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Track docstring state
            if '"""' in line or "'''" in line:
                quote = '"""' if '"""' in line else "'''"
                if not in_docstring:
                    in_docstring = True
                    docstring_quotes = quote
                elif quote == docstring_quotes:
                    in_docstring = False
                    return i + 1
            
            # Skip if in docstring
            if in_docstring:
                continue
            
            # Skip imports and comments
            if (stripped.startswith(('import ', 'from ')) or 
                stripped.startswith('#') or 
                not stripped):
                continue
                
            # Found first non-import, non-comment line
            return i
        
        return len(lines)
    
    def _find_import_insertion_point(self, lines: List[str]) -> int:
        """Find appropriate line to insert import statements."""
        # Insert after existing imports
        last_import_line = -1
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith(('import ', 'from ')) and not stripped.startswith('#'):
                last_import_line = i
        
        return last_import_line + 1 if last_import_line >= 0 else 0
    
    def _infer_phase_from_path(self, file_path: str) -> str:
        """Infer phase from file path."""
        phase_mappings = {
            'I_ingestion_preparation': 'I',
            'X_context_construction': 'X', 
            'K_knowledge_extraction': 'K',
            'A_analysis_nlp': 'A',
            'L_classification_evaluation': 'L',
            'R_search_retrieval': 'R',
            'O_orchestration_control': 'O',
            'G_aggregation_reporting': 'G',
            'T_integration_storage': 'T',
            'S_synthesis_output': 'S'
        }
        
        for pattern, phase in phase_mappings.items():
            if pattern in file_path:
                return phase
        
        return 'O'  # Default to orchestration
    
    def _infer_code_from_path(self, file_path: str) -> str:
        """Infer code from file path."""
        # Extract stage order if present in filename
        import re
        match = re.search(r'(\d+)([IXKALROGTSX])', os.path.basename(file_path))
        if match:
            return f"{match.group(1)}{match.group(2)}"
        
        # Default based on phase
        phase = self._infer_phase_from_path(file_path)
        return f"01{phase}"
    
    def _infer_stage_order_from_path(self, file_path: str) -> int:
        """Infer stage order from file path."""
        # Extract stage number if present
        import re
        match = re.search(r'(\d+)[IXKALROGTS]', os.path.basename(file_path))
        if match:
            return int(match.group(1))
        
        return 1  # Default stage order
    
    def _generate_adapter_scaffolding(self, violation: DependencyViolation) -> bool:
        """Generate adapter scaffolding for circular dependencies."""
        try:
            # Parse circular dependency
            cycle_modules = violation.file_path.split(" -> ")
            if len(cycle_modules) < 2:
                return False
            
            # Generate adapter for the first pair in cycle
            source_module = cycle_modules[0]
            target_module = cycle_modules[1]
            
            # Infer phases
            source_phase = self._infer_phase_from_path(source_module)
            target_phase = self._infer_phase_from_path(target_module)
            
            # Generate adapter file
            adapter_name = f"{source_phase}To{target_phase}Adapter"
            adapter_filename = f"adapters/{adapter_name.lower()}.py"
            adapter_path = Path(self.root_path) / adapter_filename
            
            # Create adapters directory if it doesn't exist
            adapter_path.parent.mkdir(exist_ok=True)
            
            if not adapter_path.exists():
                adapter_code = generate_adapter(source_phase, target_phase, adapter_name)
                
                # Add proper imports
                full_adapter_code = '''"""
Auto-generated adapter for breaking circular dependency.
Generated by Dependency Doctor.
"""

from typing import Dict, Any, Optional

''' + adapter_code
                
                with open(adapter_path, 'w', encoding='utf-8') as f:
                    f.write(full_adapter_code)
                
                print(f"📄 Generated adapter scaffolding: {adapter_filename}")
                return True
            
        except Exception as e:
            print(f"⚠️ Could not generate adapter scaffolding: {e}")
            
        return False


def print_violations_report(violations: List[DependencyViolation], verbose: bool = False) -> None:
    """Print formatted violations report."""
    if not violations:
        print("✅ No architecture violations detected!")
        return
    
    print(f"\n🚨 Found {len(violations)} architecture violations:\n")
    
    # Group by violation type
    by_type = defaultdict(list)
    for v in violations:
        by_type[v.violation_type].append(v)
    
    for vtype, vlist in sorted(by_type.items()):
        print(f"📋 {vtype.replace('_', ' ').title()} ({len(vlist)} violations):")
        
        if verbose:
            for violation in vlist[:10]:  # Limit to first 10 for readability
                print(f"   {violation}")
        else:
            # Show summary
            files = set(v.file_path for v in vlist)
            print(f"   Affected files: {len(files)}")
            if len(vlist) <= 3:
                for v in vlist:
                    print(f"   - {v.file_path}: {v.details}")
        
        if len(vlist) > 10 and verbose:
            print(f"   ... and {len(vlist) - 10} more")
        
        print()


def print_remediation_suggestions(suggestions: Dict[str, List[str]]) -> None:
    """Print remediation suggestions."""
    if not suggestions:
        return
    
    print("🔧 Remediation Suggestions:\n")
    
    for file_path, file_suggestions in list(suggestions.items())[:10]:  # Limit output
        print(f"📁 {file_path}:")
        for suggestion in file_suggestions:
            print(f"   • {suggestion}")
        print()
    
    if len(suggestions) > 10:
        print(f"   ... and {len(suggestions) - 10} more files need attention")


def main():
    parser = argparse.ArgumentParser(description="Dependency Doctor - Architecture Validator")
    parser.add_argument("--auto-fix", action="store_true", help="Automatically fix simple violations")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--path", default=".", help="Path to scan (default: current directory)")
    parser.add_argument("--output", help="Output JSON report to file")
    
    args = parser.parse_args()
    
    print("🏥 Dependency Doctor - Architecture Compliance Validator")
    print("=" * 60)
    
    analyzer = ArchitectureAnalyzer(args.path)
    analyzer.scan_codebase()
    
    # Print violations report
    print_violations_report(analyzer.violations, args.verbose)
    
    # Generate and print remediation suggestions
    suggestions = analyzer.generate_remediation_suggestions()
    print_remediation_suggestions(suggestions)
    
    # Auto-fix if requested
    if args.auto_fix and analyzer.violations:
        print("🔧 Applying automatic fixes...")
        fixes_applied = analyzer.auto_fix_violations()
        print(f"✅ Applied {fixes_applied} automatic fixes")
        
        if fixes_applied > 0:
            print("🔄 Re-scanning to verify fixes...")
            analyzer = ArchitectureAnalyzer(args.path)
            analyzer.scan_codebase()
            print(f"📊 Remaining violations: {len(analyzer.violations)}")
    
    # Output JSON report if requested
    if args.output:
        report = {
            "violations": [
                {
                    "type": v.violation_type,
                    "file": v.file_path,
                    "details": v.details,
                    "line": v.line_number
                }
                for v in analyzer.violations
            ],
            "suggestions": suggestions,
            "summary": {
                "total_violations": len(analyzer.violations),
                "violation_types": len(set(v.violation_type for v in analyzer.violations)),
                "files_affected": len(set(v.file_path for v in analyzer.violations))
            }
        }
        
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"📄 Report saved to {args.output}")
    
    # Exit with appropriate code for CI/CD integration
    exit_code = 0 if len(analyzer.violations) == 0 else 1
    
    if exit_code == 0:
        print("🎉 Architecture validation passed!")
    else:
        print(f"❌ Architecture validation failed with {len(analyzer.violations)} violations")
        print("💡 Run with --auto-fix to automatically resolve simple issues")
        print("💡 Run with --verbose for detailed violation descriptions")
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
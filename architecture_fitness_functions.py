#!/usr/bin/env python3
"""
Architecture Fitness Functions for Phase Layering Enforcement

This module implements comprehensive architecture fitness functions that validate
the canonical phase sequence I→X→K→A→L→R→O→G→T→S and prevent backward dependencies.
"""

import ast
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import argparse


@dataclass
class DependencyViolation:
    """Represents a detected dependency violation."""
    source_phase: str
    target_phase: str
    source_file: str
    target_file: str
    line_number: int
    import_statement: str
    severity: str = "ERROR"


@dataclass
class ValidationResult:
    """Contains the results of architecture validation."""
    passed: bool
    violations: List[DependencyViolation]
    phase_order_violations: int
    backward_dependencies: int
    total_files_scanned: int
    execution_time_ms: float


class PhaseLayeringValidator:
    """Validates phase layering constraints in the canonical flow architecture."""
    
    # Canonical phase sequence I→X→K→A→L→R→O→G→T→S
    CANONICAL_PHASES = [
        "I_ingestion_preparation",
        "X_context_construction", 
        "K_knowledge_extraction",
        "A_analysis_nlp",
        "L_classification_evaluation",
        "R_search_retrieval",
        "O_orchestration_control",
        "G_aggregation_reporting",
        "T_integration_storage",
        "S_synthesis_output"
    ]
    
    def __init__(self, root_path: Path = None):
        """Initialize the validator with the project root path."""
        self.root_path = root_path or Path.cwd()
        self.canonical_flow_path = self.root_path / "canonical_flow"
        self.phase_indices = {phase: idx for idx, phase in enumerate(self.CANONICAL_PHASES)}
        self.violations = []
        
    def get_phase_from_path(self, file_path: Path) -> Optional[str]:
        """Extract the phase name from a file path."""
        try:
            relative_path = file_path.relative_to(self.canonical_flow_path)
            parts = relative_path.parts
            if len(parts) > 0:
                potential_phase = parts[0]
                if potential_phase in self.CANONICAL_PHASES:
                    return potential_phase
        except ValueError:
            # Path is not relative to canonical_flow
            pass
        return None
    
    def extract_imports_from_file(self, file_path: Path) -> List[Tuple[str, int]]:
        """Extract all import statements from a Python file."""
        imports = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    tree = ast.parse(f.read(), filename=str(file_path))
                except SyntaxError:
                    # Skip files with syntax errors
                    return imports
                
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append((alias.name, node.lineno))
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append((node.module, node.lineno))
                        
        except (UnicodeDecodeError, FileNotFoundError):
            # Skip files that can't be read
            pass
            
        return imports
    
    def is_backward_dependency(self, source_phase: str, target_phase: str) -> bool:
        """Check if importing target_phase from source_phase violates phase ordering."""
        if source_phase not in self.phase_indices or target_phase not in self.phase_indices:
            return False
            
        source_idx = self.phase_indices[source_phase]
        target_idx = self.phase_indices[target_phase]
        
        # Backward dependency if source imports from an earlier phase
        return target_idx < source_idx
    
    def validate_file_dependencies(self, file_path: Path) -> List[DependencyViolation]:
        """Validate dependencies for a single file."""
        violations = []
        source_phase = self.get_phase_from_path(file_path)
        
        if not source_phase:
            return violations
            
        imports = self.extract_imports_from_file(file_path)
        
        for import_name, line_no in imports:
            # Check if import is from canonical_flow
            if import_name.startswith("canonical_flow."):
                parts = import_name.split(".")
                if len(parts) >= 2:
                    target_phase = parts[1]
                    
                    if self.is_backward_dependency(source_phase, target_phase):
                        violation = DependencyViolation(
                            source_phase=source_phase,
                            target_phase=target_phase,
                            source_file=str(file_path),
                            target_file=import_name,
                            line_number=line_no,
                            import_statement=f"import {import_name}",
                            severity="ERROR"
                        )
                        violations.append(violation)
                        
        return violations
    
    def scan_all_files(self) -> List[Path]:
        """Scan all Python files in the canonical flow directory."""
        python_files = []
        if self.canonical_flow_path.exists():
            python_files = list(self.canonical_flow_path.rglob("*.py"))
        return python_files
    
    def validate_architecture(self) -> ValidationResult:
        """Perform comprehensive architecture validation."""
        import time
        start_time = time.time()
        
        all_violations = []
        python_files = self.scan_all_files()
        
        for file_path in python_files:
            file_violations = self.validate_file_dependencies(file_path)
            all_violations.extend(file_violations)
        
        execution_time = (time.time() - start_time) * 1000
        
        phase_order_violations = len(all_violations)
        backward_dependencies = phase_order_violations  # All violations are backward deps
        
        result = ValidationResult(
            passed=len(all_violations) == 0,
            violations=all_violations,
            phase_order_violations=phase_order_violations,
            backward_dependencies=backward_dependencies,
            total_files_scanned=len(python_files),
            execution_time_ms=execution_time
        )
        
        return result
    
    def generate_report(self, result: ValidationResult, output_path: Optional[Path] = None) -> str:
        """Generate a detailed validation report."""
        report_lines = [
            "=" * 80,
            "PHASE LAYERING ARCHITECTURE VALIDATION REPORT",
            "=" * 80,
            f"Canonical Phase Sequence: {' → '.join(self.CANONICAL_PHASES)}",
            "",
            "SUMMARY:",
            f"  Status: {'✅ PASSED' if result.passed else '❌ FAILED'}",
            f"  Files Scanned: {result.total_files_scanned}",
            f"  Phase Order Violations: {result.phase_order_violations}",
            f"  Backward Dependencies: {result.backward_dependencies}",
            f"  Execution Time: {result.execution_time_ms:.2f}ms",
            ""
        ]
        
        if result.violations:
            report_lines.extend([
                "VIOLATIONS DETECTED:",
                "-" * 50
            ])
            
            # Group violations by phase
            violations_by_phase = defaultdict(list)
            for violation in result.violations:
                violations_by_phase[violation.source_phase].append(violation)
            
            for phase in self.CANONICAL_PHASES:
                if phase in violations_by_phase:
                    phase_violations = violations_by_phase[phase]
                    report_lines.extend([
                        f"",
                        f"📍 Phase: {phase} ({len(phase_violations)} violations)"
                    ])
                    
                    for violation in phase_violations:
                        report_lines.extend([
                            f"  ❌ {violation.source_file}:{violation.line_number}",
                            f"     Imports from earlier phase: {violation.target_phase}",
                            f"     Statement: {violation.import_statement}",
                            ""
                        ])
        else:
            report_lines.extend([
                "✅ NO VIOLATIONS DETECTED",
                "Architecture constraints are properly maintained.",
                ""
            ])
        
        report_lines.extend([
            "=" * 80,
            ""
        ])
        
        report_text = "\n".join(report_lines)
        
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(report_text)
        
        return report_text


def run_import_linter() -> Tuple[bool, str]:
    """Run import-linter and return results."""
    try:
        result = subprocess.run(
            ["lint-imports"], 
            capture_output=True, 
            text=True, 
            cwd=Path.cwd()
        )
        
        return result.returncode == 0, result.stdout + result.stderr
        
    except FileNotFoundError:
        return False, "import-linter not found. Install with: pip install import-linter"


def main():
    """Main entry point for architecture validation."""
    parser = argparse.ArgumentParser(description="Validate phase layering architecture")
    parser.add_argument("--root-path", type=Path, help="Root path of the project")
    parser.add_argument("--output", "-o", type=Path, help="Output file for report")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    parser.add_argument("--fail-on-violations", action="store_true", 
                       help="Exit with non-zero code if violations found")
    parser.add_argument("--run-import-linter", action="store_true",
                       help="Also run import-linter validation")
    
    args = parser.parse_args()
    
    # Initialize validator
    validator = PhaseLayeringValidator(args.root_path)
    
    # Run validation
    result = validator.validate_architecture()
    
    # Run import-linter if requested
    import_linter_passed = True
    import_linter_output = ""
    if args.run_import_linter:
        import_linter_passed, import_linter_output = run_import_linter()
    
    # Generate output
    if args.json:
        output_data = {
            "fitness_functions": {
                "passed": result.passed,
                "violations": [
                    {
                        "source_phase": v.source_phase,
                        "target_phase": v.target_phase,
                        "source_file": v.source_file,
                        "target_file": v.target_file,
                        "line_number": v.line_number,
                        "import_statement": v.import_statement,
                        "severity": v.severity
                    } for v in result.violations
                ],
                "phase_order_violations": result.phase_order_violations,
                "backward_dependencies": result.backward_dependencies,
                "total_files_scanned": result.total_files_scanned,
                "execution_time_ms": result.execution_time_ms
            }
        }
        
        if args.run_import_linter:
            output_data["import_linter"] = {
                "passed": import_linter_passed,
                "output": import_linter_output
            }
        
        json_output = json.dumps(output_data, indent=2)
        
        if args.output:
            with open(args.output, 'w') as f:
                f.write(json_output)
        else:
            print(json_output)
            
    else:
        # Generate text report
        report = validator.generate_report(result, args.output)
        
        if not args.output:
            print(report)
        
        if args.run_import_linter:
            print("\n" + "=" * 80)
            print("IMPORT-LINTER VALIDATION RESULTS")
            print("=" * 80)
            print(f"Status: {'✅ PASSED' if import_linter_passed else '❌ FAILED'}")
            if import_linter_output.strip():
                print("\nOutput:")
                print(import_linter_output)
    
    # Handle exit code
    overall_passed = result.passed and (import_linter_passed if args.run_import_linter else True)
    
    if args.fail_on_violations and not overall_passed:
        sys.exit(1)
    elif not overall_passed:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
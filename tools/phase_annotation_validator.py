#!/usr/bin/env python3
"""
Phase Annotation Validation Utilities

This module provides comprehensive validation utilities for phase annotations
that verify compliance against the project's phase annotation standards.

Features:
- Validates annotation format and consistency
- Checks phase sequence compliance
- Verifies component code uniqueness
- Generates detailed compliance reports
- Supports CI/CD integration

Usage:
    python tools/phase_annotation_validator.py --validate
    python tools/phase_annotation_validator.py --report --output validation_report.json
    python tools/phase_annotation_validator.py --ci-mode  # For CI/CD pipelines
"""

import os
import re
import ast
import json
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple, NamedTuple
from dataclasses import dataclass, asdict
from datetime import datetime
from collections import defaultdict

# Phase annotation standards
CANONICAL_PHASES = {
    'I': {'name': 'Ingestion Preparation', 'order': 1, 'description': 'Data ingestion and preparation'},
    'X': {'name': 'Context Construction', 'order': 2, 'description': 'Context building and immutable state'},
    'K': {'name': 'Knowledge Extraction', 'order': 3, 'description': 'Knowledge graph and entity extraction'},
    'A': {'name': 'Analysis NLP', 'order': 4, 'description': 'NLP analysis and processing'},
    'L': {'name': 'Classification Evaluation', 'order': 5, 'description': 'Classification and scoring'},
    'R': {'name': 'Search Retrieval', 'order': 6, 'description': 'Search and retrieval operations'},
    'O': {'name': 'Orchestration Control', 'order': 7, 'description': 'Workflow orchestration'},
    'G': {'name': 'Aggregation Reporting', 'order': 8, 'description': 'Aggregation and reporting'},
    'T': {'name': 'Integration Storage', 'order': 9, 'description': 'Storage and persistence'},
    'S': {'name': 'Synthesis Output', 'order': 10, 'description': 'Final synthesis and output'}
}

@dataclass
class ValidationViolation:
    """Represents a validation violation."""
    file_path: str
    violation_type: str
    severity: str  # 'error', 'warning', 'info'
    description: str
    line_number: Optional[int] = None
    current_value: Optional[str] = None
    expected_value: Optional[str] = None
    rule_id: Optional[str] = None

@dataclass
class PhaseAnnotationData:
    """Container for extracted phase annotation data."""
    phase: Optional[str] = None
    code: Optional[str] = None
    stage_order: Optional[int] = None
    phase_line: Optional[int] = None
    code_line: Optional[int] = None
    stage_order_line: Optional[int] = None

@dataclass
class ValidationResults:
    """Container for validation results."""
    total_files: int
    files_with_annotations: int
    files_missing_annotations: int
    total_violations: int
    violations_by_severity: Dict[str, int]
    violations_by_type: Dict[str, int]
    phase_distribution: Dict[str, int]
    duplicate_codes: List[Tuple[str, List[str]]]
    violations: List[ValidationViolation]

class PhaseAnnotationValidator:
    """Main validation system for phase annotations."""
    
    def __init__(self, root_dir: str = ".", strict_mode: bool = False):
        self.root_dir = Path(root_dir)
        self.strict_mode = strict_mode
        self.violations: List[ValidationViolation] = []
        self.component_codes: Dict[str, List[str]] = defaultdict(list)
        self.phase_distribution: Dict[str, int] = defaultdict(int)
        
        # Validation rules
        self.validation_rules = {
            'required_annotations': self._validate_required_annotations,
            'phase_format': self._validate_phase_format,
            'code_format': self._validate_code_format,
            'stage_order_consistency': self._validate_stage_order_consistency,
            'code_phase_consistency': self._validate_code_phase_consistency,
            'code_uniqueness': self._validate_code_uniqueness,
            'phase_directory_alignment': self._validate_phase_directory_alignment,
            'annotation_placement': self._validate_annotation_placement
        }
    
    def validate_codebase(self) -> ValidationResults:
        """Validate the entire codebase for phase annotation compliance."""
        print("🔍 Starting comprehensive phase annotation validation...")
        
        python_files = list(self.root_dir.rglob("*.py"))
        python_files = [f for f in python_files if not self._should_skip_file(f)]
        
        print(f"Scanning {len(python_files)} Python files...")
        
        files_with_annotations = 0
        files_missing_annotations = 0
        
        for file_path in python_files:
            annotations = self._extract_annotations(file_path)
            
            if self._has_any_annotation(annotations):
                files_with_annotations += 1
                self._validate_file_annotations(file_path, annotations)
            else:
                files_missing_annotations += 1
                self._add_violation(
                    file_path=str(file_path),
                    violation_type='missing_all_annotations',
                    severity='error',
                    description='File missing all required phase annotations',
                    rule_id='R001'
                )
        
        # Perform cross-file validations
        self._validate_cross_file_constraints()
        
        return self._compile_results(
            total_files=len(python_files),
            files_with_annotations=files_with_annotations,
            files_missing_annotations=files_missing_annotations
        )
    
    def _should_skip_file(self, file_path: Path) -> bool:
        """Determine if a file should be skipped during validation."""
        skip_patterns = [
            '__pycache__', '.git', 'venv', '.venv', 'node_modules',
            '.pytest_cache', '.mypy_cache', 'dist', 'build', '*.egg-info'
        ]
        
        path_str = str(file_path)
        return any(pattern in path_str for pattern in skip_patterns)
    
    def _extract_annotations(self, file_path: Path) -> PhaseAnnotationData:
        """Extract phase annotations from a Python file."""
        annotations = PhaseAnnotationData()
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Use AST parsing for robust extraction
            try:
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.Assign):
                        for target in node.targets:
                            if isinstance(target, ast.Name):
                                if target.id == '__phase__':
                                    if isinstance(node.value, (ast.Str, ast.Constant)):
                                        annotations.phase = node.value.s if hasattr(node.value, 's') else node.value.value
                                        annotations.phase_line = node.lineno
                                elif target.id == '__code__':
                                    if isinstance(node.value, (ast.Str, ast.Constant)):
                                        annotations.code = node.value.s if hasattr(node.value, 's') else node.value.value
                                        annotations.code_line = node.lineno
                                elif target.id == '__stage_order__':
                                    if isinstance(node.value, (ast.Num, ast.Constant)):
                                        annotations.stage_order = node.value.n if hasattr(node.value, 'n') else node.value.value
                                        annotations.stage_order_line = node.lineno
                                        
            except SyntaxError:
                # Fallback to regex for files with syntax errors
                self._extract_annotations_regex(content, annotations)
                
        except Exception as e:
            self._add_violation(
                file_path=str(file_path),
                violation_type='extraction_error',
                severity='warning',
                description=f'Error extracting annotations: {str(e)}',
                rule_id='R999'
            )
        
        return annotations
    
    def _extract_annotations_regex(self, content: str, annotations: PhaseAnnotationData):
        """Fallback regex-based annotation extraction."""
        phase_match = re.search(r'__phase__\s*=\s*["\']([IXKALROGTS])["\']', content)
        if phase_match:
            annotations.phase = phase_match.group(1)
            annotations.phase_line = content[:phase_match.start()].count('\n') + 1
            
        code_match = re.search(r'__code__\s*=\s*["\'](\d{2}[IXKALROGTS])["\']', content)
        if code_match:
            annotations.code = code_match.group(1)
            annotations.code_line = content[:code_match.start()].count('\n') + 1
            
        order_match = re.search(r'__stage_order__\s*=\s*(\d+)', content)
        if order_match:
            annotations.stage_order = int(order_match.group(1))
            annotations.stage_order_line = content[:order_match.start()].count('\n') + 1
    
    def _has_any_annotation(self, annotations: PhaseAnnotationData) -> bool:
        """Check if any phase annotations are present."""
        return (annotations.phase is not None or 
                annotations.code is not None or 
                annotations.stage_order is not None)
    
    def _validate_file_annotations(self, file_path: Path, annotations: PhaseAnnotationData):
        """Validate annotations for a single file."""
        file_str = str(file_path)
        
        # Run all validation rules
        for rule_name, rule_func in self.validation_rules.items():
            try:
                rule_func(file_str, annotations)
            except Exception as e:
                self._add_violation(
                    file_path=file_str,
                    violation_type='rule_error',
                    severity='warning', 
                    description=f'Error in rule {rule_name}: {str(e)}',
                    rule_id='R998'
                )
        
        # Track component codes and phases
        if annotations.code:
            self.component_codes[annotations.code].append(file_str)
        if annotations.phase:
            self.phase_distribution[annotations.phase] += 1
    
    def _validate_required_annotations(self, file_path: str, annotations: PhaseAnnotationData):
        """Validate that all required annotations are present."""
        if annotations.phase is None:
            self._add_violation(
                file_path=file_path,
                violation_type='missing_phase',
                severity='error',
                description='Missing required __phase__ annotation',
                rule_id='R002'
            )
        
        if annotations.code is None:
            self._add_violation(
                file_path=file_path,
                violation_type='missing_code',
                severity='error',
                description='Missing required __code__ annotation',
                rule_id='R003'
            )
        
        if annotations.stage_order is None:
            self._add_violation(
                file_path=file_path,
                violation_type='missing_stage_order',
                severity='error',
                description='Missing required __stage_order__ annotation',
                rule_id='R004'
            )
    
    def _validate_phase_format(self, file_path: str, annotations: PhaseAnnotationData):
        """Validate phase annotation format."""
        if annotations.phase is None:
            return
            
        if annotations.phase not in CANONICAL_PHASES:
            self._add_violation(
                file_path=file_path,
                violation_type='invalid_phase',
                severity='error',
                description=f"Invalid phase '{annotations.phase}'. Must be one of: {', '.join(CANONICAL_PHASES.keys())}",
                current_value=annotations.phase,
                line_number=annotations.phase_line,
                rule_id='R005'
            )
    
    def _validate_code_format(self, file_path: str, annotations: PhaseAnnotationData):
        """Validate code annotation format."""
        if annotations.code is None:
            return
            
        if not re.match(r'^\d{2}[IXKALROGTS]$', annotations.code):
            self._add_violation(
                file_path=file_path,
                violation_type='invalid_code_format',
                severity='error',
                description=f"Invalid code format '{annotations.code}'. Must follow pattern: NN[PHASE] (e.g., '01I', '25A')",
                current_value=annotations.code,
                line_number=annotations.code_line,
                rule_id='R006'
            )
    
    def _validate_stage_order_consistency(self, file_path: str, annotations: PhaseAnnotationData):
        """Validate stage order consistency with phase."""
        if annotations.phase is None or annotations.stage_order is None:
            return
            
        expected_order = CANONICAL_PHASES[annotations.phase]['order']
        if annotations.stage_order != expected_order:
            self._add_violation(
                file_path=file_path,
                violation_type='stage_order_mismatch',
                severity='error',
                description=f"Stage order {annotations.stage_order} doesn't match phase '{annotations.phase}' (expected {expected_order})",
                current_value=str(annotations.stage_order),
                expected_value=str(expected_order),
                line_number=annotations.stage_order_line,
                rule_id='R007'
            )
    
    def _validate_code_phase_consistency(self, file_path: str, annotations: PhaseAnnotationData):
        """Validate code suffix matches phase."""
        if annotations.code is None or annotations.phase is None:
            return
            
        code_suffix = annotations.code[-1] if len(annotations.code) >= 1 else ''
        if code_suffix != annotations.phase:
            self._add_violation(
                file_path=file_path,
                violation_type='code_phase_mismatch',
                severity='error',
                description=f"Code suffix '{code_suffix}' doesn't match phase '{annotations.phase}'",
                current_value=annotations.code,
                line_number=annotations.code_line,
                rule_id='R008'
            )
    
    def _validate_code_uniqueness(self, file_path: str, annotations: PhaseAnnotationData):
        """Validate code uniqueness (will be checked in cross-file validation)."""
        # This is handled in _validate_cross_file_constraints
        pass
    
    def _validate_phase_directory_alignment(self, file_path: str, annotations: PhaseAnnotationData):
        """Validate phase aligns with directory structure."""
        if annotations.phase is None:
            return
            
        # Check if file is in a phase-specific directory
        phase_patterns = {
            'I': r'.*[/\\]I_ingestion_preparation',
            'X': r'.*[/\\]X_context_construction',
            'K': r'.*[/\\]K_knowledge_extraction',
            'A': r'.*[/\\]A_analysis_nlp',
            'L': r'.*[/\\]L_classification_evaluation',
            'R': r'.*[/\\]R_search_retrieval',
            'O': r'.*[/\\]O_orchestration_control',
            'G': r'.*[/\\]G_aggregation_reporting',
            'T': r'.*[/\\]T_integration_storage',
            'S': r'.*[/\\]S_synthesis_output'
        }
        
        for phase, pattern in phase_patterns.items():
            if re.search(pattern, file_path, re.IGNORECASE):
                if annotations.phase != phase:
                    self._add_violation(
                        file_path=file_path,
                        violation_type='phase_directory_mismatch',
                        severity='warning',
                        description=f"Phase '{annotations.phase}' doesn't match directory structure (suggests phase '{phase}')",
                        current_value=annotations.phase,
                        expected_value=phase,
                        line_number=annotations.phase_line,
                        rule_id='R009'
                    )
                break
    
    def _validate_annotation_placement(self, file_path: str, annotations: PhaseAnnotationData):
        """Validate annotations are placed correctly in the file."""
        if not any([annotations.phase_line, annotations.code_line, annotations.stage_order_line]):
            return
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            # Check if annotations are near the top of the file (within first 50 lines)
            max_line = max(filter(None, [annotations.phase_line, annotations.code_line, annotations.stage_order_line]))
            
            if max_line > 50:
                self._add_violation(
                    file_path=file_path,
                    violation_type='annotation_placement',
                    severity='warning',
                    description=f"Annotations should be near the top of the file (found at line {max_line})",
                    line_number=max_line,
                    rule_id='R010'
                )
        except:
            pass  # Skip if can't read file
    
    def _validate_cross_file_constraints(self):
        """Validate constraints that span multiple files."""
        # Check for duplicate component codes
        for code, files in self.component_codes.items():
            if len(files) > 1:
                for file_path in files:
                    self._add_violation(
                        file_path=file_path,
                        violation_type='duplicate_code',
                        severity='error',
                        description=f"Duplicate code '{code}' found in {len(files)} files: {', '.join(files)}",
                        current_value=code,
                        rule_id='R011'
                    )
    
    def _add_violation(self, file_path: str, violation_type: str, severity: str, 
                      description: str, rule_id: str, **kwargs):
        """Add a validation violation."""
        violation = ValidationViolation(
            file_path=file_path,
            violation_type=violation_type,
            severity=severity,
            description=description,
            rule_id=rule_id,
            **kwargs
        )
        self.violations.append(violation)
    
    def _compile_results(self, total_files: int, files_with_annotations: int, 
                        files_missing_annotations: int) -> ValidationResults:
        """Compile validation results."""
        violations_by_severity = defaultdict(int)
        violations_by_type = defaultdict(int)
        
        for violation in self.violations:
            violations_by_severity[violation.severity] += 1
            violations_by_type[violation.violation_type] += 1
        
        # Find duplicate codes
        duplicate_codes = [(code, files) for code, files in self.component_codes.items() if len(files) > 1]
        
        return ValidationResults(
            total_files=total_files,
            files_with_annotations=files_with_annotations,
            files_missing_annotations=files_missing_annotations,
            total_violations=len(self.violations),
            violations_by_severity=dict(violations_by_severity),
            violations_by_type=dict(violations_by_type),
            phase_distribution=dict(self.phase_distribution),
            duplicate_codes=duplicate_codes,
            violations=self.violations
        )
    
    def generate_report(self, results: ValidationResults, output_file: Optional[str] = None) -> Dict:
        """Generate a comprehensive validation report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'validation_summary': {
                'total_files_scanned': results.total_files,
                'files_with_annotations': results.files_with_annotations,
                'files_missing_annotations': results.files_missing_annotations,
                'total_violations': results.total_violations,
                'compliance_score': self._calculate_compliance_score(results)
            },
            'violations_by_severity': results.violations_by_severity,
            'violations_by_type': results.violations_by_type,
            'phase_distribution': results.phase_distribution,
            'duplicate_codes': results.duplicate_codes,
            'canonical_phases': CANONICAL_PHASES,
            'violations': [asdict(v) for v in results.violations]
        }
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"📊 Validation report saved to {output_file}")
        
        return report
    
    def _calculate_compliance_score(self, results: ValidationResults) -> float:
        """Calculate a compliance score (0-100)."""
        if results.total_files == 0:
            return 100.0
            
        # Weight different violation types
        severity_weights = {'error': 3, 'warning': 1, 'info': 0.5}
        weighted_violations = sum(
            results.violations_by_severity.get(severity, 0) * weight 
            for severity, weight in severity_weights.items()
        )
        
        # Calculate score based on files and violations
        file_penalty = (results.files_missing_annotations / results.total_files) * 50
        violation_penalty = min(weighted_violations / results.total_files * 10, 50)
        
        score = max(0, 100 - file_penalty - violation_penalty)
        return round(score, 1)
    
    def print_summary(self, results: ValidationResults):
        """Print a validation summary."""
        print("\n" + "="*60)
        print("📋 PHASE ANNOTATION VALIDATION SUMMARY")
        print("="*60)
        
        print(f"\n📊 Files Scanned: {results.total_files}")
        print(f"✅ Files with Annotations: {results.files_with_annotations}")
        print(f"❌ Files Missing Annotations: {results.files_missing_annotations}")
        print(f"🔍 Total Violations: {results.total_violations}")
        print(f"📈 Compliance Score: {self._calculate_compliance_score(results)}%")
        
        if results.violations_by_severity:
            print("\n🚨 Violations by Severity:")
            for severity, count in sorted(results.violations_by_severity.items()):
                emoji = "🔴" if severity == "error" else "🟡" if severity == "warning" else "🔵"
                print(f"  {emoji} {severity.title()}: {count}")
        
        if results.violations_by_type:
            print("\n📝 Top Violation Types:")
            sorted_violations = sorted(results.violations_by_type.items(), key=lambda x: x[1], reverse=True)
            for violation_type, count in sorted_violations[:5]:
                print(f"  • {violation_type.replace('_', ' ').title()}: {count}")
        
        if results.phase_distribution:
            print("\n🏗️  Phase Distribution:")
            for phase, count in sorted(results.phase_distribution.items()):
                phase_name = CANONICAL_PHASES.get(phase, {}).get('name', 'Unknown')
                print(f"  • Phase {phase} ({phase_name}): {count} components")
        
        if results.duplicate_codes:
            print(f"\n⚠️  Duplicate Codes Found: {len(results.duplicate_codes)}")
            for code, files in results.duplicate_codes[:3]:
                print(f"  • {code}: {len(files)} files")
        
        # Exit code for CI
        if results.violations_by_severity.get('error', 0) > 0:
            print("\n❌ Validation failed due to errors!")
            return False
        elif results.total_violations > 0:
            print("\n⚠️  Validation completed with warnings.")
            return True
        else:
            print("\n✅ All phase annotations are compliant!")
            return True

def main():
    parser = argparse.ArgumentParser(description="Phase Annotation Validation Utilities")
    parser.add_argument('--validate', action='store_true', help='Run validation on codebase')
    parser.add_argument('--report', action='store_true', help='Generate detailed report')
    parser.add_argument('--output', type=str, help='Output file for report')
    parser.add_argument('--root', type=str, default='.', help='Root directory to validate')
    parser.add_argument('--strict', action='store_true', help='Enable strict validation mode')
    parser.add_argument('--ci-mode', action='store_true', help='CI mode - exit with error code on failures')
    
    args = parser.parse_args()
    
    if not (args.validate or args.report):
        parser.print_help()
        return
    
    validator = PhaseAnnotationValidator(args.root, args.strict)
    results = validator.validate_codebase()
    
    if args.report:
        validator.generate_report(results, args.output)
    
    success = validator.print_summary(results)
    
    if args.ci_mode and not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
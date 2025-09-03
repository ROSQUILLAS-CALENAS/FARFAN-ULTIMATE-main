#!/usr/bin/env python3
"""
Pipeline validation system that ensures filesystem reality matches index specification.
Fails builds when discrepancies are detected.
"""

import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional
from dataclasses import dataclass, asdict
import hashlib
import subprocess
from datetime import datetime, timezone


@dataclass
class ValidationError:
    """Represents a validation error."""
    error_type: str
    component_name: Optional[str]
    message: str
    severity: str  # 'error', 'warning', 'info'
    file_path: Optional[str] = None


@dataclass 
class ValidationResult:
    """Result of validation process."""
    is_valid: bool
    errors: List[ValidationError]
    warnings: List[ValidationError]
    info: List[ValidationError]
    components_validated: int
    validation_timestamp: str


class PipelineValidationSystem:
    """Validates pipeline consistency between index and filesystem."""
    
    def __init__(self, index_path: str = "pipeline_index.json",
                 canonical_root: str = "canonical_flow",
                 strict_mode: bool = True):
        self.index_path = Path(index_path)
        self.canonical_root = Path(canonical_root)
        self.repo_root = Path.cwd()
        self.strict_mode = strict_mode
        
        # Validation rules configuration
        self.validation_rules = {
            'require_index_file': True,
            'require_canonical_paths': True,
            'validate_file_hashes': True,
            'validate_dependencies': True,
            'validate_phase_consistency': True,
            'validate_code_uniqueness': True,
            'check_orphaned_files': True,
            'validate_dag_integrity': True,
            'require_descriptions': False  # Optional in non-strict mode
        }
    
    def load_index(self) -> Dict[str, Any]:
        """Load and validate index file structure."""
        if not self.index_path.exists():
            raise FileNotFoundError(f"Pipeline index not found: {self.index_path}")
        
        try:
            with open(self.index_path, 'r') as f:
                data = json.load(f)
            
            # Validate basic structure
            required_keys = ['version', 'metadata', 'components']
            for key in required_keys:
                if key not in data:
                    raise ValueError(f"Missing required key in index: {key}")
            
            return data
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in index file: {e}")
    
    def get_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file."""
        if not file_path.exists() or not file_path.is_file():
            return ""
        
        hasher = hashlib.sha256()
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception:
            return ""
    
    def validate_component(self, component: Dict[str, Any]) -> List[ValidationError]:
        """Validate a single component."""
        errors = []
        name = component.get('name', 'unnamed')
        
        # Required fields validation
        required_fields = ['name', 'phase', 'canonical_path']
        for field in required_fields:
            if field not in component:
                errors.append(ValidationError(
                    error_type='missing_field',
                    component_name=name,
                    message=f"Missing required field: {field}",
                    severity='error'
                ))
        
        # Canonical path validation
        canonical_path = component.get('canonical_path')
        if canonical_path:
            path_obj = Path(canonical_path)
            
            if not path_obj.exists():
                errors.append(ValidationError(
                    error_type='missing_file',
                    component_name=name,
                    message=f"Canonical path does not exist: {canonical_path}",
                    severity='error',
                    file_path=canonical_path
                ))
            elif not path_obj.is_file():
                errors.append(ValidationError(
                    error_type='invalid_path',
                    component_name=name, 
                    message=f"Canonical path is not a file: {canonical_path}",
                    severity='error',
                    file_path=canonical_path
                ))
            elif not canonical_path.endswith('.py'):
                errors.append(ValidationError(
                    error_type='invalid_extension',
                    component_name=name,
                    message=f"Component file must be Python (.py): {canonical_path}",
                    severity='warning',
                    file_path=canonical_path
                ))
        
        # Hash validation
        stored_hash = component.get('file_hash')
        if stored_hash and canonical_path:
            actual_hash = self.get_file_hash(Path(canonical_path))
            if stored_hash != actual_hash:
                errors.append(ValidationError(
                    error_type='hash_mismatch',
                    component_name=name,
                    message=f"File hash mismatch (file may have been modified)",
                    severity='warning',
                    file_path=canonical_path
                ))
        
        # Phase validation
        phase = component.get('phase')
        valid_phases = [
            'ingestion_preparation', 'context_construction', 'knowledge_extraction',
            'analysis_nlp', 'classification_evaluation', 'orchestration_control',
            'search_retrieval', 'synthesis_output', 'aggregation_reporting',
            'integration_storage', 'unclassified'
        ]
        
        if phase not in valid_phases:
            errors.append(ValidationError(
                error_type='invalid_phase',
                component_name=name,
                message=f"Invalid phase: {phase}. Must be one of: {', '.join(valid_phases)}",
                severity='error'
            ))
        
        # Code validation
        code = component.get('code')
        if code:
            if not isinstance(code, str) or len(code) < 2:
                errors.append(ValidationError(
                    error_type='invalid_code',
                    component_name=name,
                    message=f"Component code must be non-empty string: {code}",
                    severity='error'
                ))
        
        # Dependencies validation (basic format check)
        dependencies = component.get('dependencies', [])
        if not isinstance(dependencies, list):
            errors.append(ValidationError(
                error_type='invalid_dependencies',
                component_name=name,
                message="Dependencies must be a list",
                severity='error'
            ))
        
        # Description validation (if required)
        if self.strict_mode and not component.get('description'):
            errors.append(ValidationError(
                error_type='missing_description',
                component_name=name,
                message="Component description is required in strict mode",
                severity='warning'
            ))
        
        return errors
    
    def validate_dependency_integrity(self, components: List[Dict[str, Any]]) -> List[ValidationError]:
        """Validate dependency relationships."""
        errors = []
        
        # Build code to component mapping
        code_to_comp = {}
        name_to_comp = {}
        
        for comp in components:
            code = comp.get('code')
            name = comp.get('name')
            
            if code:
                if code in code_to_comp:
                    errors.append(ValidationError(
                        error_type='duplicate_code',
                        component_name=name,
                        message=f"Duplicate component code: {code}",
                        severity='error'
                    ))
                else:
                    code_to_comp[code] = comp
            
            if name:
                name_to_comp[name] = comp
        
        # Validate dependencies exist
        for comp in components:
            comp_name = comp.get('name', 'unnamed')
            dependencies = comp.get('dependencies', [])
            
            for dep_code in dependencies:
                if dep_code not in code_to_comp:
                    errors.append(ValidationError(
                        error_type='missing_dependency',
                        component_name=comp_name,
                        message=f"Dependency not found: {dep_code}",
                        severity='error'
                    ))
        
        # Check for circular dependencies
        cycles = self._detect_circular_dependencies(components)
        for cycle in cycles:
            cycle_str = ' -> '.join(cycle + [cycle[0]])
            errors.append(ValidationError(
                error_type='circular_dependency',
                component_name=cycle[0],
                message=f"Circular dependency detected: {cycle_str}",
                severity='error'
            ))
        
        return errors
    
    def _detect_circular_dependencies(self, components: List[Dict[str, Any]]) -> List[List[str]]:
        """Detect circular dependencies using DFS."""
        # Build dependency graph
        graph = {}
        code_to_name = {}
        
        for comp in components:
            name = comp.get('name')
            code = comp.get('code', name)
            dependencies = comp.get('dependencies', [])
            
            graph[code] = dependencies
            code_to_name[code] = name
        
        cycles = []
        visited = set()
        rec_stack = set()
        path = []
        
        def dfs(node):
            if node in rec_stack:
                # Found a cycle
                cycle_start = path.index(node)
                cycle = [code_to_name.get(c, c) for c in path[cycle_start:]]
                cycles.append(cycle)
                return
            
            if node in visited:
                return
            
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for dep in graph.get(node, []):
                dfs(dep)
            
            path.pop()
            rec_stack.remove(node)
        
        for node in graph:
            if node not in visited:
                dfs(node)
        
        return cycles
    
    def validate_filesystem_consistency(self, components: List[Dict[str, Any]]) -> List[ValidationError]:
        """Validate filesystem consistency."""
        errors = []
        
        # Check for orphaned files in canonical_flow
        indexed_paths = {comp.get('canonical_path') for comp in components}
        indexed_paths.discard(None)
        
        if self.canonical_root.exists():
            for py_file in self.canonical_root.rglob("*.py"):
                if py_file.name == "__init__.py":
                    continue
                
                rel_path = str(py_file.relative_to(self.repo_root))
                
                if rel_path not in indexed_paths:
                    errors.append(ValidationError(
                        error_type='orphaned_file',
                        component_name=None,
                        message=f"Orphaned file not in index: {rel_path}",
                        severity='warning',
                        file_path=rel_path
                    ))
        
        # Check for files outside canonical structure
        for comp in components:
            canonical_path = comp.get('canonical_path', '')
            if canonical_path and not canonical_path.startswith('canonical_flow/'):
                errors.append(ValidationError(
                    error_type='non_canonical_path',
                    component_name=comp.get('name'),
                    message=f"Component not in canonical structure: {canonical_path}",
                    severity='info',
                    file_path=canonical_path
                ))
        
        return errors
    
    def validate_phase_consistency(self, components: List[Dict[str, Any]]) -> List[ValidationError]:
        """Validate phase organization consistency."""
        errors = []
        
        for comp in components:
            name = comp.get('name')
            phase = comp.get('phase')
            canonical_path = comp.get('canonical_path', '')
            
            if phase and canonical_path:
                # Extract phase from path
                path_parts = Path(canonical_path).parts
                if len(path_parts) >= 2 and path_parts[0] == 'canonical_flow':
                    expected_prefix = self._phase_to_prefix(phase)
                    if expected_prefix and not path_parts[1].startswith(expected_prefix):
                        errors.append(ValidationError(
                            error_type='phase_path_mismatch',
                            component_name=name,
                            message=f"Phase {phase} doesn't match path structure: {canonical_path}",
                            severity='warning',
                            file_path=canonical_path
                        ))
        
        return errors
    
    def _phase_to_prefix(self, phase: str) -> Optional[str]:
        """Map phase to directory prefix."""
        phase_prefixes = {
            'ingestion_preparation': 'I_',
            'context_construction': 'X_',
            'knowledge_extraction': 'K_',
            'analysis_nlp': 'A_',
            'classification_evaluation': 'L_',
            'orchestration_control': 'O_',
            'search_retrieval': 'R_',
            'synthesis_output': 'S_',
            'aggregation_reporting': 'G_',
            'integration_storage': 'T_'
        }
        return phase_prefixes.get(phase)
    
    def run_full_validation(self) -> ValidationResult:
        """Run complete validation process."""
        print("🔍 Starting pipeline validation...")
        
        all_errors = []
        all_warnings = []
        all_info = []
        
        try:
            # Load and validate index
            index_data = self.load_index()
            components = index_data.get('components', [])
            
            print(f"📋 Validating {len(components)} components...")
            
            # Validate each component
            for i, component in enumerate(components):
                component_errors = self.validate_component(component)
                
                for error in component_errors:
                    if error.severity == 'error':
                        all_errors.append(error)
                    elif error.severity == 'warning':
                        all_warnings.append(error)
                    else:
                        all_info.append(error)
            
            # Validate dependency integrity
            print("🔗 Validating dependencies...")
            dep_errors = self.validate_dependency_integrity(components)
            
            for error in dep_errors:
                if error.severity == 'error':
                    all_errors.append(error)
                elif error.severity == 'warning':
                    all_warnings.append(error)
                else:
                    all_info.append(error)
            
            # Validate filesystem consistency
            print("📁 Validating filesystem consistency...")
            fs_errors = self.validate_filesystem_consistency(components)
            
            for error in fs_errors:
                if error.severity == 'error':
                    all_errors.append(error)
                elif error.severity == 'warning':
                    all_warnings.append(error)
                else:
                    all_info.append(error)
            
            # Validate phase consistency
            print("📊 Validating phase consistency...")
            phase_errors = self.validate_phase_consistency(components)
            
            for error in phase_errors:
                if error.severity == 'error':
                    all_errors.append(error)
                elif error.severity == 'warning':
                    all_warnings.append(error)
                else:
                    all_info.append(error)
            
        except Exception as e:
            all_errors.append(ValidationError(
                error_type='system_error',
                component_name=None,
                message=f"Validation system error: {e}",
                severity='error'
            ))
            components = []
        
        is_valid = len(all_errors) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            errors=all_errors,
            warnings=all_warnings,
            info=all_info,
            components_validated=len(components),
            validation_timestamp=datetime.now(timezone.utc).isoformat()
        )
    
    def generate_validation_report(self, result: ValidationResult, 
                                  output_path: Optional[str] = None) -> str:
        """Generate detailed validation report."""
        report_lines = [
            "# Pipeline Validation Report",
            f"Generated: {result.validation_timestamp}",
            f"Components Validated: {result.components_validated}",
            f"Overall Status: {'✅ PASS' if result.is_valid else '❌ FAIL'}",
            ""
        ]
        
        if result.errors:
            report_lines.extend([
                f"## Errors ({len(result.errors)})",
                ""
            ])
            
            for error in result.errors:
                report_lines.append(f"**{error.error_type.upper()}**: {error.message}")
                if error.component_name:
                    report_lines.append(f"  - Component: {error.component_name}")
                if error.file_path:
                    report_lines.append(f"  - File: {error.file_path}")
                report_lines.append("")
        
        if result.warnings:
            report_lines.extend([
                f"## Warnings ({len(result.warnings)})",
                ""
            ])
            
            for warning in result.warnings:
                report_lines.append(f"**{warning.error_type.upper()}**: {warning.message}")
                if warning.component_name:
                    report_lines.append(f"  - Component: {warning.component_name}")
                if warning.file_path:
                    report_lines.append(f"  - File: {warning.file_path}")
                report_lines.append("")
        
        if result.info:
            report_lines.extend([
                f"## Info ({len(result.info)})",
                ""
            ])
            
            for info in result.info:
                report_lines.append(f"**{info.error_type.upper()}**: {info.message}")
                if info.component_name:
                    report_lines.append(f"  - Component: {info.component_name}")
                if info.file_path:
                    report_lines.append(f"  - File: {info.file_path}")
                report_lines.append("")
        
        report_content = "\n".join(report_lines)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_content)
        
        return report_content
    
    def validate_for_build(self) -> int:
        """Validate for CI/CD build. Returns exit code."""
        result = self.run_full_validation()
        
        # Print summary
        print(f"\n📊 Validation Summary:")
        print(f"  Components: {result.components_validated}")
        print(f"  Errors: {len(result.errors)}")
        print(f"  Warnings: {len(result.warnings)}")
        print(f"  Info: {len(result.info)}")
        
        # Print errors
        if result.errors:
            print(f"\n❌ VALIDATION ERRORS:")
            for error in result.errors:
                print(f"  • {error.error_type}: {error.message}")
                if error.component_name:
                    print(f"    Component: {error.component_name}")
        
        # Print warnings in non-strict mode
        if result.warnings and not self.strict_mode:
            print(f"\n⚠️  WARNINGS:")
            for warning in result.warnings:
                print(f"  • {warning.error_type}: {warning.message}")
        
        # Determine exit code
        if result.errors:
            print(f"\n❌ VALIDATION FAILED")
            return 1
        elif result.warnings and self.strict_mode:
            print(f"\n❌ VALIDATION FAILED (strict mode - warnings treated as errors)")
            return 1
        else:
            print(f"\n✅ VALIDATION PASSED")
            return 0


def main():
    """Main CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Pipeline validation system")
    parser.add_argument("--index", default="pipeline_index.json", help="Index file path")
    parser.add_argument("--canonical-root", default="canonical_flow", help="Canonical flow root")
    parser.add_argument("--strict", action="store_true", help="Strict validation mode")
    parser.add_argument("--report", help="Output validation report to file")
    parser.add_argument("--json-output", help="Output results as JSON")
    parser.add_argument("--build-mode", action="store_true", help="Build validation mode (for CI/CD)")
    
    args = parser.parse_args()
    
    validator = PipelineValidationSystem(
        index_path=args.index,
        canonical_root=args.canonical_root,
        strict_mode=args.strict
    )
    
    if args.build_mode:
        return validator.validate_for_build()
    
    # Run validation
    result = validator.run_full_validation()
    
    # Generate report
    if args.report:
        report = validator.generate_validation_report(result, args.report)
        print(f"📄 Validation report saved to: {args.report}")
    
    # Output JSON
    if args.json_output:
        with open(args.json_output, 'w') as f:
            json.dump(asdict(result), f, indent=2, default=str)
        print(f"📊 JSON results saved to: {args.json_output}")
    
    # Print summary
    print(f"\n📊 Validation Result: {'✅ PASS' if result.is_valid else '❌ FAIL'}")
    print(f"  Errors: {len(result.errors)}")
    print(f"  Warnings: {len(result.warnings)}")
    print(f"  Info: {len(result.info)}")
    
    return 0 if result.is_valid else 1


if __name__ == "__main__":
    sys.exit(main())
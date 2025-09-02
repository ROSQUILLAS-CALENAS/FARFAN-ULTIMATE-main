#!/usr/bin/env python3
"""
Contracts Validation Utility
Automatically verifies all contract imports resolve to their intended canonical modules.
Flags any that still reference old shadowed module names or incorrect package references.
"""

import ast
import importlib.util
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
import json
import re
from dataclasses import dataclass, field


@dataclass
class ImportIssue:
    """Represents an import issue found in contract files."""
    file_path: str
    line_number: int
    import_statement: str
    issue_type: str
    suggested_fix: Optional[str] = None
    severity: str = "warning"  # warning, error, critical


@dataclass
class ValidationResult:
    """Results of contract validation."""
    total_files_scanned: int = 0
    contract_files_found: int = 0
    issues_found: List[ImportIssue] = field(default_factory=list)
    valid_imports: List[str] = field(default_factory=list)
    canonical_mappings: Dict[str, str] = field(default_factory=dict)


class ContractsValidator:
    """Validates contract files for canonical import paths."""
    
    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path.cwd()
        self.canonical_modules = self._discover_canonical_modules()
        self.known_issues = self._load_known_issues()
        
    def _discover_canonical_modules(self) -> Dict[str, str]:
        """Discover canonical module paths in the project."""
        canonical_modules = {}
        
        # Core EGW modules
        egw_core = self.project_root / "egw_query_expansion" / "core"
        if egw_core.exists():
            for py_file in egw_core.glob("*.py"):
                if py_file.name != "__init__.py":
                    module_name = py_file.stem
                    canonical_modules[module_name] = f"egw_query_expansion.core.{module_name}"
        
        # Root-level modules that should be imported directly
        for py_file in self.project_root.glob("*.py"):
            if py_file.name not in ["__init__.py", "setup.py"]:
                module_name = py_file.stem
                canonical_modules[module_name] = module_name
        
        # Add known canonical mappings
        canonical_modules.update({
            "snapshot_manager": "snapshot_manager",
            "contract_validator": "contract_validator",
            "deterministic_router": "egw_query_expansion.core.deterministic_router",
            "immutable_context": "egw_query_expansion.core.immutable_context",
            "conformal_risk_control": "egw_query_expansion.core.conformal_risk_control",
            "permutation_invariant_processor": "egw_query_expansion.core.permutation_invariant_processor",
            "lineage_tracker": "lineage_tracker",
            "evidence_system": "evidence_system",
        })
        
        return canonical_modules
    
    def _load_known_issues(self) -> Dict[str, str]:
        """Load known import issues and their fixes."""
        return {
            # Legacy patterns that need fixing
            r"from\s+canonical_flow\..*\.(\w+)\s+import": r"from \1 import",
            r"import\s+canonical_flow\..*\.(\w+)": r"import \1",
            
            # Non-canonical import patterns
            r"from\s+(\w+)\s+import.*(?:DeterministicRouter|RoutingContext)": 
                r"from egw_query_expansion.core.deterministic_router import",
            r"from\s+(\w+)\s+import.*(?:QuestionContext|create_question_context)": 
                r"from egw_query_expansion.core.immutable_context import",
        }
    
    def validate_all_contracts(self) -> ValidationResult:
        """Validate all contract files in the project."""
        result = ValidationResult()
        
        # Find all contract-related files
        contract_patterns = [
            "**/test_*_contract.py",
            "**/test_*contract*.py", 
            "**/*contract*.py",
            "**/test_routing*.py",
            "**/test_snapshot*.py",
        ]
        
        contract_files = set()
        for pattern in contract_patterns:
            contract_files.update(self.project_root.glob(pattern))
        
        result.contract_files_found = len(contract_files)
        result.total_files_scanned = result.contract_files_found
        
        for contract_file in contract_files:
            file_issues = self._validate_file(contract_file)
            result.issues_found.extend(file_issues)
        
        return result
    
    def _validate_file(self, file_path: Path) -> List[ImportIssue]:
        """Validate imports in a single contract file."""
        issues = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse the AST
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    issue = self._check_import_node(node, file_path, content)
                    if issue:
                        issues.append(issue)
                        
        except Exception as e:
            issues.append(ImportIssue(
                file_path=str(file_path),
                line_number=0,
                import_statement=f"Failed to parse file: {e}",
                issue_type="parse_error",
                severity="error"
            ))
        
        return issues
    
    def _check_import_node(self, node: ast.AST, file_path: Path, content: str) -> Optional[ImportIssue]:
        """Check a single import node for issues."""
        lines = content.split('\n')
        line_number = node.lineno
        import_line = lines[line_number - 1] if line_number <= len(lines) else ""
        
        if isinstance(node, ast.ImportFrom):
            module_name = node.module
            if not module_name:
                return None
                
            # Check for non-canonical paths
            if self._is_non_canonical_import(module_name):
                canonical_path = self._get_canonical_path(module_name)
                return ImportIssue(
                    file_path=str(file_path),
                    line_number=line_number,
                    import_statement=import_line.strip(),
                    issue_type="non_canonical_import",
                    suggested_fix=canonical_path,
                    severity="warning"
                )
            
            # Check for shadowed module references
            if self._is_shadowed_module(module_name):
                return ImportIssue(
                    file_path=str(file_path),
                    line_number=line_number,
                    import_statement=import_line.strip(),
                    issue_type="shadowed_module",
                    suggested_fix=self._resolve_shadowed_module(module_name),
                    severity="error"
                )
        
        elif isinstance(node, ast.Import):
            for alias in node.names:
                module_name = alias.name
                if self._is_non_canonical_import(module_name):
                    canonical_path = self._get_canonical_path(module_name)
                    return ImportIssue(
                        file_path=str(file_path),
                        line_number=line_number,
                        import_statement=import_line.strip(),
                        issue_type="non_canonical_import",
                        suggested_fix=canonical_path,
                        severity="warning"
                    )
        
        return None
    
    def _is_non_canonical_import(self, module_name: str) -> bool:
        """Check if import uses non-canonical path."""
        non_canonical_patterns = [
            r"canonical_flow\..*",
            r".*\..*\..*\..*",  # Too deeply nested
            r"test\..*",  # Should use relative imports in tests
        ]
        
        for pattern in non_canonical_patterns:
            if re.match(pattern, module_name):
                return True
        
        return False
    
    def _is_shadowed_module(self, module_name: str) -> bool:
        """Check if module name is shadowed/ambiguous."""
        shadowed_modules = [
            "contract_validator",  # Could refer to root or alias
            "deterministic_router",  # Should be from egw_query_expansion.core
            "snapshot_manager",  # Should be from root
        ]
        
        return any(module_name.endswith(name) for name in shadowed_modules)
    
    def _get_canonical_path(self, module_name: str) -> str:
        """Get canonical path for a module."""
        # Extract the actual module name from complex paths
        base_module = module_name.split('.')[-1]
        
        if base_module in self.canonical_modules:
            return self.canonical_modules[base_module]
        
        # Fallback: try to determine from known patterns
        if "deterministic_router" in module_name:
            return "egw_query_expansion.core.deterministic_router"
        elif "snapshot_manager" in module_name:
            return "snapshot_manager"
        elif "contract_validator" in module_name:
            return "contract_validator"
        
        return module_name  # Return as-is if no canonical mapping found
    
    def _resolve_shadowed_module(self, module_name: str) -> str:
        """Resolve shadowed module to canonical path."""
        return self._get_canonical_path(module_name)
    
    def fix_import_issues(self, issues: List[ImportIssue], dry_run: bool = True) -> Dict[str, int]:
        """Fix import issues automatically."""
        fixes_applied = {}
        
        for issue in issues:
            if issue.suggested_fix and issue.severity != "error":
                file_path = Path(issue.file_path)
                
                if not dry_run:
                    success = self._apply_fix(file_path, issue)
                    if success:
                        fixes_applied[issue.file_path] = fixes_applied.get(issue.file_path, 0) + 1
                else:
                    print(f"Would fix: {issue.file_path}:{issue.line_number}")
                    print(f"  Current: {issue.import_statement}")
                    print(f"  Fixed:   {issue.suggested_fix}")
        
        return fixes_applied
    
    def _apply_fix(self, file_path: Path, issue: ImportIssue) -> bool:
        """Apply a specific fix to a file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Apply the fix
            line_index = issue.line_number - 1
            if 0 <= line_index < len(lines):
                # Simple string replacement for now
                original_line = lines[line_index]
                
                # Create fixed line based on issue type
                if issue.issue_type == "non_canonical_import":
                    fixed_line = self._create_canonical_import_line(original_line, issue.suggested_fix)
                else:
                    fixed_line = issue.suggested_fix + "\n"
                
                lines[line_index] = fixed_line
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.writelines(lines)
                
                return True
                
        except Exception as e:
            print(f"Failed to fix {file_path}: {e}")
            return False
        
        return False
    
    def _create_canonical_import_line(self, original_line: str, canonical_path: str) -> str:
        """Create a canonical import line from original line."""
        # Extract the import items
        if " import " in original_line:
            import_items = original_line.split(" import ", 1)[1].strip()
            return f"from {canonical_path} import {import_items}\n"
        else:
            return f"import {canonical_path}\n"
    
    def generate_report(self, result: ValidationResult) -> str:
        """Generate a human-readable validation report."""
        report = []
        report.append("=" * 60)
        report.append("CONTRACTS VALIDATION REPORT")
        report.append("=" * 60)
        report.append(f"Files scanned: {result.total_files_scanned}")
        report.append(f"Contract files found: {result.contract_files_found}")
        report.append(f"Issues found: {len(result.issues_found)}")
        report.append("")
        
        if result.issues_found:
            report.append("ISSUES FOUND:")
            report.append("-" * 40)
            
            issues_by_file = {}
            for issue in result.issues_found:
                if issue.file_path not in issues_by_file:
                    issues_by_file[issue.file_path] = []
                issues_by_file[issue.file_path].append(issue)
            
            for file_path, issues in issues_by_file.items():
                report.append(f"\n{file_path}:")
                for issue in issues:
                    severity_mark = "⚠️" if issue.severity == "warning" else "❌"
                    report.append(f"  {severity_mark} Line {issue.line_number}: {issue.issue_type}")
                    report.append(f"     Current: {issue.import_statement}")
                    if issue.suggested_fix:
                        report.append(f"     Suggest: {issue.suggested_fix}")
        else:
            report.append("✅ No issues found! All imports are canonical.")
        
        report.append("")
        report.append("CANONICAL MODULE MAPPINGS:")
        report.append("-" * 40)
        for module, canonical in sorted(self.canonical_modules.items()):
            report.append(f"{module} -> {canonical}")
        
        return "\n".join(report)


def main():
    """Main entry point for the validation utility."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate contract imports for canonical paths")
    parser.add_argument("--project-root", "-p", type=Path, 
                       help="Project root directory (default: current directory)")
    parser.add_argument("--fix", "-f", action="store_true", 
                       help="Apply automatic fixes (default: dry run)")
    parser.add_argument("--output", "-o", type=Path, 
                       help="Output report to file")
    parser.add_argument("--json", action="store_true", 
                       help="Output results as JSON")
    
    args = parser.parse_args()
    
    # Initialize validator
    validator = ContractsValidator(args.project_root)
    
    # Run validation
    print("Scanning contract files for import issues...")
    result = validator.validate_all_contracts()
    
    # Generate report
    if args.json:
        report_data = {
            "total_files_scanned": result.total_files_scanned,
            "contract_files_found": result.contract_files_found,
            "issues_count": len(result.issues_found),
            "issues": [
                {
                    "file": issue.file_path,
                    "line": issue.line_number,
                    "statement": issue.import_statement,
                    "type": issue.issue_type,
                    "suggested_fix": issue.suggested_fix,
                    "severity": issue.severity
                }
                for issue in result.issues_found
            ],
            "canonical_mappings": validator.canonical_modules
        }
        report = json.dumps(report_data, indent=2)
    else:
        report = validator.generate_report(result)
    
    # Output report
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Report written to {args.output}")
    else:
        print(report)
    
    # Apply fixes if requested
    if args.fix and result.issues_found:
        print("\nApplying fixes...")
        fixes = validator.fix_import_issues(result.issues_found, dry_run=False)
        print(f"Applied fixes to {len(fixes)} files")
        for file_path, count in fixes.items():
            print(f"  {file_path}: {count} fixes")
    elif result.issues_found and not args.fix:
        print(f"\nTo fix issues automatically, run with --fix")
    
    # Exit with appropriate code
    sys.exit(0 if not result.issues_found else 1)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Phase Annotation Automated Refactoring System

This script automatically scans the codebase for incorrect, missing, or inconsistent
phase annotations and applies standardized corrections based on predefined patterns.

Features:
- Scans all Python files for phase annotation compliance
- Automatically fixes missing or incorrect annotations
- Validates annotation format and consistency
- Generates detailed reports of changes made
- Supports batch processing and dry-run mode

Usage:
    python tools/phase_annotation_refactor.py --scan
    python tools/phase_annotation_refactor.py --fix --dry-run
    python tools/phase_annotation_refactor.py --fix --apply
"""

import os
import re
import ast
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass
from datetime import datetime

# Phase annotation standards
CANONICAL_PHASES = {
    'I': {'name': 'Ingestion Preparation', 'order': 1},
    'X': {'name': 'Context Construction', 'order': 2},
    'K': {'name': 'Knowledge Extraction', 'order': 3},
    'A': {'name': 'Analysis NLP', 'order': 4},
    'L': {'name': 'Classification Evaluation', 'order': 5},
    'R': {'name': 'Search Retrieval', 'order': 6},
    'O': {'name': 'Orchestration Control', 'order': 7},
    'G': {'name': 'Aggregation Reporting', 'order': 8},
    'T': {'name': 'Integration Storage', 'order': 9},
    'S': {'name': 'Synthesis Output', 'order': 10}
}

# Directory to phase mapping patterns
DIRECTORY_PHASE_PATTERNS = {
    r'.*[/\\]I_ingestion_preparation': 'I',
    r'.*[/\\]X_context_construction': 'X', 
    r'.*[/\\]K_knowledge_extraction': 'K',
    r'.*[/\\]A_analysis_nlp': 'A',
    r'.*[/\\]L_classification_evaluation': 'L',
    r'.*[/\\]R_search_retrieval': 'R',
    r'.*[/\\]O_orchestration_control': 'O',
    r'.*[/\\]G_aggregation_reporting': 'G',
    r'.*[/\\]T_integration_storage': 'T',
    r'.*[/\\]S_synthesis_output': 'S',
}

# File pattern to phase inference
FILE_PHASE_PATTERNS = {
    r'.*ingestion.*|.*ingest.*|.*load.*': 'I',
    r'.*context.*|.*immutable.*': 'X',
    r'.*knowledge.*|.*extract.*|.*embed.*|.*causal.*': 'K',
    r'.*analysis.*|.*nlp.*|.*question.*|.*evidence.*': 'A',
    r'.*classification.*|.*scoring.*|.*evaluation.*': 'L',
    r'.*retrieval.*|.*search.*|.*recommendation.*': 'R',
    r'.*orchestrat.*|.*workflow.*|.*control.*|.*validate.*': 'O',
    r'.*aggregat.*|.*report.*|.*audit.*': 'G',
    r'.*integration.*|.*storage.*|.*metrics.*|.*analytics.*': 'T',
    r'.*synthesis.*|.*output.*|.*answer.*|.*format.*': 'S',
}

@dataclass
class AnnotationIssue:
    """Represents an issue with phase annotations."""
    file_path: str
    issue_type: str
    description: str
    current_value: Optional[str] = None
    suggested_value: Optional[str] = None
    line_number: Optional[int] = None

@dataclass
class PhaseAnnotations:
    """Container for phase annotation values."""
    phase: Optional[str] = None
    code: Optional[str] = None
    stage_order: Optional[int] = None
    phase_line: Optional[int] = None
    code_line: Optional[int] = None
    stage_order_line: Optional[int] = None

class PhaseAnnotationRefactor:
    """Main refactoring system for phase annotations."""
    
    def __init__(self, root_dir: str = "."):
        self.root_dir = Path(root_dir)
        self.issues: List[AnnotationIssue] = []
        self.fixes_applied: List[Dict] = []
        self.phase_counters: Dict[str, int] = {phase: 0 for phase in CANONICAL_PHASES}
        
    def scan_codebase(self) -> List[AnnotationIssue]:
        """Scan entire codebase for annotation issues."""
        print("🔍 Scanning codebase for phase annotation issues...")
        
        python_files = list(self.root_dir.rglob("*.py"))
        python_files = [f for f in python_files if not self._should_skip_file(f)]
        
        print(f"Found {len(python_files)} Python files to scan")
        
        for file_path in python_files:
            self._scan_file(file_path)
            
        return self.issues
    
    def _should_skip_file(self, file_path: Path) -> bool:
        """Determine if a file should be skipped during scanning."""
        skip_patterns = [
            '__pycache__',
            '.git',
            'venv',
            '.venv',
            'node_modules',
            '.pytest_cache',
            '.mypy_cache',
            'dist',
            'build',
            '*.egg-info',
        ]
        
        path_str = str(file_path)
        return any(pattern in path_str for pattern in skip_patterns)
    
    def _scan_file(self, file_path: Path):
        """Scan individual file for annotation issues."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            annotations = self._extract_annotations(content)
            expected_phase = self._infer_phase_from_path(file_path)
            
            # Check for missing annotations
            if not self._has_required_annotations(annotations):
                self._add_missing_annotation_issues(file_path, annotations, expected_phase)
            
            # Check for incorrect annotations
            if annotations.phase:
                self._validate_existing_annotations(file_path, annotations, expected_phase)
                
        except Exception as e:
            self.issues.append(AnnotationIssue(
                file_path=str(file_path),
                issue_type="scan_error",
                description=f"Error scanning file: {str(e)}"
            ))
    
    def _extract_annotations(self, content: str) -> PhaseAnnotations:
        """Extract phase annotations from file content."""
        annotations = PhaseAnnotations()
        
        # Parse using AST for more robust extraction
        try:
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            if target.id == '__phase__' and isinstance(node.value, ast.Str):
                                annotations.phase = node.value.s
                                annotations.phase_line = node.lineno
                            elif target.id == '__code__' and isinstance(node.value, ast.Str):
                                annotations.code = node.value.s  
                                annotations.code_line = node.lineno
                            elif target.id == '__stage_order__' and isinstance(node.value, ast.Num):
                                annotations.stage_order = node.value.n
                                annotations.stage_order_line = node.lineno
        except SyntaxError:
            # Fallback to regex parsing for files with syntax errors
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
        
        return annotations
    
    def _infer_phase_from_path(self, file_path: Path) -> Optional[str]:
        """Infer the expected phase from file path and name."""
        path_str = str(file_path)
        
        # Check directory patterns first
        for pattern, phase in DIRECTORY_PHASE_PATTERNS.items():
            if re.match(pattern, path_str, re.IGNORECASE):
                return phase
        
        # Check filename patterns
        filename = file_path.name.lower()
        for pattern, phase in FILE_PHASE_PATTERNS.items():
            if re.match(pattern, filename, re.IGNORECASE):
                return phase
        
        # Default to orchestration for unclear files
        return 'O'
    
    def _has_required_annotations(self, annotations: PhaseAnnotations) -> bool:
        """Check if all required annotations are present."""
        return (annotations.phase is not None and 
                annotations.code is not None and 
                annotations.stage_order is not None)
    
    def _add_missing_annotation_issues(self, file_path: Path, annotations: PhaseAnnotations, expected_phase: str):
        """Add issues for missing annotations."""
        if annotations.phase is None:
            self.issues.append(AnnotationIssue(
                file_path=str(file_path),
                issue_type="missing_phase",
                description="Missing __phase__ annotation",
                suggested_value=expected_phase
            ))
        
        if annotations.code is None:
            suggested_code = self._generate_component_code(expected_phase)
            self.issues.append(AnnotationIssue(
                file_path=str(file_path),
                issue_type="missing_code",
                description="Missing __code__ annotation",
                suggested_value=suggested_code
            ))
        
        if annotations.stage_order is None:
            suggested_order = CANONICAL_PHASES[expected_phase]['order']
            self.issues.append(AnnotationIssue(
                file_path=str(file_path),
                issue_type="missing_stage_order",
                description="Missing __stage_order__ annotation",
                suggested_value=str(suggested_order)
            ))
    
    def _validate_existing_annotations(self, file_path: Path, annotations: PhaseAnnotations, expected_phase: str):
        """Validate existing annotations for correctness."""
        # Validate phase
        if annotations.phase not in CANONICAL_PHASES:
            self.issues.append(AnnotationIssue(
                file_path=str(file_path),
                issue_type="invalid_phase",
                description=f"Invalid phase '{annotations.phase}'",
                current_value=annotations.phase,
                suggested_value=expected_phase,
                line_number=annotations.phase_line
            ))
        elif annotations.phase != expected_phase:
            self.issues.append(AnnotationIssue(
                file_path=str(file_path),
                issue_type="incorrect_phase",
                description=f"Phase '{annotations.phase}' doesn't match expected '{expected_phase}'",
                current_value=annotations.phase,
                suggested_value=expected_phase,
                line_number=annotations.phase_line
            ))
        
        # Validate code format
        if annotations.code:
            if not re.match(r'\d{2}[IXKALROGTS]$', annotations.code):
                suggested_code = self._generate_component_code(annotations.phase or expected_phase)
                self.issues.append(AnnotationIssue(
                    file_path=str(file_path),
                    issue_type="invalid_code_format",
                    description=f"Invalid code format '{annotations.code}'",
                    current_value=annotations.code,
                    suggested_value=suggested_code,
                    line_number=annotations.code_line
                ))
            elif annotations.phase and annotations.code[-1] != annotations.phase:
                suggested_code = self._generate_component_code(annotations.phase)
                self.issues.append(AnnotationIssue(
                    file_path=str(file_path),
                    issue_type="code_phase_mismatch",
                    description=f"Code suffix '{annotations.code[-1]}' doesn't match phase '{annotations.phase}'",
                    current_value=annotations.code,
                    suggested_value=suggested_code,
                    line_number=annotations.code_line
                ))
        
        # Validate stage order
        if annotations.stage_order and annotations.phase:
            expected_order = CANONICAL_PHASES[annotations.phase]['order']
            if annotations.stage_order != expected_order:
                self.issues.append(AnnotationIssue(
                    file_path=str(file_path),
                    issue_type="incorrect_stage_order",
                    description=f"Stage order {annotations.stage_order} doesn't match phase {annotations.phase} (expected {expected_order})",
                    current_value=str(annotations.stage_order),
                    suggested_value=str(expected_order),
                    line_number=annotations.stage_order_line
                ))
    
    def _generate_component_code(self, phase: str) -> str:
        """Generate a unique component code for the given phase."""
        self.phase_counters[phase] += 1
        return f"{self.phase_counters[phase]:02d}{phase}"
    
    def apply_fixes(self, dry_run: bool = True) -> List[Dict]:
        """Apply automatic fixes to identified issues."""
        print(f"🔧 {'Simulating' if dry_run else 'Applying'} fixes for {len(self.issues)} issues...")
        
        # Group issues by file for efficient processing
        issues_by_file = {}
        for issue in self.issues:
            if issue.file_path not in issues_by_file:
                issues_by_file[issue.file_path] = []
            issues_by_file[issue.file_path].append(issue)
        
        for file_path, file_issues in issues_by_file.items():
            if self._can_auto_fix_file(file_issues):
                fix_result = self._apply_file_fixes(file_path, file_issues, dry_run)
                if fix_result:
                    self.fixes_applied.append(fix_result)
        
        return self.fixes_applied
    
    def _can_auto_fix_file(self, issues: List[AnnotationIssue]) -> bool:
        """Check if all issues in a file can be automatically fixed."""
        auto_fixable_types = {
            'missing_phase', 'missing_code', 'missing_stage_order',
            'incorrect_phase', 'invalid_code_format', 'code_phase_mismatch',
            'incorrect_stage_order'
        }
        return all(issue.issue_type in auto_fixable_types for issue in issues)
    
    def _apply_file_fixes(self, file_path: str, issues: List[AnnotationIssue], dry_run: bool) -> Optional[Dict]:
        """Apply fixes to a single file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            original_content = content
            lines = content.split('\n')
            
            # Sort issues by line number (descending) to avoid offset issues
            issues.sort(key=lambda x: x.line_number or 0, reverse=True)
            
            changes_made = []
            
            for issue in issues:
                if issue.issue_type.startswith('missing_'):
                    # Add missing annotation
                    insertion_point = self._find_annotation_insertion_point(content)
                    annotation_line = self._format_annotation(issue.issue_type, issue.suggested_value)
                    lines.insert(insertion_point, annotation_line)
                    changes_made.append(f"Added {issue.issue_type.replace('missing_', '')}: {issue.suggested_value}")
                    
                elif issue.line_number and issue.suggested_value:
                    # Replace existing annotation
                    old_line = lines[issue.line_number - 1]
                    new_line = self._replace_annotation_value(old_line, issue.issue_type, issue.suggested_value)
                    lines[issue.line_number - 1] = new_line
                    changes_made.append(f"Updated {issue.issue_type}: {issue.current_value} -> {issue.suggested_value}")
            
            new_content = '\n'.join(lines)
            
            if not dry_run and new_content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
            
            return {
                'file_path': file_path,
                'changes': changes_made,
                'issues_fixed': len(issues),
                'dry_run': dry_run
            }
            
        except Exception as e:
            print(f"❌ Error fixing {file_path}: {str(e)}")
            return None
    
    def _find_annotation_insertion_point(self, content: str) -> int:
        """Find the best place to insert new annotations."""
        lines = content.split('\n')
        
        # Look for existing annotations
        for i, line in enumerate(lines):
            if '__phase__' in line or '__code__' in line or '__stage_order__' in line:
                return i
        
        # Look for docstring end
        in_docstring = False
        docstring_end = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('"""') or stripped.startswith("'''"):
                if in_docstring and (stripped.endswith('"""') or stripped.endswith("'''")):
                    docstring_end = i + 1
                    break
                in_docstring = not in_docstring
            elif in_docstring and (stripped.endswith('"""') or stripped.endswith("'''")):
                docstring_end = i + 1
                break
        
        if docstring_end > 0:
            return docstring_end
        
        # Look for first import
        for i, line in enumerate(lines):
            if line.strip().startswith('import ') or line.strip().startswith('from '):
                return i
        
        # Default to beginning of file
        return 0
    
    def _format_annotation(self, annotation_type: str, value: str) -> str:
        """Format an annotation line."""
        if annotation_type == 'missing_phase':
            return f'__phase__ = "{value}"'
        elif annotation_type == 'missing_code':
            return f'__code__ = "{value}"'
        elif annotation_type == 'missing_stage_order':
            return f'__stage_order__ = {value}'
        return ""
    
    def _replace_annotation_value(self, line: str, issue_type: str, new_value: str) -> str:
        """Replace the value in an existing annotation line."""
        if '__phase__' in line:
            return re.sub(r'__phase__\s*=\s*["\'][^"\']*["\']', f'__phase__ = "{new_value}"', line)
        elif '__code__' in line:
            return re.sub(r'__code__\s*=\s*["\'][^"\']*["\']', f'__code__ = "{new_value}"', line)
        elif '__stage_order__' in line:
            return re.sub(r'__stage_order__\s*=\s*\d+', f'__stage_order__ = {new_value}', line)
        return line
    
    def generate_report(self, output_file: Optional[str] = None) -> Dict:
        """Generate a comprehensive report of issues and fixes."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_issues': len(self.issues),
                'fixes_applied': len(self.fixes_applied),
                'files_scanned': len(set(issue.file_path for issue in self.issues)),
                'files_fixed': len(self.fixes_applied)
            },
            'issues_by_type': {},
            'issues_by_file': {},
            'fixes_applied': self.fixes_applied,
            'issues': [
                {
                    'file_path': issue.file_path,
                    'issue_type': issue.issue_type,
                    'description': issue.description,
                    'current_value': issue.current_value,
                    'suggested_value': issue.suggested_value,
                    'line_number': issue.line_number
                }
                for issue in self.issues
            ]
        }
        
        # Group issues by type
        for issue in self.issues:
            issue_type = issue.issue_type
            if issue_type not in report['issues_by_type']:
                report['issues_by_type'][issue_type] = 0
            report['issues_by_type'][issue_type] += 1
        
        # Group issues by file
        for issue in self.issues:
            file_path = issue.file_path
            if file_path not in report['issues_by_file']:
                report['issues_by_file'][file_path] = []
            report['issues_by_file'][file_path].append({
                'type': issue.issue_type,
                'description': issue.description,
                'line': issue.line_number
            })
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"📊 Report saved to {output_file}")
        
        return report
    
    def print_summary(self):
        """Print a summary of findings and actions."""
        print("\n" + "="*60)
        print("📋 PHASE ANNOTATION REFACTOR SUMMARY")
        print("="*60)
        
        if not self.issues:
            print("✅ No issues found! All phase annotations are compliant.")
            return
        
        # Issues by type
        issue_counts = {}
        for issue in self.issues:
            issue_counts[issue.issue_type] = issue_counts.get(issue.issue_type, 0) + 1
        
        print(f"\n🔍 Found {len(self.issues)} issues across {len(set(issue.file_path for issue in self.issues))} files:")
        for issue_type, count in sorted(issue_counts.items()):
            print(f"  • {issue_type.replace('_', ' ').title()}: {count}")
        
        # Fixes applied
        if self.fixes_applied:
            print(f"\n🔧 Applied fixes to {len(self.fixes_applied)} files:")
            for fix in self.fixes_applied:
                print(f"  • {fix['file_path']}: {fix['issues_fixed']} issues fixed")
        
        # Top problematic files
        file_issue_counts = {}
        for issue in self.issues:
            file_issue_counts[issue.file_path] = file_issue_counts.get(issue.file_path, 0) + 1
        
        if file_issue_counts:
            print(f"\n📁 Files with most issues:")
            for file_path, count in sorted(file_issue_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"  • {file_path}: {count} issues")

def main():
    parser = argparse.ArgumentParser(description="Phase Annotation Automated Refactoring System")
    parser.add_argument('--scan', action='store_true', help='Scan codebase for annotation issues')
    parser.add_argument('--fix', action='store_true', help='Apply automatic fixes')
    parser.add_argument('--dry-run', action='store_true', help='Simulate fixes without applying them')
    parser.add_argument('--apply', action='store_true', help='Actually apply fixes to files')
    parser.add_argument('--output', type=str, help='Output file for detailed report')
    parser.add_argument('--root', type=str, default='.', help='Root directory to scan')
    
    args = parser.parse_args()
    
    if not (args.scan or args.fix):
        parser.print_help()
        return
    
    refactor = PhaseAnnotationRefactor(args.root)
    
    # Scan for issues
    if args.scan or args.fix:
        refactor.scan_codebase()
    
    # Apply fixes if requested
    if args.fix:
        dry_run = not args.apply
        refactor.apply_fixes(dry_run=dry_run)
    
    # Generate and show report
    report = refactor.generate_report(args.output)
    refactor.print_summary()

if __name__ == "__main__":
    main()
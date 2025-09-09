#!/usr/bin/env python3
"""
Import Diagnostics Scanner
Comprehensive analysis tool to identify all import issues and refactoring needs
"""

import ast
import os
import sys
import json
import traceback
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict
import importlib.util
import pkgutil
import subprocess

@dataclass
class ImportIssue:
    """Represents a single import issue"""
    file_path: str
    line_number: int
    import_statement: str
    issue_type: str  # 'missing', 'circular', 'relative', 'syntax', 'deep_nested'
    severity: str    # 'critical', 'warning', 'info'
    details: str
    suggested_fix: str = ""

@dataclass
class FileAnalysis:
    """Analysis results for a single file"""
    path: str
    total_imports: int
    failed_imports: List[str]
    circular_imports: List[str]
    relative_imports: List[str]
    deep_imports: List[str]  # >3 levels deep
    external_deps: List[str]
    internal_deps: List[str]
    syntax_errors: List[str]
    complexity_score: float

class ImportDiagnostics:
    """Comprehensive import diagnostics tool"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root).resolve()
        self.issues: List[ImportIssue] = []
        self.file_analyses: Dict[str, FileAnalysis] = {}
        self.dependency_graph: Dict[str, Set[str]] = defaultdict(set)
        self.reverse_deps: Dict[str, Set[str]] = defaultdict(set)
        self.external_packages: Set[str] = set()
        self.missing_packages: Set[str] = set()
        self.import_frequency: Counter = Counter()
        
        # Exclude patterns
        self.exclude_dirs = {'.git', '__pycache__', 'venv', '.venv', 'env', 
                            'build', 'dist', 'node_modules', '.tox', 'eggs'}
        
        # Categorize issues
        self.critical_files: List[str] = []
        self.problem_patterns: Dict[str, List[str]] = defaultdict(list)
        
    def scan_project(self) -> Dict[str, Any]:
        """Run complete diagnostic scan"""
        print("🔍 Starting comprehensive import diagnostics...")
        print(f"📁 Project root: {self.project_root}\n")
        
        # Phase 1: Collect all Python files
        python_files = self._collect_python_files()
        print(f"📊 Found {len(python_files)} Python files to analyze\n")
        
        # Phase 2: Analyze each file
        print("🔬 Analyzing imports in each file...")
        for i, file_path in enumerate(python_files, 1):
            if i % 10 == 0:
                print(f"  Progress: {i}/{len(python_files)} files...")
            self._analyze_file(file_path)
        
        # Phase 3: Detect circular dependencies
        print("\n🔄 Detecting circular dependencies...")
        circular_deps = self._detect_circular_dependencies()
        
        # Phase 4: Analyze import patterns
        print("📈 Analyzing import patterns...")
        patterns = self._analyze_patterns()
        
        # Phase 5: Check external dependencies
        print("📦 Checking external package availability...")
        self._check_external_packages()
        
        # Phase 6: Identify critical issues
        print("⚠️  Identifying critical issues...")
        self._identify_critical_issues()
        
        # Phase 7: Generate refactoring recommendations
        print("💡 Generating refactoring recommendations...")
        recommendations = self._generate_recommendations()
        
        # Compile comprehensive report
        report = self._compile_report(circular_deps, patterns, recommendations)
        
        print("\n✅ Diagnostic scan complete!")
        return report
    
    def _collect_python_files(self) -> List[Path]:
        """Collect all Python files in project"""
        python_files = []
        
        for root, dirs, files in os.walk(self.project_root):
            # Remove excluded directories
            dirs[:] = [d for d in dirs if d not in self.exclude_dirs]
            
            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    # Skip if in excluded path
                    if not any(exc in str(file_path) for exc in self.exclude_dirs):
                        python_files.append(file_path)
        
        return sorted(python_files)
    
    def _analyze_file(self, file_path: Path):
        """Analyze a single Python file for import issues"""
        rel_path = str(file_path.relative_to(self.project_root))
        
        analysis = FileAnalysis(
            path=rel_path,
            total_imports=0,
            failed_imports=[],
            circular_imports=[],
            relative_imports=[],
            deep_imports=[],
            external_deps=[],
            internal_deps=[],
            syntax_errors=[],
            complexity_score=0.0
        )
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST
            tree = ast.parse(content, filename=str(file_path))
            
            # Extract imports
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        self._process_import(
                            file_path, node.lineno, 
                            f"import {alias.name}", 
                            alias.name, analysis
                        )
                
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ''
                    level = node.level  # Number of dots in relative import
                    
                    if level > 0:  # Relative import
                        import_str = '.' * level + module
                        analysis.relative_imports.append(import_str)
                        
                        if level > 2:  # Deep relative import
                            self.issues.append(ImportIssue(
                                file_path=rel_path,
                                line_number=node.lineno,
                                import_statement=import_str,
                                issue_type='relative',
                                severity='warning',
                                details=f"Deep relative import (level {level})",
                                suggested_fix=f"Consider using absolute import"
                            ))
                    
                    # Check for deep nesting
                    if module and module.count('.') >= 3:
                        analysis.deep_imports.append(module)
                        self.issues.append(ImportIssue(
                            file_path=rel_path,
                            line_number=node.lineno,
                            import_statement=f"from {module} import ...",
                            issue_type='deep_nested',
                            severity='warning',
                            details=f"Deeply nested import ({module.count('.')+1} levels)",
                            suggested_fix="Consider flattening module structure"
                        ))
                    
                    # Process the import
                    for alias in node.names:
                        import_name = alias.name
                        full_import = f"from {import_str if level > 0 else module} import {import_name}"
                        self._process_import(
                            file_path, node.lineno,
                            full_import, module or import_str,
                            analysis
                        )
            
            analysis.total_imports = len(analysis.external_deps) + len(analysis.internal_deps)
            
            # Calculate complexity score
            analysis.complexity_score = self._calculate_complexity(analysis)
            
        except SyntaxError as e:
            analysis.syntax_errors.append(f"Line {e.lineno}: {e.msg}")
            self.issues.append(ImportIssue(
                file_path=rel_path,
                line_number=e.lineno or 0,
                import_statement="",
                issue_type='syntax',
                severity='critical',
                details=str(e),
                suggested_fix="Fix syntax error before analyzing imports"
            ))
        
        except Exception as e:
            analysis.syntax_errors.append(str(e))
        
        self.file_analyses[rel_path] = analysis
    
    def _process_import(self, file_path: Path, line_no: int, 
                       import_stmt: str, module_name: str, 
                       analysis: FileAnalysis):
        """Process a single import statement"""
        rel_path = str(file_path.relative_to(self.project_root))
        
        # Track import frequency
        self.import_frequency[module_name] += 1
        
        # Build dependency graph
        self.dependency_graph[rel_path].add(module_name)
        self.reverse_deps[module_name].add(rel_path)
        
        # Determine if internal or external
        is_internal = self._is_internal_module(module_name)
        
        if is_internal:
            analysis.internal_deps.append(module_name)
            
            # Check if internal module exists
            if not self._internal_module_exists(module_name):
                analysis.failed_imports.append(module_name)
                self.issues.append(ImportIssue(
                    file_path=rel_path,
                    line_number=line_no,
                    import_statement=import_stmt,
                    issue_type='missing',
                    severity='critical',
                    details=f"Internal module '{module_name}' not found",
                    suggested_fix=f"Create module or fix import path"
                ))
        else:
            analysis.external_deps.append(module_name)
            base_package = module_name.split('.')[0]
            self.external_packages.add(base_package)
            
            # Check if package is installed
            if not self._is_package_installed(base_package):
                analysis.failed_imports.append(module_name)
                self.missing_packages.add(base_package)
                self.issues.append(ImportIssue(
                    file_path=rel_path,
                    line_number=line_no,
                    import_statement=import_stmt,
                    issue_type='missing',
                    severity='critical',
                    details=f"External package '{base_package}' not installed",
                    suggested_fix=f"pip install {base_package}"
                ))
    
    def _is_internal_module(self, module_name: str) -> bool:
        """Check if module is internal to project"""
        if not module_name:
            return False
        
        # Common external packages
        external_indicators = {
            'numpy', 'pandas', 'torch', 'tensorflow', 'sklearn',
            'matplotlib', 'requests', 'flask', 'django', 'fastapi',
            'pytest', 'setuptools', 'pip', 'transformers', 'langchain'
        }
        
        first_part = module_name.split('.')[0]
        return first_part not in external_indicators and not first_part.startswith('_')
    
    def _internal_module_exists(self, module_name: str) -> bool:
        """Check if internal module exists in project"""
        # Convert module name to path
        parts = module_name.split('.')
        
        # Try as package
        package_path = self.project_root / Path(*parts)
        if package_path.is_dir() and (package_path / '__init__.py').exists():
            return True
        
        # Try as module file
        module_path = self.project_root / Path(*parts[:-1]) / f"{parts[-1]}.py"
        if module_path.exists():
            return True
        
        return False
    
    def _is_package_installed(self, package_name: str) -> bool:
        """Check if a package is installed"""
        try:
            __import__(package_name)
            return True
        except ImportError:
            return False
    
    def _detect_circular_dependencies(self) -> List[List[str]]:
        """Detect circular import dependencies"""
        cycles = []
        visited = set()
        rec_stack = set()
        
        def dfs(node: str, path: List[str]) -> bool:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for neighbor in self.dependency_graph.get(node, []):
                # Check if neighbor is also a file that imports something
                if neighbor in self.dependency_graph:
                    if neighbor not in visited:
                        if dfs(neighbor, path.copy()):
                            return True
                    elif neighbor in rec_stack:
                        # Found cycle
                        cycle_start = path.index(neighbor)
                        cycle = path[cycle_start:] + [neighbor]
                        cycles.append(cycle)
                        
                        # Record in file analyses
                        for file in cycle[:-1]:
                            if file in self.file_analyses:
                                self.file_analyses[file].circular_imports.append(
                                    ' -> '.join(cycle)
                                )
                        
                        # Create issue
                        self.issues.append(ImportIssue(
                            file_path=cycle[0],
                            line_number=0,
                            import_statement=' -> '.join(cycle),
                            issue_type='circular',
                            severity='critical',
                            details=f"Circular dependency detected",
                            suggested_fix="Refactor to remove circular dependency"
                        ))
            
            rec_stack.remove(node)
            return False
        
        # Check all files
        for node in list(self.dependency_graph.keys()):
            if node not in visited:
                dfs(node, [])
        
        return cycles
    
    def _analyze_patterns(self) -> Dict[str, Any]:
        """Analyze import patterns and anti-patterns"""
        patterns = {
            'most_imported': self.import_frequency.most_common(10),
            'import_depth_distribution': defaultdict(int),
            'relative_import_usage': 0,
            'files_with_most_deps': [],
            'files_with_no_deps': [],
            'god_modules': [],  # Files imported by many others
            'orphan_modules': []  # Files not imported by anyone
        }
        
        # Analyze each file
        for file_path, analysis in self.file_analyses.items():
            # Import depth distribution
            for imp in analysis.deep_imports:
                depth = imp.count('.') + 1
                patterns['import_depth_distribution'][depth] += 1
            
            # Relative import usage
            if analysis.relative_imports:
                patterns['relative_import_usage'] += 1
            
            # Files with most dependencies
            total_deps = len(analysis.internal_deps) + len(analysis.external_deps)
            patterns['files_with_most_deps'].append((file_path, total_deps))
            
            if total_deps == 0:
                patterns['files_with_no_deps'].append(file_path)
        
        # Sort files by dependency count
        patterns['files_with_most_deps'].sort(key=lambda x: x[1], reverse=True)
        patterns['files_with_most_deps'] = patterns['files_with_most_deps'][:10]
        
        # Find god modules and orphans
        for module, importers in self.reverse_deps.items():
            if len(importers) > 10:  # Arbitrary threshold
                patterns['god_modules'].append((module, len(importers)))
        
        # Find orphan internal modules
        all_files = set(self.file_analyses.keys())
        imported_files = set()
        for deps in self.dependency_graph.values():
            for dep in deps:
                if self._is_internal_module(dep):
                    imported_files.add(dep)
        
        patterns['orphan_modules'] = list(all_files - imported_files - {'__init__.py'})
        
        return patterns
    
    def _check_external_packages(self):
        """Check availability of external packages"""
        # Try to get installed packages list
        try:
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'list', '--format=json'],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                installed = json.loads(result.stdout)
                installed_names = {pkg['name'].lower() for pkg in installed}
                
                # Update missing packages
                self.missing_packages = {
                    pkg for pkg in self.external_packages 
                    if pkg.lower() not in installed_names
                }
        except Exception as e:
            print(f"  Warning: Could not check pip packages: {e}")
    
    def _calculate_complexity(self, analysis: FileAnalysis) -> float:
        """Calculate import complexity score for a file"""
        score = 0.0
        
        # Factors that increase complexity
        score += len(analysis.failed_imports) * 10  # Failed imports are bad
        score += len(analysis.circular_imports) * 15  # Circular deps are very bad
        score += len(analysis.relative_imports) * 2  # Relative imports add complexity
        score += len(analysis.deep_imports) * 5  # Deep nesting is complex
        score += len(analysis.syntax_errors) * 20  # Syntax errors are critical
        
        # Normalize by total imports (avoid division by zero)
        if analysis.total_imports > 0:
            score = score / analysis.total_imports
        
        return min(score, 100.0)  # Cap at 100
    
    def _identify_critical_issues(self):
        """Identify files with critical issues that need immediate attention"""
        for file_path, analysis in self.file_analyses.items():
            if (analysis.syntax_errors or 
                analysis.failed_imports or 
                analysis.circular_imports or
                analysis.complexity_score > 50):
                self.critical_files.append(file_path)
                
                # Categorize problem patterns
                if analysis.syntax_errors:
                    self.problem_patterns['syntax_errors'].append(file_path)
                if analysis.failed_imports:
                    self.problem_patterns['missing_imports'].append(file_path)
                if analysis.circular_imports:
                    self.problem_patterns['circular_deps'].append(file_path)
                if len(analysis.deep_imports) > 5:
                    self.problem_patterns['deep_nesting'].append(file_path)
                if len(analysis.relative_imports) > 5:
                    self.problem_patterns['excessive_relative'].append(file_path)
    
    def _generate_recommendations(self) -> List[Dict[str, str]]:
        """Generate specific refactoring recommendations"""
        recommendations = []
        
        # 1. Fix syntax errors first
        if self.problem_patterns['syntax_errors']:
            recommendations.append({
                'priority': 'CRITICAL',
                'issue': 'Syntax Errors',
                'files': self.problem_patterns['syntax_errors'][:5],
                'action': 'Fix syntax errors in these files before proceeding',
                'impact': 'Blocking all functionality in affected files'
            })
        
        # 2. Install missing packages
        if self.missing_packages:
            recommendations.append({
                'priority': 'CRITICAL',
                'issue': 'Missing External Packages',
                'packages': list(self.missing_packages)[:10],
                'action': f'pip install {" ".join(list(self.missing_packages)[:10])}',
                'impact': f'{len(self.missing_packages)} packages need installation'
            })
        
        # 3. Fix circular dependencies
        if self.problem_patterns['circular_deps']:
            recommendations.append({
                'priority': 'HIGH',
                'issue': 'Circular Dependencies',
                'files': self.problem_patterns['circular_deps'][:5],
                'action': 'Refactor to break circular import chains',
                'impact': 'Can cause import failures and runtime errors'
            })
        
        # 4. Fix missing internal modules
        missing_internal = [f for f, a in self.file_analyses.items() 
                          if a.failed_imports and any(self._is_internal_module(imp) 
                          for imp in a.failed_imports)]
        if missing_internal:
            recommendations.append({
                'priority': 'HIGH',
                'issue': 'Missing Internal Modules',
                'files': missing_internal[:5],
                'action': 'Create missing modules or fix import paths',
                'impact': 'Broken internal dependencies'
            })
        
        # 5. Refactor deep nesting
        if self.problem_patterns['deep_nesting']:
            recommendations.append({
                'priority': 'MEDIUM',
                'issue': 'Deeply Nested Imports',
                'files': self.problem_patterns['deep_nesting'][:5],
                'action': 'Flatten module structure or use shorter import aliases',
                'impact': 'Complex dependency chains, hard to maintain'
            })
        
        # 6. Convert relative to absolute imports
        if self.problem_patterns['excessive_relative']:
            recommendations.append({
                'priority': 'MEDIUM',
                'issue': 'Excessive Relative Imports',
                'files': self.problem_patterns['excessive_relative'][:5],
                'action': 'Convert to absolute imports for clarity',
                'impact': 'Fragile import structure, breaks easily when moving files'
            })
        
        # 7. Break up god modules
        god_modules = [(m, c) for m, c in self.reverse_deps.items() if len(c) > 10]
        if god_modules:
            god_modules.sort(key=lambda x: len(x[1]), reverse=True)
            recommendations.append({
                'priority': 'LOW',
                'issue': 'God Modules (imported by too many files)',
                'modules': [(m, len(c)) for m, c in god_modules[:5]],
                'action': 'Consider splitting these modules into smaller, focused modules',
                'impact': 'High coupling, changes affect many files'
            })
        
        return recommendations
    
    def _compile_report(self, circular_deps: List[List[str]], 
                       patterns: Dict[str, Any],
                       recommendations: List[Dict[str, str]]) -> Dict[str, Any]:
        """Compile comprehensive diagnostic report"""
        
        # Calculate statistics
        total_files = len(self.file_analyses)
        files_with_issues = len([f for f, a in self.file_analyses.items() 
                                if a.failed_imports or a.syntax_errors or a.circular_imports])
        
        total_imports = sum(a.total_imports for a in self.file_analyses.values())
        failed_imports = sum(len(a.failed_imports) for a in self.file_analyses.values())
        
        # Top problem files
        problem_files = sorted(
            [(f, a.complexity_score) for f, a in self.file_analyses.items()],
            key=lambda x: x[1], reverse=True
        )[:20]
        
        report = {
            'summary': {
                'project_root': str(self.project_root),
                'total_python_files': total_files,
                'files_with_issues': files_with_issues,
                'issue_percentage': round(files_with_issues / total_files * 100, 2) if total_files else 0,
                'total_imports': total_imports,
                'failed_imports': failed_imports,
                'failure_rate': round(failed_imports / total_imports * 100, 2) if total_imports else 0,
                'total_issues': len(self.issues),
                'critical_issues': len([i for i in self.issues if i.severity == 'critical']),
                'circular_dependencies': len(circular_deps),
                'missing_packages': list(self.missing_packages),
                'critical_files_count': len(self.critical_files)
            },
            
            'top_issues': {
                'syntax_errors': len(self.problem_patterns['syntax_errors']),
                'missing_imports': len(self.problem_patterns['missing_imports']),
                'circular_dependencies': len(self.problem_patterns['circular_deps']),
                'deep_nesting': len(self.problem_patterns['deep_nesting']),
                'excessive_relative_imports': len(self.problem_patterns['excessive_relative'])
            },
            
            'problem_files': [
                {
                    'file': f,
                    'complexity_score': round(score, 2),
                    'issues': {
                        'failed_imports': len(self.file_analyses[f].failed_imports),
                        'syntax_errors': len(self.file_analyses[f].syntax_errors),
                        'circular_imports': len(self.file_analyses[f].circular_imports)
                    }
                }
                for f, score in problem_files
            ],
            
            'patterns': {
                'most_imported_modules': patterns['most_imported'],
                'god_modules': patterns['god_modules'],
                'orphan_modules': patterns['orphan_modules'][:10] if patterns['orphan_modules'] else [],
                'files_with_most_dependencies': patterns['files_with_most_deps']
            },
            
            'recommendations': recommendations,
            
            'detailed_issues': [asdict(issue) for issue in self.issues[:100]],  # First 100 issues
            
            'refactoring_targets': {
                'immediate': self.critical_files[:10],
                'high_priority': [f for f, a in self.file_analyses.items() 
                                 if 20 < a.complexity_score <= 50][:10],
                'medium_priority': [f for f, a in self.file_analyses.items() 
                                  if 10 < a.complexity_score <= 20][:10]
            }
        }
        
        return report

def save_report(report: Dict[str, Any], output_path: str = "import_diagnostics_report.json"):
    """Save report to JSON file"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n📄 Detailed report saved to: {output_path}")

def print_summary(report: Dict[str, Any]):
    """Print a summary of the findings"""
    print("\n" + "="*80)
    print("IMPORT DIAGNOSTICS SUMMARY")
    print("="*80)
    
    summary = report['summary']
    print(f"\n📊 Overall Statistics:")
    print(f"  - Total Python files: {summary['total_python_files']}")
    print(f"  - Files with issues: {summary['files_with_issues']} ({summary['issue_percentage']:.1f}%)")
    print(f"  - Total imports: {summary['total_imports']}")
    print(f"  - Failed imports: {summary['failed_imports']} ({summary['failure_rate']:.1f}%)")
    print(f"  - Total issues found: {summary['total_issues']}")
    print(f"  - Critical issues: {summary['critical_issues']}")
    
    print(f"\n⚠️  Top Issues:")
    for issue_type, count in report['top_issues'].items():
        if count > 0:
            print(f"  - {issue_type.replace('_', ' ').title()}: {count} files")
    
    if summary['missing_packages']:
        print(f"\n📦 Missing Packages ({len(summary['missing_packages'])}):")
        for pkg in summary['missing_packages'][:10]:
            print(f"  - {pkg}")
        if len(summary['missing_packages']) > 10:
            print(f"  ... and {len(summary['missing_packages']) - 10} more")
    
    print(f"\n🎯 Immediate Action Required:")
    if report['recommendations']:
        for i, rec in enumerate(report['recommendations'][:3], 1):
            print(f"\n  {i}. [{rec['priority']}] {rec['issue']}")
            print(f"     Action: {rec['action']}")
            print(f"     Impact: {rec['impact']}")
    
    print(f"\n📋 Top 5 Most Problematic Files:")
    for item in report['problem_files'][:5]:
        print(f"  - {item['file']}")
        print(f"    Complexity: {item['complexity_score']}, Issues: {item['issues']}")
    
    print("\n" + "="*80)
    print("See 'import_diagnostics_report.json' for complete details")
    print("="*80)

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive Import Diagnostics Scanner")
    parser.add_argument(
        "project_path",
        nargs="?",
        default=".",
        help="Path to project root (default: current directory)"
    )
    parser.add_argument(
        "-o", "--output",
        default="import_diagnostics_report.json",
        help="Output file for detailed report"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed progress"
    )
    
    args = parser.parse_args()
    
    # Run diagnostics
    diagnostics = ImportDiagnostics(args.project_path)
    report = diagnostics.scan_project()
    
    # Save report
    save_report(report, args.output)
    
    # Print summary
    print_summary(report)
    
    # Return exit code based on critical issues
    if report['summary']['critical_issues'] > 0:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Import Linter Configuration for Controlled Deletion System

This module provides integration with import-linter and custom AST-based
static analysis to enforce import bans on embargoed directories.
"""

import ast
import json
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple
import configparser
try:
    import toml
except ImportError:
    toml = None

try:
    from .controlled_deletion_system import ControlledDeletionManager
except ImportError:
    # Handle direct execution
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from controlled_deletion_system import ControlledDeletionManager


class ImportLinterIntegration:
    """Integration with import-linter tool for embargo enforcement"""
    
    def __init__(self, manager: ControlledDeletionManager):
        self.manager = manager
        self.project_root = manager.project_root
        self.config_path = self.project_root / '.importlinter'
    
    def generate_import_linter_config(self):
        """Generate import-linter configuration to ban embargoed imports"""
        embargoed_modules = []
        
        for directory, record in self.manager.embargo_registry.items():
            if record.status in ['embargoed', 'warning']:
                # Convert directory path to module path
                module_path = directory.replace('/', '.').replace('\\', '.')
                embargoed_modules.append(module_path)
        
        config = {
            'root_packages': ['canonical_flow', 'egw_query_expansion'],
            'contracts': []
        }
        
        # Add contracts to ban imports from embargoed directories
        for module in embargoed_modules:
            contract = {
                'name': f'Ban imports from embargoed {module}',
                'type': 'forbidden',
                'source_modules': ['*'],
                'forbidden_modules': [f'{module}.*'],
                'ignore_imports': []
            }
            config['contracts'].append(contract)
        
        # Write configuration
        if toml:
            with open(self.config_path, 'w') as f:
                toml.dump(config, f)
        else:
            # Fallback to JSON format if toml not available
            import json
            config_path_json = self.config_path.with_suffix('.json')
            with open(config_path_json, 'w') as f:
                json.dump(config, f, indent=2)
    
    def run_import_linter(self) -> Tuple[bool, List[str]]:
        """Run import-linter with generated configuration"""
        import subprocess
        
        self.generate_import_linter_config()
        
        try:
            result = subprocess.run(
                [sys.executable, '-m', 'importlinter'],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            violations = []
            if result.returncode != 0:
                for line in result.stdout.split('\n'):
                    if 'FAILED' in line or 'forbidden' in line.lower():
                        violations.append(line.strip())
            
            return result.returncode == 0, violations
            
        except Exception as e:
            return False, [f"Import linter execution failed: {e}"]


class CustomASTImportChecker:
    """Custom AST-based import checker for more detailed analysis"""
    
    def __init__(self, manager: ControlledDeletionManager):
        self.manager = manager
        self.project_root = manager.project_root
        self.embargoed_patterns = self._get_embargoed_patterns()
    
    def _get_embargoed_patterns(self) -> List[str]:
        """Get patterns for embargoed module paths"""
        patterns = []
        for directory in self.manager.embargo_registry.keys():
            # Convert to module pattern
            pattern = directory.replace('/', '.').replace('\\', '.')
            patterns.append(pattern)
        return patterns
    
    def check_file(self, file_path: Path) -> List[Dict]:
        """Check a single file for banned imports"""
        violations = []
        
        if not file_path.suffix == '.py':
            return violations
        
        try:
            content = file_path.read_text(encoding='utf-8')
            tree = ast.parse(content, filename=str(file_path))
            
            # Walk the AST to find import statements
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if self._is_embargoed_import(alias.name):
                            violations.append({
                                'file': str(file_path),
                                'line': node.lineno,
                                'col': node.col_offset,
                                'type': 'import',
                                'module': alias.name,
                                'message': f'Import of embargoed module: {alias.name}'
                            })
                
                elif isinstance(node, ast.ImportFrom):
                    if node.module and self._is_embargoed_import(node.module):
                        violations.append({
                            'file': str(file_path),
                            'line': node.lineno,
                            'col': node.col_offset,
                            'type': 'from_import',
                            'module': node.module,
                            'message': f'Import from embargoed module: {node.module}'
                        })
        
        except Exception as e:
            violations.append({
                'file': str(file_path),
                'line': 0,
                'col': 0,
                'type': 'parse_error',
                'module': '',
                'message': f'Failed to parse file: {e}'
            })
        
        return violations
    
    def _is_embargoed_import(self, module_name: str) -> bool:
        """Check if a module import is embargoed"""
        for pattern in self.embargoed_patterns:
            if module_name.startswith(pattern):
                return True
        return False
    
    def check_project(self) -> List[Dict]:
        """Check entire project for banned imports"""
        all_violations = []
        
        # Get embargoed directory paths to skip them
        embargoed_paths = [Path(d) for d in self.manager.embargo_registry.keys()]
        
        for py_file in self.project_root.rglob('*.py'):
            # Skip files in embargoed directories
            skip_file = any(
                self._is_path_in_directory(py_file, embargo_path)
                for embargo_path in embargoed_paths
            )
            
            if not skip_file:
                violations = self.check_file(py_file)
                all_violations.extend(violations)
        
        return all_violations
    
    def _is_path_in_directory(self, file_path: Path, directory: Path) -> bool:
        """Check if file path is within directory"""
        try:
            file_path.relative_to(directory)
            return True
        except ValueError:
            return False
    
    def generate_report(self, violations: List[Dict]) -> Dict:
        """Generate a detailed report of import violations"""
        report = {
            'summary': {
                'total_violations': len(violations),
                'files_affected': len(set(v['file'] for v in violations)),
                'embargoed_modules': list(set(v['module'] for v in violations if v['module']))
            },
            'violations_by_file': {},
            'violations_by_module': {}
        }
        
        # Group violations by file
        for violation in violations:
            file_path = violation['file']
            if file_path not in report['violations_by_file']:
                report['violations_by_file'][file_path] = []
            report['violations_by_file'][file_path].append(violation)
        
        # Group violations by embargoed module
        for violation in violations:
            module = violation['module']
            if module and module not in report['violations_by_module']:
                report['violations_by_module'][module] = []
            if module:
                report['violations_by_module'][module].append(violation)
        
        return report


class CIIntegrationHelper:
    """Helper for CI/CD pipeline integration"""
    
    def __init__(self, manager: ControlledDeletionManager):
        self.manager = manager
        self.import_linter = ImportLinterIntegration(manager)
        self.ast_checker = CustomASTImportChecker(manager)
    
    def run_full_check(self) -> Tuple[bool, Dict]:
        """Run full import checking suite"""
        results = {
            'import_linter': {'passed': True, 'violations': []},
            'ast_checker': {'passed': True, 'violations': []},
            'summary': {'total_violations': 0, 'passed': True}
        }
        
        # Run import-linter
        try:
            linter_passed, linter_violations = self.import_linter.run_import_linter()
            results['import_linter']['passed'] = linter_passed
            results['import_linter']['violations'] = linter_violations
        except Exception as e:
            results['import_linter']['passed'] = False
            results['import_linter']['violations'] = [f"Import linter failed: {e}"]
        
        # Run custom AST checker
        try:
            ast_violations = self.ast_checker.check_project()
            results['ast_checker']['passed'] = len(ast_violations) == 0
            results['ast_checker']['violations'] = ast_violations
        except Exception as e:
            results['ast_checker']['passed'] = False
            results['ast_checker']['violations'] = [{'message': f"AST checker failed: {e}"}]
        
        # Calculate summary
        total_violations = (
            len(results['import_linter']['violations']) + 
            len(results['ast_checker']['violations'])
        )
        results['summary']['total_violations'] = total_violations
        results['summary']['passed'] = (
            results['import_linter']['passed'] and 
            results['ast_checker']['passed']
        )
        
        return results['summary']['passed'], results
    
    def generate_ci_report(self, results: Dict) -> str:
        """Generate a CI-friendly report"""
        lines = []
        
        lines.append("🚫 Controlled Deletion System - Import Check Report")
        lines.append("=" * 50)
        
        if results['summary']['passed']:
            lines.append("✅ All checks passed - no embargoed imports detected")
        else:
            lines.append("❌ Import violations detected!")
            lines.append(f"Total violations: {results['summary']['total_violations']}")
        
        # Import linter results
        lines.append("\n📋 Import Linter Results:")
        if results['import_linter']['passed']:
            lines.append("  ✅ Import linter: PASSED")
        else:
            lines.append("  ❌ Import linter: FAILED")
            for violation in results['import_linter']['violations']:
                lines.append(f"    • {violation}")
        
        # AST checker results
        lines.append("\n🔍 AST Checker Results:")
        if results['ast_checker']['passed']:
            lines.append("  ✅ AST checker: PASSED")
        else:
            lines.append("  ❌ AST checker: FAILED")
            for violation in results['ast_checker']['violations']:
                if isinstance(violation, dict):
                    lines.append(f"    • {violation['file']}:{violation['line']} - {violation['message']}")
                else:
                    lines.append(f"    • {violation}")
        
        # Embargo status
        lines.append("\n📅 Current Embargo Status:")
        for directory, record in self.manager.embargo_registry.items():
            status_icon = {
                'embargoed': '🟡',
                'warning': '🟠', 
                'ready_for_deletion': '🔴',
                'deleted': '✅'
            }.get(record.status, '❓')
            
            lines.append(f"  {status_icon} {directory} - {record.status} ({record.days_remaining} days remaining)")
        
        return "\n".join(lines)
    
    def save_junit_xml(self, results: Dict, output_path: Path):
        """Save results in JUnit XML format for CI integration"""
        from xml.etree.ElementTree import Element, SubElement, tostring
        from xml.dom import minidom
        
        testsuites = Element('testsuites')
        
        # Import linter test suite
        linter_suite = SubElement(testsuites, 'testsuite')
        linter_suite.set('name', 'ImportLinter')
        linter_suite.set('tests', '1')
        
        linter_case = SubElement(linter_suite, 'testcase')
        linter_case.set('name', 'embargo_import_check')
        linter_case.set('classname', 'ImportLinter')
        
        if not results['import_linter']['passed']:
            failure = SubElement(linter_case, 'failure')
            failure.set('message', 'Import linter violations detected')
            failure.text = '\n'.join(results['import_linter']['violations'])
        
        # AST checker test suite
        ast_suite = SubElement(testsuites, 'testsuite')
        ast_suite.set('name', 'ASTChecker')
        ast_suite.set('tests', '1')
        
        ast_case = SubElement(ast_suite, 'testcase')
        ast_case.set('name', 'ast_import_check')
        ast_case.set('classname', 'ASTChecker')
        
        if not results['ast_checker']['passed']:
            failure = SubElement(ast_case, 'failure')
            failure.set('message', 'AST checker violations detected')
            failure.text = '\n'.join([
                f"{v.get('file', 'unknown')}:{v.get('line', 0)} - {v.get('message', 'Unknown error')}"
                for v in results['ast_checker']['violations']
            ])
        
        # Write XML
        rough_string = tostring(testsuites, 'utf-8')
        reparsed = minidom.parseString(rough_string)
        
        with open(output_path, 'w') as f:
            f.write(reparsed.toprettyxml(indent="  "))


def main():
    """CLI interface for import linting integration"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Import Linter Integration")
    parser.add_argument('command', choices=['check', 'generate-config', 'ci-report'])
    parser.add_argument('--output', help="Output file for reports")
    parser.add_argument('--format', choices=['json', 'xml', 'text'], default='text')
    
    args = parser.parse_args()
    
    manager = ControlledDeletionManager()
    ci_helper = CIIntegrationHelper(manager)
    
    if args.command == 'check':
        passed, results = ci_helper.run_full_check()
        
        if args.format == 'json':
            output = json.dumps(results, indent=2)
        elif args.format == 'xml':
            ci_helper.save_junit_xml(results, Path(args.output or 'import_check_results.xml'))
            output = "JUnit XML saved"
        else:
            output = ci_helper.generate_ci_report(results)
        
        if args.output:
            Path(args.output).write_text(output)
        else:
            print(output)
        
        sys.exit(0 if passed else 1)
    
    elif args.command == 'generate-config':
        ci_helper.import_linter.generate_import_linter_config()
        print(f"Import linter configuration generated: {ci_helper.import_linter.config_path}")
    
    elif args.command == 'ci-report':
        passed, results = ci_helper.run_full_check()
        report = ci_helper.generate_ci_report(results)
        
        if args.output:
            Path(args.output).write_text(report)
        else:
            print(report)
        
        sys.exit(0 if passed else 1)


if __name__ == '__main__':
    main()
"""
Phase Validation System

Implements runtime validation checks to prevent access to private module internals
across phase boundaries and enforce public API usage.
"""

import ast
import importlib.util
import inspect
import sys
from typing import Any, Dict, List, Set, Tuple
from pathlib import Path
import warnings


class PhaseAccessValidator:
    """Validates that phase access follows public API constraints."""
    
    def __init__(self):
        self.phase_names = {'I', 'X', 'K', 'A', 'L', 'R', 'O', 'G', 'T', 'S'}
        self.violation_cache: Dict[str, List[str]] = {}
    
    def validate_import(self, module_name: str, import_name: str, calling_module: str) -> bool:
        """
        Validate that an import follows phase boundary rules.
        
        Args:
            module_name: The module being imported from
            import_name: The specific item being imported
            calling_module: The module making the import
            
        Returns:
            True if import is allowed, False otherwise
        """
        # Allow imports within same phase
        calling_phase = self._extract_phase(calling_module)
        target_phase = self._extract_phase(module_name)
        
        if calling_phase == target_phase:
            return True
        
        # Allow imports from canonical_flow to phases (backward compatibility)
        if 'canonical_flow' in calling_module and 'phases' in module_name:
            return True
        
        # Block direct cross-phase imports
        if calling_phase and target_phase and calling_phase != target_phase:
            violation = f"Cross-phase direct import: {calling_module} -> {module_name}.{import_name}"
            self._record_violation(calling_module, violation)
            return False
        
        # Block access to private internals
        if self._is_private_access(module_name, import_name):
            violation = f"Private member access: {calling_module} -> {module_name}.{import_name}"
            self._record_violation(calling_module, violation)
            return False
        
        return True
    
    def validate_phase_api_compliance(self, phase: str) -> Tuple[bool, List[str]]:
        """
        Validate that a phase's __init__.py exposes only approved APIs.
        
        Args:
            phase: Phase identifier (I, X, K, etc.)
            
        Returns:
            (is_compliant, violations)
        """
        violations = []
        
        try:
            phase_module_path = Path(f"phases/{phase}/__init__.py")
            if not phase_module_path.exists():
                violations.append(f"Phase {phase} __init__.py not found")
                return False, violations
            
            # Parse the module to check __all__ declaration
            with open(phase_module_path, 'r') as f:
                tree = ast.parse(f.read())
            
            all_declaration = None
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id == '__all__':
                            all_declaration = node
                            break
            
            if all_declaration is None:
                violations.append(f"Phase {phase} missing __all__ declaration")
                return False, violations
            
            # Validate __all__ contains only approved exports
            approved_patterns = [
                f'{phase.title()}Data',
                f'{phase.title()}Context', 
                f'{phase.title()}Processor',
                f'{phase.title()}Phase'
            ]
            
            # Map special cases
            phase_mapping = {
                'I': 'Ingestion',
                'X': 'ContextConstruction',
                'K': 'KnowledgeExtraction',
                'A': 'Analysis',
                'L': 'Classification', 
                'R': 'Retrieval',
                'O': 'Orchestration',
                'G': 'Aggregation',
                'T': 'Integration',
                'S': 'Synthesis'
            }
            
            if phase in phase_mapping:
                base_name = phase_mapping[phase]
                approved_patterns = [
                    f'{base_name}Data',
                    f'{base_name}Context',
                    f'{base_name}Processor', 
                    f'{base_name}Phase'
                ]
            
            # Check exports match approved pattern
            all_value = all_declaration.value
            if isinstance(all_value, ast.List):
                exports = [elt.s for elt in all_value.elts if isinstance(elt, ast.Str)]
                for export in exports:
                    if not any(pattern in export for pattern in approved_patterns):
                        violations.append(f"Unapproved export in phase {phase}: {export}")
        
        except Exception as e:
            violations.append(f"Error validating phase {phase}: {str(e)}")
        
        return len(violations) == 0, violations
    
    def scan_codebase_violations(self, root_path: str = ".") -> Dict[str, List[str]]:
        """
        Scan the entire codebase for phase boundary violations.
        
        Args:
            root_path: Root directory to scan
            
        Returns:
            Dictionary of violations by module
        """
        violations = {}
        root = Path(root_path)
        
        for py_file in root.rglob("*.py"):
            if py_file.name == "__init__.py" or py_file.parent.name == "__pycache__":
                continue
            
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    tree = ast.parse(f.read())
                
                module_name = str(py_file.relative_to(root)).replace('/', '.').replace('.py', '')
                file_violations = self._check_file_imports(tree, module_name)
                
                if file_violations:
                    violations[str(py_file)] = file_violations
                    
            except Exception as e:
                warnings.warn(f"Could not parse {py_file}: {e}")
        
        return violations
    
    def _extract_phase(self, module_name: str) -> str:
        """Extract phase identifier from module name."""
        if 'phases.' in module_name:
            parts = module_name.split('.')
            for part in parts:
                if part in self.phase_names:
                    return part
        return ""
    
    def _is_private_access(self, module_name: str, import_name: str) -> bool:
        """Check if import accesses private module internals."""
        # Block access to private members (starting with _)
        if import_name.startswith('_'):
            return True
        
        # Block direct access to canonical_flow internals from phases
        if 'phases' in module_name and 'canonical_flow' in import_name:
            # Only allow access through orchestrator interfaces
            if 'orchestrator' not in import_name.lower():
                return True
        
        return False
    
    def _record_violation(self, module: str, violation: str):
        """Record a validation violation."""
        if module not in self.violation_cache:
            self.violation_cache[module] = []
        self.violation_cache[module].append(violation)
    
    def _check_file_imports(self, tree: ast.AST, module_name: str) -> List[str]:
        """Check all imports in a file for violations."""
        violations = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if not self.validate_import(alias.name, alias.name, module_name):
                        violations.append(f"Prohibited import: {alias.name}")
            
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    for alias in node.names:
                        import_name = alias.name
                        if not self.validate_import(node.module, import_name, module_name):
                            violations.append(f"Prohibited import: from {node.module} import {import_name}")
        
        return violations
    
    def generate_compliance_report(self) -> Dict[str, Any]:
        """Generate a comprehensive compliance report."""
        report = {
            'phase_api_compliance': {},
            'import_violations': {},
            'summary': {
                'compliant_phases': 0,
                'total_violations': 0,
                'critical_issues': []
            }
        }
        
        # Check each phase API compliance
        for phase in self.phase_names:
            is_compliant, violations = self.validate_phase_api_compliance(phase)
            report['phase_api_compliance'][phase] = {
                'compliant': is_compliant,
                'violations': violations
            }
            
            if is_compliant:
                report['summary']['compliant_phases'] += 1
            else:
                report['summary']['critical_issues'].extend([f"Phase {phase}: {v}" for v in violations])
        
        # Scan for import violations
        import_violations = self.scan_codebase_violations()
        report['import_violations'] = import_violations
        
        total_violations = sum(len(v) for v in import_violations.values())
        report['summary']['total_violations'] = total_violations
        
        return report


# Runtime import hook for validation
class PhaseImportHook:
    """Import hook to validate phase boundary compliance at runtime."""
    
    def __init__(self):
        self.validator = PhaseAccessValidator()
        # Handle both dict and module cases for __builtins__
        if isinstance(__builtins__, dict):
            self._original_import = __builtins__.get('__import__', __import__)
        else:
            self._original_import = getattr(__builtins__, '__import__', __import__)
        
    def install(self):
        """Install the import hook."""
        if isinstance(__builtins__, dict):
            __builtins__['__import__'] = self._import_wrapper
        else:
            __builtins__.__import__ = self._import_wrapper
    
    def uninstall(self):
        """Uninstall the import hook."""
        if isinstance(__builtins__, dict):
            __builtins__['__import__'] = self._original_import
        else:
            __builtins__.__import__ = self._original_import
    
    def _import_wrapper(self, name, globals=None, locals=None, fromlist=(), level=0):
        """Wrapper around __import__ to validate imports."""
        calling_module = globals.get('__name__', '') if globals else ''
        
        # Validate the import
        if fromlist:
            for item in fromlist:
                if not self.validator.validate_import(name, item, calling_module):
                    warnings.warn(
                        f"Phase boundary violation: {calling_module} -> {name}.{item}",
                        UserWarning,
                        stacklevel=2
                    )
        else:
            if not self.validator.validate_import(name, name, calling_module):
                warnings.warn(
                    f"Phase boundary violation: {calling_module} -> {name}",
                    UserWarning,
                    stacklevel=2
                )
        
        # Proceed with original import
        return self._original_import(name, globals, locals, fromlist, level)


# Module-level validator instance
_validator = PhaseAccessValidator()
_import_hook = PhaseImportHook()


def validate_phase_boundaries(install_hook: bool = False) -> Dict[str, Any]:
    """
    Validate phase boundaries and optionally install runtime checking.
    
    Args:
        install_hook: Whether to install runtime import validation
        
    Returns:
        Compliance report
    """
    if install_hook:
        _import_hook.install()
    
    return _validator.generate_compliance_report()


def uninstall_validation_hook():
    """Uninstall the runtime validation hook."""
    _import_hook.uninstall()


__all__ = [
    'PhaseAccessValidator',
    'PhaseImportHook',
    'validate_phase_boundaries',
    'uninstall_validation_hook'
]
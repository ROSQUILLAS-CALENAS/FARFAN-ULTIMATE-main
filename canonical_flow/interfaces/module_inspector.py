"""
Module inspection utilities for canonical flow components.

Provides tools to analyze existing modules, detect their interfaces,
and generate compatibility reports for standardization.
"""

import ast
import inspect
import importlib.util
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
import logging

logger = logging.getLogger(__name__)


class ModuleInspector:
    """
    Inspector for analyzing canonical flow modules and their interfaces.
    """
    
    def __init__(self):
        self.analysis_cache = {}
    
    def analyze_module_file(self, module_path: Path) -> Dict[str, Any]:
        """
        Analyze a module file using AST parsing.
        
        Args:
            module_path: Path to the module file
            
        Returns:
            Analysis results dictionary
        """
        cache_key = str(module_path)
        if cache_key in self.analysis_cache:
            return self.analysis_cache[cache_key]
        
        try:
            with open(module_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
            
            tree = ast.parse(source_code)
            
            analysis = {
                'module_path': str(module_path),
                'module_name': module_path.stem,
                'functions': [],
                'classes': [],
                'imports': [],
                'has_process_method': False,
                'process_signatures': [],
                'async_functions': [],
                'decorators_used': set(),
                'compatibility_score': 0,
                'issues': [],
                'suggestions': []
            }
            
            # Analyze AST nodes
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    self._analyze_function_node(node, analysis)
                elif isinstance(node, ast.AsyncFunctionDef):
                    self._analyze_async_function_node(node, analysis)
                elif isinstance(node, ast.ClassDef):
                    self._analyze_class_node(node, analysis)
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    self._analyze_import_node(node, analysis)
            
            # Calculate compatibility score
            analysis['compatibility_score'] = self._calculate_compatibility_score(analysis)
            
            # Generate suggestions
            analysis['suggestions'] = self._generate_suggestions(analysis)
            
            self.analysis_cache[cache_key] = analysis
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze module {module_path}: {e}")
            return {
                'module_path': str(module_path),
                'module_name': module_path.stem,
                'error': str(e),
                'compatibility_score': 0
            }
    
    def analyze_runtime_module(self, module) -> Dict[str, Any]:
        """
        Analyze a runtime module object.
        
        Args:
            module: Imported module object
            
        Returns:
            Runtime analysis results
        """
        analysis = {
            'module_name': getattr(module, '__name__', 'unknown'),
            'module_file': getattr(module, '__file__', 'unknown'),
            'functions': [],
            'classes': [],
            'has_process_method': hasattr(module, 'process'),
            'process_signature': None,
            'callable_objects': [],
            'attributes': [],
            'compatibility_score': 0,
            'runtime_issues': []
        }
        
        # Analyze module attributes
        for name in dir(module):
            if name.startswith('_'):
                continue
                
            try:
                obj = getattr(module, name)
                
                if inspect.isfunction(obj):
                    func_info = self._analyze_runtime_function(name, obj)
                    analysis['functions'].append(func_info)
                    
                    if name == 'process':
                        analysis['process_signature'] = func_info['signature']
                
                elif inspect.isclass(obj):
                    class_info = self._analyze_runtime_class(name, obj)
                    analysis['classes'].append(class_info)
                
                elif callable(obj):
                    analysis['callable_objects'].append({
                        'name': name,
                        'type': type(obj).__name__,
                        'callable': True
                    })
                
                else:
                    analysis['attributes'].append({
                        'name': name,
                        'type': type(obj).__name__,
                        'value_repr': repr(obj)[:100]  # Truncate long values
                    })
                    
            except Exception as e:
                analysis['runtime_issues'].append(f"Error analyzing {name}: {e}")
        
        # Calculate compatibility score
        analysis['compatibility_score'] = self._calculate_runtime_compatibility_score(analysis)
        
        return analysis
    
    def generate_compatibility_report(self, module_paths: List[Path]) -> Dict[str, Any]:
        """
        Generate a comprehensive compatibility report for multiple modules.
        
        Args:
            module_paths: List of module paths to analyze
            
        Returns:
            Compatibility report
        """
        report = {
            'total_modules': len(module_paths),
            'analyzed_modules': 0,
            'failed_analyses': 0,
            'compatibility_summary': {
                'high_compatibility': 0,  # Score >= 80
                'medium_compatibility': 0,  # Score 50-79
                'low_compatibility': 0,   # Score < 50
                'average_score': 0
            },
            'common_issues': {},
            'interface_patterns': {},
            'modules_with_process_method': [],
            'modules_needing_adaptation': [],
            'detailed_results': []
        }
        
        total_score = 0
        
        for module_path in module_paths:
            analysis = self.analyze_module_file(module_path)
            
            if 'error' in analysis:
                report['failed_analyses'] += 1
                continue
            
            report['analyzed_modules'] += 1
            report['detailed_results'].append(analysis)
            
            score = analysis['compatibility_score']
            total_score += score
            
            # Categorize by compatibility
            if score >= 80:
                report['compatibility_summary']['high_compatibility'] += 1
            elif score >= 50:
                report['compatibility_summary']['medium_compatibility'] += 1
            else:
                report['compatibility_summary']['low_compatibility'] += 1
            
            # Track modules with process methods
            if analysis['has_process_method']:
                report['modules_with_process_method'].append(analysis['module_name'])
            else:
                report['modules_needing_adaptation'].append(analysis['module_name'])
            
            # Collect common issues
            for issue in analysis.get('issues', []):
                issue_type = issue.get('type', 'unknown')
                if issue_type not in report['common_issues']:
                    report['common_issues'][issue_type] = 0
                report['common_issues'][issue_type] += 1
        
        # Calculate average score
        if report['analyzed_modules'] > 0:
            report['compatibility_summary']['average_score'] = total_score / report['analyzed_modules']
        
        return report
    
    def _analyze_function_node(self, node: ast.FunctionDef, analysis: Dict[str, Any]):
        """Analyze a function definition node."""
        func_info = {
            'name': node.name,
            'line_number': node.lineno,
            'args': [arg.arg for arg in node.args.args],
            'defaults_count': len(node.args.defaults),
            'has_varargs': node.args.vararg is not None,
            'has_kwargs': node.args.kwarg is not None,
            'decorators': [self._get_decorator_name(d) for d in node.decorator_list],
            'is_async': False
        }
        
        # Track decorators
        analysis['decorators_used'].update(func_info['decorators'])
        
        # Check if this is a process method
        if node.name == 'process':
            analysis['has_process_method'] = True
            analysis['process_signatures'].append(func_info)
        
        analysis['functions'].append(func_info)
    
    def _analyze_async_function_node(self, node: ast.AsyncFunctionDef, analysis: Dict[str, Any]):
        """Analyze an async function definition node."""
        func_info = {
            'name': node.name,
            'line_number': node.lineno,
            'args': [arg.arg for arg in node.args.args],
            'defaults_count': len(node.args.defaults),
            'has_varargs': node.args.vararg is not None,
            'has_kwargs': node.args.kwarg is not None,
            'decorators': [self._get_decorator_name(d) for d in node.decorator_list],
            'is_async': True
        }
        
        analysis['async_functions'].append(func_info)
        analysis['functions'].append(func_info)  # Also add to general functions list
    
    def _analyze_class_node(self, node: ast.ClassDef, analysis: Dict[str, Any]):
        """Analyze a class definition node."""
        class_info = {
            'name': node.name,
            'line_number': node.lineno,
            'bases': [self._get_node_name(base) for base in node.bases],
            'methods': [],
            'decorators': [self._get_decorator_name(d) for d in node.decorator_list]
        }
        
        # Analyze methods
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                method_info = {
                    'name': item.name,
                    'is_async': isinstance(item, ast.AsyncFunctionDef),
                    'args': [arg.arg for arg in item.args.args],
                    'is_process_method': item.name == 'process'
                }
                class_info['methods'].append(method_info)
                
                if item.name == 'process':
                    analysis['has_process_method'] = True
                    analysis['process_signatures'].append(method_info)
        
        analysis['classes'].append(class_info)
    
    def _analyze_import_node(self, node: ast.Import | ast.ImportFrom, analysis: Dict[str, Any]):
        """Analyze import statements."""
        if isinstance(node, ast.Import):
            for alias in node.names:
                analysis['imports'].append({
                    'type': 'import',
                    'module': alias.name,
                    'alias': alias.asname
                })
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ''
            for alias in node.names:
                analysis['imports'].append({
                    'type': 'from_import',
                    'module': module,
                    'name': alias.name,
                    'alias': alias.asname
                })
    
    def _analyze_runtime_function(self, name: str, func) -> Dict[str, Any]:
        """Analyze a runtime function object."""
        try:
            sig = inspect.signature(func)
            
            return {
                'name': name,
                'signature': str(sig),
                'parameters': list(sig.parameters.keys()),
                'parameter_count': len(sig.parameters),
                'has_return_annotation': sig.return_annotation != sig.empty,
                'is_coroutine': inspect.iscoroutinefunction(func),
                'module': getattr(func, '__module__', 'unknown'),
                'file': getattr(func, '__code__', {}).co_filename if hasattr(func, '__code__') else 'unknown',
                'line_number': getattr(func, '__code__', {}).co_firstlineno if hasattr(func, '__code__') else 0
            }
        except Exception as e:
            return {
                'name': name,
                'error': str(e),
                'signature': 'unknown'
            }
    
    def _analyze_runtime_class(self, name: str, cls) -> Dict[str, Any]:
        """Analyze a runtime class object."""
        try:
            methods = []
            
            for method_name in dir(cls):
                if method_name.startswith('_') and method_name not in ['__init__', '__call__']:
                    continue
                    
                method = getattr(cls, method_name)
                if callable(method):
                    methods.append({
                        'name': method_name,
                        'is_process_method': method_name == 'process'
                    })
            
            return {
                'name': name,
                'module': getattr(cls, '__module__', 'unknown'),
                'methods': methods,
                'has_process_method': any(m['is_process_method'] for m in methods),
                'bases': [base.__name__ for base in cls.__bases__]
            }
        except Exception as e:
            return {
                'name': name,
                'error': str(e)
            }
    
    def _calculate_compatibility_score(self, analysis: Dict[str, Any]) -> int:
        """Calculate compatibility score for AST analysis."""
        score = 0
        
        # Has process method (+40 points)
        if analysis['has_process_method']:
            score += 40
            
            # Check process method signature
            for proc_sig in analysis['process_signatures']:
                args = proc_sig.get('args', [])
                if len(args) >= 2 and 'data' in args and ('context' in args or 'ctx' in args):
                    score += 20  # Good signature
                elif len(args) >= 1:
                    score += 10  # Partially compatible
        
        # Has functions (+10 points, max 30)
        function_count = len(analysis['functions'])
        score += min(function_count * 10, 30)
        
        # Uses decorators (+5 points, max 15)
        if analysis['decorators_used']:
            score += min(len(analysis['decorators_used']) * 5, 15)
        
        # Penalize issues (-5 points each, max -20)
        issues_penalty = min(len(analysis.get('issues', [])) * 5, 20)
        score -= issues_penalty
        
        return max(0, min(score, 100))
    
    def _calculate_runtime_compatibility_score(self, analysis: Dict[str, Any]) -> int:
        """Calculate compatibility score for runtime analysis."""
        score = 0
        
        # Has process method (+50 points)
        if analysis['has_process_method']:
            score += 50
        
        # Function count (+5 points each, max 30)
        function_count = len(analysis['functions'])
        score += min(function_count * 5, 30)
        
        # Classes with process methods (+20 points each)
        for class_info in analysis['classes']:
            if class_info.get('has_process_method', False):
                score += 20
        
        return max(0, min(score, 100))
    
    def _generate_suggestions(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate standardization suggestions."""
        suggestions = []
        
        if not analysis['has_process_method']:
            suggestions.append(
                "Add a 'process(data, context) -> Dict[str, Any]' method for standardization"
            )
        
        if not analysis['functions']:
            suggestions.append("Module has no functions - consider adding processing logic")
        
        if not analysis.get('decorators_used'):
            suggestions.append("Consider adding telemetry decorators for monitoring")
        
        # Check for common parameter patterns
        for func in analysis['functions']:
            args = func.get('args', [])
            if len(args) == 1 and 'data' not in args:
                suggestions.append(
                    f"Function '{func['name']}' could benefit from standardized parameters"
                )
        
        return suggestions
    
    def _get_decorator_name(self, decorator) -> str:
        """Get decorator name from AST node."""
        if isinstance(decorator, ast.Name):
            return decorator.id
        elif isinstance(decorator, ast.Attribute):
            return f"{self._get_node_name(decorator.value)}.{decorator.attr}"
        elif isinstance(decorator, ast.Call):
            return self._get_decorator_name(decorator.func)
        else:
            return 'unknown_decorator'
    
    def _get_node_name(self, node) -> str:
        """Get name from AST node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_node_name(node.value)}.{node.attr}"
        else:
            return 'unknown'


def inspect_canonical_flow_modules(base_path: Optional[Path] = None,
                                  pattern: str = "**/*.py") -> Dict[str, Any]:
    """
    Convenience function to inspect all canonical flow modules.
    
    Args:
        base_path: Base path for module discovery
        pattern: File pattern for module discovery
        
    Returns:
        Comprehensive inspection report
    """
    if base_path is None:
        base_path = Path(__file__).parent.parent
    
    inspector = ModuleInspector()
    
    # Discover modules
    modules = []
    for py_file in base_path.glob(pattern):
        if (py_file.name.startswith('__') or 
            py_file.name.startswith('test_') or
            'interfaces' in py_file.parts):
            continue
        modules.append(py_file)
    
    # Generate compatibility report
    compatibility_report = inspector.generate_compatibility_report(modules)
    
    return {
        'inspection_timestamp': str(Path(__file__).stat().st_mtime),
        'base_path': str(base_path),
        'pattern': pattern,
        'compatibility_report': compatibility_report,
        'inspector_cache_size': len(inspector.analysis_cache)
    }
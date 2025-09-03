"""
Preflight validation system for analysis_nlp components.

This module validates that all 9 analysis_nlp components contain actual implementation
rather than placeholder functions, checking for expected functions and classes,
and detecting incomplete implementations before pipeline execution begins.
"""

import ast
import importlib
import inspect
import logging
import sys
# # # from pathlib import Path  # Module not found  # Module not found  # Module not found
# # # from typing import Any, Dict, List, Set, Tuple, Union  # Module not found  # Module not found  # Module not found


# Mandatory Pipeline Contract Annotations
__phase__ = "A"
__code__ = "34A"
__stage_order__ = 4

logger = logging.getLogger(__name__)


class ComponentValidationError(Exception):
    """Raised when analysis_nlp components fail validation."""
    
    def __init__(self, message: str, failed_components: Dict[str, List[str]]) -> None:
        self.failed_components: Dict[str, List[str]] = failed_components
        super().__init__(message)


class PlaceholderDetector:
    """Detects placeholder code patterns in functions and classes."""
    
    PLACEHOLDER_PATTERNS: Dict[str, List[str]] = {
        'pass_only': ['pass'],
        'not_implemented': ['raise NotImplementedError', 'raise NotImplemented'],
        'todo_comment': ['# TODO', '# FIXME', '# XXX'],
        'empty_return': ['return None', 'return', 'return {}', 'return []'],
        'placeholder_string': ['placeholder', 'TODO', 'not implemented', 'coming soon']
    }
    
    @staticmethod
    def is_placeholder_function(func: Any) -> Tuple[bool, List[str]]:
        """Check if a function contains only placeholder code."""
        issues = []
        
        try:
            source = inspect.getsource(func)
            source_lines = [line.strip() for line in source.split('\n') if line.strip()]
            
            # Remove function definition and docstring
            func_body_lines = []
            in_docstring = False
            docstring_delimiter = None
            
            for line in source_lines:
                if line.startswith('def ') or line.startswith('async def '):
                    continue
                
                # Handle docstrings
                if '"""' in line or "'''" in line:
                    if not in_docstring:
                        docstring_delimiter = '"""' if '"""' in line else "'''"
                        in_docstring = True
                        if line.count(docstring_delimiter) == 2:  # Single line docstring
                            in_docstring = False
                        continue
                    else:
                        if docstring_delimiter in line:
                            in_docstring = False
                        continue
                
                if not in_docstring and line and not line.startswith('#'):
                    func_body_lines.append(line)
            
            # Check for placeholder patterns
            body_text = ' '.join(func_body_lines).lower()
            
            # Only 'pass' statement
            if len(func_body_lines) == 1 and func_body_lines[0] == 'pass':
                issues.append('Function contains only "pass" statement')
            
            # NotImplementedError
            if any('raise notimplementederror' in line.lower() or 'raise notimplemented' in line.lower() 
                   for line in func_body_lines):
                issues.append('Function raises NotImplementedError')
            
            # Empty return only
            if len(func_body_lines) == 1 and func_body_lines[0].lower() in ['return', 'return none']:
                issues.append('Function contains only empty return')
            
            # Placeholder strings
            if any(placeholder in body_text for placeholder in 
                   ['placeholder', 'todo', 'not implemented', 'coming soon', 'fixme']):
                issues.append('Function contains placeholder text')
            
            # No meaningful implementation (very short functions)
            if len(func_body_lines) == 0:
                issues.append('Function has no implementation')
            
        except (OSError, TypeError) as e:
            issues.append(f'Could not analyze function source: {e}')
        
        return len(issues) > 0, issues
    
    @staticmethod
    def is_placeholder_class(cls: Any) -> Tuple[bool, List[str]]:
        """Check if a class contains only placeholder implementations."""
        issues = []
        
        try:
            # Get all methods defined in this class (not inherited)
            own_methods = []
            for name, method in inspect.getmembers(cls, inspect.ismethod):
                if name in cls.__dict__:
                    own_methods.append((name, method))
            
            for name, method in inspect.getmembers(cls, inspect.isfunction):
                if name in cls.__dict__:
                    own_methods.append((name, method))
            
            if not own_methods:
                issues.append('Class has no implemented methods')
                return True, issues
            
            # Check each method for placeholder patterns
            placeholder_methods = []
            for method_name, method in own_methods:
                if method_name.startswith('_') and method_name != '__init__':
                    continue  # Skip private methods except __init__
                
                is_placeholder, method_issues = PlaceholderDetector.is_placeholder_function(method)
                if is_placeholder:
                    placeholder_methods.append(f'{method_name}: {", ".join(method_issues)}')
            
            if placeholder_methods:
                issues.extend(placeholder_methods)
            
            # If all non-private methods are placeholders, class is considered placeholder
            non_private_methods = [name for name, _ in own_methods if not name.startswith('_') or name == '__init__']
            if len(placeholder_methods) >= len(non_private_methods):
                issues.append('All or most methods contain placeholder code')
        
        except Exception as e:
            issues.append(f'Could not analyze class: {e}')
        
        return len(issues) > 0, issues


class ComponentValidator:
    """Validates individual analysis_nlp components."""
    
    REQUIRED_COMPONENTS: Dict[str, Dict[str, Any]] = {
        'adaptive_analyzer': {
            'expected_classes': ['AdaptiveAnalyzer', 'AnalysisMode', 'SystemState'],
            'expected_functions': ['analyze_system'],
            'expected_methods': {
                'AdaptiveAnalyzer': ['analyze_system', 'record_metric', 'start_continuous_analysis']
            }
        },
        'question_analyzer': {
            'expected_classes': ['QuestionAnalyzer', 'CausalPosture', 'EvidenceType'],
            'expected_functions': ['analyze_question'],
            'expected_methods': {
                'QuestionAnalyzer': ['analyze_question', 'extract_search_patterns', 'identify_evidence_types']
            }
        },
        'implementacion_mapeo': {
            'expected_classes': ['QuestionDecalogoMapper', 'QuestionMapping'],
            'expected_functions': [],
            'expected_methods': {
                'QuestionDecalogoMapper': ['map_question_to_decalogo', 'validate_mapping']
            }
        },
        'evidence_processor': {
            'expected_classes': ['EvidenceProcessor', 'StructuredEvidence', 'Citation'],
            'expected_functions': ['process_evidence'],
            'expected_methods': {
                'EvidenceProcessor': ['process_evidence', 'extract_citations', 'score_evidence']
            }
        },
        'extractor_evidencias_contextual': {
            'expected_classes': ['ContextualEvidenceExtractor'],
            'expected_functions': ['extract_contextual_evidence'],
            'expected_methods': {
                'ContextualEvidenceExtractor': ['extract_evidence', 'contextualize_evidence']
            }
        },
        'evidence_validation_model': {
            'expected_classes': ['EvidenceValidationModel', 'DNPEvidenceValidator'],
            'expected_functions': ['validate_evidence'],
            'expected_methods': {
                'EvidenceValidationModel': ['validate', 'score_evidence'],
                'DNPEvidenceValidator': ['validate_against_standards']
            }
        },
        'evaluation_driven_processor': {
            'expected_classes': ['EvaluationDrivenProcessor'],
            'expected_functions': ['process_evaluation'],
            'expected_methods': {
                'EvaluationDrivenProcessor': ['process', 'evaluate_metrics', 'generate_report']
            }
        },
        'dnp_alignment_adapter': {
            'expected_classes': ['DNPAlignmentAdapter'],
            'expected_functions': ['align_with_dnp'],
            'expected_methods': {
                'DNPAlignmentAdapter': ['align_policy', 'validate_alignment', 'generate_mapping']
            }
        }
    }
    
    def __init__(self) -> None:
        self.project_root: Path = Path(__file__).resolve().parent
        self.canonical_flow_path: Path = self.project_root / 'canonical_flow' / 'A_analysis_nlp'
        
    def validate_component(self, component_name: str) -> Tuple[bool, List[str]]:
        """Validate a single component for implementation completeness."""
        issues = []
        
        try:
# # #             # First try to load from canonical flow (alias)  # Module not found  # Module not found  # Module not found
            module = self._load_component_module(component_name)
            if module is None:
                issues.append(f'Could not load module {component_name}')
                return False, issues
            
            # Check if it's just an alias that failed to load the original
            if hasattr(module, 'process') and callable(module.process):
# # #                 # Check if it's the placeholder function from failed import  # Module not found  # Module not found  # Module not found
                try:
                    result = module.process()
                    if isinstance(result, dict) and 'error' in result and 'failed to load' in str(result['error']).lower():
                        issues.append(f'Module {component_name} failed to load original implementation: {result["error"]}')
                        return False, issues
                except:
                    pass
            
            # Get expected requirements for this component
            requirements = self.REQUIRED_COMPONENTS.get(component_name, {})
            
            # Validate expected classes
            for class_name in requirements.get('expected_classes', []):
                if not hasattr(module, class_name):
                    issues.append(f'Missing expected class: {class_name}')
                else:
                    cls = getattr(module, class_name)
                    if inspect.isclass(cls):
                        is_placeholder, class_issues = PlaceholderDetector.is_placeholder_class(cls)
                        if is_placeholder:
                            issues.append(f'Class {class_name} contains placeholder code: {"; ".join(class_issues)}')
                    else:
                        issues.append(f'{class_name} is not a class')
            
            # Validate expected functions
            for func_name in requirements.get('expected_functions', []):
                if not hasattr(module, func_name):
                    issues.append(f'Missing expected function: {func_name}')
                else:
                    func = getattr(module, func_name)
                    if callable(func):
                        is_placeholder, func_issues = PlaceholderDetector.is_placeholder_function(func)
                        if is_placeholder:
                            issues.append(f'Function {func_name} contains placeholder code: {"; ".join(func_issues)}')
                    else:
                        issues.append(f'{func_name} is not callable')
            
            # Validate expected methods in classes
            for class_name, method_names in requirements.get('expected_methods', {}).items():
                if hasattr(module, class_name):
                    cls = getattr(module, class_name)
                    if inspect.isclass(cls):
                        for method_name in method_names:
                            if not hasattr(cls, method_name):
                                issues.append(f'Class {class_name} missing expected method: {method_name}')
                            else:
                                method = getattr(cls, method_name)
                                if callable(method):
                                    is_placeholder, method_issues = PlaceholderDetector.is_placeholder_function(method)
                                    if is_placeholder:
                                        issues.append(f'Method {class_name}.{method_name} contains placeholder code: {"; ".join(method_issues)}')
        
        except Exception as e:
            issues.append(f'Error validating component {component_name}: {e}')
        
        return len(issues) == 0, issues
    
    def _load_component_module(self, component_name: str) -> Optional[Any]:
# # #         """Load component module from canonical flow or project root."""  # Module not found  # Module not found  # Module not found
        module: Optional[Any] = None
        
        # Try canonical flow first (alias files)
        try:
            canonical_module_path = f'canonical_flow.A_analysis_nlp.{component_name}'
            if str(self.canonical_flow_path.parent.parent) not in sys.path:
                sys.path.insert(0, str(self.canonical_flow_path.parent.parent))
            module = importlib.import_module(canonical_module_path)
            
            # If module loaded but has errors, the alias will contain error info
            if hasattr(module, 'process') and callable(module.process):
                try:
                    test_result = module.process()
                    if isinstance(test_result, dict) and 'error' in test_result:
                        logger.warning(f'Canonical module {component_name} has loading errors: {test_result["error"]}')
                except:
                    pass
                    
        except ImportError as e:
            logger.debug(f'Could not load canonical module {component_name}: {e}')
        
        # If canonical flow failed, try project root directly
        if module is None:
            try:
                if str(self.project_root) not in sys.path:
                    sys.path.insert(0, str(self.project_root))
                module = importlib.import_module(component_name)
            except ImportError as e:
                logger.debug(f'Could not load root module {component_name}: {e}')
        
        return module


def validate_analysis_nlp_components() -> None:
    """
    Main validation function that checks all 9 analysis_nlp components.
    
    Raises:
        ComponentValidationError: If any component fails validation with details
                                 of all problematic modules.
    """
    validator = ComponentValidator()
    failed_components = {}
    
    all_components = [
        'adaptive_analyzer',
        'question_analyzer', 
        'implementacion_mapeo',
        'evidence_processor',
        'extractor_evidencias_contextual',
        'evidence_validation_model',
        'evaluation_driven_processor',
        'dnp_alignment_adapter'
    ]
    
    logger.info("Starting validation of analysis_nlp components...")
    
    for component_name in all_components:
        logger.debug(f"Validating component: {component_name}")
        is_valid, issues = validator.validate_component(component_name)
        
        if not is_valid:
            failed_components[component_name] = issues
            logger.error(f"Component {component_name} failed validation: {issues}")
        else:
            logger.info(f"Component {component_name} passed validation")
    
    if failed_components:
        # Create comprehensive error message
        error_lines = ["Analysis NLP components validation failed:"]
        error_lines.append("")
        
        for component, issues in failed_components.items():
            error_lines.append(f"❌ {component}:")
            for issue in issues:
                error_lines.append(f"   • {issue}")
            error_lines.append("")
        
        error_lines.extend([
            "These components must be properly implemented before pipeline execution.",
            "Please ensure all classes and functions contain actual implementation",
            "rather than placeholder code like 'pass' or 'raise NotImplementedError'."
        ])
        
        error_message = "\n".join(error_lines)
        
        raise ComponentValidationError(error_message, failed_components)
    
    logger.info("✅ All analysis_nlp components passed validation")


if __name__ == "__main__":
    # Enable logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        validate_analysis_nlp_components()
        print("✅ All analysis_nlp components are properly implemented")
    except ComponentValidationError as e:
        print(e)
        sys.exit(1)
    except Exception as e:
        print(f"❌ Validation failed with unexpected error: {e}")
        sys.exit(1)
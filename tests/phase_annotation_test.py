"""
Phase annotation test module that parses source files to validate required phase decorators
are present on functions.
"""

import ast
import inspect
import re
from pathlib import Path
from typing import Dict, List, Set, Optional, Any, Tuple

import pytest


class PhaseAnnotationTest:
    """Test suite for validating phase annotations and decorators."""
    
    # Expected phase decorators and their variations
    PHASE_DECORATORS = {
        'ingestion_phase': ['@ingestion_phase', '@phase("ingestion")', '@phase(PhaseType.INGESTION)'],
        'context_phase': ['@context_phase', '@phase("context")', '@phase(PhaseType.CONTEXT)'],
        'knowledge_phase': ['@knowledge_phase', '@phase("knowledge")', '@phase(PhaseType.KNOWLEDGE)'],
        'analysis_phase': ['@analysis_phase', '@phase("analysis")', '@phase(PhaseType.ANALYSIS)'],
        'classification_phase': ['@classification_phase', '@phase("classification")', '@phase(PhaseType.CLASSIFICATION)'],
        'retrieval_phase': ['@retrieval_phase', '@phase("retrieval")', '@phase(PhaseType.RETRIEVAL)'],
        'orchestration_phase': ['@orchestration_phase', '@phase("orchestration")', '@phase(PhaseType.ORCHESTRATION)'],
        'aggregation_phase': ['@aggregation_phase', '@phase("aggregation")', '@phase(PhaseType.AGGREGATION)'],
        'integration_phase': ['@integration_phase', '@phase("integration")', '@phase(PhaseType.INTEGRATION)'],
        'synthesis_phase': ['@synthesis_phase', '@phase("synthesis")', '@phase(PhaseType.SYNTHESIS)']
    }
    
    # Directories that should contain phase-decorated functions
    PHASE_DIRECTORIES = {
        'phases/',
        'canonical_flow/',
        'retrieval_engine/',
        'egw_query_expansion/core/',
        'analysis_nlp/',
        'microservices/',
        'adapters/'
    }
    
    @pytest.fixture
    def phase_source_files(self) -> List[Path]:
        """Discover Python files in phase-related directories."""
        project_root = Path(__file__).parent.parent
        phase_files = []
        
        for phase_dir in self.PHASE_DIRECTORIES:
            dir_path = project_root / phase_dir
            if dir_path.exists():
                phase_files.extend(dir_path.rglob("*.py"))
        
        # Filter out test files and __init__.py files
        filtered_files = []
        for file_path in phase_files:
            if (not file_path.name.startswith('test_') and 
                not file_path.name.endswith('_test.py') and
                file_path.name != '__init__.py'):
                filtered_files.append(file_path)
        
        return sorted(filtered_files)
    
    def _extract_decorators_from_source(self, source_code: str) -> Dict[str, List[str]]:
        """Extract decorators for each function from source code."""
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            return {}
        
        function_decorators = {}
        
        class DecoratorExtractor(ast.NodeVisitor):
            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                decorators = []
                for decorator in node.decorator_list:
                    if isinstance(decorator, ast.Name):
                        decorators.append(f"@{decorator.id}")
                    elif isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name):
                        decorators.append(f"@{decorator.func.id}(...)")
                    elif isinstance(decorator, ast.Attribute):
                        decorators.append(f"@{ast.unparse(decorator)}")
                    else:
                        decorators.append(f"@{ast.unparse(decorator)}")
                
                function_decorators[node.name] = decorators
                self.generic_visit(node)
            
            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
                decorators = []
                for decorator in node.decorator_list:
                    if isinstance(decorator, ast.Name):
                        decorators.append(f"@{decorator.id}")
                    elif isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name):
                        decorators.append(f"@{decorator.func.id}(...)")
                    elif isinstance(decorator, ast.Attribute):
                        decorators.append(f"@{ast.unparse(decorator)}")
                    else:
                        decorators.append(f"@{ast.unparse(decorator)}")
                
                function_decorators[node.name] = decorators
                self.generic_visit(node)
        
        extractor = DecoratorExtractor()
        extractor.visit(tree)
        return function_decorators
    
    def _detect_phase_from_file_path(self, file_path: Path) -> Optional[str]:
        """Detect expected phase from file path."""
        path_str = str(file_path).lower()
        
        # Direct phase directory mapping
        phase_mappings = {
            'ingestion': 'ingestion_phase',
            'context': 'context_phase', 
            'knowledge': 'knowledge_phase',
            'analysis': 'analysis_phase',
            'classification': 'classification_phase',
            'retrieval': 'retrieval_phase',
            'orchestration': 'orchestration_phase',
            'aggregation': 'aggregation_phase',
            'integration': 'integration_phase',
            'synthesis': 'synthesis_phase'
        }
        
        for phase_name, phase_decorator in phase_mappings.items():
            if phase_name in path_str:
                return phase_decorator
        
        return None
    
    def _is_phase_function(self, func_name: str, decorators: List[str], source_code: str) -> bool:
        """Determine if a function should have phase decorators."""
        # Skip private functions, test functions, and utility functions
        if (func_name.startswith('_') or 
            func_name.startswith('test_') or
            func_name in ['main', '__init__', 'setup', 'teardown']):
            return False
        
        # Check if function appears to be a phase processing function
        phase_indicators = [
            'process', 'execute', 'run', 'handle', 'transform',
            'analyze', 'classify', 'retrieve', 'orchestrate',
            'aggregate', 'integrate', 'synthesize', 'generate'
        ]
        
        func_name_lower = func_name.lower()
        if any(indicator in func_name_lower for indicator in phase_indicators):
            return True
        
        # Check function body for phase-related operations
        if any(keyword in source_code for keyword in [
            'phase_', 'Phase', 'pipeline', 'Pipeline', 'workflow', 'Workflow'
        ]):
            return True
        
        return False
    
    def _has_valid_phase_decorator(self, decorators: List[str], expected_phase: Optional[str] = None) -> bool:
        """Check if decorators include a valid phase decorator."""
        all_phase_decorators = set()
        for phase_decorators in self.PHASE_DECORATORS.values():
            all_phase_decorators.update(phase_decorators)
        
        # Also check for generic @phase decorator patterns
        phase_patterns = [
            r'@phase\s*\(',
            r'@\w*_phase\b',
            r'@Phase\w*',
            r'@phase_\w+'
        ]
        
        for decorator in decorators:
            # Direct match
            if decorator in all_phase_decorators:
                return True
            
            # Pattern match
            for pattern in phase_patterns:
                if re.search(pattern, decorator, re.IGNORECASE):
                    return True
        
        return False
    
    def test_phase_functions_have_decorators(self, phase_source_files: List[Path]) -> None:
        """Test that functions in phase directories have appropriate phase decorators."""
        missing_decorators = []
        
        for file_path in phase_source_files:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    source_code = f.read()
                
                function_decorators = self._extract_decorators_from_source(source_code)
                expected_phase = self._detect_phase_from_file_path(file_path)
                
                for func_name, decorators in function_decorators.items():
                    if self._is_phase_function(func_name, decorators, source_code):
                        if not self._has_valid_phase_decorator(decorators, expected_phase):
                            missing_decorators.append({
                                'file': str(file_path),
                                'function': func_name,
                                'expected_phase': expected_phase,
                                'current_decorators': decorators
                            })
            
            except Exception as e:
                # Log parse errors but don't fail the test for them
                print(f"Warning: Could not parse {file_path}: {e}")
        
        if missing_decorators:
            error_messages = []
            for item in missing_decorators:
                message = (
                    f"Function '{item['function']}' in {item['file']} "
                    f"appears to be a phase function but lacks phase decorators."
                )
                if item['expected_phase']:
                    message += f"\n  Expected phase: {item['expected_phase']}"
                if item['current_decorators']:
                    message += f"\n  Current decorators: {', '.join(item['current_decorators'])}"
                else:
                    message += "\n  Current decorators: None"
                error_messages.append(message)
            
            pytest.fail(
                f"Found {len(missing_decorators)} functions missing phase decorators:\n\n" +
                "\n\n".join(error_messages)
            )
    
    def test_phase_decorator_consistency(self, phase_source_files: List[Path]) -> None:
        """Test that phase decorators are consistently applied within directories."""
        phase_usage = {}
        inconsistent_usage = []
        
        for file_path in phase_source_files:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    source_code = f.read()
                
                function_decorators = self._extract_decorators_from_source(source_code)
                expected_phase = self._detect_phase_from_file_path(file_path)
                
                if expected_phase:
                    file_phases = set()
                    for decorators in function_decorators.values():
                        for decorator in decorators:
                            for phase, phase_decorators in self.PHASE_DECORATORS.items():
                                if any(pd in decorator for pd in phase_decorators):
                                    file_phases.add(phase)
                    
                    # Check if file uses multiple different phase types
                    if len(file_phases) > 1:
                        inconsistent_usage.append({
                            'file': str(file_path),
                            'expected_phase': expected_phase,
                            'found_phases': list(file_phases)
                        })
            
            except Exception as e:
                print(f"Warning: Could not analyze {file_path}: {e}")
        
        if inconsistent_usage:
            error_messages = []
            for item in inconsistent_usage:
                message = (
                    f"File {item['file']} uses inconsistent phase decorators.\n"
                    f"  Expected phase: {item['expected_phase']}\n"
                    f"  Found phases: {', '.join(item['found_phases'])}"
                )
                error_messages.append(message)
            
            pytest.fail(
                f"Found {len(inconsistent_usage)} files with inconsistent phase decorator usage:\n\n" +
                "\n\n".join(error_messages)
            )
    
    def test_valid_phase_decorator_imports(self, phase_source_files: List[Path]) -> None:
        """Test that files using phase decorators have proper imports."""
        missing_imports = []
        
        for file_path in phase_source_files:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    source_code = f.read()
                
                # Check if file uses phase decorators
                has_phase_decorators = False
                for phase_decorators in self.PHASE_DECORATORS.values():
                    if any(decorator.replace('@', '') in source_code for decorator in phase_decorators):
                        has_phase_decorators = True
                        break
                
                if has_phase_decorators:
                    # Check for phase-related imports
                    import_patterns = [
                        r'from.*phase.*import',
                        r'import.*phase',
                        r'from.*phases.*import',
                        r'from.*decorators.*import'
                    ]
                    
                    has_phase_imports = any(
                        re.search(pattern, source_code, re.IGNORECASE | re.MULTILINE)
                        for pattern in import_patterns
                    )
                    
                    if not has_phase_imports:
                        missing_imports.append(str(file_path))
            
            except Exception as e:
                print(f"Warning: Could not analyze imports in {file_path}: {e}")
        
        if missing_imports:
            pytest.fail(
                f"Found {len(missing_imports)} files using phase decorators without proper imports:\n" +
                "\n".join(missing_imports)
            )


# Global test functions for pytest discovery
def test_phase_annotation_suite():
    """Entry point for phase annotation tests."""
    test_instance = PhaseAnnotationTest()
    
    # Get phase source files manually
    project_root = Path(__file__).parent.parent
    phase_files = []
    
    for phase_dir in test_instance.PHASE_DIRECTORIES:
        dir_path = project_root / phase_dir
        if dir_path.exists():
            phase_files.extend(dir_path.rglob("*.py"))
    
    # Filter out test files and __init__.py files
    filtered_files = []
    for file_path in phase_files:
        if (not file_path.name.startswith('test_') and 
            not file_path.name.endswith('_test.py') and
            file_path.name != '__init__.py'):
            filtered_files.append(file_path)
    
    # Run individual tests
    test_instance.test_phase_functions_have_decorators(sorted(filtered_files))
    test_instance.test_phase_decorator_consistency(sorted(filtered_files))
    test_instance.test_valid_phase_decorator_imports(sorted(filtered_files))
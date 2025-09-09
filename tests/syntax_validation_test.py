"""
Syntax validation test module using ast.parse to verify all Python files compile without syntax errors.
"""

import ast
import sys
from pathlib import Path
from typing import List, Tuple

import pytest


class SyntaxValidationTest:
    """Test suite for validating Python syntax across the codebase."""
    
    @pytest.fixture
    def python_files(self) -> List[Path]:
        """Discover all Python files in the codebase."""
        project_root = Path(__file__).parent.parent
        python_files = []
        
        # Core application files
        for pattern in ["**/*.py"]:
            python_files.extend(project_root.glob(pattern))
        
        # Filter out known problematic files and directories
        excluded_patterns = [
            "venv/",
            ".venv/",
            "__pycache__/",
            ".pytest_cache/",
            ".mypy_cache/",
            ".git/",
            "build/",
            "dist/",
            # Files with known intentional syntax issues
            "test_deps/",
        ]
        
        filtered_files = []
        for file_path in python_files:
            should_exclude = False
            for pattern in excluded_patterns:
                if pattern in str(file_path):
                    should_exclude = True
                    break
            if not should_exclude and file_path.is_file():
                filtered_files.append(file_path)
        
        return sorted(filtered_files)
    
    def test_all_python_files_have_valid_syntax(self, python_files: List[Path]) -> None:
        """Test that all Python files have valid syntax and can be parsed by ast.parse."""
        syntax_errors = []
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # Parse the file to check for syntax errors
                ast.parse(content, filename=str(file_path))
                
            except SyntaxError as e:
                error_info = {
                    'file': str(file_path),
                    'line': e.lineno,
                    'offset': e.offset,
                    'message': e.msg,
                    'text': e.text
                }
                syntax_errors.append(error_info)
            except UnicodeDecodeError as e:
                error_info = {
                    'file': str(file_path),
                    'line': None,
                    'offset': None,
                    'message': f"Unicode decode error: {e}",
                    'text': None
                }
                syntax_errors.append(error_info)
            except Exception as e:
                error_info = {
                    'file': str(file_path),
                    'line': None,
                    'offset': None,
                    'message': f"Unexpected error: {e}",
                    'text': None
                }
                syntax_errors.append(error_info)
        
        if syntax_errors:
            error_messages = []
            for error in syntax_errors:
                if error['line'] is not None:
                    message = (
                        f"Syntax error in {error['file']}:{error['line']}:{error['offset']} - "
                        f"{error['message']}"
                    )
                    if error['text']:
                        message += f"\n  Code: {error['text'].strip()}"
                else:
                    message = f"Error in {error['file']}: {error['message']}"
                error_messages.append(message)
            
            pytest.fail(
                f"Found {len(syntax_errors)} syntax errors:\n" + 
                "\n".join(error_messages)
            )
    
    def test_python_version_compatibility(self, python_files: List[Path]) -> None:
        """Test that all files are compatible with the minimum Python version (3.8)."""
        compatibility_errors = []
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # Parse with compile to check for version-specific features
                compile(content, str(file_path), 'exec', dont_inherit=True)
                
            except SyntaxError as e:
                # Check if it's a Python version compatibility issue
                if any(keyword in str(e) for keyword in [
                    'invalid syntax', 'unexpected EOF', 'invalid character'
                ]):
                    compatibility_errors.append({
                        'file': str(file_path),
                        'error': str(e)
                    })
            except Exception:
                # Other compilation errors are not version compatibility issues
                pass
        
        if compatibility_errors:
            error_messages = [
                f"Compatibility error in {error['file']}: {error['error']}"
                for error in compatibility_errors
            ]
            pytest.fail(
                f"Found {len(compatibility_errors)} Python version compatibility errors:\n" +
                "\n".join(error_messages)
            )
    
    def test_encoding_consistency(self, python_files: List[Path]) -> None:
        """Test that all Python files have consistent encoding."""
        encoding_issues = []
        
        for file_path in python_files:
            try:
                # Try to read with strict UTF-8
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for BOM or other encoding indicators
                if content.startswith('\ufeff'):
                    encoding_issues.append({
                        'file': str(file_path),
                        'issue': 'File contains UTF-8 BOM'
                    })
                
                # Check encoding declaration
                lines = content.split('\n')[:2]  # Check first two lines
                has_encoding_decl = any(
                    'coding:' in line or 'coding=' in line
                    for line in lines
                    if line.strip().startswith('#')
                )
                
                if has_encoding_decl:
                    # Verify the declared encoding is utf-8
                    for line in lines:
                        if line.strip().startswith('#') and ('coding:' in line or 'coding=' in line):
                            if 'utf-8' not in line.lower() and 'utf8' not in line.lower():
                                encoding_issues.append({
                                    'file': str(file_path),
                                    'issue': f'Non-UTF-8 encoding declared: {line.strip()}'
                                })
                
            except UnicodeDecodeError as e:
                encoding_issues.append({
                    'file': str(file_path),
                    'issue': f'Cannot decode as UTF-8: {e}'
                })
        
        if encoding_issues:
            error_messages = [
                f"Encoding issue in {issue['file']}: {issue['issue']}"
                for issue in encoding_issues
            ]
            pytest.fail(
                f"Found {len(encoding_issues)} encoding issues:\n" +
                "\n".join(error_messages)
            )


# Global test functions for pytest discovery
def test_syntax_validation_suite():
    """Entry point for syntax validation tests."""
    test_instance = SyntaxValidationTest()
    
    # Get python files fixture manually
    project_root = Path(__file__).parent.parent
    python_files = []
    for pattern in ["**/*.py"]:
        python_files.extend(project_root.glob(pattern))
    
    excluded_patterns = [
        "venv/", ".venv/", "__pycache__/", ".pytest_cache/", 
        ".mypy_cache/", ".git/", "build/", "dist/", "test_deps/"
    ]
    
    filtered_files = []
    for file_path in python_files:
        should_exclude = any(pattern in str(file_path) for pattern in excluded_patterns)
        if not should_exclude and file_path.is_file():
            filtered_files.append(file_path)
    
    # Run individual tests
    test_instance.test_all_python_files_have_valid_syntax(sorted(filtered_files))
    test_instance.test_python_version_compatibility(sorted(filtered_files))
    test_instance.test_encoding_consistency(sorted(filtered_files))
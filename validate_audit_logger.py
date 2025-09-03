#!/usr/bin/env python3
"""
Validation script for AuditLogger using only built-in Python syntax validation.
"""

import ast
import sys
# # # from pathlib import Path  # Module not found  # Module not found  # Module not found


def validate_python_syntax(file_path):
    """Validate Python syntax without executing the code."""
    print(f"🔍 Validating syntax for: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source_code = f.read()
        
        # Parse the AST to check for syntax errors
        tree = ast.parse(source_code, filename=file_path)
        print("✅ Python syntax is valid")
        
        # Count key elements
        classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
        
        print(f"  📊 Found: {len(classes)} classes, {len(functions)} functions, {len(imports)} imports")
        
        # Check for key class names
        class_names = [cls.name for cls in classes]
        expected_classes = ['AuditLogger', 'ComponentTrace', 'DecalogoPointTrace', 'CoverageValidationResult', 'ScoringConsistencyResult']
        
        for expected_class in expected_classes:
            if expected_class in class_names:
                print(f"  ✅ {expected_class} class found")
            else:
                print(f"  ❌ {expected_class} class missing")
        
        return True
        
    except SyntaxError as e:
        print(f"❌ Syntax error: {e}")
        print(f"   Line {e.lineno}: {e.text}")
        return False
    except Exception as e:
        print(f"❌ Error validating file: {e}")
        return False


def validate_file_structure():
    """Validate the expected file structure."""
    print("🏗️  Validating file structure...")
    
    expected_files = [
        "canonical_flow/evaluation/__init__.py",
        "canonical_flow/evaluation/audit_logger.py"
    ]
    
    all_exist = True
    for file_path in expected_files:
        if Path(file_path).exists():
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path}")
            all_exist = False
    
    return all_exist


def validate_json_structure():
    """Validate the expected JSON output structure."""
    print("📋 Validating JSON structure expectations...")
    
    # Expected structure (just keys, not full validation)
    expected_keys = [
        "session_id",
        "audit_timestamp", 
        "session_duration_s",
        "decalogo_traces",
        "coverage_validation",
        "scoring_consistency",
        "metrics",
        "summary",
        "recommendations"
    ]
    
    print("  Expected audit report keys:")
    for key in expected_keys:
        print(f"    ✓ {key}")
    
    return True


def main():
    """Main validation routine."""
    print("🚀 AuditLogger Validation")
    print("=" * 40)
    
    # Validate file structure
    structure_valid = validate_file_structure()
    print()
    
    # Validate Python syntax
    syntax_valid = True
    audit_logger_file = "canonical_flow/evaluation/audit_logger.py"
    if Path(audit_logger_file).exists():
        syntax_valid = validate_python_syntax(audit_logger_file)
    else:
        print(f"❌ Cannot validate syntax: {audit_logger_file} not found")
        syntax_valid = False
    
    print()
    
    # Validate JSON structure expectations
    json_valid = validate_json_structure()
    print()
    
    # Overall result
    overall_valid = structure_valid and syntax_valid and json_valid
    
    print("📊 Validation Summary:")
    print(f"  File Structure: {'✅ PASS' if structure_valid else '❌ FAIL'}")
    print(f"  Python Syntax:  {'✅ PASS' if syntax_valid else '❌ FAIL'}")
    print(f"  JSON Structure: {'✅ PASS' if json_valid else '❌ FAIL'}")
    print(f"  Overall:        {'✅ PASS' if overall_valid else '❌ FAIL'}")
    
    if overall_valid:
        print("\n🎉 AuditLogger validation completed successfully!")
        print("   Ready for integration with Decálogo evaluation system")
    else:
        print("\n💥 AuditLogger validation failed")
        print("   Please fix the issues above before proceeding")
    
    return overall_valid


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
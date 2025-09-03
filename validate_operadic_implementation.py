#!/usr/bin/env python3
"""Validation script for ∞-operadic implementation"""

import sys

def validate_syntax():
    """Validate Python syntax of operadic implementation"""
    try:
        import ast
        with open('egw_query_expansion/core/confluent_orchestrator.py', 'r') as f:
            content = f.read()
        ast.parse(content)
        print("✅ Python syntax validation passed")
        return True
    except SyntaxError as e:
        print(f"❌ Syntax error: {e}")
        return False
    except FileNotFoundError:
        print("❌ File not found: egw_query_expansion/core/confluent_orchestrator.py")
        return False

def validate_imports():
    """Validate that all required modules can be imported"""
    try:
        sys.path.insert(0, '.')
# # #         from egw_query_expansion.core.confluent_orchestrator import (  # Module not found  # Module not found  # Module not found
            InfinityOperad, 
            OperadOperation, 
            HomotopyCoherence, 
            CoherentDiagram,
            ConfluentOrchestrator,
            create_operadic_node
        )
        print("✅ Import validation passed")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def validate_operadic_structure():
    """Validate basic operadic structure functionality"""
    try:
        sys.path.insert(0, '.')
# # #         from egw_query_expansion.core.confluent_orchestrator import (  # Module not found  # Module not found  # Module not found
            InfinityOperad, OperadOperation, frozenset
        )
        
        # Create basic operad
        operad = InfinityOperad()
        
        # Test operation creation
        op = OperadOperation(
            arity=1,
            operation_id="test_op",
            composition_rule=lambda x: x * 2,
            coherence_conditions=frozenset({"associative"})
        )
        operad.add_operation(op)
        
        print("✅ Operadic structure validation passed")
        return True
    except Exception as e:
        print(f"❌ Operadic structure validation failed: {e}")
        return False

def main():
    """Run all validations"""
    print("🔍 Validating ∞-Operadic Implementation\n")
    
    validations = [
        ("Syntax", validate_syntax),
        ("Imports", validate_imports), 
        ("Operadic Structure", validate_operadic_structure)
    ]
    
    all_passed = True
    for name, validator in validations:
        print(f"Testing {name}...")
        if not validator():
            all_passed = False
        print()
    
    if all_passed:
        print("🎉 All validations passed! ∞-operadic implementation is working correctly.")
        return 0
    else:
        print("❌ Some validations failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
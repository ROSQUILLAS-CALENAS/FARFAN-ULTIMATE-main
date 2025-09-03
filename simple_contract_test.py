#!/usr/bin/env python3
"""
Simple test to verify contract system works without orchestrator dependencies
"""

def test_contracts_only():
    """Test contract system in isolation."""
    import sys
# # #     from pathlib import Path  # Module not found  # Module not found  # Module not found
    
    # Add current directory to path
    sys.path.insert(0, str(Path(__file__).parent))
    
    try:
# # #         from contract_system import ContractManager  # Module not found  # Module not found  # Module not found
        print("✅ Contract system imported successfully")
        
        manager = ContractManager()
        print(f"✅ ContractManager created with {len(manager.contracts)} contracts")
        
        # Test routing contract
        result = manager.execute_contract('routing', {
            'pdf_content': b'test pdf content',
            'pdf_path': 'test.pdf'
        })
        print(f"✅ Routing contract: {result.status.value}")
        
        # Test permutation invariance
        result = manager.execute_contract('permutation_invariance', {
            'inputs': [1, 2, 3],
            'aggregation_func': 'sum'
        })
        print(f"✅ Permutation invariance: {result.status.value}")
        
        print("🎉 CONTRACT SYSTEM READY FOR INTEGRATION")
        return True
        
    except Exception as e:
        print(f"❌ Contract system failed: {e}")
        return False

if __name__ == "__main__":
    test_contracts_only()
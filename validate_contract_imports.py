#!/usr/bin/env python3
"""
Quick validation script to test contract imports
"""

import sys
# # # from pathlib import Path  # Module not found  # Module not found  # Module not found

# Add project root to path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

def test_snapshot_contract_imports():
    """Test snapshot contract imports"""
    try:
# # #         from snapshot_manager import (  # Module not found  # Module not found  # Module not found
            get_current_snapshot_id,
            mount_snapshot,
            replay_output,
            requires_snapshot,
            resolve_snapshot,
        )
        print("✅ Snapshot manager imports successful")
        return True
    except ImportError as e:
        print(f"❌ Snapshot manager import failed: {e}")
        return False

def test_routing_contract_imports():
    """Test routing contract imports"""
    try:
# # #         from egw_query_expansion.core.deterministic_router import (  # Module not found  # Module not found  # Module not found
            DeterministicRouter,
            RoutingContext,
        )
        print("✅ Deterministic router imports successful")
        return True
    except ImportError as e:
        print(f"❌ Deterministic router import failed: {e}")
        return False

def test_immutable_context_imports():
    """Test immutable context imports"""
    try:
# # #         from egw_query_expansion.core.immutable_context import (  # Module not found  # Module not found  # Module not found
            QuestionContext,
            create_question_context,
        )
        print("✅ Immutable context imports successful")
        return True
    except ImportError as e:
        print(f"❌ Immutable context import failed: {e}")
        return False

def main():
    print("🔍 Testing contract imports...")
    
    results = []
    results.append(test_snapshot_contract_imports())
    results.append(test_routing_contract_imports())  
    results.append(test_immutable_context_imports())
    
    passed = sum(results)
    total = len(results)
    
    print(f"\n📊 Results: {passed}/{total} import tests passed")
    
    if passed == total:
        print("🎉 All contract imports are working correctly!")
        return 0
    else:
        print("⚠️  Some contract imports need attention")
        return 1

if __name__ == "__main__":
    sys.exit(main())
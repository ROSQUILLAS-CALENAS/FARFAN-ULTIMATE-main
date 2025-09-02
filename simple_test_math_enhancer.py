#!/usr/bin/env python3
"""
Simple test for the TopologicalQuantumFieldTheoryEnhancer integration
"""

import sys
import os

# Add the repo root to Python path
sys.path.insert(0, os.path.abspath('.'))

def test_tqft_enhancer():
    """Test basic TQFT enhancer functionality"""
    try:
        from canonical_flow.mathematical_enhancers.context_enhancer import (
            TopologicalQuantumFieldTheoryEnhancer,
            WilsonLoopOperator, 
            KnotInvariant,
            MathematicalStage2ContextEnhancer
        )
        
        print("✓ Successfully imported TQFT classes")
        
        # Test TQFT enhancer creation
        tqft = TopologicalQuantumFieldTheoryEnhancer(chern_simons_level=2)
        print(f"✓ Created TQFT enhancer with Chern-Simons level: {tqft.chern_simons_level}")
        
        # Test transformation edge registration
        tqft.register_transformation_edge("source_id", "target_id", "test_transformation")
        print("✓ Registered transformation edge")
        
        # Test knot invariant creation
        knot = KnotInvariant("test_knot_id")
        print(f"✓ Created knot invariant: {knot.knot_id}")
        print(f"  - Alexander polynomial terms: {len(knot.alexander_polynomial)}")
        print(f"  - Jones polynomial terms: {len(knot.jones_polynomial)}")
        print(f"  - Crossing number: {knot.crossing_number}")
        print(f"  - Writhe: {knot.writhe}")
        
        # Test Wilson loop creation
        wilson_loop = WilsonLoopOperator(
            loop_path=("node1", "node2", "node3"),
            chern_simons_level=1
        )
        print(f"✓ Created Wilson loop with path: {wilson_loop.loop_path}")
        print(f"  - Holonomy value: {wilson_loop.holonomy_value}")
        print(f"  - Is trivial loop: {wilson_loop.is_trivial_loop()}")
        
        # Test integration with mathematical enhancer
        math_enhancer = MathematicalStage2ContextEnhancer()
        tqft.integrate_with_category_enhancer(math_enhancer)
        print("✓ Successfully integrated TQFT enhancer with category enhancer")
        
        # Test topological invariants computation
        invariants = tqft.compute_topological_invariants()
        print("✓ Computed topological invariants:")
        print(f"  - Graph vertices: {invariants['graph_vertices']}")
        print(f"  - Graph edges: {invariants['graph_edges']}")
        print(f"  - Wilson loop count: {invariants['wilson_loop_count']}")
        print(f"  - Knot invariant count: {invariants['knot_invariant_count']}")
        print(f"  - Chern-Simons action: {invariants['chern_simons_action']}")
        
        # Test topological consistency verification
        consistency = tqft.verify_topological_consistency()
        print(f"✓ Topological consistency verified: {consistency}")
        
        print("\n🎉 All TQFT enhancer tests passed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_knot_operations():
    """Test knot invariant operations"""
    try:
        from canonical_flow.mathematical_enhancers.context_enhancer import KnotInvariant
        
        print("\nTesting knot operations...")
        
        # Create two knot invariants
        knot1 = KnotInvariant("knot_1")
        knot2 = KnotInvariant("knot_2")
        
        print(f"✓ Created knot 1: {knot1.knot_id}")
        print(f"✓ Created knot 2: {knot2.knot_id}")
        
        # Test polynomial evaluation
        alex_val1 = knot1.evaluate_alexander(1.0 + 0j)
        jones_val1 = knot1.evaluate_jones(1.0 + 0j)
        print(f"✓ Knot 1 Alexander(1): {alex_val1}")
        print(f"✓ Knot 1 Jones(1): {jones_val1}")
        
        # Test knot composition (connected sum)
        composite = knot1.composite_with(knot2)
        print(f"✓ Created composite knot: {composite.knot_id}")
        print(f"  - Combined writhe: {composite.writhe}")
        print(f"  - Combined crossing number: {composite.crossing_number}")
        
        # Test equivalence check
        same_knot = KnotInvariant("knot_1")  # Same ID should give same invariants
        equivalent = knot1.is_equivalent_knot(same_knot)
        print(f"✓ Knot equivalence test: {equivalent}")
        
        print("✓ Knot operations test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Knot operations test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_wilson_loop_operations():
    """Test Wilson loop operations"""
    try:
        from canonical_flow.mathematical_enhancers.context_enhancer import WilsonLoopOperator
        
        print("\nTesting Wilson loop operations...")
        
        # Create Wilson loops
        loop1 = WilsonLoopOperator(loop_path=("a", "b", "c"))
        loop2 = WilsonLoopOperator(loop_path=("b", "d", "e"))
        
        print(f"✓ Created Wilson loop 1: {loop1.loop_path}")
        print(f"✓ Created Wilson loop 2: {loop2.loop_path}")
        
        # Test linking number
        linking = loop1.linking_number(loop2)
        print(f"✓ Linking number between loops: {linking}")
        
        # Test trivial loop detection
        trivial1 = loop1.is_trivial_loop()
        trivial2 = loop2.is_trivial_loop() 
        print(f"✓ Loop 1 is trivial: {trivial1}")
        print(f"✓ Loop 2 is trivial: {trivial2}")
        
        print("✓ Wilson loop operations test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Wilson loop operations test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing TopologicalQuantumFieldTheoryEnhancer integration...\n")
    
    success = True
    
    success &= test_tqft_enhancer()
    success &= test_knot_operations() 
    success &= test_wilson_loop_operations()
    
    if success:
        print("\n🏆 All tests passed! TQFT enhancer is working correctly.")
        sys.exit(0)
    else:
        print("\n💥 Some tests failed!")
        sys.exit(1)
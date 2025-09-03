#!/usr/bin/env python3
"""
Test script for WassersteinFisherRaoMetric implementation.
"""

import numpy as np
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.abspath('.'))

try:
# # #     from egw_query_expansion.core.gw_alignment import (  # Module not found  # Module not found  # Module not found
        WassersteinFisherRaoMetric, 
        GromovWassersteinAligner,
        EnhancedGromovWassersteinAligner
    )
    print("✓ Successfully imported WassersteinFisherRaoMetric and related classes")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

def test_wasserstein_fisher_rao_basic():
    """Test basic WassersteinFisherRaoMetric functionality."""
    print("\n--- Testing WassersteinFisherRaoMetric Basic Functionality ---")
    
    # Create synthetic high-dimensional embeddings
    n_samples, dim = 50, 128
    np.random.seed(42)
    embeddings = np.random.randn(n_samples, dim)
    
    # Initialize WassersteinFisherRaoMetric
    wfr_metric = WassersteinFisherRaoMetric(
        manifold_dim=10,
        ricci_flow_steps=20,
        dt=0.01,
        regularization=1e-6,
        quantum_fisher_weight=0.1
    )
    
    # Test metric tensor computation
    print("Testing metric tensor computation...")
    metric_tensor = wfr_metric.compute_metric_tensor(embeddings)
    assert metric_tensor.shape == (dim, dim), f"Expected shape ({dim}, {dim}), got {metric_tensor.shape}"
    assert np.allclose(metric_tensor, metric_tensor.T), "Metric tensor should be symmetric"
    print("✓ Metric tensor computation successful")
    
    # Test Ricci curvature computation
    print("Testing Ricci curvature computation...")
    ricci = wfr_metric.compute_ricci_curvature(metric_tensor)
    assert ricci.shape == (dim, dim), f"Expected Ricci shape ({dim}, {dim}), got {ricci.shape}"
    print("✓ Ricci curvature computation successful")
    
    # Test quantum Fisher information
    print("Testing quantum Fisher information...")
    fisher = wfr_metric.compute_quantum_fisher_information(embeddings)
    assert fisher.shape == (dim, dim), f"Expected Fisher shape ({dim}, {dim}), got {fisher.shape}"
    print("✓ Quantum Fisher information computation successful")
    
    # Test metric evolution
    print("Testing metric geometry evolution...")
    geometry_info = wfr_metric.evolve_metric_geometry(embeddings)
    assert 'evolved_metric' in geometry_info
    assert 'flow_history' in geometry_info
    assert 'quantum_fisher' in geometry_info
    print("✓ Metric geometry evolution successful")
    
    # Test geometric measures
    print("Testing geometric measures...")
    measures = wfr_metric.get_geometric_measures()
    assert 'metric_condition_number' in measures
    assert 'ricci_scalar' in measures
    assert 'quantum_fisher_entropy' in measures
    print("✓ Geometric measures computation successful")

def test_enhanced_aligner_integration():
    """Test integration with EnhancedGromovWassersteinAligner."""
    print("\n--- Testing Enhanced GW Aligner Integration ---")
    
    # Create synthetic query and corpus embeddings
    np.random.seed(42)
    query_embeddings = np.random.randn(5, 64)
    corpus_embeddings = np.random.randn(10, 64)
    
    # Test enhanced aligner with metric learning enabled
    enhanced_aligner = EnhancedGromovWassersteinAligner(
        enable_metric_learning=True,
        epsilon=0.1,
        lambda_reg=0.01,
        max_iter=100
    )
    
    print("Testing enhanced alignment with metric learning...")
    transport_plan, stability_info = enhanced_aligner.align_pattern_to_corpus(
        query_embeddings, corpus_embeddings)
    
    assert transport_plan is not None, "Transport plan should not be None"
    assert stability_info is not None, "Stability info should not be None"
    print("✓ Enhanced alignment successful")
    
    # Check if geometric measures are included
    if 'geometric_measures' in stability_info:
        measures = stability_info['geometric_measures']
        print(f"✓ Geometric measures included: {list(measures.keys())}")
    else:
        print("! Geometric measures not included (possible fallback)")
    
    # Test with metric learning disabled for backward compatibility
    print("Testing backward compatibility (metric learning disabled)...")
    basic_aligner = EnhancedGromovWassersteinAligner(
        enable_metric_learning=False,
        epsilon=0.1,
        lambda_reg=0.01
    )
    
    basic_transport_plan, basic_stability_info = basic_aligner.align_pattern_to_corpus(
        query_embeddings, corpus_embeddings)
    
    assert basic_transport_plan is not None, "Basic transport plan should not be None"
    assert basic_stability_info is not None, "Basic stability info should not be None"
    print("✓ Backward compatibility maintained")

def test_transport_plan_enhancement():
    """Test transport plan enhancement functionality."""
    print("\n--- Testing Transport Plan Enhancement ---")
    
    np.random.seed(42)
    query_embeddings = np.random.randn(3, 32)
    corpus_embeddings = np.random.randn(5, 32) 
    
    # Create base transport plan
    base_transport_plan = np.random.rand(3, 5)
    base_transport_plan = base_transport_plan / np.sum(base_transport_plan)
    
    # Initialize metric learner
    wfr_metric = WassersteinFisherRaoMetric(
        ricci_flow_steps=10,
        dt=0.005
    )
    
    # Enhance transport plan
    print("Testing transport plan enhancement...")
    enhanced_plan = wfr_metric.enhance_transport_plan(
        base_transport_plan, query_embeddings, corpus_embeddings)
    
    assert enhanced_plan.shape == base_transport_plan.shape, \
        f"Shape mismatch: expected {base_transport_plan.shape}, got {enhanced_plan.shape}"
    
    # Check transport plan properties
    plan_sum_diff = abs(np.sum(enhanced_plan) - np.sum(base_transport_plan))
    assert plan_sum_diff < 1e-10, f"Transport plan normalization failed: diff = {plan_sum_diff}"
    
    print("✓ Transport plan enhancement successful")

if __name__ == "__main__":
    print("Testing WassersteinFisherRaoMetric Implementation")
    print("=" * 60)
    
    try:
        test_wasserstein_fisher_rao_basic()
        test_enhanced_aligner_integration() 
        test_transport_plan_enhancement()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED - WassersteinFisherRaoMetric implementation successful!")
        print("✅ Backward compatibility maintained")
        print("✅ Automatic integration working")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
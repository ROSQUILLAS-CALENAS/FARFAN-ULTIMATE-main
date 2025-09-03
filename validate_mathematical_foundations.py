#!/usr/bin/env python3
"""
Simple validation script for mathematical foundations implementation
Tests core functionality without requiring full installation
"""

import sys
import numpy as np
# # # from pathlib import Path  # Module not found  # Module not found  # Module not found

# Add the project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_information_theory():
    """Test information theory components"""
    try:
# # #         from egw_query_expansion.mathematical_foundations import InformationTheory  # Module not found  # Module not found  # Module not found
        
        # Test Shannon entropy
        uniform_prob = np.ones(4) / 4
        entropy = InformationTheory.shannon_entropy(uniform_prob, base='2')
        
        print(f"Shannon entropy test: {entropy:.6f} (expected: 2.0)")
        assert abs(entropy - 2.0) < 1e-10, f"Expected 2.0, got {entropy}"
        
        # Test KL divergence
        p = np.array([0.3, 0.4, 0.3])
        q = np.array([0.3, 0.4, 0.3])
        kl_div = InformationTheory.kl_divergence(p, q)
        
        print(f"KL divergence test: {kl_div:.6f} (expected: ~0.0)")
        assert abs(kl_div) < 1e-6, f"Expected ~0.0, got {kl_div}"
        
        print("✓ Information theory tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Information theory test failed: {e}")
        return False

def test_semantic_similarity():
    """Test semantic similarity components"""
    try:
# # #         from egw_query_expansion.mathematical_foundations import SemanticSimilarity  # Module not found  # Module not found  # Module not found
        
        # Test distributional similarity with simple data
        embeddings1 = np.random.normal(0, 1, (5, 3))
        embeddings2 = np.random.normal(0, 1, (5, 3))
        
        similarity = SemanticSimilarity.distributional_similarity(
            embeddings1, embeddings2, method='wasserstein'
        )
        
        print(f"Distributional similarity: {similarity:.6f}")
        assert 0 <= similarity <= 1, f"Similarity should be in [0,1], got {similarity}"
        
        # Test coherence
        coherence = SemanticSimilarity.semantic_coherence(
            embeddings1, method='intra_cluster'
        )
        
        print(f"Semantic coherence: {coherence:.6f}")
        assert -1 <= coherence <= 1, f"Coherence should be in [-1,1], got {coherence}"
        
        print("✓ Semantic similarity tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Semantic similarity test failed: {e}")
        return False

def test_bayesian_inference():
    """Test Bayesian inference components"""
    try:
# # #         from egw_query_expansion.mathematical_foundations import BayesianInference  # Module not found  # Module not found  # Module not found
        
        # Test beta-binomial posterior
        result = BayesianInference.beta_binomial_posterior(
            successes=7, trials=10, alpha_prior=1.0, beta_prior=1.0
        )
        
        expected_mean = 8/12  # (7+1)/(10+2)
        print(f"Bayesian posterior mean: {result.posterior_mean:.6f} (expected: {expected_mean:.6f})")
        assert abs(result.posterior_mean - expected_mean) < 0.01
        
        assert result.posterior_std > 0, "Posterior std should be positive"
        assert len(result.posterior_samples) == 1000, "Should have 1000 samples"
        
        print("✓ Bayesian inference tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Bayesian inference test failed: {e}")
        return False

def test_utility_functions():
    """Test utility functions"""
    try:
# # #         from egw_query_expansion.mathematical_foundations import (  # Module not found  # Module not found  # Module not found
            compute_question_entropy_features,
            bootstrap_similarity_test
        )
        
        # Test entropy features
        embeddings = np.random.normal(0, 1, 20)
        features = compute_question_entropy_features("test question", embeddings)
        
        expected_keys = [
            'shannon_entropy', 'max_entropy', 'entropy_ratio',
            'effective_vocab_size', 'concentration_index'
        ]
        
        for key in expected_keys:
            assert key in features, f"Missing key: {key}"
            assert isinstance(features[key], (int, float)), f"Key {key} should be numeric"
        
        print(f"Entropy features computed: {len(features)} features")
        
        # Test bootstrap similarity
        emb1 = np.random.normal(0, 1, (5, 3))
        emb2 = np.random.normal(0, 1, (5, 3))
        
        result = bootstrap_similarity_test(emb1, emb2, n_bootstrap=10)
        
        expected_keys = [
            'observed_similarity', 'p_value', 'confidence_interval',
            'effect_size', 'bootstrap_mean', 'bootstrap_std'
        ]
        
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"
        
        print("✓ Utility functions tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Utility functions test failed: {e}")
        return False

def test_question_analyzer_integration():
    """Test integration with QuestionAnalyzer (minimal test without full models)"""
    try:
        # Import QuestionAnalyzer to check basic compatibility
        import question_analyzer
        
        # Check that required classes are available
        assert hasattr(question_analyzer, 'QuestionAnalyzer'), "QuestionAnalyzer class not found"
        assert hasattr(question_analyzer, 'CausalPosture'), "CausalPosture enum not found"
        assert hasattr(question_analyzer, 'QuestionRequirements'), "QuestionRequirements class not found"
        
        print("✓ QuestionAnalyzer integration structure verified")
        return True
        
    except Exception as e:
        print(f"✗ QuestionAnalyzer integration test failed: {e}")
        return False

def main():
    """Run all validation tests"""
    print("=== Mathematical Foundations Validation ===")
    print()
    
    tests = [
        ("Information Theory", test_information_theory),
        ("Semantic Similarity", test_semantic_similarity), 
        ("Bayesian Inference", test_bayesian_inference),
        ("Utility Functions", test_utility_functions),
        ("QuestionAnalyzer Integration", test_question_analyzer_integration),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"Testing {test_name}...")
        if test_func():
            passed += 1
        print()
    
    print("=== Validation Summary ===")
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All mathematical foundations tests passed!")
        return True
    else:
        print("❌ Some tests failed. Check implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
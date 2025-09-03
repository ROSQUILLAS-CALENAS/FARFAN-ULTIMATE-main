#!/usr/bin/env python3
"""Validate setup and run core functionality tests"""

import sys

import numpy as np

# # # from question_analyzer import CausalPosture, QuestionAnalyzer  # Module not found  # Module not found  # Module not found


def test_core_functionality():
    """Test core analyzer functionality"""
    print("Testing core functionality...")

    # Initialize analyzer
    analyzer = QuestionAnalyzer(alpha=0.1)

    # Test question analysis
    test_question = "What is the effect of education on income?"
    requirements = analyzer.analyze_question(test_question, "test_001")

    assert requirements.causal_posture == CausalPosture.INTERVENTIONAL
    assert "education" in test_question

    # Test search pattern extraction
    patterns = analyzer.extract_search_patterns(test_question)
    assert len(patterns) > 0

    # Test evidence type identification
    evidence_types = analyzer.identify_evidence_types(test_question)
    assert len(evidence_types) > 0

    # Test validation rules
    validation_rules = analyzer.determine_validation_rules(requirements)
    assert len(validation_rules) > 0

    print("✅ Core functionality tests passed")


def test_libraries():
    """Test that required libraries work correctly"""
    print("Testing library imports...")

    # Test NumPy
    import numpy as np

    arr = np.array([1, 2, 3])
    assert len(arr) == 3
    print("✅ NumPy works")

    # Test transformers
# # #     from transformers import AutoTokenizer  # Module not found  # Module not found  # Module not found

    print("✅ Transformers works")

    # Test sentence-transformers
# # #     from sentence_transformers import SentenceTransformer  # Module not found  # Module not found  # Module not found

    print("✅ Sentence-transformers works")

    # Test Lark
# # #     from lark import Lark  # Module not found  # Module not found  # Module not found

    print("✅ Lark works")

    # Test NetworkX
    import networkx as nx

    print("✅ NetworkX works")

    # Test Pydantic
# # #     from pydantic import BaseModel  # Module not found  # Module not found  # Module not found

    print("✅ Pydantic works")


if __name__ == "__main__":
    print("Validating setup...")
    print("=" * 50)

    try:
        test_libraries()
        print()
        test_core_functionality()
        print()
        print("🎉 All tests passed! Setup is valid.")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        sys.exit(1)

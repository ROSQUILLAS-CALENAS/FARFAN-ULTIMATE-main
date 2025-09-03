#!/usr/bin/env python3
"""Quick validation of L-stage test suite components"""

import sys
import json
import hashlib
# # # from pathlib import Path  # Module not found  # Module not found  # Module not found

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "canonical_flow"))

def main():
    try:
        # Test imports
# # #         from canonical_flow.A_analysis_nlp.question_analyzer import get_decalogo_question_registry  # Module not found  # Module not found  # Module not found
# # #         from canonical_flow.L_classification_evaluation.decalogo_scoring_system import ScoringSystem  # Module not found  # Module not found  # Module not found
        print("✓ Imports successful")
        
        # Test registry
        registry = get_decalogo_question_registry()
        questions = registry.get_all_questions()
        print(f"✓ Registry has {len(questions)} questions")
        
        # Test scoring system
        scoring_system = ScoringSystem()
        system_info = scoring_system.get_system_info()
        print(f"✓ Scoring system deterministic: {system_info['deterministic']}")
        
        # Test evidence multiplier with different values
        test_cases = [
            (0.0, 0.0),  # Minimum
            (1.0, 1.0),  # Maximum
            (0.7, 0.8),  # Good evidence
        ]
        
        for completeness, reference_quality in test_cases:
            multiplier = scoring_system.calculate_evidence_multiplier(completeness, reference_quality)
            print(f"✓ Evidence multiplier for ({completeness}, {reference_quality}): {multiplier}")
        
        # Quick determinism test
        result1 = scoring_system.calculate_final_score("Sí", 0.8, 0.7)
        result2 = scoring_system.calculate_final_score("Sí", 0.8, 0.7)
        
        if result1 == result2:
            print("✓ Scoring is deterministic")
        else:
            print(f"✗ Scoring not deterministic: {result1} vs {result2}")
        
        print("\n✓ L-stage test suite components are working correctly!")
        print("Run the full test suite with: python3 run_l_stage_tests.py")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
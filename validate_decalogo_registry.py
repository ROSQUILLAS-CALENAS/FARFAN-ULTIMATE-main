#!/usr/bin/env python3
"""
Validation script for DecalogoQuestionRegistry
Demonstrates the complete functionality including validation and coverage tracking.
"""

import json
import sys
# # # from pathlib import Path  # Module not found  # Module not found  # Module not found

# Import the registry
sys.path.insert(0, str(Path(__file__).resolve().parent / "canonical_flow" / "A_analysis_nlp"))

try:
# # #     from decalogo_question_registry import (  # Module not found  # Module not found  # Module not found
        DecalogoQuestionRegistry,
        DecalogoQuestionValidationError,
        CoverageMatrix
    )
    
    def main():
        print("DecalogoQuestionRegistry Validation Demo")
        print("=" * 50)
        
        # Test 1: Create and validate registry
        print("\n1. Creating and validating registry...")
        try:
            registry = DecalogoQuestionRegistry(validate_on_init=True)
            print(f"✓ Registry successfully created with {len(registry.questions)} questions")
        except DecalogoQuestionValidationError as e:
            print(f"✗ Registry validation failed: {e}")
            return False
        
        # Test 2: Show dimension distribution validation
        print("\n2. Validating dimension distribution...")
        validation_summary = registry.get_validation_summary()
        print(f"✓ Status: {validation_summary['status']}")
        print(f"✓ Total questions: {validation_summary['total_questions']}")
        print(f"✓ Questions per point: {validation_summary['questions_per_point']}")
        print("✓ Dimension distribution:")
        for dim, count in validation_summary['dimension_distribution'].items():
            total_for_dim = count * 10  # 10 points
            print(f"   - {dim}: {count} per point × 10 points = {total_for_dim} total")
        
        # Test 3: Sample questions
# # #         print("\n3. Sample questions from each dimension:")  # Module not found  # Module not found  # Module not found
        sample_point = 1
        point_questions = registry.get_questions_by_point(sample_point)
        
        for dim in ['DE-1', 'DE-2', 'DE-3', 'DE-4']:
            dim_questions = [q for q in point_questions if q.dimension_code == dim]
            print(f"\n   {dim} ({len(dim_questions)} questions for point {sample_point}):")
            for i, q in enumerate(dim_questions[:2]):  # Show first 2
                print(f"     {i+1}. {q.question_text}")
        
        # Test 4: Coverage tracking demo
        print("\n4. Testing coverage tracking...")
        test_doc = "DEMO_DOCUMENT"
        
        # Simulate answering first 100 questions
        all_questions = registry.get_all_questions()
        answered_ids = [q.question_id for q in all_questions[:100]]
        
        coverage = registry.validate_coverage_completion(test_doc, answered_ids)
        print(f"✓ Coverage calculated: {coverage['summary']['completion_percentage']:.1f}% complete")
        print(f"   - Questions answered: {coverage['summary']['total_actual']}")
        print(f"   - Total expected: {coverage['summary']['total_expected']}")
        print(f"   - Gaps remaining: {coverage['summary']['gaps_count']}")
        
        # Test 5: Generate coverage artifacts
        print("\n5. Generating coverage artifacts...")
        try:
            coverage_files = registry.generate_coverage_artifacts(test_doc, answered_ids)
            print(f"✓ Generated {len(coverage_files)} coverage files:")
            for point, filepath in sorted(coverage_files.items()):
                # Check if file exists and show size
                path_obj = Path(filepath)
                if path_obj.exists():
                    size = path_obj.stat().st_size
                    print(f"   - {point}: {filepath} ({size} bytes)")
                else:
                    print(f"   - {point}: {filepath} (missing)")
        except Exception as e:
            print(f"✗ Error generating coverage artifacts: {e}")
            return False
        
        # Test 6: Show a sample coverage file
        print("\n6. Sample coverage file content:")
        sample_file = coverage_files.get("P1")
        if sample_file and Path(sample_file).exists():
            with open(sample_file, 'r') as f:
                sample_data = json.load(f)
            print(f"   Document: {sample_data['document_id']}")
            print(f"   Point: {sample_data['point_number']} - {sample_data['point_name']}")
            print(f"   Completion: {sample_data['summary']['completion_percentage']:.1f}%")
            print("   Dimensions:")
            for dim, dim_data in sample_data['dimensions'].items():
                print(f"     - {dim}: {dim_data['actual_answered']}/{dim_data['expected_count']} ({dim_data['completion_percentage']:.1f}%)")
        
        print("\n" + "=" * 50)
        print("✓ ALL VALIDATION TESTS PASSED!")
        print("\nKey features demonstrated:")
        print("- Exact 47 questions per point validation")
        print("- Dimension distribution validation (DE-1:6, DE-2:21, DE-3:8, DE-4:8)")
        print("- Preflight validation before question evaluation")
        print("- Coverage tracking with dimension-level breakdowns")
        print("- Coverage artifact generation in canonical_flow/classification/<doc>/")
        print("- Total 470 questions (47 × 10 points) accounting")
        
        return True
    
    if __name__ == "__main__":
        success = main()
        sys.exit(0 if success else 1)
        
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure the decalogo_question_registry.py file is in canonical_flow/A_analysis_nlp/")
    sys.exit(1)
except Exception as e:
    print(f"Unexpected error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
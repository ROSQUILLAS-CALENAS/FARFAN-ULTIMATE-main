#!/usr/bin/env python3
"""
Validation script for decalogo JSON artifacts.
"""

import json
# # # from pathlib import Path  # Module not found  # Module not found  # Module not found

def validate_artifacts():
    """Validate the generated decalogo JSON artifacts."""
    analysis_dir = Path("canonical_flow/analysis")
    
    print("Validating Decalogo JSON Artifacts...")
    print("="*50)
    
    total_files = 0
    total_questions = 0
    
    for point_id in range(1, 11):
        filename = f"decalogo_point_{point_id}.json"
        file_path = analysis_dir / filename
        
        if not file_path.exists():
            print(f"❌ Missing: {filename}")
            continue
        
        total_files += 1
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Validate structure
            point_questions = data['total_questions']
            total_questions += point_questions
            
            print(f"✅ Point {point_id}: {point_questions} questions")
            
            # Validate dimensions
            dimensions = data.get('dimensions', {})
            print(f"   Dimensions: {list(dimensions.keys())}")
            
            for dim, questions in dimensions.items():
                print(f"   {dim}: {len(questions)} questions")
                
                # Check first question structure
                if questions:
                    q = questions[0]
                    required_fields = ['question_id', 'question_text', 'answer_classification', 
                                     'evidence_items', 'scoring']
                    missing = [f for f in required_fields if f not in q]
                    if missing:
                        print(f"   ⚠️  Missing fields in {dim}: {missing}")
                    
                    # Validate evidence items
                    for evidence in q.get('evidence_items', []):
                        if 'page_num' not in evidence or 'exact_text' not in evidence:
                            print(f"   ⚠️  Invalid evidence in {dim}: missing required fields")
            
            # Validate scoring
            dimension_scores = data.get('dimension_scores', {})
            overall_score = data.get('overall_score', 0)
            
            print(f"   Overall Score: {overall_score:.2f}")
            print(f"   Dimension Scores: {len(dimension_scores)} dimensions")
            
        except Exception as e:
            print(f"❌ Error reading {filename}: {e}")
    
    print("="*50)
    print(f"Summary:")
    print(f"Files Found: {total_files}/10")
    print(f"Total Questions: {total_questions}")
    print(f"Average Questions per Point: {total_questions/max(total_files, 1):.1f}")
    
    # Validate exactly 47 questions per point
    expected_per_point = 47
    if total_files > 0:
        avg_questions = total_questions / total_files
        if abs(avg_questions - expected_per_point) < 1:
            print(f"✅ Question count validation: PASSED")
        else:
            print(f"⚠️  Question count validation: Expected {expected_per_point}, got {avg_questions:.1f}")
    
    return total_files == 10 and total_questions == 470

if __name__ == "__main__":
    success = validate_artifacts()
    print(f"\n{'SUCCESS' if success else 'PARTIAL SUCCESS'}")
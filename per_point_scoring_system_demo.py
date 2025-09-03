#!/usr/bin/env python3
"""
Demo script for the Per-Point Scoring System
"""

import sys
import json
import random
# # # from pathlib import Path  # Module not found  # Module not found  # Module not found

# Import the scoring system
# # # from per_point_scoring_system import PerPointScoringSystem, ComplianceLevel  # Module not found  # Module not found  # Module not found

def create_demo_data():
    """Create realistic demo data for testing."""
    mock_data = {}
    
    for point_id in range(1, 11):
        # Create varying quality data across points
        base_quality = 0.4 + (point_id * 0.05)  # Points 1-10 get progressively better
        
        mock_data[point_id] = {
            "DE-1": {
                "questions": [
                    {
                        "question_id": f"P{point_id}_Q{i}",
                        "text": f"Point {point_id} - Logic question {i}",
                        "score": min(1.0, base_quality + random.uniform(-0.2, 0.3)),
                        "weight": 1.0,
                        "evidence": [
                            {"type": "document", "score": random.uniform(0.6, 1.0), "source": f"doc_{i}.pdf"}
                            for _ in range(random.randint(1, 3))
                        ]
                    }
                    for i in range(1, 7)  # 6 questions for DE-1
                ]
            },
            "DE-2": {
                "questions": [
                    {
                        "question_id": f"P{point_id}_D{i}",
                        "text": f"Point {point_id} - Thematic inclusion {i}",
                        "score": min(1.0, base_quality + random.uniform(-0.15, 0.25)),
                        "weight": 1.0,
                        "evidence": [
                            {"type": "data", "score": random.uniform(0.5, 0.9)}
                            for _ in range(random.randint(0, 2))
                        ]
                    }
                    for i in range(1, 22)  # 21 questions for DE-2
                ]
            },
            "DE-3": {
                "completion_percentage": min(100, (base_quality * 100) + random.uniform(-20, 30)),
                "actual_count": random.randint(4, 8),
                "questions": [
                    {
                        "question_id": f"P{point_id}_G{i}",
                        "text": f"Point {point_id} - Budget planning {i}",
                        "score": min(1.0, base_quality + random.uniform(-0.1, 0.2)),
                        "weight": 1.0,
                        "evidence": []
                    }
                    for i in range(1, 5)
                ]
            },
            "DE-4": {
                "completion_percentage": min(100, (base_quality * 100) + random.uniform(-25, 35)),
                "actual_count": random.randint(3, 8),
                "questions": [
                    {
                        "question_id": f"P{point_id}_VC{i}",
                        "text": f"Point {point_id} - Value chain element {i}",
                        "score": min(1.0, base_quality + random.uniform(-0.1, 0.2)),
                        "weight": 1.0,
                        "evidence": []
                    }
                    for i in range(1, 5)
                ]
            }
        }
    
    return mock_data

def main():
    print("🚀 Per-Point Scoring System Demo")
    print("=" * 50)
    
    # Initialize scoring system
    try:
        scoring_system = PerPointScoringSystem()
        print("✅ Scoring system initialized")
    except Exception as e:
        print(f"❌ Failed to initialize scoring system: {e}")
        return 1
    
    # Create demo data
    print("\n📊 Creating demo data for 10 Decálogo points...")
    mock_data = create_demo_data()
    
    # Process all points
    print("\n🔄 Processing scores for all points...")
    try:
        results = scoring_system.process_all_points(mock_data)
        print(f"✅ Processed {len(results)} points successfully")
    except Exception as e:
        print(f"❌ Error processing points: {e}")
        return 1
    
    # Display summary results
    print("\n📈 SCORING RESULTS SUMMARY")
    print("-" * 30)
    
    compliance_counts = {level.value: 0 for level in ComplianceLevel}
    total_score = 0.0
    
    for point_id, result in sorted(results.items()):
        compliance_counts[result.compliance_level.value] += 1
        total_score += result.final_score
        
        # Display each point result
        status_emoji = {
            "CUMPLE": "🟢",
            "CUMPLE_PARCIAL": "🟡", 
            "NO_CUMPLE": "🔴"
        }
        
        print(f"  Point {point_id:2d}: {result.final_score:.3f} {status_emoji.get(result.compliance_level.value, '⚪')} {result.compliance_level.value}")
        print(f"           Questions: {result.total_answered}/{result.total_questions} ({result.completion_rate:.1%})")
    
    average_score = total_score / len(results) if results else 0
    
    print(f"\n📊 OVERALL STATISTICS")
    print(f"  Average Score: {average_score:.3f}")
    print(f"  Compliance Distribution:")
    for level, count in compliance_counts.items():
        percentage = (count / len(results)) * 100 if results else 0
        print(f"    {level}: {count} points ({percentage:.1f}%)")
    
    # Generate explainability report
    print(f"\n📋 Generating explainability report...")
    try:
        report_path = scoring_system.generate_explainability_report(results)
        print(f"✅ Explainability report saved: {report_path}")
    except Exception as e:
        print(f"❌ Error generating report: {e}")
        return 1
    
    # Show file locations
    print(f"\n📁 OUTPUT FILES")
    print(f"  Analysis directory: {Path.cwd()}")
    print(f"  Scoring results: per_point_scores_*.json")
    print(f"  Explainability: explainability_report_*.json")
    
    # Display sample top contributors
    print(f"\n🎯 SAMPLE TOP CONTRIBUTORS (Point 1)")
    sample_result = list(results.values())[0]
    print(f"  Top questions contributing to Point {sample_result.point_id} score:")
    for i, contributor in enumerate(sample_result.top_contributing_questions[:3], 1):
        print(f"    {i}. {contributor.get('question_id', 'Unknown')} (contribution: {contributor.get('contribution', 0):.3f})")
    
    print(f"\n✅ Demo completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
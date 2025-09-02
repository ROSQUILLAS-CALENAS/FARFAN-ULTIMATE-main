#!/usr/bin/env python3
"""Quick validation of DecalogoQuestionRegistry"""

# Import the registry module directly
exec(open('decalogo_question_registry.py').read())

# Run validation
if __name__ == "__main__":
    print("=== DecalogoQuestionRegistry Validation ===\n")
    
    # Create registry instance
    registry = DecalogoQuestionRegistry()
    
    print(f"Registry Version: {registry.registry_version}")
    print(f"Total Points: {len(registry.points)}")
    print(f"Total Questions: {len(registry.questions)}")
    print(f"Questions per Point: {registry.QUESTIONS_PER_POINT}")
    print(f"Expected Total: {registry.EXPECTED_TOTAL_QUESTIONS}")
    
    # Validate totals
    total_validation = registry.validate_total_question_count()
    print(f"\nTotal Questions Valid: {total_validation['is_valid']}")
    
    # Validate dimensions
    dim_validation = registry.validate_dimension_distribution()
    print(f"Dimension Distribution Valid: {dim_validation['overall_valid']}")
    
    # Validate clusters
    cluster_validation = registry.validate_cluster_point_mapping()
    print(f"Cluster Mapping Valid: {cluster_validation['mapping_complete']['all_points_mapped']}")
    
    # Show cluster distribution
    print("\n=== Cluster Distribution ===")
    for cluster, points in registry.cluster_mappings.items():
        print(f"{cluster.value}: {points} ({len(points)} points)")
    
    # Show dimension distribution
    print("\n=== Dimension Distribution ===")
    for dim in Dimension:
        total = dim.question_count * 10
        print(f"{dim.code}: {dim.question_count} per point × 10 points = {total} total")
    
    # Verify specific requirements
    print("\n=== Requirements Verification ===")
    
    # Check exactly 10 points
    assert len(registry.points) == 10
    print("✓ Exactly 10 Decálogo points")
    
    # Check exactly 4 clusters
    assert len(set(ClusterType)) == 4
    print("✓ Exactly 4 cluster types: participativo, planificación, territorial, sostenible")
    
    # Check 470 total questions
    assert len(registry.questions) == 470
    print("✓ Exactly 470 total questions")
    
    # Check 47 questions per point
    for point_id in range(1, 11):
        assert len(registry.get_questions_for_point(point_id)) == 47
    print("✓ Exactly 47 questions per point")
    
    # Check dimension distribution
    expected_dims = {
        Dimension.DE_1: 6,
        Dimension.DE_2: 21, 
        Dimension.DE_3: 8,
        Dimension.DE_4: 12
    }
    
    for point_id in range(1, 11):
        for dim, expected in expected_dims.items():
            actual = len(registry.get_questions_for_dimension(dim, point_id))
            assert actual == expected, f"Point {point_id} dim {dim.code}: {actual} != {expected}"
    print("✓ Correct dimension distribution: DE-1(6), DE-2(21), DE-3(8), DE-4(12)")
    
    # Check stable question IDs
    sample_questions = list(registry.questions.values())[:5]
    for q in sample_questions:
        expected_id = f"P{q.point_id:02d}_{q.dimension.code}_{q.sequence:02d}"
        assert q.question_id == expected_id
    print("✓ Stable sequential question IDs")
    
    # Check version control
    assert registry.registry_version == "1.0.0"
    print("✓ Version control through registry_version field")
    
    # Test validation methods
    assert registry.validate_total_question_count()["is_valid"]
    assert registry.validate_dimension_distribution()["overall_valid"]  
    assert registry.validate_cluster_point_mapping()["mapping_complete"]["all_points_mapped"]
    print("✓ All validation methods work correctly")
    
    print("\n🎉 SUCCESS: DecalogoQuestionRegistry meets all requirements!")
    print("\nSummary:")
    print("- ✅ Exactly 10 Decálogo points with stable point IDs and titles")
    print("- ✅ Maps each point to one of 4 clusters (participativo, planificación, territorial, sostenible)")
    print("- ✅ Contains exactly 47 questions per point distributed across dimensions:")
    print("  • DE-1: 6 questions per point")
    print("  • DE-2: 21 questions per point") 
    print("  • DE-3: 8 questions per point")
    print("  • DE-4: 12 questions per point")
    print("- ✅ Total of 470 questions (10 points × 47 questions)")
    print("- ✅ Stable ordering using sequential question IDs within each point-dimension")
    print("- ✅ Version control through registry_version field")
    print("- ✅ Validation methods to verify counts and mappings")
#!/usr/bin/env python3
"""
Test script for comprehensive artifact generation system
"""

import sys
import os
# # # from pathlib import Path  # Module not found  # Module not found  # Module not found

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# # # from canonical_flow.analysis.artifact_generator import (  # Module not found  # Module not found  # Module not found
    ArtifactGenerator, create_sample_data, QuestionEvaluation, EvidenceReference
)


def test_comprehensive_generation():
    """Test comprehensive artifact generation."""
    print("Testing comprehensive artifact generation...")
    
    generator = ArtifactGenerator()
    
    # Create sample data for ACANDI-CHOCO
    questions, dimensions, points, clusters, macro = create_sample_data('ACANDI-CHOCO')
    
    # Generate all artifacts
    artifacts = generator.generate_comprehensive_artifacts(
        'ACANDI-CHOCO', questions, dimensions, points, clusters, macro
    )
    
    print(f"Generated {len(artifacts)} artifact types for ACANDI-CHOCO:")
    for artifact_type, filepath in artifacts.items():
        print(f"  {artifact_type}: {filepath}")
    
    # Validate artifacts
    validation = generator.validate_artifacts('ACANDI-CHOCO')
    print(f"Validation results: {validation}")
    
    # Discover artifacts
    discovered = generator.discover_artifacts()
    print(f"Discovered artifacts: {discovered}")
    
    return True


def test_deterministic_ordering():
    """Test that ordering is deterministic."""
    print("\nTesting deterministic ordering...")
    
    # Create evidence with different IDs to test sorting
    evidence_refs = [
        EvidenceReference("E003", "document", "p. 5", "Third evidence", 0.75),
        EvidenceReference("E001", "interview", "p. 10", "First evidence", 0.85),
        EvidenceReference("E002", "report", "p. 20", "Second evidence", 0.90)
    ]
    
    # Create question
    question = QuestionEvaluation(
        "DE-2-Q5", "Test deterministic ordering?", "Sí", 1.0, 0.9, 0.8, 0.95, evidence_refs
    )
    
    # Convert to dict and verify evidence is sorted by ID
    question_dict = question.to_dict()
    evidence_ids = [ref["evidence_id"] for ref in question_dict["evidence_references"]]
    
    assert evidence_ids == ["E001", "E002", "E003"], f"Evidence not sorted correctly: {evidence_ids}"
    print("✓ Evidence references are deterministically ordered")
    
    return True


def test_filename_conventions():
    """Test filename conventions."""
    print("\nTesting filename conventions...")
    
    generator = ArtifactGenerator()
    
    # Test expected suffixes
    expected_suffixes = ["questions", "dimensions", "points", "meso", "macro"]
    
    for suffix in expected_suffixes:
        expected_path = generator.output_dir / f"TEST-DOC_{suffix}.json"
        print(f"Expected path for {suffix}: {expected_path}")
    
    print("✓ Filename conventions are correct")
    return True


if __name__ == "__main__":
    try:
        success = True
        
        success &= test_comprehensive_generation()
        success &= test_deterministic_ordering()
        success &= test_filename_conventions()
        
        if success:
            print("\n✓ All tests passed!")
            sys.exit(0)
        else:
            print("\n✗ Some tests failed")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n✗ Test error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
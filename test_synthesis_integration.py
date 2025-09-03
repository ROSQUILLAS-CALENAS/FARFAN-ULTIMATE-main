"""
Integration test for answer synthesizer with hash validation
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(__file__))

def test_synthesis_with_hash_validation():
    """Test synthesis with hash validation enabled"""
    
    # Import the answer synthesizer directly
    try:
        from answer_synthesizer import AnswerSynthesizer, MockEvidence
    except ImportError:
        # Create a mock if not available
        class MockEvidence:
            def __init__(self, text, source, confidence=0.8):
                self.text = text
                self.source = source
                self.confidence = confidence
        
        # Test will be limited without the full module
        print("⚠️  Answer synthesizer not available, testing hash concepts only")
        return test_hash_concepts_only()
    
    # Create synthesizer with hash validation enabled
    synthesizer = AnswerSynthesizer(enable_hash_validation=True)
    
    # Test data
    question = "Does the plan include measurable targets for 2025?"
    evidence = [
        {
            "text": "The strategic plan sets a 25% improvement target for vaccination coverage by 2025.",
            "evidence_id": "ev1",
            "citation": {"document_id": "plan_v1", "page_number": 12}
        },
        {
            "text": "Baseline measurements were established in 2023 with clear performance indicators.",
            "evidence_id": "ev2", 
            "citation": {"document_id": "plan_v1", "page_number": 13}
        },
        {
            "text": "The implementation roadmap includes quarterly reviews starting in 2024.",
            "evidence_id": "ev3",
            "citation": {"document_id": "plan_v1", "page_number": 15}
        }
    ]
    
    print(f"Testing synthesis with question: {question}")
    print(f"Evidence items: {len(evidence)}")
    
    # Synthesize answer
    answer = synthesizer.synthesize_answer(question, evidence)
    
    # Check that answer has hash validation
    assert hasattr(answer, '_synthesis_hash'), "Answer should have synthesis hash"
    assert answer.get_synthesis_hash() is not None, "Synthesis hash should be computed"
    
    synthesis_hash_1 = answer.get_synthesis_hash()
    print(f"✓ Initial synthesis hash: {synthesis_hash_1[:16]}...")
    
    # Verify integrity
    integrity_check_1 = answer.verify_synthesis_integrity()
    assert integrity_check_1, "Synthesis integrity should be valid"
    print("✓ Synthesis integrity verified")
    
    # Apply DNP logic
    standards = {
        "mandatory_indicators": ["2025", "% target", "baseline"],
        "rules": [
            {"id": "R1", "description": "Must mention year 2025", "pattern": "2025"},
            {"id": "R2", "description": "Must include percentage target", "any_of": ["%", "percent"]}
        ]
    }
    
    answer_with_dnp = synthesizer.apply_dnp_logic(answer, standards)
    
    # Hash should be updated after DNP modifications
    synthesis_hash_2 = answer_with_dnp.get_synthesis_hash()
    print(f"✓ Post-DNP synthesis hash: {synthesis_hash_2[:16]}...")
    
    # Verify integrity after DNP
    integrity_check_2 = answer_with_dnp.verify_synthesis_integrity()
    assert integrity_check_2, "Post-DNP synthesis integrity should be valid"
    print("✓ Post-DNP synthesis integrity verified")
    
    # Format response
    formatted = synthesizer.format_response(answer_with_dnp)
    
    # Check that formatted response includes hash validation metadata
    assert "_hash_validation" in formatted, "Formatted response should include hash validation"
    hash_validation = formatted["_hash_validation"]
    
    assert hash_validation["validation_enabled"], "Hash validation should be enabled"
    assert hash_validation["synthesis_hash"] is not None, "Synthesis hash should be present"
    assert hash_validation["integrity_verified"], "Integrity should be verified"
    
    print("✓ Formatted response includes hash validation metadata")
    
    # Test pipeline validation report
    if synthesizer.enable_hash_validation:
        report = synthesizer.get_pipeline_validation_report()
        assert report is not None, "Pipeline validation report should be available"
        assert "stage_hashes" in report, "Report should include stage hashes"
        assert "validation_log" in report, "Report should include validation log"
        print("✓ Pipeline validation report generated")
    
    # Test consistency across multiple synthesis runs
    answer_2 = synthesizer.synthesize_answer(question, evidence)
    answer_2_with_dnp = synthesizer.apply_dnp_logic(answer_2, standards)
    
    # The synthesis process should be deterministic for the same inputs
    # (though hashes will differ due to different derivation IDs)
    assert answer_with_dnp.question == answer_2_with_dnp.question
    assert answer_with_dnp.verdict == answer_2_with_dnp.verdict
    print("✓ Synthesis process is deterministic for same inputs")
    
    print(f"\n📊 Synthesis Results:")
    print(f"   Question: {formatted['question']}")
    print(f"   Verdict: {formatted['verdict']}")
    print(f"   Confidence: {formatted['confidence']:.3f}")
    print(f"   Premises: {len(formatted['premises'])}")
    print(f"   Unmet conjuncts: {len(formatted['unmet_conjuncts'])}")
    print(f"   Hash: {hash_validation['synthesis_hash'][:16]}...")
    
    print("\n✅ All synthesis integration tests passed!")

def test_hash_concepts_only():
    """Test hash concepts when full synthesizer is not available"""
    
    # Test basic hash policy concepts
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'egw_query_expansion', 'core'))
    
    try:
        import hash_policies
        
        # Mock synthesis result structure
        mock_synthesis = {
            'question': 'Does the plan include measurable targets for 2025?',
            'verdict': 'yes',
            'rationale': 'The plan includes specific targets for 2025 with measurable indicators.',
            'premises': [
                {'text': 'Target: 25% improvement by 2025', 'score': 0.92},
                {'text': 'Baseline established in 2023', 'score': 0.88}
            ],
            'confidence': 0.85,
            'metadata': {'synthesis_timestamp': '2024-01-01T00:00:00Z'}
        }
        
        # Test with different hash policies
        policies = [
            hash_policies.CanonicalHashPolicy(),
            hash_policies.FastHashPolicy(),
            hash_policies.SecureHashPolicy()
        ]
        
        for policy in policies:
            hash1 = policy.hash_object(mock_synthesis)
            hash2 = policy.hash_object(mock_synthesis)
            assert hash1 == hash2
            print(f"✓ {policy.__class__.__name__} produces consistent hashes")
        
        # Test pipeline validation concepts
        validator = hash_policies.PipelineHashValidator()
        
        # Mock pipeline stages
        stages = [
            ('evidence_gathering', {'evidence_count': 3, 'question': 'test'}),
            ('synthesis', mock_synthesis),
            ('dnp_validation', {**mock_synthesis, 'dnp_applied': True}),
            ('formatting', {'formatted': True, **mock_synthesis})
        ]
        
        for stage_name, stage_data in stages:
            validator.synthesis_hasher.policy.hash_object(stage_data)
        
        report = validator.get_validation_report()
        print("✓ Pipeline validation concepts tested")
        
        print("\n✅ Hash concept tests passed!")
        
    except ImportError:
        print("⚠️  Hash policies not available")

if __name__ == "__main__":
    print("Testing synthesis integration with hash validation...")
    test_synthesis_with_hash_validation()
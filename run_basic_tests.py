#!/usr/bin/env python3
"""
Basic test runner without pytest dependency
Tests core functionality of immutable context architecture
"""

import sys
import traceback

sys.path.insert(0, ".")

# # # from datetime import datetime, timezone  # Module not found  # Module not found  # Module not found

# # # from egw_query_expansion.core import (  # Module not found  # Module not found  # Module not found
    AliasDetectionError,
    ContextAdapter,
    DerivationDAG,
    DerivationEdge,
    ImmutableDict,
    LinearTypeEnforcer,
    LinearTypeError,
    QuestionContext,
    linear_context_scope,
)


def test_immutable_dict():
    """Test ImmutableDict functionality"""
    print("Testing ImmutableDict...")

    # Creation
    data = {"key1": "value1", "key2": 42}
    immutable_dict = ImmutableDict(data)
    assert immutable_dict["key1"] == "value1"
    assert immutable_dict["key2"] == 42
    assert len(immutable_dict) == 2

    # Derivation
    derived = immutable_dict.derive(key3="value3", key1="modified")
    assert immutable_dict["key1"] == "value1"  # Original unchanged
    assert derived["key1"] == "modified"  # Derived changed
    assert derived["key3"] == "value3"  # New key added

    # Removal
    removed = immutable_dict.remove("key2")
    assert "key2" in immutable_dict  # Original unchanged
# # #     assert "key2" not in removed  # Removed from derived  # Module not found  # Module not found  # Module not found
    assert len(removed) == 1

    print("  ✓ ImmutableDict tests passed")


def test_derivation_dag():
    """Test DerivationDAG functionality"""
    print("Testing DerivationDAG...")

    # Empty DAG
    dag = DerivationDAG()
    assert len(dag.nodes) == 0
    assert len(dag.edges) == 0

    # Add edges
    now = datetime.now(timezone.utc)
    edge1 = DerivationEdge("parent", "child1", "op1", now)
    edge2 = DerivationEdge("child1", "child2", "op2", now)

    new_dag = dag.add_edge(edge1).add_edge(edge2)
    assert len(dag.edges) == 0  # Original unchanged
    assert len(new_dag.edges) == 2

    # Lineage
    lineage = new_dag.get_lineage("child2")
    assert lineage == ["parent", "child1", "child2"]

    print("  ✓ DerivationDAG tests passed")


def test_question_context():
    """Test QuestionContext functionality"""
    print("Testing QuestionContext...")

    # Creation
    context = QuestionContext(
        "What is machine learning?", {"domain": "ai", "level": "beginner"}
    )
    assert context.question_text == "What is machine learning?"
    assert context.context_data["domain"] == "ai"
    assert context.verify_integrity()

    # Immutability
    try:
        context.question_text = "Modified"
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        assert "in-situ mutation" in str(e)

    # Derivation with context
    derived = context.derive_with_context(difficulty="easy", language="en")
    assert context.context_data["domain"] == "ai"  # Original unchanged
    assert derived.context_data["difficulty"] == "easy"  # Derived changed
    assert derived.context_data["domain"] == "ai"  # Inherited
    assert derived.metadata.parent_id == context.metadata.derivation_id

    # Derivation with question
    new_question = context.derive_with_question("What is deep learning?")
    assert context.question_text == "What is machine learning?"  # Original unchanged
    assert new_question.question_text == "What is deep learning?"  # Changed
    assert new_question.context_data["domain"] == "ai"  # Context preserved

    # Expansion derivation
    expansion_data = {"terms": ["ML", "artificial intelligence"], "method": "semantic"}
    expanded = context.derive_with_expansion(expansion_data)
    assert "expansion" not in context.context_data  # Original unchanged
    assert "expansion" in expanded.context_data  # Expansion added
    # ImmutableDict deep freezes lists to tuples - this is expected behavior
    stored_expansion = expanded.context_data["expansion"]
    assert stored_expansion["method"] == expansion_data["method"]
    assert list(stored_expansion["terms"]) == expansion_data["terms"]

    print("  ✓ QuestionContext tests passed")


def test_linear_type_enforcer():
    """Test LinearTypeEnforcer functionality"""
    print("Testing LinearTypeEnforcer...")

    context = QuestionContext("Test question")
    enforcer = LinearTypeEnforcer()

    # Create linear reference
    linear_ref = enforcer.create_linear_reference(context)
    assert not linear_ref.is_consumed()
    assert context.metadata.derivation_id in linear_ref.reference_id

    # Consume reference
    consumed_context = enforcer.consume_reference(linear_ref)
    assert consumed_context == context
    assert linear_ref.is_consumed()

    # Double consumption should fail
    try:
        enforcer.consume_reference(linear_ref)
        assert False, "Should have raised LinearTypeError"
    except LinearTypeError as e:
        assert "already consumed" in str(e)

    # Aliasing detection
    fresh_context = QuestionContext("Fresh question")
    ref1 = enforcer.create_linear_reference(fresh_context)

    try:
        ref2 = enforcer.create_linear_reference(fresh_context)
        assert False, "Should have raised AliasDetectionError"
    except AliasDetectionError as e:
        assert "Aliasing detected" in str(e)

    print("  ✓ LinearTypeEnforcer tests passed")


def test_context_adapter():
    """Test ContextAdapter functionality"""
    print("Testing ContextAdapter...")

    context = QuestionContext(
        "How to implement neural networks?",
        {"domain": "ml", "complexity": "intermediate"},
    )

    adapter = ContextAdapter()

    # Query expansion adaptation
    expansion_adapted = adapter.adapt_for_query_expansion(context)
    assert expansion_adapted["question_text"] == context.question_text
    assert expansion_adapted["context_data"]["domain"] == "ml"
    assert "metadata" in expansion_adapted

    # Retrieval adaptation
    retrieval_adapted = adapter.adapt_for_retrieval(context)
    assert retrieval_adapted["query_text"] == context.question_text
    assert retrieval_adapted["context"]["domain"] == "ml"

    # Expansion context adaptation
    expansion_data = {"terms": ["neural nets", "deep learning"]}
    expanded_context = context.derive_with_expansion(expansion_data)
    retrieval_with_expansion = adapter.adapt_for_retrieval(expanded_context)
    assert "expansion_data" in retrieval_with_expansion

    print("  ✓ ContextAdapter tests passed")


def test_linear_context_scope():
    """Test linear context scope management"""
    print("Testing linear context scope...")

    context = QuestionContext("Scope test")

    with linear_context_scope(context) as scoped_context:
        assert scoped_context == context
        # Context automatically consumed in scope

    print("  ✓ Linear context scope tests passed")


def test_integration_workflow():
    """Test complete integration workflow"""
    print("Testing integration workflow...")

    # Create root context
    root = QuestionContext(
        "How can AI help with climate change?",
        {"domain": "sustainability", "urgency": "high"},
    )

    # Chain of derivations
    step1 = root.derive_with_context(analysis_type="comprehensive")
    step2 = step1.derive_with_expansion(
        {
            "terms": ["machine learning", "carbon footprint", "green technology"],
            "method": "semantic_expansion",
        }
    )
    step3 = step2.derive_with_question(
        "What AI techniques are most effective for climate modeling?"
    )

    # Verify lineage
    lineage = step3.get_lineage()
    assert len(lineage) >= 3
    assert root.metadata.derivation_id in lineage
    assert step3.metadata.derivation_id in lineage

    # Verify integrity throughout chain
    for context in [root, step1, step2, step3]:
        assert context.verify_integrity()

    # Adapter integration
    adapter = ContextAdapter()
    final_adapted = adapter.adapt_for_retrieval(step3)
    assert final_adapted["query_text"] == step3.question_text
    assert "expansion_data" in final_adapted

    print("  ✓ Integration workflow tests passed")


def run_all_tests():
    """Run all tests"""
    print("🧪 Running Immutable Context Architecture Tests")
    print("=" * 60)

    tests = [
        test_immutable_dict,
        test_derivation_dag,
        test_question_context,
        test_linear_type_enforcer,
        test_context_adapter,
        test_linear_context_scope,
        test_integration_workflow,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ❌ {test.__name__} failed: {e}")
            traceback.print_exc()
            failed += 1

    print(f"\n📊 Test Results:")
    print(f"  ✓ Passed: {passed}")
    print(f"  ❌ Failed: {failed}")
    print(f"  📈 Success Rate: {passed/(passed+failed)*100:.1f}%")

    if failed == 0:
        print(
            "\n🎉 All tests passed! Immutable context architecture is working correctly."
        )
        return True
    else:
        print(f"\n⚠️  {failed} tests failed. Please review the implementation.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

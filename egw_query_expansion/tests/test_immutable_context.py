"""
Tests for Immutable Context Architecture

Validates linear type theory compliance and immutability guarantees
"""

import hashlib
import hmac
from datetime import datetime, timezone
from typing import Any, Dict

import pytest

from egw_query_expansion.core.immutable_context import (
    DerivationDAG,
    DerivationEdge,
    ImmutableContextManager,
    ImmutableDict,
    QuestionContext,
    assert_linear_reference,
    create_expanded_context,
    create_question_context,
    is_valid_context,
)
from egw_query_expansion.core.linear_type_enforcer import (
    AliasDetectionError,
    LinearReference,
    LinearTypeEnforcer,
    LinearTypeError,
    assert_no_aliasing,
    linear_context_scope,
    linear_operation,
)


class TestImmutableDict:
    """Test immutable dictionary implementation"""

    def test_immutable_dict_creation(self):
        data = {"key1": "value1", "key2": 42}
        immutable_dict = ImmutableDict(data)

        assert immutable_dict["key1"] == "value1"
        assert immutable_dict["key2"] == 42
        assert len(immutable_dict) == 2

    def test_immutable_dict_derivation(self):
        original = ImmutableDict({"a": 1, "b": 2})
        derived = original.derive(c=3, a=10)

        # Original unchanged
        assert original["a"] == 1
        assert "c" not in original

        # Derived has changes
        assert derived["a"] == 10
        assert derived["c"] == 3
        assert derived["b"] == 2

    def test_immutable_dict_removal(self):
        original = ImmutableDict({"a": 1, "b": 2, "c": 3})
        derived = original.remove("b")

        # Original unchanged
        assert "b" in original
        assert len(original) == 3

        # Derived has removal
        assert "b" not in derived
        assert len(derived) == 2
        assert derived["a"] == 1
        assert derived["c"] == 3

    def test_immutable_dict_hashing(self):
        dict1 = ImmutableDict({"a": 1, "b": 2})
        dict2 = ImmutableDict({"b": 2, "a": 1})  # Different order
        dict3 = ImmutableDict({"a": 1, "b": 3})  # Different value

        assert hash(dict1) == hash(dict2)  # Order invariant
        assert hash(dict1) != hash(dict3)  # Value sensitive


class TestDerivationDAG:
    """Test directed acyclic graph for derivation tracking"""

    def test_dag_creation_empty(self):
        dag = DerivationDAG()
        assert len(dag.nodes) == 0
        assert len(dag.edges) == 0

    def test_dag_add_edge(self):
        dag = DerivationDAG()
        edge = DerivationEdge(
            parent_id="parent-1",
            child_id="child-1",
            operation="test_operation",
            timestamp=datetime.now(timezone.utc),
        )

        new_dag = dag.add_edge(edge)

        # Original DAG unchanged
        assert len(dag.edges) == 0

        # New DAG has edge
        assert len(new_dag.edges) == 1
        assert edge in new_dag.edges
        assert len(new_dag.nodes) == 2

    def test_dag_cycle_detection(self):
        dag = DerivationDAG()
        now = datetime.now(timezone.utc)

        edge1 = DerivationEdge("a", "b", "op1", now)
        edge2 = DerivationEdge("b", "c", "op2", now)
        edge3 = DerivationEdge("c", "a", "op3", now)  # Creates cycle

        dag = dag.add_edge(edge1).add_edge(edge2)

        with pytest.raises(ValueError, match="Cycle detected"):
            dag.add_edge(edge3)

    def test_dag_lineage_tracking(self):
        dag = DerivationDAG()
        now = datetime.now(timezone.utc)

        # Create linear chain: root -> child1 -> child2
        edge1 = DerivationEdge("root", "child1", "op1", now)
        edge2 = DerivationEdge("child1", "child2", "op2", now)

        dag = dag.add_edge(edge1).add_edge(edge2)

        lineage = dag.get_lineage("child2")
        assert lineage == ["root", "child1", "child2"]


class TestQuestionContext:
    """Test immutable question context implementation"""

    def test_context_creation(self):
        context = QuestionContext(
            question_text="What is the capital of France?",
            context_data={"domain": "geography", "difficulty": "easy"},
        )

        assert context.question_text == "What is the capital of France?"
        assert context.context_data["domain"] == "geography"
        assert context.context_data["difficulty"] == "easy"
        assert context.verify_integrity()

    def test_context_immutability(self):
        context = QuestionContext("Test question")

        # Attempt to modify should fail
        with pytest.raises(RuntimeError, match="in-situ mutation"):
            context.question_text = "Modified question"

        with pytest.raises(RuntimeError, match="in-situ mutation"):
            context._question_text = "Modified question"

    def test_context_derivation_with_context(self):
        original = QuestionContext(
            "Original question", {"type": "factual", "score": 0.8}
        )

        derived = original.derive_with_context(type="analytical", complexity="high")

        # Original unchanged
        assert original.context_data["type"] == "factual"
        assert "complexity" not in original.context_data

        # Derived has changes
        assert derived.context_data["type"] == "analytical"
        assert derived.context_data["complexity"] == "high"
        assert derived.context_data["score"] == 0.8  # Inherited

        # Different derivation IDs
        assert original.metadata.derivation_id != derived.metadata.derivation_id

        # Proper parent-child relationship
        assert derived.metadata.parent_id == original.metadata.derivation_id

    def test_context_derivation_with_question(self):
        original = QuestionContext("Original question", {"context": "test"})
        derived = original.derive_with_question("New question")

        # Question changed
        assert original.question_text == "Original question"
        assert derived.question_text == "New question"

        # Context preserved
        assert derived.context_data["context"] == "test"

        # Lineage preserved
        assert derived.metadata.parent_id == original.metadata.derivation_id

    def test_context_integrity_verification(self):
        context = QuestionContext("Test question", {"key": "value"})
        assert context.verify_integrity()

        # Manually corrupt integrity (simulating tampering)
        corrupted_context = QuestionContext("Test question", {"key": "value"})
        # Simulate corruption by changing internal HMAC
        object.__setattr__(
            corrupted_context._metadata, "integrity_hmac", "corrupted_hmac"
        )

        assert not corrupted_context.verify_integrity()

    def test_context_lineage_tracking(self):
        root = QuestionContext("Root question")
        child1 = root.derive_with_context(step=1)
        child2 = child1.derive_with_context(step=2)
        child3 = child2.derive_with_question("Modified question")

        lineage = child3.get_lineage()

        # Should have all 4 contexts in lineage
        assert len(lineage) >= 3  # At least root + 2 derivations
        assert root.metadata.derivation_id in lineage
        assert child1.metadata.derivation_id in lineage
        assert child2.metadata.derivation_id in lineage
        assert child3.metadata.derivation_id in lineage

    def test_context_hashing_and_equality(self):
        context1 = QuestionContext("Same question", {"same": "data"})
        context2 = QuestionContext("Same question", {"same": "data"})
        context3 = QuestionContext("Different question", {"same": "data"})

        # Different contexts with same content should have different hashes
        # because they have different derivation IDs
        assert context1 != context2
        assert hash(context1) != hash(context2)

        # Obviously different contexts
        assert context1 != context3

    def test_context_expansion_derivation(self):
        base_context = QuestionContext(
            "What is machine learning?", {"domain": "ai", "level": "beginner"}
        )

        expansion_data = {
            "synonyms": ["ML", "artificial intelligence", "deep learning"],
            "related_terms": ["neural networks", "supervised learning"],
            "expansion_method": "semantic_similarity",
        }

        expanded_context = base_context.derive_with_expansion(expansion_data)

        # Original preserved
        assert base_context.question_text == "What is machine learning?"
        assert "expansion" not in base_context.context_data

        # Expansion added
        assert expanded_context.question_text == "What is machine learning?"
        assert "expansion" in expanded_context.context_data
        assert expanded_context.context_data["expansion"] == expansion_data
        assert "expansion_timestamp" in expanded_context.context_data

        # Lineage preserved
        assert (
            expanded_context.metadata.parent_id == base_context.metadata.derivation_id
        )
        assert expanded_context.metadata.operation_type == "query_expansion"


class TestLinearTypeEnforcer:
    """Test linear type system enforcement"""

    def test_linear_reference_creation(self):
        context = QuestionContext("Test question")
        enforcer = LinearTypeEnforcer()

        linear_ref = enforcer.create_linear_reference(context)

        assert isinstance(linear_ref, LinearReference)
        assert not linear_ref.is_consumed()
        assert context.metadata.derivation_id in linear_ref.reference_id

    def test_linear_reference_consumption(self):
        context = QuestionContext("Test question")
        enforcer = LinearTypeEnforcer()

        linear_ref = enforcer.create_linear_reference(context)
        consumed_context = enforcer.consume_reference(linear_ref)

        assert consumed_context == context
        assert linear_ref.is_consumed()

        # Second consumption should fail
        with pytest.raises(LinearTypeError, match="already consumed"):
            enforcer.consume_reference(linear_ref)

    def test_aliasing_detection(self):
        context = QuestionContext("Test question")
        enforcer = LinearTypeEnforcer()

        # Create first reference
        ref1 = enforcer.create_linear_reference(context)

        # Attempt to create second reference should fail
        with pytest.raises(AliasDetectionError, match="Aliasing detected"):
            enforcer.create_linear_reference(context)

    def test_aliasing_check_after_consumption(self):
        context = QuestionContext("Test question")
        enforcer = LinearTypeEnforcer()

        # Create and consume reference
        ref1 = enforcer.create_linear_reference(context)
        enforcer.consume_reference(ref1)

        # Now should be able to create new reference
        ref2 = enforcer.create_linear_reference(context)
        assert not ref2.is_consumed()

    def test_linear_context_scope(self):
        context = QuestionContext("Test question")

        with linear_context_scope(context) as scoped_context:
            assert scoped_context == context
            # Context is automatically consumed in scope

    def test_linear_operation_decorator(self):
        @linear_operation
        def process_context(ctx: QuestionContext) -> str:
            return f"Processed: {ctx.question_text}"

        context = QuestionContext("Test question")
        result = process_context(context)

        assert result == "Processed: Test question"

    def test_assert_no_aliasing_success(self):
        context = QuestionContext("Test question")

        # Should not raise for unaliased context
        assert_no_aliasing(context)

    def test_assert_no_aliasing_failure(self):
        context = QuestionContext("Test question")
        enforcer = LinearTypeEnforcer()

        # Create reference to establish tracking
        ref = enforcer.create_linear_reference(context)

        # Now aliasing check should fail for new operations
        with pytest.raises(AliasDetectionError):
            assert_no_aliasing(context)


class TestContextManagerIntegration:
    """Test context manager integration"""

    def test_immutable_context_manager(self):
        with ImmutableContextManager("Test question", {"test": "data"}) as context:
            assert isinstance(context, QuestionContext)
            assert context.question_text == "Test question"
            assert context.context_data["test"] == "data"
            assert context.verify_integrity()

    def test_factory_functions(self):
        # Test create_question_context
        context = create_question_context("Test", {"key": "value"})
        assert isinstance(context, QuestionContext)
        assert context.question_text == "Test"

        # Test create_expanded_context
        expansion = {"terms": ["test", "example"]}
        expanded = create_expanded_context(context, expansion)
        assert "expansion" in expanded.context_data
        assert expanded.context_data["expansion"] == expansion

    def test_type_guards(self):
        valid_context = QuestionContext("Valid question")
        assert is_valid_context(valid_context)

        # Test with non-context
        assert not is_valid_context("not a context")
        assert not is_valid_context(None)

        # Test assert_linear_reference
        assert_linear_reference(valid_context)  # Should not raise

        with pytest.raises(TypeError):
            assert_linear_reference("not a context")


class TestSystemIntegration:
    """Test system-level integration and compliance"""

    def test_full_derivation_chain_immutability(self):
        # Create chain of derivations
        root = QuestionContext("Root question", {"step": 0})
        step1 = root.derive_with_context(step=1, process="expansion")
        step2 = step1.derive_with_question("Modified question")
        step3 = step2.derive_with_expansion({"terms": ["expanded", "terms"]})

        # Verify immutability at each step
        assert root.context_data["step"] == 0
        assert step1.context_data["step"] == 1
        assert step1.question_text == "Root question"
        assert step2.question_text == "Modified question"
        assert step2.context_data["step"] == 1
        assert "expansion" in step3.context_data

        # Verify integrity of all contexts
        for context in [root, step1, step2, step3]:
            assert context.verify_integrity()

        # Verify lineage
        lineage = step3.get_lineage()
        assert root.metadata.derivation_id in lineage
        assert step3.metadata.derivation_id in lineage

    def test_concurrent_context_operations(self):
        """Test thread safety of context operations"""
        import threading

        context = QuestionContext("Concurrent test")
        results = []

        def derive_context(index: int):
            derived = context.derive_with_context(thread_id=index)
            results.append(derived)

        threads = []
        for i in range(10):
            thread = threading.Thread(target=derive_context, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # All derivations should succeed
        assert len(results) == 10

        # All should have unique derivation IDs
        derivation_ids = {r.metadata.derivation_id for r in results}
        assert len(derivation_ids) == 10

        # All should maintain integrity
        for result in results:
            assert result.verify_integrity()

    def test_memory_cleanup_after_consumption(self):
        """Test that consumed contexts don't create memory leaks"""
        enforcer = LinearTypeEnforcer()
        initial_count = enforcer.get_active_contexts_count()

        # Create and consume multiple contexts
        for i in range(100):
            context = QuestionContext(f"Test {i}")
            with linear_context_scope(context):
                # Context automatically consumed in scope
                pass

        # Cleanup should have occurred
        enforcer.cleanup_consumed_references()
        final_count = enforcer.get_active_contexts_count()

        # Should not have grown significantly
        assert final_count <= initial_count + 5  # Allow some tolerance

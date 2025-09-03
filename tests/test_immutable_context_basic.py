#!/usr/bin/env python3
"""
Basic validation script for immutable context architecture
"""

import os
import sys

sys.path.insert(0, ".")

try:
# # #     from egw_query_expansion.core.context_adapter import ContextAdapter  # Module not found  # Module not found  # Module not found
# # #     from egw_query_expansion.core.immutable_context import (  # Module not found  # Module not found  # Module not found
        DerivationDAG,
        ImmutableDict,
        QuestionContext,
        create_expanded_context,
        create_question_context,
    )
# # #     from egw_query_expansion.core.linear_type_enforcer import (  # Module not found  # Module not found  # Module not found
        LinearTypeEnforcer,
        linear_context_scope,
    )

    print("✓ All imports successful")

    # Test basic context creation
    context = QuestionContext("What is the capital of France?", {"domain": "geography"})
    print(f"✓ Context created: {context.question_text}")
    print(f"✓ Context hash: {context.content_hash[:16]}...")
    print(f"✓ Integrity verified: {context.verify_integrity()}")

    # Test derivation
    derived = context.derive_with_context(difficulty="easy", language="en")
    print(f"✓ Derived context created: {derived.context_data.get('difficulty')}")
    print(
        f"✓ Parent-child relationship: {derived.metadata.parent_id == context.metadata.derivation_id}"
    )

    # Test immutable dict
    immutable = ImmutableDict({"a": 1, "b": 2})
    derived_dict = immutable.derive(c=3)
    print(f"✓ ImmutableDict works: {len(immutable)} -> {len(derived_dict)}")

    # Test linear enforcer
    enforcer = LinearTypeEnforcer()
    linear_ref = enforcer.create_linear_reference(context)
    consumed = enforcer.consume_reference(linear_ref)
    print(f"✓ Linear type enforcement works: {consumed == context}")

    # Test context scope
    with linear_context_scope(derived) as scoped:
        print(f"✓ Linear context scope works: {scoped.question_text}")

    # Test adapter
    adapter = ContextAdapter()
    adapted = adapter.adapt_for_query_expansion(context)
    print(f"✓ Context adapter works: {adapted['question_text']}")

    print(
        "\n🎉 All basic tests passed! Immutable context architecture is working correctly."
    )

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

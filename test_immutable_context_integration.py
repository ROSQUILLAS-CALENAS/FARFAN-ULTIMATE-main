#!/usr/bin/env python3
"""
Test integration of deterministic hashing in immutable_context.py
"""

import sys
sys.path.insert(0, 'egw_query_expansion')

def test_immutable_context_integration():
    """Test that immutable_context.py uses deterministic hashing."""
    
    import importlib.util
    
    # Load immutable_context module directly to avoid import issues
    spec = importlib.util.spec_from_file_location('immutable_context', 'egw_query_expansion/core/immutable_context.py')
    ic_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ic_module)
    
    # Create identical contexts and verify they produce identical hashes
    context1 = ic_module.QuestionContext('What is AI?', {'data': {'key': 'value'}})
    context2 = ic_module.QuestionContext('What is AI?', {'data': {'key': 'value'}})
    
    print('Testing immutable_context integration:')
    print(f'  Content hashes match: {context1.content_hash == context2.content_hash}')
    print(f'  Hash length correct: {len(context1.content_hash) == 64}')
    print(f'  Hash 1: {context1.content_hash[:16]}...')
    print(f'  Hash 2: {context2.content_hash[:16]}...')
    
    # Test with different key orders
    context3 = ic_module.QuestionContext('What is AI?', {'data': {'key': 'value'}, 'extra': 'info'})
    context4 = ic_module.QuestionContext('What is AI?', {'extra': 'info', 'data': {'key': 'value'}})
    
    print(f'  Different key order hashes match: {context3.content_hash == context4.content_hash}')
    
    if (context1.content_hash == context2.content_hash and 
        len(context1.content_hash) == 64 and
        context3.content_hash == context4.content_hash):
        print('✓ immutable_context successfully using deterministic hashing')
        return True
    else:
        print('❌ immutable_context deterministic hashing failed')
        return False

if __name__ == "__main__":
    success = test_immutable_context_integration()
    sys.exit(0 if success else 1)
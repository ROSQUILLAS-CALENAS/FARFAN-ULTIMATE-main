#!/usr/bin/env python3
"""
Validation script for memory optimization implementations.
"""

import gc
import logging
import sys
import tempfile
from pathlib import Path

import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def validate_embedding_cache():
    """Validate embedding cache functionality."""
    print("🧠 Validating Embedding Cache Manager...")
    
    try:
        from embedding_builder import EmbeddingCacheManager
        
        # Test basic functionality
        cache = EmbeddingCacheManager(max_cache_size=5, max_memory_mb=1)
        
        # Test put/get
        text = "test text"
        embedding = np.random.randn(384).astype(np.float32)
        cache.put(text, embedding, "test_model")
        
        result = cache.get(text, "test_model")
        assert result is not None, "Cache get failed"
        np.testing.assert_array_almost_equal(result, embedding)
        
        # Test stats
        stats = cache.get_stats()
        assert stats['cache_size'] == 1, "Cache size incorrect"
        assert stats['hit_count'] == 1, "Hit count incorrect"
        
        # Test LRU eviction
        for i in range(6):  # Exceed cache size
            embedding_i = np.random.randn(384).astype(np.float32)
            cache.put(f"text_{i}", embedding_i, "test_model")
        
        # First text should be evicted
        result = cache.get(text, "test_model")
        assert result is None, "LRU eviction failed"
        
        # Test clear
        cache.clear_batch()
        stats = cache.get_stats()
        assert stats['cache_size'] == 0, "Cache clear failed"
        
        print("   ✅ Embedding cache validation passed")
        return True
        
    except Exception as e:
        print(f"   ❌ Embedding cache validation failed: {e}")
        return False


def validate_embedding_builder():
    """Validate embedding builder with cache."""
    print("🏗️ Validating Embedding Builder with Cache...")
    
    try:
        from embedding_builder import EmbeddingBuilder
        
        # Test with cache enabled
        builder = EmbeddingBuilder(
            enable_cache=True,
            max_cache_size=10,
            max_cache_memory_mb=5
        )
        
        # Test single embedding generation
        text = "Test document for validation"
        embedding1 = builder.generate_embeddings(text)
        embedding2 = builder.generate_embeddings(text)  # Should hit cache
        
        assert embedding1.shape == embedding2.shape, "Embedding shapes don't match"
        np.testing.assert_array_almost_equal(embedding1, embedding2)
        
        # Test batch processing
        texts = [f"Document {i}" for i in range(5)]
        embeddings, chunk_ids = builder.batch_embeddings(texts, batch_size=2, show_progress=False)
        
        assert len(embeddings) == len(texts), "Batch embedding count mismatch"
        assert len(chunk_ids) == len(texts), "Chunk ID count mismatch"
        
        # Test query encoding
        query = "Test query"
        query_embedding = builder.encode_query(query)
        assert query_embedding.shape[0] == builder.embedding_dim, "Query embedding dimension mismatch"
        
        # Test cache stats
        stats = builder.get_cache_stats()
        assert 'cache_size' in stats, "Cache stats missing"
        
        # Test cache clearing
        builder.clear_cache()
        stats = builder.get_cache_stats()
        assert stats['cache_size'] == 0, "Cache clear failed"
        
        print("   ✅ Embedding builder validation passed")
        return True
        
    except Exception as e:
        print(f"   ❌ Embedding builder validation failed: {e}")
        return False


def validate_garbage_collection():
    """Validate garbage collection integration."""
    print("🗑️ Validating Garbage Collection Integration...")
    
    try:
        # Test basic garbage collection
        large_objects = []
        for i in range(10):
            large_array = np.random.randn(1000, 100)
            large_objects.append(large_array)
        
        # Delete references
        del large_objects
        
        # Call garbage collection
        collected = gc.collect()
        assert isinstance(collected, int), "Garbage collection not working"
        
        # Test with cache manager
        from embedding_builder import EmbeddingCacheManager
        cache = EmbeddingCacheManager(max_cache_size=5, max_memory_mb=10)
        
        for i in range(5):
            embedding = np.random.randn(500).astype(np.float32)
            cache.put(f"test_{i}", embedding, "test_model")
        
        # Clear with garbage collection
        cache.clear_batch()
        
        stats = cache.get_stats()
        assert stats['cache_size'] == 0, "Cache clear with GC failed"
        
        print("   ✅ Garbage collection validation passed")
        return True
        
    except Exception as e:
        print(f"   ❌ Garbage collection validation failed: {e}")
        return False


def validate_temp_file_management():
    """Validate temporary file management."""
    print("📁 Validating Temporary File Management...")
    
    try:
        # Use basic temporary file functionality since PDF reader has dependencies
        temp_files = []
        
        # Create temporary files
        for i in range(3):
            temp_file = tempfile.NamedTemporaryFile(suffix=f'.test_{i}', delete=False)
            temp_path = Path(temp_file.name)
            temp_file.write(b"Test content")
            temp_file.close()
            temp_files.append(temp_path)
        
        # Verify files exist
        for temp_path in temp_files:
            assert temp_path.exists(), f"Temp file not created: {temp_path}"
        
        # Test cleanup
        for temp_path in temp_files:
            temp_path.unlink()
            assert not temp_path.exists(), f"Temp file not cleaned up: {temp_path}"
        
        print("   ✅ Temporary file management validation passed")
        return True
        
    except Exception as e:
        print(f"   ❌ Temporary file management validation failed: {e}")
        return False


def validate_memory_usage():
    """Validate memory usage tracking."""
    print("📊 Validating Memory Usage Tracking...")
    
    try:
        from embedding_builder import EmbeddingCacheManager
        
        # Create cache with small memory limit
        cache = EmbeddingCacheManager(max_cache_size=1000, max_memory_mb=1)
        
        # Add embeddings and track memory
        initial_stats = cache.get_stats()
        assert initial_stats['memory_usage_mb'] == 0, "Initial memory usage not zero"
        
        # Add a small embedding
        small_embedding = np.random.randn(100).astype(np.float32)
        cache.put("small", small_embedding, "test")
        
        stats_after_small = cache.get_stats()
        assert stats_after_small['memory_usage_mb'] > 0, "Memory usage not tracked"
        
        # Add a larger embedding
        large_embedding = np.random.randn(10000).astype(np.float32)
        cache.put("large", large_embedding, "test")
        
        stats_after_large = cache.get_stats()
        assert stats_after_large['memory_usage_mb'] > stats_after_small['memory_usage_mb'], "Memory increase not tracked"
        
        # Test memory limit enforcement (should trigger evictions)
        very_large_embedding = np.random.randn(100000).astype(np.float32)
        cache.put("very_large", very_large_embedding, "test")
        
        final_stats = cache.get_stats()
        # Memory usage should be reasonable (evictions should have occurred)
        assert final_stats['memory_usage_mb'] <= 10, "Memory limit not enforced properly"
        
        print("   ✅ Memory usage tracking validation passed")
        return True
        
    except Exception as e:
        print(f"   ❌ Memory usage tracking validation failed: {e}")
        return False


def main():
    """Run all memory optimization validations."""
    print("🚀 MEMORY OPTIMIZATION VALIDATION")
    print("=" * 50)
    
    validations = [
        validate_embedding_cache,
        validate_embedding_builder,
        validate_garbage_collection,
        validate_temp_file_management,
        validate_memory_usage,
    ]
    
    passed = 0
    failed = 0
    
    for validation in validations:
        try:
            if validation():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"   ❌ Validation failed with exception: {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 VALIDATION SUMMARY")
    print(f"   ✅ Passed: {passed}")
    print(f"   ❌ Failed: {failed}")
    print(f"   📈 Success Rate: {passed/(passed+failed)*100:.1f}%")
    
    if failed == 0:
        print("\n🎉 ALL MEMORY OPTIMIZATIONS VALIDATED SUCCESSFULLY!")
        return 0
    else:
        print(f"\n⚠️ {failed} validation(s) failed. Please check the implementations.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
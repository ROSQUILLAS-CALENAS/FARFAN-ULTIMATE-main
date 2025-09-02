#!/usr/bin/env python3
"""
Example integration of DataIntegrityChecker with existing pipeline components

This demonstrates how to add integrity checking to existing pipeline components
with minimal code changes.
"""

from data_integrity_checker import (
    DataIntegrityChecker,
    integrity_validation_hook,
    add_artifact_generation_hook,
    validate_and_retry_on_corruption
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Example 1: Existing pipeline component (before integration)
class OriginalQuestionAnalyzer:
    """Original component without integrity checking"""
    
    def __init__(self):
        self.stage_name = "A_analysis_nlp"
    
    def process(self, data=None, context=None):
        """Original process method"""
        logger.info("Processing questions...")
        
        # Simulate analysis work
        result = {
            "questions_analyzed": 10,
            "categories": ["DE-1", "DE-2", "DE-3"],
            "confidence_scores": [0.85, 0.92, 0.78],
            "processing_metadata": {
                "model_version": "v1.0",
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }
        
        return result


# Example 2: Enhanced component using decorator approach
class EnhancedQuestionAnalyzer:
    """Enhanced component with integrity validation"""
    
    def __init__(self):
        self.stage_name = "A_analysis_nlp"
        self._integrity_checker = DataIntegrityChecker()
    
    @integrity_validation_hook(DataIntegrityChecker())
    def process(self, data=None, context=None):
        """Process with automatic integrity validation and retry"""
        logger.info("Processing questions with integrity validation...")
        
        # Same processing logic as original
        result = {
            "questions_analyzed": 10,
            "categories": ["DE-1", "DE-2", "DE-3"],
            "confidence_scores": [0.85, 0.92, 0.78],
            "processing_metadata": {
                "model_version": "v1.0",
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }
        
        return result


# Example 3: Using class decorator for automatic artifact saving
@add_artifact_generation_hook
class AutoSavingMesoAggregator:
    """Component with automatic artifact generation"""
    
    def __init__(self):
        self.stage_name = "G_aggregation_reporting"
    
    def process(self, data=None, context=None):
        """Process method - artifacts automatically saved"""
        logger.info("Processing meso aggregation...")
        
        result = {
            "aggregation_summary": {
                "total_documents": 25,
                "categories_processed": 4,
                "completion_rate": 0.92
            },
            "meso_synthesis": {
                "key_themes": ["governance", "transparency", "accountability"],
                "gap_analysis": {"critical_gaps": 3, "minor_gaps": 7}
            },
            "metadata": {
                "aggregation_timestamp": "2024-01-15T11:00:00Z",
                "processor_version": "v2.1"
            }
        }
        
        return result


# Example 4: Using function decorator for standalone functions
@validate_and_retry_on_corruption("R_search_retrieval", "HybridRetriever", "search_query_001")
def hybrid_search_function(query, context=None):
    """Standalone search function with integrity validation"""
    logger.info(f"Executing hybrid search for: {query}")
    
    # Simulate search results
    return {
        "query": query,
        "results": [
            {"doc_id": "doc_123", "score": 0.89, "snippet": "Relevant content..."},
            {"doc_id": "doc_456", "score": 0.76, "snippet": "More content..."},
            {"doc_id": "doc_789", "score": 0.65, "snippet": "Additional content..."}
        ],
        "search_metadata": {
            "total_documents": 10000,
            "search_time_ms": 45,
            "algorithm": "hybrid_egw_splade"
        }
    }


# Example 5: Manual integration for complex scenarios
class ManualIntegrationComponent:
    """Component with manual integrity checker integration"""
    
    def __init__(self):
        self.stage_name = "S_synthesis_output"
        self.integrity_checker = DataIntegrityChecker()
    
    def process(self, data=None, context=None):
        """Process with manual integrity handling"""
        logger.info("Processing synthesis with manual integrity checks...")
        
        # Extract document stem from context
        document_stem = context.get('document_stem', 'unknown') if context else 'unknown'
        
        # Validate input artifacts at stage boundary
        validation_report = self.integrity_checker.validate_stage_boundary(
            self.stage_name, document_stem
        )
        
        if validation_report['corruption_detected'] > 0:
            logger.warning(f"Input corruption detected: {validation_report['validation_errors']}")
            # Could implement custom recovery logic here
        
        # Process data
        result = {
            "synthesis_result": {
                "final_analysis": "Comprehensive analysis completed",
                "confidence_level": 0.94,
                "recommendations": ["Improve DE-1 coverage", "Enhance DE-3 analysis"]
            },
            "output_metadata": {
                "synthesis_timestamp": "2024-01-15T12:00:00Z",
                "quality_score": 0.91
            },
            "validation_report": validation_report
        }
        
        # Save result with integrity
        output_path, metadata = self.integrity_checker.save_artifact_with_integrity(
            result, self.stage_name, self.__class__.__name__, document_stem
        )
        
        logger.info(f"Synthesis artifact saved: {output_path}")
        
        # Add artifact info to result
        result['artifact_metadata'] = {
            'path': str(output_path),
            'hash': metadata.sha256_hash,
            'size': metadata.file_size
        }
        
        return result


def demonstrate_integration_approaches():
    """Demonstrate different integration approaches"""
    print("\n" + "="*60)
    print("Data Integrity Checker Integration Examples")
    print("="*60)
    
    # Test context
    test_context = {"document_stem": "integration_demo"}
    
    print("\n1. Original Component (no integrity checking)")
    original = OriginalQuestionAnalyzer()
    result1 = original.process(context=test_context)
    print(f"   ✅ Processed {result1['questions_analyzed']} questions")
    
    print("\n2. Enhanced Component (decorator-based validation)")
    try:
        enhanced = EnhancedQuestionAnalyzer()
        result2 = enhanced.process(context=test_context)
        print(f"   ✅ Processed {result2['questions_analyzed']} questions with integrity validation")
    except Exception as e:
        print(f"   ⚠️  Enhanced component failed: {e}")
    
    print("\n3. Auto-saving Component (class decorator)")
    auto_saving = AutoSavingMesoAggregator()
    result3 = auto_saving.process(context=test_context)
    print(f"   ✅ Processed {result3['aggregation_summary']['total_documents']} documents with auto-save")
    
    print("\n4. Function Decorator Approach")
    try:
        result4 = hybrid_search_function("governance transparency")
        print(f"   ✅ Found {len(result4['results'])} search results with integrity validation")
    except Exception as e:
        print(f"   ⚠️  Function decorator failed: {e}")
    
    print("\n5. Manual Integration")
    manual = ManualIntegrationComponent()
    result5 = manual.process(context=test_context)
    print(f"   ✅ Synthesis completed with confidence {result5['synthesis_result']['confidence_level']}")
    print(f"   📁 Artifact saved: {result5['artifact_metadata']['path'].split('/')[-1]}")
    
    print("\n" + "="*60)
    print("Integration Summary:")
    print("✅ Decorator approach: Minimal code changes, automatic retry")
    print("✅ Class decorator: Automatic artifact saving") 
    print("✅ Function decorator: Works with standalone functions")
    print("✅ Manual integration: Full control over integrity handling")
    print("="*60)


def demonstrate_corruption_recovery():
    """Demonstrate automatic corruption recovery"""
    print("\n" + "="*60)
    print("Corruption Recovery Demonstration")
    print("="*60)
    
    # Create a component that fails first, then succeeds
    class UnreliableComponent:
        def __init__(self):
            self.stage_name = "test_stage"
            self.call_count = 0
        
        def process(self, data=None, context=None):
            self.call_count += 1
            
            if self.call_count == 1:
                # Simulate transient failure
                raise RuntimeError("Transient network error")
            
            return {
                "success": True,
                "attempt": self.call_count,
                "message": "Recovered successfully"
            }
    
    # Test with integrity checker
    checker = DataIntegrityChecker()
    component = UnreliableComponent()
    
    print("Testing component with simulated transient failure...")
    
    try:
        result = checker.process_with_integrity_validation(
            component.process,
            "test_stage",
            "UnreliableComponent", 
            "recovery_demo"
        )
        
        print(f"✅ Recovery successful after {result['attempt']} attempts")
        print(f"   Message: {result['message']}")
        
        # Show integrity statistics
        integrity_report = checker.get_integrity_report()
        print(f"\n📊 Recovery Statistics:")
        print(f"   Successful recoveries: {integrity_report['statistics']['successful_recoveries']}")
        print(f"   Failed recoveries: {integrity_report['statistics']['failed_recoveries']}")
        
    except Exception as e:
        print(f"❌ Recovery failed: {e}")


def main():
    """Run integration examples"""
    print("🔧 Data Integrity Checker Integration Examples")
    
    try:
        demonstrate_integration_approaches()
        demonstrate_corruption_recovery()
        
        print("\n✅ All integration examples completed successfully!")
        
    except Exception as e:
        logger.error(f"Integration example failed: {e}")
        raise


if __name__ == "__main__":
    main()
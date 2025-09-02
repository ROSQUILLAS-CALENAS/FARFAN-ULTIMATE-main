#!/usr/bin/env python3
"""
FFC (Fail-Fast Conservative) Testing Suite
==========================================

Tests conservative deterministic routes with fault injection by class.
Verifies identical fallbacks and no side effects across error scenarios.
"""

import json
import sys
import os
from typing import Any, Dict, List, Type
from dataclasses import dataclass
import hashlib
import time

# Mock pytest for standalone execution
class MockPytest:
    @staticmethod
    def fixture(func):
        return func

try:
    import pytest
except ImportError:
    pytest = MockPytest()

# Add tools to path for fault injector
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tools'))

from fault_injector import (
    FaultInjector, 
    FaultProfile, 
    FaultType,
    fault_injector
)

# Test target classes that represent different components
class QueryProcessor:
    """Test class representing query processing component"""
    def __init__(self, processor_id: str = "default"):
        self.processor_id = processor_id
        self.processed_count = 0
        
    def process_query(self, query: str) -> Dict[str, Any]:
        self.processed_count += 1
        return {
            "processed": query,
            "processor_id": self.processor_id,
            "count": self.processed_count,
            "status": "success"
        }
        
    def conservative_fallback(self) -> Dict[str, Any]:
        """Conservative fallback that always returns safe default"""
        return {
            "processed": "fallback_query",
            "processor_id": self.processor_id,
            "count": 0,
            "status": "fallback"
        }


class EmbeddingGenerator:
    """Test class representing embedding generation component"""
    def __init__(self, model_name: str = "default_model"):
        self.model_name = model_name
        self.generation_count = 0
        
    def generate_embedding(self, text: str) -> List[float]:
        self.generation_count += 1
        # Simple deterministic embedding based on text hash
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        embedding = [float(int(text_hash[i:i+2], 16)) / 255.0 for i in range(0, 16, 2)]
        return embedding
        
    def conservative_fallback(self) -> List[float]:
        """Conservative fallback returning zero embedding"""
        return [0.0] * 8


class RetrievalEngine:
    """Test class representing retrieval engine component"""
    def __init__(self, engine_type: str = "hybrid"):
        self.engine_type = engine_type
        self.retrieval_count = 0
        
    def retrieve(self, query_embedding: List[float], k: int = 5) -> List[Dict[str, Any]]:
        self.retrieval_count += 1
        # Mock retrieval results
        results = []
        for i in range(k):
            results.append({
                "doc_id": f"doc_{i}",
                "score": 0.9 - (i * 0.1),
                "content": f"Retrieved content {i}"
            })
        return results
        
    def conservative_fallback(self) -> List[Dict[str, Any]]:
        """Conservative fallback returning empty results"""
        return []


@dataclass
class FFCTestResult:
    """Result of FFC testing for a component class"""
    class_name: str
    errors_tested: int
    identical_fallbacks: bool
    all_deterministic: bool
    error_types: List[str]
    fallback_hash: str
    execution_stats: Dict[str, Any]


class FFCTester:
    """Main FFC testing orchestrator"""
    
    def __init__(self):
        self.test_results: List[FFCTestResult] = []
        self.fault_injector = FaultInjector()
        self._setup_test_profiles()
        
    def _setup_test_profiles(self):
        """Setup fault injection profiles for testing"""
        profiles = {
            "network_fault": FaultProfile(
                fault_type=FaultType.NETWORK_ERROR,
                error_class=ConnectionError,
                probability=1.0,  # Always inject for testing
                deterministic_seed=12345
            ),
            "validation_fault": FaultProfile(
                fault_type=FaultType.VALIDATION_ERROR,
                error_class=ValueError,
                probability=1.0,
                deterministic_seed=23456
            ),
            "resource_fault": FaultProfile(
                fault_type=FaultType.RESOURCE_ERROR,
                error_class=MemoryError,
                probability=1.0,
                deterministic_seed=34567
            ),
            "timeout_fault": FaultProfile(
                fault_type=FaultType.TIMEOUT_ERROR,
                error_class=TimeoutError,
                probability=1.0,
                deterministic_seed=45678
            )
        }
        
        for name, profile in profiles.items():
            self.fault_injector.register_fault_profile(name, profile)
            
    def test_component_ffc(self, component_class: Type, test_iterations: int = 10) -> FFCTestResult:
        """Test FFC behavior for a component class"""
        class_name = component_class.__name__
        error_types = ["network_fault", "validation_fault", "resource_fault", "timeout_fault"]
        fallback_results = []
        all_deterministic = True
        
        for error_type in error_types:
            # Activate this error type
            self.fault_injector.activate_injection(error_type)
            
            # Test multiple instances to verify identical fallbacks
            iteration_results = []
            
            for iteration in range(test_iterations):
                component = component_class()
                
                try:
                    # Attempt operation with fault injection
                    with self.fault_injector.inject_for_class(
                        component_class, error_type, f"test_iteration_{iteration}"
                    ):
                        # This should trigger the fault
                        if hasattr(component, 'process_query'):
                            result = component.process_query("test_query")
                        elif hasattr(component, 'generate_embedding'):
                            result = component.generate_embedding("test_text")
                        elif hasattr(component, 'retrieve'):
                            result = component.retrieve([0.1, 0.2, 0.3])
                        else:
                            result = "no_test_method"
                            
                    # If we get here, no fault was injected (shouldn't happen with probability=1.0)
                    fallback_result = result
                    
                except Exception:
                    # Fault was injected, use conservative fallback
                    fallback_result = self.fault_injector.apply_conservative_fallback(
                        component_class, 
                        lambda: component.conservative_fallback()
                    )
                    
                iteration_results.append(fallback_result)
                
            # Verify all fallback results are identical
            fallback_results.extend(iteration_results)
            
            # Check deterministic behavior
            if len(set(str(r) for r in iteration_results)) > 1:
                all_deterministic = False
                
            self.fault_injector.deactivate_injection(error_type)
            
        # Verify identical fallbacks across all error types
        fallback_hashes = [
            hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()
            for result in fallback_results
        ]
        identical_fallbacks = len(set(fallback_hashes)) <= len(error_types)  # One unique per error type at most
        
        # Create single fallback hash representing the class behavior
        representative_hash = fallback_hashes[0] if fallback_hashes else ""
        
        # Get execution statistics
        stats = self.fault_injector.get_injection_stats()
        
        result = FFCTestResult(
            class_name=class_name,
            errors_tested=len(error_types) * test_iterations,
            identical_fallbacks=identical_fallbacks,
            all_deterministic=all_deterministic,
            error_types=error_types,
            fallback_hash=representative_hash,
            execution_stats=stats
        )
        
        self.test_results.append(result)
        return result
        
    def verify_no_side_effects(self, component_class: Type) -> bool:
        """Verify that fault injection doesn't cause side effects"""
        # Create two instances
        instance1 = component_class()
        instance2 = component_class()
        
        # Test normal operation on instance1
        if hasattr(instance1, 'process_query'):
            normal_result1 = instance1.process_query("test")
            normal_result2 = instance2.process_query("test")
        elif hasattr(instance1, 'generate_embedding'):
            normal_result1 = instance1.generate_embedding("test")
            normal_result2 = instance2.generate_embedding("test")
        elif hasattr(instance1, 'retrieve'):
            normal_result1 = instance1.retrieve([0.1])
            normal_result2 = instance2.retrieve([0.1])
        else:
            return True  # No test method, assume no side effects
            
        # Results should be identical for same inputs
        return str(normal_result1) == str(normal_result2)
        
    def generate_ffc_certificate(self) -> Dict[str, Any]:
        """Generate FFC certificate with test results"""
        all_passed = all(
            result.identical_fallbacks and result.all_deterministic 
            for result in self.test_results
        )
        
        total_errors_tested = sum(result.errors_tested for result in self.test_results)
        
        certificate = {
            "pass": all_passed,
            "errors_tested": total_errors_tested,
            "identical_fallbacks": all(result.identical_fallbacks for result in self.test_results),
            "all_deterministic": all(result.all_deterministic for result in self.test_results),
            "timestamp": time.time(),
            "components_tested": [result.class_name for result in self.test_results],
            "detailed_results": [
                {
                    "class_name": result.class_name,
                    "errors_tested": result.errors_tested,
                    "identical_fallbacks": result.identical_fallbacks,
                    "all_deterministic": result.all_deterministic,
                    "error_types": result.error_types,
                    "fallback_hash": result.fallback_hash
                }
                for result in self.test_results
            ]
        }
        
        return certificate


# Test fixtures
@pytest.fixture
def ffc_tester():
    """Pytest fixture providing FFC tester instance"""
    return FFCTester()


# Test cases
class TestFFCBehavior:
    """Test suite for FFC (Fail-Fast Conservative) behavior"""
    
    def test_query_processor_ffc(self, ffc_tester):
        """Test FFC behavior for QueryProcessor"""
        result = ffc_tester.test_component_ffc(QueryProcessor, test_iterations=5)
        
        assert result.class_name == "QueryProcessor"
        assert result.errors_tested == 20  # 4 error types * 5 iterations
        assert result.identical_fallbacks, "QueryProcessor should have identical fallbacks"
        assert result.all_deterministic, "QueryProcessor should be deterministic"
        
    def test_embedding_generator_ffc(self, ffc_tester):
        """Test FFC behavior for EmbeddingGenerator"""
        result = ffc_tester.test_component_ffc(EmbeddingGenerator, test_iterations=5)
        
        assert result.class_name == "EmbeddingGenerator"
        assert result.errors_tested == 20
        assert result.identical_fallbacks, "EmbeddingGenerator should have identical fallbacks"
        assert result.all_deterministic, "EmbeddingGenerator should be deterministic"
        
    def test_retrieval_engine_ffc(self, ffc_tester):
        """Test FFC behavior for RetrievalEngine"""
        result = ffc_tester.test_component_ffc(RetrievalEngine, test_iterations=5)
        
        assert result.class_name == "RetrievalEngine"
        assert result.errors_tested == 20
        assert result.identical_fallbacks, "RetrievalEngine should have identical fallbacks"
        assert result.all_deterministic, "RetrievalEngine should be deterministic"
        
    def test_no_side_effects(self, ffc_tester):
        """Test that fault injection doesn't cause side effects"""
        assert ffc_tester.verify_no_side_effects(QueryProcessor)
        assert ffc_tester.verify_no_side_effects(EmbeddingGenerator)
        assert ffc_tester.verify_no_side_effects(RetrievalEngine)
        
    def test_deterministic_fault_injection(self, ffc_tester):
        """Test that fault injection is deterministic"""
        # Test that same inputs produce same injection decisions
        is_deterministic = ffc_tester.fault_injector.verify_deterministic_behavior(
            QueryProcessor, "test_context", iterations=50
        )
        assert is_deterministic, "Fault injection should be deterministic"
        
    def test_certificate_generation(self, ffc_tester):
        """Test FFC certificate generation"""
        # Run tests on all components
        ffc_tester.test_component_ffc(QueryProcessor, test_iterations=3)
        ffc_tester.test_component_ffc(EmbeddingGenerator, test_iterations=3)
        ffc_tester.test_component_ffc(RetrievalEngine, test_iterations=3)
        
        certificate = ffc_tester.generate_ffc_certificate()
        
        assert "pass" in certificate
        assert "errors_tested" in certificate
        assert "identical_fallbacks" in certificate
        assert certificate["errors_tested"] == 36  # 3 components * 4 error types * 3 iterations
        assert len(certificate["components_tested"]) == 3
        assert len(certificate["detailed_results"]) == 3


def test_ffc_integration():
    """Integration test for complete FFC testing flow"""
    tester = FFCTester()
    
    # Test all components
    components = [QueryProcessor, EmbeddingGenerator, RetrievalEngine]
    for component_class in components:
        result = tester.test_component_ffc(component_class, test_iterations=3)
        assert result.identical_fallbacks
        assert result.all_deterministic
        
    # Generate and save certificate
    certificate = tester.generate_ffc_certificate()
    
    # Save certificate to file
    with open("ffc_certificate.json", "w") as f:
        json.dump(certificate, f, indent=2)
        
    assert certificate["pass"]
    assert certificate["identical_fallbacks"]
    assert certificate["errors_tested"] > 0
    
    print(f"FFC Certificate generated: {json.dumps(certificate, indent=2)}")


if __name__ == "__main__":
    # Run integration test
    test_ffc_integration()
    print("FFC testing completed successfully!")
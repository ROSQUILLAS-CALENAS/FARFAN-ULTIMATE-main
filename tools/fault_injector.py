#!/usr/bin/env python3

# Mandatory Pipeline Contract Annotations
__phase__ = "O"
__code__ = "111O"
__stage_order__ = 7

"""
Fault Injector for FFC (Fail-Fast Conservative) Testing
=====================================================

Simulates typed errors deterministically for testing conservative fallback paths.
Ensures identical fallbacks across error classes with no side effects.
"""

import hashlib
import json
# # # from enum import Enum  # Module not found  # Module not found  # Module not found
# # # from typing import Any, Callable, Dict, List, Optional, Type, Union  # Module not found  # Module not found  # Module not found
# # # from dataclasses import dataclass, field  # Module not found  # Module not found  # Module not found
# # # from contextlib import contextmanager  # Module not found  # Module not found  # Module not found
import threading
import time


class FaultType(Enum):
    """Classification of fault types for deterministic injection"""
    NETWORK_ERROR = "network_error"
    VALIDATION_ERROR = "validation_error"  
    RESOURCE_ERROR = "resource_error"
    TIMEOUT_ERROR = "timeout_error"
    DATA_ERROR = "data_error"


@dataclass(frozen=True)
class FaultProfile:
    """Immutable fault injection profile"""
    fault_type: FaultType
    error_class: Type[Exception]
    probability: float = 0.1
    deterministic_seed: Optional[int] = None
    fallback_value: Any = None
    
    def __post_init__(self):
        if not (0.0 <= self.probability <= 1.0):
            raise ValueError("Probability must be between 0.0 and 1.0")


@dataclass
class InjectionResult:
    """Result of fault injection attempt"""
    fault_triggered: bool
    fault_type: Optional[FaultType]
    error_class: Optional[Type[Exception]]
    fallback_applied: bool
    execution_time_ms: float
    deterministic_hash: str


class FaultInjector:
    """Main fault injector with deterministic error simulation"""
    
    def __init__(self):
        self._profiles: Dict[str, FaultProfile] = {}
        self._active_injections: Dict[str, bool] = {}
        self._injection_history: List[InjectionResult] = []
        self._lock = threading.Lock()
        
    def register_fault_profile(self, name: str, profile: FaultProfile) -> None:
        """Register a named fault profile"""
        with self._lock:
            self._profiles[name] = profile
            
    def activate_injection(self, profile_name: str) -> None:
        """Activate fault injection for a profile"""
        with self._lock:
            if profile_name not in self._profiles:
                raise ValueError(f"Unknown profile: {profile_name}")
            self._active_injections[profile_name] = True
            
    def deactivate_injection(self, profile_name: str) -> None:
        """Deactivate fault injection for a profile"""
        with self._lock:
            self._active_injections[profile_name] = False
            
    def clear_all_injections(self) -> None:
        """Clear all active injections"""
        with self._lock:
            self._active_injections.clear()
            
    def should_inject_fault(self, profile_name: str, context: str = "") -> bool:
        """Deterministically decide if fault should be injected"""
        with self._lock:
            if profile_name not in self._active_injections:
                return False
            if not self._active_injections[profile_name]:
                return False
                
            profile = self._profiles[profile_name]
            
            # Deterministic decision based on profile + context
            hash_input = f"{profile_name}:{context}:{profile.deterministic_seed}"
            hash_digest = hashlib.sha256(hash_input.encode()).hexdigest()
            hash_value = int(hash_digest[:8], 16) / (2**32)
            
            return hash_value < profile.probability
            
    @contextmanager
    def inject_for_class(self, target_class: Type, profile_name: str, context: str = ""):
        """Context manager for class-based fault injection"""
        start_time = time.perf_counter()
        
        if self.should_inject_fault(profile_name, context):
            profile = self._profiles[profile_name]
            
            # Create deterministic hash for this injection
            hash_input = f"{target_class.__name__}:{profile_name}:{context}"
            det_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:16]
            
            # Log injection
            result = InjectionResult(
                fault_triggered=True,
                fault_type=profile.fault_type,
                error_class=profile.error_class,
                fallback_applied=False,
                execution_time_ms=(time.perf_counter() - start_time) * 1000,
                deterministic_hash=det_hash
            )
            self._injection_history.append(result)
            
            # Raise the configured error
            raise profile.error_class(f"Injected {profile.fault_type.value} for {target_class.__name__}")
        else:
            # No injection - normal execution
            try:
                yield
            finally:
                result = InjectionResult(
                    fault_triggered=False,
                    fault_type=None,
                    error_class=None,
                    fallback_applied=False,
                    execution_time_ms=(time.perf_counter() - start_time) * 1000,
                    deterministic_hash=""
                )
                self._injection_history.append(result)
                
    def apply_conservative_fallback(self, target_class: Type, fallback_fn: Callable[[], Any]) -> Any:
        """Apply conservative fallback for a target class"""
        start_time = time.perf_counter()
        
        try:
            result = fallback_fn()
            
            # Log successful fallback
            result_entry = InjectionResult(
                fault_triggered=False,
                fault_type=None,
                error_class=None,
                fallback_applied=True,
                execution_time_ms=(time.perf_counter() - start_time) * 1000,
                deterministic_hash=hashlib.sha256(f"{target_class.__name__}:fallback".encode()).hexdigest()[:16]
            )
            self._injection_history.append(result_entry)
            
            return result
            
        except Exception as e:
            # Log failed fallback
            result_entry = InjectionResult(
                fault_triggered=True,
                fault_type=FaultType.DATA_ERROR,
                error_class=type(e),
                fallback_applied=True,
                execution_time_ms=(time.perf_counter() - start_time) * 1000,
                deterministic_hash=hashlib.sha256(f"{target_class.__name__}:fallback:error".encode()).hexdigest()[:16]
            )
            self._injection_history.append(result_entry)
            raise
            
    def get_injection_stats(self) -> Dict[str, Any]:
        """Get statistics about fault injections"""
        with self._lock:
            total = len(self._injection_history)
            if total == 0:
                return {"total": 0, "faults_triggered": 0, "fallbacks_applied": 0}
                
            faults = sum(1 for r in self._injection_history if r.fault_triggered)
            fallbacks = sum(1 for r in self._injection_history if r.fallback_applied)
            
            return {
                "total": total,
                "faults_triggered": faults,
                "fallbacks_applied": fallbacks,
                "fault_rate": faults / total,
                "fallback_rate": fallbacks / total,
                "avg_execution_time_ms": sum(r.execution_time_ms for r in self._injection_history) / total
            }
            
    def export_history(self) -> List[Dict[str, Any]]:
        """Export injection history for analysis"""
        with self._lock:
            return [
                {
                    "fault_triggered": r.fault_triggered,
                    "fault_type": r.fault_type.value if r.fault_type else None,
                    "error_class": r.error_class.__name__ if r.error_class else None,
                    "fallback_applied": r.fallback_applied,
                    "execution_time_ms": r.execution_time_ms,
                    "deterministic_hash": r.deterministic_hash
                }
                for r in self._injection_history
            ]
            
    def verify_deterministic_behavior(self, target_class: Type, context: str, iterations: int = 100) -> bool:
        """Verify that fault injection is deterministic for a given class/context"""
        results = []
        
        for i in range(iterations):
            should_inject = self.should_inject_fault(f"{target_class.__name__}_test", f"{context}:{i}")
            results.append(should_inject)
            
        # Check that same inputs produce same outputs
        first_run = results[:]
        
        # Second run with same inputs
        second_run = []
        for i in range(iterations):
            should_inject = self.should_inject_fault(f"{target_class.__name__}_test", f"{context}:{i}")
            second_run.append(should_inject)
            
        return first_run == second_run


# Global fault injector instance
fault_injector = FaultInjector()


# Common fault profiles
COMMON_PROFILES = {
    "network_timeout": FaultProfile(
        fault_type=FaultType.NETWORK_ERROR,
        error_class=ConnectionError,
        probability=0.1,
        deterministic_seed=12345
    ),
    "validation_failure": FaultProfile(
        fault_type=FaultType.VALIDATION_ERROR,
        error_class=ValueError,
        probability=0.15,
        deterministic_seed=67890
    ),
    "resource_exhausted": FaultProfile(
        fault_type=FaultType.RESOURCE_ERROR,
        error_class=MemoryError,
        probability=0.05,
        deterministic_seed=11111
    ),
    "data_corruption": FaultProfile(
        fault_type=FaultType.DATA_ERROR,
        error_class=RuntimeError,
        probability=0.08,
        deterministic_seed=22222
    )
}

# Register common profiles
for name, profile in COMMON_PROFILES.items():
    fault_injector.register_fault_profile(name, profile)


if __name__ == "__main__":
    # Demo usage
    class DemoClass:
        def process(self):
            return "success"
            
    # Activate fault injection
    fault_injector.activate_injection("validation_failure")
    
    # Test deterministic behavior
    is_deterministic = fault_injector.verify_deterministic_behavior(DemoClass, "test_context")
    print(f"Deterministic behavior verified: {is_deterministic}")
    
    # Test fault injection
    try:
        with fault_injector.inject_for_class(DemoClass, "validation_failure", "demo_context"):
            demo = DemoClass()
            result = demo.process()
            print(f"Success: {result}")
    except ValueError as e:
        print(f"Fault injected: {e}")
        
    # Show stats
    stats = fault_injector.get_injection_stats()
    print(f"Injection stats: {json.dumps(stats, indent=2)}")
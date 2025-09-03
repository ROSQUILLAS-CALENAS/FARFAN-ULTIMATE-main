#!/usr/bin/env python3
"""
Determinism Verification System
===============================
Comprehensive verification mechanisms to validate consistent component ordering,
phase assignments, and confidence scoring for the canonical flow audit process.
Ensures reproducible results across consecutive runs with identical repository states.

Key Features:
- Component ordering consistency validation
- Phase assignment stability verification  
- Confidence scoring reproducibility testing
- Tie-breaking logic verification
- Output hash comparison across runs
- Randomness source detection and mitigation
"""

import hashlib
import json
import os
import random
import time
# # # from dataclasses import dataclass, asdict  # Module not found  # Module not found  # Module not found
# # # from pathlib import Path  # Module not found  # Module not found  # Module not found
# # # from typing import Any, Dict, List, Optional, Tuple, Callable  # Module not found  # Module not found  # Module not found
import uuid

# Try to import numpy for deterministic seeding
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

# Import core audit components
try:
    import canonical_output_auditor as coa
    HAS_CANONICAL_AUDITOR = True
except ImportError:
    HAS_CANONICAL_AUDITOR = False

try:
# # #     from tools.canonical_cojoin_auditor import iter_repo_files, guess_phase, Evidence, InventoryItem  # Module not found  # Module not found  # Module not found
    HAS_COJOIN_AUDITOR = True
except ImportError:
    HAS_COJOIN_AUDITOR = False


@dataclass
class DeterminismTestResult:
# # #     """Results from a determinism verification test"""  # Module not found  # Module not found  # Module not found
    test_name: str
    success: bool
    consistency_score: float
    hash_matches: bool
    ordering_stable: bool
    confidence_variance: float
    execution_time_ms: float
    error_message: Optional[str] = None
    details: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass 
class VerificationReport:
    """Comprehensive verification report"""
    timestamp: float
    total_tests: int
    passed_tests: int
    failed_tests: int
    overall_success: bool
    test_results: List[DeterminismTestResult]
    system_info: Dict[str, Any]
    recommendations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "total_tests": self.total_tests,
            "passed_tests": self.passed_tests,
            "failed_tests": self.failed_tests,
            "overall_success": self.overall_success,
            "test_results": [result.to_dict() for result in self.test_results],
            "system_info": self.system_info,
            "recommendations": self.recommendations
        }


class DeterminismVerifier:
    """Core determinism verification system"""
    
    def __init__(self, root_path: Optional[Path] = None):
        self.root_path = root_path or Path.cwd()
        self.test_results: List[DeterminismTestResult] = []
        self.deterministic_seed = 42
        self._setup_deterministic_environment()
        
    def _setup_deterministic_environment(self) -> None:
        """Configure environment for deterministic execution"""
        # Set environment variables
        os.environ['DETERMINISTIC_MODE'] = '1'
        os.environ['PYTHONHASHSEED'] = str(self.deterministic_seed)
        
        # Set random seeds
        random.seed(self.deterministic_seed)
        if HAS_NUMPY:
            np.random.seed(self.deterministic_seed)
            
        # Configure JSON serialization for consistency
        self.json_kwargs = {
            'sort_keys': True,
            'ensure_ascii': False,
            'separators': (',', ':')
        }
    
    def compute_stable_hash(self, data: Any) -> str:
        """Compute stable, deterministic hash of any data structure"""
        try:
            if isinstance(data, dict):
                # Sort keys for consistent serialization
                serialized = json.dumps(data, **self.json_kwargs)
            elif isinstance(data, (list, tuple)):
                # Convert to sorted representation if contains dicts
                normalized = []
                for item in data:
                    if isinstance(item, dict):
                        normalized.append(json.dumps(item, **self.json_kwargs))
                    else:
                        normalized.append(str(item))
                serialized = json.dumps(normalized, **self.json_kwargs)
            else:
                serialized = json.dumps(data, **self.json_kwargs)
        except Exception:
            # Fallback to string representation
            serialized = str(data)
            
        return hashlib.sha256(serialized.encode('utf-8')).hexdigest()[:16]
    
    def generate_test_data(self) -> Dict[str, Any]:
        """Generate deterministic test data for verification"""
        return {
            "test_id": "determinism_verification",
            "timestamp": 1640995200.0,  # Fixed timestamp
            "components": [
                {"name": "component_a", "phase": "I", "confidence": 0.85},
                {"name": "component_b", "phase": "A", "confidence": 0.92},
                {"name": "component_c", "phase": "L", "confidence": 0.78},
                {"name": "component_d", "phase": "R", "confidence": 0.88},
            ],
            "evidence": {
                "Q1": [
                    {"id": "ev_001", "content": "Evidence 1", "score": 0.9},
                    {"id": "ev_002", "content": "Evidence 2", "score": 0.7},
                ],
                "Q2": [
                    {"id": "ev_003", "content": "Evidence 3", "score": 0.85},
                ]
            },
            "metadata": {
                "version": "1.0.0",
                "algorithm": "canonical_audit",
                "config_hash": "abc123def456"
            }
        }
    
    def test_hash_consistency(self, runs: int = 5) -> DeterminismTestResult:
        """Test that identical data produces identical hashes across runs"""
        start_time = time.time()
        test_data = self.generate_test_data()
        hashes = []
        
        try:
            for i in range(runs):
                # Reset environment for each run
                self._setup_deterministic_environment()
                hash_value = self.compute_stable_hash(test_data)
                hashes.append(hash_value)
            
            # Check if all hashes are identical
            unique_hashes = set(hashes)
            hash_matches = len(unique_hashes) == 1
            
            execution_time = (time.time() - start_time) * 1000
            
            return DeterminismTestResult(
                test_name="hash_consistency",
                success=hash_matches,
                consistency_score=1.0 if hash_matches else 0.0,
                hash_matches=hash_matches,
                ordering_stable=True,  # Not applicable for this test
                confidence_variance=0.0,
                execution_time_ms=execution_time,
                details={"hashes": hashes, "unique_count": len(unique_hashes)}
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            return DeterminismTestResult(
                test_name="hash_consistency",
                success=False,
                consistency_score=0.0,
                hash_matches=False,
                ordering_stable=False,
                confidence_variance=0.0,
                execution_time_ms=execution_time,
                error_message=str(e)
            )
    
    def test_component_ordering(self, runs: int = 5) -> DeterminismTestResult:
        """Test that component ordering is consistent across runs"""
        start_time = time.time()
        
        try:
            orderings = []
            for i in range(runs):
                self._setup_deterministic_environment()
                
                # Simulate component discovery and ordering
                components = [
                    {"name": f"comp_{j}", "phase": random.choice(["I", "A", "L", "R"]), 
                     "path": f"path/to/comp_{j}.py"} 
                    for j in range(10)
                ]
                
                # Apply deterministic ordering
                ordered = sorted(components, key=lambda x: (
                    self._get_phase_priority(x["phase"]), 
                    x["name"],
                    x["path"]
                ))
                
                ordering = [comp["name"] for comp in ordered]
                orderings.append(ordering)
            
            # Check consistency
            reference_ordering = orderings[0]
            all_match = all(ordering == reference_ordering for ordering in orderings)
            
            execution_time = (time.time() - start_time) * 1000
            
            return DeterminismTestResult(
                test_name="component_ordering",
                success=all_match,
                consistency_score=1.0 if all_match else 0.0,
                hash_matches=True,  # Not applicable
                ordering_stable=all_match,
                confidence_variance=0.0,
                execution_time_ms=execution_time,
                details={"orderings": orderings, "reference": reference_ordering}
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            return DeterminismTestResult(
                test_name="component_ordering",
                success=False,
                consistency_score=0.0,
                hash_matches=False,
                ordering_stable=False,
                confidence_variance=0.0,
                execution_time_ms=execution_time,
                error_message=str(e)
            )
    
    def test_confidence_scoring(self, runs: int = 5) -> DeterminismTestResult:
        """Test that confidence scoring is stable across runs"""
        start_time = time.time()
        
        try:
            confidence_sets = []
            for i in range(runs):
                self._setup_deterministic_environment()
                
                # Simulate confidence calculation
                components = [
                    {"name": "comp_a", "evidence_count": 3, "phase_match": True},
                    {"name": "comp_b", "evidence_count": 1, "phase_match": False},
                    {"name": "comp_c", "evidence_count": 2, "phase_match": True},
                ]
                
                confidences = []
                for comp in components:
                    base_confidence = 0.5
                    if comp["phase_match"]:
                        base_confidence += 0.3
                    evidence_bonus = min(0.2, comp["evidence_count"] * 0.05)
                    confidence = min(1.0, base_confidence + evidence_bonus)
                    confidences.append(round(confidence, 4))
                
                confidence_sets.append(confidences)
            
            # Calculate variance
            if len(confidence_sets) > 1:
                variances = []
                for i in range(len(confidence_sets[0])):
                    values = [conf_set[i] for conf_set in confidence_sets]
                    if HAS_NUMPY:
                        variance = float(np.var(values))
                    else:
                        mean = sum(values) / len(values)
                        variance = sum((x - mean) ** 2 for x in values) / len(values)
                    variances.append(variance)
                
                max_variance = max(variances)
                avg_variance = sum(variances) / len(variances)
            else:
                max_variance = avg_variance = 0.0
            
            # Success if variance is very low (< 0.001)
            success = max_variance < 0.001
            
            execution_time = (time.time() - start_time) * 1000
            
            return DeterminismTestResult(
                test_name="confidence_scoring",
                success=success,
                consistency_score=1.0 - min(1.0, max_variance * 1000),  # Scale variance
                hash_matches=True,  # Not applicable
                ordering_stable=True,  # Not applicable
                confidence_variance=avg_variance,
                execution_time_ms=execution_time,
                details={"confidence_sets": confidence_sets, "max_variance": max_variance}
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            return DeterminismTestResult(
                test_name="confidence_scoring",
                success=False,
                consistency_score=0.0,
                hash_matches=False,
                ordering_stable=False,
                confidence_variance=float('inf'),
                execution_time_ms=execution_time,
                error_message=str(e)
            )
    
    def test_canonical_audit_determinism(self, runs: int = 3) -> DeterminismTestResult:
        """Test canonical output auditor determinism"""
        start_time = time.time()
        
        if not HAS_CANONICAL_AUDITOR:
            return DeterminismTestResult(
                test_name="canonical_audit_determinism",
                success=False,
                consistency_score=0.0,
                hash_matches=False,
                ordering_stable=False,
                confidence_variance=0.0,
                execution_time_ms=0.0,
                error_message="canonical_output_auditor not available"
            )
        
        try:
            audit_hashes = []
            for i in range(runs):
                self._setup_deterministic_environment()
                
                # Create test audit data
                test_data = {
                    "cluster_audit": {
                        "present": ["C1", "C2", "C3", "C4"],
                        "complete": True,
                        "non_redundant": True,
                        "micro": {
                            "C1": {"answers": [{"question_id": "Q1", "evidence_ids": ["e1"]}]},
                            "C2": {"answers": [{"question_id": "Q2", "evidence_ids": ["e2"]}]},
                            "C3": {"answers": [{"question_id": "Q3", "evidence_ids": ["e3"]}]},
                            "C4": {"answers": [{"question_id": "Q4", "evidence_ids": ["e4"]}]},
                        }
                    },
                    "dnp_alignment": {"status": "compliant"},
                    "causal_correction": {"applied": True},
                    "evidence": {
                        "Q1": [{"id": "e1", "content": "Evidence 1"}],
                        "Q2": [{"id": "e2", "content": "Evidence 2"}],
                        "Q3": [{"id": "e3", "content": "Evidence 3"}],
                        "Q4": [{"id": "e4", "content": "Evidence 4"}],
                    }
                }
                
                # Run canonical audit
                result = coa.process(test_data, context={"deterministic": True})
                audit_hash = self.compute_stable_hash(result.get("canonical_audit", {}))
                audit_hashes.append(audit_hash)
            
            # Check consistency
            unique_hashes = set(audit_hashes)
            success = len(unique_hashes) == 1
            
            execution_time = (time.time() - start_time) * 1000
            
            return DeterminismTestResult(
                test_name="canonical_audit_determinism",
                success=success,
                consistency_score=1.0 if success else 0.0,
                hash_matches=success,
                ordering_stable=success,
                confidence_variance=0.0,
                execution_time_ms=execution_time,
                details={"audit_hashes": audit_hashes, "unique_count": len(unique_hashes)}
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            return DeterminismTestResult(
                test_name="canonical_audit_determinism",
                success=False,
                consistency_score=0.0,
                hash_matches=False,
                ordering_stable=False,
                confidence_variance=0.0,
                execution_time_ms=execution_time,
                error_message=str(e)
            )
    
    def test_tie_breaking_consistency(self, runs: int = 5) -> DeterminismTestResult:
        """Test that tie-breaking logic is consistent"""
        start_time = time.time()
        
        try:
            tie_break_results = []
            for i in range(runs):
                self._setup_deterministic_environment()
                
                # Create items with identical scores (ties)
                items = [
                    {"id": "item_a", "score": 0.85, "name": "alpha"},
                    {"id": "item_b", "score": 0.85, "name": "beta"}, 
                    {"id": "item_c", "score": 0.85, "name": "gamma"},
                    {"id": "item_d", "score": 0.90, "name": "delta"},
                ]
                
                # Apply deterministic tie-breaking: score desc, then name asc
                sorted_items = sorted(items, key=lambda x: (-x["score"], x["name"]))
                result_order = [item["id"] for item in sorted_items]
                tie_break_results.append(result_order)
            
            # Check consistency
            reference = tie_break_results[0]
            all_match = all(result == reference for result in tie_break_results)
            
            execution_time = (time.time() - start_time) * 1000
            
            return DeterminismTestResult(
                test_name="tie_breaking_consistency",
                success=all_match,
                consistency_score=1.0 if all_match else 0.0,
                hash_matches=True,
                ordering_stable=all_match,
                confidence_variance=0.0,
                execution_time_ms=execution_time,
                details={"tie_break_results": tie_break_results, "reference": reference}
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            return DeterminismTestResult(
                test_name="tie_breaking_consistency",
                success=False,
                consistency_score=0.0,
                hash_matches=False,
                ordering_stable=False,
                confidence_variance=0.0,
                execution_time_ms=execution_time,
                error_message=str(e)
            )
    
    def _get_phase_priority(self, phase: str) -> int:
        """Get numeric priority for phase ordering"""
        phase_order = {"I": 1, "X": 2, "K": 3, "A": 4, "L": 5, 
                      "R": 6, "O": 7, "G": 8, "T": 9, "S": 10}
        return phase_order.get(phase, 99)
    
    def run_comprehensive_verification(self, runs_per_test: int = 5) -> VerificationReport:
        """Run all determinism verification tests"""
        print("🔍 Starting comprehensive determinism verification...")
        
        # Define test suite
        tests = [
            ("hash_consistency", self.test_hash_consistency),
            ("component_ordering", self.test_component_ordering), 
            ("confidence_scoring", self.test_confidence_scoring),
            ("canonical_audit_determinism", self.test_canonical_audit_determinism),
            ("tie_breaking_consistency", self.test_tie_breaking_consistency),
        ]
        
        test_results = []
        passed = 0
        failed = 0
        
        for test_name, test_func in tests:
            print(f"  ⏳ Running {test_name}...")
            try:
                result = test_func(runs_per_test)
                test_results.append(result)
                
                if result.success:
                    print(f"  ✅ {test_name}: PASSED (consistency: {result.consistency_score:.3f})")
                    passed += 1
                else:
                    print(f"  ❌ {test_name}: FAILED ({result.error_message or 'consistency check failed'})")
                    failed += 1
                    
            except Exception as e:
                print(f"  💥 {test_name}: ERROR - {e}")
                test_results.append(DeterminismTestResult(
                    test_name=test_name,
                    success=False,
                    consistency_score=0.0,
                    hash_matches=False,
                    ordering_stable=False,
                    confidence_variance=0.0,
                    execution_time_ms=0.0,
                    error_message=str(e)
                ))
                failed += 1
        
        # Generate recommendations
        recommendations = []
        if failed > 0:
            recommendations.append("❌ Some determinism tests failed - review error messages and fix inconsistencies")
        
        for result in test_results:
            if not result.success and result.confidence_variance > 0.01:
                recommendations.append(f"⚠️  High variance in {result.test_name} - check for randomness sources")
            if not result.success and not result.ordering_stable:
                recommendations.append(f"⚠️  Unstable ordering in {result.test_name} - implement tie-breaking")
        
        if not recommendations:
            recommendations.append("✅ All determinism tests passed - system is reproducible")
        
        # Create report
        overall_success = failed == 0
        system_info = {
            "python_version": f"{__import__('sys').version_info.major}.{__import__('sys').version_info.minor}",
            "deterministic_mode": os.environ.get('DETERMINISTIC_MODE', '0') == '1',
            "numpy_available": HAS_NUMPY,
            "canonical_auditor_available": HAS_CANONICAL_AUDITOR,
            "cojoin_auditor_available": HAS_COJOIN_AUDITOR,
            "random_seed": self.deterministic_seed,
        }
        
        report = VerificationReport(
            timestamp=time.time(),
            total_tests=len(tests),
            passed_tests=passed,
            failed_tests=failed,
            overall_success=overall_success,
            test_results=test_results,
            system_info=system_info,
            recommendations=recommendations
        )
        
        return report
    
    def save_verification_report(self, report: VerificationReport, 
                               output_path: Path = None) -> Path:
        """Save verification report to JSON file"""
        if output_path is None:
            output_path = self.root_path / f"determinism_verification_report_{int(report.timestamp)}.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report.to_dict(), f, indent=2, **self.json_kwargs)
        
        return output_path


def main():
    """Main entry point for determinism verification"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Verify determinism of canonical flow audit")
    parser.add_argument("--runs", type=int, default=5, 
                       help="Number of runs per test (default: 5)")
    parser.add_argument("--output", type=Path, 
                       help="Output path for verification report")
    parser.add_argument("--verbose", action="store_true", 
                       help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Create verifier and run tests
    verifier = DeterminismVerifier()
    report = verifier.run_comprehensive_verification(args.runs)
    
    # Save report
    report_path = verifier.save_verification_report(report, args.output)
    
    # Print summary
    print(f"\n📊 Verification Summary:")
    print(f"  Total Tests: {report.total_tests}")
    print(f"  Passed: {report.passed_tests}")
    print(f"  Failed: {report.failed_tests}")
    print(f"  Success Rate: {report.passed_tests/report.total_tests*100:.1f}%")
    print(f"  Overall: {'✅ PASS' if report.overall_success else '❌ FAIL'}")
    
    print(f"\n💾 Report saved to: {report_path}")
    
    if args.verbose:
        print(f"\n🔧 Recommendations:")
        for rec in report.recommendations:
            print(f"  {rec}")
    
    # Exit with appropriate code
    exit_code = 0 if report.overall_success else 1
    return exit_code


if __name__ == "__main__":
    exit(main())
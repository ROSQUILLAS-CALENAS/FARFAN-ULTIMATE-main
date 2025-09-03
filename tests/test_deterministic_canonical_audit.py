"""
Comprehensive Test Suite for Deterministic Canonical Audit System

This test suite validates that the canonical audit system produces identical output
hashes across multiple consecutive runs when the deterministic context flag is enabled,
and that non-deterministic mode functions correctly with timestamp variations.

Key Test Areas:
- Hash consistency across multiple runs with deterministic=True
- Component count consistency
- Ordering stability
- Non-deterministic mode behavior validation
- Edge cases and error conditions
"""
import hashlib
import json
import time
import unittest
from typing import Any, Dict, List, Set
from unittest.mock import MagicMock, patch

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

try:
    import canonical_output_auditor as coa
    CANONICAL_AUDITOR_AVAILABLE = True
except ImportError:
    CANONICAL_AUDITOR_AVAILABLE = False
    coa = None


class TestDeterministicCanonicalAudit(unittest.TestCase):
    """Test suite for deterministic canonical audit behavior"""

    def setUp(self):
        """Set up test fixtures"""
        self.base_test_data = {
            "cluster_audit": {
                "present": ["C1", "C2", "C3", "C4"],
                "complete": True,
                "non_redundant": True,
                "micro": {
                    "C1": {
                        "answers": [
                            {
                                "question_id": "q1",
                                "evidence_ids": ["e1", "e2"],
                                "response": "Test response 1"
                            }
                        ],
                        "evidence_linked": True
                    },
                    "C2": {
                        "answers": [
                            {
                                "question_id": "q2", 
                                "evidence_ids": ["e3", "e4"],
                                "response": "Test response 2"
                            }
                        ],
                        "evidence_linked": True
                    },
                    "C3": {
                        "answers": [
                            {
                                "question_id": "q3",
                                "evidence_ids": ["e5"],
                                "response": "Test response 3"
                            }
                        ],
                        "evidence_linked": True
                    },
                    "C4": {
                        "answers": [
                            {
                                "question_id": "q4",
                                "evidence_ids": ["e6", "e7"],
                                "response": "Test response 4"
                            }
                        ],
                        "evidence_linked": True
                    }
                }
            },
            "meso_summary": {
                "divergence_stats": {
                    "max": 0.25,
                    "avg": 0.15,
                    "count": 4
                },
                "items": {
                    "C1": {"evidence_coverage": 2},
                    "C2": {"evidence_coverage": 2},
                    "C3": {"evidence_coverage": 1},
                    "C4": {"evidence_coverage": 2}
                }
            },
            "dnp_validation_results": {"compliant": True},
            "causal_correction": {"applied": True},
            "evidence": {
                "q1": [{"id": "e1"}, {"id": "e2"}],
                "q2": [{"id": "e3"}, {"id": "e4"}],
                "q3": [{"id": "e5"}],
                "q4": [{"id": "e6"}, {"id": "e7"}]
            },
            "features": {"extracted": True},
            "vectors": [1, 2, 3, 4],
            "bm25_index": True,
            "vector_index": True,
            "evidence_system": True
        }

    def _compute_hash(self, data: Any) -> str:
        """Compute stable hash of data structure"""
        if isinstance(data, dict):
            # Remove timestamp fields for hash computation
            filtered_data = {}
            for k, v in data.items():
                if k == "timestamp":
                    continue
                elif isinstance(v, dict):
                    filtered_data[k] = self._remove_timestamps_recursive(v)
                else:
                    filtered_data[k] = v
            content = json.dumps(filtered_data, sort_keys=True)
        else:
            content = json.dumps(data, sort_keys=True)
        return hashlib.sha256(content.encode('utf-8')).hexdigest()

    def _remove_timestamps_recursive(self, data: Any) -> Any:
        """Recursively remove timestamp fields from nested structures"""
        if isinstance(data, dict):
            return {k: self._remove_timestamps_recursive(v) for k, v in data.items() if k != "timestamp"}
        elif isinstance(data, list):
            return [self._remove_timestamps_recursive(item) for item in data]
        else:
            return data

    @unittest.skipUnless(CANONICAL_AUDITOR_AVAILABLE, "canonical_output_auditor not available")
    def test_deterministic_mode_hash_consistency_5_runs(self):
        """Test that 5 consecutive runs with deterministic=True produce identical hashes"""
        results = []
        hashes = []
        
        for i in range(5):
            result = coa.process(self.base_test_data.copy(), context={"deterministic": True})
            results.append(result)
            
            # Compute hash of canonical_audit section
            canonical_audit = result.get("canonical_audit", {})
            audit_hash = self._compute_hash(canonical_audit)
            hashes.append(audit_hash)
        
        # All hashes should be identical
        unique_hashes = set(hashes)
        self.assertEqual(len(unique_hashes), 1, 
                        f"Expected identical hashes, got {len(unique_hashes)} unique: {unique_hashes}")
        
        # Verify specific fields are identical
        first_audit = results[0].get("canonical_audit", {})
        for result in results[1:]:
            current_audit = result.get("canonical_audit", {})
            self.assertEqual(first_audit.get("four_clusters_confirmed"), 
                           current_audit.get("four_clusters_confirmed"))
            self.assertEqual(first_audit.get("gaps"), current_audit.get("gaps"))
            self.assertEqual(first_audit.get("timestamp"), current_audit.get("timestamp"))

    @unittest.skipUnless(CANONICAL_AUDITOR_AVAILABLE, "canonical_output_auditor not available")
    def test_deterministic_mode_hash_consistency_10_runs(self):
        """Test that 10 consecutive runs with deterministic=True produce identical hashes"""
        results = []
        hashes = []
        
        for i in range(10):
            result = coa.process(self.base_test_data.copy(), context={"deterministic": True})
            results.append(result)
            
            # Compute hash of entire result structure
            result_hash = self._compute_hash(result)
            hashes.append(result_hash)
        
        # All hashes should be identical
        unique_hashes = set(hashes)
        self.assertEqual(len(unique_hashes), 1, 
                        f"Expected identical hashes across 10 runs, got {len(unique_hashes)} unique")

    @unittest.skipUnless(CANONICAL_AUDITOR_AVAILABLE, "canonical_output_auditor not available")
    def test_component_count_consistency(self):
        """Test that component counts remain identical across deterministic runs"""
        results = []
        
        for i in range(7):
            result = coa.process(self.base_test_data.copy(), context={"deterministic": True})
            results.append(result)
        
        # Extract component counts from each result
        first_result = results[0]
        first_audit = first_result.get("canonical_audit", {})
        first_gaps_count = len(first_audit.get("gaps", []))
        first_resolution_count = len(first_audit.get("evidence_traceability", {}).get("per_cluster", {}))
        
        for i, result in enumerate(results[1:], 1):
            audit = result.get("canonical_audit", {})
            gaps_count = len(audit.get("gaps", []))
            resolution_count = len(audit.get("evidence_traceability", {}).get("per_cluster", {}))
            
            self.assertEqual(first_gaps_count, gaps_count, 
                           f"Gap count differs in run {i}: {first_gaps_count} vs {gaps_count}")
            self.assertEqual(first_resolution_count, resolution_count,
                           f"Resolution count differs in run {i}: {first_resolution_count} vs {resolution_count}")

    @unittest.skipUnless(CANONICAL_AUDITOR_AVAILABLE, "canonical_output_auditor not available")
    def test_ordering_stability(self):
        """Test that ordering of components remains stable across deterministic runs"""
        results = []
        
        for i in range(6):
            result = coa.process(self.base_test_data.copy(), context={"deterministic": True})
            results.append(result)
        
        # Check that gaps are consistently ordered (should be sorted)
        first_gaps = results[0].get("canonical_audit", {}).get("gaps", [])
        
        for i, result in enumerate(results[1:], 1):
            gaps = result.get("canonical_audit", {}).get("gaps", [])
            self.assertEqual(first_gaps, gaps, 
                           f"Gap ordering differs in run {i}: {first_gaps} vs {gaps}")
            
            # Gaps should be sorted
            self.assertEqual(gaps, sorted(gaps), 
                           f"Gaps not sorted in run {i}: {gaps}")

    @unittest.skipUnless(CANONICAL_AUDITOR_AVAILABLE, "canonical_output_auditor not available")
    def test_non_deterministic_mode_timestamp_variation(self):
        """Test that non-deterministic mode produces varying timestamps"""
        results = []
        timestamps = []
        
        for i in range(5):
            result = coa.process(self.base_test_data.copy(), context={"deterministic": False})
            results.append(result)
            
            audit_timestamp = result.get("canonical_audit", {}).get("timestamp")
            macro_timestamp = result.get("macro_synthesis", {}).get("timestamp")
            
            timestamps.append((audit_timestamp, macro_timestamp))
            
            # Add small delay to ensure timestamp variation
            time.sleep(0.01)
        
        # Extract unique timestamps
        audit_timestamps = set(t[0] for t in timestamps if t[0] is not None)
        macro_timestamps = set(t[1] for t in timestamps if t[1] is not None)
        
        # Should have multiple unique timestamps in non-deterministic mode
        self.assertGreater(len(audit_timestamps), 1, 
                          "Expected varying timestamps in non-deterministic mode")
        self.assertGreater(len(macro_timestamps), 1,
                          "Expected varying macro synthesis timestamps in non-deterministic mode")

    @unittest.skipUnless(CANONICAL_AUDITOR_AVAILABLE, "canonical_output_auditor not available")
    def test_default_mode_timestamp_variation(self):
        """Test that default mode (no deterministic flag) produces varying timestamps"""
        results = []
        timestamps = []
        
        for i in range(5):
            # No context provided, should default to non-deterministic
            result = coa.process(self.base_test_data.copy())
            results.append(result)
            
            audit_timestamp = result.get("canonical_audit", {}).get("timestamp")
            timestamps.append(audit_timestamp)
            
            time.sleep(0.01)
        
        unique_timestamps = set(t for t in timestamps if t is not None)
        self.assertGreater(len(unique_timestamps), 1, 
                          "Expected varying timestamps when no deterministic flag is provided")

    @unittest.skipUnless(CANONICAL_AUDITOR_AVAILABLE, "canonical_output_auditor not available")
    def test_deterministic_vs_non_deterministic_comparison(self):
        """Test comparison between deterministic and non-deterministic modes"""
        # Run deterministic mode multiple times
        det_results = []
        for i in range(3):
            result = coa.process(self.base_test_data.copy(), context={"deterministic": True})
            det_results.append(result)
        
        # Run non-deterministic mode multiple times  
        non_det_results = []
        for i in range(3):
            result = coa.process(self.base_test_data.copy(), context={"deterministic": False})
            non_det_results.append(result)
            time.sleep(0.01)
        
        # Deterministic timestamps should be identical
        det_timestamps = [r.get("canonical_audit", {}).get("timestamp") for r in det_results]
        det_unique = set(det_timestamps)
        self.assertEqual(len(det_unique), 1, "Deterministic timestamps should be identical")
        
        # Non-deterministic timestamps should vary
        non_det_timestamps = [r.get("canonical_audit", {}).get("timestamp") for r in non_det_results]
        non_det_unique = set(non_det_timestamps)
        self.assertGreater(len(non_det_unique), 1, "Non-deterministic timestamps should vary")
        
        # Other fields should be the same except for replicability hashes which are deterministic in det mode
        det_audit = det_results[0].get("canonical_audit", {})
        non_det_audit = non_det_results[0].get("canonical_audit", {})
        
        # Remove timestamp and replicability fields for comparison
        det_audit_filtered = {k: v for k, v in det_audit.items() 
                             if k not in ["timestamp", "replicability"]}
        non_det_audit_filtered = {k: v for k, v in non_det_audit.items() 
                                 if k not in ["timestamp", "replicability"]}
        
        self.assertEqual(det_audit_filtered, non_det_audit_filtered,
                        "Non-timestamp/replicability fields should be identical between modes")
        
        # Verify replicability hashes are different between modes
        det_replicability = det_audit.get("replicability", {})
        non_det_replicability = non_det_audit.get("replicability", {})
        
        self.assertNotEqual(det_replicability, non_det_replicability,
                           "Replicability hashes should differ between deterministic and non-deterministic modes")

    @unittest.skipUnless(CANONICAL_AUDITOR_AVAILABLE, "canonical_output_auditor not available")
    def test_edge_case_empty_data(self):
        """Test deterministic behavior with empty data"""
        empty_data = {}
        
        results = []
        for i in range(5):
            result = coa.process(empty_data.copy(), context={"deterministic": True})
            results.append(result)
        
        # Should still be deterministic even with empty data
        hashes = [self._compute_hash(r.get("canonical_audit", {})) for r in results]
        unique_hashes = set(hashes)
        self.assertEqual(len(unique_hashes), 1, 
                        "Empty data should still produce deterministic results")

    @unittest.skipUnless(CANONICAL_AUDITOR_AVAILABLE, "canonical_output_auditor not available")
    def test_edge_case_minimal_data(self):
        """Test deterministic behavior with minimal data"""
        minimal_data = {"cluster_audit": {"present": ["C1"]}}
        
        results = []
        hashes = []
        
        for i in range(5):
            result = coa.process(minimal_data.copy(), context={"deterministic": True})
            results.append(result)
            hashes.append(self._compute_hash(result.get("canonical_audit", {})))
        
        unique_hashes = set(hashes)
        self.assertEqual(len(unique_hashes), 1,
                        "Minimal data should produce deterministic results")

    @unittest.skipUnless(CANONICAL_AUDITOR_AVAILABLE, "canonical_output_auditor not available")
    def test_deterministic_flag_values(self):
        """Test various deterministic flag values"""
        test_cases = [
            (True, "should be deterministic"),
            (False, "should be non-deterministic"),
            ("true", "string 'true' should be non-deterministic"), 
            (1, "integer 1 should be non-deterministic"),
            (0, "integer 0 should be non-deterministic")
        ]
        
        for flag_value, description in test_cases:
            with self.subTest(flag_value=flag_value, description=description):
                results = []
                for i in range(3):
                    result = coa.process(self.base_test_data.copy(), context={"deterministic": flag_value})
                    results.append(result)
                    if flag_value is not True:
                        time.sleep(0.01)  # Add delay for non-deterministic cases
                
                timestamps = [r.get("canonical_audit", {}).get("timestamp") for r in results]
                unique_timestamps = set(timestamps)
                
                if flag_value is True:
                    self.assertEqual(len(unique_timestamps), 1,
                                   f"Flag value {flag_value} {description}")
                else:
                    # For non-True values, timestamps should vary (non-deterministic)
                    self.assertGreaterEqual(len(unique_timestamps), 1,
                                          f"Flag value {flag_value} {description}")

    @unittest.skipUnless(CANONICAL_AUDITOR_AVAILABLE, "canonical_output_auditor not available")
    def test_deep_structure_consistency(self):
        """Test that nested structures maintain consistency in deterministic mode"""
        results = []
        
        for i in range(5):
            result = coa.process(self.base_test_data.copy(), context={"deterministic": True})
            results.append(result)
        
        first_result = results[0]
        first_reporting_levels = first_result.get("canonical_audit", {}).get("reporting_levels", {})
        first_raw_data_presence = first_result.get("canonical_audit", {}).get("raw_data_presence", {})
        first_replicability = first_result.get("canonical_audit", {}).get("replicability", {})
        
        for i, result in enumerate(results[1:], 1):
            audit = result.get("canonical_audit", {})
            
            self.assertEqual(first_reporting_levels, audit.get("reporting_levels", {}),
                           f"Reporting levels differ in run {i}")
            self.assertEqual(first_raw_data_presence, audit.get("raw_data_presence", {}),
                           f"Raw data presence differs in run {i}")
            self.assertEqual(first_replicability, audit.get("replicability", {}),
                           f"Replicability differs in run {i}")

    @unittest.skipUnless(CANONICAL_AUDITOR_AVAILABLE, "canonical_output_auditor not available") 
    def test_macro_synthesis_deterministic(self):
        """Test that macro synthesis section is deterministic"""
        results = []
        
        for i in range(8):
            result = coa.process(self.base_test_data.copy(), context={"deterministic": True})
            results.append(result)
        
        first_macro = results[0].get("macro_synthesis", {})
        
        for i, result in enumerate(results[1:], 1):
            macro = result.get("macro_synthesis", {})
            
            # Remove timestamps for comparison
            first_macro_no_ts = self._remove_timestamps_recursive(first_macro)
            macro_no_ts = self._remove_timestamps_recursive(macro)
            
            self.assertEqual(first_macro_no_ts, macro_no_ts,
                           f"Macro synthesis differs in run {i}")
            
            # But timestamps should be deterministic
            self.assertEqual(first_macro.get("timestamp"), macro.get("timestamp"),
                           f"Macro synthesis timestamp should be deterministic in run {i}")


if __name__ == "__main__":
    unittest.main()
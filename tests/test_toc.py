"""
Test cases for Total Ordering with Consistent (TOC) desempates deterministas.

Casos de empate sintéticos para validar que mismo score_vector se desempata 
por orden lexicográfico de κ (uids).
"""

import unittest
from typing import List, Tuple
from egw_query_expansion.core.total_ordering import (
    OrderedRecord,
    sort_total,
    make_total_key,
    explain_tie_breaks,
    serialize_ordered
)


class TestTOCTieBreaking(unittest.TestCase):
    """Test Total Ordering with Consistent (TOC) tie-breaking"""
    
    def test_identical_scores_lexicographic_tiebreak(self):
        """Test que mismo score_vector se desempata por κ lexicográfico"""
        items = [
            OrderedRecord(scores=(0.5, 0.5), uids=("charlie", "3"), payload="C"),
            OrderedRecord(scores=(0.5, 0.5), uids=("alpha", "1"), payload="A"), 
            OrderedRecord(scores=(0.5, 0.5), uids=("beta", "2"), payload="B"),
        ]
        
        ordered = sort_total(
            items, 
            lambda r: r.scores, 
            lambda r: r.uids, 
            descending=True
        )
        
        # Con scores idénticos, debe ordenar por UIDs lexicográficos: alpha < beta < charlie
        expected_payloads = ["A", "B", "C"]
        actual_payloads = [r.payload for r in ordered]
        self.assertEqual(actual_payloads, expected_payloads)
        
    def test_multi_dimensional_identical_scores(self):
        """Test con scores multidimensionales idénticos"""
        items = [
            OrderedRecord(scores=(1.0, 2.0, 3.0), uids=("z", "9", "z"), payload="last"),
            OrderedRecord(scores=(1.0, 2.0, 3.0), uids=("a", "1", "a"), payload="first"),
            OrderedRecord(scores=(1.0, 2.0, 3.0), uids=("m", "5", "m"), payload="middle"),
        ]
        
        ordered = sort_total(
            items,
            lambda r: r.scores,
            lambda r: r.uids,
            descending=True
        )
        
        # Orden lexicográfico: ("a","1","a") < ("m","5","m") < ("z","9","z")
        expected_payloads = ["first", "middle", "last"]
        actual_payloads = [r.payload for r in ordered]
        self.assertEqual(actual_payloads, expected_payloads)
        
    def test_partial_score_ties_with_multiple_uid_dimensions(self):
        """Test empates parciales en scores con múltiples dimensiones de UID"""
        items = [
            OrderedRecord(scores=(0.9, 0.1), uids=("user_b", "doc_1"), payload="B1"),
            OrderedRecord(scores=(0.9, 0.1), uids=("user_a", "doc_2"), payload="A2"),
            OrderedRecord(scores=(0.9, 0.2), uids=("user_c", "doc_1"), payload="C1"),
            OrderedRecord(scores=(0.9, 0.1), uids=("user_a", "doc_1"), payload="A1"),
        ]
        
        ordered = sort_total(
            items,
            lambda r: r.scores,
            lambda r: r.uids,
            descending=True
        )
        
        # Orden esperado:
        # 1. (0.9, 0.2) - C1 (mejor score en segunda dimensión)
        # 2-4. Empate en (0.9, 0.1) - desempatar por UIDs:
        #   - ("user_a", "doc_1") - A1
        #   - ("user_a", "doc_2") - A2  
        #   - ("user_b", "doc_1") - B1
        expected_payloads = ["C1", "A1", "A2", "B1"]
        actual_payloads = [r.payload for r in ordered]
        self.assertEqual(actual_payloads, expected_payloads)
        
    def test_large_tie_group_stability(self):
        """Test con grupo grande de empates para validar estabilidad"""
        # Crear 10 items con mismo score
        items = []
        for i in range(10):
            uid_str = f"item_{i:02d}"  # Padding para orden lexicográfico correcto
            items.append(
                OrderedRecord(
                    scores=(0.75, 0.25), 
                    uids=(uid_str, "category_x"), 
                    payload=f"payload_{i}"
                )
            )
        
        ordered = sort_total(
            items,
            lambda r: r.scores,
            lambda r: r.uids,
            descending=True
        )
        
        # Debe ordenar por UIDs: item_00 < item_01 < ... < item_09
        expected_payloads = [f"payload_{i}" for i in range(10)]
        actual_payloads = [r.payload for r in ordered]
        self.assertEqual(actual_payloads, expected_payloads)
        
    def test_mixed_score_and_uid_scenarios(self):
        """Test escenarios mixtos con scores diferentes y empates"""
        items = [
            OrderedRecord(scores=(1.0,), uids=("high_z",), payload="high_last"),
            OrderedRecord(scores=(1.0,), uids=("high_a",), payload="high_first"), 
            OrderedRecord(scores=(0.5,), uids=("mid_z",), payload="mid_last"),
            OrderedRecord(scores=(0.5,), uids=("mid_a",), payload="mid_first"),
            OrderedRecord(scores=(0.0,), uids=("low_single",), payload="low"),
        ]
        
        ordered = sort_total(
            items,
            lambda r: r.scores,
            lambda r: r.uids,
            descending=True
        )
        
        # Orden esperado (descending=True):
        # 1.0 group: high_a < high_z
        # 0.5 group: mid_a < mid_z  
        # 0.0 group: low_single
        expected_payloads = ["high_first", "high_last", "mid_first", "mid_last", "low"]
        actual_payloads = [r.payload for r in ordered]
        self.assertEqual(actual_payloads, expected_payloads)
        
    def test_explain_tie_breaks_functionality(self):
        """Test funcionalidad de explicación de desempates"""
        items = [
            OrderedRecord(scores=(0.8, 0.2), uids=("doc_1", "query_a"), payload="1a"),
            OrderedRecord(scores=(0.8, 0.2), uids=("doc_2", "query_a"), payload="2a"),
            OrderedRecord(scores=(0.8, 0.2), uids=("doc_1", "query_b"), payload="1b"),
        ]
        
        report = explain_tie_breaks(
            items,
            lambda r: r.scores,
            lambda r: r.uids,
            descending=True
        )
        
        # Debe haber un grupo con empates que usa posiciones de UID para desempatar
        self.assertIn("groups", report)
        self.assertEqual(len(report["groups"]), 1)
        
        group = report["groups"][0]
        self.assertEqual(group["size"], 3)
        self.assertIn("uid_positions_used_for_tiebreak", group)
        # Ambas posiciones de UID (0 y 1) deberían ser usadas para desempates
        self.assertGreater(len(group["uid_positions_used_for_tiebreak"]), 0)

    def test_serialization_deterministic(self):
        """Test que la serialización sea determinística"""
        items = [
            OrderedRecord(scores=(0.6, 0.4), uids=("item_b", "cat_1"), payload="B1"),
            OrderedRecord(scores=(0.6, 0.4), uids=("item_a", "cat_1"), payload="A1"),
        ]
        
        # Serializar múltiples veces debe dar el mismo resultado
        ser1 = serialize_ordered(
            items,
            lambda r: r.scores,
            lambda r: r.uids,
            descending=True
        )
        
        ser2 = serialize_ordered(
            items,
            lambda r: r.scores,
            lambda r: r.uids,
            descending=True
        )
        
        self.assertEqual(ser1, ser2)
        
        # Y debe contener el orden correcto
        import json
        data = json.loads(ser1)
        
        self.assertIn("order", data)
        self.assertEqual(len(data["order"]), 2)
        
        # Primer elemento debe ser A1 (por orden lexicográfico de UIDs)
        first_item = data["order"][0]
        self.assertEqual(first_item["rank"], 0)
        self.assertEqual(first_item["uids"], ["item_a", "cat_1"])


if __name__ == "__main__":
    unittest.main()
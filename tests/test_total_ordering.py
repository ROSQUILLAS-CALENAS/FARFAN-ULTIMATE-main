import unittest

# # # from egw_query_expansion.core import (  # Module not found  # Module not found  # Module not found
    OrderedRecord,
    explain_comparison,
    explain_tie_breaks,
    make_total_key,
    serialize_ordered,
    sort_total,
    total_compare,
)


class TestTotalOrdering(unittest.TestCase):
    def test_basic_sort_and_tiebreak(self):
        items = [
            OrderedRecord(scores=(0.9, 2.0), uids=("b", "1"), payload="X"),
            OrderedRecord(scores=(0.9, 2.0), uids=("a", "2"), payload="Y"),
            OrderedRecord(scores=(0.95, 1.0), uids=("c", "3"), payload="Z"),
        ]
        ordered = sort_total(
            items, lambda r: r.scores, lambda r: r.uids, descending=True
        )
        # Highest first on primary coordinate; tie on first -> compare second; tie on both -> UID
        self.assertEqual(
            [r.payload for r in ordered], ["Z", "Y", "X"]
        )  # a < b so Y before X

    def test_nan_and_inf(self):
        items = [
            OrderedRecord(scores=(float("nan"), 10.0), uids=("id3",), payload=3),
            OrderedRecord(scores=(float("inf"), 0.0), uids=("id1",), payload=1),
            OrderedRecord(scores=(1.0, 0.0), uids=("id2",), payload=2),
        ]
        ordered = sort_total(
            items, lambda r: r.scores, lambda r: r.uids, descending=True
        )
        # +inf should come before finite, NaN should be worst
        self.assertEqual([r.payload for r in ordered], [1, 2, 3])

    def test_serialize_and_explain(self):
        a = OrderedRecord(scores=(1.0, 2.0), uids=("A", "1"))
        b = OrderedRecord(scores=(1.0, 2.0), uids=("B", "0"))
        cmp = total_compare((a.scores, a.uids), (b.scores, b.uids))
        self.assertIn(cmp, (-1, 0, 1))
        explain = explain_comparison(a.scores, a.uids, b.scores, b.uids)
        self.assertIn("steps", explain)
        s = serialize_ordered([a, b], lambda r: r.scores, lambda r: r.uids)
        self.assertIsInstance(s, str)
        self.assertIn("order", s)

    def test_tie_breaks_report(self):
        items = [
            OrderedRecord(scores=(5.0,), uids=("u1", "x")),
            OrderedRecord(scores=(5.0,), uids=("u2", "x")),
            OrderedRecord(scores=(5.0,), uids=("u2", "y")),
        ]
        report = explain_tie_breaks(items, lambda r: r.scores, lambda r: r.uids)
        self.assertIn("groups", report)
        self.assertGreaterEqual(len(report["groups"]), 1)


if __name__ == "__main__":
    unittest.main()

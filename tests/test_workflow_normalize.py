import copy
import json
import unittest

from workflow_definitions import canonical_json, normalize


def _strip_hashes(norm):
    d = copy.deepcopy(norm)
    d.pop("digest", None)
    for st in d.get("steps", []):
        st.pop("effective_metadata_hash", None)
    return d


class TestWorkflowNormalize(unittest.TestCase):
    def test_golden_normalization(self):
        # Input similar to the example (simplified)
        definition = {
            "schema": "2.0",
            "steps": [
                {
                    "step_id": "retrieve",
                    "handler": "handlers.retrieve",
                    "depends_on": [],
                    "metadata": {
                        "requires_snapshot": True,
                        "snapshot_id": "merkle:abc123",
                        "commutative": True,
                        "associative": True,
                        "ordered": False,
                        "join_semantics": "orset",
                        "routing_key": "qpattern",
                        "routing_hash": "rc:7e4b",
                        "cost": 1.0,
                        "gain_fn_id": "bm25_dense_fusion",
                        "risk": {
                            "alpha": 0.1,
                            "split_seed": 42,
                            "scoring_fn": "abs_error",
                        },
                        "timeout_s": 120,
                        "resources": {"cpu": 0.5, "mem_mb": 512},
                        "revision": 3,
                    },
                }
            ],
        }
        normalized = normalize(definition)
        with open("tests/golden/workflow_v2_basic.json", "r", encoding="utf-8") as f:
            golden = json.load(f)
        self.assertEqual(_strip_hashes(normalized), golden)

    def test_depends_on_permutation_commutative(self):
        base = {
            "schema": "2.0",
            "steps": [
                {
                    "step_id": "A",
                    "handler": "h.a",
                    "depends_on": [],
                    "metadata": {"commutative": True},
                },
                {
                    "step_id": "B",
                    "handler": "h.b",
                    "depends_on": ["A"],
                    "metadata": {"commutative": True},
                },
                {
                    "step_id": "C",
                    "handler": "h.c",
                    "depends_on": ["A", "B"],
                    "metadata": {"commutative": True},
                },
            ],
        }
        permuted = copy.deepcopy(base)
        permuted["steps"][2]["depends_on"] = ["B", "A"]
        n1 = normalize(base)
        n2 = normalize(permuted)
        self.assertEqual(n1["digest"], n2["digest"])  # identical outputs

    def test_budget_prefix_selection(self):
        def with_budgets(bA, bB, bC):
            return {
                "schema": "2.0",
                "steps": [
                    {
                        "step_id": "A",
                        "handler": "h.a",
                        "metadata": {"cost": 1.0, "budget": bA, "gain_fn_id": "g"},
                    },
                    {
                        "step_id": "B",
                        "handler": "h.b",
                        "depends_on": ["A"],
                        "metadata": {"cost": 2.0, "budget": bB, "gain_fn_id": "g"},
                    },
                    {
                        "step_id": "C",
                        "handler": "h.c",
                        "depends_on": ["B"],
                        "metadata": {"cost": 3.0, "budget": bC, "gain_fn_id": "g"},
                    },
                ],
            }

        n_small = normalize(with_budgets(1.0, 1.0, None))  # selects A only
        n_big = normalize(with_budgets(1.0, 2.0, 3.0))  # selects A, B, C
        sel_small = n_small["selected_step_ids"]
        sel_big = n_big["selected_step_ids"]
        # prefix property
        self.assertTrue(sel_big[: len(sel_small)] == sel_small)
        # order respects topo
        self.assertEqual(
            sel_small, [s for s in n_small["topological_order"] if s in sel_small]
        )
        self.assertEqual(
            sel_big, [s for s in n_big["topological_order"] if s in sel_big]
        )

    def test_same_snapshot_id_same_digest(self):
        d1 = {
            "schema": "2.0",
            "steps": [
                {
                    "step_id": "X",
                    "handler": "h.x",
                    "depends_on": ["A", "B"],
                    "metadata": {"requires_snapshot": True, "snapshot_id": "snap:0001"},
                },
                {"step_id": "A", "handler": "h.a", "depends_on": [], "metadata": {}},
                {"step_id": "B", "handler": "h.b", "depends_on": [], "metadata": {}},
            ],
        }
        d2 = copy.deepcopy(d1)
        # permute depends_on for X (commutative by default True in defaults)
        d2["steps"][0]["depends_on"] = ["B", "A"]
        n1 = normalize(d1)
        n2 = normalize(d2)
        self.assertEqual(n1["digest"], n2["digest"])  # same normalized digest


if __name__ == "__main__":
    unittest.main()

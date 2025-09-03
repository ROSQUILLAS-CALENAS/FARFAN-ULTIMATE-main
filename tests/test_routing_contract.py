import unittest
import sys
# # # from pathlib import Path  # Module not found  # Module not found  # Module not found

# Add project root to path for canonical imports
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# # # from egw_query_expansion.core.deterministic_router import (  # Module not found  # Module not found  # Module not found
    DeterministicRouter,
    RoutingContext,
)


class TestRoutingContract(unittest.TestCase):
    def test_pure_routing_fn_determinism(self):
        router = DeterministicRouter()
        context = RoutingContext.from_query("q", [0.1, 0.2, 0.3], 1234, "hybrid")
        steps = [
            {"step_id": "A", "content": {"k": 1}},
            {"step_id": "B", "content": {"k": 2}},
            {"step_id": "C", "content": {"k": 3}},
        ]
        # Run multiple times and with fake parallelism hints
        routes = []
        for _ in range(100):
            routes.append(tuple(router.routing_fn(context, steps)))
        self.assertEqual(len(set(routes)), 1)

    def test_tiebreak_lexicographic(self):
        router = DeterministicRouter()
        context = RoutingContext.from_query("q", [0.1, 0.2], 10, "hybrid")
        # Same contents but different step_id should break ties by step_id
        steps = [
            {"step_id": "b", "content": {"x": 1}},
            {"step_id": "a", "content": {"x": 1}},
        ]
        route = router.routing_fn(context, steps)
        # content_hash equal -> step_id lexicographic decides: 'a' before 'b'
        self.assertEqual(route, ["a", "b"])


if __name__ == "__main__":
    unittest.main()

import unittest
import sys
# # # from pathlib import Path  # Module not found  # Module not found  # Module not found

# Add project root to path for canonical imports
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# # # from snapshot_manager import (  # Module not found  # Module not found  # Module not found
    get_current_snapshot_id,
    mount_snapshot,
    replay_output,
    requires_snapshot,
    resolve_snapshot,
)


class TestSnapshotContract(unittest.TestCase):
    def test_mount_and_resolve_snapshot(self):
        state = {
            "corpus": {"docs": ["a", "b"]},
            "indices": {"dim": 3},
            "standards": {"version": 1},
        }
        sid = mount_snapshot(state)
        self.assertEqual(get_current_snapshot_id(), sid)
        ro = resolve_snapshot(sid)
        self.assertIn("frozen_json", ro)
        # Recompute resolve must produce identical frozen_json
        ro2 = resolve_snapshot(sid)
        self.assertEqual(ro["frozen_json"], ro2["frozen_json"])

    def test_requires_snapshot_decorator_and_replay(self):
        # Define a simple handler
        @requires_snapshot
        def handler(x: int, y: int, snapshot_id: str = None):
            # Use snapshot_id but behave deterministically
            return {"sum": x + y, "sid": snapshot_id[:8] if snapshot_id else None}

        # Ensure snapshot exists
        sid = get_current_snapshot_id()
        if not sid:
            sid = mount_snapshot({"corpus": {}, "indices": {}, "standards": {}})
        out1 = handler(2, 3)
        out2 = handler(2, 3)
        self.assertEqual(out1, out2)
        # Replay bytes must be identical
        replay1 = replay_output(
            sid, lambda **kw: handler(**kw), inputs={"x": 2, "y": 3}
        )
        replay2 = replay_output(
            sid, lambda **kw: handler(**kw), inputs={"x": 2, "y": 3}
        )
        self.assertEqual(replay1, replay2)


if __name__ == "__main__":
    unittest.main()

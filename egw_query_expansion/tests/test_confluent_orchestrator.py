"""
Tests for Confluent Orchestrator

Tests the deterministic concurrent orchestrator based on the confluent actor model.
Verifies reproducibility across different parallelism degrees and scheduling policies.
"""

import asyncio
import random
# # # from concurrent.futures import ThreadPoolExecutor  # Module not found  # Module not found  # Module not found
# # # from typing import List  # Module not found  # Module not found  # Module not found

import pytest

# # # from egw_query_expansion.core.confluent_orchestrator import (  # Module not found  # Module not found  # Module not found
    ConfluentOrchestrator,
    NodeType,
    TaskNode,
    associative_commutative_max,
    associative_commutative_sum,
    deterministic_merge,
)


class TestConfluentOrchestrator:
    """Test suite for ConfluentOrchestrator"""

    def test_node_creation(self):
        """Test basic node creation"""
        node = TaskNode("test", NodeType.SOURCE, lambda x: x)
        assert node.id == "test"
        assert node.node_type == NodeType.SOURCE
        assert node.dependencies == set()

    def test_dag_construction(self):
        """Test DAG construction without cycles"""
        orchestrator = ConfluentOrchestrator()

        source = TaskNode("source", NodeType.SOURCE, lambda x: [1, 2, 3])
        transform = TaskNode(
            "transform", NodeType.TRANSFORM, lambda x: sum(x), dependencies={"source"}
        )

        orchestrator.add_node(source)
        orchestrator.add_node(transform)

        assert "source" in orchestrator.actors
        assert "transform" in orchestrator.actors
        assert "transform" in orchestrator.dag["source"]

    def test_cycle_detection(self):
        """Test cycle detection in DAG"""
        orchestrator = ConfluentOrchestrator()

        node1 = TaskNode(
            "node1", NodeType.TRANSFORM, lambda x: x, dependencies={"node2"}
        )
        node2 = TaskNode(
            "node2", NodeType.TRANSFORM, lambda x: x, dependencies={"node1"}
        )

        orchestrator.add_node(node1)

        with pytest.raises(ValueError, match="would create a cycle"):
            orchestrator.add_node(node2)

    @pytest.mark.asyncio
    async def test_simple_execution(self):
        """Test simple linear execution"""
        orchestrator = ConfluentOrchestrator()

        source = TaskNode("source", NodeType.SOURCE, lambda x: 10)
        transform = TaskNode(
            "transform", NodeType.TRANSFORM, lambda x: x * 2, dependencies={"source"}
        )

        orchestrator.add_node(source)
        orchestrator.add_node(transform)

        results = await orchestrator.execute()

        assert results["results"]["source"] == 10
        assert results["results"]["transform"] == 20
        assert "execution_time" in results
        assert results["error_count"] == 0

    @pytest.mark.asyncio
    async def test_barrier_synchronization(self):
        """Test barrier synchronization for join points"""
        orchestrator = ConfluentOrchestrator()

        source1 = TaskNode("source1", NodeType.SOURCE, lambda x: [1, 2])
        source2 = TaskNode("source2", NodeType.SOURCE, lambda x: [3, 4])

        reducer = TaskNode(
            "reducer",
            NodeType.REDUCER,
            lambda inputs: sum(sum(inp) for inp in inputs),
            dependencies={"source1", "source2"},
            is_associative_commutative=True,
        )

        orchestrator.add_node(source1)
        orchestrator.add_node(source2)
        orchestrator.add_node(reducer)

        results = await orchestrator.execute()

        assert results["results"]["source1"] == [1, 2]
        assert results["results"]["source2"] == [3, 4]
        assert results["results"]["reducer"] == 10  # (1+2) + (3+4) = 10
        assert "reducer" in orchestrator.barrier_times

    @pytest.mark.asyncio
    async def test_deterministic_seeds(self):
        """Test deterministic seed generation"""
        orchestrator1 = ConfluentOrchestrator()
        orchestrator2 = ConfluentOrchestrator()

        # Same node IDs should generate same seeds
        node1_orch1 = TaskNode("test_node", NodeType.SOURCE, lambda x: random.random())
        node1_orch2 = TaskNode("test_node", NodeType.SOURCE, lambda x: random.random())

        orchestrator1.add_node(node1_orch1)
        orchestrator2.add_node(node1_orch2)

        seed1 = orchestrator1.actors["test_node"].node.seed
        seed2 = orchestrator2.actors["test_node"].node.seed

        assert seed1 == seed2, "Same node IDs should generate same seeds"

    @pytest.mark.asyncio
    async def test_reproducibility_across_parallelism(self):
        """Test reproducibility across different parallelism degrees"""

        def create_test_dag(orchestrator):
            source1 = TaskNode("source1", NodeType.SOURCE, lambda x: [1, 2, 3])
            source2 = TaskNode("source2", NodeType.SOURCE, lambda x: [4, 5, 6])

            transform1 = TaskNode(
                "transform1",
                NodeType.TRANSFORM,
                lambda x: [i * 2 for i in x],
                dependencies={"source1"},
            )

            transform2 = TaskNode(
                "transform2",
                NodeType.TRANSFORM,
                lambda x: [i + 1 for i in x],
                dependencies={"source2"},
            )

            reducer = TaskNode(
                "reducer",
                NodeType.REDUCER,
                lambda inputs: sorted([item for sublist in inputs for item in sublist]),
                dependencies={"transform1", "transform2"},
                is_associative_commutative=False,
                pre_order_inputs=True,
            )

            orchestrator.add_node(source1)
            orchestrator.add_node(source2)
            orchestrator.add_node(transform1)
            orchestrator.add_node(transform2)
            orchestrator.add_node(reducer)

        # Test with different parallelism degrees
        results_parallel_2 = None
        results_parallel_4 = None

        # Parallelism degree 2
        orch1 = ConfluentOrchestrator(max_workers=2)
        create_test_dag(orch1)
        results_parallel_2 = await orch1.execute()

        # Parallelism degree 4
        orch2 = ConfluentOrchestrator(max_workers=4)
        create_test_dag(orch2)
        results_parallel_4 = await orch2.execute()

        # Results should be identical
        assert (
            results_parallel_2["results"]["reducer"]
            == results_parallel_4["results"]["reducer"]
        )
        assert (
            results_parallel_2["results"]["transform1"]
            == results_parallel_4["results"]["transform1"]
        )
        assert (
            results_parallel_2["results"]["transform2"]
            == results_parallel_4["results"]["transform2"]
        )

    @pytest.mark.asyncio
    async def test_associative_commutative_reducers(self):
        """Test associative and commutative reducers"""
        orchestrator = ConfluentOrchestrator()

        sources = []
        for i in range(5):
            source = TaskNode(f"source{i}", NodeType.SOURCE, lambda x: [i + 1])
            sources.append(source)
            orchestrator.add_node(source)

        sum_reducer = TaskNode(
            "sum_reducer",
            NodeType.REDUCER,
            associative_commutative_sum,
            dependencies={f"source{i}" for i in range(5)},
            is_associative_commutative=True,
        )

        max_reducer = TaskNode(
            "max_reducer",
            NodeType.REDUCER,
            associative_commutative_max,
            dependencies={f"source{i}" for i in range(5)},
            is_associative_commutative=True,
        )

        orchestrator.add_node(sum_reducer)
        orchestrator.add_node(max_reducer)

        results = await orchestrator.execute()

        # Sum should be 1+2+3+4+5 = 15
        assert results["results"]["sum_reducer"] == 15
        # Max should be 5
        assert results["results"]["max_reducer"] == 5

    @pytest.mark.asyncio
    async def test_crdt_state_merge(self):
        """Test CvRDT state merging"""
# # #         from egw_query_expansion.core.confluent_orchestrator import CvRDTState  # Module not found  # Module not found  # Module not found

        state1 = CvRDTState()
        state1.version_vector = {"actor1": 2, "actor2": 1}
        state1.operations = [("actor1", "op1", 1), ("actor2", "op2", 2)]

        state2 = CvRDTState()
        state2.version_vector = {"actor1": 1, "actor3": 3}
        state2.operations = [("actor1", "op3", 3), ("actor3", "op4", 4)]

        merged = state1.merge(state2)

        assert merged.version_vector["actor1"] == 2  # max(2, 1)
        assert merged.version_vector["actor2"] == 1
        assert merged.version_vector["actor3"] == 3

        # Operations should be merged and sorted deterministically
        assert len(merged.operations) == 4

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling and recording"""
        orchestrator = ConfluentOrchestrator()

        def failing_function(x):
            raise ValueError("Test error")

        source = TaskNode("source", NodeType.SOURCE, lambda x: 10)
        failing_transform = TaskNode(
            "failing", NodeType.TRANSFORM, failing_function, dependencies={"source"}
        )

        orchestrator.add_node(source)
        orchestrator.add_node(failing_transform)

        results = await orchestrator.execute()

        assert results["error_count"] > 0
        assert len(orchestrator.error_log) > 0
        assert orchestrator.error_log[0][0] == "failing"

    def test_performance_report(self):
        """Test performance reporting functionality"""
        orchestrator = ConfluentOrchestrator(
            max_workers=8, scheduling_policy="priority"
        )

        source = TaskNode("source", NodeType.SOURCE, lambda x: 1)
        orchestrator.add_node(source)

        report = orchestrator.get_performance_report()

        assert report["worker_config"]["parallelism_degree"] == 8
        assert report["worker_config"]["scheduling_policy"] == "priority"
        assert report["execution_stats"]["total_nodes"] == 1
        assert report["determinism_guarantees"]["dag_acyclic"] == True

    @pytest.mark.asyncio
    async def test_reset_functionality(self):
        """Test orchestrator reset functionality"""
        orchestrator = ConfluentOrchestrator()

        source = TaskNode("source", NodeType.SOURCE, lambda x: 42)
        orchestrator.add_node(source)

        # Execute once
        results1 = await orchestrator.execute()
        assert len(results1["results"]) == 1

        # Reset and execute again
        orchestrator.reset()
        results2 = await orchestrator.execute()

        # Results should be the same after reset
        assert results1["results"] == results2["results"]
        assert len(orchestrator.execution_results) == 1

    def test_deterministic_merge_utility(self):
        """Test deterministic merge utility function"""
        inputs = [{"a": 1, "b": 2}, {"b": 3, "c": 4}, {"a": 0, "d": 5}]

        result = deterministic_merge(inputs)

        # Should merge deterministically with max conflict resolution for numbers
        assert result["a"] == 1  # max(1, 0)
        assert result["b"] == 3  # max(2, 3)
        assert result["c"] == 4
        assert result["d"] == 5

    @pytest.mark.asyncio
    async def test_complex_dag_execution(self):
        """Test complex DAG with multiple levels and joins"""
        orchestrator = ConfluentOrchestrator()

        # Layer 1: Sources
        s1 = TaskNode("s1", NodeType.SOURCE, lambda x: 1)
        s2 = TaskNode("s2", NodeType.SOURCE, lambda x: 2)
        s3 = TaskNode("s3", NodeType.SOURCE, lambda x: 3)

        # Layer 2: Transforms
        t1 = TaskNode("t1", NodeType.TRANSFORM, lambda x: x * 10, dependencies={"s1"})
        t2 = TaskNode("t2", NodeType.TRANSFORM, lambda x: x * 20, dependencies={"s2"})

        # Layer 3: Join and final transform
        r1 = TaskNode(
            "r1",
            NodeType.REDUCER,
            lambda inputs: sum(inputs),
            dependencies={"t1", "t2", "s3"},
            is_associative_commutative=True,
        )

        final = TaskNode(
            "final", NodeType.TRANSFORM, lambda x: x / 2, dependencies={"r1"}
        )

        nodes = [s1, s2, s3, t1, t2, r1, final]
        for node in nodes:
            orchestrator.add_node(node)

        results = await orchestrator.execute()

        # Expected: (1*10 + 2*20 + 3) / 2 = (10 + 40 + 3) / 2 = 53 / 2 = 26.5
        assert results["results"]["final"] == 26.5
        assert len(results["results"]) == len(nodes)

    @pytest.mark.asyncio
    async def test_scheduling_reproducibility(self):
        """Test reproducibility across different scheduling policies"""

        def create_test_setup(scheduling_policy):
            orch = ConfluentOrchestrator(scheduling_policy=scheduling_policy)

            source1 = TaskNode("source1", NodeType.SOURCE, lambda x: [1, 2])
            source2 = TaskNode("source2", NodeType.SOURCE, lambda x: [3, 4])
            source3 = TaskNode("source3", NodeType.SOURCE, lambda x: [5, 6])

            reducer = TaskNode(
                "reducer",
                NodeType.REDUCER,
                lambda inputs: sum(sum(inp) for inp in inputs),
                dependencies={"source1", "source2", "source3"},
                is_associative_commutative=True,
            )

            for node in [source1, source2, source3, reducer]:
                orch.add_node(node)

            return orch

        # Test different scheduling policies
        orch_round_robin = create_test_setup("round_robin")
        orch_priority = create_test_setup("priority")

        results_rr = await orch_round_robin.execute()
        results_priority = await orch_priority.execute()

        # Results should be identical regardless of scheduling policy
        assert (
            results_rr["results"]["reducer"] == results_priority["results"]["reducer"]
        )
        assert results_rr["results"]["reducer"] == 21  # (1+2) + (3+4) + (5+6) = 21


if __name__ == "__main__":
    pytest.main([__file__])

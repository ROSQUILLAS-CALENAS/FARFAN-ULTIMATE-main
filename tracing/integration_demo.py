"""
Integration Demo for DAG Observability System

This demo shows how to integrate the DAG observability system into existing pipelines
and showcases all the key features including tracing, heatmap generation, and 
circular dependency detection.
"""

import asyncio
import random
import time
import uuid
from pathlib import Path
import logging
from typing import Dict, Any, List

from dag_observability import (
    initialize_dag_observability, 
    get_dag_observability,
    trace_edge_traversal,
    trace_component_execution,
    trace_phase_transition
)
from visualization import create_performance_dashboard

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockPipelineComponent:
    """Mock component for demonstration purposes"""
    
    def __init__(self, name: str, processing_time_range: tuple = (0.1, 2.0)):
        self.name = name
        self.processing_time_range = processing_time_range
        self.error_rate = 0.05  # 5% error rate
    
    async def process(self, data: Dict[str, Any], correlation_id: str = None) -> Dict[str, Any]:
        """Simulate component processing with random delays and occasional errors"""
        
        # Generate correlation ID if not provided
        if not correlation_id:
            correlation_id = str(uuid.uuid4())
        
        with trace_component_execution(self.name, "processing", correlation_id):
            # Simulate phase transitions
            with trace_phase_transition(self.name, "init", "processing"):
                await asyncio.sleep(0.1)
            
            # Main processing
            processing_time = random.uniform(*self.processing_time_range)
            await asyncio.sleep(processing_time)
            
            # Simulate occasional errors
            if random.random() < self.error_rate:
                raise RuntimeError(f"Simulated error in {self.name}")
            
            with trace_phase_transition(self.name, "processing", "finalization"):
                await asyncio.sleep(0.05)
            
            # Return processed data
            return {
                "component": self.name,
                "processed_data": data,
                "processing_time": processing_time,
                "correlation_id": correlation_id
            }


class MockDAGPipeline:
    """Mock DAG pipeline demonstrating observability integration"""
    
    def __init__(self):
        # Create mock components
        self.components = {
            "data_ingestion": MockPipelineComponent("data_ingestion", (0.2, 0.8)),
            "preprocessing": MockPipelineComponent("preprocessing", (0.3, 1.2)),
            "feature_extraction": MockPipelineComponent("feature_extraction", (0.5, 1.5)),
            "model_inference": MockPipelineComponent("model_inference", (0.8, 2.0)),
            "post_processing": MockPipelineComponent("post_processing", (0.1, 0.5)),
            "result_aggregation": MockPipelineComponent("result_aggregation", (0.2, 0.7))
        }
        
        # Define DAG structure (adjacency list)
        self.dag_structure = {
            "data_ingestion": ["preprocessing"],
            "preprocessing": ["feature_extraction", "post_processing"],
            "feature_extraction": ["model_inference"],
            "model_inference": ["result_aggregation"],
            "post_processing": ["result_aggregation"],
            "result_aggregation": []
        }
        
        # Track processing statistics
        self.processed_items = 0
        self.errors = 0
    
    async def execute_component(self, component_name: str, data: Dict[str, Any], 
                              correlation_id: str) -> Dict[str, Any]:
        """Execute a single component"""
        component = self.components[component_name]
        try:
            result = await component.process(data, correlation_id)
            return result
        except Exception as e:
            self.errors += 1
            logger.error(f"Error in component {component_name}: {e}")
            raise
    
    async def traverse_edge(self, source: str, target: str, data: Dict[str, Any],
                          correlation_id: str) -> Dict[str, Any]:
        """Traverse an edge between two components"""
        with trace_edge_traversal(source, target, correlation_id):
            # Simulate edge traversal overhead
            await asyncio.sleep(0.01)
            return await self.execute_component(target, data, correlation_id)
    
    async def process_item(self, item_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single item through the entire DAG"""
        correlation_id = str(uuid.uuid4())
        logger.info(f"Processing item with correlation ID: {correlation_id}")
        
        try:
            # Start with data ingestion
            result = await self.execute_component("data_ingestion", item_data, correlation_id)
            current_data = result
            
            # Process through DAG using BFS-like traversal
            visited = {"data_ingestion"}
            queue = [("data_ingestion", current_data)]
            final_results = {}
            
            while queue:
                current_component, data = queue.pop(0)
                
                # Get next components
                next_components = self.dag_structure.get(current_component, [])
                
                for next_comp in next_components:
                    if next_comp not in visited:
                        try:
                            # Traverse edge and execute component
                            result = await self.traverse_edge(
                                current_component, next_comp, data, correlation_id
                            )
                            
                            visited.add(next_comp)
                            queue.append((next_comp, result))
                            
                            # Store final results
                            if not self.dag_structure[next_comp]:  # Leaf node
                                final_results[next_comp] = result
                                
                        except Exception as e:
                            logger.error(f"Failed to process {next_comp}: {e}")
                            # Continue with other branches
                            continue
            
            self.processed_items += 1
            return final_results
            
        except Exception as e:
            self.errors += 1
            logger.error(f"Pipeline execution failed: {e}")
            raise
    
    def simulate_circular_dependency(self):
        """Simulate a circular dependency scenario for testing detection"""
        logger.info("Simulating circular dependency...")
        
        try:
            # This should trigger circular dependency detection
            with trace_component_execution("component_a"):
                with trace_component_execution("component_b"):
                    with trace_component_execution("component_c"):
                        # This will create a cycle: a -> b -> c -> a
                        with trace_component_execution("component_a"):
                            pass
        except RuntimeError as e:
            logger.info(f"Circular dependency detected as expected: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        return {
            "processed_items": self.processed_items,
            "errors": self.errors,
            "error_rate": self.errors / max(1, self.processed_items + self.errors),
            "components": list(self.components.keys()),
            "dag_structure": self.dag_structure
        }


async def run_demo():
    """Run the comprehensive DAG observability demo"""
    logger.info("Starting DAG Observability System Demo")
    
    # Initialize observability system
    obs_system = initialize_dag_observability(
        service_name="mock_dag_pipeline",
        otlp_endpoint=None,  # Use None for local development
        json_fallback_path="tracing/demo_spans.json",
        enable_heatmap=True,
        enable_circular_detection=True
    )
    
    # Create mock pipeline
    pipeline = MockDAGPipeline()
    
    # Demo 1: Normal pipeline execution
    logger.info("=== Demo 1: Normal Pipeline Execution ===")
    
    # Process multiple items to generate meaningful metrics
    tasks = []
    for i in range(20):
        item_data = {
            "id": f"item_{i}",
            "content": f"Sample content for item {i}",
            "timestamp": time.time()
        }
        task = pipeline.process_item(item_data)
        tasks.append(task)
        
        # Add some delay between submissions
        if i % 5 == 0:
            await asyncio.sleep(0.5)
    
    # Execute all tasks concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Count successful vs failed executions
    successful = sum(1 for r in results if not isinstance(r, Exception))
    failed = len(results) - successful
    
    logger.info(f"Processed {len(results)} items: {successful} successful, {failed} failed")
    
    # Demo 2: Circular dependency detection
    logger.info("=== Demo 2: Circular Dependency Detection ===")
    pipeline.simulate_circular_dependency()
    
    # Demo 3: Generate performance reports
    logger.info("=== Demo 3: Generating Performance Reports ===")
    
    # Wait a bit for metrics to accumulate
    await asyncio.sleep(2)
    
    # Get observability data
    heatmap_data = obs_system.get_heatmap_data()
    violations = obs_system.get_circular_dependency_violations()
    dependency_graph = obs_system.get_dependency_graph()
    
    # Print summary
    logger.info(f"Generated {len(heatmap_data.get('hot_paths', []))} hot path entries")
    logger.info(f"Detected {len(violations)} circular dependency violations")
    logger.info(f"Mapped {len(dependency_graph)} component dependencies")
    
    # Create visualizations
    dashboard_file = create_performance_dashboard(
        heatmap_data=heatmap_data,
        violations=violations,
        dependency_graph=dependency_graph,
        output_dir="tracing/demo_visualizations"
    )
    
    if dashboard_file:
        logger.info(f"Performance dashboard created: {dashboard_file}")
    
    # Demo 4: Real-time monitoring
    logger.info("=== Demo 4: Real-time Monitoring ===")
    
    # Process items continuously for a short period
    start_time = time.time()
    continuous_items = 0
    
    while time.time() - start_time < 30:  # Run for 30 seconds
        item_data = {
            "id": f"realtime_item_{continuous_items}",
            "content": f"Real-time content {continuous_items}",
            "timestamp": time.time()
        }
        
        try:
            await pipeline.process_item(item_data)
            continuous_items += 1
            
            # Check for alerts
            alerts = obs_system.get_alerts()
            if alerts:
                logger.warning(f"Received {len(alerts)} alerts")
                for alert in alerts:
                    logger.warning(f"Alert: {alert}")
            
            await asyncio.sleep(1)  # Process one item per second
            
        except Exception as e:
            logger.error(f"Error in real-time processing: {e}")
            continue
    
    logger.info(f"Real-time monitoring processed {continuous_items} items")
    
    # Final statistics
    stats = pipeline.get_statistics()
    logger.info("=== Final Statistics ===")
    for key, value in stats.items():
        logger.info(f"{key}: {value}")
    
    # Generate final reports
    final_heatmap = obs_system.get_heatmap_data()
    final_violations = obs_system.get_circular_dependency_violations()
    
    logger.info(f"Final metrics - Hot paths: {len(final_heatmap.get('hot_paths', []))}")
    logger.info(f"Final violations: {len(final_violations)}")
    
    # Save final data
    import json
    
    final_report = {
        "demo_summary": {
            "total_items_processed": stats["processed_items"],
            "total_errors": stats["errors"],
            "error_rate": stats["error_rate"],
            "demo_duration_seconds": time.time() - start_time
        },
        "heatmap_data": final_heatmap,
        "violations": [
            {
                "id": v.violation_id,
                "timestamp": v.timestamp.isoformat(),
                "severity": v.severity,
                "cycle": v.execution_stack,
                "resolved": v.resolved
            }
            for v in final_violations
        ],
        "dependency_graph": dependency_graph
    }
    
    report_file = Path("tracing/demo_final_report.json")
    with open(report_file, 'w') as f:
        json.dump(final_report, f, indent=2, default=str)
    
    logger.info(f"Final report saved to {report_file}")
    
    # Cleanup
    logger.info("Demo completed. Shutting down observability system...")
    obs_system.stop()


if __name__ == "__main__":
    # Run the demo
    asyncio.run(run_demo())
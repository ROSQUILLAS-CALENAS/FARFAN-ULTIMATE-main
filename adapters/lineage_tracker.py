"""
Lineage Tracker

Tracks component dependencies and violations for dependency monitoring.
Routes lineage events to monitoring systems and maintains dependency graphs.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Set
from datetime import datetime, timedelta
from collections import defaultdict
from .data_transfer_objects import LineageEvent


class LineageTracker:
    """Tracks component lineage and dependency violations"""
    
    def __init__(self, monitoring_endpoint: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.monitoring_endpoint = monitoring_endpoint
        
        # Event storage
        self.lineage_events: List[LineageEvent] = []
        self.dependency_graph: Dict[str, Set[str]] = defaultdict(set)
        self.violation_history: List[LineageEvent] = []
        
        # Configure lineage logging
        self._setup_lineage_logging()
    
    def _setup_lineage_logging(self):
        """Configure dedicated lineage tracking logging"""
        lineage_handler = logging.FileHandler('logs/dependency_lineage.log')
        lineage_handler.setFormatter(
            logging.Formatter(
                '%(asctime)s - %(levelname)s - LINEAGE - %(message)s'
            )
        )
        
        lineage_logger = logging.getLogger('lineage_tracker')
        lineage_logger.addHandler(lineage_handler)
        lineage_logger.setLevel(logging.INFO)
        
        self.lineage_logger = lineage_logger
    
    def track_component_operation(
        self,
        component_id: str,
        operation_type: str,
        input_schema: str,
        output_schema: str,
        dependencies: List[str]
    ) -> LineageEvent:
        """Track a component operation for lineage"""
        
        event = LineageEvent(
            component_id=component_id,
            operation_type=operation_type,
            input_schema=input_schema,
            output_schema=output_schema,
            dependencies=dependencies
        )
        
        self.lineage_events.append(event)
        
        # Update dependency graph
        for dependency in dependencies:
            self.dependency_graph[component_id].add(dependency)
        
        # Log lineage event
        lineage_msg = {
            'component_id': component_id,
            'operation_type': operation_type,
            'input_schema': input_schema,
            'output_schema': output_schema,
            'dependencies': dependencies,
            'timestamp': event.timestamp.isoformat()
        }
        
        self.lineage_logger.info(json.dumps(lineage_msg))
        
        # Send to monitoring endpoint if configured
        if self.monitoring_endpoint:
            self._send_to_monitoring(lineage_msg)
        
        return event
    
    def track_dependency_violation(self, event: LineageEvent):
        """Track a dependency violation event"""
        
        self.violation_history.append(event)
        
        # Log violation
        violation_msg = {
            'component_id': event.component_id,
            'violation_type': event.violation_type,
            'operation_type': event.operation_type,
            'dependencies': event.dependencies,
            'timestamp': event.timestamp.isoformat()
        }
        
        self.lineage_logger.error(f"DEPENDENCY_VIOLATION: {json.dumps(violation_msg)}")
        
        # Send to monitoring endpoint
        if self.monitoring_endpoint:
            self._send_to_monitoring(violation_msg, level='error')
    
    def detect_circular_dependencies(self) -> List[List[str]]:
        """Detect circular dependencies in the dependency graph"""
        
        def dfs(node, path, visited, rec_stack):
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for neighbor in self.dependency_graph.get(node, []):
                if neighbor in rec_stack:
                    # Found cycle
                    cycle_start = path.index(neighbor)
                    return path[cycle_start:] + [neighbor]
                
                if neighbor not in visited:
                    result = dfs(neighbor, path, visited, rec_stack)
                    if result:
                        return result
            
            path.pop()
            rec_stack.remove(node)
            return None
        
        cycles = []
        visited = set()
        
        for node in self.dependency_graph:
            if node not in visited:
                cycle = dfs(node, [], visited, set())
                if cycle:
                    cycles.append(cycle)
        
        return cycles
    
    def get_component_lineage(self, component_id: str) -> Dict[str, Any]:
        """Get complete lineage for a component"""
        
        # Get direct dependencies
        direct_deps = list(self.dependency_graph.get(component_id, set()))
        
        # Get transitive dependencies
        transitive_deps = self._get_transitive_dependencies(component_id)
        
        # Get components that depend on this one
        dependents = [
            comp for comp, deps in self.dependency_graph.items()
            if component_id in deps
        ]
        
        # Get recent events for this component
        recent_events = [
            {
                'operation_type': event.operation_type,
                'input_schema': event.input_schema,
                'output_schema': event.output_schema,
                'timestamp': event.timestamp.isoformat()
            }
            for event in self.lineage_events
            if event.component_id == component_id
            and event.timestamp > datetime.now() - timedelta(hours=24)
        ]
        
        # Get violations for this component
        violations = [
            {
                'violation_type': event.violation_type,
                'operation_type': event.operation_type,
                'timestamp': event.timestamp.isoformat()
            }
            for event in self.violation_history
            if event.component_id == component_id
        ]
        
        return {
            'component_id': component_id,
            'direct_dependencies': sorted(direct_deps),
            'transitive_dependencies': sorted(transitive_deps),
            'dependents': sorted(dependents),
            'recent_events': recent_events,
            'violations': violations,
            'lineage_health': self._calculate_lineage_health(component_id)
        }
    
    def _get_transitive_dependencies(self, component_id: str, visited: Optional[Set[str]] = None) -> Set[str]:
        """Get all transitive dependencies for a component"""
        
        if visited is None:
            visited = set()
        
        if component_id in visited:
            return set()  # Avoid cycles
        
        visited.add(component_id)
        
        transitive = set()
        direct_deps = self.dependency_graph.get(component_id, set())
        
        for dep in direct_deps:
            transitive.add(dep)
            transitive.update(self._get_transitive_dependencies(dep, visited.copy()))
        
        return transitive
    
    def _calculate_lineage_health(self, component_id: str) -> Dict[str, Any]:
        """Calculate lineage health metrics for a component"""
        
        recent_violations = [
            v for v in self.violation_history
            if v.component_id == component_id
            and v.timestamp > datetime.now() - timedelta(hours=24)
        ]
        
        recent_operations = [
            e for e in self.lineage_events
            if e.component_id == component_id
            and e.timestamp > datetime.now() - timedelta(hours=24)
        ]
        
        # Calculate health score (0-100)
        health_score = 100
        
        # Penalize for recent violations
        health_score -= len(recent_violations) * 20
        
        # Penalize for excessive dependencies (coupling)
        dep_count = len(self.dependency_graph.get(component_id, set()))
        if dep_count > 5:
            health_score -= (dep_count - 5) * 5
        
        health_score = max(0, health_score)
        
        return {
            'health_score': health_score,
            'recent_violations': len(recent_violations),
            'recent_operations': len(recent_operations),
            'dependency_count': dep_count,
            'status': self._get_health_status(health_score)
        }
    
    def _get_health_status(self, score: int) -> str:
        """Get health status from score"""
        if score >= 80:
            return 'healthy'
        elif score >= 60:
            return 'warning'
        elif score >= 40:
            return 'degraded'
        else:
            return 'critical'
    
    def _send_to_monitoring(self, data: Dict[str, Any], level: str = 'info'):
        """Send lineage data to monitoring endpoint"""
        try:
            # In a real implementation, this would send to an actual monitoring system
            # For now, we'll just log that we would send it
            self.logger.info(f"Would send to monitoring ({level}): {json.dumps(data)}")
        except Exception as e:
            self.logger.error(f"Failed to send to monitoring: {e}")
    
    def get_system_lineage_summary(self) -> Dict[str, Any]:
        """Get overall system lineage summary"""
        
        # Detect cycles
        cycles = self.detect_circular_dependencies()
        
        # Count components and dependencies
        components = set(self.dependency_graph.keys())
        for deps in self.dependency_graph.values():
            components.update(deps)
        
        total_dependencies = sum(len(deps) for deps in self.dependency_graph.values())
        
        # Recent violations
        recent_violations = [
            v for v in self.violation_history
            if v.timestamp > datetime.now() - timedelta(hours=24)
        ]
        
        # Calculate system health
        unhealthy_components = [
            comp for comp in components
            if self._calculate_lineage_health(comp)['health_score'] < 60
        ]
        
        return {
            'total_components': len(components),
            'total_dependencies': total_dependencies,
            'circular_dependencies': len(cycles),
            'cycles_detected': cycles,
            'recent_violations': len(recent_violations),
            'unhealthy_components': len(unhealthy_components),
            'system_health': self._calculate_system_health(unhealthy_components, components),
            'timestamp': datetime.now().isoformat()
        }
    
    def _calculate_system_health(self, unhealthy_components: List[str], all_components: Set[str]) -> str:
        """Calculate overall system health"""
        
        if not all_components:
            return 'unknown'
        
        unhealthy_ratio = len(unhealthy_components) / len(all_components)
        
        if unhealthy_ratio == 0:
            return 'healthy'
        elif unhealthy_ratio < 0.2:
            return 'warning'
        elif unhealthy_ratio < 0.5:
            return 'degraded'
        else:
            return 'critical'
"""
Residual back-edge detection for canonical pipeline ordering violations
"""

import time
import logging
from typing import List, Dict, Set, Optional, Any
from dataclasses import dataclass
from collections import defaultdict

from .otel_tracer import get_pipeline_tracer, CANONICAL_PHASES, PHASE_ORDER


@dataclass
class DependencyViolation:
    """Represents a backward dependency violation"""
    violation_type: str  # 'direct_back_edge', 'cyclic_dependency', 'phase_skip'
    source_phase: str
    target_phase: str
    component_path: str
    timestamp: float
    severity: str  # 'critical', 'warning', 'info'
    dependency_path: List[str]
    description: str


class BackEdgeDetector:
    """Detects runtime backward dependencies that violate canonical phase ordering"""
    
    def __init__(self, log_level: str = "WARNING"):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Tracking for cycle detection
        self.dependency_graph: Dict[str, Set[str]] = defaultdict(set)
        self.violation_history: List[DependencyViolation] = []
        self.component_registry: Dict[str, str] = {}  # component -> phase mapping
        
    def analyze_span_traces(self, time_window_minutes: int = 60) -> List[DependencyViolation]:
        """Analyze OpenTelemetry span data for ordering violations"""
        tracer = get_pipeline_tracer()
        spans = tracer.get_span_data(time_window_minutes)
        
        violations = []
        
        # Build dependency graph from spans
        self._build_dependency_graph(spans)
        
        # Detect different types of violations
        violations.extend(self._detect_direct_back_edges(spans))
        violations.extend(self._detect_cyclic_dependencies(spans))
        violations.extend(self._detect_phase_ordering_violations(spans))
        violations.extend(self._detect_illegal_phase_skips(spans))
        
        # Store violations
        self.violation_history.extend(violations)
        
        # Log violations
        for violation in violations:
            self._log_violation(violation)
            
        return violations
        
    def _build_dependency_graph(self, spans):
        """Build dependency graph from span data"""
        self.dependency_graph.clear()
        self.component_registry.clear()
        
        for span in spans:
            source = span.source_phase
            target = span.target_phase
            component = span.component_name
            
            # Register component to phase mapping
            self.component_registry[component] = source
            
            # Add edge to dependency graph
            if source != target:  # Skip self-edges
                self.dependency_graph[source].add(target)
                
    def _detect_direct_back_edges(self, spans) -> List[DependencyViolation]:
        """Detect direct backward edges in canonical ordering"""
        violations = []
        
        for span in spans:
            source_order = PHASE_ORDER.get(span.source_phase, -1)
            target_order = PHASE_ORDER.get(span.target_phase, -1)
            
            # Check for direct backward edge
            if source_order > target_order and target_order >= 0:
                violation = DependencyViolation(
                    violation_type='direct_back_edge',
                    source_phase=span.source_phase,
                    target_phase=span.target_phase,
                    component_path=span.component_name,
                    timestamp=span.timing_start,
                    severity='critical',
                    dependency_path=[span.source_phase, span.target_phase],
                    description=f"Direct backward edge: {span.source_phase} -> {span.target_phase} "
                              f"violates canonical ordering I→X→K→A→L→R→O→G→T→S"
                )
                violations.append(violation)
                
        return violations
        
    def _detect_cyclic_dependencies(self, spans) -> List[DependencyViolation]:
        """Detect cyclic dependencies using DFS"""
        violations = []
        
        # DFS cycle detection
        visited = set()
        rec_stack = set()
        
        def dfs_cycle_detect(node, path):
            if node in rec_stack:
                # Found cycle
                cycle_start = path.index(node)
                cycle_path = path[cycle_start:] + [node]
                
                violation = DependencyViolation(
                    violation_type='cyclic_dependency',
                    source_phase=cycle_path[-2],
                    target_phase=cycle_path[-1], 
                    component_path=' -> '.join(cycle_path),
                    timestamp=time.time(),
                    severity='critical',
                    dependency_path=cycle_path,
                    description=f"Cyclic dependency detected: {' -> '.join(cycle_path)}"
                )
                violations.append(violation)
                return
                
            if node in visited:
                return
                
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in self.dependency_graph[node]:
                dfs_cycle_detect(neighbor, path + [neighbor])
                
            rec_stack.remove(node)
            
        # Check each phase for cycles
        for phase in CANONICAL_PHASES:
            if phase not in visited:
                dfs_cycle_detect(phase, [phase])
                
        return violations
        
    def _detect_phase_ordering_violations(self, spans) -> List[DependencyViolation]:
        """Detect violations of canonical phase ordering beyond direct back-edges"""
        violations = []
        
        # Track phase execution sequence
        phase_sequence = []
        for span in sorted(spans, key=lambda x: x.timing_start):
            if span.source_phase not in phase_sequence:
                phase_sequence.append(span.source_phase)
                
        # Check for ordering violations in the sequence
        for i in range(len(phase_sequence) - 1):
            current_phase = phase_sequence[i]
            next_phase = phase_sequence[i + 1]
            
            current_order = PHASE_ORDER.get(current_phase, -1)
            next_order = PHASE_ORDER.get(next_phase, -1)
            
            # Check if phases are out of canonical order
            if current_order >= 0 and next_order >= 0 and next_order < current_order:
                # Find the specific spans that caused this
                relevant_spans = [
                    s for s in spans 
                    if s.source_phase == next_phase and s.timing_start > 
                    max([span.timing_start for span in spans if span.source_phase == current_phase])
                ]
                
                for span in relevant_spans:
                    violation = DependencyViolation(
                        violation_type='phase_ordering_violation',
                        source_phase=current_phase,
                        target_phase=next_phase,
                        component_path=span.component_name,
                        timestamp=span.timing_start,
                        severity='warning',
                        dependency_path=phase_sequence[i:i+2],
                        description=f"Phase ordering violation: {current_phase} executed before {next_phase} "
                                  f"but canonical order is I→X→K→A→L→R→O→G→T→S"
                    )
                    violations.append(violation)
                    
        return violations
        
    def _detect_illegal_phase_skips(self, spans) -> List[DependencyViolation]:
        """Detect illegal skipping of canonical phases"""
        violations = []
        
        # Track which phases have been executed
        executed_phases = set(span.source_phase for span in spans)
        
        # Check for phase skips
        executed_orders = sorted([PHASE_ORDER[phase] for phase in executed_phases if phase in PHASE_ORDER])
        
        for i in range(len(executed_orders) - 1):
            current_order = executed_orders[i]
            next_order = executed_orders[i + 1]
            
            # Check if there's a gap (skipped phases)
            if next_order - current_order > 1:
                current_phase = CANONICAL_PHASES[current_order]
                next_phase = CANONICAL_PHASES[next_order]
                skipped_phases = CANONICAL_PHASES[current_order + 1:next_order]
                
                # Find the span that caused the skip
                relevant_span = next(
                    (s for s in spans if s.source_phase == next_phase), 
                    None
                )
                
                if relevant_span:
                    violation = DependencyViolation(
                        violation_type='phase_skip',
                        source_phase=current_phase,
                        target_phase=next_phase,
                        component_path=relevant_span.component_name,
                        timestamp=relevant_span.timing_start,
                        severity='info',
                        dependency_path=[current_phase] + skipped_phases + [next_phase],
                        description=f"Potential phase skip: {current_phase} -> {next_phase}, "
                                  f"skipped phases: {', '.join(skipped_phases)}"
                    )
                    violations.append(violation)
                    
        return violations
        
    def _log_violation(self, violation: DependencyViolation):
        """Log a dependency violation"""
        log_level = {
            'critical': logging.CRITICAL,
            'warning': logging.WARNING,
            'info': logging.INFO
        }.get(violation.severity, logging.WARNING)
        
        message = f"DEPENDENCY VIOLATION [{violation.violation_type}] " \
                 f"Component: {violation.component_path} | " \
                 f"Path: {' -> '.join(violation.dependency_path)} | " \
                 f"Description: {violation.description}"
                 
        self.logger.log(log_level, message)
        
    def get_violation_summary(self, time_window_minutes: int = 60) -> Dict[str, Any]:
        """Get summary of violations within time window"""
        cutoff_time = time.time() - (time_window_minutes * 60)
        recent_violations = [
            v for v in self.violation_history 
            if v.timestamp >= cutoff_time
        ]
        
        # Count by type and severity
        by_type = defaultdict(int)
        by_severity = defaultdict(int)
        by_component = defaultdict(int)
        
        for violation in recent_violations:
            by_type[violation.violation_type] += 1
            by_severity[violation.severity] += 1
            by_component[violation.component_path] += 1
            
        return {
            'total_violations': len(recent_violations),
            'by_type': dict(by_type),
            'by_severity': dict(by_severity),
            'by_component': dict(by_component),
            'recent_violations': [
                {
                    'type': v.violation_type,
                    'component': v.component_path,
                    'path': ' -> '.join(v.dependency_path),
                    'description': v.description,
                    'severity': v.severity,
                    'timestamp': v.timestamp
                }
                for v in sorted(recent_violations, key=lambda x: x.timestamp, reverse=True)[:20]
            ]
        }
        
    def clear_violation_history(self):
        """Clear violation history (for testing/reset)"""
        self.violation_history.clear()
        self.dependency_graph.clear()
        self.component_registry.clear()
"""
Live dependency heatmap visualization system
"""

import json
import statistics
from typing import Dict, List, Any, Optional
from collections import defaultdict, Counter
import time

from .otel_tracer import get_pipeline_tracer, PhaseTransition, CANONICAL_PHASES


class DependencyHeatmapVisualizer:
    """Generates real-time HTML dashboards from OpenTelemetry span data"""
    
    def __init__(self, refresh_interval_seconds: int = 5):
        self.refresh_interval = refresh_interval_seconds
        
    def generate_heatmap_data(self, time_window_minutes: int = 60) -> Dict[str, Any]:
        """Generate heatmap data from span traces"""
        tracer = get_pipeline_tracer()
        
        # Get recent span data
        spans = tracer.get_span_data(time_window_minutes)
        transitions = tracer.get_phase_transitions(time_window_minutes)
        
        # Calculate edge frequencies
        edge_frequencies = Counter()
        edge_latencies = defaultdict(list)
        phase_activity = defaultdict(int)
        error_counts = defaultdict(int)
        
        for span in spans:
            edge_key = f"{span.source_phase}->{span.target_phase}"
            edge_frequencies[edge_key] += 1
            phase_activity[span.source_phase] += 1
            phase_activity[span.target_phase] += 1
            
            if span.timing_end:
                latency = span.timing_end - span.timing_start
                edge_latencies[edge_key].append(latency)
                
            if span.error:
                error_counts[edge_key] += 1
                
        # Calculate latency statistics
        edge_latency_stats = {}
        for edge_key, latencies in edge_latencies.items():
            if latencies:
                edge_latency_stats[edge_key] = {
                    'min': min(latencies),
                    'max': max(latencies),
                    'mean': statistics.mean(latencies),
                    'median': statistics.median(latencies),
                    'p95': statistics.quantiles(latencies, n=20)[18] if len(latencies) > 1 else latencies[0],
                    'p99': statistics.quantiles(latencies, n=100)[98] if len(latencies) > 1 else latencies[0]
                }
                
        # Phase transition patterns
        transition_patterns = self._analyze_transition_patterns(transitions)
        
        return {
            'timestamp': time.time(),
            'time_window_minutes': time_window_minutes,
            'edge_frequencies': dict(edge_frequencies),
            'edge_latency_stats': edge_latency_stats,
            'phase_activity': dict(phase_activity),
            'error_counts': dict(error_counts),
            'transition_patterns': transition_patterns,
            'total_spans': len(spans),
            'canonical_phases': CANONICAL_PHASES
        }
        
    def _analyze_transition_patterns(self, transitions: List[PhaseTransition]) -> Dict[str, Any]:
        """Analyze phase transition patterns"""
        patterns = {
            'sequential_flows': [],
            'parallel_flows': [],
            'bottlenecks': [],
            'phase_timing': defaultdict(list)
        }
        
        # Group transitions by timestamp windows
        time_windows = defaultdict(list)
        window_size = 1.0  # 1 second windows
        
        for transition in transitions:
            window = int(transition.timestamp / window_size)
            time_windows[window].append(transition)
            patterns['phase_timing'][transition.to_phase].append(transition.duration)
            
        # Detect sequential vs parallel flows
        for window_transitions in time_windows.values():
            if len(window_transitions) == 1:
                patterns['sequential_flows'].append(window_transitions[0].component)
            else:
                patterns['parallel_flows'].append([t.component for t in window_transitions])
                
        # Calculate phase timing statistics
        phase_timing_stats = {}
        for phase, timings in patterns['phase_timing'].items():
            if timings:
                phase_timing_stats[phase] = {
                    'mean': statistics.mean(timings),
                    'median': statistics.median(timings),
                    'count': len(timings)
                }
                
        patterns['phase_timing_stats'] = phase_timing_stats
        
        return patterns
        
    def generate_html_dashboard(self, time_window_minutes: int = 60) -> str:
        """Generate complete HTML dashboard"""
        data = self.generate_heatmap_data(time_window_minutes)
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Pipeline Dependency Heatmap</title>
    <meta charset="utf-8">
    <meta http-equiv="refresh" content="{self.refresh_interval}">
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .dashboard {{ max-width: 1400px; margin: 0 auto; }}
        .header {{ background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-bottom: 20px; }}
        .metric-card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .metric-title {{ font-size: 18px; font-weight: bold; margin-bottom: 15px; color: #333; }}
        .heatmap {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .heatmap-row {{ display: flex; align-items: center; margin-bottom: 10px; }}
        .phase-label {{ min-width: 200px; font-weight: bold; }}
        .heatmap-cells {{ display: flex; flex-wrap: wrap; }}
        .heatmap-cell {{ width: 40px; height: 40px; margin: 2px; border-radius: 4px; display: flex; align-items: center; justify-content: center; color: white; font-size: 12px; font-weight: bold; }}
        .frequency-high {{ background: #d73027; }}
        .frequency-med {{ background: #fc8d59; }}
        .frequency-low {{ background: #91bfdb; }}
        .frequency-none {{ background: #e0e0e0; }}
        .error {{ color: #d32f2f; }}
        .success {{ color: #388e3c; }}
        .latency-table {{ width: 100%; border-collapse: collapse; }}
        .latency-table th, .latency-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        .latency-table th {{ background-color: #f2f2f2; }}
        .phase-flow {{ display: flex; align-items: center; margin: 10px 0; }}
        .phase-box {{ background: #1976d2; color: white; padding: 8px 16px; margin: 0 5px; border-radius: 4px; }}
        .arrow {{ margin: 0 5px; }}
        .timestamp {{ color: #666; font-size: 14px; }}
    </style>
</head>
<body>
    <div class="dashboard">
        <div class="header">
            <h1>Pipeline Dependency Heatmap</h1>
            <p class="timestamp">Last updated: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(data['timestamp']))}</p>
            <p>Time window: {data['time_window_minutes']} minutes | Total spans: {data['total_spans']}</p>
        </div>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-title">Edge Frequencies</div>
                {self._render_frequency_metrics(data['edge_frequencies'])}
            </div>
            
            <div class="metric-card">
                <div class="metric-title">Phase Activity</div>
                {self._render_phase_activity(data['phase_activity'])}
            </div>
            
            <div class="metric-card">
                <div class="metric-title">Error Counts</div>
                {self._render_error_metrics(data['error_counts'])}
            </div>
        </div>
        
        <div class="heatmap">
            <div class="metric-title">Dependency Heatmap</div>
            {self._render_heatmap(data)}
        </div>
        
        <div class="metric-card">
            <div class="metric-title">Latency Statistics</div>
            {self._render_latency_table(data['edge_latency_stats'])}
        </div>
        
        <div class="metric-card">
            <div class="metric-title">Canonical Phase Flow</div>
            {self._render_canonical_flow()}
        </div>
    </div>
    
    <script>
        // Auto-refresh functionality
        setTimeout(function(){{ location.reload(); }}, {self.refresh_interval * 1000});
    </script>
</body>
</html>"""
        
        return html
        
    def _render_frequency_metrics(self, frequencies: Dict[str, int]) -> str:
        """Render frequency metrics HTML"""
        if not frequencies:
            return "<p>No edge traversals recorded</p>"
            
        sorted_edges = sorted(frequencies.items(), key=lambda x: x[1], reverse=True)
        
        html = "<ul>"
        for edge, count in sorted_edges[:10]:  # Top 10
            html += f"<li><strong>{edge}:</strong> {count} traversals</li>"
        html += "</ul>"
        
        return html
        
    def _render_phase_activity(self, activity: Dict[str, int]) -> str:
        """Render phase activity HTML"""
        if not activity:
            return "<p>No phase activity recorded</p>"
            
        html = "<ul>"
        for phase in CANONICAL_PHASES:
            count = activity.get(phase, 0)
            html += f"<li><strong>{phase}:</strong> {count} events</li>"
        html += "</ul>"
        
        return html
        
    def _render_error_metrics(self, errors: Dict[str, int]) -> str:
        """Render error metrics HTML"""
        if not errors:
            return '<p class="success">No errors recorded</p>'
            
        html = '<ul>'
        for edge, count in errors.items():
            html += f'<li class="error"><strong>{edge}:</strong> {count} errors</li>'
        html += '</ul>'
        
        return html
        
    def _render_heatmap(self, data: Dict[str, Any]) -> str:
        """Render dependency heatmap"""
        frequencies = data['edge_frequencies']
        
        # Calculate max frequency for normalization
        max_freq = max(frequencies.values()) if frequencies else 1
        
        html = ""
        for source_phase in CANONICAL_PHASES:
            html += f'<div class="heatmap-row">'
            html += f'<div class="phase-label">{source_phase}</div>'
            html += f'<div class="heatmap-cells">'
            
            for target_phase in CANONICAL_PHASES:
                edge_key = f"{source_phase}->{target_phase}"
                freq = frequencies.get(edge_key, 0)
                
                # Determine cell color based on frequency
                if freq == 0:
                    cell_class = "frequency-none"
                elif freq / max_freq > 0.7:
                    cell_class = "frequency-high"
                elif freq / max_freq > 0.3:
                    cell_class = "frequency-med"  
                else:
                    cell_class = "frequency-low"
                    
                html += f'<div class="heatmap-cell {cell_class}" title="{edge_key}: {freq}">{freq if freq > 0 else ""}</div>'
                
            html += '</div></div>'
            
        return html
        
    def _render_latency_table(self, latency_stats: Dict[str, Dict[str, float]]) -> str:
        """Render latency statistics table"""
        if not latency_stats:
            return "<p>No latency data available</p>"
            
        html = '''
        <table class="latency-table">
            <tr>
                <th>Edge</th>
                <th>Mean (ms)</th>
                <th>Median (ms)</th>
                <th>P95 (ms)</th>
                <th>P99 (ms)</th>
                <th>Min (ms)</th>
                <th>Max (ms)</th>
            </tr>
        '''
        
        for edge, stats in sorted(latency_stats.items()):
            html += f'''
            <tr>
                <td>{edge}</td>
                <td>{stats['mean'] * 1000:.2f}</td>
                <td>{stats['median'] * 1000:.2f}</td>
                <td>{stats['p95'] * 1000:.2f}</td>
                <td>{stats['p99'] * 1000:.2f}</td>
                <td>{stats['min'] * 1000:.2f}</td>
                <td>{stats['max'] * 1000:.2f}</td>
            </tr>
            '''
            
        html += '</table>'
        return html
        
    def _render_canonical_flow(self) -> str:
        """Render canonical phase flow visualization"""
        html = '<div class="phase-flow">'
        
        for i, phase in enumerate(CANONICAL_PHASES):
            html += f'<div class="phase-box">{phase}</div>'
            if i < len(CANONICAL_PHASES) - 1:
                html += '<div class="arrow">→</div>'
                
        html += '</div>'
        html += '<p><strong>Canonical Order:</strong> I→X→K→A→L→R→O→G→T→S</p>'
        
        return html
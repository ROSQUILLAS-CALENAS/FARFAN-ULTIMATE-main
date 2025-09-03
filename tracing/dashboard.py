"""
Complete tracing dashboard with automatic refresh capabilities
"""

import os
import json
import time
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional, Dict, Any
from urllib.parse import urlparse, parse_qs

from .visualization import DependencyHeatmapVisualizer
from .back_edge_detector import BackEdgeDetector
from .otel_tracer import get_pipeline_tracer


class TracingDashboardHandler(BaseHTTPRequestHandler):
    """HTTP handler for tracing dashboard"""
    
    def __init__(self, *args, visualizer: DependencyHeatmapVisualizer = None, 
                 detector: BackEdgeDetector = None, **kwargs):
        self.visualizer = visualizer or DependencyHeatmapVisualizer()
        self.detector = detector or BackEdgeDetector()
        super().__init__(*args, **kwargs)
        
    def do_GET(self):
        """Handle GET requests"""
        parsed_url = urlparse(self.path)
        path = parsed_url.path
        query_params = parse_qs(parsed_url.query)
        
        # Extract time window parameter
        time_window = int(query_params.get('window', [60])[0])
        
        if path == '/' or path == '/dashboard':
            self._serve_main_dashboard(time_window)
        elif path == '/api/heatmap':
            self._serve_heatmap_data(time_window)
        elif path == '/api/violations':
            self._serve_violations_data(time_window)
        elif path == '/violations':
            self._serve_violations_dashboard(time_window)
        elif path == '/health':
            self._serve_health_check()
        else:
            self._serve_404()
            
    def _serve_main_dashboard(self, time_window: int):
        """Serve main dashboard HTML"""
        try:
            html = self.visualizer.generate_html_dashboard(time_window)
            self._send_response(200, html, 'text/html')
        except Exception as e:
            self._send_error(500, f"Dashboard generation error: {str(e)}")
            
    def _serve_violations_dashboard(self, time_window: int):
        """Serve violations dashboard"""
        try:
            violations = self.detector.analyze_span_traces(time_window)
            summary = self.detector.get_violation_summary(time_window)
            
            html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Dependency Violations Dashboard</title>
    <meta http-equiv="refresh" content="30">
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .dashboard {{ max-width: 1200px; margin: 0 auto; }}
        .header {{ background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 20px; }}
        .summary-card {{ background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .violation-list {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .violation-item {{ border-left: 4px solid #f44336; padding: 10px; margin: 10px 0; background: #ffebee; }}
        .violation-critical {{ border-color: #d32f2f; background: #ffebee; }}
        .violation-warning {{ border-color: #ff9800; background: #fff3e0; }}
        .violation-info {{ border-color: #2196f3; background: #e3f2fd; }}
        .timestamp {{ color: #666; font-size: 12px; }}
        .no-violations {{ color: #4caf50; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="dashboard">
        <div class="header">
            <h1>Dependency Violations Dashboard</h1>
            <p>Analyzing backward dependencies that violate I→X→K→A→L→R→O→G→T→S ordering</p>
            <p class="timestamp">Last updated: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="summary-grid">
            <div class="summary-card">
                <h3>Total Violations</h3>
                <h2>{summary['total_violations']}</h2>
            </div>
            
            <div class="summary-card">
                <h3>By Severity</h3>
                <ul>
                    {"".join([f"<li>{severity}: {count}</li>" for severity, count in summary['by_severity'].items()])}
                </ul>
            </div>
            
            <div class="summary-card">
                <h3>By Type</h3>
                <ul>
                    {"".join([f"<li>{vtype}: {count}</li>" for vtype, count in summary['by_type'].items()])}
                </ul>
            </div>
        </div>
        
        <div class="violation-list">
            <h3>Recent Violations</h3>
            {"<p class='no-violations'>No violations detected - pipeline ordering is correct!</p>" if not summary['recent_violations'] else ""}
            {"".join([
                f'''<div class="violation-item violation-{v['severity']}">
                    <strong>[{v['type'].upper()}]</strong> {v['component']}<br>
                    <strong>Path:</strong> {v['path']}<br>
                    <strong>Description:</strong> {v['description']}<br>
                    <span class="timestamp">{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(v['timestamp']))}</span>
                </div>'''
                for v in summary['recent_violations']
            ])}
        </div>
        
        <div class="header">
            <h3>Navigation</h3>
            <p><a href="/dashboard">← Back to Main Dashboard</a></p>
        </div>
    </div>
</body>
</html>"""
            
            self._send_response(200, html, 'text/html')
        except Exception as e:
            self._send_error(500, f"Violations dashboard error: {str(e)}")
            
    def _serve_heatmap_data(self, time_window: int):
        """Serve heatmap data as JSON API"""
        try:
            data = self.visualizer.generate_heatmap_data(time_window)
            self._send_response(200, json.dumps(data), 'application/json')
        except Exception as e:
            self._send_error(500, f"Heatmap data error: {str(e)}")
            
    def _serve_violations_data(self, time_window: int):
        """Serve violations data as JSON API"""
        try:
            violations = self.detector.analyze_span_traces(time_window)
            summary = self.detector.get_violation_summary(time_window)
            self._send_response(200, json.dumps(summary), 'application/json')
        except Exception as e:
            self._send_error(500, f"Violations data error: {str(e)}")
            
    def _serve_health_check(self):
        """Serve health check"""
        tracer = get_pipeline_tracer()
        active_spans = len(tracer.active_spans)
        total_spans = len(tracer.span_history)
        
        health_data = {
            'status': 'healthy',
            'active_spans': active_spans,
            'total_spans': total_spans,
            'timestamp': time.time()
        }
        
        self._send_response(200, json.dumps(health_data), 'application/json')
        
    def _serve_404(self):
        """Serve 404 error"""
        self._send_error(404, "Page not found")
        
    def _send_response(self, status_code: int, content: str, content_type: str):
        """Send HTTP response"""
        self.send_response(status_code)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(content.encode())))
        self.end_headers()
        self.wfile.write(content.encode())
        
    def _send_error(self, status_code: int, message: str):
        """Send error response"""
        self.send_error(status_code, message)


class TracingDashboard:
    """Complete tracing dashboard server"""
    
    def __init__(self, host: str = 'localhost', port: int = 8080,
                 refresh_interval: int = 5):
        self.host = host
        self.port = port
        self.refresh_interval = refresh_interval
        
        self.visualizer = DependencyHeatmapVisualizer(refresh_interval)
        self.detector = BackEdgeDetector()
        
        self.server: Optional[HTTPServer] = None
        self.server_thread: Optional[threading.Thread] = None
        
    def start(self):
        """Start the dashboard server"""
        # Create handler class with injected dependencies
        def handler_factory(*args, **kwargs):
            return TracingDashboardHandler(
                *args, 
                visualizer=self.visualizer,
                detector=self.detector,
                **kwargs
            )
            
        # Create and start server
        self.server = HTTPServer((self.host, self.port), handler_factory)
        self.server_thread = threading.Thread(target=self.server.serve_forever)
        self.server_thread.daemon = True
        self.server_thread.start()
        
        print(f"Tracing dashboard started at http://{self.host}:{self.port}")
        print(f"- Main dashboard: http://{self.host}:{self.port}/dashboard")
        print(f"- Violations dashboard: http://{self.host}:{self.port}/violations")
        print(f"- API endpoints: /api/heatmap, /api/violations")
        
    def stop(self):
        """Stop the dashboard server"""
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            
        if self.server_thread:
            self.server_thread.join(timeout=5)
            
        print("Tracing dashboard stopped")
        
    def get_dashboard_url(self) -> str:
        """Get dashboard URL"""
        return f"http://{self.host}:{self.port}/dashboard"
        
    def run_continuous_monitoring(self, check_interval: int = 30):
        """Run continuous monitoring with violation detection"""
        print(f"Starting continuous monitoring (check every {check_interval}s)")
        
        def monitoring_loop():
            while True:
                try:
                    # Analyze for violations
                    violations = self.detector.analyze_span_traces()
                    
                    # Log critical violations
                    critical_violations = [v for v in violations if v.severity == 'critical']
                    if critical_violations:
                        print(f"ALERT: {len(critical_violations)} critical dependency violations detected!")
                        
                    time.sleep(check_interval)
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"Monitoring error: {e}")
                    time.sleep(check_interval)
                    
        monitor_thread = threading.Thread(target=monitoring_loop)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        return monitor_thread


def create_dashboard(host: str = 'localhost', port: int = 8080) -> TracingDashboard:
    """Factory function to create a tracing dashboard"""
    return TracingDashboard(host, port)
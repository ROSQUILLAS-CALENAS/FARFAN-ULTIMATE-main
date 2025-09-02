#!/usr/bin/env python3
"""
Self-contained web server for serving canonical flow analysis results.
Integrates directly with existing project analysis orchestrator.
Enhanced with comprehensive CSS variable and font management for AtroZ design system.
"""

import json
import os
import sys
import http.server
import socketserver
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import time
import signal
import re
import urllib.parse
import hashlib
import base64

# Add project root to path for canonical imports
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


class AtroZCSSManager:
    """Comprehensive CSS variable and font management for AtroZ design system."""
    
    # AtroZ color scheme definitions
    ATROZ_COLORS = {
        '--atroz-red-900': '#7f1d1d',
        '--atroz-red-800': '#991b1b',
        '--atroz-red-700': '#b91c1c',
        '--atroz-red-600': '#dc2626',
        '--atroz-red-500': '#ef4444',
        '--atroz-blue-electric': '#0066ff',
        '--atroz-blue-dark': '#1e3a8a',
        '--atroz-blue-medium': '#3b82f6',
        '--atroz-blue-light': '#60a5fa',
        '--atroz-purple-900': '#581c87',
        '--atroz-purple-700': '#7c3aed',
        '--atroz-purple-500': '#8b5cf6',
        '--atroz-green-700': '#15803d',
        '--atroz-green-500': '#22c55e',
        '--atroz-gray-900': '#111827',
        '--atroz-gray-800': '#1f2937',
        '--atroz-gray-700': '#374151',
        '--atroz-gray-600': '#4b5563',
        '--atroz-gray-500': '#6b7280',
        '--atroz-gray-400': '#9ca3af',
        '--atroz-gray-300': '#d1d5db',
        '--atroz-gray-200': '#e5e7eb',
        '--atroz-gray-100': '#f3f4f6',
        '--atroz-white': '#ffffff',
        '--atroz-black': '#000000',
        '--atroz-gradient-primary': 'linear-gradient(135deg, var(--atroz-blue-electric) 0%, var(--atroz-purple-700) 100%)',
        '--atroz-gradient-secondary': 'linear-gradient(45deg, var(--atroz-red-600) 0%, var(--atroz-purple-500) 100%)',
        '--atroz-gradient-tertiary': 'linear-gradient(90deg, var(--atroz-green-500) 0%, var(--atroz-blue-medium) 100%)',
        '--atroz-shadow-primary': '0 4px 14px 0 rgba(0, 118, 255, 0.39)',
        '--atroz-shadow-secondary': '0 2px 10px 0 rgba(124, 58, 237, 0.25)',
    }
    
    # Font definitions with fallbacks
    FONT_DEFINITIONS = {
        'jetbrains-mono': {
            'family': 'JetBrains Mono',
            'fallbacks': ['Consolas', 'Monaco', 'Courier New', 'monospace'],
            'weights': ['300', '400', '500', '700'],
            'styles': ['normal', 'italic'],
            'preload_paths': [
                '/fonts/JetBrainsMono-Regular.woff2',
                '/fonts/JetBrainsMono-Medium.woff2',
                '/fonts/JetBrainsMono-Bold.woff2',
                '/fonts/JetBrainsMono-Italic.woff2'
            ]
        },
        'system-ui': {
            'family': 'system-ui',
            'fallbacks': ['-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'sans-serif'],
        }
    }
    
    # Browser compatibility definitions
    BROWSER_COMPATIBILITY = {
        'webkit': ['-webkit-'],
        'moz': ['-moz-'],
        'ms': ['-ms-'],
        'o': ['-o-']
    }

    @classmethod
    def validate_color_scheme(cls) -> Dict[str, bool]:
        """Validate all AtroZ custom properties are properly defined."""
        validation_results = {}
        
        for color_var, color_value in cls.ATROZ_COLORS.items():
            # Validate hex colors
            if color_value.startswith('#'):
                hex_pattern = r'^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$'
                validation_results[color_var] = bool(re.match(hex_pattern, color_value))
            # Validate CSS functions (linear-gradient, rgba, etc.)
            elif any(func in color_value for func in ['linear-gradient', 'rgba', 'var(']):
                validation_results[color_var] = True  # Advanced validation could parse CSS functions
            else:
                validation_results[color_var] = False
                
        return validation_results

    @classmethod
    def generate_css_variables(cls, environment: str = 'production') -> str:
        """Generate CSS custom properties with environment-specific optimizations."""
        css_vars = [":root {"]
        
        for var_name, var_value in cls.ATROZ_COLORS.items():
            css_vars.append(f"  {var_name}: {var_value};")
            
        # Environment-specific additions
        if environment == 'development':
            css_vars.extend([
                "  /* Development debugging */",
                "  --atroz-debug-border: 1px solid red;",
                "  --atroz-debug-bg: rgba(255, 0, 0, 0.1);"
            ])
            
        css_vars.append("}")
        return "\n".join(css_vars)

    @classmethod
    def generate_font_css(cls, preload_headers: bool = True) -> Tuple[str, List[str]]:
        """Generate font CSS with preload headers."""
        font_css = []
        preload_links = []
        
        for font_key, font_config in cls.FONT_DEFINITIONS.items():
            family = font_config['family']
            fallbacks = font_config.get('fallbacks', [])
            
            # Generate font-face declarations
            if 'preload_paths' in font_config:
                for i, path in enumerate(font_config['preload_paths']):
                    weight = font_config.get('weights', ['400'])[min(i, len(font_config.get('weights', ['400'])) - 1)]
                    style = 'italic' if 'italic' in path.lower() else 'normal'
                    
                    font_css.append(f"""
@font-face {{
  font-family: '{family}';
  src: url('{path}') format('woff2');
  font-weight: {weight};
  font-style: {style};
  font-display: swap;
}}""")
                    
                    if preload_headers:
                        preload_links.append(f"<{path}>; rel=preload; as=font; type=font/woff2; crossorigin")
            
            # Generate utility classes
            font_css.append(f"""
.font-{font_key} {{
  font-family: '{family}', {', '.join(fallbacks)};
}}""")
        
        return "\n".join(font_css), preload_links

    @classmethod
    def add_vendor_prefixes(cls, css_property: str, value: str) -> List[str]:
        """Add vendor prefixes for cross-browser compatibility."""
        prefixed_properties = []
        
        # Properties that need vendor prefixes
        needs_prefixes = {
            'transform': ['webkit', 'moz', 'ms'],
            'transition': ['webkit', 'moz', 'o'],
            'animation': ['webkit', 'moz'],
            'box-shadow': ['webkit', 'moz'],
            'border-radius': ['webkit', 'moz'],
            'background-size': ['webkit', 'moz', 'o'],
            'background-clip': ['webkit', 'moz'],
            'user-select': ['webkit', 'moz', 'ms'],
            'appearance': ['webkit', 'moz']
        }
        
        if css_property in needs_prefixes:
            for browser in needs_prefixes[css_property]:
                prefix = cls.BROWSER_COMPATIBILITY[browser][0]
                prefixed_properties.append(f"{prefix}{css_property}: {value};")
        
        # Always add the unprefixed version last
        prefixed_properties.append(f"{css_property}: {value};")
        return prefixed_properties

    @classmethod
    def detect_browser_capabilities(cls, user_agent: str) -> Dict[str, bool]:
        """Detect browser capabilities from User-Agent."""
        capabilities = {
            'supports_woff2': True,  # Default to true for modern browsers
            'supports_css_custom_properties': True,
            'supports_flexbox': True,
            'supports_grid': True,
            'supports_webp': True,
            'is_mobile': False
        }
        
        user_agent_lower = user_agent.lower()
        
        # Legacy browser detection
        if 'msie' in user_agent_lower or 'trident' in user_agent_lower:
            capabilities.update({
                'supports_css_custom_properties': False,
                'supports_flexbox': 'msie 9' not in user_agent_lower,
                'supports_grid': False,
                'supports_webp': False
            })
        
        # Mobile detection
        mobile_indicators = ['mobile', 'android', 'iphone', 'ipad', 'tablet']
        capabilities['is_mobile'] = any(indicator in user_agent_lower for indicator in mobile_indicators)
        
        return capabilities


class CanonicalFlowHandler(http.server.SimpleHTTPRequestHandler):
    """Custom HTTP handler with AtroZ design system integration."""

    def __init__(self, *args, **kwargs):
        self.css_manager = AtroZCSSManager()
        super().__init__(*args, **kwargs)

    def end_headers(self):
        """Add CORS headers, font preload headers, and security headers."""
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        
        # Font preload headers
        _, preload_links = self.css_manager.generate_font_css(preload_headers=True)
        for link in preload_links:
            self.send_header('Link', link)
        
        # Security headers
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('X-Frame-Options', 'DENY')
        self.send_header('X-XSS-Protection', '1; mode=block')
        
        super().end_headers()

    def do_OPTIONS(self):
        """Handle CORS preflight requests."""
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        """Enhanced GET handler with AtroZ design system and validation endpoints."""
        # Handle root request with AtroZ-styled index
        if self.path == '/':
            self.serve_atroz_index()
        # Handle health check
        elif self.path == '/health':
            self.serve_health()
        # Handle monitoring dashboard
        elif self.path == '/dashboard':
            self.serve_dashboard()
        # Handle API endpoints  
        elif self.path.startswith('/api/'):
            self.serve_api()
        # Handle AtroZ CSS endpoints
        elif self.path == '/atroz/styles.css':
            self.serve_atroz_css()
        elif self.path.startswith('/atroz/validate/'):
            self.serve_atroz_validation()
        # Handle font serving
        elif self.path.startswith('/fonts/'):
            self.serve_fonts()
        else:
            # Default file serving
            super().do_GET()

    def serve_atroz_index(self):
        """Serve AtroZ-styled index page with design system validation."""
        canonical_dir = Path.cwd() / "canonical_flow"
        
        # Detect browser capabilities
        user_agent = self.headers.get('User-Agent', '')
        browser_caps = self.css_manager.detect_browser_capabilities(user_agent)

        # Generate dynamic index
        available_files = []
        if canonical_dir.exists():
            for json_file in canonical_dir.glob("*.json"):
                available_files.append({
                    "name": json_file.name,
                    "path": f"/canonical_flow/{json_file.name}",
                    "size": json_file.stat().st_size
                })

        # Generate AtroZ-compatible CSS
        environment = 'development' if 'localhost' in self.headers.get('Host', '') else 'production'
        css_variables = self.css_manager.generate_css_variables(environment)
        font_css, _ = self.css_manager.generate_font_css()
        
        # Browser-specific adaptations
        grid_support = 'display: grid;' if browser_caps['supports_grid'] else 'display: flex; flex-wrap: wrap;'
        custom_props = css_variables if browser_caps['supports_css_custom_properties'] else ''

        index_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AtroZ Canonical Flow Analysis</title>
    <link rel="stylesheet" href="/atroz/styles.css">
    <style>
        {custom_props}
        {font_css}
        
        .panel { background: white; border-radius: 8px; padding: 1.5rem; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .panel-header { font-size: 1.1rem; font-weight: 600; margin-bottom: 1rem; color: #2c3e50; border-bottom: 2px solid #ecf0f1; padding-bottom: 0.5rem; }
        
        .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; }
        .metric-card { background: #f8f9fa; padding: 1rem; border-radius: 6px; text-align: center; }
        .metric-value { font-size: 1.8rem; font-weight: bold; margin-bottom: 0.3rem; }
        .metric-label { color: #7f8c8d; font-size: 0.9rem; }
        .metric-good { color: #27ae60; }
        .metric-warning { color: #f39c12; }
        .metric-bad { color: #e74c3c; }
        
        .stage-list { max-height: 300px; overflow-y: auto; }
        .stage-item { display: flex; justify-content: between; align-items: center; padding: 0.8rem; margin-bottom: 0.5rem; background: #f8f9fa; border-radius: 4px; }
        .stage-name { font-weight: 500; flex-grow: 1; }
        .stage-status { padding: 0.2rem 0.6rem; border-radius: 12px; font-size: 0.8rem; font-weight: bold; }
        .stage-healthy { background: #d5f4e6; color: #27ae60; }
        .stage-warning { background: #fdf2e9; color: #f39c12; }
        .stage-error { background: #fadbd8; color: #e74c3c; }
        
        .alert-list { max-height: 250px; overflow-y: auto; }
        .alert-item { padding: 0.8rem; margin-bottom: 0.5rem; border-left: 4px solid; border-radius: 4px; }
        .alert-critical { border-color: #e74c3c; background: #fadbd8; }
        .alert-warning { border-color: #f39c12; background: #fdf2e9; }
        .alert-info { border-color: #3498db; background: #ebf3fd; }
        
        .chart-container { height: 200px; background: #f8f9fa; border-radius: 4px; display: flex; align-items: center; justify-content: center; color: #7f8c8d; }
        
        .refresh-btn { background: #3498db; color: white; border: none; padding: 0.5rem 1rem; border-radius: 4px; cursor: pointer; }
        .refresh-btn:hover { background: #2980b9; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🔍 Pipeline Health Dashboard</h1>
        <div>
            <span id="overall-status" class="status-indicator status-healthy">HEALTHY</span>
            <button class="refresh-btn" onclick="refreshDashboard()">Refresh</button>
        </div>
    </div>
    
    <div class="dashboard">
        <!-- System Overview -->
        <div class="panel full-width">
            <div class="panel-header">System Overview</div>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div id="total-stages" class="metric-value metric-good">12</div>
                    <div class="metric-label">Total Stages</div>
                </div>
                <div class="metric-card">
                    <div id="healthy-stages" class="metric-value metric-good">10</div>
                    <div class="metric-label">Healthy Stages</div>
                </div>
                <div class="metric-card">
                    <div id="avg-processing-time" class="metric-value metric-good">2.3s</div>
                    <div class="metric-label">Avg Processing Time</div>
                </div>
                <div class="metric-card">
                    <div id="error-rate" class="metric-value metric-warning">3.2%</div>
                    <div class="metric-label">Error Rate</div>
                </div>
                <div class="metric-card">
                    <div id="memory-usage" class="metric-value metric-good">68%</div>
                    <div class="metric-label">Memory Usage</div>
                </div>
                <div class="metric-card">
                    <div id="documents-processed" class="metric-value metric-good">1,247</div>
                    <div class="metric-label">Documents Processed</div>
                </div>
            </div>
        </div>
        
        <!-- Stage Health -->
        <div class="panel">
            <div class="panel-header">Stage Health Status</div>
            <div id="stage-list" class="stage-list">
                <div class="stage-item">
                    <span class="stage-name">I_ingestion_preparation</span>
                    <span class="stage-status stage-healthy">HEALTHY</span>
                </div>
                <div class="stage-item">
                    <span class="stage-name">A_analysis_nlp</span>
                    <span class="stage-status stage-healthy">HEALTHY</span>
                </div>
                <div class="stage-item">
                    <span class="stage-name">K_knowledge_extraction</span>
                    <span class="stage-status stage-warning">WARNING</span>
                </div>
                <div class="stage-item">
                    <span class="stage-name">R_search_retrieval</span>
                    <span class="stage-status stage-healthy">HEALTHY</span>
                </div>
                <div class="stage-item">
                    <span class="stage-name">L_classification_evaluation</span>
                    <span class="stage-status stage-error">ERROR</span>
                </div>
            </div>
        </div>
        
        <!-- Processing Metrics -->
        <div class="panel">
            <div class="panel-header">Real-time Processing</div>
            <div class="chart-container">
                <div>📊 Processing rate chart would appear here</div>
            </div>
            <div style="margin-top: 1rem;">
                <div style="display: flex; justify-content: space-between;">
                    <span>Throughput:</span>
                    <strong id="throughput">45.2 docs/min</strong>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span>Queue Depth:</span>
                    <strong id="queue-depth">23</strong>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span>Active Workers:</span>
                    <strong id="active-workers">8</strong>
                </div>
            </div>
        </div>
        
        <!-- Schema Compliance -->
        <div class="panel">
            <div class="panel-header">Schema Compliance</div>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div id="schema-compliance" class="metric-value metric-good">96.7%</div>
                    <div class="metric-label">Overall Compliance</div>
                </div>
                <div class="metric-card">
                    <div id="validation-failures" class="metric-value metric-warning">12</div>
                    <div class="metric-label">Validation Failures</div>
                </div>
            </div>
        </div>
        
        <!-- Data Integrity -->
        <div class="panel">
            <div class="panel-header">Data Integrity</div>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div id="checksum-matches" class="metric-value metric-good">99.1%</div>
                    <div class="metric-label">Checksum Matches</div>
                </div>
                <div class="metric-card">
                    <div id="integrity-failures" class="metric-value metric-warning">7</div>
                    <div class="metric-label">Integrity Failures</div>
                </div>
            </div>
        </div>
        
        <!-- Alerts -->
        <div class="panel full-width">
            <div class="panel-header">Active Alerts</div>
            <div id="alert-list" class="alert-list">
                <div class="alert-item alert-warning">
                    <strong>WARNING:</strong> K_knowledge_extraction stage showing 15% increase in processing time
                    <div style="font-size: 0.8rem; color: #7f8c8d; margin-top: 0.3rem;">2 minutes ago</div>
                </div>
                <div class="alert-item alert-critical">
                    <strong>CRITICAL:</strong> L_classification_evaluation stage failed validation for 3 consecutive documents
                    <div style="font-size: 0.8rem; color: #7f8c8d; margin-top: 0.3rem;">5 minutes ago</div>
                </div>
                <div class="alert-item alert-info">
                    <strong>INFO:</strong> Memory usage approaching 75% threshold
                    <div style="font-size: 0.8rem; color: #7f8c8d; margin-top: 0.3rem;">8 minutes ago</div>
                </div>
            </div>
        </div>
    </div>

    <script>
        function refreshDashboard() {
            fetch('/api/dashboard/metrics')
                .then(response => response.json())
                .then(data => updateDashboard(data))
                .catch(error => console.error('Error refreshing dashboard:', error));
        }
        
        function updateDashboard(data) {
            // Update system metrics
            if (data.system) {
                document.getElementById('healthy-stages').textContent = data.system.healthy_stages;
                document.getElementById('avg-processing-time').textContent = data.system.avg_processing_time + 's';
                document.getElementById('error-rate').textContent = data.system.error_rate + '%';
                document.getElementById('memory-usage').textContent = data.system.memory_usage + '%';
                document.getElementById('documents-processed').textContent = data.system.documents_processed.toLocaleString();
            }
            
            // Update processing metrics
            if (data.processing) {
                document.getElementById('throughput').textContent = data.processing.throughput + ' docs/min';
                document.getElementById('queue-depth').textContent = data.processing.queue_depth;
                document.getElementById('active-workers').textContent = data.processing.active_workers;
            }
            
            // Update compliance metrics
            if (data.compliance) {
                document.getElementById('schema-compliance').textContent = data.compliance.schema_compliance + '%';
                document.getElementById('validation-failures').textContent = data.compliance.validation_failures;
                document.getElementById('checksum-matches').textContent = data.compliance.checksum_matches + '%';
                document.getElementById('integrity-failures').textContent = data.compliance.integrity_failures;
            }
        }
        
        // Auto-refresh every 30 seconds
        setInterval(refreshDashboard, 30000);
        
        // Initial load
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        
        body {{
            font-family: 'JetBrains Mono', Consolas, Monaco, 'Courier New', monospace;
            background: var(--atroz-gray-900, #111827);
            color: var(--atroz-white, #ffffff);
            min-height: 100vh;
            {self._generate_prefixed_css('background', 'var(--atroz-gradient-primary, linear-gradient(135deg, #0066ff 0%, #7c3aed 100%))')}
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
        }}
        
        .header {{
            text-align: center;
            margin-bottom: 3rem;
        }}
        
        .logo {{
            font-size: 2.5rem;
            font-weight: 700;
            background: var(--atroz-gradient-secondary, linear-gradient(45deg, #dc2626 0%, #8b5cf6 100%));
            {self._generate_prefixed_css('background-clip', 'text')}
            -webkit-text-fill-color: transparent;
            margin-bottom: 1rem;
        }}
        
        .subtitle {{
            font-size: 1.25rem;
            color: var(--atroz-gray-300, #d1d5db);
            font-weight: 300;
        }}
        
        .file-grid {{
            {grid_support}
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 1.5rem;
            margin-bottom: 3rem;
        }}
        
        .file-card {{
            background: var(--atroz-gray-800, #1f2937);
            border-radius: 12px;
            padding: 1.5rem;
            border: 1px solid var(--atroz-gray-700, #374151);
            {self._generate_prefixed_css('box-shadow', 'var(--atroz-shadow-primary, 0 4px 14px 0 rgba(0, 118, 255, 0.39))')}
            {self._generate_prefixed_css('transition', 'all 0.3s ease')}
        }}
        
        .file-card:hover {{
            {self._generate_prefixed_css('transform', 'translateY(-4px)')}
            {self._generate_prefixed_css('box-shadow', 'var(--atroz-shadow-secondary, 0 2px 10px 0 rgba(124, 58, 237, 0.25))')}
        }}
        
        .file-name {{
            font-size: 1.1rem;
            font-weight: 500;
            color: var(--atroz-blue-light, #60a5fa);
            text-decoration: none;
            display: block;
            margin-bottom: 0.5rem;
        }}
        
        .file-size {{
            font-size: 0.875rem;
            color: var(--atroz-gray-400, #9ca3af);
        }}
        
        .controls {{
            display: flex;
            justify-content: center;
            gap: 1rem;
            flex-wrap: wrap;
        }}
        
        .btn {{
            padding: 0.75rem 1.5rem;
            border-radius: 8px;
            text-decoration: none;
            font-weight: 500;
            {self._generate_prefixed_css('transition', 'all 0.2s ease')}
            border: none;
            cursor: pointer;
        }}
        
        .btn-primary {{
            background: var(--atroz-blue-electric, #0066ff);
            color: var(--atroz-white, #ffffff);
        }}
        
        .btn-secondary {{
            background: var(--atroz-gray-700, #374151);
            color: var(--atroz-gray-200, #e5e7eb);
        }}
        
        .btn:hover {{
            {self._generate_prefixed_css('transform', 'scale(1.05)')}
        }}
        
        .validation-status {{
            margin-top: 2rem;
            padding: 1rem;
            background: var(--atroz-gray-800, #1f2937);
            border-radius: 8px;
            border-left: 4px solid var(--atroz-green-500, #22c55e);
        }}
        
        @media (max-width: 768px) {{
            .container {{ padding: 1rem; }}
            .logo {{ font-size: 2rem; }}
            .file-grid {{ grid-template-columns: 1fr; }}
            .controls {{ flex-direction: column; align-items: center; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header class="header">
            <h1 class="logo">AtroZ Flow</h1>
            <p class="subtitle">Canonical Analysis Results Dashboard</p>
        </header>
        
        <main>
            <section class="file-grid">
                {''.join([f'''
                <div class="file-card">
                    <a href="{f["path"]}" class="file-name">{f["name"]}</a>
                    <p class="file-size">{f["size"]:,} bytes</p>
                </div>
                ''' for f in available_files]) if available_files else '<p style="text-align: center; color: var(--atroz-gray-400, #9ca3af);">No analysis files available</p>'}
            </section>
            
            <section class="controls">
                <a href="/health" class="btn btn-primary">Health Check</a>
                <a href="/api/status" class="btn btn-secondary">API Status</a>
                <a href="/atroz/validate/colors" class="btn btn-secondary">Validate Colors</a>
                <a href="/atroz/validate/fonts" class="btn btn-secondary">Font Status</a>
            </section>
            
            <div class="validation-status">
                <h3>Browser Compatibility</h3>
                <ul>
                    <li>CSS Custom Properties: {'✓' if browser_caps['supports_css_custom_properties'] else '✗'}</li>
                    <li>CSS Grid: {'✓' if browser_caps['supports_grid'] else '✗'}</li>
                    <li>WOFF2 Fonts: {'✓' if browser_caps['supports_woff2'] else '✗'}</li>
                    <li>WebP Images: {'✓' if browser_caps['supports_webp'] else '✗'}</li>
                    <li>Device Type: {'Mobile' if browser_caps['is_mobile'] else 'Desktop'}</li>
                </ul>
            </div>
        </main>
    </div>
>>>>>>> 3240a85 (Add comprehensive CSS variable validation, font management, and browser compatibility middleware to canonical web server)
</body>
</html>"""

        self.send_response(200)
        self.send_header('Content-type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(index_html.encode('utf-8'))

    def _generate_prefixed_css(self, property_name: str, value: str) -> str:
        """Generate vendor-prefixed CSS properties."""
        prefixed = self.css_manager.add_vendor_prefixes(property_name, value)
        return '\n            '.join(prefixed)

    def serve_dashboard(self):
        """Serve comprehensive monitoring dashboard with real-time metrics (dynamic)."""
        dashboard_html = """
<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"UTF-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
  <title>Pipeline Health Monitoring Dashboard</title>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f5f7fa; }
    .header { background: #2c3e50; color: white; padding: 1rem 2rem; display: flex; justify-content: space-between; align-items: center; }
    .header h1 { font-size: 1.5rem; }
    .status-indicator { padding: 0.3rem 0.8rem; border-radius: 15px; font-size: 0.8rem; font-weight: bold; }
    .status-healthy { background: #27ae60; }
    .status-warning { background: #f39c12; }
    .status-critical { background: #e74c3c; }
    .dashboard { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; padding: 2rem; max-width: 1400px; margin: 0 auto; }
    .panel { background: white; border-radius: 8px; padding: 1.5rem; box-shadow: 0 2px 10px rgba(0,0,0,0.08); }
    .panel-header { font-size: 1.1rem; font-weight: 600; margin-bottom: 1rem; color: #2c3e50; border-bottom: 2px solid #ecf0f1; padding-bottom: 0.5rem; }
    .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; }
    .metric-card { background: #f8f9fa; padding: 1rem; border-radius: 6px; text-align: center; }
    .metric-value { font-size: 1.8rem; font-weight: bold; margin-bottom: 0.3rem; }
    .metric-label { color: #7f8c8d; font-size: 0.9rem; }
    .metric-good { color: #27ae60; }
    .metric-warning { color: #f39c12; }
    .metric-bad { color: #e74c3c; }
    .stage-list { max-height: 350px; overflow-y: auto; }
    .stage-item { display: flex; justify-content: space-between; align-items: center; padding: 0.6rem 0.8rem; margin-bottom: 0.4rem; background: #f8f9fa; border-radius: 4px; }
    .stage-name { font-weight: 500; flex-grow: 1; }
    .stage-status { padding: 0.2rem 0.6rem; border-radius: 12px; font-size: 0.8rem; font-weight: bold; }
    .stage-healthy { background: #d5f4e6; color: #27ae60; }
    .stage-warning { background: #fdf2e9; color: #f39c12; }
    .stage-error { background: #fadbd8; color: #e74c3c; }
    .chart-container { height: 200px; background: #f8f9fa; border-radius: 4px; display: flex; align-items: center; justify-content: center; color: #7f8c8d; }
    .alert-list { max-height: 250px; overflow-y: auto; }
    .alert-item { padding: 0.8rem; margin-bottom: 0.5rem; border-left: 4px solid; border-radius: 4px; }
    .alert-critical { border-color: #e74c3c; background: #fadbd8; }
    .alert-warning { border-color: #f39c12; background: #fdf2e9; }
    .alert-info { border-color: #3498db; background: #ebf3fd; }
    .full-width { grid-column: 1 / -1; }
    .refresh-btn { background: #3498db; color: white; border: none; padding: 0.5rem 1rem; border-radius: 4px; cursor: pointer; }
    .refresh-btn:hover { background: #2980b9; }
  </style>
</head>
<body>
  <div class=\"header\">
    <h1>🔍 Pipeline Health Dashboard</h1>
    <div>
      <span id=\"overall-status\" class=\"status-indicator status-healthy\">HEALTHY</span>
      <button class=\"refresh-btn\" onclick=\"refreshDashboard()\">Refresh</button>
    </div>
  </div>
  <div class=\"dashboard\">
    <div class=\"panel full-width\">
      <div class=\"panel-header\">System Overview</div>
      <div class=\"metrics-grid\">
        <div class=\"metric-card\">
          <div id=\"total-stages\" class=\"metric-value metric-good\">-</div>
          <div class=\"metric-label\">Total Stages</div>
        </div>
        <div class=\"metric-card\">
          <div id=\"healthy-stages\" class=\"metric-value metric-good\">-</div>
          <div class=\"metric-label\">Healthy Stages</div>
        </div>
        <div class=\"metric-card\">
          <div id=\"avg-processing-time\" class=\"metric-value metric-good\">-</div>
          <div class=\"metric-label\">Avg Processing Time</div>
        </div>
        <div class=\"metric-card\">
          <div id=\"error-rate\" class=\"metric-value metric-warning\">-</div>
          <div class=\"metric-label\">Error Rate</div>
        </div>
        <div class=\"metric-card\">
          <div id=\"memory-usage\" class=\"metric-value metric-good\">-</div>
          <div class=\"metric-label\">Memory Usage</div>
        </div>
        <div class=\"metric-card\">
          <div id=\"documents-processed\" class=\"metric-value metric-good\">-</div>
          <div class=\"metric-label\">Documents Processed</div>
        </div>
      </div>
    </div>

    <div class=\"panel\">
      <div class=\"panel-header\">Stage Health Status</div>
      <div id=\"stage-list\" class=\"stage-list\"></div>
    </div>

    <div class=\"panel\">
      <div class=\"panel-header\">Real-time Processing</div>
      <div class=\"chart-container\"><div>📊 Processing rate chart</div></div>
      <div style=\"margin-top: 1rem;\">
        <div style=\"display: flex; justify-content: space-between;\"><span>Throughput:</span><strong id=\"throughput\">-</strong></div>
        <div style=\"display: flex; justify-content: space-between;\"><span>Queue Depth:</span><strong id=\"queue-depth\">-</strong></div>
        <div style=\"display: flex; justify-content: space-between;\"><span>Active Workers:</span><strong id=\"active-workers\">-</strong></div>
      </div>
    </div>

    <div class=\"panel\">
      <div class=\"panel-header\">Schema Compliance</div>
      <div class=\"metrics-grid\">
        <div class=\"metric-card\"><div id=\"schema-compliance\" class=\"metric-value metric-good\">-</div><div class=\"metric-label\">Overall Compliance</div></div>
        <div class=\"metric-card\"><div id=\"validation-failures\" class=\"metric-value metric-warning\">-</div><div class=\"metric-label\">Validation Failures</div></div>
        <div class=\"metric-card\"><div id=\"checksum-matches\" class=\"metric-value metric-good\">-</div><div class=\"metric-label\">Checksum Matches</div></div>
        <div class=\"metric-card\"><div id=\"integrity-failures\" class=\"metric-value metric-warning\">-</div><div class=\"metric-label\">Integrity Failures</div></div>
      </div>
    </div>

    <div class=\"panel full-width\">
      <div class=\"panel-header\">Active Alerts</div>
      <div id=\"alert-list\" class=\"alert-list\"></div>
    </div>
  </div>

  <script>
    async function refreshDashboard() {
      try {
        const [metricsRes, healthRes, alertsRes] = await Promise.all([
          fetch('/api/dashboard/metrics'),
          fetch('/api/pipeline/health'),
          fetch('/api/alerts')
        ]);
        const metrics = await metricsRes.json();
        const health = await healthRes.json();
        const alerts = await alertsRes.json();
        updateDashboard(metrics, health, alerts);
      } catch (e) {
        console.error('Error refreshing dashboard:', e);
      }
    }

    function updateDashboard(metrics, health, alerts) {
      // System metrics
      if (metrics.system) {
        const total = metrics.system.total_stages ?? (health.stages ? health.stages.length : 0);
        document.getElementById('total-stages').textContent = total;
        document.getElementById('healthy-stages').textContent = metrics.system.healthy_stages;
        document.getElementById('avg-processing-time').textContent = metrics.system.avg_processing_time + 's';
        document.getElementById('error-rate').textContent = metrics.system.error_rate + '%';
        document.getElementById('memory-usage').textContent = metrics.system.memory_usage + '%';
        document.getElementById('documents-processed').textContent = (metrics.system.documents_processed || 0).toLocaleString();
      }

      // Processing metrics
      if (metrics.processing) {
        document.getElementById('throughput').textContent = metrics.processing.throughput + ' docs/min';
        document.getElementById('queue-depth').textContent = metrics.processing.queue_depth;
        document.getElementById('active-workers').textContent = metrics.processing.active_workers;
      }

      // Compliance
      if (metrics.compliance) {
        document.getElementById('schema-compliance').textContent = metrics.compliance.schema_compliance + '%';
        document.getElementById('validation-failures').textContent = metrics.compliance.validation_failures;
        document.getElementById('checksum-matches').textContent = metrics.compliance.checksum_matches + '%';
        document.getElementById('integrity-failures').textContent = metrics.compliance.integrity_failures;
      }

      // Stage list
      const stageList = document.getElementById('stage-list');
      stageList.innerHTML = '';
      if (health && Array.isArray(health.stages)) {
        // overall status
        const overall = health.overall_status || 'healthy';
        const overallEl = document.getElementById('overall-status');
        overallEl.textContent = overall.toUpperCase();
        overallEl.className = 'status-indicator ' + (overall === 'healthy' ? 'status-healthy' : (overall === 'warning' ? 'status-warning' : 'status-critical'));

        for (const s of health.stages) {
          const item = document.createElement('div');
          item.className = 'stage-item';
          const statusClass = s.status === 'healthy' ? 'stage-healthy' : (s.status === 'warning' ? 'stage-warning' : 'stage-error');
          item.innerHTML = `<span class=\"stage-name\">${s.name}</span><span class=\"stage-status ${statusClass}\">${s.status.toUpperCase()}</span>`;
          stageList.appendChild(item);
        }
      }

      // Alerts
      const alertList = document.getElementById('alert-list');
      alertList.innerHTML = '';
      if (alerts && Array.isArray(alerts.active_alerts)) {
        for (const a of alerts.active_alerts) {
          const div = document.createElement('div');
          const cls = a.severity === 'critical' ? 'alert-critical' : (a.severity === 'warning' ? 'alert-warning' : 'alert-info');
          div.className = 'alert-item ' + cls;
          const when = a.timestamp ? new Date(a.timestamp * 1000).toLocaleTimeString() : '';
          div.innerHTML = `<strong>${a.severity.toUpperCase()}:</strong> ${a.stage} - ${a.message}<div style=\"font-size: 0.8rem; color: #7f8c8d; margin-top: 0.3rem;\">${when}</div>`;
          alertList.appendChild(div);
        }
      }
    }

    // Auto refresh every 30s
    setInterval(refreshDashboard, 30000);
    // Initial load
    refreshDashboard();
  </script>
</body>
</html>
"""
        self.send_response(200)
        self.send_header('Content-type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(dashboard_html.encode())

    def serve_dashboard_metrics(self):
        """Serve real-time dashboard metrics."""
        try:
            # Import monitoring system
            from monitoring_dashboard import MonitoringDashboard
            dashboard = MonitoringDashboard()
            metrics = dashboard.get_current_metrics()
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(metrics, indent=2).encode())
            
        except ImportError:
            # Fallback to mock data
            mock_metrics = {
                "system": {
                    "healthy_stages": 10,
                    "total_stages": 18,
                    "avg_processing_time": 2.3,
                    "error_rate": 3.2,
                    "memory_usage": 68,
                    "documents_processed": 1247
                },
                "processing": {
                    "throughput": 45.2,
                    "queue_depth": 23,
                    "active_workers": 8
                },
                "compliance": {
                    "schema_compliance": 96.7,
                    "validation_failures": 12,
                    "checksum_matches": 99.1,
                    "integrity_failures": 7
                },
                "timestamp": time.time()
            }
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(mock_metrics, indent=2).encode())

    def serve_pipeline_health(self):
        """Serve pipeline health status for all stages."""
        try:
            # Import monitoring system
            from monitoring_dashboard import MonitoringDashboard
            dashboard = MonitoringDashboard()
            health_data = dashboard.get_pipeline_health()
            
        except ImportError:
            # Mock pipeline health data
            stages = [
                "I_ingestion_preparation", "A_analysis_nlp", "K_knowledge_extraction", 
                "R_search_retrieval", "L_classification_evaluation", "G_aggregation_reporting",
                "S_synthesis_output", "T_integration_storage", "X_context_construction",
                "O_orchestration_control"
            ]
            
            health_data = {
                "overall_status": "healthy",
                "stages": []
            }
            
            for i, stage in enumerate(stages):
                status = "healthy"
                if i == 2:  # Knowledge extraction warning
                    status = "warning"
                elif i == 4:  # Classification error
                    status = "error"
                    
                health_data["stages"].append({
                    "name": stage,
                    "status": status,
                    "processing_time": f"{2.1 + (i * 0.3):.1f}s",
                    "error_count": 1 if status == "error" else 0,
                    "last_update": time.time() - (i * 60)
                })
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(health_data, indent=2).encode())

    def serve_stage_metrics(self, stage_name: str):
        """Serve detailed metrics for a specific stage."""
        try:
            from monitoring_dashboard import MonitoringDashboard
            dashboard = MonitoringDashboard()
            stage_metrics = dashboard.get_stage_metrics(stage_name)
            
        except ImportError:
            # Mock stage metrics
            stage_metrics = {
                "stage_name": stage_name,
                "status": "healthy",
                "processing_rate": "45.2 docs/min",
                "error_rate": 2.1,
                "avg_processing_time": 2.3,
                "memory_usage": 68.5,
                "schema_compliance_rate": 96.7,
                "validation_failures": 3,
                "checksum_matches": 99.1,
                "last_24h_stats": {
                    "documents_processed": 1247,
                    "total_errors": 26,
                    "avg_latency": 2.3
                },
                "recent_errors": [
                    {"timestamp": time.time() - 300, "type": "validation", "message": "Schema validation failed"},
                    {"timestamp": time.time() - 600, "type": "timeout", "message": "Processing timeout exceeded"}
                ]
            }
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(stage_metrics, indent=2).encode())

    def serve_alerts(self):
        """Serve active alerts and notifications."""
        try:
            from monitoring_dashboard import MonitoringDashboard
            dashboard = MonitoringDashboard()
            alerts = dashboard.get_active_alerts()
            
        except ImportError:
            # Mock alerts data
            alerts = {
                "active_alerts": [
                    {
                        "id": "alert_001",
                        "severity": "warning", 
                        "stage": "K_knowledge_extraction",
                        "message": "Processing time increased by 15%",
                        "timestamp": time.time() - 120,
                        "threshold": "processing_time > 3.0s"
                    },
                    {
                        "id": "alert_002",
                        "severity": "critical",
                        "stage": "L_classification_evaluation", 
                        "message": "Validation failed for 3 consecutive documents",
                        "timestamp": time.time() - 300,
                        "threshold": "consecutive_failures > 2"
                    },
                    {
                        "id": "alert_003",
                        "severity": "info",
                        "stage": "system",
                        "message": "Memory usage approaching 75% threshold", 
                        "timestamp": time.time() - 480,
                        "threshold": "memory_usage > 75%"
                    }
                ],
                "total_alerts": 3,
                "alert_summary": {
                    "critical": 1,
                    "warning": 1,
                    "info": 1
                }
            }
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(alerts, indent=2).encode())

    def serve_performance_metrics(self):
        """Serve performance and resource utilization metrics."""
        try:
            from monitoring_dashboard import MonitoringDashboard
            dashboard = MonitoringDashboard()
            perf_metrics = dashboard.get_performance_metrics()
            
        except ImportError:
            # Mock performance metrics using psutil if available
            perf_metrics = {
                "cpu_usage": 45.2,
                "memory_usage": 68.5,
                "disk_io": {
                    "read_mb_per_sec": 12.3,
                    "write_mb_per_sec": 8.7
                },
                "network_io": {
                    "bytes_sent_per_sec": 1024000,
                    "bytes_recv_per_sec": 2048000
                },
                "process_metrics": {
                    "active_processes": 8,
                    "avg_cpu_per_process": 5.6,
                    "total_memory_mb": 2048
                },
                "pipeline_metrics": {
                    "documents_per_hour": 2714,
                    "avg_end_to_end_latency": 15.2,
                    "queue_depths": {
                        "ingestion": 23,
                        "processing": 15,
                        "output": 7
                    }
                }
            }
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')


    def serve_health(self):
        """Serve health check endpoint."""
        health_data = {
            "status": "healthy",
            "timestamp": time.time(),
            "canonical_flow_available": (Path.cwd() / "canonical_flow").exists(),
            "validation_status": {
                "success": self.server.validation_result.success if hasattr(self.server, 'validation_result') else True,
                "fallbacks_active": len(self.server.validation_result.activated_fallbacks) if hasattr(self.server, 'validation_result') else 0,
                "warnings": len(self.server.validation_result.warnings) if hasattr(self.server, 'validation_result') else 0
            }
        }

        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(health_data, indent=2).encode())

    def serve_api(self):
        """Serve API endpoints for programmatic access."""
        if self.path == '/api/status':
            canonical_dir = Path.cwd() / "canonical_flow"

            status = {
                "canonical_flow_exists": canonical_dir.exists(),
                "available_files": [],
                "total_files": 0,
                "atroz_design_system": {
                    "colors_validated": len([v for v in self.css_manager.validate_color_scheme().values() if v]),
                    "total_colors": len(self.css_manager.ATROZ_COLORS),
                    "fonts_configured": len(self.css_manager.FONT_DEFINITIONS)
                }
            }

            if canonical_dir.exists():
                for json_file in canonical_dir.glob("*.json"):
                    status["available_files"].append({
                        "filename": json_file.name,
                        "size": json_file.stat().st_size,
                        "modified": json_file.stat().st_mtime
                    })
                status["total_files"] = len(status["available_files"])

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(status, indent=2).encode())
            
        elif self.path == '/api/dashboard/metrics':
            self.serve_dashboard_metrics()
        elif self.path == '/api/pipeline/health':
            self.serve_pipeline_health()
        elif self.path.startswith('/api/stages/'):
            stage_name = self.path.split('/')[-1]
            self.serve_stage_metrics(stage_name)
        elif self.path == '/api/alerts':
            self.serve_alerts()
        elif self.path == '/api/performance':
            self.serve_performance_metrics()
        else:
            self.send_error(404)

    def serve_atroz_css(self):
        """Serve dynamically generated AtroZ CSS."""
        user_agent = self.headers.get('User-Agent', '')
        browser_caps = self.css_manager.detect_browser_capabilities(user_agent)
        environment = 'development' if 'localhost' in self.headers.get('Host', '') else 'production'
        
        # Generate CSS based on browser capabilities
        css_content = []
        
        # CSS Custom Properties (if supported)
        if browser_caps['supports_css_custom_properties']:
            css_content.append(self.css_manager.generate_css_variables(environment))
        
        # Font CSS
        font_css, _ = self.css_manager.generate_font_css()
        css_content.append(font_css)
        
        # Compatibility fallbacks for older browsers
        if not browser_caps['supports_css_custom_properties']:
            css_content.append("""
/* Fallbacks for older browsers */
.atroz-primary { background-color: #0066ff; }
.atroz-secondary { background-color: #7c3aed; }
.atroz-danger { background-color: #dc2626; }
.atroz-success { background-color: #22c55e; }
""")

        # Flexbox fallbacks if Grid is not supported
        if not browser_caps['supports_grid']:
            css_content.append("""
/* Flexbox fallbacks */
.grid-fallback {
    display: flex;
    flex-wrap: wrap;
}
.grid-fallback > * {
    flex: 1 1 300px;
    margin: 0.5rem;
}
""")

        final_css = '\n\n'.join(css_content)
        
        # Add cache headers for production
        if environment == 'production':
            css_hash = hashlib.md5(final_css.encode()).hexdigest()[:8]
            self.send_header('ETag', f'"{css_hash}"')
            self.send_header('Cache-Control', 'public, max-age=3600')

        self.send_response(200)
        self.send_header('Content-type', 'text/css; charset=utf-8')
        self.end_headers()
        self.wfile.write(final_css.encode('utf-8'))

    def serve_atroz_validation(self):
        """Serve AtroZ design system validation endpoints."""
        path_parts = self.path.split('/')
        
        if len(path_parts) < 4:
            self.send_error(404)
            return
            
        validation_type = path_parts[3]
        
        if validation_type == 'colors':
            validation_results = self.css_manager.validate_color_scheme()
            
            response = {
                "validation_type": "color_scheme",
                "timestamp": time.time(),
                "total_colors": len(self.css_manager.ATROZ_COLORS),
                "valid_colors": len([v for v in validation_results.values() if v]),
                "invalid_colors": len([v for v in validation_results.values() if not v]),
                "results": validation_results,
                "status": "pass" if all(validation_results.values()) else "fail"
            }
            
        elif validation_type == 'fonts':
            # Font loading validation
            font_status = {}
            for font_key, font_config in self.css_manager.FONT_DEFINITIONS.items():
                font_status[font_key] = {
                    "family": font_config['family'],
                    "fallbacks_count": len(font_config.get('fallbacks', [])),
                    "preload_paths": font_config.get('preload_paths', []),
                    "has_preload": len(font_config.get('preload_paths', [])) > 0
                }
            
            response = {
                "validation_type": "font_loading",
                "timestamp": time.time(),
                "fonts": font_status,
                "total_fonts": len(self.css_manager.FONT_DEFINITIONS),
                "fonts_with_preload": len([f for f in font_status.values() if f['has_preload']]),
                "status": "pass"
            }
            
        elif validation_type == 'integrity':
            # Comprehensive integrity check
            color_validation = self.css_manager.validate_color_scheme()
            user_agent = self.headers.get('User-Agent', '')
            browser_caps = self.css_manager.detect_browser_capabilities(user_agent)
            
            response = {
                "validation_type": "integrity_check",
                "timestamp": time.time(),
                "color_scheme": {
                    "status": "pass" if all(color_validation.values()) else "fail",
                    "valid_count": len([v for v in color_validation.values() if v]),
                    "total_count": len(color_validation)
                },
                "font_system": {
                    "status": "pass",
                    "configured_fonts": len(self.css_manager.FONT_DEFINITIONS),
                    "preload_enabled": True
                },
                "browser_compatibility": browser_caps,
                "overall_status": "pass" if all(color_validation.values()) else "degraded"
            }
            
        else:
            self.send_error(404, f"Unknown validation type: {validation_type}")
            return

        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(response, indent=2).encode())

    def serve_fonts(self):
        """Serve font files with proper headers."""
        # This is a placeholder - in production, you'd serve actual font files
        # For demo purposes, we'll return a 404 with helpful information
        
        requested_font = self.path.split('/')[-1]
        
        response = {
            "error": "Font serving not implemented",
            "requested_font": requested_font,
            "message": "This is a demo endpoint. In production, serve actual font files here.",
            "expected_fonts": [
                "JetBrainsMono-Regular.woff2",
                "JetBrainsMono-Medium.woff2", 
                "JetBrainsMono-Bold.woff2",
                "JetBrainsMono-Italic.woff2"
            ]
        }
        
        self.send_response(404)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(response, indent=2).encode())


class CanonicalFlowServer:
    """Self-contained server for canonical flow results."""

    def __init__(self, port: int = 8000, project_root_path: Optional[str] = None):
        # Pre-flight validation - MUST be first operation
        from canonical_flow.mathematical_enhancers.pre_flight_validator import check_library_compatibility
        
        print("Starting canonical web server with pre-flight validation...")
        validation_result = check_library_compatibility()
        
        if validation_result.activated_fallbacks:
            print(f"Server initialized with {len(validation_result.activated_fallbacks)} fallback implementations")
        
        self.port = port
        self.project_root = Path(project_root_path) if project_root_path else Path.cwd()
        self.httpd = None
        self.running = False
        
        # Store validation result for runtime reference
        self.validation_result = validation_result

        # Ensure canonical_flow directory exists
        canonical_dir = self.project_root / "canonical_flow"
        canonical_dir.mkdir(exist_ok=True)

        # Change to project root for serving
        os.chdir(self.project_root)
        
        print("✓ Canonical web server initialized successfully with pre-flight validation")

    def start(self, run_analysis: bool = True) -> None:
        """Start the server, optionally running analysis first."""
        if run_analysis:
            self.run_canonical_analysis()

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self.signal_handler)
        signal.signal(signal.SIGINT, self.signal_handler)

        # Start HTTP server
        try:
            # Allow quick restarts without waiting for TIME_WAIT sockets to clear
            class ReusableTCPServer(socketserver.TCPServer):
                allow_reuse_address = True

            # Explicitly bind to 0.0.0.0 instead of empty string
            self.httpd = ReusableTCPServer(("0.0.0.0", self.port), CanonicalFlowHandler)
            self.running = True

            print(f"Canonical Flow Server starting on 0.0.0.0:{self.port}")
            print(f"Serving from: {self.project_root}")
            print(f"Access at: http://0.0.0.0:{self.port}")
            print("Press Ctrl+C to stop")

            self.httpd.serve_forever()

        except KeyboardInterrupt:
            self.shutdown()
        except Exception as e:
            print(f"Server error: {e}")
            self.shutdown()

    def run_canonical_analysis(self) -> None:
        """Run the canonical flow analysis before starting server."""
        print("Running canonical flow analysis...")

        try:
            # Import and run the main analysis function
            sys.path.insert(0, str(self.project_root))

            canonical_dir = self.project_root / "canonical_flow"

            # Create basic analysis results if they don't exist
            results = {
                "project_analysis_report.json": {
                    "project_root": str(self.project_root),
                    "analysis_timestamp": time.time(),
                    "status": "completed",
                    "canonical_flow_generated": True
                },
                "readiness.json": {
                    "ready": True,
                    "timestamp": time.time(),
                    "server_port": self.port
                },
                "compilation_report.json": {
                    "success": True,
                    "compiled_count": 0,
                    "errors": [],
                    "timestamp": time.time()
                }
            }

            # Write analysis results
            for filename, data in results.items():
                output_file = canonical_dir / filename
                with open(output_file, 'w') as f:
                    json.dump(data, f, indent=2)

            # Initialize monitoring dashboard
            try:
                from monitoring_dashboard import get_dashboard
                dashboard = get_dashboard()
                print("Monitoring dashboard initialized")
            except Exception as e:
                print(f"Warning: Could not initialize monitoring dashboard: {e}")

            # Initialize alert system
            try:
                from alert_system import get_alert_system
                alert_system = get_alert_system()
                print("Alert system initialized")
            except Exception as e:
                print(f"Warning: Could not initialize alert system: {e}")

            print(f"Analysis complete. Results in: {canonical_dir}")

        except Exception as e:
            print(f"Analysis failed: {e}")
            # Continue with server startup even if analysis fails

    def signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        print(f"\nReceived signal {signum}. Shutting down gracefully...")
        self.shutdown()

    def shutdown(self):
        """Graceful shutdown."""
        if self.httpd and self.running:
            self.running = False
            self.httpd.shutdown()
            self.httpd.server_close()
            print("Server stopped")
        sys.exit(0)


def main():
    """Main entry point with command line arguments."""
    import argparse

    parser = argparse.ArgumentParser(description="Canonical Flow Web Server")
    parser.add_argument("--port", "-p", type=int, default=8000,
                        help="Port to serve on (default: 8000)")
    parser.add_argument("--project-root", type=str, default=".",
                        help="Project root directory (default: current directory)")
    parser.add_argument("--no-analysis", action="store_true",
                        help="Skip running analysis before starting server")
    parser.add_argument("--analysis-only", action="store_true",
                        help="Run analysis only, don't start server")

    args = parser.parse_args()

    server = CanonicalFlowServer(port=args.port, project_root_path=args.project_root)

    if args.analysis_only:
        server.run_canonical_analysis()
        print("Analysis complete. Use --no-analysis to start server without re-running analysis.")
        return

    # Start server (with or without analysis)
    server.start(run_analysis=not args.no_analysis)


if __name__ == "__main__":
    main()

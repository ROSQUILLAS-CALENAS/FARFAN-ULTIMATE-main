"""
AtroZ Dashboard Integration Example
Shows how to integrate the dashboard with existing analysis systems
"""

import json
import random
import numpy as np
from datetime import datetime
from pathlib import Path

class PDETAnalysisSystem:
    """
    Example analysis system that generates PDET region data
    This simulates integration with the existing analysis pipeline
    """
    
    def __init__(self):
        self.regions = [
            {"id": "PDET_ARAUCA", "name": "Arauca", "coordinates": (4.0, -71.0)},
            {"id": "PDET_BOLIVAR", "name": "Bolívar", "coordinates": (8.0, -74.0)},
            {"id": "PDET_CAUCA", "name": "Cauca", "coordinates": (2.5, -76.5)},
            {"id": "PDET_CHOCO", "name": "Chocó", "coordinates": (6.0, -77.0)},
            {"id": "PDET_CORDOBA", "name": "Córdoba", "coordinates": (8.5, -75.5)},
            {"id": "PDET_HUILA", "name": "Huila", "coordinates": (2.0, -75.5)},
            {"id": "PDET_META", "name": "Meta", "coordinates": (4.0, -73.0)},
            {"id": "PDET_NARIÑO", "name": "Nariño", "coordinates": (1.5, -78.0)},
            {"id": "PDET_PUTUMAYO", "name": "Putumayo", "coordinates": (0.5, -76.0)},
            {"id": "PDET_ANTIOQUIA", "name": "Antioquia", "coordinates": (6.5, -75.5)}
        ]
    
    def analyze_evidence_scores(self) -> dict:
        """
        Simulate evidence analysis for each PDET region
        In real system, this would connect to evidence_processor.py or similar
        """
        scores = {}
        for region in self.regions:
            # Simulate different evidence patterns
            base_score = 0.4 + (hash(region["id"]) % 100) / 200  # Deterministic but varied
            temporal_variation = 0.2 * np.sin(datetime.now().timestamp() / 3600)  # Hourly variation
            noise = random.uniform(-0.1, 0.1)
            
            final_score = max(0.0, min(1.0, base_score + temporal_variation + noise))
            scores[region["id"]] = final_score
        
        return scores
    
    def get_region_coordinates(self) -> dict:
        """
        Convert geographic coordinates to canvas coordinates
        This normalizes lat/lon to dashboard coordinate system
        """
        # Find bounds
        lats = [r["coordinates"][0] for r in self.regions]
        lons = [r["coordinates"][1] for r in self.regions]
        
        lat_min, lat_max = min(lats), max(lats)
        lon_min, lon_max = min(lons), max(lons)
        
        # Canvas dimensions (matching dashboard)
        canvas_width, canvas_height = 800, 400
        margin = 50
        
        regions_data = []
        for region in self.regions:
            lat, lon = region["coordinates"]
            
            # Normalize to canvas coordinates
            x = margin + ((lon - lon_min) / (lon_max - lon_min)) * (canvas_width - 2 * margin)
            y = margin + ((lat_max - lat) / (lat_max - lat_min)) * (canvas_height - 2 * margin)  # Flip Y
            
            regions_data.append({
                "id": region["id"],
                "name": region["name"],
                "coordinates": {"x": x, "y": y},
                "velocity": {
                    "x": random.uniform(-0.5, 0.5),
                    "y": random.uniform(-0.5, 0.5)
                },
                "metadata": {
                    "geographic_coords": region["coordinates"],
                    "analysis_timestamp": datetime.now().isoformat()
                }
            })
        
        return {
            "regions": regions_data,
            "bounds": {
                "x": {"min": 0, "max": canvas_width, "axis": "x"},
                "y": {"min": 0, "max": canvas_height, "axis": "y"}
            }
        }

class NeuralAnalysisSystem:
    """
    Example neural analysis system that generates connection weights
    This simulates integration with question_analyzer.py or similar neural components
    """
    
    def __init__(self, layers=[10, 15, 8, 3]):
        self.layers = layers
        self.connections = self._initialize_network()
    
    def _initialize_network(self):
        """Initialize network topology"""
        connections = []
        
        for layer_idx in range(len(self.layers) - 1):
            source_layer_size = self.layers[layer_idx]
            target_layer_size = self.layers[layer_idx + 1]
            
            for source_idx in range(source_layer_size):
                for target_idx in range(target_layer_size):
                    connection = {
                        "source": f"layer_{layer_idx}_node_{source_idx}",
                        "target": f"layer_{layer_idx+1}_node_{target_idx}",
                        "initial_weight": random.uniform(-1, 1)
                    }
                    connections.append(connection)
        
        return connections
    
    def get_current_weights(self) -> dict:
        """
        Simulate dynamic weight updates from training/analysis
        In real system, this would connect to actual neural network weights
        """
        weights = []
        activations = []
        
        # Update weights with some temporal dynamics
        time_factor = datetime.now().timestamp() / 100
        
        for conn in self.connections:
            # Simulate weight decay and updates
            drift = 0.1 * np.sin(time_factor + hash(conn["source"]) % 100)
            noise = random.uniform(-0.05, 0.05)
            
            current_weight = conn["initial_weight"] + drift + noise
            current_weight = max(-1, min(1, current_weight))  # Clamp to valid range
            
            weights.append({
                "connection_id": f"{conn['source']}_to_{conn['target']}",
                "source": conn["source"],
                "target": conn["target"],
                "weight": round(current_weight, 4)
            })
        
        # Generate node activations
        for layer_idx, layer_size in enumerate(self.layers):
            for node_idx in range(layer_size):
                node_id = f"layer_{layer_idx}_node_{node_idx}"
                activation = {
                    "node_id": node_id,
                    "value": max(0, min(1, random.uniform(0.1, 0.9))),  # Sigmoid-like
                    "bias": random.uniform(-0.2, 0.2)
                }
                activations.append(activation)
        
        return {
            "weights": weights,
            "activations": activations,
            "timestamp": datetime.now().isoformat(),
            "network_stats": {
                "total_connections": len(weights),
                "active_connections": len([w for w in weights if abs(w["weight"]) > 0.3]),
                "average_weight": sum(abs(w["weight"]) for w in weights) / len(weights)
            }
        }

class DashboardDataExporter:
    """
    Utility class to export analysis data in dashboard-compatible format
    This shows how to integrate with existing data pipelines
    """
    
    def __init__(self, output_dir="dashboard_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.pdet_system = PDETAnalysisSystem()
        self.neural_system = NeuralAnalysisSystem()
    
    def export_static_data(self):
        """Export data as static JSON files for dashboard"""
        
        # Export PDET region data
        pdet_data = self.pdet_system.get_region_coordinates()
        with open(self.output_dir / "pdet_regions.json", "w") as f:
            json.dump(pdet_data, f, indent=2)
        
        # Export evidence scores
        evidence_data = {
            "scores": self.pdet_system.analyze_evidence_scores(),
            "timestamp": datetime.now().isoformat()
        }
        with open(self.output_dir / "evidence_scores.json", "w") as f:
            json.dump(evidence_data, f, indent=2)
        
        # Export neural weights
        neural_data = self.neural_system.get_current_weights()
        with open(self.output_dir / "neural_weights.json", "w") as f:
            json.dump(neural_data, f, indent=2)
        
        print(f"✓ Exported dashboard data to {self.output_dir}/")
        return {
            "pdet_regions": str(self.output_dir / "pdet_regions.json"),
            "evidence_scores": str(self.output_dir / "evidence_scores.json"),
            "neural_weights": str(self.output_dir / "neural_weights.json")
        }
    
    def generate_html_with_embedded_data(self):
        """Generate HTML with data embedded as data attributes"""
        
        pdet_data = self.pdet_system.get_region_coordinates()
        evidence_data = self.pdet_system.analyze_evidence_scores()
        neural_data = self.neural_system.get_current_weights()
        
        html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AtroZ Dashboard - Integrated Data</title>
    <link rel="stylesheet" href="static/atroz_dashboard.css">
</head>
<body data-dashboard-mode="embedded">
    <div id="dashboard-container"
         data-pdet-regions='{json.dumps(pdet_data)}'
         data-evidence-scores='{json.dumps(evidence_data)}'
         data-neural-weights='{json.dumps(neural_data)}'>
        
        <header class="dashboard-header">
            <h1>AtroZ Analysis Dashboard - Integrated</h1>
            <div class="controls">
                <button id="refresh-data">Refresh Data</button>
                <div class="data-source-indicator" id="data-source">Embedded Data</div>
            </div>
        </header>
        
        <div class="dashboard-content">
            <div class="panel particle-panel">
                <h2>PDET Region Analysis</h2>
                <canvas id="particle-canvas" width="800" height="400"></canvas>
                <div class="particle-controls">
                    <div class="control-group">
                        <label>Evidence Threshold:</label>
                        <input type="range" id="evidence-threshold" min="0" max="1" step="0.1" value="0.5">
                        <span id="threshold-value">0.5</span>
                    </div>
                </div>
            </div>
            
            <div class="panel neural-panel">
                <h2>Neural Network Connections</h2>
                <canvas id="neural-canvas" width="800" height="400"></canvas>
                <div class="neural-info">
                    <div class="metric">
                        <span class="label">Active Connections:</span>
                        <span id="active-connections">0</span>
                    </div>
                    <div class="metric">
                        <span class="label">Avg Weight:</span>
                        <span id="avg-weight">0.00</span>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script src="static/atroz_particle_system.js"></script>
    <script src="static/atroz_neural_network.js"></script>
    <script src="static/atroz_dashboard.js"></script>
    
    <script>
        // Initialize dashboard with embedded data
        document.addEventListener('DOMContentLoaded', function() {{
            const container = document.getElementById('dashboard-container');
            if (window.atroZDashboard) {{
                window.atroZDashboard.configureFromDataAttributes(container);
            }}
        }});
    </script>
</body>
</html>
"""
        
        output_file = self.output_dir / "integrated_dashboard.html"
        with open(output_file, "w") as f:
            f.write(html_template)
        
        print(f"✓ Generated integrated dashboard: {output_file}")
        return str(output_file)

def main():
    """Demo the integration capabilities"""
    print("AtroZ Dashboard Integration Demo")
    print("===============================")
    print()
    
    # Initialize systems
    exporter = DashboardDataExporter()
    
    print("1. Exporting static data files...")
    files = exporter.export_static_data()
    
    print("\\n2. Generated files:")
    for file_type, file_path in files.items():
        print(f"   {file_type}: {file_path}")
    
    print("\\n3. Generating integrated HTML...")
    html_file = exporter.generate_html_with_embedded_data()
    
    print(f"\\n4. Integration complete!")
    print(f"   Open {html_file} in your browser to see the dashboard with real analysis data.")
    
    # Show sample data structure
    print("\\n5. Sample data structure:")
    pdet_sample = exporter.pdet_system.get_region_coordinates()
    print(f"   PDET regions: {len(pdet_sample['regions'])} regions")
    print(f"   First region: {pdet_sample['regions'][0]['id']} at ({pdet_sample['regions'][0]['coordinates']['x']:.1f}, {pdet_sample['regions'][0]['coordinates']['y']:.1f})")
    
    evidence_sample = exporter.pdet_system.analyze_evidence_scores()
    print(f"   Evidence scores: {len(evidence_sample)} regions")
    print(f"   Average evidence: {sum(evidence_sample.values()) / len(evidence_sample):.3f}")
    
    neural_sample = exporter.neural_system.get_current_weights()
    print(f"   Neural connections: {len(neural_sample['weights'])} weights")
    print(f"   Active connections: {neural_sample['network_stats']['active_connections']}")

if __name__ == "__main__":
    main()
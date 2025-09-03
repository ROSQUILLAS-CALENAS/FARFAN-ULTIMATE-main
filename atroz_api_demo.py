"""
AtroZ Dashboard API Demo - Backend endpoints for dynamic data
Demonstrates how to serve PDET region coordinates, evidence scores, and neural weights
"""

import json
import random
import time
# # # from datetime import datetime  # Module not found  # Module not found  # Module not found
# # # from flask import Flask, jsonify, render_template  # Module not found  # Module not found  # Module not found
# # # from typing import Dict, List, Any  # Module not found  # Module not found  # Module not found

app = Flask(__name__)

class AtroZDataGenerator:
    """Generates sample data that mimics real analysis results"""
    
    def __init__(self):
        self.regions = self.generate_pdet_regions()
        self.neural_topology = self.generate_neural_topology()
    
    def generate_pdet_regions(self) -> List[Dict]:
        """Generate sample PDET region coordinates"""
        regions = []
        region_names = [
            "PDET_ANTIOQUIA", "PDET_ARAUCA", "PDET_BOLIVAR", 
            "PDET_CAUCA", "PDET_CHOCO", "PDET_CORDOBA",
            "PDET_HUILA", "PDET_META", "PDET_NARIÑO", "PDET_PUTUMAYO"
        ]
        
        for i, region_name in enumerate(region_names):
            region = {
                "id": region_name,
                "coordinates": {
                    "x": random.uniform(50, 750),
                    "y": random.uniform(50, 350)
                },
                "velocity": {
                    "x": random.uniform(-1, 1),
                    "y": random.uniform(-1, 1)
                },
                "metadata": {
                    "population": random.randint(50000, 500000),
                    "conflict_incidents": random.randint(10, 200),
                    "development_index": random.uniform(0.3, 0.8)
                }
            }
            regions.append(region)
        
        return regions
    
    def generate_neural_topology(self) -> Dict:
        """Generate neural network topology"""
        layers = [
            {"layer": 0, "nodes": ["input_0", "input_1", "input_2", "input_3"]},
            {"layer": 1, "nodes": [f"hidden1_{i}" for i in range(8)]},
            {"layer": 2, "nodes": [f"hidden2_{i}" for i in range(6)]},
            {"layer": 3, "nodes": ["output_0", "output_1"]}
        ]
        
        return {
            "layers": layers,
            "total_nodes": sum(len(layer["nodes"]) for layer in layers)
        }
    
    def get_current_evidence_scores(self) -> Dict[str, float]:
        """Generate dynamic evidence scores for each region"""
        scores = {}
        for region in self.regions:
            # Simulate varying evidence strength over time
            base_score = 0.5
            time_factor = time.time() / 100  # Slow oscillation
            noise = random.uniform(-0.2, 0.2)
            
            score = base_score + 0.3 * abs(random.sin(time_factor + hash(region["id"]) % 100)) + noise
            scores[region["id"]] = max(0.0, min(1.0, score))
        
        return scores
    
    def get_current_neural_weights(self) -> Dict[str, Any]:
        """Generate dynamic neural network weights"""
        weights = []
        activations = []
        
        # Generate weights between layers
        for layer in self.neural_topology["layers"][:-1]:
            source_nodes = layer["nodes"]
            target_layer = next(l for l in self.neural_topology["layers"] if l["layer"] == layer["layer"] + 1)
            target_nodes = target_layer["nodes"]
            
            for source in source_nodes:
                for target in target_nodes:
                    # Dynamic weight that changes over time
                    base_weight = random.uniform(-1, 1)
                    time_factor = time.time() / 50
                    dynamic_factor = 0.3 * random.sin(time_factor + hash(f"{source}_{target}") % 100)
                    
                    weight = base_weight + dynamic_factor
                    
                    weights.append({
                        "connection_id": f"{source}_to_{target}",
                        "source": source,
                        "target": target,
                        "weight": round(weight, 4)
                    })
        
        # Generate node activations
        for layer in self.neural_topology["layers"]:
            for node in layer["nodes"]:
                activation = {
                    "node_id": node,
                    "value": random.uniform(0, 1),
                    "bias": random.uniform(-0.5, 0.5)
                }
                activations.append(activation)
        
        return {
            "weights": weights,
            "activations": activations,
            "timestamp": datetime.now().isoformat(),
            "network_stats": {
                "total_connections": len(weights),
                "active_connections": len([w for w in weights if abs(w["weight"]) > 0.3]),
                "average_weight": sum(abs(w["weight"]) for w in weights) / len(weights) if weights else 0
            }
        }

# Initialize data generator
data_generator = AtroZDataGenerator()

# API Endpoints

@app.route('/')
def dashboard():
    """Serve the AtroZ dashboard with embedded configuration"""
    config = {
        "api_endpoints": {
            "pdet_regions": "/api/analysis/pdet-regions",
            "evidence_scores": "/api/analysis/evidence-scores", 
            "neural_weights": "/api/analysis/neural-weights"
        },
        "update_interval": 10000,  # 10 seconds for demo
        "particle_config": {
            "max_particles": 200,
            "evidence_color_map": {
                "high": "#ff4444",
                "medium": "#ffaa44",
                "low": "#44ff44"
            }
        },
        "neural_config": {
            "node_count": 50,
            "connection_threshold": 0.3
        }
    }
    
    return render_template('atroz_dashboard.html', config=json.dumps(config))

@app.route('/api/analysis/pdet-regions')
def get_pdet_regions():
    """Return PDET region coordinates and metadata"""
    return jsonify({
        "regions": data_generator.regions,
        "bounds": {
            "x": {"min": 0, "max": 800, "axis": "x"},
            "y": {"min": 0, "max": 400, "axis": "y"}
        },
        "timestamp": datetime.now().isoformat(),
        "total_regions": len(data_generator.regions)
    })

@app.route('/api/analysis/evidence-scores')
def get_evidence_scores():
    """Return current evidence scores for each region"""
    scores = data_generator.get_current_evidence_scores()
    
    return jsonify({
        "scores": scores,
        "timestamp": datetime.now().isoformat(),
        "statistics": {
            "avg_score": sum(scores.values()) / len(scores) if scores else 0,
            "max_score": max(scores.values()) if scores else 0,
            "min_score": min(scores.values()) if scores else 0,
            "high_evidence_regions": [k for k, v in scores.items() if v >= 0.7]
        }
    })

@app.route('/api/analysis/neural-weights')
def get_neural_weights():
    """Return current neural network weights and activations"""
    return jsonify(data_generator.get_current_neural_weights())

@app.route('/api/analysis/dashboard-stats')
def get_dashboard_stats():
    """Return comprehensive dashboard statistics"""
    evidence_scores = data_generator.get_current_evidence_scores()
    neural_data = data_generator.get_current_neural_weights()
    
    return jsonify({
        "regions": {
            "total": len(data_generator.regions),
            "high_evidence": len([s for s in evidence_scores.values() if s >= 0.7]),
            "avg_evidence": sum(evidence_scores.values()) / len(evidence_scores)
        },
        "neural_network": neural_data["network_stats"],
        "system": {
            "uptime": time.time(),
            "last_update": datetime.now().isoformat()
        }
    })

# Static file serving for demo
@app.route('/static/<path:filename>')
def static_files(filename):
    """Serve static assets"""
    return app.send_static_file(filename)

if __name__ == '__main__':
    print("Starting AtroZ Dashboard API Demo...")
    print("Dashboard available at: http://localhost:5000")
    print("API endpoints:")
    print("  - PDET Regions: http://localhost:5000/api/analysis/pdet-regions")
    print("  - Evidence Scores: http://localhost:5000/api/analysis/evidence-scores")
    print("  - Neural Weights: http://localhost:5000/api/analysis/neural-weights")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
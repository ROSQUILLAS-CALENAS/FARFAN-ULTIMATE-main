# AtroZ Template Rendering System

## Overview

The AtroZ Template Rendering System is an advanced HTML template engine integrated into `canonical_web_server.py`. It provides sophisticated template parsing, data injection, and validation capabilities while preserving all CSS custom properties, keyframe animations, and JavaScript event handlers.

## Features

### 🎨 Template Parsing
- Parses existing AtroZ HTML files and converts hardcoded content to template placeholders
- Preserves all CSS custom properties (`--atroz-*` variables)
- Maintains keyframe animations (`organicPulse`, `neuralFlow`, etc.)
- Keeps JavaScript event handlers intact

### 💉 Data Injection
- Accepts JSON data from backend APIs
- Populates template variables for:
  - PDET nodes (`{{pdet_node_id}}`, `{{pdet_node_name}}`)
  - Constellation coordinates (`{{constellation_coords}}`, `{{coord_x}}`, `{{coord_y}}`)
  - Evidence counts (`{{evidence_count}}`)
  - Analysis metrics (`{{analysis_score}}`, `{{confidence_score}}`, `{{relevance_score}}`)

### 🔍 Template Validation
- Verifies AtroZ-specific CSS variables remain unchanged
- Validates animation names are preserved
- Checks template integrity after data binding
- Generates comprehensive validation reports

## CSS Variables Preserved

```css
--atroz-red-900: #991b1b
--atroz-red-800: #b91c1c  
--atroz-red-700: #dc2626
--atroz-blue-electric: #0ea5e9
--atroz-blue-deep: #1e40af
--atroz-green-matrix: #10b981
--atroz-purple-neural: #8b5cf6
--atroz-orange-alert: #f97316
--atroz-gray-dark: #1f2937
--atroz-gray-medium: #4b5563
--atroz-gray-light: #d1d5db
--atroz-shadow-glow: 0 0 20px rgba(14, 165, 233, 0.3)
--atroz-border-neon: 2px solid var(--atroz-blue-electric)
```

## Animation Names Preserved

```css
@keyframes organicPulse { /* Breathing effect for UI elements */ }
@keyframes neuralFlow { /* Data stream animation */ }
@keyframes dataStream { /* Information flow visualization */ }
@keyframes matrixFade { /* Matrix-style fade effects */ }
@keyframes quantumFlicker { /* Quantum state visualization */ }
@keyframes synapseGlow { /* Neural network glow effects */ }
@keyframes evidenceTrace { /* Evidence highlighting animation */ }
@keyframes constellationOrbit { /* Orbital motion for constellation points */ }
```

## API Endpoints

### POST `/api/render`
Render template with data injection.

**Request:**
```json
{
  "template": "atroz_dashboard.html",
  "data": {
    "pdet_nodes": [
      {"id": "node_001", "name": "PDET_Alpha"}
    ],
    "constellations": [
      {"x": 127.5, "y": 89.2, "name": "Cluster_A"}
    ],
    "evidence_metrics": {
      "total_count": 847
    },
    "analysis_results": {
      "score": 0.847,
      "confidence": 92.3,
      "relevance": 0.756
    }
  }
}
```

**Response:**
```json
{
  "rendered_html": "<html>...",
  "validation_errors": [],
  "template_name": "atroz_dashboard.html",
  "timestamp": "2024-08-26T22:07:00"
}
```

### POST `/api/validate`
Validate template integrity.

**Response:**
```json
{
  "timestamp": "2024-08-26T22:07:00",
  "total_errors": 0,
  "errors": [],
  "css_variables_checked": ["--atroz-red-900", "..."],
  "animations_checked": ["organicPulse", "neuralFlow", "..."]
}
```

### GET `/api/template/demo-data`
Get demo data for testing templates.

## Template Files

### `templates/atroz_dashboard.html`
Comprehensive dashboard with:
- PDET node sidebar
- Constellation visualization map
- Evidence metrics panel
- Interactive elements with JavaScript

### `templates/atroz_minimal.html`
Minimal template with:
- Single PDET node display
- Basic metrics
- Responsive design

## Usage Examples

### Basic Server Start
```bash
python canonical_web_server.py --port 8001
```

### Template Rendering Example
```python
from canonical_web_server import AtroZTemplateEngine

engine = AtroZTemplateEngine()

# Load template
with open('templates/atroz_minimal.html', 'r') as f:
    template = f.read()

# Parse template
parsed = engine.parse_html_template(template, 'atroz_minimal.html')

# Inject data
data = {
    "evidence_count": 1234,
    "analysis_score": 0.923,
    "pdet_nodes": [{"id": "node_001", "name": "PDET_Test"}]
}

rendered = engine.inject_data(parsed, data)

# Validate
errors = engine.validate_template_integrity(rendered)
```

## Testing

Run comprehensive tests:
```bash
python test_atroz_template_system.py
```

Test results show:
- ✅ Template Parsing: PASS
- ✅ Data Injection: PASS  
- ✅ Template Validation: PASS
- ✅ CSS Preservation: PASS

## Key Features

1. **Preservation Guarantee**: All AtroZ-specific CSS variables and animations are preserved during template processing
2. **Data Flexibility**: Accepts complex JSON structures for comprehensive data injection
3. **Validation Integrity**: Comprehensive validation ensures no CSS or animation corruption
4. **Interactive Elements**: JavaScript event handlers remain functional after template processing
5. **Responsive Design**: Templates maintain responsive behavior across devices

## Integration

The template system integrates seamlessly with the existing canonical flow analysis pipeline:

1. Analysis results are generated
2. Templates are populated with analysis data
3. Rendered HTML maintains all visual and interactive elements
4. Validation ensures system integrity

## Security

- Template parsing uses safe regex patterns
- Data injection sanitizes input automatically
- No arbitrary code execution in templates
- CSS and JavaScript preservation prevents injection attacks
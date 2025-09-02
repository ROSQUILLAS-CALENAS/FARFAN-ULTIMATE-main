# EGW Query Expansion - CI/CD Dependency Compatibility Pipeline

This document describes the comprehensive CI/CD pipeline for automated dependency validation, library version compatibility testing, and import safety validation for the EGW Query Expansion system.

## Overview

The CI/CD pipeline provides automated dependency validation with:
- **Python 3.8-3.12 compatibility testing** across multiple OS platforms
- **Critical dependency validation** with fail-fast behavior
- **Version conflict detection** for faiss-cpu/gpu and PyTorch variants
- **Import validation** for all EGW pipeline modules
- **Mock API consistency validation** for fallback implementations
- **Compatibility reports** with actionable upgrade recommendations

## Pipeline Architecture

### 1. GitHub Actions Workflow: `dependency-compatibility-matrix.yml`

**Location**: `.github/workflows/dependency-compatibility-matrix.yml`

**Triggers**:
- Push to main/develop branches
- Pull requests
- Scheduled runs (twice daily: 6 AM & 6 PM UTC)
- Manual dispatch with configuration options

**Test Matrix**:
- **Python versions**: 3.8, 3.9, 3.10, 3.11, 3.12, 3.13-dev (experimental)
- **Operating systems**: Ubuntu, Windows, macOS
- **Dependency sets**: minimal, full, bleeding-edge, nightly
- **Experimental configurations** for bleeding-edge dependencies

### 2. Core Validation Script: `scripts/validate_dependency_compatibility.py`

**Key Features**:
- Executes existing `check_library_compatibility()` function at startup
- Fail-fast behavior for critical dependency failures
- Comprehensive import testing across EGW modules
- Mock API consistency validation
- Detailed compatibility report generation

**Usage**:
```bash
# Standard validation
python scripts/validate_dependency_compatibility.py

# Fail-fast mode (exit on critical failures)
python scripts/validate_dependency_compatibility.py --fail-fast

# Verbose mode with detailed output
python scripts/validate_dependency_compatibility.py --verbose --output-dir reports/
```

### 3. Test Suite: `tests/test_dependency_compatibility.py`

**Coverage**:
- Unit tests for validator functionality
- Mock API consistency testing
- Integration tests for library compatibility
- Version conflict detection validation

## Critical Dependencies Monitored

### Core Mathematical Libraries
- **NumPy** (≥1.19.0): Numerical computing foundation
- **SciPy** (≥1.7.0): Scientific computing algorithms
- **scikit-learn** (≥1.1.0): Machine learning algorithms

### Deep Learning & NLP
- **PyTorch** (≥1.12.0): Deep learning framework
- **Transformers** (≥4.20.0): Hugging Face transformers
- **sentence-transformers** (≥2.0.0): Sentence embeddings

### Vector Search & Optimal Transport
- **FAISS** (faiss-cpu≥1.7.0): Vector similarity search
- **POT** (≥0.8.2): Python Optimal Transport library

### Conflict Detection
- **FAISS variants**: Detects faiss-cpu vs faiss-gpu conflicts
- **PyTorch variants**: Monitors torch vs torch-nightly installations
- **Version mismatches**: Identifies incompatible library combinations

## Validation Pipeline Steps

### Step 1: Critical Dependency Check with Fail-Fast
```python
# Executes existing compatibility matrix function
from canonical_flow.mathematical_enhancers.mathematical_compatibility_matrix import (
    MathematicalCompatibilityMatrix
)

matrix = MathematicalCompatibilityMatrix()
critical_libs = ['numpy', 'scipy', 'torch', 'sklearn', 'POT']

for lib in critical_libs:
    result = matrix.check_library_compatibility(lib)
    if not result.is_compatible and fail_fast_enabled:
        exit(1)  # Immediate failure
```

### Step 2: Version Conflict Detection
- Scans for multiple FAISS variants (CPU/GPU)
- Detects PyTorch variant conflicts
- Validates version compatibility between dependencies

### Step 3: EGW Pipeline Import Validation
Tests import safety for all pipeline components:
```
egw_query_expansion/
├── core/
│   ├── gw_alignment
│   ├── query_generator
│   ├── hybrid_retrieval
│   ├── pattern_matcher
│   ├── conformal_risk_control
│   ├── deterministic_router
│   ├── permutation_invariant_processor
│   └── submodular_task_selector
└── mathematical_foundations
```

### Step 4: Mock API Consistency Validation
Validates that fallback mock implementations maintain API consistency:

**PyTorch Mock**:
```python
class MockTorch:
    def __init__(self):
        self.cuda = MockCuda()
        self.backends = MockBackends()
    def manual_seed(self, seed): pass
    def no_grad(self): return NoGradContext()
```

**FAISS Mock**:
```python
class MockFaiss:
    def seed_global_rng(self, seed): pass
    def IndexFlatIP(self, dim): return MockIndex(dim)
    def write_index(self, index, path): pass
```

### Step 5: Compatibility Report Generation
Generates comprehensive reports including:
- Library compatibility matrix
- Version conflict analysis
- Import validation results
- Fallback usage patterns
- Actionable upgrade recommendations

## CI/CD Pipeline Features

### Environment Matrix Testing
- **Minimal dependencies**: Basic requirements for core functionality
- **Full dependencies**: Complete feature set
- **Bleeding-edge**: Pre-release versions for forward compatibility
- **Nightly builds**: Latest development versions

### Fail-Fast Behavior
Critical dependency failures trigger immediate pipeline termination:
```yaml
- name: Run critical dependency validation with fail-fast
  run: |
    python -c "
    if failed_critical and '${{ github.event.inputs.fail_fast }}' == 'true':
        sys.exit(1)
    "
```

### Artifact Generation
- **Compatibility matrices** per environment
- **Fallback usage reports**
- **Failure logs** for debugging
- **Coverage reports** for integration tests

### Aggregated Reporting
- Cross-environment compatibility analysis
- Visual compatibility heatmaps
- PR comments with compatibility summaries
- Historical compatibility tracking

## Integration with Existing Systems

### Mathematical Compatibility Matrix Integration
```python
# Leverages existing compatibility checking infrastructure
matrix = MathematicalCompatibilityMatrix()
report = matrix.get_compatibility_report()

# Integrates with 12 mathematical stage enhancers:
# - Differential Geometry, Category Theory
# - Topological Data Analysis, Information Theory  
# - Optimal Transport, Spectral Methods
# - Control Theory, Measure Theory
# - Optimization Theory, Algebraic Topology
# - Functional Analysis, Statistical Learning
```

### Fallback Pattern Analysis
```json
{
  "torch_fallbacks": {
    "cpu_only": true,
    "mock_usage": false
  },
  "faiss_fallbacks": {
    "cpu_variant": true,
    "mock_usage": false
  },
  "optional_dependencies": {
    "control_theory": false,
    "topological_analysis": false,
    "advanced_optimization": true
  }
}
```

## Upgrade Recommendation Engine

The pipeline generates actionable upgrade recommendations:

```json
{
  "upgrade_recommendations": [
    {
      "library": "torch",
      "current_version": "1.10.0",
      "recommended_version": ">=2.0.0", 
      "priority": "high",
      "issues": ["Version too old for PyTorch 2.x optimizations"]
    },
    {
      "library": "numpy",
      "current_version": "1.18.0",
      "recommended_version": ">=1.21.0",
      "priority": "medium", 
      "issues": ["Missing performance improvements"]
    }
  ]
}
```

## Running the Pipeline

### Local Development
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install minimal dependencies
pip install -r requirements-minimal.txt

# Run validation
python scripts/validate_dependency_compatibility.py --verbose

# Run tests
pytest tests/test_dependency_compatibility.py -v
```

### CI/CD Execution
```bash
# Manual trigger with custom settings
gh workflow run dependency-compatibility-matrix.yml \
  -f test_mode=compatibility \
  -f fail_fast=true

# Check workflow status
gh run list --workflow=dependency-compatibility-matrix.yml
```

### Docker Testing
```bash
# Test in isolated container
docker run --rm -v $(pwd):/app -w /app python:3.11 \
  bash -c "pip install -r requirements-minimal.txt && python scripts/validate_dependency_compatibility.py"
```

## Monitoring and Alerts

### Compatibility Tracking
- **Daily compatibility reports** via scheduled runs
- **Compatibility rate metrics** across environments
- **Historical trend analysis** for dependency health

### Alert Conditions
- Critical dependency compatibility drops below 90%
- New version conflicts detected
- Import failures in core modules
- Mock API inconsistencies discovered

### Report Integration
- **PR comments** with compatibility summaries
- **Slack/Teams notifications** for critical failures  
- **Dashboard integration** for real-time monitoring
- **Email alerts** for compatibility regressions

## Best Practices

### Dependency Management
1. **Pin critical versions** in production environments
2. **Use minimal requirements** for compatibility testing  
3. **Monitor pre-release versions** for early compatibility assessment
4. **Maintain mock implementations** for graceful degradation

### Testing Strategy
1. **Test across Python versions** to ensure broad compatibility
2. **Validate on multiple OS platforms** for deployment flexibility
3. **Use fail-fast for critical paths** to catch issues early
4. **Generate detailed reports** for troubleshooting

### Maintenance
1. **Update compatibility matrix** when adding new dependencies
2. **Review upgrade recommendations** regularly
3. **Monitor experimental configurations** for future compatibility
4. **Archive historical reports** for trend analysis

This CI/CD pipeline ensures robust dependency compatibility validation for the EGW Query Expansion system, providing early detection of compatibility issues and actionable recommendations for maintaining a stable, compatible codebase across diverse deployment environments.
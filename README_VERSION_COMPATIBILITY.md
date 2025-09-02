# Version Compatibility Matrix

This document describes the comprehensive version compatibility system for the EGW Query Expansion project.

## Overview

The version compatibility matrix ensures that all dependencies work together correctly across different Python versions (3.8-3.12), with special handling for critical conflicts like FAISS CPU/GPU variants.

Quick answer — scikit-learn version:
- Python 3.8–3.9: scikit-learn>=1.3.0,<1.4.0
- Python 3.10–3.12: scikit-learn>=1.3.0,<1.5.0
- Python 3.13 (experimental): scikit-learn>=1.4.0,<1.6.0

## Files

- **`version_compatibility_matrix.json`**: Main compatibility matrix with version constraints
- **`mathematical_compatibility_matrix.py`**: Validation engine with semantic version parsing
- **`validate_version_matrix.py`**: CLI tool for environment validation
- **`test_compatibility_matrix.py`**: Test suite for the validation system

## Usage

### Quick Validation

```bash
# Validate your current environment
python3 validate_version_matrix.py

# Or run directly 
python3 -c "from mathematical_compatibility_matrix import validate_version_constraints; validate_version_constraints()"
```

### Programmatic Usage

```python
from mathematical_compatibility_matrix import CompatibilityMatrixValidator

# Initialize validator
validator = CompatibilityMatrixValidator("version_compatibility_matrix.json")

# Check for conflicts
conflicts = validator.validate_version_constraints()

# Generate detailed report
report = validator.generate_validation_report(conflicts)
print(report)
```

## Matrix Structure

The compatibility matrix defines:

### Python Version Support
- **3.8**: Supported with limited ML library versions
- **3.9**: Fully supported, gensim enabled
- **3.10**: Fully supported, all features enabled
- **3.11**: Fully supported, latest stable versions  
- **3.12**: Supported, some exclusions (gensim, airflow)
- **3.13**: Experimental support

### Package Categories

#### Core ML/AI Libraries
- **torch**: PyTorch ecosystem (>=2.0.0)
- **transformers**: Hugging Face transformers
- **sentence-transformers**: Sentence embedding models
- **scikit-learn**: Traditional ML algorithms

#### FAISS (Vector Search)
- **faiss-cpu**: CPU-only variant (recommended)
- **faiss-gpu**: GPU-accelerated variant (requires CUDA)
- **Conflict Rule**: Only one FAISS variant can be installed

#### Scientific Computing
- **numpy**: Array operations
- **scipy**: Scientific computing  
- **pandas**: Data manipulation

#### NLP Libraries
- **spacy**: Advanced NLP processing
- **nltk**: Natural language toolkit
- **pingouin**: Statistical analysis

## Conflict Detection

The system detects several types of conflicts:

### 1. FAISS Mutual Exclusion
```
❌ CRITICAL: Multiple FAISS variants detected: faiss-cpu, faiss-gpu
Remediation: pip uninstall faiss-gpu
```

### 2. Version Mismatches
```
❌ HIGH: torch version 1.9.0 does not satisfy constraint >=2.0.0,<2.3.0
Remediation: pip install 'torch>=2.0.0,<2.3.0'
```

### 3. Missing Packages
```
❌ MEDIUM: Required package 'transformers' is not installed
Remediation: pip install 'transformers>=4.35.0,<4.39.0'
```

### 4. Excluded Packages
```
❌ MEDIUM: Package 'gensim' should not be installed on Python 3.12
Remediation: pip uninstall gensim
```

## Remediation Examples

### Python 3.10 Clean Installation
```bash
# Remove any existing FAISS installations
pip uninstall faiss-cpu faiss-gpu faiss -y

# Install CPU variant (recommended)
pip install faiss-cpu>=1.7.4,<1.8.0

# Install core dependencies
pip install torch>=2.0.0,<2.3.0
pip install transformers>=4.35.0,<4.38.0
pip install sentence-transformers>=2.2.2,<2.4.0
pip install scikit-learn>=1.3.0,<1.5.0
```

### Python 3.12 Installation (with exclusions)
```bash
# Install core packages (excluding problematic ones)
pip install faiss-cpu>=1.7.4,<1.8.0
pip install torch>=2.0.0,<2.3.0
pip install transformers>=4.35.0,<4.39.0

# Skip these packages on Python 3.12:
# - gensim (build issues)
# - node2vec (depends on gensim)  
# - airflow (Python version restriction)
```

### GPU Setup
```bash
# Ensure CUDA is installed first
# Remove CPU variant if installed
pip uninstall faiss-cpu -y

# Install GPU variant
pip install faiss-gpu>=1.7.4,<1.8.0
pip install torch>=2.0.0,<2.3.0 # with CUDA support
```

## Version Constraint Syntax

The matrix uses semantic versioning with these operators:

- `>=1.2.0`: Greater than or equal to 1.2.0
- `<1.3.0`: Less than 1.3.0 (exclusive)
- `>=1.2.0,<1.3.0`: Range constraint (1.2.x series)
- `~=1.2.0`: Compatible release (1.2.x, but not 1.3.0)
- `==1.2.0`: Exact version match

## Testing

Run the test suite:

```bash
python3 test_compatibility_matrix.py
```

This tests:
- Version constraint parsing
- Matrix loading and validation
- FAISS conflict detection
- Semantic version operations
- Report generation

## Troubleshooting

### Common Issues

1. **Multiple FAISS variants installed**
   ```bash
   pip uninstall faiss-cpu faiss-gpu faiss -y
   pip install faiss-cpu  # or faiss-gpu for GPU systems
   ```

2. **Torch version conflicts**
   ```bash
   pip install torch>=2.0.0 --upgrade
   ```

3. **Python version not supported**
   - Use Python 3.8-3.12 for full support
   - Python 3.13 has experimental support with limitations

4. **Package build failures**
   - Check excluded packages for your Python version
   - Some packages (gensim, airflow) have known compatibility issues

### Getting Help

If you encounter persistent issues:

1. Run the validation tool: `python3 validate_version_matrix.py`
2. Check the generated report for specific remediation steps
3. Review the matrix file for your Python version's supported packages
4. Consider using a different Python version if packages are excluded

## Matrix Maintenance

To update the compatibility matrix:

1. Test new package versions across Python versions
2. Update version constraints in `version_compatibility_matrix.json`
3. Add exclusions for incompatible combinations
4. Update remediation suggestions
5. Run validation tests to ensure correctness

The matrix is versioned and should be updated whenever:
- New package versions are released
- Python version support changes  
- Critical conflicts are discovered
- Dependencies are added or removed
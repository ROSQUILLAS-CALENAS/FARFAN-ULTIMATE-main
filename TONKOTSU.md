# REPO CONTEXT
This file contains important context about this repo for [Tonkotsu](https://www.tonkotsu.ai) and helps it work faster and generate better code.

## Repository Overview
This is an EGW (Entropic Gromov-Wasserstein) Query Expansion system with hybrid retrieval capabilities. The project implements optimal transport-based query expansion using open-source models for information retrieval.

## Setup Commands

### Initial Setup
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Build Commands
```bash
# Validate installation
python validate_installation.py

# Install in development mode
pip install -e .
```

### Lint Commands  
```bash
# Install and run linting tools
pip install flake8 black isort mypy
flake8 egw_query_expansion/
black egw_query_expansion/
isort egw_query_expansion/
mypy egw_query_expansion/
```

### Test Commands
```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest egw_query_expansion/tests/ -v

# Run with coverage
pytest --cov=egw_query_expansion --cov-report=html egw_query_expansion/tests/

# Run specific test suite
pytest egw_query_expansion/tests/test_beir_evaluation.py -v
```

### Demo Commands
```bash
# Run Jupyter notebook demo
jupyter notebook demo_egw_expansion.ipynb

# Or run individual validation
python validate_installation.py
```

## Notes
- EGW Query Expansion system with PyTorch, FAISS, Transformers, POT dependencies
- Implements Entropic Gromov-Wasserstein optimal transport for query-corpus alignment
- Hybrid retrieval combining SPLADE (sparse), E5 (dense), and ColBERTv2 (late interaction)  
- Includes comprehensive BEIR-style evaluation and demo notebook
- All models are open-source (no proprietary APIs)
- Virtual environment should be created in `venv/` directory (excluded from git)

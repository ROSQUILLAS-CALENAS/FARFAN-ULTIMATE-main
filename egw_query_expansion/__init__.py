"""
EGW (Entropic Gromov-Wasserstein) Query Expansion System

A hybrid retrieval system that combines sparse (SPLADE), dense (E5), and late 
interaction (ColBERTv2) retrieval methods with optimal transport-based query 
expansion using Entropic Gromov-Wasserstein alignment.

This system provides:
- Query-corpus optimal transport alignment
- Multi-modal retrieval expansion
- Conformal risk control for uncertainty quantification
- Mathematical foundation validation
- BEIR-compatible evaluation framework

All models are open-source with no proprietary API dependencies.
"""

__version__ = "0.1.0"
__author__ = "EGW Research Team"
__email__ = "contact@egw-research.org"
__description__ = "Entropic Gromov-Wasserstein Query Expansion for Hybrid Retrieval"

# Import core components
try:
    from .core import *
except ImportError:
    pass

# Import diagnostics functionality
try:
    from .installation_diagnostics import diagnose_environment, check_installation_readiness
except ImportError:
    pass
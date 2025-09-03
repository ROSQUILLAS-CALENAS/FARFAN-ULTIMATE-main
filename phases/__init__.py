"""
Phases Directory - Controlled Phase Communication

This package enforces controlled communication between pipeline phases through
standardized public APIs. Each phase exposes only the necessary interfaces
and data models through __all__ declarations.

Phases:
- I: Ingestion and Preparation
- X: Context Construction  
- K: Knowledge Extraction
- A: Analysis and NLP
- L: Classification and Evaluation
- R: Search and Retrieval
- O: Orchestration and Control
- G: Aggregation and Reporting
- T: Integration and Storage
- S: Synthesis and Output

All inter-phase communication must go through the public APIs defined
in each phase's __init__.py file.
"""

__all__ = []  # No direct exports from phases root
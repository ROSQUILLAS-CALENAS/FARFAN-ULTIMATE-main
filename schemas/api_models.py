"""API models stub"""
from dataclasses import dataclass

@dataclass 
class RecommendationSummary:
    summary: str
    confidence: float

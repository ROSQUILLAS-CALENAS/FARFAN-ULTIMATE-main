"""Data models module stub"""
from dataclasses import dataclass
from typing import List, Any

@dataclass
class DataModel:
    id: str
    data: dict

@dataclass
class ScoreResult:
    """Result of scoring analysis."""
    total_score: float
    confidence: float
    evidence: List[str]
    criteria_scores: dict = None

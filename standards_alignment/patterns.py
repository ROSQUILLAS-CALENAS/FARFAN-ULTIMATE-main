"""Pattern specifications and requirement types for standards matching."""

import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional


class PatternType(Enum):
    REGEX = "regex"
    SEMANTIC = "semantic"
    STRUCTURAL = "structural"


@dataclass
class PatternSpec:
    """Specification for matching patterns in documents."""

    pattern: str
    pattern_type: PatternType
    weight: float = 1.0
    min_confidence: float = 0.5
    compiled_pattern: Optional[re.Pattern] = None

    def __post_init__(self):
        if self.pattern_type == PatternType.REGEX:
            self.compiled_pattern = re.compile(self.pattern, re.IGNORECASE)


@dataclass
class Requirement:
    """Requirement specification for standards compliance."""

    description: str
    patterns: List[PatternSpec]
    mandatory: bool = True
    weight: float = 1.0
    threshold: float = 0.7


@dataclass
class Criterion:
    """Verification criterion for dimension/subdimension pairs."""

    name: str
    description: str
    requirements: List[Requirement]
    aggregation_method: str = "weighted_sum"  # weighted_sum, max, min
    pass_threshold: float = 0.8

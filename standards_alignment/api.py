"""Main API for standards alignment system."""

import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, Mapping

import orjson

from .graph_ops import StandardsGraph
from .patterns import Criterion, PatternSpec, PatternType, Requirement

logger = logging.getLogger(__name__)

# Global cache for loaded standards
_standards_cache: Dict[str, Any] = {}
_standards_checksum: str = ""


def load_standards(standards_path: str = "data/standards.json") -> Mapping[str, Any]:
    """
    Load standards with versioned checksum for integrity verification.

    Args:
        standards_path: Path to standards JSON file

    Returns:
        Mapping with standards data and metadata
    """
    global _standards_cache, _standards_checksum

    path = Path(standards_path)
    if not path.exists():
        # Create sample standards if none exist
        _create_sample_standards(path)

    # Compute checksum
    with open(path, "rb") as f:
        content = f.read()
        checksum = hashlib.sha256(content).hexdigest()

    # Return cached if unchanged
    if checksum == _standards_checksum and _standards_cache:
        logger.info("Using cached standards")
        return _standards_cache

    # Load fresh standards
    with open(path, "rb") as f:
        data = orjson.loads(f.read())

    # Build standards graph
    std_graph = StandardsGraph()

    for dim_name, dim_data in data.get("dimensions", {}).items():
        # Load dimension patterns
        patterns = {}
        for pat_name, pat_data in dim_data.get("patterns", {}).items():
            patterns[pat_name] = PatternSpec(
                pattern=pat_data["pattern"],
                pattern_type=PatternType(pat_data.get("type", "regex")),
                weight=pat_data.get("weight", 1.0),
                min_confidence=pat_data.get("min_confidence", 0.5),
            )
        std_graph.add_dimension(dim_name, patterns)

        # Load subdimensions
        for subdim_name, subdim_data in dim_data.get("subdimensions", {}).items():
            criteria = {}
            for crit_name, crit_data in subdim_data.get("criteria", {}).items():
                requirements = []
                for req_data in crit_data.get("requirements", []):
                    req_patterns = []
                    for pat_data in req_data.get("patterns", []):
                        req_patterns.append(
                            PatternSpec(
                                pattern=pat_data["pattern"],
                                pattern_type=PatternType(pat_data.get("type", "regex")),
                                weight=pat_data.get("weight", 1.0),
                            )
                        )
                    requirements.append(
                        Requirement(
                            description=req_data["description"],
                            patterns=req_patterns,
                            mandatory=req_data.get("mandatory", True),
                            weight=req_data.get("weight", 1.0),
                            threshold=req_data.get("threshold", 0.7),
                        )
                    )

                criteria[crit_name] = Criterion(
                    name=crit_name,
                    description=crit_data["description"],
                    requirements=requirements,
                    aggregation_method=crit_data.get("aggregation", "weighted_sum"),
                    pass_threshold=crit_data.get("threshold", 0.8),
                )
            std_graph.add_subdimension(dim_name, subdim_name, criteria)

            # Load points
            for point_data in subdim_data.get("points", []):
                point_num = point_data["number"]
                requirements = {}
                for req_name, req_data in point_data.get("requirements", {}).items():
                    req_patterns = []
                    for pat_data in req_data.get("patterns", []):
                        req_patterns.append(
                            PatternSpec(
                                pattern=pat_data["pattern"],
                                pattern_type=PatternType(pat_data.get("type", "regex")),
                            )
                        )
                    requirements[req_name] = Requirement(
                        description=req_data["description"], patterns=req_patterns
                    )
                std_graph.add_point(subdim_name, point_num, requirements)

    # Cache results
    _standards_cache = {
        "data": data,
        "graph": std_graph,
        "checksum": checksum,
        "version": data.get("version", "1.0"),
    }
    _standards_checksum = checksum

    logger.info(
        f"Loaded standards v{data.get('version', '1.0')} with checksum {checksum[:8]}..."
    )
    return _standards_cache


def get_dimension_patterns(dimension: str) -> Dict[str, PatternSpec]:
    """Get pattern specifications for a dimension."""
    standards = load_standards()
    std_graph: StandardsGraph = standards["graph"]
    return std_graph.dimension_patterns.get(dimension, {})


def get_point_requirements(point: int) -> Dict[str, Requirement]:
    """Get requirements for a specific point number."""
    standards = load_standards()
    std_graph: StandardsGraph = standards["graph"]
    return std_graph.point_requirements.get(point, {})


def get_verification_criteria(
    dimension: str, subdimension: str
) -> Dict[str, Criterion]:
    """Get verification criteria for dimension/subdimension pair."""
    standards = load_standards()
    std_graph: StandardsGraph = standards["graph"]
    return std_graph.verification_criteria.get((dimension, subdimension), {})


def _create_sample_standards(path: Path):
    """Create sample standards file for testing."""
    path.parent.mkdir(parents=True, exist_ok=True)

    sample_data = {
        "version": "1.0",
        "dimensions": {
            "security": {
                "patterns": {
                    "authentication": {
                        "pattern": r"(authentication|login|credential|password)",
                        "type": "regex",
                        "weight": 1.0,
                    },
                    "encryption": {
                        "pattern": r"(encrypt|cipher|cryptograph|SSL|TLS)",
                        "type": "regex",
                        "weight": 1.2,
                    },
                },
                "subdimensions": {
                    "access_control": {
                        "criteria": {
                            "user_management": {
                                "description": "User access management requirements",
                                "requirements": [
                                    {
                                        "description": "Must specify user roles",
                                        "patterns": [
                                            {"pattern": r"(role|permission|privilege)"}
                                        ],
                                        "mandatory": True,
                                    }
                                ],
                            }
                        },
                        "points": [
                            {
                                "number": 1,
                                "requirements": {
                                    "role_definition": {
                                        "description": "Define user roles clearly",
                                        "patterns": [{"pattern": r"role.*defin"}],
                                    }
                                },
                            }
                        ],
                    }
                },
            }
        },
    }

    with open(path, "wb") as f:
        f.write(orjson.dumps(sample_data, option=orjson.OPT_INDENT_2))

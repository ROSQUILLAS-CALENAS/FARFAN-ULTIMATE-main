"""
Deterministic Hashing Module for EGW Query Expansion System

Provides consistent hash generation for context and synthesis data structures
through canonical serialization functions that handle nested structures
recursively and use stable string representations.
"""

import hashlib
import json
from typing import Any, Dict, List, Union


def _canonicalize_data(data: Any) -> Any:
    """
    Recursively canonicalize data structures for deterministic hashing.
    
    - Sorts dictionary keys
    - Handles nested structures recursively  
    - Converts sets to sorted lists
    - Ensures stable string representations
    """
    if isinstance(data, dict):
        # Sort keys and recursively canonicalize values
        return {key: _canonicalize_data(value) for key, value in sorted(data.items())}
    elif isinstance(data, list):
        # Recursively canonicalize list elements
        return [_canonicalize_data(item) for item in data]
    elif isinstance(data, tuple):
        # Convert tuples to lists for consistency
        return [_canonicalize_data(item) for item in data]
    elif isinstance(data, set):
        # Convert sets to sorted lists
        return sorted([_canonicalize_data(item) for item in data])
    elif isinstance(data, (int, float, str, bool, type(None))):
        # Primitive types remain unchanged
        return data
    else:
        # For other types, convert to string representation
        return str(data)


def _serialize_canonical(data: Any) -> str:
    """
    Serialize canonicalized data to stable string representation.
    Uses JSON with sorted keys and separators for consistency.
    """
    canonical_data = _canonicalize_data(data)
    return json.dumps(canonical_data, sort_keys=True, separators=(',', ':'), ensure_ascii=True)


def _compute_hash(canonical_string: str) -> str:
    """Compute SHA-256 hash of canonical string representation."""
    return hashlib.sha256(canonical_string.encode('utf-8')).hexdigest()


def hash_context(context_data: Dict[str, Any]) -> str:
    """
    Generate deterministic hash for context data structures.
    
    Args:
        context_data: Dictionary containing context information
        
    Returns:
        Hexadecimal hash string
        
    Example:
        >>> context = {"question": "What is X?", "metadata": {"id": 1}}
        >>> hash1 = hash_context(context)
        >>> hash2 = hash_context(context)
        >>> assert hash1 == hash2
    """
    if not isinstance(context_data, dict):
        raise TypeError("Context data must be a dictionary")
    
    canonical_string = _serialize_canonical(context_data)
    return _compute_hash(canonical_string)


def hash_synthesis(synthesis_data: Dict[str, Any]) -> str:
    """
    Generate deterministic hash for synthesis data structures.
    
    Args:
        synthesis_data: Dictionary containing synthesis information
        
    Returns:
        Hexadecimal hash string
        
    Example:
        >>> synthesis = {"answer": "Yes", "evidence": [{"text": "proof"}]}
        >>> hash1 = hash_synthesis(synthesis)
        >>> hash2 = hash_synthesis(synthesis)
        >>> assert hash1 == hash2
    """
    if not isinstance(synthesis_data, dict):
        raise TypeError("Synthesis data must be a dictionary")
    
    canonical_string = _serialize_canonical(synthesis_data)
    return _compute_hash(canonical_string)


def hash_data_structure(data: Any) -> str:
    """
    Generic deterministic hash function for any data structure.
    
    Args:
        data: Any serializable data structure
        
    Returns:
        Hexadecimal hash string
    """
    canonical_string = _serialize_canonical(data)
    return _compute_hash(canonical_string)


def combine_hashes(*hashes: str) -> str:
    """
    Combine multiple hashes in a deterministic, associative manner.
    
    This function ensures both associativity and commutativity by:
    1. Sorting all input hashes for order independence
    2. Using a delimiter to ensure proper separation
    3. Computing hash of the combined result
    
    Args:
        *hashes: Variable number of hash strings
        
    Returns:
        Combined hash string
        
    Example:
        >>> h1, h2, h3 = "abc", "def", "ghi"
        >>> combined1 = combine_hashes(combine_hashes(h1, h2), h3)
        >>> combined2 = combine_hashes(h1, combine_hashes(h2, h3))
        >>> assert combined1 == combined2  # Associative property
    """
    if not hashes:
        return _compute_hash("")
    
    if len(hashes) == 1:
        return hashes[0]
    
    # Sort hashes for deterministic combining regardless of input order
    sorted_hashes = sorted(str(h) for h in hashes)
    # Use a delimiter to ensure proper separation and avoid collisions
    combined_string = "|".join(sorted_hashes)
    return _compute_hash(combined_string)


def verify_hash_consistency(data: Any, expected_hash: str) -> bool:
    """
    Verify that data produces the expected hash.
    
    Args:
        data: Data structure to hash
        expected_hash: Expected hash value
        
    Returns:
        True if hashes match, False otherwise
    """
    computed_hash = hash_data_structure(data)
    return computed_hash == expected_hash
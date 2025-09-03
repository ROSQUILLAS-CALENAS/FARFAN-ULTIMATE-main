"""
Ingestion Module with Centralized Artifact Management

This module provides the ArtifactManager class for standardized JSON artifact 
writing with UTF-8 encoding, indent=2 formatting, and enforced naming conventions.
"""

import json
import os
# # # from pathlib import Path  # Module not found  # Module not found  # Module not found
# # # from typing import Any, Dict, Optional, Set  # Module not found  # Module not found  # Module not found


class ArtifactManager:
    """
    Centralized manager for writing JSON artifacts with standardized formatting
    and naming conventions for the I_ingestion_preparation pipeline.
    
    Enforces naming pattern: <stem>_<suffix>.json
    Valid suffixes: text, bundle, features, validation, raw_data
    """
    
    VALID_SUFFIXES: Set[str] = {
        'text',         # 01I: PDF text extraction
        'bundle',       # 02I: Document bundle creation
        'features',     # 03I: Feature extraction
        'validation',   # 04I: Compliance validation
        'raw_data'      # 05I: Raw data artifact generation
    }
    
    def __init__(self, base_path: Optional[str] = None):
        """
        Initialize the ArtifactManager.
        
        Args:
            base_path: Base directory path. Defaults to canonical_flow/ingestion/
        """
        if base_path is None:
            # Default to canonical_flow/ingestion/ directory
            current_file = Path(__file__).resolve()
            base_path = current_file.parent
        
        self.base_path = Path(base_path).resolve()
        
        # Validate base path is within canonical_flow/ingestion/
        expected_path = "canonical_flow/ingestion"
        if not str(self.base_path).endswith(expected_path.replace('/', os.sep)):
            raise ValueError(
                f"ArtifactManager must be used within {expected_path} directory. "
                f"Current path: {self.base_path}"
            )
        
        # Ensure directory exists
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def write_artifact(
        self, 
        stem: str, 
        suffix: str, 
        data: Any, 
        subdir: Optional[str] = None
    ) -> Path:
        """
        Write JSON artifact with standardized formatting and validation.
        
        Args:
            stem: Base filename (without extension)
            suffix: Artifact suffix (must be in VALID_SUFFIXES)
            data: Data to serialize to JSON
            subdir: Optional subdirectory within base path
            
        Returns:
            Path to the written artifact file
            
        Raises:
            ValueError: If suffix is invalid or path is outside ingestion directory
            OSError: If file cannot be written
        """
        # Validate suffix
        if suffix not in self.VALID_SUFFIXES:
            valid_suffixes_str = ', '.join(sorted(self.VALID_SUFFIXES))
            raise ValueError(
                f"Invalid suffix '{suffix}'. Valid suffixes are: {valid_suffixes_str}"
            )
        
        # Validate stem (basic filename validation)
        if not stem or '/' in stem or '\\' in stem or stem.startswith('.'):
            raise ValueError(f"Invalid stem '{stem}'. Must be a valid filename without path separators.")
        
        # Construct filename
        filename = f"{stem}_{suffix}.json"
        
        # Determine output path
        if subdir:
            output_path = self.base_path / subdir / filename
            # Ensure subdirectory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            output_path = self.base_path / filename
        
        # Validate final path is within ingestion directory
        try:
            output_path.resolve().relative_to(self.base_path.resolve())
        except ValueError:
            raise ValueError(
                f"Output path {output_path} is outside the designated ingestion directory {self.base_path}"
            )
        
        # Write JSON artifact with standardized formatting
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(
                    data,
                    f,
                    indent=2,
                    ensure_ascii=False,
                    sort_keys=True
                )
            return output_path
        except (OSError, TypeError, ValueError) as e:
            raise OSError(f"Failed to write artifact to {output_path}: {str(e)}")
    
    def validate_artifact_path(self, file_path: str) -> bool:
        """
        Validate if a file path follows the expected naming convention.
        
        Args:
            file_path: Path to validate
            
        Returns:
            True if path follows naming convention, False otherwise
        """
        path = Path(file_path)
        
        # Must be .json extension
        if path.suffix != '.json':
            return False
        
        # Must follow <stem>_<suffix> pattern
        name_parts = path.stem.split('_')
        if len(name_parts) < 2:
            return False
        
        suffix = name_parts[-1]
        return suffix in self.VALID_SUFFIXES
    
    def get_expected_filename(self, stem: str, suffix: str) -> str:
        """
        Get the expected filename for given stem and suffix.
        
        Args:
            stem: Base filename
            suffix: Artifact suffix
            
        Returns:
            Expected filename
            
        Raises:
            ValueError: If suffix is invalid
        """
        if suffix not in self.VALID_SUFFIXES:
            valid_suffixes_str = ', '.join(sorted(self.VALID_SUFFIXES))
            raise ValueError(
                f"Invalid suffix '{suffix}'. Valid suffixes are: {valid_suffixes_str}"
            )
        
        return f"{stem}_{suffix}.json"


# Export the ArtifactManager class
__all__ = ['ArtifactManager']
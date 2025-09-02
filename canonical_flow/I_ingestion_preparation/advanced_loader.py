"""
Canonical Flow Component: Advanced Loader (02I)

This module handles advanced document loading and bundle creation
for the ingestion preparation stage of the canonical flow.

Component: 02I
Source: advanced_loader.py
Stage: ingestion_preparation
"""

import sys
import warnings
from pathlib import Path
from importlib import util as importlib_util

# Component metadata
source_module = r"advanced_loader.py"
stage = "ingestion_preparation"
component_id = "02I"

# Dynamically load and re-export the original module
try:
    # Add project root to path for imports
    project_root = Path(__file__).resolve().parents[2]  # Go up two levels from canonical_flow/I_ingestion_preparation/
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # Load original module
    original_file = project_root / "advanced_loader.py"
    if original_file.exists():
        spec = importlib_util.spec_from_file_location(
            f"original_{component_id.lower()}", 
            str(original_file)
        )
        
        if spec and spec.loader:
            original_module = importlib_util.module_from_spec(spec)
            spec.loader.exec_module(original_module)
            
            # Re-export all public symbols
            for attr_name in dir(original_module):
                if not attr_name.startswith("_"):
                    globals()[attr_name] = getattr(original_module, attr_name)
        else:
            raise ImportError(f"Could not load spec for {original_file}")
    else:
        raise FileNotFoundError(f"Original file not found: {original_file}")
        
except Exception as e:
    warnings.warn(f"Failed to load original module {source_module}: {e}")
    
    # Import ArtifactManager for process API
    from . import ArtifactManager
    
    # Create placeholder functions to prevent import errors
    def process(data=None, context=None):
        """
        Process API for advanced loader component (02I).
        
        Uses ArtifactManager to write standardized _bundle.json artifacts.
        """
        try:
            artifact_manager = ArtifactManager()
            
            # Handle failed module loading case
            error_data = {
                "error": f"Module {source_module} failed to load: {e}",
                "component": "02I",
                "status": "failed",
                "timestamp": str(__import__('datetime').datetime.now())
            }
            
            # Determine stem from context or use default
            stem = "unknown"
            if context and isinstance(context, dict):
                stem = context.get('document_stem', context.get('filename', 'unknown'))
            
            # Write error artifact using ArtifactManager
            output_path = artifact_manager.write_artifact(stem, "bundle", error_data)
            
            return {
                "success": False,
                "error": f"Module {source_module} failed to load: {e}",
                "output_path": str(output_path),
                "artifact_type": "bundle"
            }
        except Exception as inner_e:
            return {"error": f"Module {source_module} failed to load and artifact writing failed: {inner_e}"}

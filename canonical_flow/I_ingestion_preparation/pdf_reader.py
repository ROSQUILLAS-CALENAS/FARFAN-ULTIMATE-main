"""
Canonical Flow: PDF Reader (01I)

This module handles PDF document reading and text extraction
for the ingestion preparation stage of the canonical flow.

Component: 01I
Source: pdf_reader.py
Stage: ingestion_preparation
"""

# Mandatory Pipeline Contract Annotations
__phase__ = "I"
__code__ = "01I"
__stage_order__ = 1

import sys
import warnings
# # # from pathlib import Path  # Module not found  # Module not found  # Module not found
# # # from importlib import util as importlib_util  # Module not found  # Module not found  # Module not found

# Module metadata
source_module = r"pdf_reader.py"
stage = "ingestion_preparation"
component_id = "01I"

# Dynamically load and re-export the original module
try:
    # Add project root to path for imports
# # #     project_root = Path(__file__).resolve().parents[2]  # Go up two levels from canonical_flow/I_ingestion_preparation/  # Module not found  # Module not found  # Module not found
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # Load original module
    original_file = project_root / "pdf_reader.py"
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
# # #     from . import ArtifactManager  # Module not found  # Module not found  # Module not found
    
    # Create placeholder functions to prevent import errors
    def process(data=None, context=None):
        """
        Process API for PDF reader component (01I).
        
        Uses ArtifactManager to write standardized _text.json artifacts.
        """
        try:
            artifact_manager = ArtifactManager()
            
            # Handle failed module loading case
            error_data = {
                "error": f"Module {source_module} failed to load: {e}",
                "component": "01I",
                "status": "failed",
                "timestamp": str(__import__('datetime').datetime.now())
            }
            
# # #             # Determine stem from context or use default  # Module not found  # Module not found  # Module not found
            stem = "unknown"
            if context and isinstance(context, dict):
                stem = context.get('document_stem', context.get('filename', 'unknown'))
            
            # Write error artifact using ArtifactManager
            output_path = artifact_manager.write_artifact(stem, "text", error_data)
            
            return {
                "success": False,
                "error": f"Module {source_module} failed to load: {e}",
                "output_path": str(output_path),
                "artifact_type": "text"
            }
        except Exception as inner_e:
            return {"error": f"Module {source_module} failed to load and artifact writing failed: {inner_e}"}
    
    def stream_pdf_documents(*args, **kwargs):
        """Placeholder stream function for failed import."""
        return []
    
    class PDFPageIterator:
        """Placeholder class for failed import."""
        def __init__(self, *args, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def __iter__(self):
            return iter([])
        def process_full_document_with_ocr(self):
            return {"success": False, "error": "PDF reader not available"}

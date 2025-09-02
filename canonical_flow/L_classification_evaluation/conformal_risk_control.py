"""
Canonical Flow Alias: 23L

This is an auto-generated alias file that re-exports the original module.
DO NOT EDIT - changes will be overwritten by organize_canonical_structure.py

Source: egw_query_expansion/core/conformal_risk_control.py
Stage: classification_evaluation
Code: 23L
"""

import sys
from pathlib import Path
from importlib import util as importlib_util

# Alias metadata
alias_source = r"egw_query_expansion/core/conformal_risk_control.py"
alias_stage = "classification_evaluation"
alias_code = "23L"

# Dynamically load and re-export the original module
try:
    # Add project root to path for imports
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # Load original module
    original_file = project_root / "egw_query_expansion/core/conformal_risk_control.py"
    if original_file.exists():
        spec = importlib_util.spec_from_file_location(
            f"original_{alias_code.lower()}", 
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
    import warnings
    warnings.warn(f"Failed to load original module {alias_source}: {e}")
    
    # Create placeholder functions to prevent import errors
    def process(data=None, context=None):
        """Placeholder process function for failed import."""
        return {"error": f"Module {alias_source} failed to load: {e}"}

"""
Helper loader to integrate advanced orchestrator modules placed in
"F.A.R.F.A.N-1.O/Nueva carpeta con elementos" without relocating files.

It enables optional use of the advanced implementations via dynamic loading
so that existing imports and runtime remain backward compatible unless explicitly
enabled via environment variables.

Env flags:
- FARFAN_USE_ADVANCED_AIRFLOW = "1" to prefer HyperAirflowOrchestrator/AdvancedDAGGenerator
- FARFAN_USE_ENHANCED_CORE   = "1" to prefer HyperAdvancedOrchestrator
"""
# # # from __future__ import annotations  # Module not found  # Module not found  # Module not found

import os
import sys
# # # from importlib.machinery import SourceFileLoader  # Module not found  # Module not found  # Module not found
# # # from importlib.util import module_from_spec, spec_from_loader  # Module not found  # Module not found  # Module not found
# # # from pathlib import Path  # Module not found  # Module not found  # Module not found
# # # from types import ModuleType  # Module not found  # Module not found  # Module not found
# # # from typing import Any, Dict, Optional  # Module not found  # Module not found  # Module not found

# # # from total_ordering_base import TotalOrderingBase  # Module not found  # Module not found  # Module not found


# Mandatory Pipeline Contract Annotations
__phase__ = "I"
__code__ = "09I"
__stage_order__ = 1

ADV_DIR_NAME = "Nueva carpeta con elementos"


class AdvancedLoader(TotalOrderingBase):
    """Main advanced loader class with deterministic module loading"""
    
    def __init__(self):
        super().__init__(component_name="AdvancedLoader")
        
        # State tracking
        self._loaded_modules: Dict[str, ModuleType] = {}
        self._load_attempts = 0
        self._successful_loads = 0
        
        # Generate ID based on environment configuration
        env_config = {
            "FARFAN_USE_ADVANCED_AIRFLOW": os.getenv("FARFAN_USE_ADVANCED_AIRFLOW", "0"),
            "FARFAN_USE_ENHANCED_CORE": os.getenv("FARFAN_USE_ENHANCED_CORE", "0"),
            "FARFAN_FORCE_ADVANCED": os.getenv("FARFAN_FORCE_ADVANCED", "0")
        }
        self._env_config_id = self.generate_stable_id(env_config, prefix="env")
    
    def __lt__(self, other):
        """Comparison based on component ID for stable sorting"""
        if not isinstance(other, AdvancedLoader):
            return NotImplemented
        return self.component_id < other.component_id


def _env_enabled(name: str) -> bool:
    return os.getenv(name, "0") == "1"


def _advanced_dir() -> Optional[Path]:
    here = Path(__file__).resolve().parents[1]  # points to F.A.R.F.A.N-1.O
    candidate = here / ADV_DIR_NAME
    if candidate.exists() and candidate.is_dir():
        return candidate
    return None


def _load_module(module_filename: str, module_qualname: str, loader_instance: Optional[AdvancedLoader] = None) -> Optional[ModuleType]:
    """Load module with deterministic tracking if loader instance provided"""
    if loader_instance:
        loader_instance._load_attempts += 1
        
        # Check if already loaded
        if module_qualname in loader_instance._loaded_modules:
            return loader_instance._loaded_modules[module_qualname]
    
    adv_dir = _advanced_dir()
    if not adv_dir:
        return None
    src_path = adv_dir / module_filename
    if not src_path.exists():
        return None
        
    # Ensure a parent package exists for relative imports inside advanced modules
    pkg_name = module_qualname.split(".")[0]
    if pkg_name not in sys.modules:
        pkg = ModuleType(pkg_name)
        pkg.__path__ = [str(adv_dir)]  # type: ignore[attr-defined]
        sys.modules[pkg_name] = pkg
        
    # Map expected sibling modules used by advanced files to existing orchestration modules
    alias_map = {
        f"{pkg_name}.process_inventory": "orchestration.process_inventory",
        f"{pkg_name}.service_discovery": "orchestration.service_discovery",
        f"{pkg_name}.circuit_breaker": "orchestration.circuit_breaker",
        f"{pkg_name}.models": "orchestration.models",
        f"{pkg_name}.monitoring_stack": "orchestration.monitoring_stack",
        f"{pkg_name}.event_bus": "orchestration.event_bus",
        f"{pkg_name}.workflow_engine": "orchestration.workflow_engine",
        f"{pkg_name}.compensation_engine": "orchestration.compensation_engine",
        f"{pkg_name}.workflow_definitions": "orchestration.workflow_definitions",
        f"{pkg_name}.step_handlers": "orchestration.step_handlers",
        f"{pkg_name}.telemetry_collector": "orchestration.telemetry_collector",
# # #         f"{pkg_name}.optimization_engine": f"{pkg_name}.optimization_engine",  # load from advanced dir  # Module not found  # Module not found  # Module not found
    }
    
    # Create deterministic alias processing order
    sorted_aliases = sorted(alias_map.items())
    
    for alias, target in sorted_aliases:
        if alias not in sys.modules:
            try:
                if target.startswith(f"{pkg_name}."):
                    # Load advanced-local module lazily when requested
                    pass
                else:
                    # Import the orchestration module and alias it
                    target_mod = __import__(target, fromlist=["*"])
                    sys.modules[alias] = target_mod
            except Exception:
                # Skip if not available; the advanced module may not need it at runtime
                pass

    loader = SourceFileLoader(module_qualname, str(src_path))
    spec = spec_from_loader(module_qualname, loader)
    if spec is None:
        return None
    module = module_from_spec(spec)
    try:
        loader.exec_module(module)  # type: ignore[arg-type]
        sys.modules[module_qualname] = module
        
        # Track successful load
        if loader_instance:
            loader_instance._successful_loads += 1
            loader_instance._loaded_modules[module_qualname] = module
            
            # Update state hash
            state_data = {
                "loaded_modules": sorted(loader_instance._loaded_modules.keys()),
                "load_attempts": loader_instance._load_attempts,
                "successful_loads": loader_instance._successful_loads
            }
            loader_instance.update_state_hash(state_data)
        
        return module
    except Exception:
        # Do not crash the orchestrator if advanced module fails to load
        return None


def load_advanced_airflow_module(loader_instance: Optional[AdvancedLoader] = None) -> Optional[ModuleType]:
    """Attempt to load advanced_airflow_orchestrator.py module."""
    return _load_module("advanced_airflow_orchestrator.py", "farfan_advanced.airflow", loader_instance)


def load_enhanced_core_module(loader_instance: Optional[AdvancedLoader] = None) -> Optional[ModuleType]:
    """Attempt to load enhanced_core_orchestrator.py module."""
    return _load_module("enhanced_core_orchestrator.py", "farfan_advanced.core", loader_instance)


# Global loader instance for deterministic tracking
_global_loader = AdvancedLoader()


def get_hyper_airflow_orchestrator(*args, **kwargs) -> Optional[Any]:
    """Return HyperAirflowOrchestrator if available; can be forced by env flags.

    Flags:
      - FARFAN_USE_ADVANCED_AIRFLOW=1 enables optional usage (default: off)
      - FARFAN_FORCE_ADVANCED_AIRFLOW=1 (or FARFAN_FORCE_ADVANCED=1) requires advanced
        orchestrator to be present; raises RuntimeError if not available.
    """
    use = (
        _env_enabled("FARFAN_USE_ADVANCED_AIRFLOW")
        or _env_enabled("FARFAN_FORCE_ADVANCED_AIRFLOW")
        or _env_enabled("FARFAN_FORCE_ADVANCED")
    )
    force = _env_enabled("FARFAN_FORCE_ADVANCED_AIRFLOW") or _env_enabled(
        "FARFAN_FORCE_ADVANCED"
    )
    if not use:
        return None
    mod = load_advanced_airflow_module(_global_loader)
    if not mod:
        if force:
            raise RuntimeError(
                "Advanced Airflow module required but could not be loaded"
            )
        return None
    cls = getattr(mod, "HyperAirflowOrchestrator", None)
    if cls is None:
        if force:
            raise RuntimeError(
                "HyperAirflowOrchestrator class not found in advanced module"
            )
        return None
    try:
        return cls(*args, **kwargs)
    except Exception:
        if force:
            raise
        return None


def get_advanced_dag_generator(*args, **kwargs) -> Optional[Any]:
    """Return AdvancedDAGGenerator if available; can be forced by env flags.

    Flags:
      - FARFAN_USE_ADVANCED_AIRFLOW=1 enables optional usage (default: off)
      - FARFAN_FORCE_ADVANCED_AIRFLOW=1 (or FARFAN_FORCE_ADVANCED=1) requires advanced
        DAG generator to be present; raises RuntimeError if not available.
    """
    use = (
        _env_enabled("FARFAN_USE_ADVANCED_AIRFLOW")
        or _env_enabled("FARFAN_FORCE_ADVANCED_AIRFLOW")
        or _env_enabled("FARFAN_FORCE_ADVANCED")
    )
    force = _env_enabled("FARFAN_FORCE_ADVANCED_AIRFLOW") or _env_enabled(
        "FARFAN_FORCE_ADVANCED"
    )
    if not use:
        return None
    mod = load_advanced_airflow_module(_global_loader)
    if not mod:
        if force:
            raise RuntimeError(
                "Advanced Airflow module required but could not be loaded"
            )
        return None
    cls = getattr(mod, "AdvancedDAGGenerator", None)
    if cls is None:
        if force:
            raise RuntimeError(
                "AdvancedDAGGenerator class not found in advanced module"
            )
        return None
    try:
        return cls(*args, **kwargs)
    except Exception:
        if force:
            raise
        return None


def get_hyper_advanced_core(*args, **kwargs) -> Optional[Any]:
    """Return HyperAdvancedOrchestrator if available; can be forced by env flags.

    Flags:
      - FARFAN_USE_ENHANCED_CORE=1 enables optional usage (default: off)
      - FARFAN_FORCE_ENHANCED_CORE=1 (or FARFAN_FORCE_ADVANCED=1) requires enhanced
        core orchestrator; raises RuntimeError if not available.
    """
    use = (
        _env_enabled("FARFAN_USE_ENHANCED_CORE")
        or _env_enabled("FARFAN_FORCE_ENHANCED_CORE")
        or _env_enabled("FARFAN_FORCE_ADVANCED")
    )
    force = _env_enabled("FARFAN_FORCE_ENHANCED_CORE") or _env_enabled(
        "FARFAN_FORCE_ADVANCED"
    )
    if not use:
        return None
    mod = load_enhanced_core_module(_global_loader)
    if not mod:
        if force:
            raise RuntimeError("Enhanced core module required but could not be loaded")
        return None
    cls = getattr(mod, "HyperAdvancedOrchestrator", None)
    if cls is None:
        if force:
            raise RuntimeError(
                "HyperAdvancedOrchestrator class not found in enhanced module"
            )
        return None
    try:
        return cls(*args, **kwargs)
    except Exception:
        if force:
            raise
        return None


def process(data=None, context=None) -> Dict[str, Any]:
    """
    Process API for advanced loader component (02I).
    
# # #     Creates document bundles from text extraction results and writes   # Module not found  # Module not found  # Module not found
    standardized artifacts using ArtifactManager.
    
    Args:
        data: Input data (text extraction results or file paths)
        context: Processing context with metadata
        
    Returns:
        Dictionary with processing results and output paths
    """
    # Import ArtifactManager locally to avoid circular imports
    try:
# # #         from canonical_flow.ingestion import ArtifactManager  # Module not found  # Module not found  # Module not found
    except ImportError:
        return {"error": "ArtifactManager not available"}
    
    artifact_manager = ArtifactManager()
    
    # Process input data
    if not data:
        return {"error": "No input data provided"}
    
    results = []
    
    # Handle different input formats
    if isinstance(data, dict) and 'results' in data:
# # #         # Input from 01I component  # Module not found  # Module not found  # Module not found
        text_results = data['results']
    elif isinstance(data, list):
        text_results = data
    else:
        text_results = [data]
    
    for text_result in text_results:
        try:
            # Extract stem and text data
            if isinstance(text_result, dict):
                stem = text_result.get('stem', text_result.get('document_stem', 'unknown'))
                pages = text_result.get('pages', [])
                file_path = text_result.get('file_path', '')
            else:
                stem = 'unknown'
                pages = []
                file_path = ''
            
            # Create document bundle
            bundle_data = {
                "document_stem": stem,
                "source_file": file_path,
                "bundle_metadata": {
                    "component": "02I",
                    "processor": "AdvancedLoader",
                    "timestamp": str(__import__('datetime').datetime.now()),
                    "total_pages": len(pages)
                },
                "text_sections": [],
                "document_structure": {
                    "has_toc": False,
                    "sections_detected": 0,
                    "page_count": len(pages)
                }
            }
            
            # Process pages into sections
            full_text = ""
            for page in pages:
                if isinstance(page, dict) and 'text' in page:
                    page_text = page['text'].strip()
                    if page_text:
                        bundle_data["text_sections"].append({
                            "section_type": "page_content",
                            "page_number": page.get('page_num', 0),
                            "content": page_text,
                            "word_count": len(page_text.split())
                        })
                        full_text += page_text + "\n"
            
            # Update structure info
            bundle_data["document_structure"]["sections_detected"] = len(bundle_data["text_sections"])
            bundle_data["full_text"] = full_text.strip()
            bundle_data["document_stats"] = {
                "total_characters": len(full_text),
                "total_words": len(full_text.split()),
                "non_empty_pages": len([s for s in bundle_data["text_sections"] if s["word_count"] > 0])
            }
            
            # Write artifact using ArtifactManager
            output_path = artifact_manager.write_artifact(stem, "bundle", bundle_data)
            
            results.append({
                "stem": stem,
                "success": True,
                "output_path": str(output_path),
                "sections_created": len(bundle_data["text_sections"]),
                "artifact_type": "bundle"
            })
            
        except Exception as e:
            # Write error artifact
            error_stem = text_result.get('stem', 'unknown') if isinstance(text_result, dict) else 'unknown'
            error_data = {
                "document_stem": error_stem,
                "error": str(e),
                "processing_metadata": {
                    "component": "02I",
                    "status": "failed", 
                    "timestamp": str(__import__('datetime').datetime.now())
                }
            }
            
            try:
                output_path = artifact_manager.write_artifact(error_stem, "bundle", error_data)
                results.append({
                    "stem": error_stem,
                    "success": False,
                    "error": str(e),
                    "output_path": str(output_path),
                    "artifact_type": "bundle"
                })
            except Exception as artifact_error:
                results.append({
                    "stem": error_stem,
                    "success": False,
                    "error": f"Processing failed: {str(e)}, Artifact writing failed: {str(artifact_error)}"
                })
    
    return {
        "component": "02I",
        "results": results,
        "total_inputs": len(text_results),
        "successful_bundles": len([r for r in results if r.get('success', False)])
    }


def get_advanced_loader_state() -> Dict[str, Any]:
    """Get current state of the global advanced loader"""
    return _global_loader.serialize_output(_global_loader.get_deterministic_metadata())

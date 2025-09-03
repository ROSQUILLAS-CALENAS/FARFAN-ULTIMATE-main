#!/usr/bin/env python3
"""
Pipeline Dependency Validation Script

Validates all modules defined in the comprehensive_pipeline_orchestrator.py dependency graph:
- Checks if each module file exists at the expected path
- Attempts to import each module successfully 
- Verifies each module contains at least one required entrypoint method
- Handles subdirectory module paths correctly
- Reports detailed validation failures

Run: python validate_pipeline_dependencies.py
"""

import importlib
import importlib.util
import inspect
import sys
import traceback
# # # from pathlib import Path  # Module not found  # Module not found  # Module not found
# # # from typing import Dict, List, Set, Tuple  # Module not found  # Module not found  # Module not found

# # # from comprehensive_pipeline_orchestrator import ComprehensivePipelineOrchestrator  # Module not found  # Module not found  # Module not found


class PipelineValidationResult:
    """Container for validation results"""
    
    def __init__(self):
        self.total_modules = 0
        self.validated_modules = 0
        self.missing_files: List[str] = []
        self.import_errors: List[Tuple[str, str]] = []  # (module, error)
        self.missing_entrypoints: List[Tuple[str, List[str]]] = []  # (module, available_methods)
        self.successful_modules: List[str] = []

    @property
    def success_rate(self) -> float:
        return (self.validated_modules / self.total_modules * 100) if self.total_modules > 0 else 0.0

    @property
    def has_failures(self) -> bool:
        return bool(self.missing_files or self.import_errors or self.missing_entrypoints)


class PipelineDependencyValidator:
    """Validates pipeline module dependencies and entrypoints"""
    
    REQUIRED_ENTRYPOINTS = {"process", "run", "execute", "main", "handle"}
    
    def __init__(self, root_path: Path = None):
        self.root_path = root_path or Path(__file__).resolve().parent
        self.orchestrator = ComprehensivePipelineOrchestrator()
        
    def validate_all_modules(self) -> PipelineValidationResult:
        """Validate all modules in the dependency graph"""
        result = PipelineValidationResult()
        
# # #         # Get all modules from the dependency graph  # Module not found  # Module not found  # Module not found
        modules = list(self.orchestrator.process_graph.keys())
        result.total_modules = len(modules)
        
        print(f"🔍 Validating {result.total_modules} pipeline modules...")
        print("=" * 60)
        
        for module_name in sorted(modules):
            print(f"\n📦 Validating: {module_name}")
            
            # Check if file exists
            if not self._check_file_exists(module_name):
                result.missing_files.append(module_name)
                print(f"  ❌ File not found at expected path")
                continue
                
            # Try to import module
            try:
                imported_module = self._import_module(module_name)
                if imported_module is None:
                    result.import_errors.append((module_name, "Failed to import (unknown error)"))
                    print(f"  ❌ Import failed")
                    continue
            except Exception as e:
                error_msg = f"{type(e).__name__}: {str(e)}"
                result.import_errors.append((module_name, error_msg))
                print(f"  ❌ Import error: {error_msg}")
                continue
                
            # Check for required entrypoint methods
            available_methods = self._get_available_methods(imported_module)
            entrypoint_methods = self._find_entrypoint_methods(available_methods)
            
            if not entrypoint_methods:
                result.missing_entrypoints.append((module_name, available_methods))
                print(f"  ❌ No entrypoint methods found")
                print(f"     Available methods: {', '.join(available_methods) if available_methods else 'None'}")
            else:
                result.validated_modules += 1
                result.successful_modules.append(module_name)
                print(f"  ✅ Valid - entrypoints: {', '.join(entrypoint_methods)}")
        
        return result
    
    def _check_file_exists(self, module_name: str) -> bool:
        """Check if module file exists at expected path"""
        module_path = self._resolve_module_path(module_name)
        return module_path.exists()
    
    def _resolve_module_path(self, module_name: str) -> Path:
        """Resolve the full path to a module file"""
        if "/" in module_name:
            # Handle subdirectory paths like "retrieval_engine/lexical_index.py"
            return self.root_path / module_name
        else:
            # Handle root-level modules
            return self.root_path / module_name
    
    def _import_module(self, module_name: str):
        """Import a module by name, handling subdirectories correctly"""
        module_path = self._resolve_module_path(module_name)
        
        # Generate module spec name for import
        if "/" in module_name:
            # Convert path separators to dots and remove .py extension
            spec_name = module_name.replace("/", ".").replace(".py", "")
        else:
            # Remove .py extension for root modules
            spec_name = module_name.replace(".py", "")
        
        # Handle special characters in filenames
        spec_name = spec_name.replace(" ", "_").replace("-", "_")
        
        try:
# # #             # Use importlib to load the module directly from file path  # Module not found  # Module not found  # Module not found
            spec = importlib.util.spec_from_file_location(spec_name, module_path)
            if spec is None or spec.loader is None:
                return None
                
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
            
        except Exception:
            # Re-raise for caller to handle
            raise
    
    def _get_available_methods(self, module) -> List[str]:
        """Get all available public methods/functions in a module"""
        methods = []
        
        for name in dir(module):
            if name.startswith('_'):
                continue
                
            attr = getattr(module, name)
            if inspect.isfunction(attr) or inspect.ismethod(attr):
                methods.append(name)
            elif inspect.isclass(attr):
                # Check for entrypoint methods in classes
                for class_method_name in dir(attr):
                    if class_method_name.startswith('_'):
                        continue
                    class_attr = getattr(attr, class_method_name)
                    if inspect.isfunction(class_attr) or inspect.ismethod(class_attr):
                        methods.append(f"{name}.{class_method_name}")
        
        return methods
    
    def _find_entrypoint_methods(self, available_methods: List[str]) -> List[str]:
# # #         """Find entrypoint methods from available methods"""  # Module not found  # Module not found  # Module not found
        entrypoints = []
        
        for method in available_methods:
            # Check direct method names
            method_name = method.split('.')[-1]  # Get last part for class methods
            if method_name.lower() in self.REQUIRED_ENTRYPOINTS:
                entrypoints.append(method)
        
        return entrypoints
    
    def print_detailed_report(self, result: PipelineValidationResult):
        """Print detailed validation report"""
        print("\n" + "=" * 60)
        print("📊 PIPELINE VALIDATION REPORT")
        print("=" * 60)
        
        print(f"Total modules: {result.total_modules}")
        print(f"Successfully validated: {result.validated_modules}")
        print(f"Success rate: {result.success_rate:.1f}%")
        print()
        
        if result.missing_files:
            print(f"❌ MISSING FILES ({len(result.missing_files)}):")
            for module_name in result.missing_files:
                expected_path = self._resolve_module_path(module_name)
                print(f"  • {module_name}")
                print(f"    Expected at: {expected_path}")
            print()
        
        if result.import_errors:
            print(f"❌ IMPORT ERRORS ({len(result.import_errors)}):")
            for module_name, error in result.import_errors:
                print(f"  • {module_name}")
                print(f"    Error: {error}")
            print()
        
        if result.missing_entrypoints:
            print(f"❌ MISSING ENTRYPOINTS ({len(result.missing_entrypoints)}):")
            print(f"    Required: {', '.join(sorted(self.REQUIRED_ENTRYPOINTS))}")
            for module_name, available in result.missing_entrypoints:
                print(f"  • {module_name}")
                if available:
                    print(f"    Available methods: {', '.join(available)}")
                else:
                    print(f"    No public methods found")
            print()
        
        if result.successful_modules:
            print(f"✅ SUCCESSFULLY VALIDATED ({len(result.successful_modules)}):")
            for module_name in result.successful_modules:
                print(f"  • {module_name}")
            print()
        
        if result.has_failures:
            print("⚠️  VALIDATION FAILED - Pipeline dependencies not properly configured")
            return False
        else:
            print("🎉 ALL MODULES VALIDATED SUCCESSFULLY")
            return True


def main():
    """Main validation function"""
    print("🚀 Pipeline Dependency Validation")
    print("Checking comprehensive_pipeline_orchestrator.py dependency graph...")
    
    validator = PipelineDependencyValidator()
    
    try:
        result = validator.validate_all_modules()
        success = validator.print_detailed_report(result)
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)
        
    except Exception as e:
        print(f"\n💥 VALIDATION SCRIPT ERROR: {e}")
        print("Traceback:")
        traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    main()
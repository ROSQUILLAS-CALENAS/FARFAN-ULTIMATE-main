#!/usr/bin/env python3
"""
Runtime import guard for detecting and preventing circular imports at runtime.
This module provides utilities to detect import cycles during module loading.
"""

import sys
import threading
import weakref
from typing import Dict, List, Optional, Set
from types import ModuleType
import warnings


class ImportCycleDetector:
    """Thread-safe detector for import cycles at runtime."""
    
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._import_stack: Dict[int, List[str]] = {}
        self._module_refs: Dict[str, weakref.ref] = {}
        self._detected_cycles: Set[tuple] = set()
        
    def _get_thread_id(self) -> int:
        """Get current thread ID."""
        return threading.get_ident()
        
    def enter_import(self, module_name: str) -> None:
        """Called when starting to import a module."""
        with self._lock:
            thread_id = self._get_thread_id()
            if thread_id not in self._import_stack:
                self._import_stack[thread_id] = []
                
            current_stack = self._import_stack[thread_id]
            
            # Check for cycle
            if module_name in current_stack:
                cycle_start = current_stack.index(module_name)
                cycle = current_stack[cycle_start:] + [module_name]
                cycle_tuple = tuple(cycle)
                
                if cycle_tuple not in self._detected_cycles:
                    self._detected_cycles.add(cycle_tuple)
                    self._handle_cycle_detected(cycle)
                    
            current_stack.append(module_name)
            
    def exit_import(self, module_name: str, module: Optional[ModuleType] = None) -> None:
        """Called when finishing importing a module."""
        with self._lock:
            thread_id = self._get_thread_id()
            if thread_id in self._import_stack:
                current_stack = self._import_stack[thread_id]
                if current_stack and current_stack[-1] == module_name:
                    current_stack.pop()
                    
                # Store weak reference to module
                if module is not None:
                    self._module_refs[module_name] = weakref.ref(module)
                    
    def _handle_cycle_detected(self, cycle: List[str]) -> None:
        """Handle detection of an import cycle."""
        cycle_str = " → ".join(cycle)
        
        # Issue warning
        warnings.warn(
            f"Circular import detected: {cycle_str}",
            ImportWarning,
            stacklevel=3
        )
        
        # Log to stderr for visibility
        print(f"❌ RUNTIME CIRCULAR IMPORT: {cycle_str}", file=sys.stderr)
        
    def get_import_stack(self, thread_id: Optional[int] = None) -> List[str]:
        """Get current import stack for thread."""
        with self._lock:
            if thread_id is None:
                thread_id = self._get_thread_id()
            return self._import_stack.get(thread_id, []).copy()
            
    def get_detected_cycles(self) -> List[List[str]]:
        """Get all detected cycles."""
        with self._lock:
            return [list(cycle) for cycle in self._detected_cycles]
            
    def clear_detected_cycles(self) -> None:
        """Clear detected cycles (useful for testing)."""
        with self._lock:
            self._detected_cycles.clear()


# Global detector instance
_global_detector = ImportCycleDetector()


class ImportGuardMetaFinder:
    """Meta path finder that detects import cycles."""
    
    def find_spec(self, fullname: str, path, target=None):
        """Hook into import machinery to detect cycles."""
        _global_detector.enter_import(fullname)
        return None  # Let default finders handle actual import
        

class ImportGuard:
    """Context manager for guarded imports."""
    
    def __init__(self, module_name: str):
        self.module_name = module_name
        
    def __enter__(self):
        _global_detector.enter_import(self.module_name)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        module = sys.modules.get(self.module_name)
        _global_detector.exit_import(self.module_name, module)


def install_import_guard() -> None:
    """Install the import guard in the meta path."""
    finder = ImportGuardMetaFinder()
    if finder not in sys.meta_path:
        sys.meta_path.insert(0, finder)


def uninstall_import_guard() -> None:
    """Remove the import guard from the meta path."""
    sys.meta_path[:] = [
        finder for finder in sys.meta_path
        if not isinstance(finder, ImportGuardMetaFinder)
    ]


def get_import_cycles() -> List[List[str]]:
    """Get all detected import cycles."""
    return _global_detector.get_detected_cycles()


def clear_import_cycles() -> None:
    """Clear detected import cycles."""
    _global_detector.clear_detected_cycles()


def check_for_cycles() -> bool:
    """
    Check if any import cycles have been detected.
    
    Returns:
        True if cycles detected, False otherwise
    """
    cycles = get_import_cycles()
    if cycles:
        print("Runtime circular imports detected:")
        for i, cycle in enumerate(cycles, 1):
            print(f"  {i}. {' → '.join(cycle)}")
        return True
    return False


def main() -> int:
    """Main entry point for runtime validation."""
    print("Installing runtime import guard...")
    install_import_guard()
    
    try:
        # Import main modules to test for cycles
        import egw_query_expansion
        print("✅ Main package imported successfully")
        
        # Check for any detected cycles
        if check_for_cycles():
            print("❌ Runtime circular imports detected!")
            return 1
        else:
            print("✅ No runtime circular imports detected")
            return 0
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return 1
    finally:
        uninstall_import_guard()


if __name__ == "__main__":
    sys.exit(main())
"""
Import Blocker

Prevents analysis modules from importing retrieval components directly,
forcing all communication through the adapter translation layer.
"""

import sys
import importlib.util
import logging
from typing import Set, Dict, Any, Optional
from .lineage_tracker import LineageTracker


class ImportBlocker:
    """Blocks direct imports between restricted component pairs"""
    
    # Define restricted import patterns
    RESTRICTED_PATTERNS = {
        # Analysis modules cannot import retrieval modules
        'analysis': [
            'retrieval_engine',
            'hybrid_retrieval', 
            'canonical_flow.R_search_retrieval',
            'retrieval_engine.py',
            'hybrid_retrieval.py'
        ],
        # Retrieval modules cannot import analysis modules
        'retrieval': [
            'analysis_nlp_orchestrator',
            'canonical_flow.A_analysis_nlp',
            'analysis_nlp_orchestrator.py'
        ]
    }
    
    def __init__(self, lineage_tracker: Optional[LineageTracker] = None):
        self.logger = logging.getLogger(__name__)
        self.lineage_tracker = lineage_tracker
        self.blocked_imports: Dict[str, Set[str]] = {}
        self.violation_count = 0
        
        # Install import hook
        self._install_import_hook()
        
        # Setup violation logging
        self._setup_violation_logging()
    
    def _setup_violation_logging(self):
        """Configure violation logging"""
        violation_handler = logging.FileHandler('logs/import_violations.log')
        violation_handler.setFormatter(
            logging.Formatter(
                '%(asctime)s - %(levelname)s - IMPORT_VIOLATION - %(message)s'
            )
        )
        
        violation_logger = logging.getLogger('import_violations')
        violation_logger.addHandler(violation_handler)
        violation_logger.setLevel(logging.ERROR)
        
        self.violation_logger = violation_logger
    
    def _install_import_hook(self):
        """Install import hook to intercept imports"""
        
        # Store original import function
        self._original_import = __builtins__['__import__']
        
        # Replace with our intercepting import
        __builtins__['__import__'] = self._intercepting_import
    
    def _intercepting_import(self, name, globals_=None, locals_=None, fromlist=(), level=0):
        """Intercept and validate imports"""
        
        # Get calling module info
        calling_module = self._get_calling_module(globals_)
        
        # Check if this import should be blocked
        if self._should_block_import(calling_module, name):
            self._handle_blocked_import(calling_module, name)
            raise ImportError(
                f"Import of '{name}' blocked: analysis modules cannot import "
                f"retrieval components directly. Use adapters instead."
            )
        
        # Allow the import
        return self._original_import(name, globals_, locals_, fromlist, level)
    
    def _get_calling_module(self, globals_dict: Optional[Dict[str, Any]]) -> str:
        """Get the name of the calling module"""
        
        if not globals_dict:
            return 'unknown'
        
        module_name = globals_dict.get('__name__', 'unknown')
        module_file = globals_dict.get('__file__', '')
        
        # Try to get more specific module identification
        if module_file:
            if 'analysis' in module_file.lower():
                return f"{module_name} (analysis)"
            elif 'retrieval' in module_file.lower():
                return f"{module_name} (retrieval)"
        
        return module_name
    
    def _should_block_import(self, calling_module: str, imported_module: str) -> bool:
        """Check if an import should be blocked"""
        
        # Check analysis -> retrieval blocks
        if 'analysis' in calling_module.lower():
            for pattern in self.RESTRICTED_PATTERNS['analysis']:
                if pattern in imported_module:
                    return True
        
        # Check retrieval -> analysis blocks  
        if 'retrieval' in calling_module.lower():
            for pattern in self.RESTRICTED_PATTERNS['retrieval']:
                if pattern in imported_module:
                    return True
        
        return False
    
    def _handle_blocked_import(self, calling_module: str, imported_module: str):
        """Handle a blocked import violation"""
        
        self.violation_count += 1
        
        # Track the violation
        if calling_module not in self.blocked_imports:
            self.blocked_imports[calling_module] = set()
        self.blocked_imports[calling_module].add(imported_module)
        
        # Log the violation
        violation_msg = {
            'calling_module': calling_module,
            'attempted_import': imported_module,
            'violation_count': self.violation_count,
            'timestamp': self._get_timestamp()
        }
        
        self.violation_logger.error(f"BLOCKED_IMPORT: {violation_msg}")
        
        # Report to lineage tracker if available
        if self.lineage_tracker:
            from .data_transfer_objects import LineageEvent
            
            event = LineageEvent(
                component_id=calling_module,
                operation_type="blocked_import",
                input_schema="",
                output_schema="",
                dependencies=[imported_module],
                violation_type="direct_import_violation"
            )
            
            self.lineage_tracker.track_dependency_violation(event)
    
    def _get_timestamp(self) -> str:
        """Get current timestamp as string"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def add_restriction(self, calling_pattern: str, blocked_patterns: list[str]):
        """Add custom import restriction"""
        
        if calling_pattern not in self.RESTRICTED_PATTERNS:
            self.RESTRICTED_PATTERNS[calling_pattern] = []
        
        self.RESTRICTED_PATTERNS[calling_pattern].extend(blocked_patterns)
    
    def remove_restriction(self, calling_pattern: str, blocked_pattern: str):
        """Remove a specific import restriction"""
        
        if calling_pattern in self.RESTRICTED_PATTERNS:
            patterns = self.RESTRICTED_PATTERNS[calling_pattern]
            if blocked_pattern in patterns:
                patterns.remove(blocked_pattern)
    
    def get_violation_summary(self) -> Dict[str, Any]:
        """Get summary of import violations"""
        
        violation_by_module = {
            module: len(imports) 
            for module, imports in self.blocked_imports.items()
        }
        
        most_violating = sorted(
            violation_by_module.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return {
            'total_violations': self.violation_count,
            'violating_modules': len(self.blocked_imports),
            'violations_by_module': violation_by_module,
            'most_violating_modules': most_violating[:10],
            'restricted_patterns': dict(self.RESTRICTED_PATTERNS)
        }
    
    def is_import_allowed(self, calling_module: str, imported_module: str) -> bool:
        """Check if an import would be allowed without actually importing"""
        return not self._should_block_import(calling_module, imported_module)
    
    def disable(self):
        """Disable the import blocker"""
        if hasattr(self, '_original_import'):
            __builtins__['__import__'] = self._original_import
    
    def enable(self):
        """Re-enable the import blocker"""
        __builtins__['__import__'] = self._intercepting_import
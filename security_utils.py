#!/usr/bin/env python3
"""
Security utilities with safe wrappers for eval/exec and secure serialization.

This module provides secure alternatives to dangerous Python functions
and implements safe serialization using jsonpickle and msgpack.
"""

import ast
import logging
import re
import traceback
from typing import Any, Dict, List, Optional, Set, Union
from pathlib import Path
import functools
import time

try:
    import jsonpickle
    import msgpack
except ImportError as e:
    logging.error(f"Required security dependencies not installed: {e}")
    raise ImportError("Please install jsonpickle and msgpack: pip install jsonpickle msgpack")

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Allowlisted names for safe evaluation
SAFE_NAMES = {
    'abs', 'all', 'any', 'bool', 'chr', 'dict', 'divmod', 'enumerate',
    'filter', 'float', 'frozenset', 'hasattr', 'hash', 'hex', 'int',
    'isinstance', 'len', 'list', 'map', 'max', 'min', 'oct', 'ord',
    'pow', 'range', 'repr', 'reversed', 'round', 'set', 'slice', 'sorted',
    'str', 'sum', 'tuple', 'zip', 'True', 'False', 'None'
}

# Allowlisted modules for imports
SAFE_MODULES = {
    'math', 'random', 'datetime', 'collections', 'itertools', 'functools',
    'operator', 'json', 'base64', 'hashlib', 'uuid', 'typing'
}

# Dangerous patterns that should never be evaluated
DANGEROUS_PATTERNS = [
    r'__\w+__',  # Dunder methods
    r'import\s+',  # Import statements
    r'exec\s*\(',  # Exec calls
    r'eval\s*\(',  # Eval calls
    r'compile\s*\(',  # Compile calls
    r'open\s*\(',  # File operations
    r'input\s*\(',  # Input operations
    r'getattr',  # Attribute access
    r'setattr',  # Attribute setting
    r'delattr',  # Attribute deletion
    r'globals\s*\(',  # Global scope access
    r'locals\s*\(',  # Local scope access
    r'vars\s*\(',  # Variable introspection
    r'dir\s*\(',  # Directory listing
    r'help\s*\(',  # Help function
    r'quit\s*\(',  # Quit function
    r'exit\s*\(',  # Exit function
]


class SecurityError(Exception):
    """Raised when a security violation is detected."""
    pass


class SafeEvaluator:
    """Safe evaluation context with allowlisted operations."""
    
    def __init__(self, allowed_names: Optional[Set[str]] = None, 
                 allowed_modules: Optional[Set[str]] = None,
                 max_execution_time: float = 5.0):
        self.allowed_names = allowed_names or SAFE_NAMES.copy()
        self.allowed_modules = allowed_modules or SAFE_MODULES.copy()
        self.max_execution_time = max_execution_time
        
    def _validate_expression(self, expression: str) -> None:
        """Validate expression for security threats."""
        if not isinstance(expression, str):
            raise SecurityError("Expression must be a string")
            
        if len(expression) > 10000:  # Prevent DoS with huge expressions
            raise SecurityError("Expression too long")
            
        # Check for dangerous patterns
        for pattern in DANGEROUS_PATTERNS:
            if re.search(pattern, expression, re.IGNORECASE):
                raise SecurityError(f"Dangerous pattern detected: {pattern}")
        
        # Parse AST to validate syntax and structure
        try:
            tree = ast.parse(expression, mode='eval')
        except SyntaxError as e:
            raise SecurityError(f"Invalid syntax: {e}")
        
        # Check for only allowed node types
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                raise SecurityError("Import statements not allowed")
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id not in self.allowed_names:
                    raise SecurityError(f"Function '{node.func.id}' not allowed")
            elif isinstance(node, ast.Attribute):
                if isinstance(node.value, ast.Name) and node.value.id not in self.allowed_names:
                    raise SecurityError(f"Attribute access on '{node.value.id}' not allowed")
                    
    def safe_eval(self, expression: str, global_vars: Optional[Dict[str, Any]] = None,
                  local_vars: Optional[Dict[str, Any]] = None) -> Any:
        """Safely evaluate a Python expression."""
        start_time = time.time()
        
        try:
            # Validate the expression
            self._validate_expression(expression)
            
            # Create safe namespace
            safe_globals = {name: __builtins__[name] for name in self.allowed_names 
                           if name in __builtins__}
            
            # Add provided globals if safe
            if global_vars:
                for name, value in global_vars.items():
                    if name in self.allowed_names:
                        safe_globals[name] = value
                    else:
                        logger.warning(f"Ignoring unsafe global variable: {name}")
            
            # Add math module if allowed
            if 'math' in self.allowed_modules:
                import math
                safe_globals['math'] = math
            
            # Use restricted locals
            safe_locals = local_vars or {}
            
            # Check execution time limit
            def timeout_handler():
                if time.time() - start_time > self.max_execution_time:
                    raise SecurityError("Execution timeout exceeded")
            
            # Evaluate with timeout protection
            result = eval(expression, {"__builtins__": {}}, 
                         {**safe_globals, **safe_locals})
            
            timeout_handler()
            
            logger.info(f"Safe evaluation successful: {expression[:100]}...")
            return result
            
        except SecurityError:
            logger.error(f"Security violation in expression: {expression[:100]}...")
            raise
        except Exception as e:
            logger.error(f"Evaluation error: {e}")
            raise SecurityError(f"Evaluation failed: {e}")


def safe_eval(expression: str, global_vars: Optional[Dict[str, Any]] = None,
              local_vars: Optional[Dict[str, Any]] = None) -> Any:
    """Safe wrapper for eval() with security validation."""
    evaluator = SafeEvaluator()
    return evaluator.safe_eval(expression, global_vars, local_vars)


def safe_exec(code: str, global_vars: Optional[Dict[str, Any]] = None,
              local_vars: Optional[Dict[str, Any]] = None) -> None:
    """Safe wrapper for exec() - RESTRICTED TO VERY LIMITED CASES."""
    if not isinstance(code, str):
        raise SecurityError("Code must be a string")
        
    if len(code) > 5000:  # Prevent DoS
        raise SecurityError("Code too long")
    
    # Check for dangerous patterns
    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, code, re.IGNORECASE):
            raise SecurityError(f"Dangerous pattern detected in exec: {pattern}")
    
    # Only allow very simple assignments and expressions
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Module, ast.Assign, ast.Name, 
                                   ast.Constant, ast.Expr, ast.BinOp, 
                                   ast.UnaryOp, ast.Compare)):
                raise SecurityError(f"Unsupported AST node type: {type(node).__name__}")
    except SyntaxError as e:
        raise SecurityError(f"Invalid syntax in exec: {e}")
    
    logger.warning(f"Executing code (use with extreme caution): {code[:100]}...")
    
    # Create minimal safe environment
    safe_globals = {"__builtins__": {}}
    if global_vars:
        safe_globals.update({k: v for k, v in global_vars.items() 
                           if k in SAFE_NAMES})
    
    safe_locals = local_vars or {}
    
    try:
        exec(code, safe_globals, safe_locals)
    except Exception as e:
        logger.error(f"Exec error: {e}")
        raise SecurityError(f"Execution failed: {e}")


class SecureSerialization:
    """Secure serialization using jsonpickle and msgpack."""
    
    @staticmethod
    def serialize_jsonpickle(obj: Any, keys: bool = True) -> str:
        """Serialize object using jsonpickle (safer than pickle)."""
        try:
            # Configure jsonpickle for security
            jsonpickle.set_encoder_options('json', ensure_ascii=False, sort_keys=True)
            return jsonpickle.encode(obj, keys=keys, warn=True)
        except Exception as e:
            logger.error(f"Jsonpickle serialization error: {e}")
            raise SecurityError(f"Serialization failed: {e}")
    
    @staticmethod
    def deserialize_jsonpickle(data: str, keys: bool = True) -> Any:
        """Deserialize object using jsonpickle."""
        try:
            return jsonpickle.decode(data, keys=keys)
        except Exception as e:
            logger.error(f"Jsonpickle deserialization error: {e}")
            raise SecurityError(f"Deserialization failed: {e}")
    
    @staticmethod
    def serialize_msgpack(obj: Any) -> bytes:
        """Serialize object using msgpack (fast and secure)."""
        try:
            return msgpack.packb(obj, use_bin_type=True)
        except Exception as e:
            logger.error(f"Msgpack serialization error: {e}")
            raise SecurityError(f"Serialization failed: {e}")
    
    @staticmethod  
    def deserialize_msgpack(data: bytes) -> Any:
        """Deserialize object using msgpack."""
        try:
            return msgpack.unpackb(data, raw=False, strict_map_key=False)
        except Exception as e:
            logger.error(f"Msgpack deserialization error: {e}")
            raise SecurityError(f"Deserialization failed: {e}")


def secure_pickle_replacement(obj: Any, use_msgpack: bool = False) -> Union[str, bytes]:
    """Drop-in replacement for pickle.dumps() using secure alternatives."""
    serializer = SecureSerialization()
    
    if use_msgpack:
        return serializer.serialize_msgpack(obj)
    else:
        return serializer.serialize_jsonpickle(obj)


def secure_unpickle_replacement(data: Union[str, bytes]) -> Any:
    """Drop-in replacement for pickle.loads() using secure alternatives."""
    serializer = SecureSerialization()
    
    if isinstance(data, bytes):
        return serializer.deserialize_msgpack(data)
    else:
        return serializer.deserialize_jsonpickle(data)


def security_audit_decorator(func):
    """Decorator to log function calls for security auditing."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"Security audit: {func.__name__} executed successfully in {execution_time:.3f}s")
            return result
        except Exception as e:
            logger.error(f"Security audit: {func.__name__} failed with {type(e).__name__}: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    return wrapper


if __name__ == "__main__":
    # Demo of safe evaluation
    print("=== Security Utils Demo ===")
    
    # Safe evaluation examples
    evaluator = SafeEvaluator()
    
    try:
        result = evaluator.safe_eval("2 + 3 * 4")
        print(f"Safe eval: 2 + 3 * 4 = {result}")
        
        result = evaluator.safe_eval("max([1, 2, 3, 4])")
        print(f"Safe eval: max([1, 2, 3, 4]) = {result}")
        
        # This should fail
        result = evaluator.safe_eval("__import__('os').system('ls')")
        print("This should not print!")
        
    except SecurityError as e:
        print(f"Security error (expected): {e}")
    
    # Secure serialization example
    serializer = SecureSerialization()
    
    data = {"test": [1, 2, 3], "nested": {"key": "value"}}
    
    # JSONPickle
    serialized = serializer.serialize_jsonpickle(data)
    deserialized = serializer.deserialize_jsonpickle(serialized)
    print(f"JSONPickle roundtrip: {data} -> {deserialized}")
    
    # MessagePack
    serialized_mp = serializer.serialize_msgpack(data)
    deserialized_mp = serializer.deserialize_msgpack(serialized_mp)
    print(f"MessagePack roundtrip: {data} -> {deserialized_mp}")
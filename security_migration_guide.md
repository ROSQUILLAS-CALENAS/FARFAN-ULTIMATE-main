# Security Migration Guide

## Overview

This guide explains the security improvements implemented in the codebase to replace unsafe serialization and evaluation practices with secure alternatives.

## Changes Made

### 1. Secure Evaluation Wrappers

Created `security_utils.py` with safe wrappers for `eval()` and `exec()`:

- **SafeEvaluator class**: Validates expressions using AST parsing and allowlisted operations
- **safe_eval() function**: Drop-in replacement for eval() with security validation
- **safe_exec() function**: Restricted exec() for very limited use cases

#### Security Features:
- Expression length limits (10,000 chars max)
- Dangerous pattern detection (imports, file operations, etc.)
- AST validation to prevent unsafe operations
- Execution timeout protection (5 seconds default)
- Allowlisted function and module names
- Comprehensive logging and error handling

### 2. Secure Serialization Migration

Replaced all `pickle.loads()` and `pickle.dumps()` calls with secure alternatives:

#### Files Updated:
- `adaptive_scoring_engine.py` - Model serialization
- `serializable_wrappers.py` - Wrapper serialization  
- `test_serializable_wrappers.py` - Test serialization
- `retrieval_engine/vector_index.py` - Metadata serialization
- `retrieval_engine/lexical_index.py` - Index serialization

#### Secure Alternatives:
- **JSONPickle**: Safer JSON-based serialization for complex objects
- **MessagePack**: Fast binary serialization for simple data types
- **Backward compatibility**: Legacy pickle files still supported with warnings

### 3. Safe Code Execution

Updated `validate_decalogo.py` to use secure execution instead of raw `exec()`:

- Uses `safe_exec()` for controlled code execution
- Fallback to proper imports when safe execution fails
- Security error handling and logging

### 4. Enhanced CI Security Scanning

Updated `.github/workflows/static-analysis.yml` with comprehensive Bandit configuration:

#### New Features:
- Full test suite coverage (B101-B703)
- High-severity issues fail the build
- Medium-severity issues generate warnings
- Detailed security reports (JSON + text)
- Configurable exclusions and rules

#### Security Rules Enabled:
- **B101-B108**: Assert and test vulnerabilities
- **B201**: Flask debug mode detection
- **B301-B325**: Import and module vulnerabilities
- **B401-B413**: Cryptographic vulnerabilities
- **B501-B507**: Certificate and SSL issues
- **B601-B611**: Shell injection vulnerabilities
- **B701-B703**: Code injection vulnerabilities

## Usage Examples

### Safe Evaluation
```python
from security_utils import safe_eval, SafeEvaluator

# Basic safe evaluation
result = safe_eval("2 + 3 * 4")  # Returns: 14

# Custom evaluator with extended allowlist
evaluator = SafeEvaluator(allowed_names={'custom_func'})
result = evaluator.safe_eval("custom_func(5)")
```

### Secure Serialization
```python
from security_utils import secure_pickle_replacement, secure_unpickle_replacement

# Serialize with MessagePack
data = {"key": [1, 2, 3]}
serialized = secure_pickle_replacement(data, use_msgpack=True)
deserialized = secure_unpickle_replacement(serialized)

# Serialize with JSONPickle  
serialized = secure_pickle_replacement(data, use_msgpack=False)
deserialized = secure_unpickle_replacement(serialized)
```

## Dependencies Added

Added security-focused dependencies to `requirements.txt`:
```
jsonpickle>=3.0.0
msgpack>=1.0.0
```

## Migration Checklist

- ✅ Created secure evaluation wrappers
- ✅ Replaced pickle with secure serialization
- ✅ Updated CI pipeline for security scanning
- ✅ Added backward compatibility for legacy files
- ✅ Implemented comprehensive logging
- ✅ Added security dependencies

## Security Best Practices Implemented

1. **Defense in Depth**: Multiple validation layers (syntax, AST, patterns)
2. **Fail-Safe Defaults**: Restrictive allowlists by default
3. **Comprehensive Logging**: All security events logged for audit
4. **Graceful Degradation**: Backward compatibility with warnings
5. **Automated Testing**: CI pipeline fails on high-severity issues
6. **Documentation**: Clear migration path and usage examples

## Recommendations

1. **Regenerate Indexes**: Consider regenerating lexical/vector indexes to use secure format
2. **Monitor Logs**: Watch for legacy pickle format warnings
3. **Regular Updates**: Keep security dependencies updated
4. **Security Reviews**: Regular code reviews focusing on security implications
5. **Testing**: Test all serialization paths with new security measures

## Breaking Changes

**Minimal Breaking Changes**: The implementation maintains backward compatibility while encouraging migration to secure formats through warnings.

- Legacy pickle files still work but generate warnings
- New files use secure serialization formats (.msgpack extensions)
- CI pipeline now fails on high-severity security issues
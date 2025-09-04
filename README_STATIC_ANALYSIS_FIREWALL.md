# Static Analysis Firewall

This document describes the comprehensive static analysis firewall implemented for the EGW Query Expansion system.

## Overview

The static analysis firewall provides multiple layers of code quality and type safety enforcement:

1. **MyPy Strict Type Checking** - Comprehensive static type analysis
2. **Pyright Type Checking** - Microsoft's advanced type checker 
3. **Ruff Linting** - Fast Python linter with import organization
4. **Import Validation** - Custom scripts to detect problematic imports
5. **Circular Import Detection** - Dependency graph analysis
6. **Pre-commit Hooks** - Automated checks before commits
7. **CI Pipeline Integration** - Continuous validation

## Configuration Files

### MyPy Configuration (`mypy.ini`)

Strict type checking configuration:
- `strict = true` - Enable all strict mode checks
- `disallow_untyped_defs = true` - Require type annotations for all functions
- `disallow_any_generics = true` - Prevent usage of bare generics
- `warn_return_any = true` - Warn when returning `Any` type
- Additional strict settings for comprehensive type safety

### Pyright Configuration (`pyproject.toml`)

Advanced static analysis settings:
- `typeCheckingMode = "strict"` - Maximum type checking strictness
- `reportImportCycles = "error"` - Detect circular imports
- `reportWildcardImportFromLibrary = "error"` - Block star imports
- Comprehensive diagnostic settings for all error types

### Ruff Configuration (`pyproject.toml`)

Import organization and linting:
- **TCH (Type-Checking)** rules - Enforce proper import organization
- **I (Import)** rules - Import sorting and organization
- Additional rules for code quality and security
- Automatic fixing capabilities

## Static Analysis Tools

### 1. Import Validation (`scripts/validate_imports.py`)

Detects and prevents:
- Star imports (`from module import *`)
- Syntax errors in Python files
- Problematic import patterns

Usage:
```bash
python3 scripts/validate_imports.py
```

### 2. Circular Import Detection (`scripts/detect_circular_imports.py`)

Features:
- Builds dependency graphs for all Python modules
- Detects circular dependencies using DFS algorithm
- Provides detailed cycle information
- Dependency statistics and analysis

Usage:
```bash
python3 scripts/detect_circular_imports.py
```

### 3. Runtime Import Guard (`scripts/runtime_import_guard.py`)

Runtime protection:
- Thread-safe circular import detection
- Automatic warning system
- Meta path finder integration
- Context manager for guarded imports

Usage:
```python
from scripts.runtime_import_guard import install_import_guard

install_import_guard()
# Your imports here...
```

### 4. Setup Script (`scripts/setup_static_analysis.py`)

Automated configuration:
- Installs all required dependencies
- Validates tool configurations
- Sets up pre-commit hooks
- Generates validation reports

Usage:
```bash
python3 scripts/setup_static_analysis.py
```

## Pre-commit Hooks

The `.pre-commit-config.yaml` includes:

1. **Ruff** - Linting and formatting
2. **MyPy** - Static type checking
3. **Pyright** - Advanced type analysis
4. **Import Validation** - Custom import checks
5. **Circular Import Detection** - Dependency validation
6. **Bandit** - Security analysis
7. **YAML/JSON Validation** - Configuration file validation

## CI Pipeline Integration

### GitHub Workflow (`.github/workflows/static-analysis.yml`)

Multi-job validation:

#### Static Analysis Job
- Runs on Python 3.8, 3.9, 3.10, 3.11
- MyPy strict type checking
- Pyright type analysis
- Ruff linting (TCH and I rules)
- Bandit security scanning
- Import validation
- Circular import detection

#### Dependency Analysis Job
- Dependency tree analysis with `pipdeptree`
- Security audits with `safety` and `pip-audit`
- Vulnerability reporting

#### Pre-commit Validation Job
- Validates pre-commit configuration
- Runs all hooks on all files

#### Type Coverage Job
- Generates type coverage reports
- HTML and text format reports

## Installation and Setup

### Prerequisites

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

### Install Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# Install static analysis tools
pip install mypy pyright ruff bandit pre-commit
pip install types-PyYAML types-requests types-setuptools types-toml
```

### Setup Static Analysis

```bash
# Run automated setup
python3 scripts/setup_static_analysis.py

# Or manually install pre-commit hooks
pre-commit install
pre-commit install --hook-type pre-push
```

## Usage

### Manual Validation

```bash
# Type checking
mypy --config-file=mypy.ini egw_query_expansion/ src/ scripts/
pyright --project=pyproject.toml

# Linting
ruff check --select=TCH,I .
ruff format --check .

# Security analysis
bandit -r egw_query_expansion/ src/ scripts/

# Import validation
python3 scripts/validate_imports.py
python3 scripts/detect_circular_imports.py

# Run all pre-commit hooks
pre-commit run --all-files
```

### Automated Validation

Pre-commit hooks run automatically on:
- `git commit` - Basic validation
- `git push` - Full validation

CI pipeline runs on:
- Push to main/develop branches
- Pull requests to main/develop

## Error Types and Fixes

### MyPy Errors

Common fixes:
- Add type annotations to functions
- Use proper generic types instead of `Any`
- Add return type annotations
- Import types in `TYPE_CHECKING` blocks

### Ruff TCH/I Errors

Common fixes:
- Move type-only imports to `TYPE_CHECKING` blocks
- Sort imports properly
- Remove star imports
- Organize import sections

### Import Validation Errors

Common fixes:
- Replace `from module import *` with specific imports
- Fix syntax errors
- Resolve circular imports

### Circular Import Errors

Common fixes:
- Move imports inside functions (lazy imports)
- Refactor code to break dependency cycles
- Use forward references with `TYPE_CHECKING`
- Extract common interfaces to separate modules

## Integration with Development Workflow

### Git Hooks Integration

The firewall integrates with the development workflow through:
1. Pre-commit hooks prevent problematic code from being committed
2. Pre-push hooks run comprehensive validation
3. CI pipeline provides final verification

### IDE Integration

Configure your IDE to use the same settings:
- MyPy configuration: `mypy.ini`
- Pyright configuration: `pyproject.toml` 
- Ruff configuration: `pyproject.toml`

### Build Integration

The static analysis firewall is integrated with the build process:
- CI fails if any static analysis violations are found
- Pull requests are automatically rejected for violations
- Type coverage reports are generated and archived

## Maintenance

### Updating Configurations

When updating static analysis configurations:
1. Test changes locally first
2. Update version pins in requirements.txt
3. Validate with `pre-commit run --all-files`
4. Update this documentation

### Adding New Rules

To add new validation rules:
1. Update appropriate configuration files
2. Test on existing codebase
3. Fix any new violations
4. Update CI pipeline if needed

## Benefits

This static analysis firewall provides:

1. **Type Safety** - Comprehensive type checking prevents runtime type errors
2. **Import Safety** - Prevents circular imports and star import anti-patterns
3. **Code Quality** - Consistent code style and organization
4. **Security** - Automated security vulnerability detection
5. **Maintainability** - Clear dependency relationships and documentation
6. **CI Integration** - Automated validation in continuous integration

## Troubleshooting

### Common Issues

1. **Type stub missing** - Install appropriate `types-*` packages
2. **Import not found** - Check PYTHONPATH and module structure
3. **Circular import detected** - Use lazy imports or refactor code
4. **Pre-commit slow** - Use `--hook-stage manual` for expensive checks

### Performance Optimization

For large codebases:
- Use MyPy daemon mode (`dmypy`)
- Configure Ruff to exclude large directories
- Use incremental type checking
- Parallelize CI jobs where possible

## Future Enhancements

Planned improvements:
- Integration with additional type checkers (pyre, pytype)
- Custom rule development for domain-specific validation
- Performance optimization for large codebases
- Integration with code coverage tools
- Automated refactoring suggestions
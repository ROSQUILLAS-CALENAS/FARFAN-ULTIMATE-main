# Dependency Doctor - Architecture Compliance Validator

The Dependency Doctor (`tools/dep_doctor.py`) is a comprehensive CLI tool that performs pre-commit validation by scanning the codebase for architecture violations that compromise the canonical pipeline's phase ordering and interface contracts.

## Features

### 🔍 Violation Detection
- **Phase Ordering Violations**: Detects backward dependencies that violate the I→X→K→A→L→R→O→G→T→S canonical flow
- **Missing Annotations**: Identifies modules lacking mandatory `__phase__`, `__code__`, and `__stage_order__` annotations  
- **Signature Drift**: Validates that modules conform to the standard `process(data, context) -> Dict[str, Any]` interface
- **Circular Dependencies**: Detects import cycles that create hot nodes requiring adapter patterns

### 🔧 Auto-Fix Capabilities
- **Annotation Injection**: Automatically adds missing `__phase__`, `__code__`, and `__stage_order__` annotations
- **Process Function Scaffolding**: Generates standard `process()` function templates with proper signatures
- **Adapter Generation**: Creates adapter/bridge pattern scaffolding for breaking circular dependencies
- **Port Interface Stubs**: Auto-generates interface stubs for quarantining hot nodes

### 🏥 Remediation Suggestions
Provides specific, actionable suggestions for each violation type:
- Dependency injection patterns for backward dependencies
- Adapter/bridge patterns for circular dependencies  
- Interface segregation for hot node quarantine
- Proper phase ordering restructuring recommendations

## Usage

### Basic Validation
```bash
# Scan entire codebase
python3 tools/dep_doctor.py

# Scan specific path
python3 tools/dep_doctor.py --path canonical_flow

# Verbose output with detailed violations
python3 tools/dep_doctor.py --verbose
```

### Auto-Fix Mode
```bash
# Automatically fix simple violations
python3 tools/dep_doctor.py --auto-fix

# Fix specific path and save report
python3 tools/dep_doctor.py --auto-fix --path canonical_flow --output report.json
```

### Pre-Commit Integration
The tool is configured to run automatically on pre-commit via `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: local
    hooks:
      - id: dep-doctor
        name: Dependency Doctor - Architecture Validator
        entry: python3 tools/dep_doctor.py
        language: system
        pass_filenames: false
        files: '\.py$'
        always_run: true
```

Install pre-commit hooks:
```bash
pip install pre-commit
pre-commit install
```

## Violation Types

### 1. Phase Ordering Violations
**Type**: `BACKWARD_DEPENDENCY_VIOLATION`
**Description**: Module in earlier phase depends on later phase module

**Example**:
```
Phase I module depends on later phase R module canonical_flow/retrieval_engine/hybrid_retriever.py
```

**Fix**: Use dependency injection or adapter pattern to reverse the dependency flow.

### 2. Missing Annotations
**Types**: 
- `MISSING_PHASE_ANNOTATION` 
- `MISSING_CODE_ANNOTATION`
- `MISSING_STAGE_ORDER_ANNOTATION`

**Description**: Canonical flow modules missing mandatory metadata annotations

**Auto-fix**: Tool infers appropriate values from file path and adds annotations:
```python
__phase__ = "I"           # Ingestion & Preparation
__code__ = "01I"          # Stage 1 Ingestion  
__stage_order__ = 1       # Execution order
```

### 3. Interface Violations  
**Types**:
- `MISSING_PROCESS_FUNCTION`
- `INVALID_PROCESS_SIGNATURE` 
- `INVALID_PROCESS_RETURN_TYPE`

**Description**: Modules not conforming to standard `process(data, context) -> Dict[str, Any]` interface

**Auto-fix**: Generates compliant process function template:
```python
def process(data=None, context=None) -> Dict[str, Any]:
    """Standard process function interface."""
    return {
        "status": "success",
        "data": data,
        "processed_at": str(datetime.now())
    }
```

### 4. Circular Dependencies
**Type**: `CIRCULAR_DEPENDENCY`
**Description**: Import cycles detected in dependency graph

**Auto-fix**: Generates adapter scaffolding in `adapters/` directory to break cycles using:
- Adapter pattern for delegation
- Bridge pattern for implementation decoupling
- Port interfaces for hot node quarantine

## Phase Mapping

The tool understands the canonical phase ordering:

| Phase | Name | Description |
|-------|------|-------------|
| **I** | Ingestion & Preparation | Data loading, validation, preprocessing |
| **X** | Context Construction | Immutable context building |  
| **K** | Knowledge Extraction | Entity extraction, knowledge graphs |
| **A** | Analysis & NLP | Text analysis, evidence processing |
| **L** | Classification & Evaluation | Scoring, classification, risk assessment |
| **R** | Search & Retrieval | Vector search, hybrid retrieval |
| **O** | Orchestration & Control | Workflow management, routing |
| **G** | Aggregation & Reporting | Result compilation, reporting |
| **T** | Integration & Storage | Persistence, external integration |
| **S** | Synthesis & Output | Final answer synthesis |

## Exit Codes

- **0**: No violations detected - architecture compliant
- **1**: Violations found - architecture validation failed

Perfect for CI/CD pipeline integration where architecture compliance blocks deployment.

## Adapter Templates

The tool includes `tools/adapter_bridge_templates.py` with code generation templates:

### Adapter Pattern
```python
class IToRAdapter:
    """Adapter to break I→R backward dependency"""
    def __init__(self):
        self._target_instance = None
    
    def set_target(self, target_instance):
        self._target_instance = target_instance
    
    def process(self, data, context=None):
        return self._target_instance.process(data, context)
```

### Bridge Pattern  
```python
class IRBridge:
    """Bridge for I→R implementation decoupling"""
    def set_implementation(self, implementation):
        self._implementation = implementation
        
    def process(self, data, context=None):
        return self._implementation.process(data, context)
```

### Port Interface
```python
class IProcessorInterface(ABC):
    """Port interface for I phase quarantine"""
    @abstractmethod
    def process(self, data, context=None) -> Dict[str, Any]:
        pass
```

## Integration Example

```bash
# Development workflow
git add .
git commit -m "feature: add new analysis module"
# → Dependency Doctor runs automatically
# → Blocks commit if architecture violations detected
# → Provides specific remediation suggestions

# Fix violations and re-commit
python3 tools/dep_doctor.py --auto-fix
git add .
git commit -m "fix: resolve architecture violations"
# → Commit succeeds once compliant
```

This ensures the canonical pipeline architecture remains deterministic and maintainable across all development cycles.
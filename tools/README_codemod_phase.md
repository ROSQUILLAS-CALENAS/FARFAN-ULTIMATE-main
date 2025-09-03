# CodeMod Phase - Advanced AST-based Refactoring Tool

A comprehensive refactoring tool that uses LibCST and Bowler for safe AST-based codebase transformations with integrated validation, rollback capabilities, and detailed logging.

## 🎯 Features

### Core Functionality
- **Safe AST-based refactoring** using LibCST and Bowler
- **Canonical naming cleanup** for 05I_* prefix patterns
- **Automated import updates** throughout the codebase
- **Phase directory reorganization** following canonical structure
- **Content cleanup** removing old naming pattern references

### Safety & Validation
- **Pre-commit validation gates** with MyPy and Ruff integration
- **Post-refactoring validation** ensuring code quality
- **Syntax checking** before and after changes
- **Dry-run capabilities** for safe preview of changes
- **Automatic rollback mechanisms** on validation failures

### Reliability Features
- **Comprehensive backup system** before any changes
- **Detailed operation logging** with timestamps
- **Graceful dependency handling** with fallback mechanisms
- **Regex-based fallbacks** when LibCST is unavailable
- **Transaction-like operations** with full rollback support

## 📦 Installation

### Required Dependencies
```bash
pip install libcst bowler mypy ruff
```

### Optional Dependencies (for full functionality)
```bash
# For enhanced AST parsing and transformations
pip install bowler libcst

# For validation and code quality
pip install mypy ruff black isort
```

## 🚀 Usage

### Basic Usage
```bash
# Dry run to preview changes
python3 tools/codemod_phase.py --dry-run --verbose

# Generate detailed report only
python3 tools/codemod_phase.py --report-only

# Execute actual refactoring
python3 tools/codemod_phase.py --verbose

# Rollback last operation
python3 tools/codemod_phase.py --rollback
```

### Command Line Options
- `--project-root PATH` - Project root directory (default: current directory)
- `--dry-run` - Perform dry run without making changes
- `--verbose, -v` - Enable verbose logging
- `--rollback` - Rollback the last refactoring operation
- `--report-only` - Generate report without making changes

### Python API Usage
```python
from tools.codemod_phase import CodemodPhase

# Initialize refactoring tool
codemod = CodemodPhase(
    project_root=".",
    dry_run=True,  # Safe preview mode
    verbose=True
)

# Execute refactoring
result = codemod.execute_refactoring()

# Check results
if result.success:
    print(f"✅ Refactoring completed: {len(result.operations)} operations")
    print(f"Validation: {result.validation_results}")
else:
    print(f"❌ Refactoring failed: {result.errors}")
    if result.rollback_info:
        print(f"Rollback available: {result.rollback_info['backup_dir']}")
```

## 🔍 What It Detects & Fixes

### Naming Pattern Issues
- **Files with 05I_* prefixes** → Clean canonical naming
- **Content references** to old patterns → Updated references
- **Import statements** using old names → Corrected imports

### Canonical Structure Issues
- **Misplaced components** → Moved to correct phase directories
- **Missing phase directories** → Created with proper structure
- **Inconsistent imports** → Updated throughout codebase

### Example Transformations
```python
# BEFORE
"""
05I_raw_data_generator.py - Deterministic Raw Data Artifacts Generator
Canonical Flow Module: I_ingestion_preparation/05I_raw_data_generator.py
"""

# AFTER  
"""
raw_data_generator.py - Deterministic Raw Data Artifacts Generator
Canonical Flow Module: I_ingestion_preparation/raw_data_generator.py
"""

# Import updates throughout codebase:
# from canonical_flow.I_ingestion_preparation.05I_raw_data_generator import process
# ↓
# from canonical_flow.I_ingestion_preparation.raw_data_generator import process
```

## 📊 Validation & Quality Gates

### Pre-commit Validation
- **Syntax checking** of existing Python files
- **Tool availability** verification (mypy, ruff)
- **Codebase health** assessment

### Post-commit Validation  
- **MyPy type checking** on changed files
- **Ruff linting** and format checking
- **Syntax validation** of all modified files
- **Import resolution** verification

### Validation Results
```python
validation_results = {
    'mypy': True,     # Type checking passed
    'ruff': True,     # Linting passed  
    'syntax': True,   # All files have valid syntax
}
```

## 🔒 Safety Features

### Backup & Rollback System
```python
# Automatic backup creation
with codemod.create_backup():
    # Perform refactoring operations
    operations = execute_refactoring()
    
    # Automatic rollback on failure
    if validation_fails():
        codemod.rollback()
```

### Operation Tracking
```json
{
    "backup_dir": "/tmp/codemod_backup_20231203_143022/",
    "timestamp": "2023-12-03T14:30:22.123456",
    "operations": [
        {
            "type": "move",
            "from": "new/path/file.py", 
            "to": "original/path/file.py"
        }
    ]
}
```

## 📋 Phase Directory Mapping

The tool recognizes these canonical phase mappings:

| Old Prefix | Canonical Phase Directory | Purpose |
|------------|---------------------------|---------|
| `05I_*` | `I_ingestion_preparation` | Data ingestion & preparation |
| `06K_*` | `K_knowledge_extraction` | Knowledge extraction |
| `07A_*` | `A_analysis_nlp` | NLP analysis |
| `08R_*` | `R_search_retrieval` | Search & retrieval |
| `09S_*` | `S_synthesis_output` | Output synthesis |
| `10T_*` | `T_integration_storage` | Integration & storage |
| `11L_*` | `L_classification_evaluation` | Classification & evaluation |
| `12G_*` | `G_aggregation_reporting` | Aggregation & reporting |
| `13O_*` | `O_orchestration_control` | Orchestration & control |
| `14X_*` | `X_context_construction` | Context construction |

## 📈 Reports & Logging

### Detailed Reports
Generated reports include:
- **Operation summary** with file counts and types
- **Validation results** for each quality gate
- **Error details** with specific failure reasons
- **Rollback information** for recovery
- **Performance metrics** and timing data

### Log Files
- **Comprehensive logging** to timestamped files
- **Debug information** for troubleshooting
- **Operation tracking** for audit trails
- **Performance monitoring** data

### Sample Report
```
================================================================================
CODEMOD PHASE REFACTORING REPORT
================================================================================
Timestamp: 2023-12-03T14:30:22.123456
Project: /path/to/project
Dry Run: False
Success: True

OPERATIONS PERFORMED (3):
----------------------------------------
1. RENAME_AND_MOVE
   Source: tools/05I_data_generator.py
   Target: canonical_flow/I_ingestion_preparation/data_generator.py
   
2. UPDATE_CONTENT  
   Source: canonical_flow/I_ingestion_preparation/raw_data_generator.py
   Action: Cleaned 05I_ pattern references

VALIDATION RESULTS:
----------------------------------------
mypy: PASSED
ruff: PASSED
syntax: PASSED
```

## 🛠 Troubleshooting

### Common Issues

**Missing Dependencies**
```bash
# Install required packages
pip install libcst bowler mypy ruff
```

**Validation Failures**
```bash
# Check specific validation results
python3 tools/codemod_phase.py --dry-run --verbose
```

**Rollback Needed**
```bash
# Rollback last operation
python3 tools/codemod_phase.py --rollback
```

### Debug Mode
```bash
# Maximum verbosity for troubleshooting
python3 tools/codemod_phase.py --dry-run --verbose
```

## 🔧 Extending the Tool

### Custom Refactoring Operations
```python
from tools.codemod_phase import RefactoringOperation

# Define custom operation
custom_op = RefactoringOperation(
    operation_type='custom_rename',
    source_path=Path('old/file.py'),
    target_path=Path('new/file.py'),
    metadata={'custom_flag': True}
)
```

### Custom Validation Gates
```python
class CustomValidator(SafetyValidator):
    def run_custom_validation(self, files):
        # Custom validation logic
        return success, output
```

## 📝 Best Practices

1. **Always run dry-run first** to preview changes
2. **Install all dependencies** for full functionality
3. **Review generated reports** before applying changes
4. **Keep backups** of critical codebase states
5. **Run tests** after refactoring operations
6. **Use version control** for additional safety

## 🤝 Contributing

To extend or modify the tool:
1. Follow existing code patterns
2. Add comprehensive tests
3. Update documentation
4. Ensure backward compatibility
5. Test with various codebases

## 📄 License

MIT License - see LICENSE file for details.
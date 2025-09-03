# Pipeline Index System

A comprehensive system for managing, validating, and visualizing pipeline components with automated synchronization between the definitive specification and filesystem reality.

## Overview

This system provides:

1. **📋 Index Specification** (`pipeline_index.json`) - Definitive source of truth for all pipeline components
2. **🔍 Autoscan System** (`pipeline_autoscan.py`) - Automated reconciliation between index and filesystem  
3. **🎨 DAG Visualization** (`pipeline_dag_visualizer.py`) - Generate visual representations of pipeline flow
4. **✅ Validation System** (`pipeline_validation_system.py`) - Ensure consistency and fail builds on mismatches
5. **🤖 CI/CD Integration** - Automated validation in GitHub Actions
6. **🪝 Git Hooks** - Pre-commit validation to catch issues early

## Quick Start

### 1. Install System Dependencies

```bash
# Ubuntu/Debian
sudo apt-get install graphviz

# macOS
brew install graphviz

# Windows (via chocolatey)
choco install graphviz
```

### 2. Set Up Git Hooks

```bash
./install_git_hooks.sh
```

### 3. Run Initial Sync

```bash
# Scan filesystem and update index
python3 pipeline_autoscan.py

# Validate everything is correct
python3 pipeline_validation_system.py --strict --report validation_report.md

# Generate visualizations
python3 pipeline_dag_visualizer.py --format all
```

## Core Components

### Pipeline Index (`pipeline_index.json`)

The authoritative specification containing:

```json
{
  "version": "1.0.0",
  "metadata": {
    "description": "Canonical pipeline component specification index",
    "total_components": 58,
    "phases": ["ingestion_preparation", "context_construction", ...],
    "maintainer": "automated_index_system"
  },
  "components": [
    {
      "name": "pdf_reader",
      "code": "01I",
      "phase": "ingestion_preparation", 
      "dependencies": [],
      "canonical_path": "canonical_flow/I_ingestion_preparation/pdf_reader.py",
      "original_path": "pdf_reader.py",
      "description": "PDF document reader and text extractor",
      "enabled": true,
      "entry_point": true
    }
  ]
}
```

**Key Fields:**
- `name` - Component identifier
- `code` - Short code (e.g., 01I, 02X) for ordering and dependencies
- `phase` - Pipeline stage classification
- `dependencies` - Array of component codes this depends on
- `canonical_path` - Normalized location in canonical structure
- `original_path` - Original source location (if moved)
- `enabled` - Whether component is active
- `entry_point` - True for pipeline entry points

### Autoscan System (`pipeline_autoscan.py`)

Maintains synchronization between the index and filesystem reality.

**Features:**
- 🔍 **Filesystem Scanning** - Discovers components automatically
- 🔄 **Change Detection** - Identifies added, removed, modified, and moved components
- 📝 **Index Updates** - Keeps index synchronized with filesystem
- 🏷️ **Auto-Classification** - Infers phases from directory structure
- 🔒 **Hash Validation** - Detects file modifications

**Usage:**
```bash
# Scan and update index
python3 pipeline_autoscan.py

# Scan without updating (dry run)
python3 pipeline_autoscan.py --no-update

# Validate index integrity
python3 pipeline_autoscan.py --validate

# Output results to JSON
python3 pipeline_autoscan.py --output scan_results.json
```

### DAG Visualization (`pipeline_dag_visualizer.py`)

Generates visual representations of the pipeline dependency graph.

**Output Formats:**
- 🖼️ PNG/SVG - High-quality graphics via Graphviz
- 📊 DOT - Graphviz source format  
- 🌊 Mermaid - Text-based diagrams for documentation

**Features:**
- 📈 **Dependency Graph** - Shows component relationships
- 🎨 **Phase Layering** - Visual grouping by pipeline phases
- 🔴 **Status Indicators** - Disabled components marked clearly
- ✅ **Cycle Detection** - Validates DAG integrity

**Usage:**
```bash
# Generate all formats
python3 pipeline_dag_visualizer.py --format all

# Generate specific format
python3 pipeline_dag_visualizer.py --format png --output my_dag

# Validate DAG structure
python3 pipeline_dag_visualizer.py --validate
```

### Validation System (`pipeline_validation_system.py`)

Ensures filesystem reality matches the index specification.

**Validation Checks:**
- ✅ **File Existence** - All indexed components exist on filesystem
- 🔐 **Hash Integrity** - File contents match stored hashes
- 🔗 **Dependency Integrity** - All dependencies exist and are valid
- 🚫 **Cycle Detection** - No circular dependencies
- 📁 **Path Consistency** - Components in correct canonical locations
- 🏷️ **Phase Organization** - Proper phase directory structure

**Usage:**
```bash
# Full validation with report
python3 pipeline_validation_system.py --strict --report validation_report.md

# Build validation mode (for CI/CD)
python3 pipeline_validation_system.py --build-mode

# JSON output for automation
python3 pipeline_validation_system.py --json-output results.json
```

## Pipeline Phases

The system organizes components into logical phases:

| Phase | Code | Directory | Description |
|-------|------|-----------|-------------|
| **Ingestion Preparation** | I | `I_ingestion_preparation/` | Data input and preprocessing |
| **Context Construction** | X | `X_context_construction/` | Build processing context |
| **Knowledge Extraction** | K | `K_knowledge_extraction/` | Extract semantic knowledge |
| **Analysis NLP** | A | `A_analysis_nlp/` | Natural language processing |
| **Classification Evaluation** | L | `L_classification_evaluation/` | Scoring and classification |
| **Orchestration Control** | O | `O_orchestration_control/` | Process coordination |
| **Search Retrieval** | R | `R_search_retrieval/` | Information retrieval |
| **Synthesis Output** | S | `S_synthesis_output/` | Generate final outputs |
| **Aggregation Reporting** | G | `G_aggregation_reporting/` | Compile reports and metrics |
| **Integration Storage** | T | `T_integration_storage/` | Data persistence and integration |

## CI/CD Integration

### GitHub Actions Workflow

The system includes a comprehensive CI/CD workflow (`.github/workflows/pipeline_validation.yml`) that:

1. 🔍 **Runs autoscan** to detect component changes
2. ✅ **Validates DAG structure** for cycles and integrity
3. 🧪 **Performs strict validation** of index consistency  
4. 🎨 **Generates visualizations** for documentation
5. 📊 **Comments on PRs** with validation results
6. 🤖 **Auto-updates index** on main branch after validation passes

### Pre-commit Hooks

Git hooks prevent invalid commits:

```bash
# Install hooks
./install_git_hooks.sh

# Hooks will run automatically on commit
git commit -m "Update component"

# Bypass hooks if needed (not recommended)
git commit --no-verify -m "Emergency fix"
```

## Development Workflow

### Adding New Components

1. **Create component** in appropriate phase directory:
   ```bash
   # Example: new ingestion component
   touch canonical_flow/I_ingestion_preparation/new_processor.py
   ```

2. **Run autoscan** to detect and add to index:
   ```bash
   python3 pipeline_autoscan.py
   ```

3. **Update dependencies** in `pipeline_index.json` if needed:
   ```json
   {
     "name": "new_processor",
     "dependencies": ["01I", "02I"]
   }
   ```

4. **Validate changes**:
   ```bash
   python3 pipeline_validation_system.py --strict
   ```

### Modifying Existing Components

1. **Edit component files** as needed
2. **Autoscan detects changes** automatically on commit via pre-commit hook
3. **CI/CD validates** changes and updates visualizations
4. **Index updates** automatically on merge to main

### Troubleshooting

**Common Issues:**

- **Missing dependencies**: Add required component codes to `dependencies` array
- **Hash mismatches**: Normal after editing files - autoscan will update hashes
- **Circular dependencies**: Check dependency graph and break cycles
- **Orphaned files**: Move files to canonical structure or add to index
- **Phase mismatches**: Ensure components are in correct phase directories

**Debug Commands:**
```bash
# Detailed validation with report
python3 pipeline_validation_system.py --strict --report debug_report.md

# Check DAG for cycles
python3 pipeline_dag_visualizer.py --validate

# View autoscan changes without updating
python3 pipeline_autoscan.py --no-update --output debug_scan.json
```

## Advanced Usage

### Custom Phase Organization

To add new phases, update `phase_prefixes` in the autoscan system:

```python
# In pipeline_autoscan.py
self.phase_prefixes = {
    'I_': 'ingestion_preparation',
    'N_': 'new_phase',  # Add custom phase
    # ...
}
```

### Validation Rules

Customize validation behavior in `pipeline_validation_system.py`:

```python
self.validation_rules = {
    'require_index_file': True,
    'validate_file_hashes': True,
    'validate_dependencies': True,
    'require_descriptions': True,  # Set to False for optional descriptions
    # ...
}
```

### Visualization Styling

Modify colors and styling in `pipeline_dag_visualizer.py`:

```python
self.phase_colors = {
    'ingestion_preparation': '#FF6B6B',  # Red
    'custom_phase': '#123ABC',           # Custom color
    # ...
}
```

## API Reference

### PipelineAutoscan

```python
from pipeline_autoscan import PipelineAutoscan

scanner = PipelineAutoscan("pipeline_index.json", "canonical_flow")
result = scanner.run_autoscan(update_index=True)
```

### PipelineDAGVisualizer

```python
from pipeline_dag_visualizer import PipelineDAGVisualizer

visualizer = PipelineDAGVisualizer("pipeline_index.json")
results = visualizer.generate_all_formats("my_dag")
```

### PipelineValidationSystem

```python
from pipeline_validation_system import PipelineValidationSystem

validator = PipelineValidationSystem("pipeline_index.json", strict_mode=True)
result = validator.run_full_validation()
```

## Contributing

1. 🔧 **Install hooks**: `./install_git_hooks.sh`
2. 🧪 **Run tests**: `python3 -m pytest tests/`
3. ✅ **Validate changes**: `python3 pipeline_validation_system.py --strict`
4. 📊 **Update docs**: Regenerate visualizations if needed
5. 🚀 **Submit PR**: CI/CD will validate automatically

## License

This pipeline index system is part of the larger project and follows the same licensing terms.
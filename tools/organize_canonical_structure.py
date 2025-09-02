#!/usr/bin/env python3
"""
Canonical Flow Organization Tool

Generates a canonical, ordered view of the pipeline by creating alias files
that re-export original modules in deterministic topological order.

Usage:
    python tools/organize_canonical_structure.py [--dry-run] [--copy-tests]
    
Features:
- Creates canonical_flow/ directory structure by stage
- Generates alias files with pattern: NN<Letter>_<slug>.py
- Builds index.json and README.md for human/machine reference
- Does NOT modify original files - only creates organized aliases
"""

import argparse
import json
import logging
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Stage mapping: ProcessStage -> single letter code
STAGE_MAP = {
    "ingestion_preparation": "I",
    "context_construction": "X", 
    "knowledge_extraction": "K",
    "analysis_nlp": "A",
    "classification_evaluation": "L",
    "search_retrieval": "R",
    "orchestration_control": "O",
    "aggregation_reporting": "G",
    "integration_storage": "T",
    "synthesis_output": "S",
}

# Stage folder mapping
STAGE_FOLDERS = {
    "I": "I_ingestion_preparation",
    "X": "X_context_construction",
    "K": "K_knowledge_extraction", 
    "A": "A_analysis_nlp",
    "L": "L_classification_evaluation",
    "R": "R_search_retrieval",
    "O": "O_orchestration_control",
    "G": "G_aggregation_reporting",
    "T": "T_integration_storage",
    "S": "S_synthesis_output",
}

logger = logging.getLogger(__name__)


def sanitize_slug(filename: str) -> str:
    """Convert filename to import-safe slug."""
    # Remove .py extension
    base = filename.replace(".py", "")
    # Replace non-alphanumeric with underscore
    slug = re.sub(r"[^a-zA-Z0-9]", "_", base)
    # Remove leading/trailing underscores
    slug = slug.strip("_")
    # Fallback for empty slug
    return slug or "module"


def create_alias_content(original_path: str, stage: str, alias_code: str) -> str:
    """Generate the content for an alias file."""
    return f'''"""
Canonical Flow Alias: {alias_code}

This is an auto-generated alias file that re-exports the original module.
DO NOT EDIT - changes will be overwritten by organize_canonical_structure.py

Source: {original_path}
Stage: {stage}
Code: {alias_code}
"""

import sys
from pathlib import Path
from importlib import util as importlib_util

# Alias metadata
alias_source = r"{original_path}"
alias_stage = "{stage}"
alias_code = "{alias_code}"

# Dynamically load and re-export the original module
try:
    # Add project root to path for imports
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # Load original module
    original_file = project_root / "{original_path}"
    if original_file.exists():
        spec = importlib_util.spec_from_file_location(
            f"original_{{alias_code.lower()}}", 
            str(original_file)
        )
        
        if spec and spec.loader:
            original_module = importlib_util.module_from_spec(spec)
            spec.loader.exec_module(original_module)
            
            # Re-export all public symbols
            for attr_name in dir(original_module):
                if not attr_name.startswith("_"):
                    globals()[attr_name] = getattr(original_module, attr_name)
        else:
            raise ImportError(f"Could not load spec for {{original_file}}")
    else:
        raise FileNotFoundError(f"Original file not found: {{original_file}}")
        
except Exception as e:
    import warnings
    warnings.warn(f"Failed to load original module {{alias_source}}: {{e}}")
    
    # Create placeholder functions to prevent import errors
    def process(data=None, context=None):
        """Placeholder process function for failed import."""
        return {{"error": f"Module {{alias_source}} failed to load: {{e}}"}}
'''


def get_orchestrator_graph() -> Tuple[Dict[str, Any], List[str]]:
    """Load the orchestrator graph and compute execution order."""
    try:
        # Import the orchestrator with fallback for missing dependencies
        import sys
        project_root = Path(__file__).resolve().parent.parent
        sys.path.insert(0, str(project_root))
        
        # Mock numpy if not available
        if 'numpy' not in sys.modules:
            import types
            numpy_mock = types.ModuleType('numpy')
            numpy_mock.random = types.ModuleType('random')
            numpy_mock.random.seed = lambda x: None
            numpy_mock.random.normal = lambda *args, **kwargs: [0.0] * kwargs.get('size', (1,))[0] if 'size' in kwargs else 0.0
            numpy_mock.random.choice = lambda arr, **kwargs: arr[0] if arr else None
            numpy_mock.random.uniform = lambda *args: 0.5
            numpy_mock.random.randint = lambda *args: 1
            numpy_mock.linalg = types.ModuleType('linalg')
            numpy_mock.linalg.norm = lambda x, **kwargs: 1.0
            numpy_mock.array = lambda x, **kwargs: x
            numpy_mock.mean = lambda x, **kwargs: sum(x) / len(x) if x else 0
            numpy_mock.std = lambda x, **kwargs: 0.0
            numpy_mock.var = lambda x, **kwargs: 0.0
            numpy_mock.zeros = lambda shape, **kwargs: [0.0] * (shape if isinstance(shape, int) else shape[0])
            sys.modules['numpy'] = numpy_mock
        
        # Mock pandas if not available
        if 'pandas' not in sys.modules:
            import types
            pandas_mock = types.ModuleType('pandas')
            
            class MockDataFrame:
                def __init__(self, data=None):
                    self.data = data or {}
                    self.values = MockValues()
                def to_parquet(self, path, **kwargs):
                    pass
                def __len__(self):
                    return 1000  # Mock length
                    
            class MockValues:
                def tobytes(self):
                    return b'mock_data'
                    
            pandas_mock.DataFrame = MockDataFrame
            sys.modules['pandas'] = pandas_mock
            
        from comprehensive_pipeline_orchestrator import ComprehensivePipelineOrchestrator
        
        orchestrator = ComprehensivePipelineOrchestrator()
        graph = orchestrator.process_graph
        
        # Compute topological order using the orchestrator's method
        execution_order = list(orchestrator._topological_sort())
        
        logger.info(f"Loaded graph with {len(graph)} nodes, execution order: {len(execution_order)} items")
        
        return graph, execution_order
        
    except Exception as e:
        logger.error(f"Failed to load orchestrator graph: {e}")
        # Create a minimal fallback graph based on known modules
        fallback_graph = create_fallback_graph()
        fallback_order = list(fallback_graph.keys())
        logger.warning("Using fallback graph due to import issues")
        return fallback_graph, fallback_order


def create_fallback_graph() -> Dict[str, Any]:
    """Create a fallback graph when orchestrator can't be loaded."""
    from types import SimpleNamespace
    
    # Define ProcessStage enum values
    stages = {
        "INGESTION": "ingestion_preparation",
        "CONTEXT_BUILD": "context_construction", 
        "KNOWLEDGE": "knowledge_extraction",
        "ANALYSIS": "analysis_nlp",
        "CLASSIFICATION": "classification_evaluation",
        "SEARCH": "search_retrieval",
        "ORCHESTRATION": "orchestration_control",
        "AGGREGATION": "aggregation_reporting",
        "INTEGRATION": "integration_storage",
        "SYNTHESIS": "synthesis_output"
    }
    
    def create_node(file_path: str, stage: str):
        node = SimpleNamespace()
        node.file_path = file_path
        node.stage = SimpleNamespace()
        node.stage.value = stage
        return node
    
    # Core modules we know exist
    fallback_graph = {
        "comprehensive_pipeline_orchestrator.py": create_node("comprehensive_pipeline_orchestrator.py", stages["ORCHESTRATION"]),
        "cluster_execution_controller.py": create_node("cluster_execution_controller.py", stages["ORCHESTRATION"]),
        "evidence_processor.py": create_node("evidence_processor.py", stages["ANALYSIS"]),
        "evidence_validation_model.py": create_node("evidence_validation_model.py", stages["ANALYSIS"]),
        "adaptive_scoring_engine.py": create_node("adaptive_scoring_engine.py", stages["CLASSIFICATION"]),
        "retrieval_engine/lexical_index.py": create_node("retrieval_engine/lexical_index.py", stages["SEARCH"]),
        "retrieval_engine/vector_index.py": create_node("retrieval_engine/vector_index.py", stages["SEARCH"]),
        "retrieval_engine/hybrid_retriever.py": create_node("retrieval_engine/hybrid_retriever.py", stages["SEARCH"]),
        "question_analyzer.py": create_node("question_analyzer.py", stages["ANALYSIS"]),
        "pdf_reader.py": create_node("pdf_reader.py", stages["INGESTION"]),
        "advanced_loader.py": create_node("advanced_loader.py", stages["INGESTION"]),
        "feature_extractor.py": create_node("feature_extractor.py", stages["INGESTION"]),
        "embedding_builder.py": create_node("embedding_builder.py", stages["KNOWLEDGE"]),
        "embedding_generator.py": create_node("embedding_generator.py", stages["KNOWLEDGE"]),
        "deterministic_router.py": create_node("deterministic_router.py", stages["ORCHESTRATION"]),
        "decision_engine.py": create_node("decision_engine.py", stages["ORCHESTRATION"]),
        "score_calculator.py": create_node("score_calculator.py", stages["CLASSIFICATION"]),
        "answer_formatter.py": create_node("answer_formatter.py", stages["SYNTHESIS"]),
        "report_compiler.py": create_node("report_compiler.py", stages["AGGREGATION"]),
        "canonical_output_auditor.py": create_node("canonical_output_auditor.py", stages["INTEGRATION"]),
    }
    
    return fallback_graph


def organize_canonical_structure(dry_run: bool = False, copy_tests: bool = False) -> Dict[str, Any]:
    """Main function to organize canonical flow structure."""
    
    project_root = Path(__file__).resolve().parent.parent
    canonical_dir = project_root / "canonical_flow"
    
    logger.info("Loading orchestrator graph...")
    graph, execution_order = get_orchestrator_graph()
    
    # Build canonical mapping
    canonical_mapping = {}
    stage_contents = {}
    
    for i, node_name in enumerate(execution_order, 1):
        if node_name not in graph:
            logger.warning(f"Node {node_name} in execution order but not in graph")
            continue
            
        node = graph[node_name]
        stage_name = node.stage.value
        stage_letter = STAGE_MAP.get(stage_name, "O")  # Default to O if unknown
        
        # Generate alias code
        alias_number = f"{i:02d}"
        alias_code = f"{alias_number}{stage_letter}"
        
        # Generate slug from filename
        slug = sanitize_slug(node.file_path.split("/")[-1])  # Get just filename
        alias_filename = f"{alias_code}_{slug}.py"
        
        # Stage folder
        stage_folder = STAGE_FOLDERS.get(stage_letter, f"{stage_letter}_unknown")
        
        canonical_mapping[node_name] = {
            "original_path": node.file_path,
            "stage": stage_name,
            "stage_letter": stage_letter,
            "alias_code": alias_code,
            "alias_filename": alias_filename,
            "stage_folder": stage_folder,
            "order": i
        }
        
        # Track stage contents
        if stage_folder not in stage_contents:
            stage_contents[stage_folder] = []
        stage_contents[stage_folder].append({
            "alias_filename": alias_filename,
            "original_path": node.file_path,
            "alias_code": alias_code,
            "order": i
        })
    
    if dry_run:
        print("DRY RUN - Would create:")
        for stage_folder, contents in stage_contents.items():
            print(f"  {stage_folder}/")
            for item in contents:
                print(f"    {item['alias_filename']} -> {item['original_path']}")
        return {"dry_run": True, "mapping": canonical_mapping}
    
    # Create directory structure and alias files
    logger.info(f"Creating canonical flow structure in {canonical_dir}")
    
    # Clean and create canonical_flow directory
    if canonical_dir.exists():
        shutil.rmtree(canonical_dir)
    canonical_dir.mkdir()
    
    # Create __init__.py for canonical_flow
    (canonical_dir / "__init__.py").write_text('''"""
Canonical Flow - Organized Pipeline View

This directory contains auto-generated alias files that provide a canonical,
ordered view of the pipeline. Each alias re-exports the original module.

DO NOT EDIT FILES IN THIS DIRECTORY - they will be overwritten.
Edit original files and regenerate using tools/organize_canonical_structure.py
"""
''')
    
    # Create stage directories and alias files
    for stage_folder, contents in stage_contents.items():
        stage_dir = canonical_dir / stage_folder
        stage_dir.mkdir()
        
        # Create stage __init__.py
        (stage_dir / "__init__.py").write_text(f'"""Stage: {stage_folder}"""\n')
        
        # Create alias files
        for item in contents:
            alias_path = stage_dir / item["alias_filename"]
            alias_content = create_alias_content(
                item["original_path"],
                canonical_mapping[execution_order[item["order"]-1]]["stage"],
                item["alias_code"]
            )
            alias_path.write_text(alias_content)
            logger.debug(f"Created alias: {alias_path}")
    
    # Create index.json
    index_data = {
        "generation_timestamp": "2025-01-24T00:00:00Z",
        "total_modules": len(canonical_mapping),
        "stages": list(stage_contents.keys()),
        "mapping": canonical_mapping,
        "execution_order": execution_order
    }
    
    with open(canonical_dir / "index.json", "w") as f:
        json.dump(index_data, f, indent=2, sort_keys=True)
    
    # Create README.md
    readme_content = generate_readme(canonical_mapping, stage_contents, execution_order)
    (canonical_dir / "README.md").write_text(readme_content)
    
    # Optionally copy tests
    if copy_tests:
        copy_test_files(project_root)
    
    logger.info(f"Successfully organized {len(canonical_mapping)} modules into {len(stage_contents)} stages")
    
    return {
        "success": True,
        "modules_organized": len(canonical_mapping),
        "stages_created": len(stage_contents),
        "canonical_dir": str(canonical_dir)
    }


def generate_readme(mapping: Dict[str, Any], stage_contents: Dict[str, List], execution_order: List[str]) -> str:
    """Generate README.md content for canonical flow."""
    
    readme = """# Canonical Flow - Pipeline Organization

This directory provides a canonical, deterministic view of the pipeline modules organized by stage and execution order.

## Structure

Each alias file follows the pattern: `NN<Letter>_<slug>.py` where:
- `NN` = two-digit order number (01, 02, ...)
- `<Letter>` = single letter stage code
- `<slug>` = sanitized original filename

## Stage Mapping

| Code | Stage | Folder |
|------|-------|--------|
| I | Ingestion & Preparation | I_ingestion_preparation |
| X | Context Construction | X_context_construction |
| K | Knowledge Extraction | K_knowledge_extraction |
| A | Analysis & NLP | A_analysis_nlp |
| L | Classification & Evaluation | L_classification_evaluation |
| R | Search & Retrieval | R_search_retrieval |
| O | Orchestration & Control | O_orchestration_control |
| G | Aggregation & Reporting | G_aggregation_reporting |
| T | Integration & Storage | T_integration_storage |
| S | Synthesis & Output | S_synthesis_output |

## Module Index

"""
    
    # Add modules by stage
    for stage_folder in sorted(stage_contents.keys()):
        contents = stage_contents[stage_folder]
        stage_letter = stage_folder[0]
        stage_name = stage_folder[2:]  # Remove "X_" prefix
        
        readme += f"### {stage_letter} - {stage_name.title().replace('_', ' ')}\n\n"
        
        # Sort by order
        sorted_contents = sorted(contents, key=lambda x: x["order"])
        
        for item in sorted_contents:
            alias_name = item["alias_filename"]
            original_path = item["original_path"] 
            order = item["order"]
            
            readme += f"- `{order:02d}` - [`{alias_name}`]({stage_folder}/{alias_name}) → `{original_path}`\n"
        
        readme += "\n"
    
    readme += """## Usage

Import from canonical flow for stable, ordered access:

```python
# Import specific module alias
from canonical_flow.R_search_retrieval import _28R_lexical_index as lexical_index

# Use re-exported functions
result = lexical_index.process(data, context)
```

## Important Notes

- **DO NOT EDIT** alias files - they are auto-generated and will be overwritten
- Edit original files when changing behavior
- Update orchestrator graph when changing pipeline structure
- Regenerate using: `python tools/organize_canonical_structure.py`

## Metadata

Each alias file contains:
- `alias_source` - path to original module
- `alias_stage` - pipeline stage name  
- `alias_code` - the NN<Letter> identifier

---

*Generated automatically by organize_canonical_structure.py*
"""
    
    return readme


def copy_test_files(project_root: Path) -> None:
    """Copy test files to central test directory."""
    test_dir = project_root / "test"
    test_dir.mkdir(exist_ok=True)
    
    # Find test files
    test_patterns = ["test_*.py", "*_test.py"]
    copied_count = 0
    
    for pattern in test_patterns:
        for test_file in project_root.glob(f"**/{pattern}"):
            if "canonical_flow" in str(test_file):
                continue  # Skip canonical flow directory
                
            dest_file = test_dir / test_file.name
            if not dest_file.exists():
                shutil.copy2(test_file, dest_file)
                copied_count += 1
                logger.debug(f"Copied test: {test_file} -> {dest_file}")
    
    logger.info(f"Copied {copied_count} test files to {test_dir}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Organize canonical flow structure",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true", 
        help="Preview without creating files"
    )
    parser.add_argument(
        "--copy-tests", 
        action="store_true", 
        help="Copy test files to central test directory"
    )
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true", 
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    try:
        result = organize_canonical_structure(
            dry_run=args.dry_run, 
            copy_tests=args.copy_tests
        )
        
        if args.dry_run:
            print("\nDry run completed successfully!")
        else:
            print(f"\nSuccessfully organized canonical flow:")
            print(f"  Modules: {result['modules_organized']}")
            print(f"  Stages: {result['stages_created']}")
            print(f"  Location: {result['canonical_dir']}")
            print("\nSee canonical_flow/README.md for details")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to organize canonical structure: {e}")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
#!/usr/bin/env python3
"""
Canonical Module Scaffolding Tool

Generates missing canonical modules in 08X-12X context construction and 52S-57S synthesis output 
ranges using Jinja2 templates. Includes standardized headers, OpenTelemetry spans, invariant 
assertions, and property-based test generation with Hypothesis.

Usage:
    python tools/scaffold_canonical_modules.py [--dry-run] [--verbose]
"""

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    from jinja2 import Environment, FileSystemLoader, Template
except ImportError:
    print("Error: jinja2 not installed. Run: pip install jinja2")
    exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModuleSpec:
    """Specification for a canonical module."""
    code: str
    phase: str
    stage: str
    stage_order: int
    dependencies: List[str] = field(default_factory=list)
    description: str = ""
    
    @property
    def file_name(self) -> str:
        """Generate the canonical file name."""
        return f"{self.code.lower()}_{self.stage}.py"
    
    @property
    def test_file_name(self) -> str:
        """Generate the test file name."""
        return f"test_{self.code.lower()}_{self.stage}.py"
    
    @property
    def directory_path(self) -> Path:
        """Get the phase directory path."""
        phase_mapping = {
            'context_construction': 'X_context_construction',
            'synthesis_output': 'S_synthesis_output',
            'knowledge_extraction': 'K_knowledge_extraction',
            'analysis_nlp': 'A_analysis_nlp',
            'classification_evaluation': 'L_classification_evaluation',
            'orchestration_control': 'O_orchestration_control',
            'search_retrieval': 'R_search_retrieval',
            'aggregation_reporting': 'G_aggregation_reporting',
            'integration_storage': 'T_integration_storage',
            'ingestion_preparation': 'I_ingestion_preparation'
        }
        return Path(f"canonical_flow/{phase_mapping[self.stage]}")


class CanonicalScaffolder:
    """Main scaffolding class for canonical modules."""
    
    def __init__(self, project_root: Path = None):
        """Initialize the scaffolder."""
        self.project_root = project_root or Path.cwd()
        self.canonical_flow_dir = self.project_root / "canonical_flow"
        self.index_file = self.canonical_flow_dir / "index.json"
        self.tools_dir = self.project_root / "tools"
        
        # Initialize Jinja2 environment
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(self.tools_dir / "templates")),
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        # Ensure templates directory exists
        (self.tools_dir / "templates").mkdir(exist_ok=True)
        self._create_templates()
        
        self.existing_modules: List[Dict] = []
        self.missing_modules: List[ModuleSpec] = []
        
    def _create_templates(self):
        """Create Jinja2 templates for module generation."""
        
        # Module template
        module_template = '''"""
{{ description }}

Canonical Module: {{ code }}
Phase: {{ phase }}
Stage: {{ stage }}
Order: {{ stage_order }}

This module implements standard process(data, context) -> Dict[str, Any] signature
with OpenTelemetry tracing and invariant validation.
"""

from typing import Any, Dict, Optional
import logging

try:
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode
    TELEMETRY_AVAILABLE = True
except ImportError:
    TELEMETRY_AVAILABLE = False
    # Mock tracer for environments without OpenTelemetry
    class MockTracer:
        def start_as_current_span(self, name):
            return self
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def set_status(self, status):
            pass
        def add_event(self, name, attributes=None):
            pass
    
    class MockTrace:
        def get_tracer(self, name):
            return MockTracer()
    
    trace = MockTrace()

# Module metadata
__phase__ = "{{ phase }}"
__code__ = "{{ code }}"
__stage_order__ = {{ stage_order }}

# Logger and tracer setup
logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


def _validate_input_invariants(data: Any, context: Optional[Dict[str, Any]] = None) -> None:
    """
    Validate input invariants for the {{ code }} module.
    
    Args:
        data: Input data to validate
        context: Optional context dictionary
        
    Raises:
        TypeError: If input types are invalid
        ValueError: If input values are invalid
    """
    if data is None:
        raise ValueError(f"{{ code }}: Data cannot be None")
    
    if context is not None and not isinstance(context, dict):
        raise TypeError(f"{{ code }}: Context must be a dictionary or None")
    
    # Add module-specific invariants here
    logger.debug(f"{{ code }}: Input validation passed")


def _validate_output_invariants(result: Dict[str, Any]) -> None:
    """
    Validate output invariants for the {{ code }} module.
    
    Args:
        result: Output result to validate
        
    Raises:
        TypeError: If output type is invalid
        ValueError: If output structure is invalid
    """
    if not isinstance(result, dict):
        raise TypeError(f"{{ code }}: Result must be a dictionary")
    
    if "status" not in result:
        raise ValueError(f"{{ code }}: Result must contain 'status' field")
    
    logger.debug(f"{{ code }}: Output validation passed")


@tracer.start_as_current_span("{{ code.lower() }}_process")
def process(data: Any, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Process data through the {{ code }} canonical module.
    
    This is the standard entry point for all canonical modules, implementing
    the process(data, context) -> Dict[str, Any] signature.
    
    Args:
        data: Input data to process
        context: Optional processing context
        
    Returns:
        Dict[str, Any]: Processing results with status and output data
        
    Raises:
        TypeError: If input types are invalid
        ValueError: If input values are invalid
    """
    span = trace.get_current_span() if TELEMETRY_AVAILABLE else None
    
    try:
        # Input validation
        _validate_input_invariants(data, context)
        
        if span:
            span.add_event("{{ code }}: Input validation completed")
        
        logger.info(f"{{ code }}: Processing started")
        
        # TODO: Implement actual processing logic
        # This is a placeholder implementation
        result = {
            "status": "success",
            "module": "{{ code }}",
            "phase": "{{ phase }}",
            "stage": "{{ stage }}",
            "stage_order": {{ stage_order }},
            "processed_data": data,
            "context": context,
            "metadata": {
                "processing_complete": True,
                "validation_passed": True
            }
        }
        
        # Output validation
        _validate_output_invariants(result)
        
        if span:
            span.set_status(Status(StatusCode.OK))
            span.add_event("{{ code }}: Processing completed successfully")
        
        logger.info(f"{{ code }}: Processing completed successfully")
        return result
        
    except Exception as e:
        if span:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.add_event(f"{{ code }}: Error occurred", {"error": str(e)})
        
        logger.error(f"{{ code }}: Processing failed: {str(e)}")
        
        return {
            "status": "error",
            "module": "{{ code }}",
            "phase": "{{ phase }}",
            "stage": "{{ stage }}",
            "error": str(e),
            "metadata": {
                "processing_complete": False,
                "validation_passed": False
            }
        }


# Export main function for module interface
__all__ = ["process", "__phase__", "__code__", "__stage_order__"]
'''

        # Test template
        test_template = '''"""
Property-based tests for {{ code }} canonical module using Hypothesis.

Tests deterministic behavior, invariant validation, and metamorphic properties.
"""

import json
import logging
from typing import Any, Dict, List, Optional

try:
    import pytest
    from hypothesis import given, settings, strategies as st
    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False
    # Mock decorators for environments without Hypothesis
    def given(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    def settings(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    class st:
        @staticmethod
        def dictionaries(keys, values):
            return lambda: {"test_key": "test_value"}
        
        @staticmethod
        def text():
            return lambda: "test_string"
        
        @staticmethod
        def integers():
            return lambda: 42
        
        @staticmethod
        def one_of(*args):
            return lambda: args[0]() if args else None

# Import the module under test
try:
    from canonical_flow.{{ directory_name }}.{{ code.lower() }}_{{ stage }} import process
    MODULE_AVAILABLE = True
except ImportError:
    MODULE_AVAILABLE = False
    def process(data, context=None):
        return {"status": "error", "error": "Module not available"}

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test configuration
NUMERICAL_TOLERANCE = 1e-6
DEFAULT_TRIALS = 50 if HYPOTHESIS_AVAILABLE else 1


class Test{{ code }}Properties:
    """Property-based tests for {{ code }} module."""
    
    @pytest.mark.skipif(not MODULE_AVAILABLE, reason="Module not available")
    @given(
        data=st.one_of(
            st.text(),
            st.integers(),
            st.dictionaries(st.text(), st.text())
        )
    )
    @settings(max_examples=DEFAULT_TRIALS)
    def test_deterministic_output(self, data):
        """Test that identical inputs produce identical outputs."""
        if not HYPOTHESIS_AVAILABLE:
            data = "test_data"
        
        # Run the same input twice
        result1 = process(data)
        result2 = process(data)
        
        # Results should be identical
        assert result1 == result2, f"Non-deterministic behavior detected for input: {data}"
    
    @pytest.mark.skipif(not MODULE_AVAILABLE, reason="Module not available")
    @given(
        data=st.one_of(
            st.text(),
            st.integers(),
            st.dictionaries(st.text(), st.text())
        ),
        context=st.one_of(
            st.none(),
            st.dictionaries(st.text(), st.text())
        )
    )
    @settings(max_examples=DEFAULT_TRIALS)
    def test_output_structure_invariant(self, data, context):
        """Test that output always has required structure."""
        if not HYPOTHESIS_AVAILABLE:
            data, context = "test_data", {"test": "context"}
        
        result = process(data, context)
        
        # Check required fields
        assert isinstance(result, dict), "Output must be a dictionary"
        assert "status" in result, "Output must contain 'status' field"
        assert "module" in result, "Output must contain 'module' field"
        
        # Check status is valid
        assert result["status"] in ["success", "error"], f"Invalid status: {result['status']}"
    
    @pytest.mark.skipif(not MODULE_AVAILABLE, reason="Module not available")
    def test_input_validation_none_data(self):
        """Test that None data raises appropriate error."""
        result = process(None)
        
        assert result["status"] == "error"
        assert "error" in result
        assert "cannot be None" in result["error"] or "Data" in result["error"]
    
    @pytest.mark.skipif(not MODULE_AVAILABLE, reason="Module not available")
    def test_input_validation_invalid_context(self):
        """Test that invalid context type is handled appropriately."""
        result = process("test_data", context="invalid_context")
        
        # Should either succeed or return error status
        assert isinstance(result, dict)
        assert "status" in result
    
    @pytest.mark.skipif(not MODULE_AVAILABLE, reason="Module not available")
    @given(data=st.text())
    @settings(max_examples=DEFAULT_TRIALS)
    def test_metadata_consistency(self, data):
        """Test that metadata fields are consistent."""
        if not HYPOTHESIS_AVAILABLE:
            data = "test_data"
        
        result = process(data)
        
        if "metadata" in result:
            metadata = result["metadata"]
            assert isinstance(metadata, dict), "Metadata must be a dictionary"
            
            if "processing_complete" in metadata:
                assert isinstance(metadata["processing_complete"], bool)
            
            if "validation_passed" in metadata:
                assert isinstance(metadata["validation_passed"], bool)


class Test{{ code }}Integration:
    """Integration tests for {{ code }} module."""
    
    @pytest.mark.skipif(not MODULE_AVAILABLE, reason="Module not available")
    def test_basic_functionality(self):
        """Test basic module functionality with simple inputs."""
        test_data = "test_input"
        test_context = {"source": "test"}
        
        result = process(test_data, test_context)
        
        assert isinstance(result, dict)
        assert result.get("module") == "{{ code }}"
        assert result.get("phase") == "{{ phase }}"
        assert result.get("stage") == "{{ stage }}"
    
    @pytest.mark.skipif(not MODULE_AVAILABLE, reason="Module not available")
    def test_empty_context(self):
        """Test processing with empty context."""
        result = process("test_data", {})
        
        assert isinstance(result, dict)
        assert "status" in result
    
    @pytest.mark.skipif(not MODULE_AVAILABLE, reason="Module not available")
    def test_complex_data_structure(self):
        """Test processing with complex data structures."""
        complex_data = {
            "nested": {
                "list": [1, 2, 3],
                "dict": {"key": "value"}
            },
            "array": ["a", "b", "c"]
        }
        
        result = process(complex_data)
        
        assert isinstance(result, dict)
        assert "status" in result


if __name__ == "__main__":
    # Run tests if executed directly
    if HYPOTHESIS_AVAILABLE:
        pytest.main([__file__, "-v"])
    else:
        print("Hypothesis not available - running basic tests")
        test_instance = Test{{ code }}Integration()
        test_instance.test_basic_functionality()
        print("Basic tests completed")
'''

        # Save templates
        templates_dir = self.tools_dir / "templates"
        with open(templates_dir / "canonical_module.py.jinja2", "w") as f:
            f.write(module_template)
        
        with open(templates_dir / "canonical_test.py.jinja2", "w") as f:
            f.write(test_template)
    
    def load_existing_modules(self) -> None:
        """Load existing modules from index.json."""
        try:
            with open(self.index_file, 'r') as f:
                self.existing_modules = json.load(f)
            logger.info(f"Loaded {len(self.existing_modules)} existing modules")
        except FileNotFoundError:
            logger.warning("index.json not found, starting with empty module list")
            self.existing_modules = []
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing index.json: {e}")
            self.existing_modules = []
    
    def identify_missing_modules(self) -> None:
        """Identify missing modules in the specified ranges."""
        existing_codes = {module['code'] for module in self.existing_modules}
        
        # Define ranges and their corresponding stages
        ranges = {
            # Context Construction: 08X-12X
            ('08X', '09X', '10X', '11X', '12X'): 'context_construction',
            # Synthesis Output: 52S-57S
            ('52S', '53S', '54S', '55S', '56S', '57S'): 'synthesis_output'
        }
        
        for codes, stage in ranges.items():
            for i, code in enumerate(codes):
                if code not in existing_codes:
                    # Generate description based on code pattern
                    descriptions = {
                        '08X': 'Context transformation and preparation module',
                        '09X': 'Context validation and normalization module', 
                        '10X': 'Context enrichment and augmentation module',
                        '11X': 'Context integration and merging module',
                        '12X': 'Context finalization and optimization module',
                        '52S': 'Pre-synthesis preparation module',
                        '53S': 'Core synthesis processing module',
                        '54S': 'Synthesis refinement and enhancement module',
                        '55S': 'Synthesis validation and verification module',
                        '56S': 'Synthesis formatting and structuring module',
                        '57S': 'Final synthesis output preparation module'
                    }
                    
                    # Determine dependencies based on sequence
                    dependencies = []
                    if i > 0:
                        dependencies.append(codes[i-1])
                    
                    # Add cross-stage dependencies
                    if stage == 'synthesis_output':
                        dependencies.extend(['24O', '25O'])  # Orchestration dependencies
                    elif stage == 'context_construction':
                        dependencies.extend(['04I', '03I'])  # Ingestion dependencies
                    
                    module_spec = ModuleSpec(
                        code=code,
                        phase=self._get_phase_from_code(code),
                        stage=stage,
                        stage_order=self._get_stage_order(code),
                        dependencies=dependencies,
                        description=descriptions.get(code, f"Canonical module {code}")
                    )
                    
                    self.missing_modules.append(module_spec)
        
        logger.info(f"Identified {len(self.missing_modules)} missing modules")
    
    def _get_phase_from_code(self, code: str) -> str:
        """Extract phase from module code."""
        suffix = code[-1]
        phase_mapping = {
            'I': 'ingestion',
            'X': 'context', 
            'K': 'knowledge',
            'A': 'analysis',
            'L': 'classification',
            'O': 'orchestration',
            'R': 'retrieval',
            'S': 'synthesis',
            'G': 'aggregation',
            'T': 'integration'
        }
        return phase_mapping.get(suffix, 'unknown')
    
    def _get_stage_order(self, code: str) -> int:
        """Extract stage order from module code."""
        match = re.match(r'(\d+)', code)
        return int(match.group(1)) if match else 0
    
    def generate_module_file(self, module_spec: ModuleSpec, dry_run: bool = False) -> bool:
        """Generate a single module file."""
        try:
            template = self.jinja_env.get_template("canonical_module.py.jinja2")
            
            # Get directory mapping for template
            directory_mapping = {
                'context_construction': 'X_context_construction',
                'synthesis_output': 'S_synthesis_output'
            }
            
            content = template.render(
                code=module_spec.code,
                phase=module_spec.phase,
                stage=module_spec.stage,
                stage_order=module_spec.stage_order,
                description=module_spec.description,
                directory_name=directory_mapping.get(module_spec.stage, module_spec.stage)
            )
            
            if dry_run:
                logger.info(f"[DRY RUN] Would create: {module_spec.directory_path / module_spec.file_name}")
                return True
            
            # Ensure directory exists
            module_spec.directory_path.mkdir(parents=True, exist_ok=True)
            
            # Write module file
            module_file = module_spec.directory_path / module_spec.file_name
            with open(module_file, 'w') as f:
                f.write(content)
            
            logger.info(f"Generated module: {module_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to generate module {module_spec.code}: {e}")
            return False
    
    def generate_test_file(self, module_spec: ModuleSpec, dry_run: bool = False) -> bool:
        """Generate a test file for the module."""
        try:
            template = self.jinja_env.get_template("canonical_test.py.jinja2")
            
            directory_mapping = {
                'context_construction': 'X_context_construction',
                'synthesis_output': 'S_synthesis_output'
            }
            
            content = template.render(
                code=module_spec.code,
                phase=module_spec.phase,
                stage=module_spec.stage,
                stage_order=module_spec.stage_order,
                description=module_spec.description,
                directory_name=directory_mapping.get(module_spec.stage, module_spec.stage)
            )
            
            if dry_run:
                logger.info(f"[DRY RUN] Would create: tests/{module_spec.test_file_name}")
                return True
            
            # Ensure tests directory exists
            tests_dir = self.project_root / "tests"
            tests_dir.mkdir(parents=True, exist_ok=True)
            
            # Write test file
            test_file = tests_dir / module_spec.test_file_name
            with open(test_file, 'w') as f:
                f.write(content)
            
            logger.info(f"Generated test file: {test_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to generate test for {module_spec.code}: {e}")
            return False
    
    def update_index(self, dry_run: bool = False) -> bool:
        """Update index.json with newly scaffolded modules."""
        try:
            # Add new modules to existing ones
            for module_spec in self.missing_modules:
                new_entry = {
                    "code": module_spec.code,
                    "stage": module_spec.stage,
                    "alias_path": str(module_spec.directory_path / module_spec.file_name),
                    "original_path": module_spec.file_name,
                    "dependencies": module_spec.dependencies,
                    "scaffolded": True,
                    "description": module_spec.description
                }
                self.existing_modules.append(new_entry)
            
            # Sort by code for consistency
            self.existing_modules.sort(key=lambda x: (int(re.match(r'(\d+)', x['code']).group(1)), x['code']))
            
            if dry_run:
                logger.info(f"[DRY RUN] Would update index.json with {len(self.missing_modules)} new entries")
                return True
            
            # Write updated index
            with open(self.index_file, 'w') as f:
                json.dump(self.existing_modules, f, indent=2)
            
            logger.info(f"Updated index.json with {len(self.missing_modules)} new modules")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update index.json: {e}")
            return False
    
    def scaffold_missing_modules(self, dry_run: bool = False) -> bool:
        """Main scaffolding process."""
        logger.info("Starting canonical module scaffolding...")
        
        # Load existing modules
        self.load_existing_modules()
        
        # Identify missing modules
        self.identify_missing_modules()
        
        if not self.missing_modules:
            logger.info("No missing modules found in specified ranges")
            return True
        
        # Generate modules and tests
        success_count = 0
        
        for module_spec in self.missing_modules:
            logger.info(f"Scaffolding module: {module_spec.code}")
            
            # Generate module file
            if self.generate_module_file(module_spec, dry_run):
                # Generate test file
                if self.generate_test_file(module_spec, dry_run):
                    success_count += 1
                else:
                    logger.warning(f"Module {module_spec.code} generated but test failed")
                    success_count += 1  # Still count as success
            else:
                logger.error(f"Failed to generate module {module_spec.code}")
        
        # Update index
        if success_count > 0:
            if self.update_index(dry_run):
                logger.info(f"Successfully scaffolded {success_count} modules")
            else:
                logger.error("Failed to update index.json")
                return False
        
        return success_count > 0


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate missing canonical modules")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without making changes")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create scaffolder and run
    scaffolder = CanonicalScaffolder()
    success = scaffolder.scaffold_missing_modules(dry_run=args.dry_run)
    
    if success:
        logger.info("Scaffolding completed successfully")
        return 0
    else:
        logger.error("Scaffolding failed")
        return 1


if __name__ == "__main__":
    exit(main())
#!/usr/bin/env python3
"""
Canonical Module Scaffolding Generator

This tool scaffolds missing canonical modules in the 08X-12X context construction
and 52S-57S synthesis output ranges by building template-based generators.

Each generated module includes:
- Standardized headers with phase annotations (__phase__, __code__, __stage_order__)
- process(data, context) -> Dict[str, Any] interface signatures
- OpenTelemetry span initialization hooks
- Basic property test stubs using Hypothesis
- DAG compliance infrastructure

Usage:
    python tools/scaffold_canonical.py --ranges 08X-12X,52S-57S
    python tools/scaffold_canonical.py --dry-run --ranges 08X-12X
    python tools/scaffold_canonical.py --phase context_construction
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


@dataclass
class ModuleSpec:
    """Specification for a canonical module to be generated."""
    code: str
    phase: str
    stage_order: int
    directory: str
    filename: str
    class_name: Optional[str] = None
    description: Optional[str] = None


@dataclass
class ScaffoldConfig:
    """Configuration for the scaffolding generator."""
    target_ranges: List[str] = field(default_factory=list)
    target_phases: List[str] = field(default_factory=list)
    dry_run: bool = False
    force: bool = False
    output_dir: Path = field(default_factory=lambda: Path("canonical_flow"))


class CanonicalScaffolder:
    """Main scaffolding generator for canonical modules."""
    
    def __init__(self, config: ScaffoldConfig):
        self.config = config
        self.project_root = project_root
        self.index_path = self.project_root / "canonical_flow" / "index.json"
        self.existing_modules: Dict[str, Dict[str, Any]] = {}
        self.phase_mapping = {
            'I': ('ingestion_preparation', 'I_ingestion_preparation'),
            'X': ('context_construction', 'X_context_construction'),
            'K': ('knowledge_extraction', 'K_knowledge_extraction'),
            'A': ('analysis_nlp', 'A_analysis_nlp'),
            'L': ('classification_evaluation', 'L_classification_evaluation'),
            'O': ('orchestration_control', 'O_orchestration_control'),
            'R': ('search_retrieval', 'R_search_retrieval'),
            'S': ('synthesis_output', 'S_synthesis_output'),
            'G': ('aggregation_reporting', 'G_aggregation_reporting'),
            'T': ('integration_storage', 'T_integration_storage'),
        }
        self.load_existing_modules()
    
    def load_existing_modules(self) -> None:
        """Load existing modules from index.json."""
        if self.index_path.exists():
            with open(self.index_path, 'r') as f:
                modules = json.load(f)
                for module in modules:
                    self.existing_modules[module['code']] = module
    
    def parse_ranges(self, ranges: List[str]) -> Set[str]:
        """Parse range specifications like '08X-12X' into individual codes."""
        codes = set()
        
        for range_spec in ranges:
            if '-' in range_spec:
                start, end = range_spec.split('-', 1)
                start_num = int(start[:-1])
                start_letter = start[-1]
                end_num = int(end[:-1])
                end_letter = end[-1]
                
                if start_letter != end_letter:
                    raise ValueError(f"Range {range_spec} spans different phases")
                
                for num in range(start_num, end_num + 1):
                    codes.add(f"{num:02d}{start_letter}")
            else:
                codes.add(range_spec)
        
        return codes
    
    def identify_gaps(self) -> List[ModuleSpec]:
        """Identify missing modules in the specified ranges/phases."""
        target_codes = set()
        
        # Process ranges
        if self.config.target_ranges:
            target_codes.update(self.parse_ranges(self.config.target_ranges))
        
        # Process phases
        if self.config.target_phases:
            for phase in self.config.target_phases:
                phase_letter = None
                for letter, (phase_name, _) in self.phase_mapping.items():
                    if phase_name == phase:
                        phase_letter = letter
                        break
                
                if phase_letter:
                    # Generate typical range for phase (adjust as needed)
                    if phase == 'context_construction':
                        target_codes.update([f"{i:02d}X" for i in range(8, 13)])
                    elif phase == 'synthesis_output':
                        target_codes.update([f"{i:02d}S" for i in range(52, 58)])
        
        # Find gaps
        missing_specs = []
        for code in sorted(target_codes):
            if code not in self.existing_modules:
                phase_letter = code[-1]
                if phase_letter in self.phase_mapping:
                    phase_name, phase_dir = self.phase_mapping[phase_letter]
                    stage_order = int(code[:-1])
                    
                    spec = ModuleSpec(
                        code=code,
                        phase=phase_name,
                        stage_order=stage_order,
                        directory=phase_dir,
                        filename=f"{code.lower()}_{phase_name}_component.py",
                        class_name=self._generate_class_name(code, phase_name),
                        description=f"Component {code} for {phase_name} phase"
                    )
                    missing_specs.append(spec)
        
        return missing_specs
    
    def _generate_class_name(self, code: str, phase: str) -> str:
        """Generate a class name for the module."""
        phase_words = phase.replace('_', ' ').title().replace(' ', '')
        # Handle numeric prefixes in code by adding underscore
        clean_code = f"Component{code}" if code[0].isdigit() else f"{code}Component"
        return f"{clean_code}_{phase_words}"
    
    def generate_module(self, spec: ModuleSpec) -> str:
        """Generate the module content from template."""
        template = self._get_module_template()
        
        # Replace placeholders
        content = template.format(
            code=spec.code,
            code_lower=spec.code.lower(),
            phase=spec.phase,
            stage_order=spec.stage_order,
            class_name=spec.class_name or f"{spec.code}Component",
            description=spec.description or f"Component {spec.code}",
            filename=spec.filename,
            directory=spec.directory
        )
        
        return content
    
    def generate_test_module(self, spec: ModuleSpec) -> str:
        """Generate test module content with Hypothesis property tests."""
        template = self._get_test_template()
        
        content = template.format(
            code=spec.code,
            phase=spec.phase,
            stage_order=spec.stage_order,
            class_name=spec.class_name or f"{spec.code}Component",
            filename=spec.filename,
            directory=spec.directory,
            test_class_name=f"Test{spec.class_name or spec.code}Component"
        )
        
        return content
    
    def _get_module_template(self) -> str:
        """Get the module template with all required infrastructure."""
        return '''"""
Canonical Module: {code} - {description}

Component for {phase} phase in the canonical pipeline.
Generated by scaffold_canonical.py - customize as needed.

Architecture:
- Phase: {phase}
- Stage Order: {stage_order}
- Component Code: {code}
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

# OpenTelemetry imports with graceful fallback
try:
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode
    TELEMETRY_AVAILABLE = True
except ImportError:
    TELEMETRY_AVAILABLE = False
    # Mock classes for fallback
    class MockSpan:
        def set_attribute(self, key: str, value: Any) -> None: pass
        def set_status(self, status: Any, description: str = "") -> None: pass
        def record_exception(self, exception: Exception) -> None: pass
        def __enter__(self): return self
        def __exit__(self, exc_type, exc_val, exc_tb): pass
    
    class MockTracer:
        def start_as_current_span(self, name: str) -> MockSpan: return MockSpan()
    
    trace = type('MockTrace', (), {{'get_tracer': lambda name: MockTracer()}})()
    Status = type('MockStatus', (), {{}})()
    StatusCode = type('MockStatusCode', (), {{'OK': 'OK', 'ERROR': 'ERROR'}})()

# Module metadata for DAG compliance
__phase__ = "{phase}"
__code__ = "{code}"
__stage_order__ = {stage_order}

# Configure logging
logger = logging.getLogger(__name__)


class {class_name}:
    """
    Canonical component {code} for {phase} phase.
    
    Implements standardized process(data, context) interface
    with OpenTelemetry observability and DAG compliance.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {{}}
        self.tracer = trace.get_tracer(__name__) if TELEMETRY_AVAILABLE else trace.get_tracer(__name__)
        
        # Component-specific initialization
        self._initialize_component()
    
    def _initialize_component(self) -> None:
        """Initialize component-specific resources."""
        # TODO: Add component-specific initialization logic
        logger.info(f"Initialized component {{code}}")
    
    def process(self, data: Any = None, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Main processing interface for component {code}.
        
        Args:
            data: Input data to process
            context: Processing context with metadata
            
        Returns:
            Dictionary containing processing results with required fields:
            - status: "success" | "error" | "warning"
            - data: Processed data
            - metadata: Processing metadata
            - component_info: Component identification
        """
        with self.tracer.start_as_current_span(f"component-{code_lower}-process") as span:
            try:
                # Set span attributes for observability
                span.set_attribute("component.code", "{code}")
                span.set_attribute("component.phase", "{phase}")
                span.set_attribute("component.stage_order", {stage_order})
                
                if context:
                    span.set_attribute("context.keys", list(context.keys()))
                
                start_time = time.time()
                
                # Validate inputs
                validation_result = self._validate_inputs(data, context)
                if not validation_result["valid"]:
                    raise ValueError(f"Input validation failed: {{validation_result['errors']}}")
                
                # Core processing logic
                result = self._process_core(data, context)
                
                processing_time = time.time() - start_time
                span.set_attribute("processing.duration_seconds", processing_time)
                
                # Build standardized response
                response = {{
                    "status": "success",
                    "data": result,
                    "metadata": {{
                        "processing_time_seconds": processing_time,
                        "timestamp": time.time(),
                        "input_size": len(str(data)) if data else 0,
                    }},
                    "component_info": {{
                        "code": "{code}",
                        "phase": "{phase}",
                        "stage_order": {stage_order},
                        "class": self.__class__.__name__,
                    }},
                    "invariants": self._check_invariants(result, context),
                }}
                
                span.set_status(Status(StatusCode.OK))
                logger.info(f"Component {{code}} processed successfully in {{processing_time:.3f}}s")
                
                return response
                
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR), str(e))
                logger.error(f"Component {{code}} processing failed: {{e}}")
                
                return {{
                    "status": "error",
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "component_info": {{
                        "code": "{code}",
                        "phase": "{phase}",
                        "stage_order": {stage_order},
                        "class": self.__class__.__name__,
                    }},
                    "metadata": {{
                        "timestamp": time.time(),
                        "error_context": str(context) if context else None,
                    }}
                }}
    
    def _validate_inputs(self, data: Any, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate input parameters."""
        errors = []
        
        # TODO: Add component-specific validation logic
        # Example validations:
        # if data is None:
        #     errors.append("Data parameter is required")
        # if context and not isinstance(context, dict):
        #     errors.append("Context must be a dictionary")
        
        return {{
            "valid": len(errors) == 0,
            "errors": errors
        }}
    
    def _process_core(self, data: Any, context: Optional[Dict[str, Any]]) -> Any:
        """Core processing logic - implement component-specific functionality here."""
        # TODO: Implement actual processing logic
        logger.info(f"Core processing for component {{code}}")
        
        # Placeholder processing
        processed_data = {{
            "input_data": data,
            "processed_by": "{code}",
            "phase": "{phase}",
            "processing_note": "This is a scaffolded component - implement actual logic"
        }}
        
        return processed_data
    
    def _check_invariants(self, result: Any, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Check component invariants for DAG compliance."""
        invariants = {{}}
        
        # Standard invariants for all components
        invariants["result_not_none"] = result is not None
        invariants["result_serializable"] = self._is_serializable(result)
        
        # TODO: Add component-specific invariants
        # Example:
        # invariants["output_has_required_fields"] = all(
        #     field in result for field in ["field1", "field2"]
        # )
        
        return invariants
    
    def _is_serializable(self, obj: Any) -> bool:
        """Check if object is JSON serializable."""
        try:
            import json
            json.dumps(obj, default=str)
            return True
        except (TypeError, ValueError):
            return False
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get component health status."""
        return {{
            "component": "{code}",
            "phase": "{phase}",
            "status": "healthy",
            "initialized": True,
            "telemetry_enabled": TELEMETRY_AVAILABLE,
        }}


# Standalone function interface for compatibility
def process(data: Any = None, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Standalone process function for component {code}.
    
    Creates a component instance and processes the data.
    """
    component = {class_name}()
    return component.process(data, context)


if __name__ == "__main__":
    # Demo usage
    print(f"Component {{code}} - {{'{phase}'.replace('_', ' ').title()}}")
    
    # Test with sample data
    sample_data = {{"test": "data", "component": "{code}"}}
    sample_context = {{"source": "demo", "debug": True}}
    
    result = process(sample_data, sample_context)
    print(f"Sample processing result: {{result}}")
'''
    
    def _get_test_template(self) -> str:
        """Get the test template with Hypothesis property tests."""
        return '''"""
Property Tests for Component {code}

Generated by scaffold_canonical.py
Uses Hypothesis for property-based testing.
"""

import json
import pytest
from typing import Any, Dict, Optional

# Hypothesis imports with graceful fallback
try:
    from hypothesis import given, strategies as st, assume, settings
    from hypothesis.strategies import composite
    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False
    # Mock decorators for fallback
    def given(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    class MockStrategies:
        def text(self, **kwargs): return lambda: "test_string"
        def dictionaries(self, **kwargs): return lambda: {{"test": "dict"}}
        def integers(self, **kwargs): return lambda: 42
        def booleans(self): return lambda: True
        def none(self): return lambda: None
        def one_of(self, *args): return lambda: args[0]() if args else lambda: None
    
    st = MockStrategies()
    assume = lambda x: True
    settings = lambda **kwargs: lambda func: func
    composite = lambda func: func

# Import component under test
try:
    from canonical_flow.{directory}.{filename} import {class_name}, process
except ImportError:
    # Fallback for testing during development
    import sys
    from pathlib import Path
    
    # Add canonical_flow to path
    canonical_path = Path(__file__).parent.parent / "canonical_flow" / "{directory}"
    sys.path.insert(0, str(canonical_path))
    
    from {filename} import {class_name}, process


class {test_class_name}:
    """Property-based tests for component {code}."""
    
    def test_component_initialization(self):
        """Test component can be initialized."""
        component = {class_name}()
        assert component is not None
        assert hasattr(component, 'process')
        
        # Test health status
        health = component.get_health_status()
        assert health["component"] == "{code}"
        assert health["phase"] == "{phase}"
    
    @pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="Hypothesis not available")
    @given(
        data=st.one_of(
            st.none(),
            st.text(),
            st.dictionaries(st.text(), st.text()),
            st.integers(),
        ),
        context=st.one_of(
            st.none(),
            st.dictionaries(st.text(), st.text()),
        )
    )
    def test_process_always_returns_dict(self, data: Any, context: Optional[Dict[str, Any]]):
        """Property: process() always returns a dictionary."""
        component = {class_name}()
        result = component.process(data, context)
        
        assert isinstance(result, dict)
        assert "status" in result
        assert "component_info" in result
        assert result["component_info"]["code"] == "{code}"
    
    @pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="Hypothesis not available")
    @given(context_dict=st.dictionaries(st.text(), st.text()))
    def test_process_with_valid_context(self, context_dict: Dict[str, str]):
        """Property: process() handles valid context dictionaries."""
        component = {class_name}()
        result = component.process(data="test", context=context_dict)
        
        assert isinstance(result, dict)
        assert result["status"] in ["success", "error", "warning"]
    
    @pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="Hypothesis not available")
    @given(data=st.text(min_size=1, max_size=1000))
    def test_process_with_text_data(self, data: str):
        """Property: process() handles text data consistently."""
        component = {class_name}()
        result = component.process(data)
        
        assert isinstance(result, dict)
        assert "metadata" in result
        assert "processing_time_seconds" in result["metadata"]
        assert result["metadata"]["processing_time_seconds"] >= 0
    
    def test_process_idempotency(self):
        """Test that processing the same data multiple times is consistent."""
        component = {class_name}()
        test_data = {{"test": "idempotency"}}
        
        result1 = component.process(test_data)
        result2 = component.process(test_data)
        
        # Results should be structurally similar (excluding timestamps)
        assert result1["status"] == result2["status"]
        assert result1["component_info"] == result2["component_info"]
    
    def test_process_error_handling(self):
        """Test error handling in process method."""
        component = {class_name}()
        
        # Test with potentially problematic input
        # (Exact behavior depends on component implementation)
        result = component.process(None, None)
        
        assert isinstance(result, dict)
        assert "status" in result
        assert result["component_info"]["code"] == "{code}"
    
    def test_invariants_checking(self):
        """Test that invariants are checked properly."""
        component = {class_name}()
        result = component.process("test_data")
        
        if "invariants" in result:
            invariants = result["invariants"]
            assert isinstance(invariants, dict)
            
            # Standard invariants should be present
            if "result_not_none" in invariants:
                assert isinstance(invariants["result_not_none"], bool)
            if "result_serializable" in invariants:
                assert isinstance(invariants["result_serializable"], bool)
    
    def test_standalone_process_function(self):
        """Test the standalone process function."""
        result = process("test_data")
        
        assert isinstance(result, dict)
        assert result["component_info"]["code"] == "{code}"
    
    @pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="Hypothesis not available")
    @given(
        data=st.dictionaries(
            st.text(min_size=1, max_size=50),
            st.one_of(st.text(), st.integers(), st.booleans())
        )
    )
    @settings(max_examples=50, deadline=5000)  # Limit examples for CI
    def test_serialization_property(self, data: Dict[str, Any]):
        """Property: Output should be JSON serializable."""
        assume(len(str(data)) < 10000)  # Avoid huge test cases
        
        component = {class_name}()
        result = component.process(data)
        
        # Should be JSON serializable
        try:
            json_str = json.dumps(result, default=str)
            assert len(json_str) > 0
        except (TypeError, ValueError) as ex:
            pytest.fail(f"Result not serializable: {{ex}}")
    
    def test_component_metadata(self):
        """Test component metadata is correctly set."""
        component = {class_name}()
        
        # Check class attributes
        assert hasattr(component, '__class__')
        
        # Check module-level metadata
        import canonical_flow.{directory}.{filename} as component_module
        assert hasattr(component_module, '__phase__')
        assert hasattr(component_module, '__code__')
        assert hasattr(component_module, '__stage_order__')
        
        assert component_module.__phase__ == "{phase}"
        assert component_module.__code__ == "{code}"
        assert component_module.__stage_order__ == {stage_order}


if __name__ == "__main__":
    # Run basic tests without pytest
    test_instance = {test_class_name}()
    
    print(f"Running basic tests for component {{code}}...")
    
    try:
        test_instance.test_component_initialization()
        print("✓ Component initialization test passed")
        
        test_instance.test_process_idempotency()
        print("✓ Process idempotency test passed")
        
        test_instance.test_standalone_process_function()
        print("✓ Standalone process function test passed")
        
        test_instance.test_component_metadata()
        print("✓ Component metadata test passed")
        
        print(f"Basic tests for component {{code}} completed successfully!")
        
    except Exception as ex:
        print(f"✗ Test failed: {{ex}}")
        raise
'''
    
    def scaffold_modules(self, specs: List[ModuleSpec]) -> None:
        """Generate and write modules based on specifications."""
        for spec in specs:
            if self.config.dry_run:
                print(f"[DRY RUN] Would create module: {spec.code} -> {spec.directory}/{spec.filename}")
                continue
            
            # Create directory structure
            module_dir = self.config.output_dir / spec.directory
            module_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate module content
            module_path = module_dir / spec.filename
            test_path = module_dir / f"test_{spec.filename}"
            
            # Check if files exist
            if module_path.exists() and not self.config.force:
                print(f"Skipping {spec.code}: {module_path} already exists (use --force to overwrite)")
                continue
            
            # Write module
            module_content = self.generate_module(spec)
            with open(module_path, 'w') as f:
                f.write(module_content)
            
            # Write test module
            test_content = self.generate_test_module(spec)
            with open(test_path, 'w') as f:
                f.write(test_content)
            
            print(f"✓ Generated {spec.code}: {module_path}")
            print(f"✓ Generated test: {test_path}")
    
    def update_index(self, specs: List[ModuleSpec]) -> None:
        """Update index.json with new modules."""
        if self.config.dry_run:
            print("[DRY RUN] Would update index.json")
            return
        
        # Load existing index
        existing_modules = []
        if self.index_path.exists():
            with open(self.index_path, 'r') as f:
                existing_modules = json.load(f)
        
        # Add new modules
        for spec in specs:
            entry = {
                "code": spec.code,
                "stage": spec.phase,
                "alias_path": f"canonical_flow/{spec.directory}/{spec.filename}",
                "original_path": f"{spec.filename}",
                "generated": True,
                "scaffolded_by": "scaffold_canonical.py"
            }
            existing_modules.append(entry)
        
        # Sort by code
        existing_modules.sort(key=lambda x: (int(x['code'][:-1]), x['code'][-1]))
        
        # Write updated index
        with open(self.index_path, 'w') as f:
            json.dump(existing_modules, f, indent=2)
        
        print(f"✓ Updated {self.index_path}")
    
    def run(self) -> None:
        """Run the scaffolding process."""
        print("Canonical Module Scaffolder")
        print("=" * 40)
        
        # Identify gaps
        missing_specs = self.identify_gaps()
        
        if not missing_specs:
            print("No missing modules found in specified ranges/phases.")
            return
        
        print(f"Found {len(missing_specs)} missing modules:")
        for spec in missing_specs:
            print(f"  {spec.code}: {spec.description}")
        
        if self.config.dry_run:
            print("\nDry run mode - no files will be created")
        
        print("\nGenerating modules...")
        self.scaffold_modules(missing_specs)
        
        print("\nUpdating index...")
        self.update_index(missing_specs)
        
        print(f"\n✓ Scaffolding complete! Generated {len(missing_specs)} modules.")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Scaffold missing canonical modules",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --ranges 08X-12X,52S-57S
  %(prog)s --phase context_construction --dry-run  
  %(prog)s --ranges 08X-10X --force
        """
    )
    
    parser.add_argument(
        '--ranges', 
        help='Comma-separated ranges like "08X-12X,52S-57S"'
    )
    parser.add_argument(
        '--phase', '--phases',
        action='append',
        help='Target phase(s): context_construction, synthesis_output, etc.'
    )
    parser.add_argument(
        '--dry-run', 
        action='store_true',
        help='Show what would be generated without creating files'
    )
    parser.add_argument(
        '--force',
        action='store_true', 
        help='Overwrite existing files'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('canonical_flow'),
        help='Output directory (default: canonical_flow)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.ranges and not args.phase:
        parser.error("Must specify either --ranges or --phase")
    
    # Build config
    config = ScaffoldConfig(
        target_ranges=args.ranges.split(',') if args.ranges else [],
        target_phases=args.phase or [],
        dry_run=args.dry_run,
        force=args.force,
        output_dir=args.output_dir
    )
    
    # Run scaffolder
    scaffolder = CanonicalScaffolder(config)
    scaffolder.run()


if __name__ == '__main__':
    main()
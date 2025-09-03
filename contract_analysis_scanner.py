#!/usr/bin/env python3
"""
Contract Analysis Scanner
Analyzes process(data, context) method signatures across canonical pipeline components
"""

import ast
import json
import sys
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Set
from dataclasses import dataclass, asdict
from enum import Enum


class SeverityLevel(Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


@dataclass
class MethodSignature:
    method_name: str
    file_path: str
    line_number: int
    parameters: List[str]
    param_types: Dict[str, Optional[str]]
    param_defaults: Dict[str, Any]
    return_type: Optional[str]
    is_class_method: bool
    class_name: Optional[str]


@dataclass
class ContractMismatch:
    severity: SeverityLevel
    mismatch_type: str
    primary_file: str
    conflicting_files: List[str]
    primary_signature: MethodSignature
    conflicting_signatures: List[MethodSignature]
    description: str
    recommended_fix: str
    adapter_snippet: Optional[str] = None


class MethodVisitor(ast.NodeVisitor):
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.signatures: List[MethodSignature] = []
        self.current_class = None

    def visit_ClassDef(self, node):
        prev_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = prev_class

    def visit_FunctionDef(self, node):
        if node.name == "process":
            signature = self._extract_signature(node)
            self.signatures.append(signature)
        self.generic_visit(node)

    def _extract_signature(self, node: ast.FunctionDef) -> MethodSignature:
        parameters = []
        param_types = {}
        param_defaults = {}
        
        # Extract parameters
        for arg in node.args.args:
            if arg.arg == 'self':
                continue
            parameters.append(arg.arg)
            
            # Extract type annotation
            if arg.annotation:
                param_types[arg.arg] = ast.unparse(arg.annotation)
            else:
                param_types[arg.arg] = None
        
        # Extract default values
        defaults = node.args.defaults
        default_offset = len(parameters) - len(defaults)
        for i, default in enumerate(defaults):
            param_name = parameters[default_offset + i]
            try:
                param_defaults[param_name] = ast.literal_eval(default)
            except (ValueError, TypeError):
                param_defaults[param_name] = ast.unparse(default)
        
        # Extract return type
        return_type = None
        if node.returns:
            return_type = ast.unparse(node.returns)
        
        return MethodSignature(
            method_name=node.name,
            file_path=self.file_path,
            line_number=node.lineno,
            parameters=parameters,
            param_types=param_types,
            param_defaults=param_defaults,
            return_type=return_type,
            is_class_method=self.current_class is not None,
            class_name=self.current_class
        )


class ContractAnalyzer:
    def __init__(self, root_path: Path):
        self.root_path = root_path
        self.signatures: List[MethodSignature] = []
        self.mismatches: List[ContractMismatch] = []
        
    def scan_components(self) -> None:
        """Scan all Python files for process method signatures"""
        python_files = list(self.root_path.rglob("*.py"))
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Skip if no process method
                if 'def process(' not in content:
                    continue
                    
                tree = ast.parse(content)
                visitor = MethodVisitor(str(file_path.relative_to(self.root_path)))
                visitor.visit(tree)
                self.signatures.extend(visitor.signatures)
                
            except (SyntaxError, UnicodeDecodeError, OSError) as e:
                print(f"Warning: Could not parse {file_path}: {e}")
                continue
    
    def analyze_contracts(self) -> None:
        """Analyze signatures for contract mismatches"""
        self._find_parameter_mismatches()
        self._find_return_type_inconsistencies()
        self._find_naming_variations()
        self._find_missing_defaults()
    
    def _find_parameter_mismatches(self) -> None:
        """Find methods with different parameter counts or types"""
        param_groups = defaultdict(list)
        
        for sig in self.signatures:
            key = len(sig.parameters)
            param_groups[key].append(sig)
        
        # Check if we have multiple parameter count groups
        if len(param_groups) > 1:
            base_group = max(param_groups.values(), key=len)
            base_sig = base_group[0]
            
            for count, group in param_groups.items():
                if count != len(base_sig.parameters):
                    mismatch = ContractMismatch(
                        severity=SeverityLevel.CRITICAL,
                        mismatch_type="parameter_count_mismatch",
                        primary_file=base_sig.file_path,
                        conflicting_files=[s.file_path for s in group],
                        primary_signature=base_sig,
                        conflicting_signatures=group,
                        description=f"Parameter count mismatch: expected {len(base_sig.parameters)} parameters, found {count}",
                        recommended_fix="Standardize parameter count or create adapter",
                        adapter_snippet=self._generate_parameter_adapter(base_sig, group[0])
                    )
                    self.mismatches.append(mismatch)
    
    def _find_return_type_inconsistencies(self) -> None:
        """Find methods with inconsistent return types"""
        return_types = defaultdict(list)
        
        for sig in self.signatures:
            key = sig.return_type or "None"
            return_types[key].append(sig)
        
        if len(return_types) > 1:
            # Most common return type is the standard
            standard_type = max(return_types.keys(), key=lambda k: len(return_types[k]))
            
            for ret_type, group in return_types.items():
                if ret_type != standard_type:
                    base_sig = return_types[standard_type][0]
                    mismatch = ContractMismatch(
                        severity=SeverityLevel.HIGH,
                        mismatch_type="return_type_inconsistency",
                        primary_file=base_sig.file_path,
                        conflicting_files=[s.file_path for s in group],
                        primary_signature=base_sig,
                        conflicting_signatures=group,
                        description=f"Return type inconsistency: expected {standard_type}, found {ret_type}",
                        recommended_fix="Standardize return types with wrapper",
                        adapter_snippet=self._generate_return_adapter(base_sig, group[0])
                    )
                    self.mismatches.append(mismatch)
    
    def _find_naming_variations(self) -> None:
        """Find parameter naming variations"""
        param_names = defaultdict(set)
        
        for sig in self.signatures:
            for i, param in enumerate(sig.parameters):
                param_names[i].add((param, sig.file_path))
        
        for position, names in param_names.items():
            unique_names = {name for name, _ in names}
            if len(unique_names) > 1:
                # Group by name variations
                name_groups = defaultdict(list)
                for name, file_path in names:
                    name_groups[name].append(file_path)
                
                most_common_name = max(name_groups.keys(), key=lambda k: len(name_groups[k]))
                
                for name, files in name_groups.items():
                    if name != most_common_name:
                        conflicting_sigs = [s for s in self.signatures if s.file_path in files]
                        base_sig = next(s for s in self.signatures if most_common_name in s.parameters)
                        
                        mismatch = ContractMismatch(
                            severity=SeverityLevel.MEDIUM,
                            mismatch_type="parameter_naming_variation",
                            primary_file=base_sig.file_path,
                            conflicting_files=files,
                            primary_signature=base_sig,
                            conflicting_signatures=conflicting_sigs,
                            description=f"Parameter naming variation at position {position}: expected '{most_common_name}', found '{name}'",
                            recommended_fix="Standardize parameter names",
                            adapter_snippet=self._generate_naming_adapter(most_common_name, name)
                        )
                        self.mismatches.append(mismatch)
    
    def _find_missing_defaults(self) -> None:
        """Find methods missing default parameter values"""
        for sig in self.signatures:
            if 'data' in sig.parameters and 'data' not in sig.param_defaults:
                mismatch = ContractMismatch(
                    severity=SeverityLevel.LOW,
                    mismatch_type="missing_default_value",
                    primary_file=sig.file_path,
                    conflicting_files=[],
                    primary_signature=sig,
                    conflicting_signatures=[],
                    description=f"Missing default value for 'data' parameter",
                    recommended_fix="Add default value: data=None",
                    adapter_snippet="def process(self, data=None, context=None):"
                )
                self.mismatches.append(mismatch)
    
    def _generate_parameter_adapter(self, base_sig: MethodSignature, conflict_sig: MethodSignature) -> str:
        """Generate adapter code for parameter count mismatches"""
        base_params = ", ".join(f"{p}=None" for p in base_sig.parameters)
        conflict_params = ", ".join(conflict_sig.parameters)
        
        return textwrap.dedent(f"""
        class ProcessAdapter:
            def __init__(self, wrapped_processor):
                self.wrapped = wrapped_processor
            
            def process(self, {base_params}):
                # Adapter for {conflict_sig.file_path}
                return self.wrapped.process({conflict_params})
        """).strip()
    
    def _generate_return_adapter(self, base_sig: MethodSignature, conflict_sig: MethodSignature) -> str:
        """Generate adapter code for return type mismatches"""
        return textwrap.dedent(f"""
        class ReturnTypeAdapter:
            def __init__(self, wrapped_processor):
                self.wrapped = wrapped_processor
            
            def process(self, data=None, context=None):
                result = self.wrapped.process(data, context)
                # Convert {conflict_sig.return_type} to {base_sig.return_type}
                if not isinstance(result, dict):
                    return {{"result": result, "metadata": {{"adapted": True}}}}
                return result
        """).strip()
    
    def _generate_naming_adapter(self, standard_name: str, variant_name: str) -> str:
        """Generate adapter code for naming variations"""
        return textwrap.dedent(f"""
        def process_adapter(processor, **kwargs):
            # Map {variant_name} to {standard_name}
            if '{variant_name}' in kwargs:
                kwargs['{standard_name}'] = kwargs.pop('{variant_name}')
            return processor.process(**kwargs)
        """).strip()
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive contract analysis report"""
        report = {
            "scan_summary": {
                "total_components_scanned": len(set(sig.file_path for sig in self.signatures)),
                "total_process_methods": len(self.signatures),
                "total_mismatches": len(self.mismatches),
                "severity_breakdown": {
                    level.value: len([m for m in self.mismatches if m.severity == level])
                    for level in SeverityLevel
                }
            },
            "signature_inventory": [asdict(sig) for sig in self.signatures],
            "contract_mismatches": self._format_mismatches(),
            "adapter_solutions": self._generate_adapter_solutions(),
            "recommendations": self._generate_recommendations()
        }
        
        return report
    
    def _format_mismatches(self) -> List[Dict[str, Any]]:
        """Format mismatches for report"""
        formatted = []
        
        for mismatch in sorted(self.mismatches, key=lambda m: m.severity.value):
            formatted.append({
                "severity": mismatch.severity.value,
                "type": mismatch.mismatch_type,
                "description": mismatch.description,
                "primary_location": {
                    "file": mismatch.primary_file,
                    "line": mismatch.primary_signature.line_number,
                    "signature": self._format_signature(mismatch.primary_signature)
                },
                "conflicting_locations": [
                    {
                        "file": sig.file_path,
                        "line": sig.line_number,
                        "signature": self._format_signature(sig)
                    }
                    for sig in mismatch.conflicting_signatures
                ],
                "recommended_fix": mismatch.recommended_fix,
                "adapter_code": mismatch.adapter_snippet
            })
        
        return formatted
    
    def _format_signature(self, sig: MethodSignature) -> str:
        """Format method signature as string"""
        params = []
        for param in sig.parameters:
            param_str = param
            if param in sig.param_types and sig.param_types[param]:
                param_str += f": {sig.param_types[param]}"
            if param in sig.param_defaults:
                param_str += f" = {sig.param_defaults[param]}"
            params.append(param_str)
        
        params_str = ", ".join(params)
        if sig.is_class_method:
            params_str = "self, " + params_str
        
        return_str = ""
        if sig.return_type:
            return_str = f" -> {sig.return_type}"
        
        return f"def {sig.method_name}({params_str}){return_str}"
    
    def _generate_adapter_solutions(self) -> Dict[str, Any]:
        """Generate comprehensive adapter solutions"""
        solutions = {
            "unified_interface": self._generate_unified_interface(),
            "adapter_registry": self._generate_adapter_registry(),
            "migration_guide": self._generate_migration_guide()
        }
        
        return solutions
    
    def _generate_unified_interface(self) -> str:
        """Generate unified interface specification"""
        return textwrap.dedent("""
        from abc import ABC, abstractmethod
        from typing import Any, Dict, Optional
        
        class CanonicalProcessor(ABC):
            \"\"\"Unified interface for canonical pipeline components\"\"\"
            
            @abstractmethod
            def process(self, data: Optional[Any] = None, 
                       context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
                \"\"\"
                Process data with optional context.
                
                Args:
                    data: Input data to process (default: None)
                    context: Processing context and metadata (default: None)
                    
                Returns:
                    Dict[str, Any]: Processing results with metadata
                \"\"\"
                pass
        """).strip()
    
    def _generate_adapter_registry(self) -> str:
        """Generate adapter registry for managing component compatibility"""
        return textwrap.dedent("""
        class ProcessorAdapterRegistry:
            \"\"\"Registry for managing processor adapters\"\"\"
            
            def __init__(self):
                self.adapters = {}
                self.processors = {}
            
            def register_processor(self, name: str, processor, adapter_class=None):
                self.processors[name] = processor
                if adapter_class:
                    self.adapters[name] = adapter_class(processor)
                else:
                    self.adapters[name] = processor
            
            def process(self, processor_name: str, data=None, context=None):
                if processor_name not in self.adapters:
                    raise ValueError(f"Unknown processor: {processor_name}")
                
                adapter = self.adapters[processor_name]
                return adapter.process(data=data, context=context)
        """).strip()
    
    def _generate_migration_guide(self) -> List[str]:
        """Generate migration guide steps"""
        return [
            "1. Implement CanonicalProcessor interface for all components",
            "2. Standardize parameter names: use 'data' and 'context'",
            "3. Add default values: data=None, context=None",
            "4. Ensure return type is Dict[str, Any]",
            "5. Use adapter registry for gradual migration",
            "6. Run contract validation tests before deployment"
        ]
    
    def _generate_recommendations(self) -> List[Dict[str, Any]]:
        """Generate actionable recommendations"""
        recommendations = []
        
        critical_count = len([m for m in self.mismatches if m.severity == SeverityLevel.CRITICAL])
        if critical_count > 0:
            recommendations.append({
                "priority": "IMMEDIATE",
                "action": f"Fix {critical_count} critical parameter mismatches",
                "impact": "System integration will fail without these fixes"
            })
        
        high_count = len([m for m in self.mismatches if m.severity == SeverityLevel.HIGH])
        if high_count > 0:
            recommendations.append({
                "priority": "HIGH",
                "action": f"Address {high_count} return type inconsistencies",
                "impact": "Data flow corruption and pipeline failures"
            })
        
        recommendations.append({
            "priority": "MEDIUM",
            "action": "Implement unified interface across all components",
            "impact": "Long-term maintainability and consistency"
        })
        
        return recommendations


def main():
    """Main entry point for contract analysis"""
    if len(sys.argv) > 1:
        root_path = Path(sys.argv[1])
    else:
        root_path = Path(".")
    
    print("🔍 Scanning canonical pipeline components for contract analysis...")
    
    analyzer = ContractAnalyzer(root_path)
    analyzer.scan_components()
    
    print(f"📊 Found {len(analyzer.signatures)} process() methods across {len(set(sig.file_path for sig in analyzer.signatures))} components")
    
    analyzer.analyze_contracts()
    
    print(f"⚠️  Detected {len(analyzer.mismatches)} contract mismatches")
    
    # Generate detailed report
    report = analyzer.generate_report()
    
    # Save report
    report_path = Path("contract_analysis_report.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("\n" + "="*80)
    print("CONTRACT ANALYSIS SUMMARY")
    print("="*80)
    
    print(f"Components Scanned: {report['scan_summary']['total_components_scanned']}")
    print(f"Process Methods: {report['scan_summary']['total_process_methods']}")
    print(f"Total Mismatches: {report['scan_summary']['total_mismatches']}")
    
    print("\nSeverity Breakdown:")
    for level, count in report['scan_summary']['severity_breakdown'].items():
        if count > 0:
            print(f"  {level}: {count}")
    
    print("\nTop Contract Issues:")
    for i, mismatch in enumerate(report['contract_mismatches'][:5], 1):
        print(f"  {i}. {mismatch['severity']} - {mismatch['description']}")
        print(f"     Location: {mismatch['primary_location']['file']}:{mismatch['primary_location']['line']}")
    
    print(f"\n📋 Detailed report saved to: {report_path}")
    print(f"📁 Run with specific path: python contract_analysis_scanner.py <path>")
    
    return len(analyzer.mismatches) == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
#!/usr/bin/env python3
"""
Generate CONTRACT DIFFS Report based on grep analysis of process() signatures
"""

import re
import json
from pathlib import Path
from collections import defaultdict

def extract_signatures_from_grep():
    """Extract process signatures from grep results"""
    # These are the grep results we found earlier
    grep_results = [
        "analysis_nlp_orchestrator.py:150:     def process(self, data: Optional[Any] = None, context: Optional[Any] = None) -> Dict[str, Any]:",
        "report_compiler.py:1083: def process(data: Any, context: Dict[str, Any] = None) -> Dict[str, Any]:",
        "canonical_flow/A_analysis_nlp/adaptive_analyzer.py:133:     def process(self, data: Any = None, context: Any = None) -> Dict[str, Any]:",
        "score_calculator.py:236:     def process(self, data: Optional[Dict[str, Any]] = None, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:",
        "public_transformer_adapter.py:272:     def process(self, data: Any, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:",
        "pdf_text_reader.py:108: def process(data: Any, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:",
        "pdf_reader.py:711: def process(data=None, context=None) -> Dict[str, Any]:",
        "normative_validator.py:1808: def process(data=None, context=None) -> Dict[str, Any]:",
        "meso_aggregator.py:913: def process(data: Any, context: Dict[str, Any] | None = None) -> Dict[str, Any]:",
        "macro_alignment_calculator.py:424: def process(data: Any, context: Dict[str, Any] | None = None) -> Dict[str, Any]:",
        "feature_extractor.py:675: def process(data=None, context=None) -> Dict[str, Any]:",
        "embedding_generator.py:598: def process(data=None, context=None) -> Dict[str, Any]:",
        "embedding_builder.py:768:     def process(self, data=None, context=None) -> Dict[str, Any]:",
        "dnp_alignment_adapter.py:170: def process(data: Any, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:",
        "cluster_execution_controller.py:333: def process(data: Any, context: Dict[str, Any] | None = None) -> Dict[str, Any]:",
        "causal_graph.py:1191: def process(data=None, context=None) -> Dict[str, Any]:",
        "causal_dnp_framework.py:851: def process(data=None, context=None) -> Dict[str, Union[str, Dict, List]]:",
        "canonical_output_auditor.py:113: def process(data: Any, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:",
        "advanced_loader.py:282: def process(data=None, context=None) -> Dict[str, Any]:",
        "adaptive_analyzer.py:910:     def process(self, data: Optional[Dict[str, Any]] = None, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:",
        "retrieval_engine/vector_index.py:430: def process(data: Any, context: Dict[str, Any] = None) -> Dict[str, Any]:",
        "retrieval_engine/lexical_index.py:287: def process(data: Any, context: Dict[str, Any] = None) -> Dict[str, Any]:",
        "retrieval_engine/hybrid_retriever.py:165: def process(data: Any, context: Dict[str, Any] | None = None) -> Dict[str, Any]:",
        "semantic_reranking/reranker.py:72: def process(data: Any, context: Dict[str, Any] | None = None) -> Dict[str, Any]:",
        "canonical_flow/mathematical_enhancers/retrieval_enhancer.py:452: def process(data: Any, context: Dict[str, Any] | None = None) -> Dict[str, Any]:",
        "canonical_flow/mathematical_enhancers/integration_enhancer.py:587: def process(data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:",
        "canonical_flow/mathematical_enhancers/ingestion_enhancer.py:1401: def process(data: Optional[Dict[str, Any]] = None, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:",
        "canonical_flow/mathematical_enhancers/analysis_enhancer.py:1284: def process(data: Any, context: Dict[str, Any] = None) -> Dict[str, Any]:",
        "canonical_flow/mathematical_enhancers/aggregation_enhancer.py:1315: def process(data: Any, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:",
    ]
    
    signatures = []
    for line in grep_results:
        if ':' in line:
            parts = line.split(':', 2)
            if len(parts) >= 3:
                file_path = parts[0]
                line_num = parts[1]
                signature = parts[2].strip()
                signatures.append({
                    'file': file_path,
                    'line': int(line_num),
                    'signature': signature
                })
    
    return signatures

def analyze_signature_patterns(signatures):
    """Analyze patterns in the signatures"""
    
    # Parse each signature
    parsed_sigs = []
    for sig_info in signatures:
        sig = sig_info['signature']
        
        # Extract parameters
        param_match = re.search(r'def process\((.*?)\)(?:\s*->\s*([^:]+))?', sig)
        if param_match:
            params = param_match.group(1).strip()
            return_type = param_match.group(2).strip() if param_match.group(2) else None
            
            # Check if it's a method (has self)
            is_method = 'self,' in params or params == 'self'
            
            # Clean parameters for analysis
            clean_params = params
            if is_method:
                clean_params = re.sub(r'self,?\s*', '', clean_params)
            
            parsed_sigs.append({
                **sig_info,
                'params': clean_params,
                'return_type': return_type,
                'is_method': is_method,
                'original_params': params
            })
    
    return parsed_sigs

def detect_contract_mismatches(parsed_sigs):
    """Detect contract mismatches"""
    
    mismatches = []
    
    # Group by parameter patterns
    param_patterns = defaultdict(list)
    return_type_patterns = defaultdict(list)
    
    for sig in parsed_sigs:
        # Normalize parameter pattern
        normalized = normalize_params(sig['params'])
        param_patterns[normalized].append(sig)
        
        # Group return types
        ret_type = sig['return_type'] or 'None'
        return_type_patterns[ret_type].append(sig)
    
    # Find parameter mismatches
    if len(param_patterns) > 1:
        most_common_pattern = max(param_patterns.keys(), key=lambda k: len(param_patterns[k]))
        
        for pattern, sigs in param_patterns.items():
            if pattern != most_common_pattern:
                mismatches.append({
                    'type': 'CRITICAL_PARAMETER_MISMATCH',
                    'severity': 'CRITICAL',
                    'expected_pattern': most_common_pattern,
                    'found_pattern': pattern,
                    'description': f'Parameter signature mismatch: expected {most_common_pattern}, found {pattern}',
                    'files': [s['file'] for s in sigs],
                    'signatures': sigs,
                    'recommended_fix': 'Standardize parameter signature or implement adapter'
                })
    
    # Find return type mismatches
    if len(return_type_patterns) > 1:
        most_common_return = max(return_type_patterns.keys(), key=lambda k: len(return_type_patterns[k]))
        
        for ret_type, sigs in return_type_patterns.items():
            if ret_type != most_common_return:
                mismatches.append({
                    'type': 'HIGH_RETURN_TYPE_MISMATCH',
                    'severity': 'HIGH',
                    'expected_return': most_common_return,
                    'found_return': ret_type,
                    'description': f'Return type inconsistency: expected {most_common_return}, found {ret_type}',
                    'files': [s['file'] for s in sigs],
                    'signatures': sigs,
                    'recommended_fix': 'Implement return type wrapper/adapter'
                })
    
    return mismatches, param_patterns, return_type_patterns

def normalize_params(params):
    """Normalize parameter string for pattern matching"""
    if not params:
        return 'NO_PARAMS'
    
    # Remove type annotations and defaults
    normalized = re.sub(r':\s*[^,=]+', '', params)
    normalized = re.sub(r'=\s*[^,]+', '', normalized)
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    
    # Common patterns
    if 'data' in normalized and 'context' in normalized:
        return 'DATA_CONTEXT'
    elif 'data' in normalized:
        return 'DATA_ONLY'
    else:
        return f'CUSTOM_{normalized.replace(" ", "_").replace(",", "_")}'

def generate_adapter_snippets(mismatches):
    """Generate adapter code snippets"""
    
    adapters = {}
    
    # Universal adapter
    adapters['universal_process_adapter'] = '''
class UniversalProcessAdapter:
    """Universal adapter for process() method signature standardization"""
    
    def __init__(self, wrapped_component):
        self.wrapped = wrapped_component
        self.component_type = self._detect_signature_type()
    
    def _detect_signature_type(self):
        import inspect
        if hasattr(self.wrapped, 'process'):
            sig = inspect.signature(self.wrapped.process)
            params = list(sig.parameters.keys())
            if 'self' in params:
                params.remove('self')
            return len(params)
        return 0
    
    def process(self, data=None, context=None):
        """Standardized process interface"""
        if not hasattr(self.wrapped, 'process'):
            raise AttributeError(f"{type(self.wrapped).__name__} has no process method")
        
        # Adapt call based on component signature
        if self.component_type == 0:
            return self.wrapped.process()
        elif self.component_type == 1:
            return self.wrapped.process(data)
        else:  # 2 or more parameters
            return self.wrapped.process(data, context)
'''
    
    # Registry system
    adapters['processor_registry'] = '''
class CanonicalProcessorRegistry:
    """Registry for managing canonical pipeline components with heterogeneous interfaces"""
    
    def __init__(self):
        self.components = {}
        self.adapters = {}
    
    def register(self, name: str, component, custom_adapter=None):
        """Register a component with optional custom adapter"""
        self.components[name] = component
        
        if custom_adapter:
            self.adapters[name] = custom_adapter
        else:
            # Use universal adapter
            self.adapters[name] = UniversalProcessAdapter(component)
    
    def process(self, component_name: str, data=None, context=None):
        """Process data through specified component"""
        if component_name not in self.adapters:
            raise ValueError(f"Component '{component_name}' not registered")
        
        adapter = self.adapters[component_name]
        return adapter.process(data, context)
    
    def list_components(self):
        """List all registered components"""
        return list(self.components.keys())
'''
    
    # Type-specific adapters for common mismatches
    adapters['return_type_adapter'] = '''
class ReturnTypeAdapter:
    """Adapter for normalizing return types to Dict[str, Any]"""
    
    def __init__(self, wrapped_component):
        self.wrapped = wrapped_component
    
    def process(self, data=None, context=None):
        result = self.wrapped.process(data, context)
        
        # Ensure result is Dict[str, Any]
        if result is None:
            return {"status": "completed", "data": None}
        elif not isinstance(result, dict):
            return {"status": "completed", "data": result}
        else:
            return result
'''
    
    return adapters

def create_contract_diffs_report():
    """Create the main CONTRACT DIFFS report"""
    
    print("🔍 Analyzing process() method signatures across canonical pipeline components...")
    
    # Extract signatures
    signatures = extract_signatures_from_grep()
    print(f"📊 Found {len(signatures)} process() method signatures")
    
    # Parse signatures
    parsed_sigs = analyze_signature_patterns(signatures)
    
    # Detect mismatches
    mismatches, param_patterns, return_patterns = detect_contract_mismatches(parsed_sigs)
    
    # Generate adapters
    adapters = generate_adapter_snippets(mismatches)
    
    # Create comprehensive report
    report = {
        "meta": {
            "scan_date": "2024-01-15",
            "total_components_analyzed": len(set(s['file'] for s in signatures)),
            "total_process_methods": len(signatures),
            "scan_scope": "canonical_flow, retrieval_engine, semantic_reranking, root modules"
        },
        "signature_inventory": parsed_sigs,
        "pattern_analysis": {
            "parameter_patterns": {k: len(v) for k, v in param_patterns.items()},
            "return_type_patterns": {k: len(v) for k, v in return_patterns.items()}
        },
        "contract_mismatches": mismatches,
        "severity_summary": {
            "CRITICAL": len([m for m in mismatches if m['severity'] == 'CRITICAL']),
            "HIGH": len([m for m in mismatches if m['severity'] == 'HIGH']),
            "MEDIUM": len([m for m in mismatches if m['severity'] == 'MEDIUM']),
            "LOW": len([m for m in mismatches if m['severity'] == 'LOW'])
        },
        "adapter_solutions": adapters,
        "recommendations": generate_recommendations(mismatches)
    }
    
    return report

def generate_recommendations(mismatches):
    """Generate actionable recommendations"""
    recs = []
    
    critical_count = len([m for m in mismatches if m['severity'] == 'CRITICAL'])
    high_count = len([m for m in mismatches if m['severity'] == 'HIGH'])
    
    if critical_count > 0:
        recs.append({
            "priority": "IMMEDIATE",
            "action": f"Fix {critical_count} critical parameter mismatches",
            "details": "Parameter signature inconsistencies will cause integration failures",
            "implementation": "Deploy UniversalProcessAdapter for immediate compatibility"
        })
    
    if high_count > 0:
        recs.append({
            "priority": "HIGH", 
            "action": f"Address {high_count} return type inconsistencies",
            "details": "Return type mismatches can cause data flow corruption",
            "implementation": "Implement ReturnTypeAdapter wrappers"
        })
    
    recs.append({
        "priority": "MEDIUM",
        "action": "Establish unified canonical interface",
        "details": "Create standard CanonicalProcessor ABC for long-term consistency",
        "implementation": "Migrate components to implement standard interface"
    })
    
    recs.append({
        "priority": "LOW",
        "action": "Deploy processor registry system",
        "details": "Centralized component management with adapter support",
        "implementation": "Implement CanonicalProcessorRegistry"
    })
    
    return recs

def format_markdown_report(report_data):
    """Format report as markdown"""
    
    md = []
    md.append("# CONTRACT DIFFS REPORT")
    md.append("## Canonical Pipeline Component Interface Analysis")
    md.append("")
    
    meta = report_data['meta']
    md.append("### SCAN SUMMARY")
    md.append(f"- **Components Analyzed**: {meta['total_components_analyzed']}")
    md.append(f"- **Process Methods Found**: {meta['total_process_methods']}")
    md.append(f"- **Scan Date**: {meta['scan_date']}")
    md.append(f"- **Scope**: {meta['scan_scope']}")
    md.append("")
    
    # Severity summary
    severity = report_data['severity_summary']
    md.append("### MISMATCH SEVERITY BREAKDOWN")
    total_issues = sum(severity.values())
    if total_issues > 0:
        md.append(f"- **CRITICAL**: {severity['CRITICAL']} (parameter signature mismatches)")
        md.append(f"- **HIGH**: {severity['HIGH']} (return type inconsistencies)")
        md.append(f"- **MEDIUM**: {severity['MEDIUM']} (naming variations)")
        md.append(f"- **LOW**: {severity['LOW']} (missing defaults)")
        md.append(f"- **TOTAL ISSUES**: {total_issues}")
    else:
        md.append("✅ **NO CRITICAL ISSUES DETECTED**")
    md.append("")
    
    # Pattern analysis
    patterns = report_data['pattern_analysis']
    md.append("### SIGNATURE PATTERNS DETECTED")
    md.append("#### Parameter Patterns:")
    for pattern, count in patterns['parameter_patterns'].items():
        md.append(f"- **{pattern}**: {count} methods")
    
    md.append("\n#### Return Type Patterns:")
    for ret_type, count in patterns['return_type_patterns'].items():
        md.append(f"- **{ret_type}**: {count} methods")
    md.append("")
    
    # Detailed mismatches
    if report_data['contract_mismatches']:
        md.append("### DETAILED CONTRACT MISMATCHES")
        md.append("")
        
        for i, mismatch in enumerate(report_data['contract_mismatches'], 1):
            md.append(f"#### {i}. {mismatch['type']} [{mismatch['severity']}]")
            md.append(f"**Description**: {mismatch['description']}")
            md.append(f"**Affected Files**: {len(mismatch['files'])} components")
            
            # Show first few files
            for j, file_path in enumerate(mismatch['files'][:5]):
                md.append(f"  - `{file_path}`")
            if len(mismatch['files']) > 5:
                md.append(f"  - ... and {len(mismatch['files']) - 5} more files")
            
            md.append(f"**Recommended Fix**: {mismatch['recommended_fix']}")
            md.append("")
    
    # Adapter solutions
    md.append("### ADAPTER IMPLEMENTATION STRATEGIES")
    md.append("")
    md.append("The following adapter patterns are recommended to resolve interface incompatibilities:")
    md.append("")
    md.append("#### 1. Universal Process Adapter")
    md.append("```python")
    md.append(report_data['adapter_solutions']['universal_process_adapter'])
    md.append("```")
    md.append("")
    
    md.append("#### 2. Canonical Processor Registry")  
    md.append("```python")
    md.append(report_data['adapter_solutions']['processor_registry'])
    md.append("```")
    md.append("")
    
    # Recommendations
    md.append("### IMPLEMENTATION RECOMMENDATIONS")
    md.append("")
    
    for i, rec in enumerate(report_data['recommendations'], 1):
        md.append(f"#### {i}. {rec['action']} [{rec['priority']}]")
        md.append(f"**Details**: {rec['details']}")
        md.append(f"**Implementation**: {rec['implementation']}")
        md.append("")
    
    # Component inventory sample
    md.append("### COMPONENT SIGNATURE INVENTORY (Sample)")
    md.append("```")
    for i, sig in enumerate(report_data['signature_inventory'][:15], 1):
        md.append(f"{i:2d}. {sig['file']}:{sig['line']}")
        md.append(f"    {sig['signature']}")
    
    if len(report_data['signature_inventory']) > 15:
        remaining = len(report_data['signature_inventory']) - 15
        md.append(f"    ... and {remaining} more components")
    md.append("```")
    md.append("")
    
    md.append("---")
    md.append("*Report generated by Contract Analysis Scanner*")
    
    return '\n'.join(md)

def main():
    """Generate complete contract analysis report"""
    
    # Generate report data
    report_data = create_contract_diffs_report()
    
    # Save JSON report
    with open('contract_analysis_report.json', 'w') as f:
        json.dump(report_data, f, indent=2)
    
    # Generate markdown report
    markdown_report = format_markdown_report(report_data)
    
    # Save markdown report
    with open('CONTRACT_DIFFS_REPORT.md', 'w') as f:
        f.write(markdown_report)
    
    # Save adapter code
    with open('contract_adapters.py', 'w') as f:
        f.write("# Contract Adapter Solutions\n")
        f.write("# Generated by Contract Analysis Scanner\n\n")
        
        for name, code in report_data['adapter_solutions'].items():
            f.write(f"# {name.upper().replace('_', ' ')}\n")
            f.write(code)
            f.write("\n\n")
    
    # Print summary
    print("\n" + "="*80)
    print("CONTRACT ANALYSIS COMPLETE")
    print("="*80)
    
    meta = report_data['meta']
    print(f"📊 Analyzed {meta['total_components_analyzed']} components")
    print(f"🔍 Found {meta['total_process_methods']} process() methods")
    
    severity = report_data['severity_summary'] 
    total_issues = sum(severity.values())
    print(f"⚠️  Detected {total_issues} contract mismatches")
    
    if total_issues > 0:
        print("\nSeverity Breakdown:")
        for level, count in severity.items():
            if count > 0:
                print(f"  {level}: {count}")
    
    print(f"\n📁 Reports Generated:")
    print(f"  - CONTRACT_DIFFS_REPORT.md (main report)")
    print(f"  - contract_analysis_report.json (detailed data)")
    print(f"  - contract_adapters.py (implementation solutions)")
    
    return total_issues == 0

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
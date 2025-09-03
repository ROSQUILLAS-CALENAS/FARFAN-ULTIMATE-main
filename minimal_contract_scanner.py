#!/usr/bin/env python3
"""
Minimal Contract Scanner - Quick analysis of process() method signatures
"""

import re
import json
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any

class MinimalContractScanner:
    def __init__(self, root_path: Path = Path(".")):
        self.root_path = root_path
        self.process_signatures = []
        
    def scan_files(self):
        """Scan all Python files for process method signatures using regex"""
        pattern = r'def process\(([^)]*)\)(?:\s*->\s*([^:]+))?:'
        
        for py_file in self.root_path.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue
                
            try:
                content = py_file.read_text(encoding='utf-8')
                matches = re.finditer(pattern, content)
                
                for match in matches:
                    params = match.group(1).strip()
                    return_type = match.group(2).strip() if match.group(2) else None
                    line_num = content[:match.start()].count('\n') + 1
                    
                    self.process_signatures.append({
                        'file': str(py_file.relative_to(self.root_path)),
                        'line': line_num,
                        'parameters': params,
                        'return_type': return_type,
                        'signature': f"def process({params})" + (f" -> {return_type}" if return_type else "") + ":"
                    })
                    
            except (UnicodeDecodeError, OSError):
                continue
    
    def analyze_patterns(self) -> Dict[str, Any]:
        """Analyze signature patterns and detect inconsistencies"""
        if not self.process_signatures:
            return {"error": "No process() methods found"}
        
        # Parameter patterns
        param_patterns = Counter()
        return_patterns = Counter()
        
        # Categorize signatures
        signature_categories = defaultdict(list)
        
        for sig in self.process_signatures:
            # Clean parameters for analysis
            params = re.sub(r'\s+', ' ', sig['parameters']).strip()
            params_normalized = self._normalize_params(params)
            
            param_patterns[params_normalized] += 1
            return_patterns[sig['return_type']] += 1
            
            # Categorize by signature pattern
            signature_categories[params_normalized].append(sig)
        
        # Identify mismatches
        mismatches = []
        if len(param_patterns) > 1:
            most_common = param_patterns.most_common(1)[0][0]
            for pattern, count in param_patterns.items():
                if pattern != most_common:
                    conflicting_files = [s['file'] for s in signature_categories[pattern]]
                    mismatches.append({
                        'type': 'parameter_mismatch',
                        'expected_pattern': most_common,
                        'found_pattern': pattern,
                        'count': count,
                        'files': conflicting_files
                    })
        
        if len(return_patterns) > 1:
            most_common_return = return_patterns.most_common(1)[0][0]
            for ret_type, count in return_patterns.items():
                if ret_type != most_common_return and ret_type is not None:
                    mismatches.append({
                        'type': 'return_type_mismatch',
                        'expected_return': most_common_return,
                        'found_return': ret_type,
                        'count': count
                    })
        
        return {
            'summary': {
                'total_process_methods': len(self.process_signatures),
                'unique_parameter_patterns': len(param_patterns),
                'unique_return_types': len(return_patterns),
                'total_files': len(set(s['file'] for s in self.process_signatures))
            },
            'parameter_patterns': dict(param_patterns),
            'return_type_patterns': dict(return_patterns),
            'mismatches': mismatches,
            'all_signatures': self.process_signatures
        }
    
    def _normalize_params(self, params: str) -> str:
        """Normalize parameter strings for comparison"""
        if not params:
            return "no_params"
        
        # Remove type annotations and defaults for pattern matching
        normalized = re.sub(r':\s*[^,=]+', '', params)  # Remove type hints
        normalized = re.sub(r'=\s*[^,]+', '=DEFAULT', normalized)  # Normalize defaults
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        # Common patterns
        if "self, data" in normalized and "context" in normalized:
            return "self_data_context_pattern"
        elif "data" in normalized and "context" in normalized:
            return "data_context_pattern"
        elif "self" in normalized and len(normalized.split(',')) == 1:
            return "self_only_pattern"
        else:
            return f"custom_pattern: {normalized}"
    
    def generate_adapters(self) -> Dict[str, str]:
        """Generate adapter code snippets for common mismatches"""
        adapters = {}
        
        # Universal adapter
        adapters['universal_adapter'] = '''
class UniversalProcessAdapter:
    """Adapter to normalize all process method calls"""
    def __init__(self, wrapped_processor):
        self.wrapped = wrapped_processor
    
    def process(self, data=None, context=None):
        """Standardized process interface"""
        if hasattr(self.wrapped, 'process'):
            import inspect
            sig = inspect.signature(self.wrapped.process)
            params = list(sig.parameters.keys())
            
            if 'self' in params:
                params.remove('self')
            
            # Adapt call based on expected parameters
            if len(params) == 0:
                return self.wrapped.process()
            elif len(params) == 1:
                return self.wrapped.process(data)
            elif len(params) == 2:
                return self.wrapped.process(data, context)
            else:
                # Pass all as kwargs
                kwargs = {}
                if 'data' in params: kwargs['data'] = data
                if 'context' in params: kwargs['context'] = context
                return self.wrapped.process(**kwargs)
        else:
            raise AttributeError(f"Wrapped object has no process method")
'''
        
        # Registry adapter
        adapters['registry_system'] = '''
class ProcessorRegistry:
    """Registry for managing heterogeneous processors"""
    def __init__(self):
        self.processors = {}
        self.adapters = {}
    
    def register(self, name: str, processor, adapter_func=None):
        self.processors[name] = processor
        if adapter_func:
            self.adapters[name] = adapter_func
        else:
            self.adapters[name] = self._auto_adapt
    
    def _auto_adapt(self, processor, data=None, context=None):
        return UniversalProcessAdapter(processor).process(data, context)
    
    def process(self, processor_name: str, data=None, context=None):
        if processor_name not in self.processors:
            raise ValueError(f"Unknown processor: {processor_name}")
        
        processor = self.processors[processor_name]
        adapter = self.adapters[processor_name]
        return adapter(processor, data, context)
'''
        
        return adapters
    
    def create_report(self) -> str:
        """Create a formatted contract analysis report"""
        analysis = self.analyze_patterns()
        adapters = self.generate_adapters()
        
        report = []
        report.append("=" * 80)
        report.append("CANONICAL PIPELINE CONTRACT ANALYSIS REPORT")
        report.append("=" * 80)
        
        if 'error' in analysis:
            report.append(f"ERROR: {analysis['error']}")
            return '\n'.join(report)
        
        # Summary
        summary = analysis['summary']
        report.append(f"\nSCAN SUMMARY:")
        report.append(f"  Total Components: {summary['total_files']}")
        report.append(f"  Total process() methods: {summary['total_process_methods']}")
        report.append(f"  Parameter pattern variations: {summary['unique_parameter_patterns']}")
        report.append(f"  Return type variations: {summary['unique_return_types']}")
        
        # Parameter patterns
        report.append(f"\nPARAMETER PATTERNS:")
        for pattern, count in analysis['parameter_patterns'].items():
            report.append(f"  {pattern}: {count} methods")
        
        # Return type patterns
        report.append(f"\nRETURN TYPE PATTERNS:")
        for ret_type, count in analysis['return_type_patterns'].items():
            ret_type_display = ret_type if ret_type else "None/Unspecified"
            report.append(f"  {ret_type_display}: {count} methods")
        
        # Contract mismatches
        if analysis['mismatches']:
            report.append(f"\nCONTRACT MISMATCHES ({len(analysis['mismatches'])} found):")
            for i, mismatch in enumerate(analysis['mismatches'], 1):
                report.append(f"\n  {i}. {mismatch['type'].upper()}:")
                if 'expected_pattern' in mismatch:
                    report.append(f"     Expected: {mismatch['expected_pattern']}")
                    report.append(f"     Found: {mismatch['found_pattern']} ({mismatch['count']} methods)")
                    report.append(f"     Files: {', '.join(mismatch['files'][:3])}")
                    if len(mismatch['files']) > 3:
                        report.append(f"            ... and {len(mismatch['files']) - 3} more")
        else:
            report.append(f"\n✅ NO CONTRACT MISMATCHES DETECTED")
        
        # Sample signatures
        report.append(f"\nSAMPLE SIGNATURES:")
        for i, sig in enumerate(analysis['all_signatures'][:10], 1):
            report.append(f"  {i}. {sig['file']}:{sig['line']}")
            report.append(f"     {sig['signature']}")
        
        if len(analysis['all_signatures']) > 10:
            remaining = len(analysis['all_signatures']) - 10
            report.append(f"     ... and {remaining} more signatures")
        
        # Recommendations
        report.append(f"\nRECOMMENDATIONS:")
        if analysis['mismatches']:
            report.append("  1. CRITICAL: Implement universal adapter pattern")
            report.append("  2. HIGH: Standardize on unified interface")
            report.append("  3. MEDIUM: Create processor registry system")
            report.append("  4. LOW: Update documentation with contract specs")
        else:
            report.append("  ✅ All process() methods follow consistent contract")
            report.append("  📋 Consider documenting the standard interface")
        
        # Adapter solutions
        report.append(f"\nADAPTER SOLUTIONS:")
        report.append("  See generated adapter code in contract_adapters.py")
        
        report.append("\n" + "=" * 80)
        
        return '\n'.join(report)

def main():
    """Main entry point"""
    scanner = MinimalContractScanner()
    
    print("🔍 Scanning for process() method signatures...")
    scanner.scan_files()
    
    print(f"📊 Found {len(scanner.process_signatures)} process() methods")
    
    # Generate analysis
    analysis = scanner.analyze_patterns()
    
    # Save detailed JSON report
    with open('contract_analysis_detailed.json', 'w') as f:
        json.dump(analysis, f, indent=2)
    
    # Generate and save adapters
    adapters = scanner.generate_adapters()
    with open('contract_adapters.py', 'w') as f:
        f.write("# Generated Contract Adapters\n\n")
        for name, code in adapters.items():
            f.write(f"# {name.upper().replace('_', ' ')}\n")
            f.write(code)
            f.write("\n\n")
    
    # Generate formatted report
    report = scanner.create_report()
    
    # Save report
    with open('CONTRACT_ANALYSIS_REPORT.md', 'w') as f:
        f.write(report)
    
    # Print to console
    print(report)
    
    print(f"\n📁 Reports saved:")
    print(f"  - CONTRACT_ANALYSIS_REPORT.md (formatted)")
    print(f"  - contract_analysis_detailed.json (raw data)")
    print(f"  - contract_adapters.py (adapter solutions)")
    
    return len(analysis.get('mismatches', [])) == 0

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
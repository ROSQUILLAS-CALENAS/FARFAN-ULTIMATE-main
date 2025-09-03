#!/usr/bin/env python3
"""Run contract analysis for canonical pipeline components"""

import sys
# # # from pathlib import Path  # Module not found  # Module not found  # Module not found
# # # from contract_analysis_scanner import ContractAnalyzer  # Module not found  # Module not found  # Module not found

def main():
    print("🔍 Starting contract analysis...")
    
    # Initialize analyzer
    analyzer = ContractAnalyzer(Path('.'))
    
    # Scan for process methods
    analyzer.scan_components()
    print(f"📊 Found {len(analyzer.signatures)} process() methods")
    
    if analyzer.signatures:
        print("\nProcess methods found in:")
        for sig in analyzer.signatures[:10]:  # Show first 10
            print(f"  - {sig.file_path}:{sig.line_number}")
        if len(analyzer.signatures) > 10:
            print(f"  ... and {len(analyzer.signatures) - 10} more")
    
    # Analyze contracts
    analyzer.analyze_contracts()
    print(f"\n⚠️  Found {len(analyzer.mismatches)} contract mismatches")
    
    # Generate report
    report = analyzer.generate_report()
    
    # Save report
    import json
    with open('contract_analysis_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    
    # Print key findings
    print("\n" + "="*60)
    print("CONTRACT ANALYSIS RESULTS")
    print("="*60)
    
    summary = report['scan_summary']
    print(f"Components Scanned: {summary['total_components_scanned']}")
    print(f"Process Methods: {summary['total_process_methods']}")
    print(f"Contract Mismatches: {summary['total_mismatches']}")
    
    if summary['severity_breakdown']:
        print("\nSeverity Breakdown:")
        for level, count in summary['severity_breakdown'].items():
            if count > 0:
                print(f"  {level}: {count}")
    
    if report['contract_mismatches']:
        print(f"\nTop Issues:")
        for i, issue in enumerate(report['contract_mismatches'][:3], 1):
            print(f"  {i}. [{issue['severity']}] {issue['description']}")
            print(f"     File: {issue['primary_location']['file']}")
    
    print(f"\n📄 Full report saved to: contract_analysis_report.json")
    
    return len(analyzer.mismatches) == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
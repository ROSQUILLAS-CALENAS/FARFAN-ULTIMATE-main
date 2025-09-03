#!/usr/bin/env python3

from repository_scanner import CanonicalPipelineScanner

def main():
    scanner = CanonicalPipelineScanner()
    components = scanner.scan_repository()
    scanner.generate_inventory_jsonl()
    summary = scanner.generate_summary_report()
    
    print("\n=== CANONICAL PIPELINE COMPONENT INVENTORY SUMMARY ===")
    print(f"Total Components Discovered: {summary['total_components']}")
    print(f"Canonical Flow Components: {summary['canonical_flow_components']}")
    print(f"External Components: {summary['external_components']}")
    print(f"High Confidence Canonical: {summary['high_confidence_canonical']}")
    print(f"Canonical Process Functions: {summary['canonical_process_function_count']}")
    print(f"Pipeline Classes: {summary['pipeline_classes_count']}")
    print(f"Mathematical Enhancers: {summary['mathematical_enhancers_count']}")
    
    print("\nPhase Distribution:")
    for phase, count in summary['phase_distribution'].items():
        print(f"  {phase}: {count}")
    
    print("\nStatus Distribution:")
    for status, count in summary['status_distribution'].items():
        print(f"  {status}: {count}")
        
    print("\nConfidence Distribution:")
    for level, count in summary['confidence_distribution'].items():
        print(f"  {level.capitalize()}: {count}")
        
    print("\nTop Evidence Patterns:")
    for pattern, count in summary['top_evidence_patterns'].items():
        print(f"  {pattern}: {count}")

if __name__ == "__main__":
    main()
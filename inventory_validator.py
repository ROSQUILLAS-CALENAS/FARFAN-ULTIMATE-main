#!/usr/bin/env python3
"""
Inventory Validator - Validates INVENTORY.jsonl completeness and accuracy
"""

import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any

def load_inventory(inventory_path: str = "INVENTORY.jsonl") -> List[Dict]:
    """Load inventory records"""
    records = []
    try:
        with open(inventory_path, 'r') as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line.strip()))
        return records
    except FileNotFoundError:
        print(f"Inventory file {inventory_path} not found")
        return []

def validate_inventory_completeness():
    """Validate inventory completeness"""
    records = load_inventory()
    
    print("=== INVENTORY VALIDATION REPORT ===")
    print(f"Total Records: {len(records)}")
    
    # Phase analysis
    phase_counts = defaultdict(int)
    status_counts = defaultdict(int) 
    confidence_ranges = {"0.9+": 0, "0.8-0.89": 0, "0.6-0.79": 0, "<0.6": 0}
    evidence_stats = defaultdict(int)
    
    canonical_flow_files = []
    external_files = []
    high_confidence_canonical = []
    
    for record in records:
        # Phase distribution
        phase_counts[record['phase_assignment']] += 1
        
        # Status distribution
        status_counts[record['status_classification']] += 1
        
        # Confidence ranges
        conf = record['confidence_score']
        if conf >= 0.9:
            confidence_ranges["0.9+"] += 1
        elif conf >= 0.8:
            confidence_ranges["0.8-0.89"] += 1
        elif conf >= 0.6:
            confidence_ranges["0.6-0.79"] += 1
        else:
            confidence_ranges["<0.6"] += 1
            
        # Evidence patterns
        for pattern in record['evidence_patterns']:
            evidence_stats[pattern] += 1
            
        # File categorization
        file_path = record['file_path']
        if 'canonical_flow' in file_path:
            canonical_flow_files.append(file_path)
        else:
            external_files.append(file_path)
            
        # High confidence canonical
        if (record['confidence_score'] >= 0.9 and 
            record['status_classification'] == 'canonical_confirmed'):
            high_confidence_canonical.append(file_path)
    
    print(f"\nCanonical Flow Directory Components: {len(canonical_flow_files)}")
    print(f"External Components: {len(external_files)}")
    print(f"High Confidence Canonical: {len(high_confidence_canonical)}")
    
    print(f"\nPhase Distribution:")
    for phase, count in sorted(phase_counts.items()):
        print(f"  {phase}: {count}")
    
    print(f"\nStatus Distribution:")
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count}")
    
    print(f"\nConfidence Distribution:")
    for range_name, count in confidence_ranges.items():
        print(f"  {range_name}: {count}")
    
    print(f"\nTop 10 Evidence Patterns:")
    top_evidence = sorted(evidence_stats.items(), key=lambda x: x[1], reverse=True)[:10]
    for pattern, count in top_evidence:
        print(f"  {pattern}: {count}")
    
    # Canonical process function analysis
    process_function_files = [r['file_path'] for r in records 
                             if 'canonical_process_function' in r['evidence_patterns']]
    print(f"\nFiles with canonical process() function: {len(process_function_files)}")
    
    # Pipeline class analysis
    pipeline_class_files = [r['file_path'] for r in records 
                           if any(p in r['evidence_patterns'] for p in 
                                 ['pipeline_class', 'orchestrator_class', 'processor_class'])]
    print(f"Files with pipeline classes: {len(pipeline_class_files)}")
    
    # Mathematical enhancers analysis
    math_enhancer_files = [r['file_path'] for r in records 
                          if 'mathematical_enhancers' in r['file_path']]
    print(f"Mathematical enhancer components: {len(math_enhancer_files)}")
    
    print("\n=== SAMPLE HIGH-CONFIDENCE CANONICAL COMPONENTS ===")
    for i, file_path in enumerate(high_confidence_canonical[:10]):
        record = next(r for r in records if r['file_path'] == file_path)
        print(f"{i+1:2}. {file_path}")
        print(f"    Phase: {record['phase_assignment']}")
        print(f"    Confidence: {record['confidence_score']:.3f}")
        print(f"    Evidence: {', '.join(record['evidence_patterns'][:3])}")
        print()
    
    return records

def generate_phase_breakdown():
    """Generate detailed phase breakdown"""
    records = load_inventory()
    
    print("\n=== DETAILED PHASE BREAKDOWN ===")
    
    phase_files = defaultdict(list)
    for record in records:
        phase_files[record['phase_assignment']].append(record)
    
    for phase, files in sorted(phase_files.items()):
        print(f"\n{phase} ({len(files)} components):")
        
        # Show high-confidence files in this phase
        high_conf_files = [f for f in files if f['confidence_score'] >= 0.8]
        if high_conf_files:
            print(f"  High Confidence ({len(high_conf_files)}):")
            for f in sorted(high_conf_files, key=lambda x: -x['confidence_score'])[:5]:
                print(f"    {f['file_path']} (conf: {f['confidence_score']:.2f})")
        
        # Show canonical flow vs external split
        canonical_files = [f for f in files if 'canonical_flow' in f['file_path']]
        external_files = [f for f in files if 'canonical_flow' not in f['file_path']]
        print(f"  Canonical Flow: {len(canonical_files)}, External: {len(external_files)}")

if __name__ == "__main__":
    validate_inventory_completeness()
    generate_phase_breakdown()
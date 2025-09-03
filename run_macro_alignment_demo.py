#!/usr/bin/env python3
"""
Comprehensive demonstration of the Macro Alignment System
"""

import sys
import json
# # # from pathlib import Path  # Module not found  # Module not found  # Module not found

# Add current directory to path
sys.path.insert(0, '.')

# # # from canonical_flow.analysis.macro_alignment import MacroAggregationSystem, create_sample_data  # Module not found  # Module not found  # Module not found

def main():
    """Run comprehensive macro alignment demonstration"""
    
    print("=" * 70)
    print("MACRO AGGREGATION SYSTEM FOR DECÁLOGO COMPLIANCE ANALYSIS")
    print("=" * 70)
    
    # Initialize system with custom weights
    custom_weights = {
        "P1": 0.18,  # Vida y seguridad - increased weight
        "P2": 0.12,  # Dignidad humana  
        "P3": 0.08,  # Igualdad
        "P4": 0.06,  # Participación
        "P5": 0.15,  # Servicios básicos - increased
        "P6": 0.10,  # Protección ambiental
        "P7": 0.09,  # Desarrollo económico
        "P8": 0.05,  # Derechos culturales
        "P9": 0.11,  # Acceso a justicia
        "P10": 0.06  # Transparencia
    }
    
    system = MacroAggregationSystem(weights=custom_weights)
    print("✓ Initialized MacroAggregationSystem with custom weights")
    
    # Generate sample data
    sample_data = create_sample_data()
    print(f"✓ Generated sample data for {len(sample_data)} points")
    
    print("\nInput Point Scores:")
    print("-" * 50)
    for point_id, data in sample_data.items():
        print(f"  {point_id}: {data['name']} = {data['score']:.3f} (evidence: {data['evidence_count']})")
    
    # Compute macro alignment
    print("\n" + "=" * 50)
    print("COMPUTING MACRO ALIGNMENT...")
    print("=" * 50)
    
    result = system.compute_macro_alignment(sample_data)
    
    # Display results
    print(f"\n📊 GLOBAL DECÁLOGO SCORE: {result.global_decalogo_score:.3f}")
    print(f"📈 COMPLIANCE LEVEL: {result.compliance_level.value}")
    
    print(f"\n🔍 WEIGHTED CONTRIBUTIONS:")
    print("-" * 30)
    for point_id, weight_score in sorted(result.weighted_scores.items()):
        original_score = result.point_scores[point_id].score
        weight = custom_weights[point_id]
        print(f"  {point_id}: {original_score:.3f} × {weight:.3f} = {weight_score:.3f}")
    
    print(f"\n🏆 CLUSTER ANALYSIS:")
    print("-" * 30)
    cluster_names = {
        1: "Paz, Seguridad y Protección",
        2: "Derechos Sociales Fundamentales", 
        3: "Igualdad y No Discriminación",
        4: "Derechos Territoriales y Ambientales"
    }
    
    for cluster_id, analysis in result.cluster_analysis.items():
        name = cluster_names.get(cluster_id, f"Cluster {cluster_id}")
        print(f"  {name}:")
        print(f"    Average Score: {analysis['average_score']:.3f}")
        print(f"    Score Range: {analysis['min_score']:.3f} - {analysis['max_score']:.3f}")
        print(f"    Completion: {analysis['completion_rate']:.1%}")
        print(f"    Contribution: {analysis['weighted_contribution']:.3f}")
        print()
    
    if result.systemic_contradictions:
        print(f"⚠️  SYSTEMIC CONTRADICTIONS DETECTED: {len(result.systemic_contradictions)}")
        print("-" * 40)
        for contradiction in result.systemic_contradictions:
            print(f"  • {contradiction.contradiction_type.value.upper()}")
            print(f"    Severity: {contradiction.severity}")
            print(f"    Affected Points: {', '.join(contradiction.affected_points)}")
            print(f"    Description: {contradiction.description}")
            print()
    else:
        print("✅ No systemic contradictions detected")
    
    print(f"🎯 TOP PRIORITY ACTIONS:")
    print("-" * 30)
    for rec in result.action_recommendations:
        print(f"  {rec.priority_rank}. {rec.point_name} ({rec.point_id})")
        print(f"     Current Score: {rec.current_score:.3f}")
        print(f"     Potential Impact: {rec.estimated_impact:.3f}")
        print(f"     Complexity: {rec.implementation_complexity}")
        print(f"     Actions: {len(rec.required_actions)} required")
        for i, action in enumerate(rec.required_actions[:2], 1):
            print(f"       {i}. {action}")
        if len(rec.required_actions) > 2:
            print(f"       ... and {len(rec.required_actions) - 2} more")
        print()
    
    # Save results
    output_path = system.save_results(result)
    print(f"💾 Results saved to: {output_path}")
    
    # Display audit trail summary
    audit_events = result.audit_trail['events']
    print(f"\n📋 AUDIT TRAIL: {len(audit_events)} events logged")
    print("-" * 30)
    for event in audit_events[-3:]:  # Show last 3 events
        print(f"  • {event['event_type']}: {event['timestamp']}")
    
    print(f"\n⏱️  Processing completed in {result.processing_metadata['processing_duration_seconds']:.2f} seconds")
    print(f"🔢 Processing ID: {result.processing_metadata['processing_id']}")
    
    print("\n" + "=" * 70)
    print("MACRO ALIGNMENT ANALYSIS COMPLETE")
    print("=" * 70)
    
    return output_path

if __name__ == "__main__":
    output_path = main()
    print(f"\nFor detailed results, see: {output_path}")
"""
Demo script for the Auto-Enhancement Orchestration System
========================================================

Demonstrates comprehensive preflight validation, auto-deactivation monitoring,
and provenance tracking capabilities.
"""

import json
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path

from orchestrator import EnhancementOrchestrator, OrchestrationConfig, OrchestrationMode
from preflight_validator import PreflightValidator
from auto_deactivation_monitor import AutoDeactivationMonitor
from provenance_tracker import ProvenanceTracker, ActivationCriteriaType


def setup_logging():
    """Setup logging for demonstration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('orchestration_demo.log'),
            logging.StreamHandler()
        ]
    )


def demo_preflight_validation():
    """Demonstrate preflight validation capabilities"""
    print("\n" + "="*60)
    print("PREFLIGHT VALIDATION DEMONSTRATION")
    print("="*60)
    
    validator = PreflightValidator()
    
    # Test 1: Valid enhancement request
    print("\n1. Testing valid enhancement request...")
    valid_request = {
        "enhancement_id": "adaptive_scoring_v1",
        "enhancement_type": "adaptive_scoring",
        "configuration": {
            "learning_rate": 0.01,
            "adaptation_window": 100,
            "stability_threshold": 0.85
        },
        "priority": "high",
        "activation_criteria": {
            "performance_threshold": 0.8,
            "stability_requirement": 0.85
        },
        "metadata": {
            "dependencies": ["base_scoring"],
            "tags": ["ml", "adaptive"]
        }
    }
    
    current_metrics = {
        "mandatory_compliance": 0.95,
        "proxy_score": 0.85,
        "confidence_alpha": 0.9,
        "sigma_presence": 0.1,
        "governance_completeness": 0.88
    }
    
    validation_results = validator.run_comprehensive_validation(
        input_data=valid_request,
        schema_type="enhancement_request",
        current_metrics=current_metrics
    )
    
    summary = validator.get_validation_summary(validation_results)
    print(f"✓ Validation passed: {summary['overall_passed']}")
    print(f"✓ Overall score: {summary['overall_score']:.3f}")
    print(f"✓ Checks passed: {summary['checks_passed']}/{summary['total_checks']}")
    
    # Test 2: Invalid enhancement request
    print("\n2. Testing invalid enhancement request...")
    invalid_request = {
        "enhancement_id": "invalid_enhancement",
        "enhancement_type": "unknown_type",  # Invalid type
        "configuration": {},  # Missing required config
        # Missing required fields
    }
    
    validation_results = validator.run_comprehensive_validation(
        input_data=invalid_request,
        schema_type="enhancement_request",
        current_metrics=current_metrics
    )
    
    summary = validator.get_validation_summary(validation_results)
    print(f"✗ Validation passed: {summary['overall_passed']}")
    print(f"✗ Errors found: {len(summary['errors'])}")
    for error in summary['errors'][:3]:  # Show first 3 errors
        print(f"  - {error}")


def demo_auto_deactivation_monitoring():
    """Demonstrate auto-deactivation monitoring capabilities"""
    print("\n" + "="*60)
    print("AUTO-DEACTIVATION MONITORING DEMONSTRATION")
    print("="*60)
    
    monitor = AutoDeactivationMonitor()
    
    enhancement_id = "test_enhancement_monitor"
    
    print(f"\n1. Monitoring enhancement: {enhancement_id}")
    
    # Simulate good performance initially
    print("   - Recording stable performance metrics...")
    for i in range(5):
        monitor.monitor_enhancement(
            enhancement_id=enhancement_id,
            performance_metrics={
                "response_time": 0.5 + (i * 0.01),
                "accuracy": 0.85,
                "throughput": 100.0,
                "error_rate": 0.01
            },
            evidence_quality={
                "overall_quality": 0.85,
                "consistency": 0.9,
                "coverage": 0.88,
                "coherence": 0.87
            },
            score=0.85
        )
    
    status = monitor.get_enhancement_status(enhancement_id)
    print(f"   ✓ Enhancement active: {status['is_active']}")
    
    # Simulate performance degradation
    print("\n2. Simulating performance degradation...")
    degradation_results = []
    
    for i in range(4):
        result = monitor.monitor_enhancement(
            enhancement_id=enhancement_id,
            performance_metrics={
                "response_time": 0.8 + (i * 0.3),  # Increasing response time
                "accuracy": 0.85 - (i * 0.05),    # Decreasing accuracy
                "throughput": 100.0 - (i * 10),   # Decreasing throughput
                "error_rate": 0.01 + (i * 0.01)   # Increasing error rate
            },
            evidence_quality={
                "overall_quality": 0.85 - (i * 0.1),  # Degrading quality
                "consistency": 0.9 - (i * 0.05),
                "coverage": 0.88 - (i * 0.03),
                "coherence": 0.87 - (i * 0.04)
            },
            score=0.85 - (i * 0.1)  # Decreasing score
        )
        
        degradation_results.append(result)
        print(f"   Cycle {i+1}: Deactivation triggers = {result['deactivation_triggers']}")
        
        if result['deactivation_decision']['should_deactivate']:
            print(f"   🚨 DEACTIVATION TRIGGERED: {result['deactivation_decision']['reason']}")
            break
    
    # Show monitoring summary
    summary = monitor.get_monitoring_summary()
    print(f"\n3. Monitoring Summary:")
    print(f"   - Total deactivation events: {summary['total_deactivation_events']}")
    print(f"   - Active cooldowns: {summary['active_cooldowns']}")
    print(f"   - Recent deactivations (24h): {summary['recent_deactivations']}")


def demo_provenance_tracking():
    """Demonstrate provenance tracking capabilities"""
    print("\n" + "="*60)
    print("PROVENANCE TRACKING DEMONSTRATION")
    print("="*60)
    
    tracker = ProvenanceTracker()
    
    # Create enhancement metadata
    print("\n1. Creating enhancement metadata...")
    enhancement_id = "demo_adaptive_enhancement"
    
    activation_criteria = [
        {
            "type": "performance_threshold",
            "description": "Minimum performance score requirement",
            "threshold": 0.8
        },
        {
            "type": "stability_requirement", 
            "description": "System stability requirement",
            "threshold": 0.85
        },
        {
            "type": "safety_margin",
            "description": "Safety margin requirement",
            "threshold": 0.95
        }
    ]
    
    baseline_metrics = {
        "response_time": 0.5,
        "accuracy": 0.82,
        "throughput": 100.0,
        "error_rate": 0.015
    }
    
    metadata = tracker.create_enhancement_metadata(
        enhancement_id=enhancement_id,
        enhancement_type="adaptive_scoring",
        description="Adaptive scoring enhancement with real-time learning",
        configuration={
            "algorithm": "gradient_descent",
            "learning_rate": 0.01,
            "batch_size": 32
        },
        activation_criteria=activation_criteria,
        baseline_metrics=baseline_metrics,
        dependencies=["base_scoring", "metrics_collector"],
        tags=["ml", "adaptive", "real-time"]
    )
    
    print(f"   ✓ Created metadata for {enhancement_id}")
    print(f"   ✓ Activation criteria: {len(metadata.activation_criteria)}")
    
    # Evaluate activation criteria
    print("\n2. Evaluating activation criteria...")
    current_metrics = {
        "performance_score": 0.85,
        "stability_score": 0.88,
        "safety_score": 0.96,
        "confidence_score": 0.9
    }
    
    decision = tracker.evaluate_activation_criteria(enhancement_id, current_metrics)
    print(f"   ✓ Should activate: {decision['should_activate']}")
    print(f"   ✓ Satisfaction score: {decision['satisfaction_score']:.3f}")
    print(f"   ✓ Satisfied criteria: {decision['satisfied_criteria']}/{decision['total_criteria']}")
    
    # Record activation
    if decision['should_activate']:
        print("\n3. Recording activation...")
        success = tracker.record_activation(enhancement_id, {
            "activation_reason": "criteria_satisfied",
            "system_load": 0.6
        })
        print(f"   ✓ Activation recorded: {success}")
        
        # Record performance impact
        print("\n4. Recording performance impacts...")
        for i in range(3):
            performance_metrics = {
                "response_time": baseline_metrics["response_time"] - (i * 0.05),
                "accuracy": baseline_metrics["accuracy"] + (i * 0.02),
                "throughput": baseline_metrics["throughput"] + (i * 5),
                "error_rate": baseline_metrics["error_rate"] - (i * 0.002)
            }
            
            tracker.record_performance_impact(enhancement_id, performance_metrics)
            time.sleep(0.1)  # Small delay to show time progression
            
        print(f"   ✓ Recorded 3 performance impact measurements")
        
        # Generate enhancement report
        print("\n5. Generating enhancement report...")
        report = tracker.generate_enhancement_report(enhancement_id)
        
        print(f"   ✓ Current state: {report['current_state']}")
        print(f"   ✓ Total lifecycle events: {report['lifecycle_summary']['total_lifecycle_events']}")
        print(f"   ✓ Criteria satisfaction: {report['activation_analysis']['criteria_satisfaction_score']:.3f}")
        print(f"   ✓ Performance measurements: {report['performance_analysis']['total_measurements']}")


def demo_full_orchestration():
    """Demonstrate full orchestration system"""
    print("\n" + "="*60)
    print("FULL ORCHESTRATION SYSTEM DEMONSTRATION")
    print("="*60)
    
    # Initialize orchestrator
    config = OrchestrationConfig(
        mode=OrchestrationMode.AUTOMATIC,
        monitoring_interval_seconds=5,
        max_concurrent_enhancements=3
    )
    
    orchestrator = EnhancementOrchestrator(config)
    
    # Submit enhancement requests
    print("\n1. Submitting enhancement requests...")
    
    enhancements = [
        {
            "enhancement_id": "adaptive_scoring_v2",
            "enhancement_type": "adaptive_scoring",
            "description": "Advanced adaptive scoring with neural networks",
            "configuration": {
                "model_type": "neural_network",
                "hidden_layers": [64, 32, 16],
                "activation": "relu",
                "learning_rate": 0.001
            },
            "activation_criteria": [
                {"type": "performance_threshold", "description": "Performance requirement", "threshold": 0.85},
                {"type": "stability_requirement", "description": "Stability requirement", "threshold": 0.9}
            ],
            "baseline_metrics": {"accuracy": 0.82, "response_time": 0.3},
            "priority": "high"
        },
        {
            "enhancement_id": "dynamic_thresholding_v1",
            "enhancement_type": "dynamic_thresholding",
            "description": "Dynamic threshold adjustment based on data patterns",
            "configuration": {
                "adjustment_algorithm": "statistical",
                "window_size": 100,
                "sensitivity": 0.1
            },
            "activation_criteria": [
                {"type": "performance_threshold", "description": "Performance requirement", "threshold": 0.75},
                {"type": "confidence_level", "description": "Confidence requirement", "threshold": 0.85}
            ],
            "baseline_metrics": {"precision": 0.8, "recall": 0.78},
            "priority": "medium"
        }
    ]
    
    for enhancement in enhancements:
        result = orchestrator.submit_enhancement_request(**enhancement)
        print(f"   ✓ {enhancement['enhancement_id']}: {result['status']}")
        if 'immediate_activation' in result:
            print(f"     - Immediate activation: {result['immediate_activation']}")
    
    # Start monitoring
    print("\n2. Starting continuous monitoring...")
    orchestrator.start_monitoring()
    
    # Let it run for a short time
    print("   - Monitoring active enhancements...")
    time.sleep(10)
    
    # Get orchestration status
    print("\n3. Orchestration status:")
    status = orchestrator.get_orchestration_status()
    print(f"   ✓ Active enhancements: {status['current_state']['active_enhancements']}")
    print(f"   ✓ Pending enhancements: {status['current_state']['pending_enhancements']}")
    print(f"   ✓ Monitoring active: {status['current_state']['monitoring_active']}")
    
    # Generate comprehensive report
    print("\n4. Generating orchestration report...")
    report = orchestrator.generate_orchestration_report()
    
    print(f"   ✓ System metrics recorded: {len(report['system_metrics'])}")
    print(f"   ✓ Enhancement details: {len(report['enhancement_details'])}")
    print(f"   ✓ Recent activities: {len(report['recent_activities'])}")
    print(f"   ✓ Recommendations: {len(report['recommendations'])}")
    
    # Export enhancement metadata
    print("\n5. Exporting enhancement metadata...")
    export_result = orchestrator.export_enhancement_metadata()
    
    if export_result["success"]:
        print(f"   ✓ Metadata exported to: {export_result['output_path']}")
        print(f"   ✓ Enhancements exported: {export_result['enhancements_exported']}")
        print(f"   ✓ File size: {export_result['file_size_bytes']} bytes")
    else:
        print(f"   ✗ Export failed: {export_result['error']}")
    
    # Stop monitoring and shutdown
    print("\n6. Shutting down orchestrator...")
    orchestrator.shutdown()
    print("   ✓ Orchestrator shutdown complete")


def main():
    """Main demonstration function"""
    setup_logging()
    
    print("AUTO-ENHANCEMENT ORCHESTRATION SYSTEM DEMO")
    print("=" * 60)
    
    try:
        # Individual component demos
        demo_preflight_validation()
        demo_auto_deactivation_monitoring()
        demo_provenance_tracking()
        
        # Full system demo
        demo_full_orchestration()
        
        print("\n" + "="*60)
        print("DEMONSTRATION COMPLETE")
        print("="*60)
        print("✓ All components demonstrated successfully")
        print("✓ Check orchestration_demo.log for detailed logs")
        print("✓ Check calibration_safety_governance/metadata/ for enhancement metadata")
        print("✓ Check calibration_safety_governance/enhancement_metadata.json for exported data")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        logging.exception("Demo failed")


if __name__ == "__main__":
    main()
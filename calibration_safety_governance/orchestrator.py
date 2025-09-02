"""
Enhancement Orchestration System
================================

Main coordination system that integrates preflight validation, auto-deactivation monitoring,
and provenance tracking for comprehensive enhancement lifecycle management.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
from enum import Enum
import threading
import time

from .preflight_validator import PreflightValidator, ValidationResult
from .auto_deactivation_monitor import AutoDeactivationMonitor
from .provenance_tracker import ProvenanceTracker, ActivationCriteriaType


class OrchestrationMode(Enum):
    """Orchestration operation modes"""
    AUTOMATIC = "automatic"
    SEMI_AUTOMATIC = "semi_automatic"
    MANUAL = "manual"


@dataclass
class OrchestrationConfig:
    """Orchestration system configuration"""
    mode: OrchestrationMode
    monitoring_interval_seconds: int = 30
    preflight_validation_enabled: bool = True
    auto_deactivation_enabled: bool = True
    provenance_tracking_enabled: bool = True
    max_concurrent_enhancements: int = 5
    safety_override_enabled: bool = True


class EnhancementOrchestrator:
    """Main enhancement orchestration system"""
    
    def __init__(
        self,
        config: OrchestrationConfig = None,
        thresholds_path: str = "calibration_safety_governance/thresholds.json"
    ):
        self.logger = logging.getLogger(__name__)
        self.config = config or OrchestrationConfig(OrchestrationMode.AUTOMATIC)
        
        # Initialize subsystems
        self.preflight_validator = PreflightValidator(thresholds_path)
        self.monitor = AutoDeactivationMonitor(thresholds_path)
        self.provenance = ProvenanceTracker()
        
        # Orchestration state
        self.active_enhancements: Set[str] = set()
        self.pending_enhancements: Dict[str, Dict[str, Any]] = {}
        self.failed_validations: Dict[str, List[ValidationResult]] = {}
        
        # Monitoring control
        self._monitoring_thread = None
        self._monitoring_active = False
        self._shutdown_event = threading.Event()
        
        self.logger.info(f"Enhancement orchestrator initialized in {self.config.mode.value} mode")

    def submit_enhancement_request(
        self,
        enhancement_id: str,
        enhancement_type: str,
        description: str,
        configuration: Dict[str, Any],
        activation_criteria: List[Dict[str, Any]],
        baseline_metrics: Dict[str, float],
        priority: str = "medium",
        dependencies: List[str] = None,
        tags: List[str] = None
    ) -> Dict[str, Any]:
        """Submit new enhancement request for orchestration"""
        
        self.logger.info(f"Received enhancement request: {enhancement_id}")
        
        try:
            # Create provenance metadata
            if self.config.provenance_tracking_enabled:
                metadata = self.provenance.create_enhancement_metadata(
                    enhancement_id=enhancement_id,
                    enhancement_type=enhancement_type,
                    description=description,
                    configuration=configuration,
                    activation_criteria=activation_criteria,
                    baseline_metrics=baseline_metrics,
                    dependencies=dependencies or [],
                    tags=tags or []
                )
            
            # Store pending request
            request_data = {
                "enhancement_id": enhancement_id,
                "enhancement_type": enhancement_type,
                "description": description,
                "configuration": configuration,
                "activation_criteria": activation_criteria,
                "baseline_metrics": baseline_metrics,
                "priority": priority,
                "dependencies": dependencies or [],
                "tags": tags or [],
                "submitted_at": datetime.now(),
                "validation_attempts": 0
            }
            
            self.pending_enhancements[enhancement_id] = request_data
            
            # Immediate validation if in automatic mode
            if self.config.mode == OrchestrationMode.AUTOMATIC:
                validation_result = self._run_preflight_validation(request_data)
                
                if validation_result["overall_passed"]:
                    activation_result = self._attempt_activation(enhancement_id)
                    return {
                        "status": "submitted",
                        "immediate_activation": activation_result.get("activated", False),
                        "validation_result": validation_result
                    }
                else:
                    return {
                        "status": "validation_failed",
                        "validation_result": validation_result
                    }
            else:
                return {
                    "status": "submitted",
                    "message": f"Enhancement queued for {self.config.mode.value} processing"
                }
                
        except Exception as e:
            self.logger.error(f"Failed to submit enhancement request {enhancement_id}: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

    def _run_preflight_validation(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run comprehensive preflight validation"""
        
        if not self.config.preflight_validation_enabled:
            return {"overall_passed": True, "message": "Preflight validation disabled"}
            
        enhancement_id = request_data["enhancement_id"]
        self.logger.info(f"Running preflight validation for {enhancement_id}")
        
        try:
            # Prepare validation input
            input_data = {
                "enhancement_id": enhancement_id,
                "enhancement_type": request_data["enhancement_type"],
                "configuration": request_data["configuration"],
                "priority": request_data["priority"],
                "activation_criteria": request_data["activation_criteria"],
                "metadata": {
                    "dependencies": request_data["dependencies"],
                    "tags": request_data["tags"]
                }
            }
            
            # Get current system metrics
            current_metrics = self._get_current_system_metrics()
            
            # Run validation
            validation_results = self.preflight_validator.run_comprehensive_validation(
                input_data=input_data,
                schema_type="enhancement_request",
                current_metrics=current_metrics
            )
            
            # Generate summary
            summary = self.preflight_validator.get_validation_summary(validation_results)
            
            # Store validation results
            if enhancement_id not in self.failed_validations:
                self.failed_validations[enhancement_id] = []
                
            if not summary["overall_passed"]:
                self.failed_validations[enhancement_id].extend(validation_results.values())
                
            self.logger.info(f"Preflight validation for {enhancement_id}: {summary['overall_passed']}")
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Preflight validation failed for {enhancement_id}: {e}")
            return {
                "overall_passed": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def _get_current_system_metrics(self) -> Dict[str, float]:
        """Get current system metrics for validation"""
        # This would integrate with actual system monitoring
        # For now, return mock metrics
        return {
            "mandatory_compliance": 0.95,
            "proxy_score": 0.85,
            "confidence_alpha": 0.9,
            "sigma_presence": 0.1,
            "governance_completeness": 0.88,
            "performance_score": 0.82,
            "stability_score": 0.9,
            "safety_score": 0.95,
            "resource_availability": 0.8
        }

    def _attempt_activation(self, enhancement_id: str) -> Dict[str, Any]:
        """Attempt to activate enhancement"""
        
        if enhancement_id not in self.pending_enhancements:
            return {"activated": False, "reason": "enhancement_not_pending"}
            
        if len(self.active_enhancements) >= self.config.max_concurrent_enhancements:
            return {"activated": False, "reason": "max_concurrent_limit_reached"}
            
        try:
            # Check dependencies
            request_data = self.pending_enhancements[enhancement_id]
            dependencies = request_data.get("dependencies", [])
            
            unmet_dependencies = [dep for dep in dependencies if dep not in self.active_enhancements]
            if unmet_dependencies:
                return {
                    "activated": False,
                    "reason": "unmet_dependencies",
                    "unmet_dependencies": unmet_dependencies
                }
            
            # Evaluate activation criteria
            current_metrics = self._get_current_system_metrics()
            
            if self.config.provenance_tracking_enabled:
                activation_decision = self.provenance.evaluate_activation_criteria(
                    enhancement_id, current_metrics
                )
                
                if not activation_decision.get("should_activate", False):
                    return {
                        "activated": False,
                        "reason": "activation_criteria_not_met",
                        "activation_decision": activation_decision
                    }
            
            # Establish performance baseline
            baseline_metrics = request_data["baseline_metrics"]
            self.monitor.performance_detector.establish_baseline(enhancement_id, baseline_metrics)
            
            # Activate enhancement
            self.active_enhancements.add(enhancement_id)
            del self.pending_enhancements[enhancement_id]
            
            if self.config.provenance_tracking_enabled:
                self.provenance.record_activation(
                    enhancement_id,
                    {
                        "activation_mode": self.config.mode.value,
                        "system_metrics": current_metrics,
                        "concurrent_enhancements": len(self.active_enhancements)
                    }
                )
            
            self.logger.info(f"Enhancement {enhancement_id} activated successfully")
            
            return {
                "activated": True,
                "activation_timestamp": datetime.now().isoformat(),
                "concurrent_enhancements": len(self.active_enhancements)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to activate enhancement {enhancement_id}: {e}")
            return {
                "activated": False,
                "reason": "activation_error",
                "error": str(e)
            }

    def start_monitoring(self):
        """Start continuous monitoring of active enhancements"""
        if self._monitoring_active:
            self.logger.warning("Monitoring already active")
            return
            
        self._monitoring_active = True
        self._shutdown_event.clear()
        
        self._monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitoring_thread.start()
        
        self.logger.info("Enhancement monitoring started")

    def stop_monitoring(self):
        """Stop continuous monitoring"""
        if not self._monitoring_active:
            return
            
        self._monitoring_active = False
        self._shutdown_event.set()
        
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5)
            
        self.logger.info("Enhancement monitoring stopped")

    def _monitoring_loop(self):
        """Main monitoring loop"""
        self.logger.info("Monitoring loop started")
        
        while self._monitoring_active and not self._shutdown_event.is_set():
            try:
                if self.active_enhancements:
                    self._monitor_active_enhancements()
                    
                # Process pending enhancements in automatic mode
                if self.config.mode == OrchestrationMode.AUTOMATIC:
                    self._process_pending_enhancements()
                    
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                
            # Wait for next monitoring cycle
            self._shutdown_event.wait(self.config.monitoring_interval_seconds)
            
        self.logger.info("Monitoring loop stopped")

    def _monitor_active_enhancements(self):
        """Monitor all active enhancements for deactivation triggers"""
        
        if not self.config.auto_deactivation_enabled:
            return
            
        current_metrics = self._get_current_system_metrics()
        
        enhancements_to_deactivate = []
        
        for enhancement_id in self.active_enhancements.copy():
            try:
                # Simulate performance and evidence quality metrics
                # In real implementation, these would come from actual monitoring systems
                performance_metrics = {
                    "response_time": current_metrics.get("response_time", 0.5),
                    "accuracy": current_metrics.get("accuracy", 0.85),
                    "throughput": current_metrics.get("throughput", 100.0),
                    "error_rate": current_metrics.get("error_rate", 0.01)
                }
                
                evidence_quality = {
                    "overall_quality": current_metrics.get("quality_score", 0.8),
                    "consistency": current_metrics.get("consistency", 0.85),
                    "coverage": current_metrics.get("coverage", 0.9),
                    "coherence": current_metrics.get("coherence", 0.82)
                }
                
                score = current_metrics.get("overall_score", 0.8)
                
                # Monitor for deactivation triggers
                monitoring_result = self.monitor.monitor_enhancement(
                    enhancement_id, performance_metrics, evidence_quality, score
                )
                
                # Record performance impact if provenance tracking enabled
                if self.config.provenance_tracking_enabled:
                    self.provenance.record_performance_impact(enhancement_id, performance_metrics)
                
                # Check for deactivation
                if monitoring_result["deactivation_decision"]["should_deactivate"]:
                    enhancements_to_deactivate.append({
                        "enhancement_id": enhancement_id,
                        "reason": monitoring_result["deactivation_decision"]["reason"],
                        "trigger_type": monitoring_result["deactivation_decision"]["trigger_type"],
                        "metrics": performance_metrics
                    })
                    
            except Exception as e:
                self.logger.error(f"Error monitoring enhancement {enhancement_id}: {e}")
                
        # Deactivate enhancements that triggered deactivation conditions
        for deactivation in enhancements_to_deactivate:
            self._deactivate_enhancement(
                deactivation["enhancement_id"],
                deactivation["reason"],
                deactivation["metrics"],
                {
                    "trigger_type": deactivation["trigger_type"],
                    "monitoring_result": monitoring_result
                }
            )

    def _process_pending_enhancements(self):
        """Process pending enhancements for potential activation"""
        
        for enhancement_id in list(self.pending_enhancements.keys()):
            try:
                request_data = self.pending_enhancements[enhancement_id]
                
                # Check if maximum concurrent limit allows activation
                if len(self.active_enhancements) >= self.config.max_concurrent_enhancements:
                    break
                    
                # Retry validation if it previously failed
                if enhancement_id in self.failed_validations:
                    validation_result = self._run_preflight_validation(request_data)
                    
                    if not validation_result["overall_passed"]:
                        # Increment retry count and check if we should give up
                        request_data["validation_attempts"] += 1
                        if request_data["validation_attempts"] >= 3:
                            # Move to permanently failed
                            del self.pending_enhancements[enhancement_id]
                            self.logger.warning(f"Enhancement {enhancement_id} failed validation after 3 attempts")
                        continue
                        
                # Attempt activation
                activation_result = self._attempt_activation(enhancement_id)
                if not activation_result["activated"]:
                    self.logger.debug(f"Enhancement {enhancement_id} activation deferred: {activation_result['reason']}")
                    
            except Exception as e:
                self.logger.error(f"Error processing pending enhancement {enhancement_id}: {e}")

    def _deactivate_enhancement(
        self,
        enhancement_id: str,
        reason: str,
        triggering_metrics: Dict[str, float],
        context: Dict[str, Any]
    ):
        """Deactivate enhancement due to monitoring triggers"""
        
        if enhancement_id not in self.active_enhancements:
            self.logger.warning(f"Enhancement {enhancement_id} not active, cannot deactivate")
            return
            
        try:
            # Remove from active set
            self.active_enhancements.remove(enhancement_id)
            
            # Record deactivation in provenance
            if self.config.provenance_tracking_enabled:
                self.provenance.record_deactivation(
                    enhancement_id, reason, triggering_metrics, context
                )
            
            self.logger.warning(f"Enhancement {enhancement_id} deactivated: {reason}")
            
        except Exception as e:
            self.logger.error(f"Error deactivating enhancement {enhancement_id}: {e}")

    def get_orchestration_status(self) -> Dict[str, Any]:
        """Get comprehensive orchestration system status"""
        
        # Get monitoring summary
        monitoring_summary = {}
        if self.config.auto_deactivation_enabled:
            monitoring_summary = self.monitor.get_monitoring_summary()
            
        # Get provenance summary
        provenance_summary = {}
        if self.config.provenance_tracking_enabled:
            provenance_summary = self.provenance.get_system_provenance_summary()
        
        return {
            "orchestration_config": {
                "mode": self.config.mode.value,
                "monitoring_interval_seconds": self.config.monitoring_interval_seconds,
                "max_concurrent_enhancements": self.config.max_concurrent_enhancements,
                "subsystems_enabled": {
                    "preflight_validation": self.config.preflight_validation_enabled,
                    "auto_deactivation": self.config.auto_deactivation_enabled,
                    "provenance_tracking": self.config.provenance_tracking_enabled
                }
            },
            "current_state": {
                "active_enhancements": len(self.active_enhancements),
                "pending_enhancements": len(self.pending_enhancements),
                "failed_validations": len(self.failed_validations),
                "monitoring_active": self._monitoring_active,
                "active_enhancement_ids": list(self.active_enhancements),
                "pending_enhancement_ids": list(self.pending_enhancements.keys())
            },
            "monitoring_summary": monitoring_summary,
            "provenance_summary": provenance_summary,
            "status_timestamp": datetime.now().isoformat()
        }

    def generate_orchestration_report(self, include_detailed_logs: bool = False) -> Dict[str, Any]:
        """Generate comprehensive orchestration report"""
        
        status = self.get_orchestration_status()
        
        # Enhancement details
        enhancement_details = {}
        
        for enhancement_id in self.active_enhancements:
            if self.config.provenance_tracking_enabled:
                try:
                    enhancement_report = self.provenance.generate_enhancement_report(enhancement_id)
                    enhancement_details[enhancement_id] = enhancement_report
                except Exception as e:
                    self.logger.error(f"Failed to generate report for {enhancement_id}: {e}")
                    enhancement_details[enhancement_id] = {"error": str(e)}
        
        # System metrics
        current_metrics = self._get_current_system_metrics()
        
        report = {
            "report_metadata": {
                "generated_at": datetime.now().isoformat(),
                "orchestrator_version": "1.0.0",
                "report_scope": "comprehensive"
            },
            "orchestration_status": status,
            "system_metrics": current_metrics,
            "enhancement_details": enhancement_details,
            "recent_activities": self._get_recent_activities(),
            "recommendations": self._generate_recommendations()
        }
        
        if include_detailed_logs:
            report["detailed_logs"] = self._get_recent_logs()
            
        return report

    def _get_recent_activities(self) -> List[Dict[str, Any]]:
        """Get recent orchestration activities"""
        activities = []
        
        # Add recent activations
        for enhancement_id in self.active_enhancements:
            if self.config.provenance_tracking_enabled and enhancement_id in self.provenance.enhancement_metadata:
                metadata = self.provenance.enhancement_metadata[enhancement_id]
                if metadata.activated_at:
                    activities.append({
                        "type": "activation",
                        "enhancement_id": enhancement_id,
                        "timestamp": metadata.activated_at.isoformat(),
                        "details": f"Enhancement activated in {self.config.mode.value} mode"
                    })
        
        # Add recent deactivations
        if self.config.auto_deactivation_enabled:
            recent_deactivations = [
                event for event in self.monitor.deactivation_events
                if event.timestamp > datetime.now() - timedelta(hours=24)
            ]
            
            for event in recent_deactivations[-10:]:  # Last 10 deactivations
                activities.append({
                    "type": "deactivation",
                    "enhancement_id": event.enhancement_id,
                    "timestamp": event.timestamp.isoformat(),
                    "details": f"Deactivated due to {event.trigger_condition}",
                    "severity": event.severity.value
                })
        
        # Sort by timestamp
        activities.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return activities[:20]  # Return most recent 20 activities

    def _generate_recommendations(self) -> List[Dict[str, Any]]:
        """Generate system recommendations"""
        recommendations = []
        
        # Check for overloaded system
        if len(self.active_enhancements) >= self.config.max_concurrent_enhancements * 0.8:
            recommendations.append({
                "type": "capacity_warning",
                "priority": "medium",
                "message": "System approaching maximum concurrent enhancement limit",
                "suggestion": "Consider increasing max_concurrent_enhancements or optimizing active enhancements"
            })
        
        # Check for frequent deactivations
        if self.config.auto_deactivation_enabled:
            recent_deactivations = len([
                event for event in self.monitor.deactivation_events
                if event.timestamp > datetime.now() - timedelta(hours=1)
            ])
            
            if recent_deactivations >= 3:
                recommendations.append({
                    "type": "stability_warning",
                    "priority": "high",
                    "message": f"{recent_deactivations} deactivations in the last hour",
                    "suggestion": "Review system stability and consider adjusting thresholds"
                })
        
        # Check for pending enhancements
        if len(self.pending_enhancements) > 5:
            recommendations.append({
                "type": "queue_warning",
                "priority": "low",
                "message": f"{len(self.pending_enhancements)} enhancements pending activation",
                "suggestion": "Review activation criteria and system capacity"
            })
        
        return recommendations

    def _get_recent_logs(self) -> List[str]:
        """Get recent log entries (placeholder for actual log integration)"""
        # This would integrate with actual logging system
        return [
            "Enhancement monitoring cycle completed",
            "Preflight validation passed for enhancement_001",
            "Performance metrics collected for active enhancements"
        ]

    def export_enhancement_metadata(self, output_path: str = "calibration_safety_governance/enhancement_metadata.json") -> Dict[str, Any]:
        """Export all enhancement metadata to JSON artifact"""
        
        if not self.config.provenance_tracking_enabled:
            return {"error": "Provenance tracking not enabled"}
            
        try:
            # Generate comprehensive metadata export
            export_data = {
                "export_metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "orchestrator_version": "1.0.0",
                    "total_enhancements": len(self.provenance.enhancement_metadata),
                    "export_scope": "all_enhancements"
                },
                "system_summary": self.provenance.get_system_provenance_summary(),
                "enhancement_details": {},
                "audit_trail": self.provenance.export_audit_trail()
            }
            
            # Add detailed reports for all enhancements
            for enhancement_id in self.provenance.enhancement_metadata.keys():
                try:
                    enhancement_report = self.provenance.generate_enhancement_report(enhancement_id)
                    export_data["enhancement_details"][enhancement_id] = enhancement_report
                except Exception as e:
                    export_data["enhancement_details"][enhancement_id] = {"error": str(e)}
            
            # Write to file
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            self.logger.info(f"Enhancement metadata exported to {output_path}")
            
            return {
                "success": True,
                "output_path": str(output_file),
                "enhancements_exported": len(export_data["enhancement_details"]),
                "file_size_bytes": output_file.stat().st_size
            }
            
        except Exception as e:
            self.logger.error(f"Failed to export enhancement metadata: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def shutdown(self):
        """Gracefully shutdown orchestration system"""
        self.logger.info("Shutting down enhancement orchestrator")
        
        # Stop monitoring
        self.stop_monitoring()
        
        # Export final metadata
        if self.config.provenance_tracking_enabled:
            self.export_enhancement_metadata("calibration_safety_governance/final_enhancement_metadata.json")
        
        self.logger.info("Enhancement orchestrator shutdown complete")
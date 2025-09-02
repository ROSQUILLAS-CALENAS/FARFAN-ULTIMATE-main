"""
Comprehensive Pipeline Monitoring Dashboard System

This module provides real-time monitoring of pipeline health across all 12 stages
with metrics collection, automated alerting, and visual feedback for production use.
"""

import json
import logging
import os
import time
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import hashlib
import psutil

# Import audit system
try:
    from audit_logger import AuditLogger
    AUDIT_AVAILABLE = True
except ImportError:
    AUDIT_AVAILABLE = False

# Import comprehensive orchestrator for metrics
try:
    from comprehensive_pipeline_orchestrator import MonitoringMetrics, ProcessingStageMetrics, ErrorMetrics
    ORCHESTRATOR_AVAILABLE = True
except ImportError:
    ORCHESTRATOR_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class StageHealth:
    """Health status for an individual pipeline stage"""
    stage_name: str
    status: str = "unknown"  # healthy, warning, error, unknown
    processing_time_avg: float = 0.0
    error_count: int = 0
    error_rate: float = 0.0
    documents_processed: int = 0
    schema_compliance_rate: float = 100.0
    validation_failures: int = 0
    checksum_matches: float = 100.0
    memory_usage: float = 0.0
    last_update: float = 0.0
    recent_errors: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class PipelineAlert:
    """Alert for pipeline health issues"""
    alert_id: str
    severity: str  # info, warning, critical
    stage: str
    message: str
    timestamp: float
    threshold: str
    is_active: bool = True
    
@dataclass
class SystemMetrics:
    """Overall system performance metrics"""
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    disk_io_read: float = 0.0
    disk_io_write: float = 0.0
    network_bytes_sent: float = 0.0
    network_bytes_recv: float = 0.0
    active_processes: int = 0

class MonitoringDashboard:
    """
    Comprehensive monitoring dashboard that tracks pipeline health across all stages
    with real-time metrics, automated alerting, and visual feedback.
    """
    
    def __init__(self, 
                 alert_thresholds: Optional[Dict[str, Dict[str, float]]] = None,
                 metrics_window_minutes: int = 60):
        """
        Initialize monitoring dashboard.
        
        Args:
            alert_thresholds: Configurable thresholds for different alert types
            metrics_window_minutes: Time window for metrics aggregation
        """
        self.metrics_window_minutes = metrics_window_minutes
        self.start_time = datetime.now()
        
        # Default alert thresholds (configurable for production)
        self.alert_thresholds = alert_thresholds or {
            "processing_time": {"warning": 5.0, "critical": 10.0},
            "error_rate": {"warning": 5.0, "critical": 10.0}, 
            "memory_usage": {"warning": 75.0, "critical": 90.0},
            "schema_compliance": {"warning": 95.0, "critical": 90.0},
            "consecutive_failures": {"warning": 3, "critical": 5}
        }
        
        # Stage tracking
        self.stages = [
            "I_ingestion_preparation", "A_analysis_nlp", "K_knowledge_extraction",
            "R_search_retrieval", "L_classification_evaluation", "G_aggregation_reporting", 
            "S_synthesis_output", "T_integration_storage", "X_context_construction",
            "O_orchestration_control", "mathematical_enhancers", "evaluation",
            "ingestion_gate_validation", "orchestration_control", "aggregation_reporting",
            "integration_storage", "search_retrieval", "analysis_quality_assurance"
        ]
        
        # Data structures for metrics
        self.stage_health: Dict[str, StageHealth] = {}
        self.active_alerts: Dict[str, PipelineAlert] = {}
        self.metrics_history = deque(maxlen=1000)
        self.performance_data = deque(maxlen=500)
        
        # Initialize stage health
        for stage in self.stages:
            self.stage_health[stage] = StageHealth(stage_name=stage)
        
        # Threading for background data collection
        self._running = False
        self._collector_thread: Optional[threading.Thread] = None
        
        # Audit log paths
        self.canonical_flow_dir = Path("canonical_flow")
        self.audit_patterns = ["*_audit.json", "*audit*.json"]
        
    def start_monitoring(self):
        """Start background monitoring and metrics collection."""
        if self._running:
            return
            
        self._running = True
        self._collector_thread = threading.Thread(target=self._collect_metrics_loop, daemon=True)
        self._collector_thread.start()
        logger.info("Pipeline monitoring started")
        
    def stop_monitoring(self):
        """Stop background monitoring."""
        self._running = False
        if self._collector_thread:
            self._collector_thread.join(timeout=5)
        logger.info("Pipeline monitoring stopped")
        
    def _collect_metrics_loop(self):
        """Background loop for collecting metrics and detecting issues."""
        while self._running:
            try:
                self._collect_stage_metrics()
                self._collect_system_metrics()
                self._detect_alerts()
                self._cleanup_old_data()
                
                # Sleep between collection cycles
                time.sleep(30)  # Collect every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in metrics collection loop: {e}")
                time.sleep(60)  # Back off on errors
    
    def _collect_stage_metrics(self):
        """Collect metrics for each pipeline stage from audit logs."""
        current_time = time.time()
        
        for stage_name in self.stages:
            stage_health = self.stage_health[stage_name]
            
            try:
                # Look for audit files in canonical_flow/<stage_name>/
                stage_dir = self.canonical_flow_dir / stage_name
                if not stage_dir.exists():
                    continue
                    
                # Process recent audit files
                audit_files = []
                for pattern in self.audit_patterns:
                    audit_files.extend(stage_dir.glob(pattern))
                
                if not audit_files:
                    continue
                    
                # Sort by modification time, get recent files
                recent_files = sorted(audit_files, key=lambda f: f.stat().st_mtime, reverse=True)[:10]
                
                processing_times = []
                error_count = 0
                docs_processed = 0
                validation_failures = 0
                checksum_failures = 0
                
                for audit_file in recent_files:
                    try:
                        with open(audit_file, 'r') as f:
                            audit_data = json.load(f)
                            
                        # Extract metrics from audit data
                        if isinstance(audit_data, dict):
                            # Processing time
                            if 'duration_ms' in audit_data and audit_data['duration_ms']:
                                processing_times.append(audit_data['duration_ms'] / 1000.0)
                                
                            # Error tracking
                            if audit_data.get('status') == 'failed':
                                error_count += 1
                            elif audit_data.get('error_count', 0) > 0:
                                error_count += audit_data['error_count']
                                
                            docs_processed += 1
                            
                            # Schema validation failures
                            if 'validation' in audit_data.get('metadata', {}):
                                if not audit_data['metadata']['validation'].get('schema_valid', True):
                                    validation_failures += 1
                                    
                            # Checksum failures 
                            if 'integrity' in audit_data.get('metadata', {}):
                                if not audit_data['metadata']['integrity'].get('checksum_valid', True):
                                    checksum_failures += 1
                                    
                    except (json.JSONDecodeError, IOError) as e:
                        logger.debug(f"Could not process audit file {audit_file}: {e}")
                        continue
                
                # Update stage health metrics
                if processing_times:
                    stage_health.processing_time_avg = sum(processing_times) / len(processing_times)
                    
                stage_health.error_count = error_count
                stage_health.documents_processed = docs_processed
                stage_health.validation_failures = validation_failures
                
                if docs_processed > 0:
                    stage_health.error_rate = (error_count / docs_processed) * 100
                    stage_health.schema_compliance_rate = ((docs_processed - validation_failures) / docs_processed) * 100
                    stage_health.checksum_matches = ((docs_processed - checksum_failures) / docs_processed) * 100
                    
                # Determine stage status
                stage_health.status = self._determine_stage_status(stage_health)
                stage_health.last_update = current_time
                
            except Exception as e:
                logger.error(f"Error collecting metrics for stage {stage_name}: {e}")
                stage_health.status = "unknown"
                
    def _collect_system_metrics(self):
        """Collect system-level performance metrics."""
        try:
            # CPU and memory usage
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()
            
            # Network I/O  
            network_io = psutil.net_io_counters()
            
            # Process count
            process_count = len(psutil.pids())
            
            system_metrics = SystemMetrics(
                cpu_usage=cpu_percent,
                memory_usage=memory.percent,
                disk_io_read=disk_io.read_bytes if disk_io else 0,
                disk_io_write=disk_io.write_bytes if disk_io else 0,
                network_bytes_sent=network_io.bytes_sent if network_io else 0,
                network_bytes_recv=network_io.bytes_recv if network_io else 0,
                active_processes=process_count
            )
            
            self.performance_data.append({
                "timestamp": time.time(),
                "metrics": system_metrics
            })
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            
    def _determine_stage_status(self, stage_health: StageHealth) -> str:
        """Determine stage status based on metrics and thresholds."""
        # Critical conditions
        if (stage_health.error_rate >= self.alert_thresholds["error_rate"]["critical"] or
            stage_health.processing_time_avg >= self.alert_thresholds["processing_time"]["critical"] or
            stage_health.schema_compliance_rate <= self.alert_thresholds["schema_compliance"]["critical"]):
            return "error"
            
        # Warning conditions  
        if (stage_health.error_rate >= self.alert_thresholds["error_rate"]["warning"] or
            stage_health.processing_time_avg >= self.alert_thresholds["processing_time"]["warning"] or
            stage_health.schema_compliance_rate <= self.alert_thresholds["schema_compliance"]["warning"]):
            return "warning"
            
        return "healthy"
        
    def _detect_alerts(self):
        """Detect and generate alerts based on current metrics."""
        current_time = time.time()
        
        for stage_name, stage_health in self.stage_health.items():
            # Processing time alerts
            if stage_health.processing_time_avg >= self.alert_thresholds["processing_time"]["critical"]:
                self._create_alert(
                    f"processing_time_critical_{stage_name}",
                    "critical", 
                    stage_name,
                    f"Processing time ({stage_health.processing_time_avg:.1f}s) exceeded critical threshold",
                    f"processing_time > {self.alert_thresholds['processing_time']['critical']}s"
                )
            elif stage_health.processing_time_avg >= self.alert_thresholds["processing_time"]["warning"]:
                self._create_alert(
                    f"processing_time_warning_{stage_name}",
                    "warning",
                    stage_name, 
                    f"Processing time ({stage_health.processing_time_avg:.1f}s) exceeded warning threshold",
                    f"processing_time > {self.alert_thresholds['processing_time']['warning']}s"
                )
                
            # Error rate alerts
            if stage_health.error_rate >= self.alert_thresholds["error_rate"]["critical"]:
                self._create_alert(
                    f"error_rate_critical_{stage_name}",
                    "critical",
                    stage_name,
                    f"Error rate ({stage_health.error_rate:.1f}%) exceeded critical threshold", 
                    f"error_rate > {self.alert_thresholds['error_rate']['critical']}%"
                )
            elif stage_health.error_rate >= self.alert_thresholds["error_rate"]["warning"]:
                self._create_alert(
                    f"error_rate_warning_{stage_name}",
                    "warning",
                    stage_name,
                    f"Error rate ({stage_health.error_rate:.1f}%) exceeded warning threshold",
                    f"error_rate > {self.alert_thresholds['error_rate']['warning']}%"
                )
                
            # Schema compliance alerts
            if stage_health.schema_compliance_rate <= self.alert_thresholds["schema_compliance"]["critical"]:
                self._create_alert(
                    f"schema_compliance_critical_{stage_name}", 
                    "critical",
                    stage_name,
                    f"Schema compliance ({stage_health.schema_compliance_rate:.1f}%) below critical threshold",
                    f"schema_compliance < {self.alert_thresholds['schema_compliance']['critical']}%"
                )
                
        # System-level alerts
        if self.performance_data:
            latest_perf = self.performance_data[-1]["metrics"]
            
            if latest_perf.memory_usage >= self.alert_thresholds["memory_usage"]["critical"]:
                self._create_alert(
                    "memory_usage_critical",
                    "critical", 
                    "system",
                    f"Memory usage ({latest_perf.memory_usage:.1f}%) exceeded critical threshold",
                    f"memory_usage > {self.alert_thresholds['memory_usage']['critical']}%"
                )
            elif latest_perf.memory_usage >= self.alert_thresholds["memory_usage"]["warning"]:
                self._create_alert(
                    "memory_usage_warning",
                    "warning",
                    "system", 
                    f"Memory usage ({latest_perf.memory_usage:.1f}%) exceeded warning threshold",
                    f"memory_usage > {self.alert_thresholds['memory_usage']['warning']}%"
                )
    
    def _create_alert(self, alert_id: str, severity: str, stage: str, message: str, threshold: str):
        """Create or update an alert."""
        if alert_id not in self.active_alerts:
            self.active_alerts[alert_id] = PipelineAlert(
                alert_id=alert_id,
                severity=severity,
                stage=stage, 
                message=message,
                timestamp=time.time(),
                threshold=threshold
            )
            logger.warning(f"Alert created: {severity.upper()} - {stage} - {message}")
            
    def _cleanup_old_data(self):
        """Clean up old metrics and resolved alerts."""
        current_time = time.time()
        window_seconds = self.metrics_window_minutes * 60
        
        # Remove old performance data
        while (self.performance_data and 
               current_time - self.performance_data[0]["timestamp"] > window_seconds):
            self.performance_data.popleft()
            
        # Remove resolved alerts (older than 1 hour)
        expired_alerts = []
        for alert_id, alert in self.active_alerts.items():
            if current_time - alert.timestamp > 3600:  # 1 hour
                expired_alerts.append(alert_id)
                
        for alert_id in expired_alerts:
            del self.active_alerts[alert_id]
            
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current dashboard metrics for API."""
        healthy_stages = sum(1 for stage in self.stage_health.values() if stage.status == "healthy")
        total_docs = sum(stage.documents_processed for stage in self.stage_health.values())
        avg_processing_time = sum(stage.processing_time_avg for stage in self.stage_health.values()) / len(self.stages)
        avg_error_rate = sum(stage.error_rate for stage in self.stage_health.values()) / len(self.stages)
        
        # System metrics
        memory_usage = 0.0
        if self.performance_data:
            memory_usage = self.performance_data[-1]["metrics"].memory_usage
            
        return {
            "system": {
                "healthy_stages": healthy_stages,
                "total_stages": len(self.stages),
                "avg_processing_time": round(avg_processing_time, 1),
                "error_rate": round(avg_error_rate, 1),
                "memory_usage": round(memory_usage, 1),
                "documents_processed": total_docs
            },
            "processing": {
                "throughput": round(total_docs / max(1, (time.time() - self.start_time.timestamp()) / 3600), 1),
                "queue_depth": len(self.active_alerts),  # Approximation
                "active_workers": psutil.cpu_count()
            },
            "compliance": {
                "schema_compliance": round(sum(stage.schema_compliance_rate for stage in self.stage_health.values()) / len(self.stages), 1),
                "validation_failures": sum(stage.validation_failures for stage in self.stage_health.values()),
                "checksum_matches": round(sum(stage.checksum_matches for stage in self.stage_health.values()) / len(self.stages), 1),
                "integrity_failures": sum(1 for stage in self.stage_health.values() if stage.checksum_matches < 100)
            },
            "timestamp": time.time()
        }
        
    def get_pipeline_health(self) -> Dict[str, Any]:
        """Get detailed pipeline health for all stages."""
        healthy_count = sum(1 for stage in self.stage_health.values() if stage.status == "healthy")
        total_count = len(self.stages)
        
        overall_status = "healthy"
        if any(stage.status == "error" for stage in self.stage_health.values()):
            overall_status = "error"
        elif any(stage.status == "warning" for stage in self.stage_health.values()):
            overall_status = "warning"
            
        return {
            "overall_status": overall_status,
            "healthy_stages": healthy_count,
            "total_stages": total_count,
            "stages": [
                {
                    "name": stage.stage_name,
                    "status": stage.status,
                    "processing_time": f"{stage.processing_time_avg:.1f}s",
                    "error_count": stage.error_count,
                    "error_rate": f"{stage.error_rate:.1f}%",
                    "documents_processed": stage.documents_processed,
                    "schema_compliance": f"{stage.schema_compliance_rate:.1f}%",
                    "last_update": stage.last_update
                }
                for stage in self.stage_health.values()
            ]
        }
        
    def get_stage_metrics(self, stage_name: str) -> Dict[str, Any]:
        """Get detailed metrics for a specific stage."""
        if stage_name not in self.stage_health:
            return {"error": "Stage not found"}
            
        stage = self.stage_health[stage_name]
        
        return {
            "stage_name": stage_name,
            "status": stage.status,
            "processing_rate": f"{stage.documents_processed / max(1, (time.time() - self.start_time.timestamp()) / 3600):.1f} docs/hr",
            "error_rate": stage.error_rate,
            "avg_processing_time": stage.processing_time_avg,
            "memory_usage": stage.memory_usage,
            "schema_compliance_rate": stage.schema_compliance_rate,
            "validation_failures": stage.validation_failures,
            "checksum_matches": stage.checksum_matches,
            "last_24h_stats": {
                "documents_processed": stage.documents_processed,
                "total_errors": stage.error_count,
                "avg_latency": stage.processing_time_avg
            },
            "recent_errors": stage.recent_errors[-5:]  # Last 5 errors
        }
        
    def get_active_alerts(self) -> Dict[str, Any]:
        """Get all active alerts."""
        alerts_list = []
        for alert in self.active_alerts.values():
            alerts_list.append({
                "id": alert.alert_id,
                "severity": alert.severity,
                "stage": alert.stage,
                "message": alert.message,
                "timestamp": alert.timestamp,
                "threshold": alert.threshold
            })
            
        # Sort by severity and timestamp
        severity_order = {"critical": 0, "warning": 1, "info": 2}
        alerts_list.sort(key=lambda a: (severity_order.get(a["severity"], 3), -a["timestamp"]))
        
        alert_summary = {"critical": 0, "warning": 0, "info": 0}
        for alert in alerts_list:
            alert_summary[alert["severity"]] = alert_summary.get(alert["severity"], 0) + 1
            
        return {
            "active_alerts": alerts_list,
            "total_alerts": len(alerts_list),
            "alert_summary": alert_summary
        }
        
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance and resource utilization metrics."""
        if not self.performance_data:
            return {"error": "No performance data available"}
            
        latest = self.performance_data[-1]["metrics"]
        
        # Calculate pipeline metrics
        total_docs = sum(stage.documents_processed for stage in self.stage_health.values())
        uptime_hours = (time.time() - self.start_time.timestamp()) / 3600
        docs_per_hour = total_docs / max(1, uptime_hours)
        
        avg_latency = sum(stage.processing_time_avg for stage in self.stage_health.values()) / len(self.stages)
        
        return {
            "cpu_usage": latest.cpu_usage,
            "memory_usage": latest.memory_usage,
            "disk_io": {
                "read_mb_per_sec": latest.disk_io_read / (1024 * 1024),
                "write_mb_per_sec": latest.disk_io_write / (1024 * 1024)
            },
            "network_io": {
                "bytes_sent_per_sec": latest.network_bytes_sent,
                "bytes_recv_per_sec": latest.network_bytes_recv
            },
            "process_metrics": {
                "active_processes": latest.active_processes,
                "avg_cpu_per_process": latest.cpu_usage / max(1, latest.active_processes),
                "total_memory_mb": psutil.virtual_memory().total / (1024 * 1024)
            },
            "pipeline_metrics": {
                "documents_per_hour": round(docs_per_hour, 1),
                "avg_end_to_end_latency": round(avg_latency, 1),
                "queue_depths": {
                    "ingestion": len([a for a in self.active_alerts.values() if "ingestion" in a.stage]),
                    "processing": len([a for a in self.active_alerts.values() if a.severity in ["warning", "critical"]]),
                    "output": len([a for a in self.active_alerts.values() if "output" in a.stage or "synthesis" in a.stage])
                }
            }
        }

# Global dashboard instance
_dashboard_instance: Optional[MonitoringDashboard] = None

def get_dashboard() -> MonitoringDashboard:
    """Get or create global dashboard instance."""
    global _dashboard_instance
    if _dashboard_instance is None:
        _dashboard_instance = MonitoringDashboard()
        _dashboard_instance.start_monitoring()
    return _dashboard_instance
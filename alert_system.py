"""
Automated Alert System for Pipeline Health Monitoring

Provides configurable threshold-based alerting with escalation policies
and notification channels for production pipeline monitoring.
"""

import json
import logging
import smtplib
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set
import threading
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning" 
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class AlertStatus(Enum):
    """Alert status"""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"

@dataclass
class AlertThreshold:
    """Configurable alert threshold"""
    metric_name: str
    operator: str  # '>', '<', '>=', '<=', '=='
    value: float
    window_minutes: int = 5
    consecutive_breaches: int = 1
    severity: AlertSeverity = AlertSeverity.WARNING

@dataclass
class AlertRule:
    """Complete alert rule configuration"""
    rule_id: str
    name: str
    description: str
    stage_filter: Optional[str] = None  # Filter by stage name, None for all stages
    thresholds: List[AlertThreshold] = field(default_factory=list)
    enabled: bool = True
    escalation_minutes: int = 30  # Time before escalation
    max_alerts_per_hour: int = 10  # Rate limiting
    notification_channels: List[str] = field(default_factory=list)

@dataclass 
class Alert:
    """Individual alert instance"""
    alert_id: str
    rule_id: str
    stage: str
    metric_name: str
    current_value: float
    threshold_value: float
    severity: AlertSeverity
    status: AlertStatus = AlertStatus.ACTIVE
    created_at: datetime = field(default_factory=datetime.now)
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    escalation_count: int = 0
    
    def age_minutes(self) -> float:
        """Get alert age in minutes."""
        return (datetime.now() - self.created_at).total_seconds() / 60

class NotificationChannel:
    """Base class for alert notification channels"""
    
    def __init__(self, channel_id: str, config: Dict[str, Any]):
        self.channel_id = channel_id
        self.config = config
        
    async def send_alert(self, alert: Alert) -> bool:
        """Send alert notification. Return True if successful."""
        raise NotImplementedError

class EmailNotificationChannel(NotificationChannel):
    """Email notification channel"""
    
    def __init__(self, channel_id: str, config: Dict[str, Any]):
        super().__init__(channel_id, config)
        self.smtp_server = config.get("smtp_server", "localhost")
        self.smtp_port = config.get("smtp_port", 587)
        self.username = config.get("username")
        self.password = config.get("password")
        self.from_email = config.get("from_email")
        self.to_emails = config.get("to_emails", [])
        
    async def send_alert(self, alert: Alert) -> bool:
        """Send email alert."""
        try:
            subject = f"[{alert.severity.value.upper()}] Pipeline Alert: {alert.stage}"
            
            body = f"""
Pipeline Alert Notification

Alert ID: {alert.alert_id}
Stage: {alert.stage}
Severity: {alert.severity.value.upper()}
Metric: {alert.metric_name}
Current Value: {alert.current_value}
Threshold: {alert.threshold_value}
Created: {alert.created_at.strftime('%Y-%m-%d %H:%M:%S')}

Message: {alert.message}

Dashboard: http://localhost:8000/dashboard
API Details: http://localhost:8000/api/stages/{alert.stage}
            """
            
            msg = MIMEMultipart()
            msg['From'] = self.from_email
            msg['To'] = ", ".join(self.to_emails)
            msg['Subject'] = subject
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                if self.username and self.password:
                    server.starttls()
                    server.login(self.username, self.password)
                server.send_message(msg)
                
            logger.info(f"Email alert sent for {alert.alert_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email alert {alert.alert_id}: {e}")
            return False

class SlackNotificationChannel(NotificationChannel):
    """Slack notification channel (webhook)"""
    
    def __init__(self, channel_id: str, config: Dict[str, Any]):
        super().__init__(channel_id, config)
        self.webhook_url = config.get("webhook_url")
        self.channel = config.get("channel", "#alerts")
        
    async def send_alert(self, alert: Alert) -> bool:
        """Send Slack alert via webhook."""
        try:
            import requests
            
            color_map = {
                AlertSeverity.INFO: "good",
                AlertSeverity.WARNING: "warning", 
                AlertSeverity.CRITICAL: "danger",
                AlertSeverity.EMERGENCY: "danger"
            }
            
            payload = {
                "channel": self.channel,
                "username": "Pipeline Monitor",
                "icon_emoji": ":warning:",
                "attachments": [{
                    "color": color_map.get(alert.severity, "warning"),
                    "title": f"Pipeline Alert: {alert.stage}",
                    "text": alert.message,
                    "fields": [
                        {"title": "Severity", "value": alert.severity.value.upper(), "short": True},
                        {"title": "Metric", "value": alert.metric_name, "short": True},
                        {"title": "Current", "value": str(alert.current_value), "short": True},
                        {"title": "Threshold", "value": str(alert.threshold_value), "short": True},
                    ],
                    "footer": f"Alert ID: {alert.alert_id}",
                    "ts": int(alert.created_at.timestamp())
                }]
            }
            
            response = requests.post(self.webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            
            logger.info(f"Slack alert sent for {alert.alert_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send Slack alert {alert.alert_id}: {e}")
            return False

class WebhookNotificationChannel(NotificationChannel):
    """Generic webhook notification channel"""
    
    def __init__(self, channel_id: str, config: Dict[str, Any]):
        super().__init__(channel_id, config)
        self.url = config.get("url")
        self.headers = config.get("headers", {})
        self.timeout = config.get("timeout", 10)
        
    async def send_alert(self, alert: Alert) -> bool:
        """Send webhook alert."""
        try:
            import requests
            
            payload = {
                "alert_id": alert.alert_id,
                "rule_id": alert.rule_id,
                "stage": alert.stage,
                "severity": alert.severity.value,
                "status": alert.status.value,
                "metric_name": alert.metric_name,
                "current_value": alert.current_value,
                "threshold_value": alert.threshold_value,
                "message": alert.message,
                "created_at": alert.created_at.isoformat(),
                "metadata": alert.metadata
            }
            
            response = requests.post(
                self.url,
                json=payload,
                headers=self.headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            logger.info(f"Webhook alert sent for {alert.alert_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send webhook alert {alert.alert_id}: {e}")
            return False

class AlertSystem:
    """
    Comprehensive alert system with configurable thresholds, escalation policies,
    and multiple notification channels for production pipeline monitoring.
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize alert system.
        
        Args:
            config_file: Path to alert configuration file
        """
        self.config_file = config_file or "alert_config.json"
        
        # Core data structures
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.notification_channels: Dict[str, NotificationChannel] = {}
        
        # Rate limiting and deduplication
        self.alert_counts: Dict[str, List[datetime]] = {}
        self.last_alert_times: Dict[str, datetime] = {}
        
        # Background processing
        self._running = False
        self._processor_thread: Optional[threading.Thread] = None
        
        # Load configuration
        self._load_configuration()
        
    def _load_configuration(self):
        """Load alert rules and notification channels from configuration."""
        try:
            if Path(self.config_file).exists():
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
            else:
                config = self._get_default_config()
                self._save_configuration(config)
                
            # Load alert rules
            for rule_data in config.get("alert_rules", []):
                rule = AlertRule(**rule_data)
                self.alert_rules[rule.rule_id] = rule
                
            # Load notification channels
            for channel_data in config.get("notification_channels", []):
                channel_type = channel_data.get("type")
                channel_id = channel_data.get("channel_id")
                channel_config = channel_data.get("config", {})
                
                if channel_type == "email":
                    channel = EmailNotificationChannel(channel_id, channel_config)
                elif channel_type == "slack":
                    channel = SlackNotificationChannel(channel_id, channel_config)  
                elif channel_type == "webhook":
                    channel = WebhookNotificationChannel(channel_id, channel_config)
                else:
                    logger.warning(f"Unknown notification channel type: {channel_type}")
                    continue
                    
                self.notification_channels[channel_id] = channel
                
            logger.info(f"Loaded {len(self.alert_rules)} alert rules and {len(self.notification_channels)} notification channels")
            
        except Exception as e:
            logger.error(f"Error loading alert configuration: {e}")
            self._load_default_rules()
            
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default alert configuration."""
        return {
            "alert_rules": [
                {
                    "rule_id": "high_error_rate",
                    "name": "High Error Rate",
                    "description": "Alert when error rate exceeds thresholds",
                    "thresholds": [
                        {
                            "metric_name": "error_rate",
                            "operator": ">=",
                            "value": 5.0,
                            "severity": "warning"
                        },
                        {
                            "metric_name": "error_rate", 
                            "operator": ">=",
                            "value": 10.0,
                            "severity": "critical"
                        }
                    ],
                    "notification_channels": ["email", "slack"]
                },
                {
                    "rule_id": "slow_processing",
                    "name": "Slow Processing",
                    "description": "Alert when processing time is too high",
                    "thresholds": [
                        {
                            "metric_name": "processing_time_avg",
                            "operator": ">=", 
                            "value": 5.0,
                            "severity": "warning"
                        },
                        {
                            "metric_name": "processing_time_avg",
                            "operator": ">=",
                            "value": 10.0,
                            "severity": "critical"
                        }
                    ],
                    "notification_channels": ["email"]
                },
                {
                    "rule_id": "low_schema_compliance",
                    "name": "Low Schema Compliance",
                    "description": "Alert when schema compliance drops",
                    "thresholds": [
                        {
                            "metric_name": "schema_compliance_rate",
                            "operator": "<=",
                            "value": 95.0,
                            "severity": "warning"
                        },
                        {
                            "metric_name": "schema_compliance_rate",
                            "operator": "<=",
                            "value": 90.0,
                            "severity": "critical"
                        }
                    ],
                    "notification_channels": ["email", "slack"]
                },
                {
                    "rule_id": "high_memory_usage",
                    "name": "High Memory Usage",
                    "description": "Alert when system memory usage is high",
                    "stage_filter": None,  # System-level
                    "thresholds": [
                        {
                            "metric_name": "memory_usage",
                            "operator": ">=",
                            "value": 80.0,
                            "severity": "warning"
                        },
                        {
                            "metric_name": "memory_usage",
                            "operator": ">=",
                            "value": 90.0,
                            "severity": "critical"
                        }
                    ],
                    "notification_channels": ["email"]
                }
            ],
            "notification_channels": [
                {
                    "type": "email",
                    "channel_id": "email",
                    "config": {
                        "smtp_server": "localhost",
                        "smtp_port": 587,
                        "from_email": "alerts@pipeline.local",
                        "to_emails": ["admin@pipeline.local"]
                    }
                },
                {
                    "type": "slack", 
                    "channel_id": "slack",
                    "config": {
                        "webhook_url": "https://hooks.slack.com/services/YOUR/WEBHOOK/URL",
                        "channel": "#pipeline-alerts"
                    }
                }
            ]
        }
        
    def _save_configuration(self, config: Dict[str, Any]):
        """Save configuration to file."""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving alert configuration: {e}")
            
    def _load_default_rules(self):
        """Load minimal default alert rules."""
        self.alert_rules = {
            "high_error_rate": AlertRule(
                rule_id="high_error_rate",
                name="High Error Rate",
                description="Alert when error rate exceeds 5%",
                thresholds=[
                    AlertThreshold("error_rate", ">=", 5.0, severity=AlertSeverity.WARNING),
                    AlertThreshold("error_rate", ">=", 10.0, severity=AlertSeverity.CRITICAL)
                ]
            )
        }
        
    def start_monitoring(self):
        """Start background alert processing."""
        if self._running:
            return
            
        self._running = True
        self._processor_thread = threading.Thread(target=self._process_alerts_loop, daemon=True)
        self._processor_thread.start()
        logger.info("Alert system started")
        
    def stop_monitoring(self):
        """Stop background alert processing."""
        self._running = False
        if self._processor_thread:
            self._processor_thread.join(timeout=5)
        logger.info("Alert system stopped")
        
    def _process_alerts_loop(self):
        """Background loop for processing alerts and escalations."""
        while self._running:
            try:
                self._process_escalations()
                self._cleanup_resolved_alerts()
                time.sleep(60)  # Process every minute
                
            except Exception as e:
                logger.error(f"Error in alert processing loop: {e}")
                time.sleep(60)
                
    def _process_escalations(self):
        """Process alert escalations."""
        current_time = datetime.now()
        
        for alert in self.active_alerts.values():
            if alert.status != AlertStatus.ACTIVE:
                continue
                
            # Get corresponding rule
            rule = self.alert_rules.get(alert.rule_id)
            if not rule or not rule.enabled:
                continue
                
            # Check if escalation is needed
            if alert.age_minutes() >= rule.escalation_minutes:
                if alert.escalation_count == 0:  # First escalation
                    alert.escalation_count += 1
                    alert.severity = AlertSeverity.CRITICAL
                    logger.warning(f"Escalating alert {alert.alert_id} to CRITICAL")
                    
                    # Send escalated notification
                    self._send_notifications(alert, escalation=True)
                    
    def _cleanup_resolved_alerts(self):
        """Clean up old resolved alerts."""
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        # Remove old alerts from history
        self.alert_history = [
            alert for alert in self.alert_history
            if alert.resolved_at is None or alert.resolved_at > cutoff_time
        ]
        
        # Clean up rate limiting data
        for rule_id in list(self.alert_counts.keys()):
            self.alert_counts[rule_id] = [
                timestamp for timestamp in self.alert_counts[rule_id]
                if timestamp > cutoff_time
            ]
            
    def evaluate_metrics(self, stage: str, metrics: Dict[str, float]):
        """
        Evaluate metrics against alert rules and generate alerts as needed.
        
        Args:
            stage: Pipeline stage name
            metrics: Dictionary of metric name -> value pairs
        """
        current_time = datetime.now()
        
        for rule in self.alert_rules.values():
            if not rule.enabled:
                continue
                
            # Apply stage filter if specified
            if rule.stage_filter and rule.stage_filter != stage:
                continue
                
            # Check rate limiting
            if self._is_rate_limited(rule.rule_id):
                continue
                
            # Evaluate each threshold
            for threshold in rule.thresholds:
                if threshold.metric_name not in metrics:
                    continue
                    
                current_value = metrics[threshold.metric_name]
                
                # Check threshold
                if self._check_threshold(threshold, current_value):
                    # Generate alert
                    alert_id = f"{rule.rule_id}_{stage}_{threshold.metric_name}"
                    
                    # Check if alert already exists and is active
                    if alert_id in self.active_alerts:
                        existing_alert = self.active_alerts[alert_id]
                        if existing_alert.status == AlertStatus.ACTIVE:
                            # Update existing alert
                            existing_alert.current_value = current_value
                            continue
                            
                    # Create new alert
                    alert = Alert(
                        alert_id=alert_id,
                        rule_id=rule.rule_id,
                        stage=stage,
                        metric_name=threshold.metric_name,
                        current_value=current_value,
                        threshold_value=threshold.value,
                        severity=threshold.severity,
                        message=f"{rule.name}: {threshold.metric_name} {threshold.operator} {threshold.value} (current: {current_value})"
                    )
                    
                    self.active_alerts[alert_id] = alert
                    self.alert_history.append(alert)
                    
                    # Update rate limiting
                    self._update_rate_limiting(rule.rule_id)
                    
                    # Send notifications
                    self._send_notifications(alert)
                    
                    logger.warning(f"Generated alert: {alert.message}")
                    
                else:
                    # Check if we should resolve an existing alert
                    alert_id = f"{rule.rule_id}_{stage}_{threshold.metric_name}"
                    if alert_id in self.active_alerts:
                        existing_alert = self.active_alerts[alert_id]
                        if existing_alert.status == AlertStatus.ACTIVE:
                            existing_alert.status = AlertStatus.RESOLVED
                            existing_alert.resolved_at = current_time
                            logger.info(f"Resolved alert: {alert_id}")
                            
    def _check_threshold(self, threshold: AlertThreshold, value: float) -> bool:
        """Check if value breaches threshold."""
        if threshold.operator == ">":
            return value > threshold.value
        elif threshold.operator == ">=":
            return value >= threshold.value
        elif threshold.operator == "<":
            return value < threshold.value
        elif threshold.operator == "<=":
            return value <= threshold.value
        elif threshold.operator == "==":
            return value == threshold.value
        return False
        
    def _is_rate_limited(self, rule_id: str) -> bool:
        """Check if rule is rate limited."""
        if rule_id not in self.alert_counts:
            return False
            
        rule = self.alert_rules.get(rule_id)
        if not rule:
            return False
            
        # Count alerts in the last hour
        cutoff_time = datetime.now() - timedelta(hours=1)
        recent_alerts = [
            timestamp for timestamp in self.alert_counts[rule_id]
            if timestamp > cutoff_time
        ]
        
        return len(recent_alerts) >= rule.max_alerts_per_hour
        
    def _update_rate_limiting(self, rule_id: str):
        """Update rate limiting counters."""
        current_time = datetime.now()
        
        if rule_id not in self.alert_counts:
            self.alert_counts[rule_id] = []
            
        self.alert_counts[rule_id].append(current_time)
        self.last_alert_times[rule_id] = current_time
        
    def _send_notifications(self, alert: Alert, escalation: bool = False):
        """Send alert notifications through configured channels."""
        rule = self.alert_rules.get(alert.rule_id)
        if not rule:
            return
            
        for channel_id in rule.notification_channels:
            if channel_id not in self.notification_channels:
                logger.warning(f"Notification channel {channel_id} not configured")
                continue
                
            channel = self.notification_channels[channel_id]
            
            try:
                # Run notification in background
                import asyncio
                asyncio.create_task(channel.send_alert(alert))
                
            except Exception as e:
                logger.error(f"Error sending notification via {channel_id}: {e}")
                
    def acknowledge_alert(self, alert_id: str, user: str = "system") -> bool:
        """Acknowledge an active alert."""
        if alert_id not in self.active_alerts:
            return False
            
        alert = self.active_alerts[alert_id]
        if alert.status == AlertStatus.ACTIVE:
            alert.status = AlertStatus.ACKNOWLEDGED
            alert.acknowledged_at = datetime.now()
            alert.metadata["acknowledged_by"] = user
            
            logger.info(f"Alert {alert_id} acknowledged by {user}")
            return True
            
        return False
        
    def resolve_alert(self, alert_id: str, user: str = "system") -> bool:
        """Manually resolve an alert."""
        if alert_id not in self.active_alerts:
            return False
            
        alert = self.active_alerts[alert_id]
        alert.status = AlertStatus.RESOLVED
        alert.resolved_at = datetime.now() 
        alert.metadata["resolved_by"] = user
        
        logger.info(f"Alert {alert_id} resolved by {user}")
        return True
        
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get list of active alerts."""
        return [
            {
                "alert_id": alert.alert_id,
                "rule_id": alert.rule_id,
                "stage": alert.stage,
                "severity": alert.severity.value,
                "status": alert.status.value,
                "metric_name": alert.metric_name,
                "current_value": alert.current_value,
                "threshold_value": alert.threshold_value,
                "message": alert.message,
                "created_at": alert.created_at.isoformat(),
                "age_minutes": alert.age_minutes(),
                "escalation_count": alert.escalation_count
            }
            for alert in self.active_alerts.values()
            if alert.status in [AlertStatus.ACTIVE, AlertStatus.ACKNOWLEDGED]
        ]
        
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary statistics of alerts."""
        active_alerts = [a for a in self.active_alerts.values() if a.status == AlertStatus.ACTIVE]
        
        severity_counts = {
            "info": 0,
            "warning": 0,
            "critical": 0,
            "emergency": 0
        }
        
        for alert in active_alerts:
            severity_counts[alert.severity.value] += 1
            
        return {
            "total_active": len(active_alerts),
            "total_acknowledged": len([a for a in self.active_alerts.values() if a.status == AlertStatus.ACKNOWLEDGED]),
            "total_resolved_24h": len([a for a in self.alert_history if a.resolved_at and a.resolved_at > datetime.now() - timedelta(hours=24)]),
            "severity_breakdown": severity_counts,
            "top_alerting_stages": self._get_top_alerting_stages()
        }
        
    def _get_top_alerting_stages(self) -> List[Dict[str, Any]]:
        """Get stages with most alerts."""
        stage_counts = {}
        for alert in self.active_alerts.values():
            if alert.status == AlertStatus.ACTIVE:
                stage_counts[alert.stage] = stage_counts.get(alert.stage, 0) + 1
                
        return [
            {"stage": stage, "alert_count": count}
            for stage, count in sorted(stage_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        ]

# Global alert system instance
_alert_system: Optional[AlertSystem] = None

def get_alert_system() -> AlertSystem:
    """Get or create global alert system instance."""
    global _alert_system
    if _alert_system is None:
        _alert_system = AlertSystem()
        _alert_system.start_monitoring()
    return _alert_system
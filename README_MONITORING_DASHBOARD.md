# Comprehensive Pipeline Monitoring Dashboard System

This document describes the comprehensive monitoring dashboard system that tracks pipeline health across all 12 stages with real-time metrics collection, automated alerting, and visual feedback for production monitoring.

## Overview

The monitoring dashboard system provides:

- **Real-time metrics collection** from audit logs and stage artifacts
- **Automated alerting** with configurable thresholds and escalation policies  
- **Visual web interface** integrated into canonical_web_server.py
- **RESTful API endpoints** for programmatic access
- **Pipeline health tracking** across 18+ stages
- **Performance monitoring** with memory and resource utilization
- **Data integrity validation** with checksum verification
- **Schema compliance monitoring** per stage

## Architecture

### Core Components

1. **MonitoringDashboard** (`monitoring_dashboard.py`)
   - Collects metrics from audit logs across all stages
   - Provides real-time pipeline health assessment
   - Tracks processing rates, error rates, and compliance metrics
   - Generates performance and reliability reports

2. **AlertSystem** (`alert_system.py`) 
   - Configurable threshold-based alerting
   - Multiple notification channels (email, Slack, webhook)
   - Alert escalation and rate limiting
   - Automated alert resolution and acknowledgment

3. **Web Interface** (integrated into `canonical_web_server.py`)
   - Interactive dashboard with real-time updates
   - Stage health visualization
   - Performance metrics charts
   - Active alerts and notifications panel

4. **Configuration System** (`alert_config.json`)
   - Configurable alert thresholds for production
   - Notification channel setup
   - Escalation policies and rate limiting

## Pipeline Stages Monitored

The system monitors 18+ canonical pipeline stages:

1. **I_ingestion_preparation** - Document ingestion and preprocessing
2. **A_analysis_nlp** - Natural language processing and analysis
3. **K_knowledge_extraction** - Knowledge and entity extraction
4. **R_search_retrieval** - Search and retrieval operations
5. **L_classification_evaluation** - Document classification and evaluation
6. **G_aggregation_reporting** - Data aggregation and reporting
7. **S_synthesis_output** - Output synthesis and formatting
8. **T_integration_storage** - Data integration and storage
9. **X_context_construction** - Context building and analysis
10. **O_orchestration_control** - Pipeline orchestration and control
11. **mathematical_enhancers** - Mathematical processing enhancements
12. **evaluation** - Final evaluation and validation

## Metrics Tracked

### Per-Stage Metrics

- **Processing Rate**: Documents processed per minute/hour
- **Error Rate**: Percentage of failed processing attempts
- **Average Processing Time**: Mean duration per document
- **Schema Compliance Rate**: Percentage passing schema validation
- **Data Integrity**: Checksum validation success rate
- **Memory Usage**: Peak memory consumption per stage
- **Validation Failures**: Count of schema/validation errors

### System-Wide Metrics

- **Overall Pipeline Health**: Aggregate health across all stages
- **End-to-End Latency**: Total processing time across pipeline
- **Throughput**: Overall documents processed per unit time  
- **Resource Utilization**: CPU, memory, disk, and network usage
- **Queue Depths**: Processing backlogs per stage
- **Inter-Stage Handoff Status**: Success rate of stage transitions

### Data Quality Metrics

- **Schema Compliance**: Per-stage validation against expected schemas
- **Data Integrity Checksums**: Content validation and corruption detection
- **Processing Failure Categorization**: Timeout, memory, validation, connection errors
- **Evidence Quality Scores**: Content completeness and accuracy metrics

## Alert Types and Thresholds

### Production Alert Rules (Configurable)

1. **High Error Rate**
   - Warning: >5% error rate
   - Critical: >10% error rate
   - Escalation: 30 minutes

2. **Slow Processing**
   - Warning: >5 seconds average processing time
   - Critical: >10 seconds average processing time
   - Escalation: 45 minutes

3. **Schema Compliance Drop**
   - Warning: <95% compliance rate
   - Critical: <90% compliance rate
   - Escalation: 20 minutes

4. **Data Integrity Issues**  
   - Warning: <98% checksum matches
   - Critical: <95% checksum matches
   - Escalation: 25 minutes

5. **High Memory Usage**
   - Warning: >80% system memory
   - Critical: >90% system memory
   - Emergency: >95% system memory
   - Escalation: 15 minutes

6. **Pipeline Throughput Drop**
   - Warning: <30 documents/hour
   - Critical: <20 documents/hour
   - Escalation: 30 minutes

7. **Stage Failure Cascade**
   - Critical: 3+ stages reporting errors
   - Emergency: 5+ stages reporting errors
   - Escalation: 10 minutes

## API Endpoints

The dashboard exposes the following RESTful API endpoints:

### Dashboard Metrics
- `GET /api/dashboard/metrics` - Current dashboard metrics summary
- `GET /api/pipeline/health` - Pipeline health status for all stages
- `GET /api/stages/{stage_name}` - Detailed metrics for specific stage

### Alerting
- `GET /api/alerts` - Active alerts and notifications
- `POST /api/alerts/{alert_id}/acknowledge` - Acknowledge an alert
- `POST /api/alerts/{alert_id}/resolve` - Resolve an alert

### Performance  
- `GET /api/performance` - Performance and resource utilization metrics
- `GET /api/metrics/history` - Historical metrics data
- `GET /health` - System health check

### Stage-Specific Endpoints
- `GET /api/stages/{stage}/compliance` - Schema compliance rates
- `GET /api/stages/{stage}/integrity` - Data integrity checksums  
- `GET /api/stages/{stage}/errors` - Processing failures with categorization
- `GET /api/stages/{stage}/performance` - Memory and performance metrics

## Web Interface Features

### Dashboard Layout

The web dashboard (`/dashboard`) provides:

1. **System Overview Panel**
   - Total/healthy stages count
   - Average processing time
   - Overall error rate
   - Memory usage
   - Total documents processed

2. **Stage Health Status**
   - Real-time status for each of the 18+ stages
   - Color-coded health indicators (green/yellow/red)
   - Processing time and error counts per stage

3. **Real-time Processing Metrics**
   - Live throughput chart
   - Queue depth monitoring
   - Active worker count
   - Processing rate visualization

4. **Schema Compliance Panel**
   - Overall compliance percentage
   - Validation failure counts
   - Per-stage compliance breakdown

5. **Data Integrity Panel**
   - Checksum match rates
   - Integrity failure counts
   - Data quality metrics

6. **Active Alerts Panel**
   - Critical, warning, and info alerts
   - Alert timestamps and details
   - Quick acknowledgment actions

### Auto-Refresh and Real-Time Updates

- Dashboard auto-refreshes every 30 seconds
- Real-time metrics via WebSocket (future enhancement)
- Configurable refresh intervals
- Manual refresh capability

## Installation and Setup

### 1. Dependencies

The monitoring system requires:

```bash
pip install psutil  # System metrics
pip install requests  # Webhook notifications (optional)
```

### 2. Configuration

Edit `alert_config.json` to configure:

- Alert thresholds for your environment
- Notification channels (email, Slack, webhook)
- Escalation policies
- Rate limiting settings

### 3. Integration

The monitoring dashboard is fully integrated into the existing `canonical_web_server.py`. Simply run:

```bash
python canonical_web_server.py --port 8000
```

### 4. Web Access

- **Dashboard**: http://localhost:8000/dashboard
- **Health Check**: http://localhost:8000/health  
- **API Documentation**: http://localhost:8000/api/status

## Usage Examples

### 1. Basic Setup and Demo

```bash
# Generate test data and start monitoring
python demo_monitoring_dashboard.py

# In another terminal, start the web server
python canonical_web_server.py --port 8000

# Visit the dashboard
open http://localhost:8000/dashboard
```

### 2. Programmatic API Access

```python
import requests

# Get current pipeline health
response = requests.get('http://localhost:8000/api/pipeline/health')
health_data = response.json()

print(f"Overall Status: {health_data['overall_status']}")
print(f"Healthy Stages: {health_data['healthy_stages']}/{health_data['total_stages']}")

# Get stage-specific metrics  
response = requests.get('http://localhost:8000/api/stages/K_knowledge_extraction')
stage_metrics = response.json()

print(f"Stage Status: {stage_metrics['status']}")
print(f"Error Rate: {stage_metrics['error_rate']}%")
print(f"Processing Rate: {stage_metrics['processing_rate']}")
```

### 3. Alert Integration

```python
from alert_system import get_alert_system

# Get alert system instance
alert_system = get_alert_system()

# Evaluate metrics and generate alerts
stage_metrics = {
    'error_rate': 12.5,  # Above critical threshold
    'processing_time_avg': 8.2,  # Above warning threshold
    'schema_compliance_rate': 94.0  # Below warning threshold
}

alert_system.evaluate_metrics('K_knowledge_extraction', stage_metrics)

# Check active alerts
active_alerts = alert_system.get_active_alerts()
print(f"Active alerts: {len(active_alerts)}")
```

## Production Deployment

### 1. Environment Configuration

For production use:

1. **Configure SMTP** for email alerts in `alert_config.json`
2. **Set up Slack webhook** for team notifications  
3. **Configure webhook endpoints** for external monitoring systems
4. **Adjust alert thresholds** based on your SLA requirements
5. **Set up log rotation** for audit files

### 2. Security Considerations

- Use authentication for web dashboard access
- Secure API endpoints with API keys
- Configure firewall rules for monitoring ports
- Use HTTPS in production environments
- Implement role-based access for alert management

### 3. Performance Tuning

- Adjust metrics collection frequency based on load
- Configure audit log retention policies
- Use database backend for large-scale deployments
- Implement metrics aggregation for high-volume pipelines

### 4. High Availability

- Deploy monitoring system across multiple instances
- Use load balancer for dashboard access
- Implement failover for alert notifications
- Monitor the monitoring system itself

## Troubleshooting

### Common Issues

1. **Dashboard not loading**: Check canonical_web_server.py is running and accessible
2. **No metrics appearing**: Verify audit logs are being generated in canonical_flow/
3. **Alerts not triggering**: Check alert_config.json thresholds and notification settings
4. **High memory usage**: Adjust metrics retention and collection frequency

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Log Files

Monitor these log locations:
- Dashboard logs: Check console output from canonical_web_server.py
- Alert logs: Check console output for alert system messages
- Audit logs: canonical_flow/{stage_name}/*_audit.json

## Future Enhancements

### Planned Features

1. **Real-time WebSocket updates** for dashboard
2. **Metrics database backend** for historical analysis  
3. **Advanced alerting rules** with machine learning
4. **Performance prediction** based on historical trends
5. **Integration with external APM systems**
6. **Mobile-responsive dashboard interface**
7. **Grafana/Prometheus integration**
8. **Advanced data visualization** with charts and graphs

### Contributing

To contribute to the monitoring system:

1. Follow existing code style and patterns
2. Add comprehensive tests for new features
3. Update documentation for API changes
4. Ensure backward compatibility
5. Test with realistic data volumes

## License

This monitoring dashboard system is part of the EGW Query Expansion project and follows the same licensing terms.
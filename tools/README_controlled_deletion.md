# Controlled Deletion System

A sophisticated staged removal process for duplicate non-canonical directories identified in the canonical flow audit. This system provides safe, automated cleanup with comprehensive safeguards and monitoring.

## 🎯 Features

- **Embargo Mechanism**: Configurable grace periods (default 30 days) before deletion
- **Deprecation Warnings**: Automatic injection into modules in marked directories
- **Import Bans**: Static analysis enforcement through import-linter and custom AST walkers
- **Dead Code Detection**: Integration with vulture and custom reference analysis
- **Automated Verification**: Nightly scans for external references and usage patterns
- **Safe Deletion**: Zero-downtime removal with backup creation
- **CI/CD Integration**: GitHub Actions workflow with import violation detection
- **Notification System**: Email and webhook alerts for status updates

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install vulture import-linter schedule requests
```

### 2. Configure the System

Edit `tools/deletion_config.json` to set up your canonical mappings and notification preferences:

```json
{
  "default_grace_period_days": 30,
  "canonical_mapping": {
    "analysis_nlp": "canonical_flow/A_analysis_nlp",
    "retrieval_engine": "canonical_flow/R_search_retrieval"
  },
  "notification_emails": ["dev-team@example.com"]
}
```

### 3. Place a Directory Under Embargo

```bash
python -m tools.embargo_cli embargo analysis_nlp "Duplicate of canonical structure" --grace-period 30
```

### 4. Monitor Status

```bash
python -m tools.embargo_cli list
python -m tools.embargo_cli report
```

### 5. Check for Import Violations

```bash
python -m tools.embargo_cli check-imports --fail-on-violations
```

## 📋 Command Reference

### Embargo Management

```bash
# Place directory under embargo
embargo-cli embargo <directory> "<reason>" [--grace-period DAYS]

# List all embargoed directories
embargo-cli list

# Remove embargo (cancel deletion)
embargo-cli remove-embargo <directory>

# Extend embargo period
embargo-cli extend-embargo <directory> --days 14
```

### Scanning and Analysis

```bash
# Scan specific directory for external references
embargo-cli scan --directory <directory>

# Scan all embargoed directories
embargo-cli scan

# Check for deprecated imports
embargo-cli check-imports [--fail-on-violations]

# Run full nightly scan
embargo-cli nightly-scan
```

### Reporting

```bash
# Generate text report
embargo-cli report

# Generate JSON report
embargo-cli report --format json

# Show current configuration
embargo-cli config
```

### Deletion

```bash
# Preview deletions (dry run)
embargo-cli delete --dry-run

# Perform actual deletions
embargo-cli delete
```

## 🔧 System Components

### 1. Controlled Deletion Manager (`controlled_deletion_system.py`)

The core system that manages embargo records, tracks directory status, and coordinates all deletion activities.

**Key Features:**
- Embargo registry with JSON persistence
- Deprecation warning injection
- External reference tracking
- Safe deletion with backups

### 2. Import Linter Integration (`import_linter_config.py`)

Enforces import bans through static analysis tools.

**Components:**
- Import-linter configuration generation
- Custom AST-based import checking
- CI integration with JUnit XML output
- Detailed violation reporting

### 3. Nightly Scanner (`nightly_deletion_scanner.py`)

Automated service for regular scanning and maintenance.

**Capabilities:**
- Dead code analysis with vulture
- Custom reference analysis
- Notification dispatch
- Safe deletion execution
- Comprehensive reporting

### 4. CLI Interface (`embargo_cli.py`)

User-friendly command line interface for all operations.

**Commands:**
- Directory embargo management
- Status monitoring and reporting
- Import violation checking
- Manual deletion control

## 🤖 CI/CD Integration

The system includes GitHub Actions workflow (`.github/workflows/embargo_check.yml`) that:

1. **On Push/PR**: Checks for embargoed imports and fails CI if violations found
2. **Nightly**: Runs full scan and creates issues for directories ready for deletion
3. **Reporting**: Uploads test results and generates PR comments

### Setting up CI

1. The workflow is automatically active once the files are in your repository
2. Configure secrets for notifications:
   ```
   SLACK_WEBHOOK_URL  # Optional: for Slack notifications
   SMTP_PASSWORD      # Optional: for email notifications
   ```

3. Customize notification settings in `tools/deletion_config.json`

## 🔍 Analysis Tools Integration

### Vulture Integration

The system uses vulture for dead code detection:

```bash
# Manual vulture run
python -m vulture canonical_flow/A_analysis_nlp --min-confidence 60
```

### Import-Linter Integration

Automatically generates import-linter configuration:

```bash
# Generate configuration
python tools/import_linter_config.py generate-config

# Run import linter
python -m importlinter
```

## 📊 Monitoring and Notifications

### Email Notifications

Configure SMTP settings in `deletion_config.json`:

```json
{
  "notifications": {
    "smtp": {
      "host": "smtp.example.com",
      "port": 587,
      "use_tls": true,
      "from": "noreply@example.com"
    }
  }
}
```

### Webhook Notifications (Slack/Teams)

```json
{
  "notifications": {
    "webhook": {
      "url": "https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
    }
  }
}
```

### Reports Location

All reports are saved to `tools/deletion_reports/`:
- `embargo_summary.json` - Current status summary
- `nightly_scan_YYYYMMDD_HHMMSS.json` - Detailed scan results
- `<directory>_embargo_report.json` - Individual directory reports

## 🛡️ Safety Features

### Backup Creation

Before deletion, the system:
1. Creates tar.gz backup in `tools/deletion_reports/backups/`
2. Retains backups for configured retention period
3. Logs backup location in embargo record

### Multi-Stage Verification

1. **Embargo Phase**: Grace period with warnings
2. **Warning Phase**: External references detected
3. **Ready Phase**: No references, grace period expired
4. **Deletion Phase**: Safe removal with backup

### Rollback Capability

```bash
# Manual restore from backup (if needed)
cd tools/deletion_reports/backups
tar -xzf analysis_nlp_20240101_120000.tar.gz
```

## 🔧 Configuration Options

### Core Settings

```json
{
  "default_grace_period_days": 30,     // Default embargo period
  "enable_vulture": true,              // Use vulture for dead code detection
  "enable_import_linter": true,        // Use import-linter
  "auto_delete": false,                // Enable automatic deletion (use with caution)
  "canonical_mapping": {               // Map old paths to canonical equivalents
    "old_path": "canonical_path"
  }
}
```

### Scanning Settings

```json
{
  "scanning": {
    "vulture_min_confidence": 60,      // Minimum confidence for vulture
    "custom_analysis_enabled": true,   // Enable custom AST analysis
    "backup_before_deletion": true,    // Create backups before deletion
    "backup_retention_days": 90        // How long to keep backups
  }
}
```

### CI Integration Settings

```json
{
  "ci_integration": {
    "fail_on_deprecated_imports": true,  // Fail CI on import violations
    "generate_reports": true,            // Generate CI reports
    "junit_xml_output": true             // Create JUnit XML for test results
  }
}
```

## 🚨 Troubleshooting

### Common Issues

1. **Import violations not detected**
   - Check that embargoed directories are in the registry
   - Verify import-linter configuration is generated
   - Run manual check: `embargo-cli check-imports`

2. **Vulture not finding dead code**
   - Increase minimum confidence: `vulture_min_confidence: 40`
   - Check vulture is installed: `pip install vulture`

3. **Notifications not working**
   - Verify SMTP/webhook configuration
   - Check network connectivity
   - Test with manual send

4. **Directories not ready for deletion**
   - Check external references: `embargo-cli scan --directory <dir>`
   - Verify grace period has expired: `embargo-cli list`

### Debug Mode

Enable debug logging by setting environment variable:

```bash
export DELETION_SYSTEM_DEBUG=1
python -m tools.embargo_cli list
```

### Manual Recovery

If something goes wrong:

1. **Restore from backup**:
   ```bash
   tar -xzf tools/deletion_reports/backups/directory_backup.tar.gz
   ```

2. **Remove from embargo**:
   ```bash
   embargo-cli remove-embargo <directory>
   ```

3. **Reset system state**:
   ```bash
   rm tools/embargo_registry.json  # Clears all embargo records
   ```

## 📚 API Reference

### ControlledDeletionManager

```python
from tools.controlled_deletion_system import ControlledDeletionManager

manager = ControlledDeletionManager()

# Embargo a directory
manager.embargo_directory("analysis_nlp", "Duplicate structure", grace_period_days=30)

# Scan for references
results = manager.scan_embargoed_directories()

# Safe deletion
deleted = manager.safe_delete_ready_directories(dry_run=True)
```

### ImportAnalyzer

```python
from tools.import_linter_config import CustomASTImportChecker

checker = CustomASTImportChecker(manager)
violations = checker.check_project()
```

## 🤝 Contributing

To extend the system:

1. **Add new analysis tools**: Extend `DeadCodeAnalyzer` class
2. **Custom notification providers**: Extend `NotificationService` 
3. **Additional safety checks**: Add to `_generate_recommendations`
4. **New CLI commands**: Add to `EmbargoCLI` class

## 📄 License

This controlled deletion system is part of the EGW Query Expansion project and follows the same licensing terms.
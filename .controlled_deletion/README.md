# Controlled Deletion System

This directory contains the data and configuration files for the automated controlled deletion system.

## System Components

### Core System
- **controlled_deletion_system.py**: Main deletion system with embargo management
- **setup_deletion_cron.py**: Automated setup for cron jobs and systemd timers

### Key Features

1. **Duplicate Directory Detection**: Automatically scans for duplicate non-canonical directories
2. **30-Day Embargo Period**: Implements a waiting period with deprecation warnings
3. **Import Ban Enforcement**: Uses import-linter to prevent usage of deprecated paths
4. **Dead Code Verification**: Nightly scans using vulture to detect unused code
5. **Safe Deletion**: Automated deletion with backup and verification

## Directory Structure

```
.controlled_deletion/
├── README.md                 # This file
├── deletion_config.json      # System configuration
├── embargo_records.json      # Current embargo records
├── deletion_system.log       # System log file
├── backups/                  # Backup directory for deleted files
└── verification_report_*.json # Nightly verification reports
```

## Usage

### Initial Setup
```bash
# Install dependencies
pip install -r requirements-deletion-system.txt

# Setup automation (creates cron jobs or systemd timers)
python tools/setup_deletion_cron.py

# Manual setup without automation
python tools/setup_deletion_cron.py --no-automation
```

### Manual Operations

#### Scan for Duplicates
```bash
python tools/controlled_deletion_system.py --scan
```

#### Initiate Embargo for Found Duplicates
```bash
python tools/controlled_deletion_system.py --scan --embargo
```

#### Run Nightly Verification
```bash
python tools/controlled_deletion_system.py --verify
```

#### Execute Safe Deletion (Dry Run)
```bash
python tools/controlled_deletion_system.py --delete --dry-run
```

#### Execute Actual Deletion
```bash
python tools/controlled_deletion_system.py --delete
```

### Monitoring

#### Check System Health
```bash
python tools/deletion_monitor.py
```

#### View Current Embargo Records
```bash
cat .controlled_deletion/embargo_records.json
```

#### View Latest Verification Report
```bash
ls -la .controlled_deletion/verification_report_*.json | tail -1 | xargs cat
```

## Automated Schedule

When automation is set up, the system runs:

- **Weekly** (Sundays 1 AM): Scan for duplicates and initiate embargoes
- **Daily** (2 AM): Verify embargoed directories and check for deletion readiness  
- **Monthly** (1st day, 3 AM): Execute safe deletion of verified directories

## Configuration

Edit `.controlled_deletion/deletion_config.json` to customize:

```json
{
  "embargo_days": 30,
  "scan_frequency": "weekly",
  "verification_frequency": "daily", 
  "deletion_frequency": "monthly",
  "backup_retention_days": 90,
  "excluded_paths": [".git", ".venv", "__pycache__"],
  "canonical_path_prefixes": ["egw_query_expansion/", "canonical_flow/"],
  "dry_run_by_default": true
}
```

## Safety Features

1. **Backup Creation**: All deleted directories are backed up before deletion
2. **Import Analysis**: Checks for active imports before deletion
3. **Dead Code Detection**: Uses vulture to verify no active references
4. **Dry Run Mode**: Default dry-run prevents accidental deletions
5. **Embargo Period**: 30-day waiting period allows for intervention
6. **Deprecation Warnings**: Automatic warnings injected into deprecated modules

## Import Linter Integration

The system automatically adds import ban rules to `pyproject.toml`:

```toml
[tool.importlinter]
root_package = "egw_query_expansion"

[[tool.importlinter.contracts]]
name = "Deprecated module bans"
type = "forbidden"
source_modules = ["*"]
forbidden_modules = ["path.to.deprecated.module"]
```

Run import linting:
```bash
lint-imports
```

## Troubleshooting

### Common Issues

1. **Permission Errors**: Ensure write permissions to project directory
2. **Import Detection False Positives**: Review and manually exclude critical imports
3. **Cron Job Failures**: Check system logs and ensure Python environment is available
4. **Vulture Not Found**: Install with `pip install vulture`

### Manual Recovery

If a directory was incorrectly deleted:
```bash
# Restore from backup
cp -r .controlled_deletion/backups/path_to_directory path/to/directory
```

### Disable System

```bash
# Remove cron jobs
crontab -e  # Remove deletion system entries

# Disable systemd timers
systemctl --user disable deletion-verify.timer
systemctl --user disable deletion-scan.timer  
systemctl --user disable deletion-execute.timer
```

## Logs and Monitoring

- **System Log**: `.controlled_deletion/deletion_system.log`
- **Health Reports**: `.controlled_deletion/health_report_*.json`
- **Verification Reports**: `.controlled_deletion/verification_report_*.json`

Monitor system status:
```bash
tail -f .controlled_deletion/deletion_system.log
```
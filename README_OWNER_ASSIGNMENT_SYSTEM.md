# Owner Assignment System

A comprehensive system for automatically determining and managing component ownership through git blame analysis, team mapping integration, and governance UI controls.

## Features

### Automatic Ownership Assignment
- **Git Blame Analysis**: Analyzes git blame data to identify primary contributors for each file
- **Contributor Mapping**: Maps contributors to teams using configurable email patterns and explicit assignments
- **Confidence Scoring**: Provides confidence scores for ownership assignments based on contribution patterns
- **Bridge Registry Integration**: Uses existing bridge_registry team mapping patterns for automatic team assignments

### Manual Override System
- **Manual Assignment**: Allows explicit owner assignment through the ComponentRegistryService
- **Change Tracking**: Records ownership change history with timestamps, reasons, and change authors
- **Email Notifications**: Sends notifications to old and new owners when ownership changes occur
- **Override Justification**: Requires reasons for manual overrides and tracks who made changes

### SQL-Based Persistence
- **Component Ownership Table**: Stores current ownership information with confidence scores
- **Ownership History Table**: Maintains complete change history for audit purposes  
- **Team Mappings Table**: Configurable team definitions with email patterns and bridge phase mappings
- **SQLite Backend**: Lightweight, embedded database for easy deployment

### Governance UI Integration
- **Dashboard API**: Provides comprehensive dashboard data for governance interfaces
- **Search & Filter**: Advanced component search with team, owner, and override filters
- **Bulk Operations**: Support for bulk ownership reassignment operations
- **Team Management**: APIs for viewing and updating team mapping configurations

## Architecture

### Core Components

#### OwnerAssignmentSystem
- Main orchestrator for ownership determination and management
- Handles git blame analysis and contributor mapping
- Manages database persistence and change notifications

#### ComponentRegistryService  
- Service layer for component discovery and ownership assignment
- Provides APIs for both automated and manual ownership operations
- Integrates with bridge registry for component detection

#### GovernanceUIIntegration
- Web-friendly API layer for governance UI integration
- Provides dashboard data aggregation and component management endpoints
- Handles bulk operations and advanced search functionality

### Data Models

#### ContributorInfo
```python
@dataclass
class ContributorInfo:
    email: str
    name: str
    commits_count: int
    lines_contributed: int
    first_contribution: datetime
    last_contribution: datetime
    ownership_percentage: float
```

#### ComponentOwnership
```python
@dataclass
class ComponentOwnership:
    component_path: str
    primary_owner: str
    secondary_owners: List[str]
    team: Optional[str]
    confidence_score: float
    last_updated: datetime
    manual_override: bool = False
    override_reason: Optional[str] = None
```

#### TeamMapping
```python
@dataclass
class TeamMapping:
    team_id: str
    team_name: str
    email_patterns: List[str]
    contributors: List[str]
    bridge_phases: List[str]
```

## Installation & Setup

### Dependencies
```bash
# Core dependencies
pip install sqlite3  # Built-in with Python
pip install subprocess  # Built-in with Python

# Optional email notification dependencies
pip install smtplib  # Built-in with Python (for notifications)
```

### Database Setup
The system automatically creates the required SQLite tables on first use:
- `component_ownership` - Current ownership information
- `ownership_history` - Change audit trail
- `team_mappings` - Team configuration data

### Email Configuration (Optional)
```python
# Configure SMTP for email notifications
owner_system = OwnerAssignmentSystem()
owner_system.smtp_user = "your-smtp-user@domain.com"
owner_system.smtp_password = "your-smtp-password"
owner_system.smtp_server = "smtp.gmail.com"  
owner_system.smtp_port = 587
```

## Usage Examples

### Basic Ownership Assignment

```python
from canonical_flow.owner_assignment_system import OwnerAssignmentSystem
from pathlib import Path

# Initialize system
owner_system = OwnerAssignmentSystem()

# Automatically assign ownership for a component
component_path = Path("canonical_flow/my_component.py")
ownership = owner_system.assign_ownership_automatically(component_path)

print(f"Assigned to: {ownership.primary_owner}")
print(f"Team: {ownership.team}")
print(f"Confidence: {ownership.confidence_score}")
```

### Manual Override
```python
# Override ownership manually
success = owner_system.override_ownership_manually(
    component_path="canonical_flow/my_component.py",
    new_primary_owner="new.owner@example.com",
    new_team="new_team",
    reason="Organizational restructure",
    changed_by="admin@example.com"
)
```

### Bulk Assignment
```python
from canonical_flow.owner_assignment_system import ComponentRegistryService

# Discover and assign ownership for multiple components
registry_service = ComponentRegistryService(owner_system)
result = registry_service.discover_and_assign_components([
    "canonical_flow/",
    "src/components/"
])

print(f"Discovered: {result['discovered_components']} components")
print(f"Assigned: {result['assigned_ownership']} ownerships")
```

### Governance UI Integration
```python
from canonical_flow.governance_ui_integration import GovernanceUIIntegration

# Initialize governance integration
governance = GovernanceUIIntegration()

# Get dashboard data
dashboard_data = governance.get_dashboard_data()

# Search components
search_results = governance.search_components(
    query="processor",
    team_filter="analysis_team",
    manual_override_only=True
)

# Reassign ownership via UI
result = governance.reassign_ownership(
    component_path="path/to/component.py",
    new_owner="new.owner@example.com",
    new_team="new_team",
    reason="Team restructure",
    changed_by="ui_admin"
)
```

### API Endpoints for Web Integration
```python
from canonical_flow.governance_ui_integration import create_governance_api_routes

# Create API routes for web framework integration
api_routes = create_governance_api_routes()

# Available endpoints:
# - dashboard(): Get dashboard overview data
# - search_components(): Search and filter components
# - reassign_ownership(): Manual ownership reassignment
# - bulk_reassign(): Bulk ownership operations
# - discover_components(): Component discovery and assignment
# - component_details(): Detailed component information
# - team_details(): Team information and member lists
# - update_team(): Update team mapping configuration
```

## Team Mapping Configuration

The system includes default team mappings based on bridge registry phases:

- **ingestion_team**: Data ingestion and ETL components
- **context_team**: Context construction and semantic processing
- **knowledge_team**: Knowledge extraction and NLP
- **analysis_team**: Analysis and machine learning
- **evaluation_team**: Classification and evaluation
- **orchestration_team**: Pipeline orchestration and control
- **retrieval_team**: Search and retrieval systems
- **synthesis_team**: Synthesis and output generation
- **aggregation_team**: Aggregation and reporting
- **integration_team**: Integration and storage

### Customizing Team Mappings
```python
# Update team mapping via governance UI
result = governance.update_team_mapping(
    team_id="custom_team",
    team_name="Custom Development Team",
    email_patterns=["*custom.com", "*team@company.com"],
    contributors=["explicit.member@company.com"],
    bridge_phases=["CUSTOM_PHASE"]
)
```

## Monitoring & Health Reporting

### Ownership Health Report
```python
# Generate comprehensive health report
health_report = owner_system.get_ownership_health_report()

print(f"Total components: {health_report['total_components']}")
print(f"Average confidence: {health_report['avg_confidence_score']}")
print(f"Manual overrides: {health_report['manual_overrides']}")
print(f"Unassigned teams: {health_report['unassigned_teams']}")
print(f"Low confidence components: {health_report['low_confidence_components']}")
```

### Team Ownership Summary
```python
# Get ownership summary by team
team_summary = owner_system.get_team_ownership_summary()

for team_id, stats in team_summary.items():
    print(f"Team {team_id}:")
    print(f"  Components: {stats['component_count']}")
    print(f"  Avg Confidence: {stats['avg_confidence']}")
    print(f"  Manual Overrides: {stats['manual_overrides']}")
```

## Integration Points

### Bridge Registry Integration
The system integrates with the existing bridge_registry for:
- Team mapping patterns based on processing phases
- Component detection and classification
- Bridge metadata for ownership confidence scoring

### SQL Database Schema
```sql
-- Component ownership table
CREATE TABLE component_ownership (
    component_path TEXT PRIMARY KEY,
    primary_owner TEXT NOT NULL,
    secondary_owners TEXT,  -- JSON array
    team TEXT,
    confidence_score REAL NOT NULL,
    last_updated TIMESTAMP NOT NULL,
    manual_override BOOLEAN DEFAULT 0,
    override_reason TEXT
);

-- Ownership change history
CREATE TABLE ownership_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    component_path TEXT NOT NULL,
    old_owner TEXT,
    new_owner TEXT,
    old_team TEXT,
    new_team TEXT,
    change_timestamp TIMESTAMP NOT NULL,
    change_reason TEXT,
    changed_by TEXT
);

-- Team mapping configuration
CREATE TABLE team_mappings (
    team_id TEXT PRIMARY KEY,
    team_name TEXT NOT NULL,
    email_patterns TEXT,  -- JSON array
    contributors TEXT,    -- JSON array
    bridge_phases TEXT    -- JSON array
);
```

### Governance UI APIs
RESTful APIs for web-based governance interfaces:
- `GET /dashboard` - Dashboard overview data
- `GET /components/search` - Component search with filters
- `POST /components/reassign` - Manual ownership reassignment
- `POST /components/bulk-reassign` - Bulk ownership operations
- `POST /components/discover` - Component discovery and assignment
- `GET /components/{path}/details` - Detailed component information
- `GET /teams/{id}/details` - Team information
- `PUT /teams/{id}` - Update team mapping

## Testing

### Validation Script
```bash
# Run comprehensive validation tests
python validate_owner_assignment_system.py
```

### Unit Tests
```bash
# Run unit test suite
python -m pytest test_owner_assignment_system.py -v
```

The validation script tests:
- Database initialization and schema creation
- Team mapping initialization and retrieval
- Component ownership save/retrieve operations
- API endpoint creation and availability
- Bridge registry integration
- Component discovery functionality
- Governance UI integration

## Security Considerations

- **Email Credentials**: SMTP credentials should be stored securely (environment variables)
- **Database Access**: SQLite database should have appropriate file permissions
- **Git Access**: Requires read access to git repository for blame analysis
- **API Authentication**: Governance UI endpoints should include authentication middleware
- **Input Validation**: All user inputs are validated before database operations
- **SQL Injection Prevention**: Uses parameterized queries for all database operations

## Performance Considerations

- **Git Blame Caching**: Consider caching git blame results for large repositories
- **Database Indexing**: Automatic indexes on primary keys and foreign keys
- **Bulk Operations**: Optimized for processing large numbers of components
- **Background Processing**: Git blame analysis can be run asynchronously
- **Database Cleanup**: Includes optional history cleanup for long-running systems

## Future Enhancements

- **Machine Learning Integration**: Use ML for improved ownership prediction
- **GitHub/GitLab Integration**: Direct integration with Git hosting services
- **Slack/Teams Notifications**: Additional notification channels
- **Ownership Analytics**: Detailed analytics and ownership trends
- **Component Lifecycle**: Integration with component lifecycle management
- **Access Control**: Role-based access control for ownership changes
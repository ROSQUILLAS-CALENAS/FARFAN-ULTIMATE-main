"""
Owner Assignment System

Automatically determines component ownership by analyzing git blame data
to identify primary contributors, then maps to teams using bridge_registry
team mapping patterns. Provides manual override functionality with email
notifications when ownership changes occur.
"""

import subprocess
import json
import logging
import sqlite3
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter

# Email imports (optional for notifications)
try:
    import smtplib
    from email.mime.text import MimeText
    from email.mime.multipart import MimeMultipart
    EMAIL_AVAILABLE = True
except ImportError:
    EMAIL_AVAILABLE = False
    logging.warning("Email functionality not available - notifications will be disabled")

# Import bridge registry with fallback
try:
    from canonical_flow.bridge_registry import get_bridge_registry, BridgeRegistry
except ImportError as e:
    logging.warning(f"Bridge registry import failed: {e} - using mock implementation")
    
    class MockBridgeRegistry:
        def get_registry_info(self):
            return {"bridge_summary": {}, "total_bridges": 0}
    
    def get_bridge_registry():
        return MockBridgeRegistry()
    
    BridgeRegistry = MockBridgeRegistry

logger = logging.getLogger(__name__)


@dataclass
class ContributorInfo:
    """Information about a contributor"""
    email: str
    name: str
    commits_count: int
    lines_contributed: int
    first_contribution: datetime
    last_contribution: datetime
    ownership_percentage: float


@dataclass
class ComponentOwnership:
    """Component ownership information"""
    component_path: str
    primary_owner: str
    secondary_owners: List[str]
    team: Optional[str]
    confidence_score: float
    last_updated: datetime
    manual_override: bool = False
    override_reason: Optional[str] = None


@dataclass
class TeamMapping:
    """Team mapping information"""
    team_id: str
    team_name: str
    email_patterns: List[str]
    contributors: List[str]
    bridge_phases: List[str]


class OwnerAssignmentSystem:
    """
    System for determining component ownership through git blame analysis
    and team mapping integration with the bridge registry system.
    """
    
    def __init__(self, repo_path: str = ".", db_path: str = "component_ownership.db"):
        self.repo_path = Path(repo_path)
        self.db_path = db_path
        self.bridge_registry = get_bridge_registry()
        self._init_database()
        self._init_team_mappings()
        
        # Email configuration (to be set via environment variables)
        self.smtp_server = "smtp.gmail.com"
        self.smtp_port = 587
        self.smtp_user = None
        self.smtp_password = None
        
    def _init_database(self) -> None:
        """Initialize SQLite database for ownership persistence"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS component_ownership (
                    component_path TEXT PRIMARY KEY,
                    primary_owner TEXT NOT NULL,
                    secondary_owners TEXT,  -- JSON array
                    team TEXT,
                    confidence_score REAL NOT NULL,
                    last_updated TIMESTAMP NOT NULL,
                    manual_override BOOLEAN DEFAULT 0,
                    override_reason TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ownership_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    component_path TEXT NOT NULL,
                    old_owner TEXT,
                    new_owner TEXT,
                    old_team TEXT,
                    new_team TEXT,
                    change_timestamp TIMESTAMP NOT NULL,
                    change_reason TEXT,
                    changed_by TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS team_mappings (
                    team_id TEXT PRIMARY KEY,
                    team_name TEXT NOT NULL,
                    email_patterns TEXT,  -- JSON array
                    contributors TEXT,    -- JSON array
                    bridge_phases TEXT    -- JSON array
                )
            """)
            
    def _init_team_mappings(self) -> None:
        """Initialize team mappings based on bridge registry patterns"""
        # Default team mappings based on bridge registry phases
        default_teams = [
            TeamMapping(
                team_id="ingestion_team",
                team_name="Data Ingestion Team", 
                email_patterns=["*ingestion*", "*data*", "*etl*"],
                contributors=[],
                bridge_phases=["INGESTION_PREPARATION"]
            ),
            TeamMapping(
                team_id="context_team",
                team_name="Context Construction Team",
                email_patterns=["*context*", "*semantic*"],
                contributors=[],
                bridge_phases=["CONTEXT_CONSTRUCTION"]
            ),
            TeamMapping(
                team_id="knowledge_team",
                team_name="Knowledge Extraction Team",
                email_patterns=["*knowledge*", "*extraction*", "*nlp*"],
                contributors=[],
                bridge_phases=["KNOWLEDGE_EXTRACTION"]
            ),
            TeamMapping(
                team_id="analysis_team",
                team_name="Analysis & NLP Team",
                email_patterns=["*analysis*", "*nlp*", "*ml*"],
                contributors=[],
                bridge_phases=["ANALYSIS_NLP"]
            ),
            TeamMapping(
                team_id="evaluation_team", 
                team_name="Classification & Evaluation Team",
                email_patterns=["*eval*", "*classification*", "*scoring*"],
                contributors=[],
                bridge_phases=["CLASSIFICATION_EVALUATION"]
            ),
            TeamMapping(
                team_id="orchestration_team",
                team_name="Orchestration & Control Team",
                email_patterns=["*orchestration*", "*control*", "*pipeline*"],
                contributors=[],
                bridge_phases=["ORCHESTRATION_CONTROL"]
            ),
            TeamMapping(
                team_id="retrieval_team",
                team_name="Search & Retrieval Team", 
                email_patterns=["*search*", "*retrieval*", "*index*"],
                contributors=[],
                bridge_phases=["SEARCH_RETRIEVAL"]
            ),
            TeamMapping(
                team_id="synthesis_team",
                team_name="Synthesis & Output Team",
                email_patterns=["*synthesis*", "*output*", "*generation*"],
                contributors=[],
                bridge_phases=["SYNTHESIS_OUTPUT"]
            ),
            TeamMapping(
                team_id="aggregation_team",
                team_name="Aggregation & Reporting Team",
                email_patterns=["*aggregation*", "*reporting*", "*metrics*"],
                contributors=[],
                bridge_phases=["AGGREGATION_REPORTING"] 
            ),
            TeamMapping(
                team_id="integration_team",
                team_name="Integration & Storage Team",
                email_patterns=["*integration*", "*storage*", "*database*"],
                contributors=[],
                bridge_phases=["INTEGRATION_STORAGE"]
            )
        ]
        
        # Store team mappings in database
        with sqlite3.connect(self.db_path) as conn:
            for team in default_teams:
                conn.execute("""
                    INSERT OR REPLACE INTO team_mappings 
                    (team_id, team_name, email_patterns, contributors, bridge_phases)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    team.team_id,
                    team.team_name,
                    json.dumps(team.email_patterns),
                    json.dumps(team.contributors),
                    json.dumps(team.bridge_phases)
                ))
                
    def analyze_git_blame(self, file_path: Path) -> List[ContributorInfo]:
        """Analyze git blame data to identify primary contributors"""
        try:
            # Run git blame to get line-by-line authorship
            result = subprocess.run([
                'git', 'blame', '--porcelain', str(file_path)
            ], capture_output=True, text=True, cwd=self.repo_path)
            
            if result.returncode != 0:
                logger.warning(f"Git blame failed for {file_path}: {result.stderr}")
                return []
                
            # Parse git blame output
            contributor_stats = defaultdict(lambda: {
                'commits': set(),
                'lines': 0,
                'first_date': None,
                'last_date': None,
                'name': '',
                'email': ''
            })
            
            lines = result.stdout.split('\n')
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                if not line:
                    i += 1
                    continue
                    
                # Parse commit hash and other info
                parts = line.split()
                if len(parts) < 2:
                    i += 1
                    continue
                    
                commit_hash = parts[0]
                
                # Look for author info in next lines
                author_name = ""
                author_email = ""
                author_time = None
                
                j = i + 1
                while j < len(lines) and not lines[j].startswith(commit_hash):
                    if lines[j].startswith('author '):
                        author_name = lines[j][7:]
                    elif lines[j].startswith('author-mail '):
                        author_email = lines[j][12:].strip('<>')
                    elif lines[j].startswith('author-time '):
                        try:
                            timestamp = int(lines[j][12:])
                            author_time = datetime.fromtimestamp(timestamp)
                        except (ValueError, OSError):
                            pass
                    elif lines[j].startswith('\t'):
                        # This is the actual code line
                        break
                    j += 1
                
                if author_email and author_name:
                    stats = contributor_stats[author_email]
                    stats['commits'].add(commit_hash)
                    stats['lines'] += 1
                    stats['name'] = author_name
                    stats['email'] = author_email
                    
                    if author_time:
                        if not stats['first_date'] or author_time < stats['first_date']:
                            stats['first_date'] = author_time
                        if not stats['last_date'] or author_time > stats['last_date']:
                            stats['last_date'] = author_time
                            
                i = j + 1
                
            # Convert to ContributorInfo objects
            contributors = []
            total_lines = sum(stats['lines'] for stats in contributor_stats.values())
            
            for email, stats in contributor_stats.items():
                if stats['lines'] > 0:  # Only include contributors with actual lines
                    ownership_percentage = stats['lines'] / total_lines if total_lines > 0 else 0
                    contributors.append(ContributorInfo(
                        email=email,
                        name=stats['name'],
                        commits_count=len(stats['commits']),
                        lines_contributed=stats['lines'],
                        first_contribution=stats['first_date'] or datetime.now(),
                        last_contribution=stats['last_date'] or datetime.now(),
                        ownership_percentage=ownership_percentage
                    ))
                    
            # Sort by ownership percentage (most contributing first)
            contributors.sort(key=lambda x: x.ownership_percentage, reverse=True)
            return contributors
            
        except Exception as e:
            logger.error(f"Failed to analyze git blame for {file_path}: {e}")
            return []
    
    def map_contributor_to_team(self, contributor: ContributorInfo) -> Optional[str]:
        """Map contributor to team based on email patterns and bridge registry"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT * FROM team_mappings")
                teams = cursor.fetchall()
                
            for team_row in teams:
                team_id, team_name, email_patterns_json, contributors_json, bridge_phases_json = team_row
                
                email_patterns = json.loads(email_patterns_json)
                contributors_list = json.loads(contributors_json)
                
                # Check if contributor is explicitly listed
                if contributor.email in contributors_list:
                    return team_id
                    
                # Check email patterns
                for pattern in email_patterns:
                    pattern_regex = pattern.replace('*', '.*')
                    if re.search(pattern_regex, contributor.email, re.IGNORECASE):
                        return team_id
                        
            return None
            
        except Exception as e:
            logger.error(f"Failed to map contributor {contributor.email} to team: {e}")
            return None
    
    def determine_component_ownership(self, component_path: Path) -> Optional[ComponentOwnership]:
        """Determine ownership for a component based on git blame analysis"""
        if not component_path.exists():
            logger.warning(f"Component path does not exist: {component_path}")
            return None
            
        # Analyze git blame
        contributors = self.analyze_git_blame(component_path)
        if not contributors:
            logger.warning(f"No contributors found for {component_path}")
            return None
            
        # Primary owner is the contributor with highest ownership percentage
        primary_contributor = contributors[0]
        
        # Secondary owners are others with significant contribution (>10%)
        secondary_contributors = [
            c for c in contributors[1:] 
            if c.ownership_percentage > 0.10
        ]
        
        # Map primary contributor to team
        team = self.map_contributor_to_team(primary_contributor)
        
        # Calculate confidence score based on ownership distribution
        confidence_score = primary_contributor.ownership_percentage
        
        # If primary owner has less than 30%, reduce confidence
        if confidence_score < 0.30:
            confidence_score *= 0.7
            
        # If there are many secondary owners, reduce confidence
        if len(secondary_contributors) > 3:
            confidence_score *= 0.8
            
        return ComponentOwnership(
            component_path=str(component_path),
            primary_owner=primary_contributor.email,
            secondary_owners=[c.email for c in secondary_contributors],
            team=team,
            confidence_score=min(confidence_score, 1.0),
            last_updated=datetime.now(),
            manual_override=False
        )
    
    def save_ownership(self, ownership: ComponentOwnership) -> None:
        """Save ownership information to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO component_ownership 
                    (component_path, primary_owner, secondary_owners, team, 
                     confidence_score, last_updated, manual_override, override_reason)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    ownership.component_path,
                    ownership.primary_owner,
                    json.dumps(ownership.secondary_owners),
                    ownership.team,
                    ownership.confidence_score,
                    ownership.last_updated,
                    ownership.manual_override,
                    ownership.override_reason
                ))
                
        except Exception as e:
            logger.error(f"Failed to save ownership for {ownership.component_path}: {e}")
    
    def get_ownership(self, component_path: str) -> Optional[ComponentOwnership]:
        """Retrieve ownership information from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT component_path, primary_owner, secondary_owners, team,
                           confidence_score, last_updated, manual_override, override_reason
                    FROM component_ownership WHERE component_path = ?
                """, (component_path,))
                
                row = cursor.fetchone()
                if row:
                    return ComponentOwnership(
                        component_path=row[0],
                        primary_owner=row[1],
                        secondary_owners=json.loads(row[2] or '[]'),
                        team=row[3],
                        confidence_score=row[4],
                        last_updated=datetime.fromisoformat(row[5]),
                        manual_override=bool(row[6]),
                        override_reason=row[7]
                    )
                return None
                
        except Exception as e:
            logger.error(f"Failed to get ownership for {component_path}: {e}")
            return None
    
    def assign_ownership_automatically(self, component_path: Path) -> Optional[ComponentOwnership]:
        """Automatically assign ownership during component discovery"""
        ownership = self.determine_component_ownership(component_path)
        if ownership:
            self.save_ownership(ownership)
            logger.info(f"Automatically assigned ownership for {component_path} to {ownership.primary_owner}")
        return ownership
    
    def override_ownership_manually(
        self,
        component_path: str,
        new_primary_owner: str,
        new_team: Optional[str] = None,
        reason: str = "",
        changed_by: str = ""
    ) -> bool:
        """Manually override component ownership"""
        try:
            # Get current ownership
            current_ownership = self.get_ownership(component_path)
            old_owner = current_ownership.primary_owner if current_ownership else None
            old_team = current_ownership.team if current_ownership else None
            
            # Create new ownership record
            new_ownership = ComponentOwnership(
                component_path=component_path,
                primary_owner=new_primary_owner,
                secondary_owners=current_ownership.secondary_owners if current_ownership else [],
                team=new_team,
                confidence_score=1.0,  # Manual override has maximum confidence
                last_updated=datetime.now(),
                manual_override=True,
                override_reason=reason
            )
            
            # Save to database
            self.save_ownership(new_ownership)
            
            # Record change history
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO ownership_history
                    (component_path, old_owner, new_owner, old_team, new_team,
                     change_timestamp, change_reason, changed_by)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    component_path, old_owner, new_primary_owner, old_team, new_team,
                    datetime.now(), reason, changed_by
                ))
            
            # Send email notifications
            self._send_ownership_change_notification(
                component_path, old_owner, new_primary_owner, old_team, new_team, reason
            )
            
            logger.info(f"Manually overrode ownership for {component_path}: {old_owner} -> {new_primary_owner}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to override ownership for {component_path}: {e}")
            return False
    
    def _send_ownership_change_notification(
        self,
        component_path: str,
        old_owner: Optional[str],
        new_owner: str,
        old_team: Optional[str],
        new_team: Optional[str],
        reason: str
    ) -> None:
        """Send email notification about ownership change"""
        if not EMAIL_AVAILABLE:
            logger.info(f"Email not available - would notify about ownership change: {component_path} -> {new_owner}")
            return
            
        if not self.smtp_user or not self.smtp_password:
            logger.warning("SMTP credentials not configured, skipping email notification")
            return
            
        try:
            # Create email message
            msg = MimeMultipart()
            msg['From'] = self.smtp_user
            msg['Subject'] = f"Component Ownership Change: {component_path}"
            
            # Recipients: old owner, new owner, and team leads
            recipients = []
            if old_owner and '@' in old_owner:
                recipients.append(old_owner)
            if new_owner and '@' in new_owner:
                recipients.append(new_owner)
                
            msg['To'] = ', '.join(recipients)
            
            # Email body
            body = f"""
Component Ownership Change Notification

Component: {component_path}
Old Owner: {old_owner or 'None'}
New Owner: {new_owner}
Old Team: {old_team or 'None'} 
New Team: {new_team or 'None'}
Reason: {reason}

Changed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

This is an automated notification from the Owner Assignment System.
"""
            
            msg.attach(MimeText(body, 'plain'))
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_password)
                server.send_message(msg)
                
            logger.info(f"Sent ownership change notification for {component_path}")
            
        except Exception as e:
            logger.error(f"Failed to send ownership change notification: {e}")
    
    def bulk_assign_ownership(self, component_paths: List[Path]) -> Dict[str, ComponentOwnership]:
        """Bulk assign ownership for multiple components"""
        results = {}
        for path in component_paths:
            try:
                ownership = self.assign_ownership_automatically(path)
                if ownership:
                    results[str(path)] = ownership
            except Exception as e:
                logger.error(f"Failed to assign ownership for {path}: {e}")
                
        logger.info(f"Bulk assigned ownership for {len(results)}/{len(component_paths)} components")
        return results
    
    def get_team_ownership_summary(self) -> Dict[str, Any]:
        """Get ownership summary by team"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT team, COUNT(*) as component_count,
                           AVG(confidence_score) as avg_confidence,
                           SUM(CASE WHEN manual_override = 1 THEN 1 ELSE 0 END) as manual_overrides
                    FROM component_ownership
                    WHERE team IS NOT NULL
                    GROUP BY team
                """)
                
                team_stats = {}
                for row in cursor.fetchall():
                    team_stats[row[0]] = {
                        'component_count': row[1],
                        'avg_confidence': round(row[2], 3),
                        'manual_overrides': row[3]
                    }
                    
                return team_stats
                
        except Exception as e:
            logger.error(f"Failed to get team ownership summary: {e}")
            return {}
    
    def get_ownership_health_report(self) -> Dict[str, Any]:
        """Generate ownership health report"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Overall statistics
                cursor = conn.execute("""
                    SELECT 
                        COUNT(*) as total_components,
                        AVG(confidence_score) as avg_confidence,
                        SUM(CASE WHEN manual_override = 1 THEN 1 ELSE 0 END) as manual_overrides,
                        SUM(CASE WHEN team IS NULL THEN 1 ELSE 0 END) as unassigned_teams,
                        SUM(CASE WHEN confidence_score < 0.5 THEN 1 ELSE 0 END) as low_confidence
                    FROM component_ownership
                """)
                
                stats = cursor.fetchone()
                
                return {
                    'total_components': stats[0],
                    'avg_confidence_score': round(stats[1] or 0, 3),
                    'manual_overrides': stats[2],
                    'unassigned_teams': stats[3],
                    'low_confidence_components': stats[4],
                    'team_distribution': self.get_team_ownership_summary(),
                    'report_timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Failed to generate ownership health report: {e}")
            return {}


class ComponentRegistryService:
    """
    Service for managing component registry with ownership integration
    """
    
    def __init__(self, owner_assignment_system: OwnerAssignmentSystem):
        self.owner_system = owner_assignment_system
        self.bridge_registry = get_bridge_registry()
        
    def discover_and_assign_components(self, scan_paths: List[str]) -> Dict[str, Any]:
        """Discover components and automatically assign ownership"""
        discovered_components = []
        ownership_results = {}
        
        for scan_path in scan_paths:
            path_obj = Path(scan_path)
            if path_obj.is_dir():
                # Scan directory for Python files
                for py_file in path_obj.rglob("*.py"):
                    if self._is_component_file(py_file):
                        discovered_components.append(py_file)
            elif path_obj.is_file() and path_obj.suffix == '.py':
                if self._is_component_file(path_obj):
                    discovered_components.append(path_obj)
                    
        # Assign ownership for discovered components
        ownership_results = self.owner_system.bulk_assign_ownership(discovered_components)
        
        return {
            'discovered_components': len(discovered_components),
            'assigned_ownership': len(ownership_results),
            'results': ownership_results
        }
    
    def _is_component_file(self, file_path: Path) -> bool:
        """Check if file is a pipeline component"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Check for pipeline component patterns
            component_patterns = [
                r'__phase__\s*=',
                r'__code__\s*=',
                r'__stage_order__\s*=',
                r'class.*Processor',
                r'class.*Engine',
                r'def process\('
            ]
            
            return any(re.search(pattern, content) for pattern in component_patterns)
            
        except Exception:
            return False
    
    def reassign_component_owner(
        self,
        component_path: str,
        new_owner: str,
        new_team: Optional[str] = None,
        reason: str = "",
        changed_by: str = ""
    ) -> Dict[str, Any]:
        """API endpoint for manual ownership reassignment"""
        success = self.owner_system.override_ownership_manually(
            component_path=component_path,
            new_primary_owner=new_owner,
            new_team=new_team,
            reason=reason,
            changed_by=changed_by
        )
        
        return {
            'success': success,
            'component_path': component_path,
            'new_owner': new_owner,
            'new_team': new_team,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_component_ownership_info(self, component_path: str) -> Dict[str, Any]:
        """Get ownership information for a component"""
        ownership = self.owner_system.get_ownership(component_path)
        
        if ownership:
            return {
                'found': True,
                'ownership': asdict(ownership)
            }
        else:
            return {
                'found': False,
                'message': 'Component ownership not found'
            }
    
    def get_ownership_dashboard_data(self) -> Dict[str, Any]:
        """Get data for governance UI dashboard"""
        health_report = self.owner_system.get_ownership_health_report()
        team_summary = self.owner_system.get_team_ownership_summary()
        
        return {
            'health_report': health_report,
            'team_summary': team_summary,
            'bridge_registry_info': self.bridge_registry.get_registry_info()
        }


# API for governance UI integration

def create_owner_assignment_api():
    """Create owner assignment system API for governance UI"""
    owner_system = OwnerAssignmentSystem()
    registry_service = ComponentRegistryService(owner_system)
    
    def assign_ownership_endpoint(component_paths: List[str]):
        return registry_service.discover_and_assign_components(component_paths)
    
    def reassign_ownership_endpoint(
        component_path: str,
        new_owner: str,
        new_team: str = None,
        reason: str = "",
        changed_by: str = ""
    ):
        return registry_service.reassign_component_owner(
            component_path, new_owner, new_team, reason, changed_by
        )
    
    def get_ownership_info_endpoint(component_path: str):
        return registry_service.get_component_ownership_info(component_path)
    
    def get_dashboard_data_endpoint():
        return registry_service.get_ownership_dashboard_data()
    
    return {
        'assign_ownership': assign_ownership_endpoint,
        'reassign_ownership': reassign_ownership_endpoint,
        'get_ownership_info': get_ownership_info_endpoint,
        'get_dashboard_data': get_dashboard_data_endpoint
    }
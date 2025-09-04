"""
Governance UI Integration for Owner Assignment System

Provides web API endpoints and dashboard data for the governance UI
to manage component ownership and team assignments.
"""

import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

from canonical_flow.owner_assignment_system import (
    OwnerAssignmentSystem, 
    ComponentRegistryService,
    create_owner_assignment_api
)

logger = logging.getLogger(__name__)


class GovernanceUIIntegration:
    """
    Integration layer for governance UI to interact with owner assignment system
    """
    
    def __init__(self, owner_assignment_system: Optional[OwnerAssignmentSystem] = None):
        self.owner_system = owner_assignment_system or OwnerAssignmentSystem()
        self.registry_service = ComponentRegistryService(self.owner_system)
        self.api = create_owner_assignment_api()
        
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data for governance UI"""
        try:
            health_report = self.owner_system.get_ownership_health_report()
            team_summary = self.owner_system.get_team_ownership_summary()
            
            # Get component list with ownership info
            components = self._get_all_components_with_ownership()
            
            # Get recent ownership changes
            recent_changes = self._get_recent_ownership_changes()
            
            return {
                'status': 'success',
                'data': {
                    'overview': health_report,
                    'team_summary': team_summary,
                    'components': components,
                    'recent_changes': recent_changes,
                    'alerts': self._generate_alerts(health_report),
                    'last_updated': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get dashboard data: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def search_components(
        self, 
        query: str = "", 
        team_filter: Optional[str] = None,
        owner_filter: Optional[str] = None,
        manual_override_only: bool = False
    ) -> Dict[str, Any]:
        """Search components with filters"""
        try:
            components = self._get_all_components_with_ownership()
            
            # Apply filters
            filtered = []
            for component in components:
                if query and query.lower() not in component['component_path'].lower():
                    continue
                if team_filter and component['team'] != team_filter:
                    continue
                if owner_filter and component['primary_owner'] != owner_filter:
                    continue
                if manual_override_only and not component['manual_override']:
                    continue
                    
                filtered.append(component)
            
            return {
                'status': 'success',
                'data': {
                    'components': filtered,
                    'total_count': len(filtered),
                    'query': query,
                    'filters': {
                        'team': team_filter,
                        'owner': owner_filter,
                        'manual_override_only': manual_override_only
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to search components: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def reassign_ownership(
        self,
        component_path: str,
        new_owner: str,
        new_team: Optional[str] = None,
        reason: str = "",
        changed_by: str = "ui_user"
    ) -> Dict[str, Any]:
        """Reassign component ownership via UI"""
        try:
            success = self.owner_system.override_ownership_manually(
                component_path=component_path,
                new_primary_owner=new_owner,
                new_team=new_team,
                reason=reason,
                changed_by=changed_by
            )
            
            if success:
                return {
                    'status': 'success',
                    'message': f'Successfully reassigned {component_path} to {new_owner}',
                    'data': {
                        'component_path': component_path,
                        'new_owner': new_owner,
                        'new_team': new_team,
                        'timestamp': datetime.now().isoformat()
                    }
                }
            else:
                return {
                    'status': 'error',
                    'message': 'Failed to reassign ownership'
                }
                
        except Exception as e:
            logger.error(f"Failed to reassign ownership for {component_path}: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def bulk_reassign_ownership(
        self,
        assignments: List[Dict[str, Any]],
        changed_by: str = "ui_user"
    ) -> Dict[str, Any]:
        """Bulk reassign ownership for multiple components"""
        try:
            results = []
            success_count = 0
            
            for assignment in assignments:
                component_path = assignment['component_path']
                new_owner = assignment['new_owner']
                new_team = assignment.get('new_team')
                reason = assignment.get('reason', 'Bulk reassignment')
                
                success = self.owner_system.override_ownership_manually(
                    component_path=component_path,
                    new_primary_owner=new_owner,
                    new_team=new_team,
                    reason=reason,
                    changed_by=changed_by
                )
                
                results.append({
                    'component_path': component_path,
                    'success': success,
                    'new_owner': new_owner,
                    'new_team': new_team
                })
                
                if success:
                    success_count += 1
            
            return {
                'status': 'success',
                'data': {
                    'total_assignments': len(assignments),
                    'successful_assignments': success_count,
                    'failed_assignments': len(assignments) - success_count,
                    'results': results
                }
            }
            
        except Exception as e:
            logger.error(f"Failed bulk reassignment: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def discover_new_components(
        self,
        scan_paths: List[str]
    ) -> Dict[str, Any]:
        """Discover and assign ownership to new components"""
        try:
            result = self.registry_service.discover_and_assign_components(scan_paths)
            
            return {
                'status': 'success',
                'data': {
                    'discovered_components': result['discovered_components'],
                    'assigned_ownership': result['assigned_ownership'],
                    'scan_paths': scan_paths,
                    'timestamp': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to discover new components: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def get_component_details(self, component_path: str) -> Dict[str, Any]:
        """Get detailed information about a specific component"""
        try:
            ownership = self.owner_system.get_ownership(component_path)
            
            if not ownership:
                return {
                    'status': 'error',
                    'message': 'Component not found'
                }
            
            # Get git blame analysis for detailed contributor info
            path_obj = Path(component_path)
            if path_obj.exists():
                contributors = self.owner_system.analyze_git_blame(path_obj)
            else:
                contributors = []
            
            # Get ownership history
            history = self._get_component_ownership_history(component_path)
            
            return {
                'status': 'success',
                'data': {
                    'ownership': {
                        'component_path': ownership.component_path,
                        'primary_owner': ownership.primary_owner,
                        'secondary_owners': ownership.secondary_owners,
                        'team': ownership.team,
                        'confidence_score': ownership.confidence_score,
                        'last_updated': ownership.last_updated.isoformat(),
                        'manual_override': ownership.manual_override,
                        'override_reason': ownership.override_reason
                    },
                    'contributors': [
                        {
                            'email': c.email,
                            'name': c.name,
                            'commits_count': c.commits_count,
                            'lines_contributed': c.lines_contributed,
                            'ownership_percentage': c.ownership_percentage,
                            'first_contribution': c.first_contribution.isoformat(),
                            'last_contribution': c.last_contribution.isoformat()
                        } for c in contributors
                    ],
                    'history': history
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get component details for {component_path}: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def get_team_details(self, team_id: str) -> Dict[str, Any]:
        """Get detailed information about a team"""
        try:
            import sqlite3
            
            with sqlite3.connect(self.owner_system.db_path) as conn:
                # Get team info
                cursor = conn.execute("""
                    SELECT team_name, email_patterns, contributors, bridge_phases
                    FROM team_mappings WHERE team_id = ?
                """, (team_id,))
                
                team_row = cursor.fetchone()
                if not team_row:
                    return {
                        'status': 'error',
                        'message': 'Team not found'
                    }
                
                team_name, email_patterns_json, contributors_json, bridge_phases_json = team_row
                
                # Get team components
                cursor = conn.execute("""
                    SELECT component_path, primary_owner, confidence_score, manual_override
                    FROM component_ownership WHERE team = ?
                    ORDER BY last_updated DESC
                """, (team_id,))
                
                components = [
                    {
                        'component_path': row[0],
                        'primary_owner': row[1],
                        'confidence_score': row[2],
                        'manual_override': bool(row[3])
                    } for row in cursor.fetchall()
                ]
            
            return {
                'status': 'success',
                'data': {
                    'team_id': team_id,
                    'team_name': team_name,
                    'email_patterns': json.loads(email_patterns_json),
                    'contributors': json.loads(contributors_json),
                    'bridge_phases': json.loads(bridge_phases_json),
                    'components': components,
                    'component_count': len(components)
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get team details for {team_id}: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def update_team_mapping(
        self,
        team_id: str,
        team_name: Optional[str] = None,
        email_patterns: Optional[List[str]] = None,
        contributors: Optional[List[str]] = None,
        bridge_phases: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Update team mapping configuration"""
        try:
            import sqlite3
            
            with sqlite3.connect(self.owner_system.db_path) as conn:
                # Get current team data
                cursor = conn.execute("""
                    SELECT team_name, email_patterns, contributors, bridge_phases
                    FROM team_mappings WHERE team_id = ?
                """, (team_id,))
                
                current_row = cursor.fetchone()
                if not current_row:
                    return {
                        'status': 'error',
                        'message': 'Team not found'
                    }
                
                current_name, current_patterns, current_contributors, current_phases = current_row
                
                # Update with provided values or keep current
                updated_name = team_name or current_name
                updated_patterns = json.dumps(email_patterns or json.loads(current_patterns))
                updated_contributors = json.dumps(contributors or json.loads(current_contributors))
                updated_phases = json.dumps(bridge_phases or json.loads(current_phases))
                
                # Update database
                conn.execute("""
                    UPDATE team_mappings 
                    SET team_name = ?, email_patterns = ?, contributors = ?, bridge_phases = ?
                    WHERE team_id = ?
                """, (updated_name, updated_patterns, updated_contributors, updated_phases, team_id))
            
            return {
                'status': 'success',
                'message': f'Successfully updated team {team_id}',
                'data': {
                    'team_id': team_id,
                    'team_name': updated_name,
                    'timestamp': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to update team mapping for {team_id}: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def _get_all_components_with_ownership(self) -> List[Dict[str, Any]]:
        """Get all components with ownership information"""
        import sqlite3
        
        components = []
        
        try:
            with sqlite3.connect(self.owner_system.db_path) as conn:
                cursor = conn.execute("""
                    SELECT component_path, primary_owner, secondary_owners, team,
                           confidence_score, last_updated, manual_override, override_reason
                    FROM component_ownership
                    ORDER BY last_updated DESC
                """)
                
                for row in cursor.fetchall():
                    components.append({
                        'component_path': row[0],
                        'primary_owner': row[1],
                        'secondary_owners': json.loads(row[2] or '[]'),
                        'team': row[3],
                        'confidence_score': row[4],
                        'last_updated': row[5],
                        'manual_override': bool(row[6]),
                        'override_reason': row[7]
                    })
                    
        except Exception as e:
            logger.error(f"Failed to get components with ownership: {e}")
            
        return components
    
    def _get_recent_ownership_changes(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent ownership changes"""
        import sqlite3
        
        changes = []
        
        try:
            with sqlite3.connect(self.owner_system.db_path) as conn:
                cursor = conn.execute("""
                    SELECT component_path, old_owner, new_owner, old_team, new_team,
                           change_timestamp, change_reason, changed_by
                    FROM ownership_history
                    ORDER BY change_timestamp DESC
                    LIMIT ?
                """, (limit,))
                
                for row in cursor.fetchall():
                    changes.append({
                        'component_path': row[0],
                        'old_owner': row[1],
                        'new_owner': row[2],
                        'old_team': row[3],
                        'new_team': row[4],
                        'change_timestamp': row[5],
                        'change_reason': row[6],
                        'changed_by': row[7]
                    })
                    
        except Exception as e:
            logger.error(f"Failed to get recent ownership changes: {e}")
            
        return changes
    
    def _get_component_ownership_history(self, component_path: str) -> List[Dict[str, Any]]:
        """Get ownership history for a specific component"""
        import sqlite3
        
        history = []
        
        try:
            with sqlite3.connect(self.owner_system.db_path) as conn:
                cursor = conn.execute("""
                    SELECT old_owner, new_owner, old_team, new_team,
                           change_timestamp, change_reason, changed_by
                    FROM ownership_history
                    WHERE component_path = ?
                    ORDER BY change_timestamp DESC
                """, (component_path,))
                
                for row in cursor.fetchall():
                    history.append({
                        'old_owner': row[0],
                        'new_owner': row[1],
                        'old_team': row[2],
                        'new_team': row[3],
                        'change_timestamp': row[4],
                        'change_reason': row[5],
                        'changed_by': row[6]
                    })
                    
        except Exception as e:
            logger.error(f"Failed to get component ownership history: {e}")
            
        return history
    
    def _generate_alerts(self, health_report: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate alerts based on health report"""
        alerts = []
        
        # Alert for low confidence components
        if health_report.get('low_confidence_components', 0) > 0:
            alerts.append({
                'type': 'warning',
                'message': f"{health_report['low_confidence_components']} components have low ownership confidence",
                'action': 'Review and consider manual assignment'
            })
        
        # Alert for unassigned teams
        if health_report.get('unassigned_teams', 0) > 0:
            alerts.append({
                'type': 'warning',
                'message': f"{health_report['unassigned_teams']} components have no team assignment",
                'action': 'Update team mappings or manually assign teams'
            })
        
        # Alert for high manual override ratio
        total_components = health_report.get('total_components', 0)
        manual_overrides = health_report.get('manual_overrides', 0)
        
        if total_components > 0 and (manual_overrides / total_components) > 0.3:
            alerts.append({
                'type': 'info',
                'message': f"High ratio of manual overrides ({manual_overrides}/{total_components})",
                'action': 'Consider updating team mapping patterns'
            })
        
        return alerts


# REST API wrapper functions for web framework integration

def create_governance_api_routes():
    """Create API routes for web framework integration"""
    governance = GovernanceUIIntegration()
    
    def dashboard_endpoint():
        return governance.get_dashboard_data()
    
    def search_components_endpoint(
        query: str = "",
        team_filter: str = None,
        owner_filter: str = None,
        manual_override_only: bool = False
    ):
        return governance.search_components(query, team_filter, owner_filter, manual_override_only)
    
    def reassign_ownership_endpoint(
        component_path: str,
        new_owner: str,
        new_team: str = None,
        reason: str = "",
        changed_by: str = "api_user"
    ):
        return governance.reassign_ownership(component_path, new_owner, new_team, reason, changed_by)
    
    def bulk_reassign_endpoint(assignments: List[Dict[str, Any]], changed_by: str = "api_user"):
        return governance.bulk_reassign_ownership(assignments, changed_by)
    
    def discover_components_endpoint(scan_paths: List[str]):
        return governance.discover_new_components(scan_paths)
    
    def component_details_endpoint(component_path: str):
        return governance.get_component_details(component_path)
    
    def team_details_endpoint(team_id: str):
        return governance.get_team_details(team_id)
    
    def update_team_endpoint(
        team_id: str,
        team_name: str = None,
        email_patterns: List[str] = None,
        contributors: List[str] = None,
        bridge_phases: List[str] = None
    ):
        return governance.update_team_mapping(team_id, team_name, email_patterns, contributors, bridge_phases)
    
    return {
        'dashboard': dashboard_endpoint,
        'search_components': search_components_endpoint,
        'reassign_ownership': reassign_ownership_endpoint,
        'bulk_reassign': bulk_reassign_endpoint,
        'discover_components': discover_components_endpoint,
        'component_details': component_details_endpoint,
        'team_details': team_details_endpoint,
        'update_team': update_team_endpoint
    }
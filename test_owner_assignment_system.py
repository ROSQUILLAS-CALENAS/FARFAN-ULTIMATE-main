"""
Test suite for Owner Assignment System
"""

import pytest
import sqlite3
import tempfile
import json
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, MagicMock

from canonical_flow.owner_assignment_system import (
    OwnerAssignmentSystem,
    ComponentRegistryService,
    ContributorInfo,
    ComponentOwnership,
    TeamMapping,
    create_owner_assignment_api
)


@pytest.fixture
def temp_repo():
    """Create a temporary repository for testing"""
    with tempfile.TemporaryDirectory() as temp_dir:
        repo_path = Path(temp_dir)
        
        # Create test files
        test_files = {
            "test_component.py": '''
"""Test component"""
__phase__ = "A"
__code__ = "01A"  
__stage_order__ = 4

class TestProcessor:
    def process(self, data):
        return data
''',
            "canonical_flow/bridge_test.py": '''
"""Bridge test"""
__phase__ = "K"
__code__ = "02K"
__stage_order__ = 3

def process():
    pass
'''
        }
        
        for file_path, content in test_files.items():
            full_path = repo_path / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content)
            
        yield repo_path


@pytest.fixture  
def owner_system(temp_repo):
    """Create owner assignment system for testing"""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as temp_db:
        system = OwnerAssignmentSystem(
            repo_path=str(temp_repo),
            db_path=temp_db.name
        )
        yield system
        # Cleanup handled by temp file


class TestOwnerAssignmentSystem:
    
    def test_init_database(self, owner_system):
        """Test database initialization"""
        # Check tables exist
        with sqlite3.connect(owner_system.db_path) as conn:
            cursor = conn.execute("""
                SELECT name FROM sqlite_master WHERE type='table'
            """)
            tables = {row[0] for row in cursor.fetchall()}
            
        expected_tables = {
            'component_ownership',
            'ownership_history', 
            'team_mappings'
        }
        
        assert expected_tables.issubset(tables)
    
    def test_team_mappings_initialization(self, owner_system):
        """Test team mappings are properly initialized"""
        with sqlite3.connect(owner_system.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM team_mappings")
            count = cursor.fetchone()[0]
            
        assert count > 0  # Should have default team mappings
        
        # Check specific team exists
        with sqlite3.connect(owner_system.db_path) as conn:
            cursor = conn.execute("""
                SELECT team_name FROM team_mappings WHERE team_id = ?
            """, ("analysis_team",))
            result = cursor.fetchone()
            
        assert result is not None
        assert "Analysis" in result[0]
    
    @patch('subprocess.run')
    def test_analyze_git_blame(self, mock_run, owner_system, temp_repo):
        """Test git blame analysis"""
        # Mock git blame output
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = """
a1b2c3d4 1 1 1
author John Doe
author-mail <john@example.com>
author-time 1640995200
\tdef process():

e5f6g7h8 2 2 1  
author Jane Smith
author-mail <jane@example.com>
author-time 1641081600
\t    return data
"""
        
        test_file = temp_repo / "test_component.py"
        contributors = owner_system.analyze_git_blame(test_file)
        
        assert len(contributors) == 2
        assert contributors[0].email in ["john@example.com", "jane@example.com"]
        assert all(c.ownership_percentage > 0 for c in contributors)
        assert sum(c.ownership_percentage for c in contributors) == pytest.approx(1.0)
    
    def test_map_contributor_to_team(self, owner_system):
        """Test contributor to team mapping"""
        # Test contributor with analysis pattern
        contributor = ContributorInfo(
            email="ml.engineer@analysis.com",
            name="ML Engineer",
            commits_count=10,
            lines_contributed=100,
            first_contribution=datetime.now(),
            last_contribution=datetime.now(),
            ownership_percentage=0.5
        )
        
        team = owner_system.map_contributor_to_team(contributor)
        assert team == "analysis_team"
        
        # Test contributor with no matching pattern
        contributor.email = "random@external.com"
        team = owner_system.map_contributor_to_team(contributor)
        assert team is None
    
    @patch.object(OwnerAssignmentSystem, 'analyze_git_blame')
    def test_determine_component_ownership(self, mock_blame, owner_system, temp_repo):
        """Test component ownership determination"""
        # Mock contributors
        mock_blame.return_value = [
            ContributorInfo(
                email="primary@example.com",
                name="Primary Dev",
                commits_count=20,
                lines_contributed=80,
                first_contribution=datetime.now(),
                last_contribution=datetime.now(),
                ownership_percentage=0.8
            ),
            ContributorInfo(
                email="secondary@example.com", 
                name="Secondary Dev",
                commits_count=5,
                lines_contributed=20,
                first_contribution=datetime.now(),
                last_contribution=datetime.now(),
                ownership_percentage=0.2
            )
        ]
        
        test_file = temp_repo / "test_component.py"
        ownership = owner_system.determine_component_ownership(test_file)
        
        assert ownership is not None
        assert ownership.primary_owner == "primary@example.com"
        assert ownership.secondary_owners == ["secondary@example.com"]
        assert ownership.confidence_score > 0.5
    
    def test_save_and_get_ownership(self, owner_system):
        """Test saving and retrieving ownership"""
        ownership = ComponentOwnership(
            component_path="test/component.py",
            primary_owner="test@example.com",
            secondary_owners=["secondary@example.com"],
            team="test_team",
            confidence_score=0.85,
            last_updated=datetime.now(),
            manual_override=False
        )
        
        # Save ownership
        owner_system.save_ownership(ownership)
        
        # Retrieve ownership
        retrieved = owner_system.get_ownership("test/component.py")
        
        assert retrieved is not None
        assert retrieved.primary_owner == "test@example.com"
        assert retrieved.team == "test_team"
        assert retrieved.confidence_score == 0.85
    
    def test_manual_override(self, owner_system):
        """Test manual ownership override"""
        # Create initial ownership
        initial_ownership = ComponentOwnership(
            component_path="test/override.py",
            primary_owner="original@example.com",
            secondary_owners=[],
            team="original_team",
            confidence_score=0.6,
            last_updated=datetime.now(),
            manual_override=False
        )
        
        owner_system.save_ownership(initial_ownership)
        
        # Override ownership
        success = owner_system.override_ownership_manually(
            component_path="test/override.py",
            new_primary_owner="new@example.com",
            new_team="new_team",
            reason="Management decision",
            changed_by="admin@example.com"
        )
        
        assert success
        
        # Verify override
        updated = owner_system.get_ownership("test/override.py")
        assert updated.primary_owner == "new@example.com"
        assert updated.team == "new_team"
        assert updated.manual_override is True
        assert updated.confidence_score == 1.0
        
        # Check history
        with sqlite3.connect(owner_system.db_path) as conn:
            cursor = conn.execute("""
                SELECT old_owner, new_owner FROM ownership_history
                WHERE component_path = ?
            """, ("test/override.py",))
            history = cursor.fetchone()
            
        assert history[0] == "original@example.com"
        assert history[1] == "new@example.com"
    
    def test_team_ownership_summary(self, owner_system):
        """Test team ownership summary"""
        # Create test ownerships
        ownerships = [
            ComponentOwnership(
                component_path="team1/comp1.py",
                primary_owner="dev1@example.com",
                secondary_owners=[],
                team="team1",
                confidence_score=0.9,
                last_updated=datetime.now()
            ),
            ComponentOwnership(
                component_path="team1/comp2.py", 
                primary_owner="dev2@example.com",
                secondary_owners=[],
                team="team1",
                confidence_score=0.8,
                last_updated=datetime.now()
            ),
            ComponentOwnership(
                component_path="team2/comp1.py",
                primary_owner="dev3@example.com",
                secondary_owners=[],
                team="team2",
                confidence_score=0.7,
                last_updated=datetime.now()
            )
        ]
        
        for ownership in ownerships:
            owner_system.save_ownership(ownership)
            
        summary = owner_system.get_team_ownership_summary()
        
        assert "team1" in summary
        assert "team2" in summary
        assert summary["team1"]["component_count"] == 2
        assert summary["team2"]["component_count"] == 1
        assert summary["team1"]["avg_confidence"] == pytest.approx(0.85)


class TestComponentRegistryService:
    
    def test_discover_and_assign_components(self, owner_system, temp_repo):
        """Test component discovery and assignment"""
        registry_service = ComponentRegistryService(owner_system)
        
        with patch.object(owner_system, 'bulk_assign_ownership') as mock_assign:
            mock_assign.return_value = {"test_component.py": MagicMock()}
            
            result = registry_service.discover_and_assign_components([str(temp_repo)])
            
            assert result['discovered_components'] > 0
            assert result['assigned_ownership'] == 1
    
    def test_is_component_file(self, owner_system, temp_repo):
        """Test component file detection"""
        registry_service = ComponentRegistryService(owner_system)
        
        # Test positive case
        test_file = temp_repo / "test_component.py"
        assert registry_service._is_component_file(test_file)
        
        # Test negative case
        non_component = temp_repo / "regular.py"
        non_component.write_text("print('hello')")
        assert not registry_service._is_component_file(non_component)
    
    def test_reassign_component_owner(self, owner_system):
        """Test component owner reassignment API"""
        registry_service = ComponentRegistryService(owner_system)
        
        with patch.object(owner_system, 'override_ownership_manually') as mock_override:
            mock_override.return_value = True
            
            result = registry_service.reassign_component_owner(
                component_path="test/component.py",
                new_owner="new@example.com",
                new_team="new_team",
                reason="Testing",
                changed_by="admin"
            )
            
            assert result['success'] is True
            assert result['new_owner'] == "new@example.com"
            mock_override.assert_called_once()


class TestAPI:
    
    def test_create_owner_assignment_api(self):
        """Test API creation"""
        api = create_owner_assignment_api()
        
        expected_endpoints = [
            'assign_ownership',
            'reassign_ownership', 
            'get_ownership_info',
            'get_dashboard_data'
        ]
        
        for endpoint in expected_endpoints:
            assert endpoint in api
            assert callable(api[endpoint])


if __name__ == "__main__":
    pytest.main([__file__])
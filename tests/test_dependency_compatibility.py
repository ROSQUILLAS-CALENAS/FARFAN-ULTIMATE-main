#!/usr/bin/env python3
"""
Test suite for dependency compatibility validation

This test suite validates the dependency compatibility matrix functionality,
import safety, version conflict detection, and mock API consistency.
"""

import pytest
import sys
import tempfile
import json
# # # from pathlib import Path  # Module not found  # Module not found  # Module not found
# # # from unittest.mock import patch, MagicMock  # Module not found  # Module not found  # Module not found

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# # # from scripts.validate_dependency_compatibility import DependencyCompatibilityValidator  # Module not found  # Module not found  # Module not found

class TestDependencyCompatibilityValidator:
    """Test suite for dependency compatibility validator"""
    
    @pytest.fixture
    def validator(self):
        """Create validator instance for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield DependencyCompatibilityValidator(
                fail_fast=False,
                output_dir=Path(temp_dir),
                verbose=False
            )
    
    @pytest.fixture  
    def fail_fast_validator(self):
        """Create fail-fast validator for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield DependencyCompatibilityValidator(
                fail_fast=True,
                output_dir=Path(temp_dir),
                verbose=False
            )
    
    def test_validator_initialization(self, validator):
        """Test validator initializes correctly"""
        assert validator.fail_fast is False
        assert validator.output_dir.exists()
        assert len(validator.critical_dependencies) > 0
        assert len(validator.egw_modules) > 0
        assert validator.errors == []
        assert validator.warnings == []
    
    def test_python_version_compatibility(self, validator):
        """Test Python version compatibility check"""
        # This should pass for supported Python versions (3.8-3.12)
        result = validator.validate_python_version_compatibility()
        
        current_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        supported_versions = ['3.8', '3.9', '3.10', '3.11', '3.12']
        
        if current_version in supported_versions:
            assert result is True
        else:
            assert result is False
            assert len(validator.warnings) > 0
    
    def test_version_conflict_detection(self, validator):
        """Test version conflict detection"""
        # Mock importlib.metadata to simulate conflicts
        with patch('importlib.metadata.version') as mock_version:
            # Simulate FAISS conflict
            def side_effect(pkg):
                if pkg == 'faiss-cpu':
                    return '1.7.4'
                elif pkg == 'faiss-gpu':
                    return '1.7.4'  # Both present = conflict
                elif pkg == 'torch':
                    return '2.0.0'
                raise ImportError(f"Package {pkg} not found")
            
            mock_version.side_effect = side_effect
            
            result = validator.detect_version_conflicts()
            assert result is False  # Should detect conflict
            assert any('FAISS' in error for error in validator.errors)
    
    def test_mock_torch_api_consistency(self, validator):
        """Test PyTorch mock API consistency"""
        result = validator._test_torch_mock_api()
        assert result is True  # Mock should have consistent API
        
    def test_mock_faiss_api_consistency(self, validator):
        """Test FAISS mock API consistency"""
        result = validator._test_faiss_mock_api()
        assert result is True  # Mock should have consistent API
        
    def test_mock_transformers_api_consistency(self, validator):
        """Test Transformers mock API consistency"""  
        result = validator._test_transformers_mock_api()
        assert result is True  # Mock should have consistent API
    
    def test_egw_pipeline_import_validation(self, validator):
        """Test EGW pipeline module import validation"""
        # This will test actual imports - some may fail in test environment
        result = validator.validate_egw_pipeline_imports()
        
        # Check that results were recorded
        assert len(validator.import_results) > 0
        
        # At least the main package should be importable
        assert 'egw_query_expansion' in validator.import_results
        
    @patch('canonical_flow.mathematical_enhancers.mathematical_compatibility_matrix.MathematicalCompatibilityMatrix')
    def test_existing_compatibility_check_success(self, mock_matrix_class, validator):
        """Test successful execution of existing compatibility check"""
        # Mock the compatibility matrix and results
        mock_matrix = MagicMock()
        mock_matrix_class.return_value = mock_matrix
        
        # Mock successful results for critical dependencies
        mock_result = MagicMock()
        mock_result.is_compatible = True
        mock_result.installed_version = '2.0.0'
        mock_result.required_version = '>=1.0.0'
        mock_result.issues = None
        mock_result.warnings = None
        
        mock_matrix.check_library_compatibility.return_value = mock_result
        mock_matrix.check_all_compatibility.return_value = {
            'numpy': mock_result,
            'torch': mock_result
        }
        
        result = validator.execute_existing_compatibility_check()
        assert result is True
        assert len(validator.compatibility_results) > 0
        
    @patch('canonical_flow.mathematical_enhancers.mathematical_compatibility_matrix.MathematicalCompatibilityMatrix')
    def test_existing_compatibility_check_failure(self, mock_matrix_class, fail_fast_validator):
        """Test failure in existing compatibility check with fail-fast"""
        # Mock the compatibility matrix with failures
        mock_matrix = MagicMock()
        mock_matrix_class.return_value = mock_matrix
        
        # Mock failed result for critical dependency
        mock_result = MagicMock()
        mock_result.is_compatible = False
        mock_result.installed_version = '1.0.0'
        mock_result.required_version = '>=2.0.0'
        mock_result.issues = ['Version too old']
        mock_result.warnings = None
        
        mock_matrix.check_library_compatibility.return_value = mock_result
        
        # This should trigger fail-fast and exit
        with pytest.raises(SystemExit):
            fail_fast_validator.execute_existing_compatibility_check()
    
    def test_compatibility_report_generation(self, validator):
        """Test compatibility report generation"""
        # Add some mock data
        validator.compatibility_results = {
            'numpy': {
                'compatible': True,
                'installed_version': '1.21.0',
                'required_version': '>=1.19.0',
                'issues': None,
                'warnings': None
            }
        }
        
        validator.import_results = {
            'egw_query_expansion': {
                'status': 'success',
                'path': '/mock/path',
                'submodules': ['core']
            }
        }
        
        validator.upgrade_recommendations = [{
            'library': 'torch',
            'current_version': '1.10.0', 
            'required_version': '>=2.0.0',
            'priority': 'high',
            'issues': ['Version too old']
        }]
        
        report = validator.generate_compatibility_report()
        
        # Check report structure
        assert 'metadata' in report
        assert 'summary' in report
        assert 'compatibility_results' in report
        assert 'import_validation' in report
        assert 'upgrade_recommendations' in report
        
        # Check that file was saved
        report_file = validator.output_dir / 'dependency_compatibility_report.json'
        assert report_file.exists()
        
        # Verify file content
        with open(report_file) as f:
            saved_report = json.load(f)
        assert saved_report['summary']['total_libraries_tested'] == 1
        
    def test_fallback_patterns_generation(self, validator):
        """Test fallback usage patterns generation"""
        patterns = validator._generate_fallback_patterns()
        
        assert 'torch_fallbacks' in patterns
        assert 'faiss_fallbacks' in patterns  
        assert 'optional_dependencies' in patterns
        
        # Check structure
        assert 'cpu_only' in patterns['torch_fallbacks']
        assert 'mock_usage' in patterns['torch_fallbacks']
        assert 'cpu_variant' in patterns['faiss_fallbacks']
        
    def test_error_and_warning_logging(self, validator):
        """Test error and warning logging functionality"""
        # Test error logging
        validator.log_error("Test error message")
        assert len(validator.errors) == 1
        assert "Test error message" in validator.errors[0]
        
        # Test warning logging  
        validator.log_warning("Test warning message")
        assert len(validator.warnings) == 1
        assert "Test warning message" in validator.warnings[0]
        
        # Test fail-fast error logging
        fail_fast_validator = DependencyCompatibilityValidator(fail_fast=True)
        with pytest.raises(SystemExit):
            fail_fast_validator.log_error("Critical error", critical=True)

class TestIntegrationCompatibility:
    """Integration tests for library compatibility"""
    
    def test_numpy_scipy_compatibility(self):
        """Test NumPy/SciPy compatibility"""
        try:
            import numpy as np
            import scipy
            
            # Test basic operations work together
            a = np.random.rand(10, 10)
# # #             from scipy.linalg import svd  # Module not found  # Module not found  # Module not found
            u, s, vt = svd(a)
            
            # Test reconstruction
            reconstructed = u @ np.diag(s) @ vt
            error = np.max(np.abs(a - reconstructed))
            assert error < 1e-10
            
        except ImportError:
            pytest.skip("NumPy/SciPy not available for compatibility test")
    
    def test_torch_compatibility(self):
        """Test PyTorch compatibility and basic operations"""
        try:
            import torch
            
            # Test basic tensor operations
            a = torch.randn(10, 10)
            b = torch.randn(10, 10)
            c = torch.mm(a, b)
            
            assert c.shape == (10, 10)
            
            # Test CUDA availability check doesn't crash
            cuda_available = torch.cuda.is_available()
            assert isinstance(cuda_available, bool)
            
        except ImportError:
            pytest.skip("PyTorch not available for compatibility test")
    
    def test_faiss_compatibility(self):
        """Test FAISS compatibility and basic operations"""
        try:
            import faiss
            import numpy as np
            
            # Test basic index creation and operations
            dimension = 128
            index = faiss.IndexFlatL2(dimension)
            
            # Test adding vectors
            vectors = np.random.random((100, dimension)).astype('float32')
            index.add(vectors)
            
            assert index.ntotal == 100
            
            # Test search
            query = np.random.random((1, dimension)).astype('float32')
            distances, indices = index.search(query, 5)
            
            assert distances.shape == (1, 5)
            assert indices.shape == (1, 5)
            
        except ImportError:
            pytest.skip("FAISS not available for compatibility test")
    
    def test_pot_compatibility(self):
        """Test POT (Python Optimal Transport) compatibility"""
        try:
            import ot
            import numpy as np
            
            # Test basic optimal transport
            n = 100
            a = np.random.rand(n)
            a = a / a.sum()
            b = np.random.rand(n) 
            b = b / b.sum()
            
            M = np.random.rand(n, n)
            
            # Test EMD
            T = ot.emd(a, b, M)
            
            # Test marginal constraints
            assert np.allclose(T.sum(axis=1), a, atol=1e-6)
            assert np.allclose(T.sum(axis=0), b, atol=1e-6)
            
        except ImportError:
            pytest.skip("POT not available for compatibility test")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
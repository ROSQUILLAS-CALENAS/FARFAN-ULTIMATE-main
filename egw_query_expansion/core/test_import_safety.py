"""
Tests for Import Safety Utility
"""

import pytest
import sys
from unittest.mock import patch, MagicMock

from import_safety import ImportSafety, safe_import, ImportResult


class TestImportSafety:
    """Test ImportSafety functionality"""
    
    def setup_method(self):
        """Reset ImportSafety singleton for each test"""
        ImportSafety._instance = None
    
    def test_singleton_pattern(self):
        """Test that ImportSafety follows singleton pattern"""
        safety1 = ImportSafety()
        safety2 = ImportSafety()
        assert safety1 is safety2
    
    def test_successful_import(self):
        """Test successful module import"""
        safety = ImportSafety()
        result = safety.safe_import('os')
        
        assert result.success
        assert result.module is not None
        assert result.error is None
        assert result.fallback_used is None
    
    def test_failed_import_with_fallback(self):
        """Test failed import with fallback factory"""
        safety = ImportSafety()
        
        def mock_fallback():
            return MagicMock()
        
        result = safety.safe_import('nonexistent_module', fallback_factory=mock_fallback)
        
        assert result.success
        assert result.module is not None
        assert result.fallback_used == "factory"
    
    def test_failed_import_no_fallback(self):
        """Test failed import without fallback"""
        safety = ImportSafety()
        result = safety.safe_import('nonexistent_module', required=False)
        
        assert not result.success
        assert result.module is None
        assert result.error is not None
        assert result.fallback_used is None
    
    def test_version_checking(self):
        """Test minimum version checking"""
        safety = ImportSafety()
        
        # Test with a module that has version
        with patch('builtins.__import__') as mock_import:
            mock_module = MagicMock()
            mock_module.__version__ = '1.0.0'
            mock_import.return_value = mock_module
            
            result = safety.safe_import('test_module', min_version='0.5.0')
            assert result.success
            
            result = safety.safe_import('test_module', min_version='2.0.0')
            assert not result.success
    
    def test_attribute_validation(self):
        """Test required attribute validation"""
        safety = ImportSafety()
        
        with patch('builtins.__import__') as mock_import:
            mock_module = MagicMock()
            mock_module.required_attr = True
            mock_import.return_value = mock_module
            
            result = safety.safe_import('test_module', attributes=['required_attr'])
            assert result.success
            
            result = safety.safe_import('test_module', attributes=['missing_attr'])
            assert not result.success
    
    def test_alternative_names(self):
        """Test alternative module names"""
        safety = ImportSafety()
        
        with patch('builtins.__import__') as mock_import:
            def side_effect(name):
                if name == 'primary_name':
                    raise ImportError("Module not found")
                elif name == 'alternative_name':
                    return MagicMock()
                raise ImportError("Module not found")
            
            mock_import.side_effect = side_effect
            
            result = safety.safe_import(
                'primary_name', 
                alternative_names=['alternative_name']
            )
            assert result.success
            assert result.metadata['actual_name'] == 'alternative_name'
    
    def test_register_fallback(self):
        """Test fallback registration"""
        safety = ImportSafety()
        
        def fallback_factory():
            return MagicMock()
        
        safety.register_fallback('test_module', fallback_factory)
        result = safety.safe_import('nonexistent_test_module')
        
        # Should not use registered fallback for different module name
        assert not result.success
        
        # Register with exact name match
        safety.register_fallback('nonexistent_test_module', fallback_factory)
        result = safety.safe_import('nonexistent_test_module')
        assert result.success
        assert result.fallback_used == "registry"
    
    def test_register_mock(self):
        """Test mock registration"""
        safety = ImportSafety()
        mock_module = MagicMock()
        
        safety.register_mock('test_module', mock_module)
        result = safety.safe_import('nonexistent_test_module')
        
        # Should not use registered mock for different module name
        assert not result.success
        
        # Register with exact name match
        safety.register_mock('nonexistent_test_module', mock_module)
        result = safety.safe_import('nonexistent_test_module')
        assert result.success
        assert result.fallback_used == "mock_registry"
        assert result.module is mock_module
    
    def test_import_report(self):
        """Test import status reporting"""
        safety = ImportSafety()
        
        # Perform some imports
        safety.safe_import('os')  # Should succeed
        safety.safe_import('nonexistent_module', required=False)  # Should fail
        
        report = safety.get_import_report()
        
        assert 'summary' in report
        assert report['summary']['total_attempts'] >= 2
        assert report['summary']['successful_imports'] >= 1
        assert report['summary']['failed_imports'] >= 1
    
    def test_convenience_functions(self):
        """Test convenience wrapper functions"""
        result = safe_import('os')
        assert result.success
        assert result.module is not None
    
    def test_specialized_import_methods(self):
        """Test specialized import methods for common libraries"""
        safety = ImportSafety()
        
        # Test safe_import_numpy
        result = safety.safe_import_numpy()
        # Should either succeed or fail gracefully
        assert isinstance(result, ImportResult)
        
        # Test safe_import_torch
        result = safety.safe_import_torch()
        assert isinstance(result, ImportResult)
        
        # Test safe_import_sklearn
        result = safety.safe_import_sklearn()
        assert isinstance(result, ImportResult)
        
        # Test safe_import_faiss (with fallback)
        result = safety.safe_import_faiss()
        assert isinstance(result, ImportResult)
        # Should either succeed or provide fallback
        if not result.success:
            assert result.fallback_used is not None
    
    def test_caching(self):
        """Test import result caching"""
        safety = ImportSafety()
        
        result1 = safety.safe_import('os')
        result2 = safety.safe_import('os')
        
        # Should return same result from cache
        assert result1.module is result2.module
    
    def test_clear_cache(self):
        """Test cache clearing"""
        safety = ImportSafety()
        
        # Import and cache result
        safety.safe_import('os')
        assert len(safety.successful_imports) > 0
        
        # Clear cache
        safety.clear_cache()
        assert len(safety.successful_imports) == 0
        assert len(safety.failed_imports) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
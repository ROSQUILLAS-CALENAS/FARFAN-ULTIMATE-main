"""
Tests for PDF Error Handling System

This module contains comprehensive tests for the PDF processing error handling,
checkpointing, validation, and resource monitoring capabilities.
"""

import json
import tempfile
import time
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from pdf_processing_error_handler import (
    PDFErrorHandler,
    PDFValidator,
    ResourceMonitor,
    CheckpointManager,
    ExponentialBackoffRetry,
    ProcessingState,
    ResourceMetrics,
    PDFValidationError,
    ResourceExhaustionError,
    CheckpointError,
    process_pdf_batch_with_error_handling
)


class TestPDFValidator(unittest.TestCase):
    """Test PDF validation functionality"""
    
    def setUp(self):
        self.validator = PDFValidator(max_file_size_mb=10)
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def tearDown(self):
        # Clean up temp files
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def create_temp_file(self, name: str, content: bytes) -> Path:
        """Helper to create temporary files"""
        file_path = self.temp_dir / name
        with open(file_path, 'wb') as f:
            f.write(content)
        return file_path
    
    def test_valid_pdf_header(self):
        """Test validation of valid PDF header"""
        valid_pdf_content = b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n>>\nendobj"
        pdf_path = self.create_temp_file("valid.pdf", valid_pdf_content)
        
        is_valid, message = self.validator._validate_pdf_header(pdf_path)
        self.assertTrue(is_valid)
        self.assertEqual(message, "Valid PDF header")
    
    def test_invalid_pdf_header(self):
        """Test validation of invalid PDF header"""
        invalid_content = b"NOT A PDF FILE"
        pdf_path = self.create_temp_file("invalid.pdf", invalid_content)
        
        is_valid, message = self.validator._validate_pdf_header(pdf_path)
        self.assertFalse(is_valid)
        self.assertIn("Invalid PDF header", message)
    
    def test_file_size_validation(self):
        """Test file size limits"""
        # Test file too small
        small_file = self.create_temp_file("small.pdf", b"")
        is_valid, message = self.validator.validate_pdf(small_file)
        self.assertFalse(is_valid)
        self.assertIn("too small", message)
        
        # Test file too large (mock large file)
        large_content = b"x" * (15 * 1024 * 1024)  # 15MB
        large_file = self.create_temp_file("large.pdf", large_content)
        is_valid, message = self.validator.validate_pdf(large_file)
        self.assertFalse(is_valid)
        self.assertIn("too large", message)
    
    def test_nonexistent_file(self):
        """Test validation of non-existent file"""
        nonexistent = Path("/nonexistent/file.pdf")
        is_valid, message = self.validator.validate_pdf(nonexistent)
        self.assertFalse(is_valid)
        self.assertIn("does not exist", message)
    
    def test_pdf_structure_validation(self):
        """Test basic PDF structure validation"""
        # Valid structure
        valid_content = b"%PDF-1.4\nsome content\nxref\ntrailer\n%%EOF"
        valid_file = self.create_temp_file("valid_structure.pdf", valid_content)
        is_valid, message = self.validator._validate_pdf_structure(valid_file)
        self.assertTrue(is_valid)
        
        # Invalid structure (missing EOF)
        invalid_content = b"%PDF-1.4\nsome content\nxref\ntrailer"
        invalid_file = self.create_temp_file("invalid_structure.pdf", invalid_content)
        is_valid, message = self.validator._validate_pdf_structure(invalid_file)
        self.assertFalse(is_valid)
        self.assertIn("end-of-file marker", message)


class TestResourceMonitor(unittest.TestCase):
    """Test resource monitoring functionality"""
    
    def setUp(self):
        self.monitor = ResourceMonitor(
            memory_threshold_mb=1024,
            memory_percent_threshold=80.0
        )
    
    def tearDown(self):
        self.monitor.stop_monitoring()
    
    def test_get_current_metrics(self):
        """Test getting current resource metrics"""
        metrics = self.monitor.get_current_metrics()
        
        self.assertIsInstance(metrics, ResourceMetrics)
        self.assertGreater(metrics.memory_usage_mb, 0)
        self.assertGreater(metrics.memory_percent, 0)
        self.assertGreaterEqual(metrics.cpu_percent, 0)
        self.assertGreater(metrics.disk_usage_gb, 0)
        self.assertIsInstance(metrics.timestamp, datetime)
    
    @patch('psutil.virtual_memory')
    @patch('psutil.Process')
    def test_memory_threshold_exceeded(self, mock_process, mock_memory):
        """Test memory threshold detection"""
        # Mock high memory usage
        mock_memory.return_value.percent = 90.0  # Above 80% threshold
        mock_process.return_value.memory_info.return_value.rss = 2048 * 1024 * 1024  # Above 1024MB
        
        within_limits, message = self.monitor.check_resource_thresholds()
        self.assertFalse(within_limits)
        self.assertIn("Memory usage", message)
    
    @patch('psutil.cpu_percent')
    def test_cpu_threshold_exceeded(self, mock_cpu):
        """Test CPU threshold detection"""
        mock_cpu.return_value = 98.0  # Above 95% threshold
        
        within_limits, message = self.monitor.check_resource_thresholds()
        self.assertFalse(within_limits)
        self.assertIn("CPU usage", message)
    
    def test_monitoring_start_stop(self):
        """Test starting and stopping monitoring"""
        self.assertFalse(self.monitor._monitoring)
        
        self.monitor.start_monitoring(interval_seconds=1)
        self.assertTrue(self.monitor._monitoring)
        time.sleep(0.1)  # Let it start
        
        self.monitor.stop_monitoring()
        self.assertFalse(self.monitor._monitoring)
    
    def test_metrics_history(self):
        """Test metrics history collection"""
        self.monitor.start_monitoring(interval_seconds=0.1)
        time.sleep(0.3)  # Collect a few samples
        self.monitor.stop_monitoring()
        
        history = self.monitor.get_metrics_history()
        self.assertGreater(len(history), 0)
        self.assertIsInstance(history[0], ResourceMetrics)


class TestExponentialBackoffRetry(unittest.TestCase):
    """Test exponential backoff retry mechanism"""
    
    def setUp(self):
        self.retry = ExponentialBackoffRetry(
            max_attempts=3,
            base_delay=0.01,  # Fast for testing
            max_delay=0.1
        )
    
    def test_successful_retry(self):
        """Test successful function after retries"""
        attempt_count = 0
        
        @self.retry
        def failing_function():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise ValueError("Temporary failure")
            return "Success"
        
        result = failing_function()
        self.assertEqual(result, "Success")
        self.assertEqual(attempt_count, 3)
    
    def test_max_attempts_exceeded(self):
        """Test failure after max attempts"""
        @self.retry
        def always_failing():
            raise ValueError("Always fails")
        
        with self.assertRaises(ValueError):
            always_failing()
    
    def test_immediate_success(self):
        """Test function that succeeds immediately"""
        @self.retry
        def immediate_success():
            return "Immediate success"
        
        result = immediate_success()
        self.assertEqual(result, "Immediate success")


class TestProcessingState(unittest.TestCase):
    """Test processing state serialization"""
    
    def test_state_serialization(self):
        """Test ProcessingState to/from dict conversion"""
        state = ProcessingState(
            batch_id="test_batch",
            total_documents=10,
            processed_documents=["doc1.pdf", "doc2.pdf"],
            failed_documents=[("bad.pdf", "Error message")],
            current_index=2,
            checkpoint_frequency=5,
            start_time=datetime(2024, 1, 1, 12, 0, 0),
            last_checkpoint_time=datetime(2024, 1, 1, 12, 5, 0),
            metadata={"test": True}
        )
        
        # Convert to dict
        state_dict = state.to_dict()
        self.assertIsInstance(state_dict['start_time'], str)
        self.assertIsInstance(state_dict['last_checkpoint_time'], str)
        
        # Convert back from dict
        restored_state = ProcessingState.from_dict(state_dict)
        self.assertEqual(restored_state.batch_id, state.batch_id)
        self.assertEqual(restored_state.total_documents, state.total_documents)
        self.assertEqual(restored_state.processed_documents, state.processed_documents)
        self.assertEqual(restored_state.start_time, state.start_time)
        self.assertEqual(restored_state.last_checkpoint_time, state.last_checkpoint_time)


class TestCheckpointManager(unittest.TestCase):
    """Test checkpoint management functionality"""
    
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.manager = CheckpointManager(self.temp_dir)
        self.test_state = ProcessingState(
            batch_id="test_checkpoint",
            total_documents=5,
            processed_documents=["doc1.pdf"],
            failed_documents=[],
            current_index=1,
            checkpoint_frequency=2,
            start_time=datetime.now()
        )
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_save_and_load_checkpoint(self):
        """Test saving and loading checkpoints"""
        # Save checkpoint
        checkpoint_path = self.manager.save_checkpoint(self.test_state)
        self.assertTrue(Path(checkpoint_path).exists())
        
        # Load checkpoint
        loaded_state = self.manager.load_checkpoint(checkpoint_path)
        self.assertEqual(loaded_state.batch_id, self.test_state.batch_id)
        self.assertEqual(loaded_state.total_documents, self.test_state.total_documents)
        self.assertEqual(loaded_state.current_index, self.test_state.current_index)
    
    def test_find_latest_checkpoint(self):
        """Test finding latest checkpoint"""
        # No checkpoints initially
        latest = self.manager.find_latest_checkpoint("test_checkpoint")
        self.assertIsNone(latest)
        
        # Create checkpoints
        checkpoint1 = self.manager.save_checkpoint(self.test_state)
        time.sleep(0.01)  # Ensure different timestamps
        
        self.test_state.current_index = 2
        checkpoint2 = self.manager.save_checkpoint(self.test_state)
        
        # Find latest
        latest = self.manager.find_latest_checkpoint("test_checkpoint")
        self.assertEqual(str(latest), checkpoint2)
    
    def test_cleanup_old_checkpoints(self):
        """Test cleaning up old checkpoints"""
        # Create multiple checkpoints
        for i in range(5):
            self.test_state.current_index = i
            self.manager.save_checkpoint(self.test_state)
            time.sleep(0.01)
        
        # Clean up, keeping only 2
        self.manager.cleanup_old_checkpoints("test_checkpoint", keep_count=2)
        
        # Count remaining checkpoints
        checkpoints = list(self.temp_dir.glob("checkpoint_test_checkpoint_*"))
        self.assertEqual(len(checkpoints), 4)  # 2 pkl + 2 json files
    
    def test_invalid_checkpoint_path(self):
        """Test loading from invalid checkpoint path"""
        with self.assertRaises(CheckpointError):
            self.manager.load_checkpoint("/nonexistent/checkpoint.pkl")


class TestPDFErrorHandler(unittest.TestCase):
    """Test main PDF error handler"""
    
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.handler = PDFErrorHandler(
            checkpoint_frequency=2,
            checkpoint_dir=self.temp_dir / "checkpoints",
            max_file_size_mb=10,
            memory_threshold_mb=2048,
            max_retry_attempts=2,
            enable_resource_monitoring=False  # Disable for testing
        )
        
        # Create test PDF files
        self.test_files = []
        for i in range(3):
            pdf_content = b"%PDF-1.4\ntest content\nxref\ntrailer\n%%EOF"
            pdf_path = self.temp_dir / f"test_{i}.pdf"
            with open(pdf_path, 'wb') as f:
                f.write(pdf_content)
            self.test_files.append(pdf_path)
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def mock_processing_function(self, file_path: Path) -> dict:
        """Mock processing function for tests"""
        if "fail" in file_path.name:
            raise ValueError(f"Mock failure for {file_path.name}")
        return {"file": str(file_path), "status": "processed"}
    
    def test_validate_pdf_batch(self):
        """Test batch PDF validation"""
        # Add an invalid file
        invalid_file = self.temp_dir / "invalid.pdf"
        with open(invalid_file, 'w') as f:
            f.write("not a pdf")
        
        all_files = self.test_files + [invalid_file]
        valid_files, invalid_files = self.handler.validate_pdf_batch(all_files)
        
        self.assertEqual(len(valid_files), 3)
        self.assertEqual(len(invalid_files), 1)
        self.assertEqual(invalid_files[0][0], str(invalid_file))
    
    def test_batch_processing_success(self):
        """Test successful batch processing"""
        results = self.handler.process_pdf_batch(
            file_paths=self.test_files,
            processing_function=self.mock_processing_function
        )
        
        self.assertEqual(results["successful_files"], 3)
        self.assertEqual(results["failed_files"], 0)
        self.assertEqual(len(results["results"]), 3)
        
        for result in results["results"]:
            self.assertEqual(result["status"], "success")
    
    def test_batch_processing_with_failures(self):
        """Test batch processing with some failures"""
        # Create a file that will trigger failure
        fail_file = self.temp_dir / "fail_test.pdf"
        with open(fail_file, 'wb') as f:
            f.write(b"%PDF-1.4\ntest\nxref\ntrailer\n%%EOF")
        
        files_with_failure = self.test_files + [fail_file]
        
        results = self.handler.process_pdf_batch(
            file_paths=files_with_failure,
            processing_function=self.mock_processing_function
        )
        
        self.assertEqual(results["successful_files"], 3)
        self.assertEqual(results["failed_files"], 1)
        
        # Check that failure was recorded
        failed_results = [r for r in results["results"] if r["status"] == "failed"]
        self.assertEqual(len(failed_results), 1)
        self.assertIn("Mock failure", failed_results[0]["error"])
    
    @patch('pdf_processing_error_handler.ResourceMonitor.check_resource_thresholds')
    def test_resource_exhaustion(self, mock_resource_check):
        """Test handling of resource exhaustion"""
        mock_resource_check.return_value = (False, "Memory limit exceeded")
        
        with self.assertRaises(Exception):  # Should raise resource exhaustion
            self.handler.process_pdf_batch(
                file_paths=self.test_files[:1],
                processing_function=self.mock_processing_function
            )
    
    def test_checkpointing_during_processing(self):
        """Test that checkpoints are created during processing"""
        results = self.handler.process_pdf_batch(
            file_paths=self.test_files,
            processing_function=self.mock_processing_function,
            batch_id="checkpoint_test"
        )
        
        # Check that checkpoint was created
        checkpoint_dir = self.temp_dir / "checkpoints"
        checkpoints = list(checkpoint_dir.glob("checkpoint_checkpoint_test_*.pkl"))
        self.assertGreater(len(checkpoints), 0)


class TestBatchProcessingFunction(unittest.TestCase):
    """Test the convenience batch processing function"""
    
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Create test files
        self.test_files = []
        for i in range(2):
            pdf_content = b"%PDF-1.4\ntest\nxref\ntrailer\n%%EOF"
            pdf_path = self.temp_dir / f"batch_test_{i}.pdf"
            with open(pdf_path, 'wb') as f:
                f.write(pdf_content)
            self.test_files.append(str(pdf_path))
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def simple_processor(self, file_path: Path) -> dict:
        """Simple processing function"""
        return {"processed": True, "file": str(file_path)}
    
    def test_convenience_function(self):
        """Test the convenience batch processing function"""
        results = process_pdf_batch_with_error_handling(
            file_paths=self.test_files,
            processing_function=self.simple_processor,
            checkpoint_frequency=5,
            enable_resource_monitoring=False
        )
        
        self.assertEqual(results["successful_files"], 2)
        self.assertEqual(results["failed_files"], 0)
        self.assertEqual(len(results["results"]), 2)


class TestErrorTypes(unittest.TestCase):
    """Test custom error types"""
    
    def test_pdf_validation_error(self):
        """Test PDFValidationError"""
        with self.assertRaises(PDFValidationError):
            raise PDFValidationError("Invalid PDF format")
    
    def test_resource_exhaustion_error(self):
        """Test ResourceExhaustionError"""
        with self.assertRaises(ResourceExhaustionError):
            raise ResourceExhaustionError("Out of memory")
    
    def test_checkpoint_error(self):
        """Test CheckpointError"""
        with self.assertRaises(CheckpointError):
            raise CheckpointError("Checkpoint corruption")


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
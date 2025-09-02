# PDF Processing Error Handling System

A comprehensive error handling, checkpointing, and resource monitoring system for PDF processing pipelines with retry mechanisms and validation.

## Features

### 🛡️ Comprehensive Error Handling
- **PDFErrorHandler**: Main class that wraps PDF processing operations in try-catch blocks
- **Isolation of failures**: Prevents cascade effects across batch processing
- **Graceful degradation**: System continues processing even when individual documents fail

### 💾 Checkpoint System
- **Automatic checkpointing**: Saves processing state after every N documents (configurable, default 10)
- **Resume functionality**: `resume_from_checkpoint()` to restart batch jobs from last successful state
- **Persistent state**: Checkpoints saved in both binary (pickle) and JSON formats
- **Automatic cleanup**: Old checkpoints are cleaned up automatically

### ✅ File Validation
- **PDF header verification**: Validates PDF signature and version
- **File size limits**: Configurable minimum and maximum file sizes
- **Corruption detection**: Basic PDF structure parsing to detect corrupted files
- **Batch validation**: Validates entire batches before processing

### 📊 Resource Monitoring
- **Memory usage tracking**: Monitors process and system memory usage
- **CPU monitoring**: Tracks CPU utilization
- **Disk space monitoring**: Ensures adequate disk space available
- **Threshold alerts**: Raises `ResourceExhaustionError` when limits are exceeded
- **Background monitoring**: Optional continuous monitoring with configurable intervals

### 🔄 Retry Mechanisms
- **Exponential backoff**: Intelligent retry with increasing delays
- **Configurable attempts**: Default 3 attempts, customizable
- **Base delay**: Default 2 seconds, adjustable
- **Max delay cap**: Prevents excessively long waits
- **Transient failure handling**: Automatically retries operations that may succeed on subsequent attempts

## Installation

The system is designed to work with minimal dependencies. Core functionality works without external dependencies, while advanced features require:

```bash
# Optional: For full resource monitoring
pip install psutil

# For PDF processing (if not already installed)
pip install PyMuPDF pdfplumber
```

## Quick Start

### Basic Batch Processing

```python
from pdf_processing_error_handler import process_pdf_batch_with_error_handling
from pathlib import Path

def my_pdf_processor(file_path: Path) -> dict:
    """Your PDF processing function"""
    # Process the PDF file
    return {"pages": 10, "text_length": 5000}

# Process a batch of PDFs with error handling
results = process_pdf_batch_with_error_handling(
    file_paths=["doc1.pdf", "doc2.pdf", "doc3.pdf"],
    processing_function=my_pdf_processor,
    checkpoint_frequency=5,  # Save checkpoint every 5 documents
    max_retry_attempts=3,
    enable_resource_monitoring=True
)

print(f"Processed: {results['successful_files']} successful, {results['failed_files']} failed")
```

### Advanced Usage with Full Control

```python
from pdf_processing_error_handler import PDFErrorHandler

# Initialize with custom settings
handler = PDFErrorHandler(
    checkpoint_frequency=10,
    max_file_size_mb=100,
    memory_threshold_mb=2048,
    max_retry_attempts=3,
    base_retry_delay=2.0,
    enable_resource_monitoring=True,
    monitoring_interval=30
)

# Process batch with resumption capability
results = handler.process_pdf_batch(
    file_paths=["doc1.pdf", "doc2.pdf", "doc3.pdf"],
    processing_function=my_pdf_processor,
    batch_id="my_batch_001",
    resume_from_checkpoint=True
)

# Resume from specific checkpoint if needed
if checkpoint_path:
    results = handler.resume_from_checkpoint(
        checkpoint_path="checkpoints/checkpoint_my_batch_001_12345.pkl",
        processing_function=my_pdf_processor
    )
```

### Individual Component Usage

#### PDF Validation

```python
from pdf_processing_error_handler import PDFValidator

validator = PDFValidator(max_file_size_mb=50)
is_valid, message = validator.validate_pdf("document.pdf")

if is_valid:
    print("PDF is valid for processing")
else:
    print(f"PDF validation failed: {message}")
```

#### Resource Monitoring

```python
from pdf_processing_error_handler import ResourceMonitor

monitor = ResourceMonitor(
    memory_threshold_mb=1024,
    memory_percent_threshold=85.0
)

# Check current resource usage
within_limits, message = monitor.check_resource_thresholds()
if not within_limits:
    print(f"Resource warning: {message}")

# Start background monitoring
monitor.start_monitoring(interval_seconds=30)
```

#### Retry Mechanism

```python
from pdf_processing_error_handler import ExponentialBackoffRetry

retry = ExponentialBackoffRetry(max_attempts=3, base_delay=1.0)

@retry
def unreliable_operation():
    # Your operation that might fail
    pass

result = unreliable_operation()  # Will retry up to 3 times
```

#### Checkpoint Management

```python
from pdf_processing_error_handler import CheckpointManager, ProcessingState
from datetime import datetime

manager = CheckpointManager("./my_checkpoints")

# Create processing state
state = ProcessingState(
    batch_id="my_batch",
    total_documents=100,
    processed_documents=["doc1.pdf", "doc2.pdf"],
    failed_documents=[],
    current_index=2,
    checkpoint_frequency=10,
    start_time=datetime.now()
)

# Save checkpoint
checkpoint_path = manager.save_checkpoint(state)
print(f"Saved checkpoint: {checkpoint_path}")

# Load checkpoint
loaded_state = manager.load_checkpoint(checkpoint_path)
print(f"Loaded state with {len(loaded_state.processed_documents)} processed documents")
```

## Integration with Existing PDF Pipeline

The system is designed to integrate seamlessly with existing PDF processing code:

### Before (without error handling):
```python
def process_pdfs(file_paths):
    results = []
    for pdf_path in file_paths:
        try:
            result = process_single_pdf(pdf_path)
            results.append(result)
        except Exception as e:
            print(f"Failed to process {pdf_path}: {e}")
    return results
```

### After (with comprehensive error handling):
```python
from pdf_processing_error_handler import PDFErrorHandler

def process_pdfs(file_paths):
    handler = PDFErrorHandler(checkpoint_frequency=10)
    return handler.process_pdf_batch(
        file_paths=file_paths,
        processing_function=process_single_pdf
    )
```

## Configuration Options

### PDFErrorHandler Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `checkpoint_frequency` | 10 | Save checkpoint every N documents |
| `max_file_size_mb` | 100 | Maximum PDF file size in MB |
| `memory_threshold_mb` | 2048 | Memory usage threshold in MB |
| `max_retry_attempts` | 3 | Maximum retry attempts for failures |
| `base_retry_delay` | 2.0 | Base delay for exponential backoff (seconds) |
| `enable_resource_monitoring` | True | Enable background resource monitoring |
| `monitoring_interval` | 30 | Resource monitoring interval (seconds) |

### PDFValidator Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_file_size_mb` | 100 | Maximum file size in MB |
| `min_file_size_bytes` | 1024 | Minimum file size in bytes |

### ResourceMonitor Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `memory_threshold_mb` | 2048 | Process memory threshold |
| `memory_percent_threshold` | 85.0 | System memory percentage threshold |
| `cpu_threshold_percent` | 95.0 | CPU usage percentage threshold |
| `disk_threshold_gb` | 1.0 | Minimum free disk space in GB |

## Error Types

The system defines several custom exception types:

- **`PDFProcessingError`**: Base exception for all PDF processing errors
- **`PDFValidationError`**: Raised when PDF validation fails
- **`ResourceExhaustionError`**: Raised when resource thresholds are exceeded
- **`CheckpointError`**: Raised during checkpoint save/load operations

## Checkpoint Format

Checkpoints are saved in both formats for flexibility:

### Pickle Format (`.pkl`)
Binary format for complete object serialization, including complex data types.

### JSON Format (`.json`)
Human-readable format for inspection and debugging:

```json
{
  "batch_id": "my_batch_001",
  "total_documents": 100,
  "processed_documents": ["doc1.pdf", "doc2.pdf"],
  "failed_documents": [["bad_doc.pdf", "Validation failed"]],
  "current_index": 2,
  "checkpoint_frequency": 10,
  "start_time": "2024-01-15T10:30:00",
  "last_checkpoint_time": "2024-01-15T10:35:00",
  "metadata": {"additional_info": "value"}
}
```

## Integration with Main Pipeline

The system has been integrated into the main PDF processing pipeline in `main.py`:

```python
# Enhanced batch processing with error handling
from pdf_reader import process_pdf_files_with_error_handling

batch_results = process_pdf_files_with_error_handling(
    file_paths=[str(p) for p in pdf_files],
    checkpoint_frequency=5,
    max_retry_attempts=3,
    enable_intelligent_ocr=True
)
```

## Demos and Examples

### Quick Demo
```bash
python3 run_demo_quick.py
```

### Full Demo (interactive)
```bash
python3 demo_pdf_error_handling.py
```

### Basic Functionality Test
```bash
python3 test_basic_pdf_error_handler.py
```

## Best Practices

1. **Choose appropriate checkpoint frequency**: Balance between recovery granularity and I/O overhead
2. **Set realistic resource thresholds**: Based on your system's capacity and other running processes
3. **Monitor logs**: Error handling produces detailed logs for troubleshooting
4. **Handle validation results**: Check the validation results before processing
5. **Clean up checkpoints**: Old checkpoints are automatically cleaned, but consider manual cleanup for long-running processes

## Monitoring and Logging

The system provides comprehensive logging:

```python
import logging
logging.basicConfig(level=logging.INFO)

# Will log:
# - Validation results for each file
# - Processing progress and timing
# - Retry attempts and failures
# - Checkpoint save/load operations
# - Resource usage warnings
```

## Performance Considerations

- **Checkpointing overhead**: Minimal impact with appropriate frequency settings
- **Resource monitoring**: Optional background monitoring with configurable intervals
- **Memory usage**: Efficient batch processing with resource-aware processing
- **Retry delays**: Exponential backoff prevents system overload during failures

## Contributing

When extending the system:

1. Maintain backward compatibility
2. Add comprehensive tests
3. Follow the established error handling patterns
4. Update documentation

## License

This system is part of the larger EGW Query Expansion project. See the main project LICENSE for details.
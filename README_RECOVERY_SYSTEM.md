# Document Recovery System

## Overview

The Document Recovery System provides a comprehensive recovery mechanism for the distributed processing system that tracks previously failed document processing attempts and automatically reprocesses them after import issues are resolved. The system is designed to handle various failure scenarios including import errors, memory issues, timeouts, and other processing failures.

## Architecture

### Components

1. **FailedDocumentRecord**: Data class that represents a failed document processing attempt with metadata including:
   - Document path and task ID
   - Error message and failure timestamp
   - Retry count and recovery attempts
   - Query, worker ID, and metadata

2. **FailedDocumentsTracker**: Redis-backed storage system that:
   - Tracks failed document processing attempts
   - Manages retry logic and failure metadata
   - Provides cleanup and retention management
   - Collects recovery metrics

3. **DocumentRecoveryManager**: Main recovery orchestrator that:
   - Manages recovery operations and scheduling
   - Integrates with the distributed processor
   - Provides periodic recovery capabilities
   - Analyzes failure patterns and recovery success rates

## Features

### Automatic Recovery Integration

The recovery system is automatically integrated into the distributed processing pipeline:

- **Initialization**: Recovery system starts when the distributed processor initializes
- **Failure Tracking**: Failed documents are automatically tracked with metadata
- **Success Cleanup**: Successfully processed documents are removed from failed tracking
- **Periodic Recovery**: Automatic retry of failed documents at configurable intervals

### Intelligent Retry Logic

- **Exponential Backoff**: Configurable retry intervals to avoid overwhelming the system
- **Max Retry Limits**: Prevents infinite retry loops
- **Failure Age Tracking**: Allows filtering by failure age
- **Error Categorization**: Groups similar errors for analysis

### Redis-Based Persistence

- **Persistent Storage**: Failed documents survive system restarts
- **TTL Management**: Automatic cleanup of old records
- **Atomic Operations**: Thread-safe operations for concurrent access
- **Metrics Storage**: Recovery statistics and performance tracking

### Recovery Metrics and Analytics

- **Success Rates**: Track recovery success/failure ratios
- **Processing Times**: Monitor average recovery processing times
- **Error Analysis**: Categorize and analyze common failure types
- **Age Distribution**: Analyze failed documents by failure age

## Usage

### Integration with Distributed Processor

The recovery system is automatically integrated when initializing the distributed processor:

```python
from distributed_processor import DistributedProcessor

# Recovery system is automatically initialized
processor = DistributedProcessor(
    worker_id="worker-1",
    redis_url="redis://localhost:6379"
)

# Start worker (includes automatic recovery attempt)
await processor.start_worker()
```

### Manual Recovery

You can trigger recovery manually using the command line:

```bash
# Run recovery for all eligible failed documents
python main.py --recover

# Run recovery with custom Redis URL
python main.py --recover --redis-url redis://custom-host:6379

# Run with verbose output
python main.py --recover --verbose
```

### Standalone Recovery Function

```python
import asyncio
from recovery_system import run_document_recovery

async def main():
    config = {
        'max_retry_count': 3,
        'min_retry_interval_hours': 1.0,
        'recovery_batch_size': 10,
        'enable_periodic_recovery': False
    }
    
    result = await run_document_recovery(
        redis_url="redis://localhost:6379",
        config=config
    )
    
    print(f"Recovery completed: {result}")

asyncio.run(main())
```

## Configuration

### Recovery Configuration Parameters

```python
config = {
    # Maximum retry attempts per document
    'max_retry_count': 3,
    
    # Minimum interval between retry attempts (hours)
    'min_retry_interval_hours': 1.0,
    
    # Number of documents to process in each recovery batch
    'recovery_batch_size': 10,
    
    # Enable automatic periodic recovery
    'enable_periodic_recovery': True,
    
    # Interval between periodic recovery runs (minutes)
    'recovery_interval_minutes': 30,
    
    # Timeout for individual recovery attempts (seconds)
    'recovery_timeout': 300,
    
    # Retention period for failed document records (days)
    'failed_docs_retention_days': 7
}
```

### Distributed Processor Integration

```python
from distributed_processor import DistributedProcessor

processor = DistributedProcessor(
    worker_id="worker-1",
    redis_url="redis://localhost:6379"
)

# Recovery system configuration is passed in the processor config
processor.config.update({
    'max_retry_count': 3,
    'min_retry_interval_hours': 0.5,
    'recovery_batch_size': 5,
    'enable_periodic_recovery': True,
    'recovery_interval_minutes': 15
})
```

## Recovery Process Flow

1. **Failure Detection**: When a document processing task fails, it's automatically tracked
2. **Metadata Storage**: Failure details, error messages, and context are stored in Redis
3. **Eligibility Check**: System determines if a document is eligible for retry based on:
   - Retry count vs. maximum allowed
   - Time since last attempt vs. minimum interval
   - Document age vs. retention period
4. **Recovery Attempt**: Eligible documents are resubmitted for processing
5. **Result Tracking**: Recovery attempts are monitored and metrics updated
6. **Cleanup**: Successfully recovered documents are removed from failed tracking

## Monitoring and Analytics

### Recovery Status

```python
# Get current recovery status
status = await recovery_manager.get_recovery_status()

print(f"Total failed documents: {status['total_failed_documents']}")
print(f"Eligible for recovery: {status['eligible_for_recovery']}")
print(f"Recovery success rate: {status['recovery_metrics']['recovery_success_rate']:.2%}")
```

### Failed Documents Analysis

The system provides analysis of failed documents by:

- **Age Distribution**: Categorizes failures by age (< 1 hour, 1-6 hours, etc.)
- **Error Types**: Groups similar errors for pattern identification
- **Worker Performance**: Tracks which workers have higher failure rates
- **Query Patterns**: Analyzes if certain query types fail more often

### Metrics Available

- `total_failed_documents`: Current number of failed documents
- `documents_eligible_for_retry`: Documents that can be retried now
- `recovery_attempts`: Total recovery attempts made
- `successful_recoveries`: Number of successful recoveries
- `failed_recoveries`: Number of failed recovery attempts
- `recovery_success_rate`: Percentage of successful recoveries
- `average_recovery_time`: Average time per recovery attempt
- `last_recovery_run`: Timestamp of last recovery run

## Error Handling

The recovery system handles various error types:

- **Import/Module Errors**: Often resolved after system updates
- **Memory Errors**: May succeed with different resource allocation
- **Timeout Errors**: May succeed with longer timeout values
- **File Not Found**: Documents that may have been moved or restored
- **Permission Errors**: May be resolved after permission fixes
- **Network Errors**: Transient connectivity issues

## Best Practices

1. **Configure Appropriate Intervals**: Set retry intervals based on your system's characteristics
2. **Monitor Success Rates**: Track recovery effectiveness and adjust parameters
3. **Regular Cleanup**: Ensure old failed records are cleaned up to avoid memory issues
4. **Error Analysis**: Review common error patterns to address root causes
5. **Resource Management**: Monitor Redis memory usage for failed document storage
6. **Testing**: Test recovery scenarios in development environments

## Troubleshooting

### Common Issues

1. **Redis Connection Errors**: Verify Redis server is running and accessible
2. **High Memory Usage**: Reduce retention period or increase cleanup frequency
3. **Low Success Rates**: Check if root causes of failures have been addressed
4. **Slow Recovery**: Reduce batch size or increase worker count

### Debugging

Enable verbose logging to troubleshoot recovery issues:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run recovery with detailed logging
result = await run_document_recovery(config={'verbose': True})
```

### Recovery State Inspection

```python
# Get detailed failed documents information
failed_docs = await tracker.get_failed_documents()

for doc in failed_docs:
    print(f"Document: {doc.document_path}")
    print(f"Error: {doc.error_message}")
    print(f"Retry count: {doc.retry_count}")
    print(f"Age: {doc.get_failure_age_hours():.1f} hours")
    print("---")
```

## Integration with Main Pipeline

The recovery system is integrated into the main processing pipeline through several mechanisms:

1. **Automatic Initialization**: Recovery components are initialized when the distributed processor starts
2. **Failure Tracking**: Every processing failure is automatically tracked
3. **Success Cleanup**: Successfully processed documents are removed from failed tracking
4. **Periodic Recovery**: Background recovery runs at configurable intervals
5. **Command Line Access**: Manual recovery via `python main.py --recover`

This ensures that the recovery system operates seamlessly as part of the overall document processing workflow, providing resilience against various failure scenarios.
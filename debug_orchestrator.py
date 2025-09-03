#!/usr/bin/env python3
"""
Debug script for orchestrator issues
"""

from event_bus import EventBus
from event_driven_orchestrator import EventDrivenOrchestrator
import time

def main():
    # Test basic setup
    event_bus = EventBus()
    orchestrator = EventDrivenOrchestrator(event_bus, 'test')
    print('Components initialized successfully')

    # Test pipeline execution
    def test_handler(data):
        print(f'Handler called with: {data}')
        return {'test': 'result'}

    orchestrator.register_stage_handler('test_stage', test_handler)
    print('Handler registered')

    execution_id = orchestrator.start_pipeline_execution(
        {'stages': ['test_stage']}, 
        {'input': 'data'}
    )
    print(f'Execution started: {execution_id}')

    # Monitor execution
    for i in range(10):
        time.sleep(0.1)
        status = orchestrator.get_execution_status(execution_id)
        print(f'Attempt {i}: Status = {status}')
        if status and status.get('is_complete'):
            break

    final_status = orchestrator.get_execution_status(execution_id)
    print(f'Final Status: {final_status}')

    orchestrator.shutdown()
    event_bus.shutdown()
    print('Test completed')

if __name__ == '__main__':
    main()
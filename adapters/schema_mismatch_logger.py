"""
Schema Mismatch Logger

Captures incompatibilities between retrieval outputs and analysis inputs,
routing logs to a lineage tracking system for dependency violation monitoring.
"""

import json
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime
from .data_transfer_objects import SchemaMismatchEvent, LineageEvent


class SchemaMismatchLogger:
    """Logs and tracks schema mismatches between components"""
    
    def __init__(self, lineage_tracker=None):
        self.logger = logging.getLogger(__name__)
        self.lineage_tracker = lineage_tracker
        self.mismatch_events: List[SchemaMismatchEvent] = []
        
        # Configure schema mismatch logging
        self._setup_schema_logging()
    
    def _setup_schema_logging(self):
        """Configure dedicated schema mismatch logging"""
        schema_handler = logging.FileHandler('logs/schema_mismatches.log')
        schema_handler.setFormatter(
            logging.Formatter(
                '%(asctime)s - %(levelname)s - SCHEMA_MISMATCH - %(message)s'
            )
        )
        
        schema_logger = logging.getLogger('schema_mismatch')
        schema_logger.addHandler(schema_handler)
        schema_logger.setLevel(logging.WARNING)
        
        self.schema_logger = schema_logger
    
    def log_mismatch(
        self, 
        source_schema: str,
        target_schema: str,
        source_data: Dict[str, Any],
        mismatch_details: List[str],
        adapter_id: str
    ) -> SchemaMismatchEvent:
        """Log a schema mismatch event"""
        
        event = SchemaMismatchEvent(
            source_schema=source_schema,
            target_schema=target_schema,
            source_data=source_data,
            mismatch_details=mismatch_details,
            adapter_id=adapter_id
        )
        
        self.mismatch_events.append(event)
        
        # Log to dedicated schema mismatch logger
        mismatch_msg = {
            'adapter_id': adapter_id,
            'source_schema': source_schema,
            'target_schema': target_schema,
            'mismatch_count': len(mismatch_details),
            'mismatches': mismatch_details,
            'timestamp': event.timestamp.isoformat()
        }
        
        self.schema_logger.warning(json.dumps(mismatch_msg))
        
        # Route to lineage tracker if available
        if self.lineage_tracker:
            lineage_event = LineageEvent(
                component_id=adapter_id,
                operation_type="schema_translation",
                input_schema=source_schema,
                output_schema=target_schema,
                dependencies=[],
                violation_type="schema_mismatch"
            )
            
            self.lineage_tracker.track_dependency_violation(lineage_event)
        
        return event
    
    def validate_retrieval_to_analysis(
        self, 
        retrieval_output: Dict[str, Any], 
        expected_analysis_input: Dict[str, str]
    ) -> List[str]:
        """Validate compatibility between retrieval output and analysis input schemas"""
        
        mismatches = []
        
        # Check required fields for analysis input
        for field, expected_type in expected_analysis_input.items():
            if field not in retrieval_output:
                mismatches.append(f"Missing required field: {field}")
            else:
                actual_value = retrieval_output[field]
                if not self._type_matches(actual_value, expected_type):
                    mismatches.append(
                        f"Type mismatch for {field}: expected {expected_type}, "
                        f"got {type(actual_value).__name__}"
                    )
        
        # Check for unexpected fields that might indicate schema drift
        expected_fields = set(expected_analysis_input.keys())
        actual_fields = set(retrieval_output.keys())
        unexpected_fields = actual_fields - expected_fields
        
        if unexpected_fields:
            mismatches.append(f"Unexpected fields: {list(unexpected_fields)}")
        
        return mismatches
    
    def _type_matches(self, value: Any, expected_type: str) -> bool:
        """Check if value matches expected type string"""
        
        type_map = {
            'str': str,
            'int': int,
            'float': float,
            'bool': bool,
            'list': list,
            'dict': dict,
            'Any': object
        }
        
        if expected_type in type_map:
            return isinstance(value, type_map[expected_type])
        
        # Handle complex types like List[str], Dict[str, Any]
        if expected_type.startswith('List['):
            return isinstance(value, list)
        elif expected_type.startswith('Dict['):
            return isinstance(value, dict)
        elif expected_type.startswith('Optional['):
            return value is None or self._type_matches(value, expected_type[9:-1])
        
        return True  # Default to accept for unknown types
    
    def get_mismatch_summary(self, adapter_id: Optional[str] = None) -> Dict[str, Any]:
        """Get summary of schema mismatches"""
        
        events = self.mismatch_events
        if adapter_id:
            events = [e for e in events if e.adapter_id == adapter_id]
        
        if not events:
            return {
                'total_events': 0,
                'adapters_affected': 0,
                'most_common_mismatches': [],
                'schema_pairs': []
            }
        
        # Analyze mismatches
        mismatch_counts = {}
        schema_pairs = set()
        adapters = set()
        
        for event in events:
            adapters.add(event.adapter_id)
            schema_pairs.add((event.source_schema, event.target_schema))
            
            for detail in event.mismatch_details:
                mismatch_counts[detail] = mismatch_counts.get(detail, 0) + 1
        
        # Get most common mismatches
        most_common = sorted(
            mismatch_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:10]
        
        return {
            'total_events': len(events),
            'adapters_affected': len(adapters),
            'most_common_mismatches': most_common,
            'schema_pairs': list(schema_pairs),
            'latest_event': events[-1].timestamp.isoformat() if events else None
        }
    
    def clear_events(self, older_than: Optional[datetime] = None):
        """Clear mismatch events, optionally only those older than specified time"""
        
        if older_than:
            self.mismatch_events = [
                e for e in self.mismatch_events 
                if e.timestamp > older_than
            ]
        else:
            self.mismatch_events.clear()
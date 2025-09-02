"""
Provenance Tracking System
==========================

Generates enhancement metadata and audit trails documenting auto-activated features,
activation criteria satisfaction, performance impact metrics, and complete lifecycle events.
"""

import json
import logging
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
from pydantic import BaseModel


class EnhancementLifecycleState(Enum):
    """Enhancement lifecycle states"""
    PENDING_ACTIVATION = "pending_activation"
    ACTIVE = "active"
    DEACTIVATED = "deactivated"
    COOLDOWN = "cooldown"
    PERMANENTLY_DISABLED = "permanently_disabled"


class ActivationCriteriaType(Enum):
    """Types of activation criteria"""
    PERFORMANCE_THRESHOLD = "performance_threshold"
    STABILITY_REQUIREMENT = "stability_requirement"
    SAFETY_MARGIN = "safety_margin"
    CONFIDENCE_LEVEL = "confidence_level"
    RESOURCE_AVAILABILITY = "resource_availability"


@dataclass
class ActivationCriteria:
    """Single activation criteria definition"""
    criteria_id: str
    criteria_type: ActivationCriteriaType
    description: str
    threshold_value: float
    current_value: float
    satisfied: bool
    evaluation_timestamp: datetime
    metadata: Dict[str, Any] = None


@dataclass
class PerformanceImpact:
    """Performance impact metrics"""
    metric_name: str
    baseline_value: float
    current_value: float
    impact_percentage: float
    measurement_timestamp: datetime
    confidence_interval: Optional[Tuple[float, float]] = None


@dataclass
class LifecycleEvent:
    """Enhancement lifecycle event"""
    event_id: str
    enhancement_id: str
    event_type: str
    state_from: Optional[EnhancementLifecycleState]
    state_to: EnhancementLifecycleState
    trigger_condition: str
    triggering_metrics: Dict[str, float]
    timestamp: datetime
    metadata: Dict[str, Any] = None


class EnhancementMetadata(BaseModel):
    """Complete enhancement metadata"""
    enhancement_id: str
    enhancement_type: str
    description: str
    version: str
    
    # Lifecycle tracking
    current_state: EnhancementLifecycleState
    created_at: datetime
    activated_at: Optional[datetime] = None
    deactivated_at: Optional[datetime] = None
    total_active_duration: Optional[timedelta] = None
    
    # Activation criteria
    activation_criteria: List[ActivationCriteria]
    activation_decision: Dict[str, Any]
    criteria_satisfaction_score: float
    
    # Performance tracking
    performance_impacts: List[PerformanceImpact]
    baseline_metrics: Dict[str, float]
    current_metrics: Dict[str, float]
    
    # Audit trail
    lifecycle_events: List[LifecycleEvent]
    
    # Configuration and dependencies
    configuration: Dict[str, Any]
    dependencies: List[str] = []
    conflicts: List[str] = []
    
    # Risk and safety
    risk_assessment: Dict[str, Any]
    safety_validations: List[Dict[str, Any]] = []
    
    # Metadata
    checksum: str
    last_updated: datetime
    created_by: str = "auto_enhancement_orchestrator"
    tags: List[str] = []


class ProvenanceTracker:
    """Main provenance tracking system"""
    
    def __init__(self, metadata_dir: str = "calibration_safety_governance/metadata"):
        self.logger = logging.getLogger(__name__)
        self.metadata_dir = Path(metadata_dir)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory tracking
        self.enhancement_metadata: Dict[str, EnhancementMetadata] = {}
        self.event_sequence: List[LifecycleEvent] = []
        
        # Load existing metadata
        self._load_existing_metadata()
        
    def _load_existing_metadata(self):
        """Load existing enhancement metadata from disk"""
        try:
            metadata_files = list(self.metadata_dir.glob("enhancement_*.json"))
            
            for metadata_file in metadata_files:
                try:
                    with open(metadata_file, 'r') as f:
                        data = json.load(f)
                        
                    # Convert timestamps back to datetime objects
                    data = self._deserialize_timestamps(data)
                    
                    # Create EnhancementMetadata object
                    metadata = EnhancementMetadata(**data)
                    self.enhancement_metadata[metadata.enhancement_id] = metadata
                    
                    # Add lifecycle events to sequence
                    self.event_sequence.extend(metadata.lifecycle_events)
                    
                except Exception as e:
                    self.logger.error(f"Failed to load metadata from {metadata_file}: {e}")
                    
            self.logger.info(f"Loaded metadata for {len(self.enhancement_metadata)} enhancements")
            
        except Exception as e:
            self.logger.error(f"Failed to load existing metadata: {e}")
            
    def _deserialize_timestamps(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert ISO timestamp strings back to datetime objects"""
        timestamp_fields = [
            'created_at', 'activated_at', 'deactivated_at', 'last_updated'
        ]
        
        for field in timestamp_fields:
            if field in data and data[field]:
                data[field] = datetime.fromisoformat(data[field])
                
        # Handle nested timestamp fields
        if 'activation_criteria' in data:
            for criteria in data['activation_criteria']:
                if 'evaluation_timestamp' in criteria:
                    criteria['evaluation_timestamp'] = datetime.fromisoformat(criteria['evaluation_timestamp'])
                    
        if 'performance_impacts' in data:
            for impact in data['performance_impacts']:
                if 'measurement_timestamp' in impact:
                    impact['measurement_timestamp'] = datetime.fromisoformat(impact['measurement_timestamp'])
                    
        if 'lifecycle_events' in data:
            for event in data['lifecycle_events']:
                if 'timestamp' in event:
                    event['timestamp'] = datetime.fromisoformat(event['timestamp'])
                    
        # Handle timedelta fields
        if 'total_active_duration' in data and data['total_active_duration']:
            # Convert seconds to timedelta
            data['total_active_duration'] = timedelta(seconds=data['total_active_duration'])
            
        return data

    def create_enhancement_metadata(
        self,
        enhancement_id: str,
        enhancement_type: str,
        description: str,
        configuration: Dict[str, Any],
        activation_criteria: List[Dict[str, Any]],
        baseline_metrics: Dict[str, float],
        dependencies: List[str] = None,
        tags: List[str] = None
    ) -> EnhancementMetadata:
        """Create new enhancement metadata"""
        
        # Convert activation criteria to structured format
        criteria_objects = []
        for i, criteria in enumerate(activation_criteria):
            criteria_obj = ActivationCriteria(
                criteria_id=f"{enhancement_id}_criteria_{i}",
                criteria_type=ActivationCriteriaType(criteria.get("type", "performance_threshold")),
                description=criteria.get("description", ""),
                threshold_value=criteria.get("threshold", 0.0),
                current_value=0.0,  # Will be updated during evaluation
                satisfied=False,
                evaluation_timestamp=datetime.now(),
                metadata=criteria.get("metadata", {})
            )
            criteria_objects.append(criteria_obj)
            
        # Generate initial checksum
        config_str = json.dumps(configuration, sort_keys=True)
        checksum = hashlib.sha256(config_str.encode()).hexdigest()[:16]
        
        # Create metadata object
        metadata = EnhancementMetadata(
            enhancement_id=enhancement_id,
            enhancement_type=enhancement_type,
            description=description,
            version="1.0.0",
            current_state=EnhancementLifecycleState.PENDING_ACTIVATION,
            created_at=datetime.now(),
            activation_criteria=criteria_objects,
            activation_decision={},
            criteria_satisfaction_score=0.0,
            performance_impacts=[],
            baseline_metrics=baseline_metrics,
            current_metrics=baseline_metrics.copy(),
            lifecycle_events=[],
            configuration=configuration,
            dependencies=dependencies or [],
            conflicts=[],
            risk_assessment={},
            safety_validations=[],
            checksum=checksum,
            last_updated=datetime.now(),
            tags=tags or []
        )
        
        # Record creation event
        creation_event = LifecycleEvent(
            event_id=str(uuid.uuid4()),
            enhancement_id=enhancement_id,
            event_type="created",
            state_from=None,
            state_to=EnhancementLifecycleState.PENDING_ACTIVATION,
            trigger_condition="enhancement_requested",
            triggering_metrics=baseline_metrics,
            timestamp=datetime.now(),
            metadata={"configuration_checksum": checksum}
        )
        
        metadata.lifecycle_events.append(creation_event)
        self.event_sequence.append(creation_event)
        
        # Store metadata
        self.enhancement_metadata[enhancement_id] = metadata
        self._persist_metadata(metadata)
        
        self.logger.info(f"Created metadata for enhancement {enhancement_id}")
        
        return metadata

    def evaluate_activation_criteria(
        self,
        enhancement_id: str,
        current_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """Evaluate activation criteria for enhancement"""
        
        if enhancement_id not in self.enhancement_metadata:
            raise ValueError(f"Enhancement {enhancement_id} not found")
            
        metadata = self.enhancement_metadata[enhancement_id]
        evaluation_timestamp = datetime.now()
        
        # Evaluate each criteria
        satisfied_count = 0
        total_weight = 0.0
        weighted_satisfaction = 0.0
        
        for criteria in metadata.activation_criteria:
            # Get current value for this criteria type
            current_value = self._extract_criteria_value(criteria.criteria_type, current_metrics)
            criteria.current_value = current_value
            criteria.evaluation_timestamp = evaluation_timestamp
            
            # Evaluate satisfaction based on criteria type
            if criteria.criteria_type == ActivationCriteriaType.PERFORMANCE_THRESHOLD:
                criteria.satisfied = current_value >= criteria.threshold_value
            elif criteria.criteria_type == ActivationCriteriaType.STABILITY_REQUIREMENT:
                criteria.satisfied = current_value >= criteria.threshold_value
            elif criteria.criteria_type == ActivationCriteriaType.SAFETY_MARGIN:
                criteria.satisfied = current_value >= criteria.threshold_value
            elif criteria.criteria_type == ActivationCriteriaType.CONFIDENCE_LEVEL:
                criteria.satisfied = current_value >= criteria.threshold_value
            else:
                criteria.satisfied = current_value >= criteria.threshold_value
                
            if criteria.satisfied:
                satisfied_count += 1
                
            # Weight criteria (safety and stability have higher weights)
            weight = self._get_criteria_weight(criteria.criteria_type)
            total_weight += weight
            
            if criteria.satisfied:
                weighted_satisfaction += weight
                
        # Calculate satisfaction score
        satisfaction_score = weighted_satisfaction / total_weight if total_weight > 0 else 0.0
        metadata.criteria_satisfaction_score = satisfaction_score
        
        # Make activation decision
        activation_decision = {
            "should_activate": satisfaction_score >= 0.8,  # Require 80% weighted satisfaction
            "satisfied_criteria": satisfied_count,
            "total_criteria": len(metadata.activation_criteria),
            "satisfaction_score": satisfaction_score,
            "evaluation_timestamp": evaluation_timestamp.isoformat(),
            "blocking_criteria": [
                {
                    "criteria_id": c.criteria_id,
                    "description": c.description,
                    "required": c.threshold_value,
                    "current": c.current_value,
                    "gap": c.threshold_value - c.current_value
                }
                for c in metadata.activation_criteria if not c.satisfied
            ]
        }
        
        metadata.activation_decision = activation_decision
        metadata.current_metrics = current_metrics
        metadata.last_updated = datetime.now()
        
        # Record evaluation event
        evaluation_event = LifecycleEvent(
            event_id=str(uuid.uuid4()),
            enhancement_id=enhancement_id,
            event_type="criteria_evaluated",
            state_from=metadata.current_state,
            state_to=metadata.current_state,
            trigger_condition="scheduled_evaluation",
            triggering_metrics=current_metrics,
            timestamp=evaluation_timestamp,
            metadata={
                "satisfaction_score": satisfaction_score,
                "satisfied_criteria": satisfied_count
            }
        )
        
        metadata.lifecycle_events.append(evaluation_event)
        self.event_sequence.append(evaluation_event)
        
        self._persist_metadata(metadata)
        
        return activation_decision

    def _extract_criteria_value(self, criteria_type: ActivationCriteriaType, metrics: Dict[str, float]) -> float:
        """Extract relevant metric value for criteria type"""
        type_mapping = {
            ActivationCriteriaType.PERFORMANCE_THRESHOLD: "performance_score",
            ActivationCriteriaType.STABILITY_REQUIREMENT: "stability_score",
            ActivationCriteriaType.SAFETY_MARGIN: "safety_score",
            ActivationCriteriaType.CONFIDENCE_LEVEL: "confidence_score",
            ActivationCriteriaType.RESOURCE_AVAILABILITY: "resource_availability"
        }
        
        metric_key = type_mapping.get(criteria_type, "overall_score")
        return metrics.get(metric_key, 0.0)

    def _get_criteria_weight(self, criteria_type: ActivationCriteriaType) -> float:
        """Get weight for criteria type in activation decision"""
        weights = {
            ActivationCriteriaType.SAFETY_MARGIN: 0.4,
            ActivationCriteriaType.STABILITY_REQUIREMENT: 0.3,
            ActivationCriteriaType.PERFORMANCE_THRESHOLD: 0.2,
            ActivationCriteriaType.CONFIDENCE_LEVEL: 0.1,
            ActivationCriteriaType.RESOURCE_AVAILABILITY: 0.1
        }
        
        return weights.get(criteria_type, 0.1)

    def record_activation(self, enhancement_id: str, activation_context: Dict[str, Any]) -> bool:
        """Record enhancement activation"""
        if enhancement_id not in self.enhancement_metadata:
            raise ValueError(f"Enhancement {enhancement_id} not found")
            
        metadata = self.enhancement_metadata[enhancement_id]
        
        if metadata.current_state != EnhancementLifecycleState.PENDING_ACTIVATION:
            self.logger.warning(f"Enhancement {enhancement_id} not in pending state: {metadata.current_state}")
            return False
            
        # Update state
        previous_state = metadata.current_state
        metadata.current_state = EnhancementLifecycleState.ACTIVE
        metadata.activated_at = datetime.now()
        metadata.last_updated = datetime.now()
        
        # Record activation event
        activation_event = LifecycleEvent(
            event_id=str(uuid.uuid4()),
            enhancement_id=enhancement_id,
            event_type="activated",
            state_from=previous_state,
            state_to=EnhancementLifecycleState.ACTIVE,
            trigger_condition="criteria_satisfied",
            triggering_metrics=metadata.current_metrics,
            timestamp=metadata.activated_at,
            metadata=activation_context
        )
        
        metadata.lifecycle_events.append(activation_event)
        self.event_sequence.append(activation_event)
        
        self._persist_metadata(metadata)
        
        self.logger.info(f"Enhancement {enhancement_id} activated")
        
        return True

    def record_deactivation(
        self,
        enhancement_id: str,
        deactivation_reason: str,
        triggering_metrics: Dict[str, float],
        deactivation_context: Dict[str, Any]
    ) -> bool:
        """Record enhancement deactivation"""
        if enhancement_id not in self.enhancement_metadata:
            raise ValueError(f"Enhancement {enhancement_id} not found")
            
        metadata = self.enhancement_metadata[enhancement_id]
        
        if metadata.current_state != EnhancementLifecycleState.ACTIVE:
            self.logger.warning(f"Enhancement {enhancement_id} not active: {metadata.current_state}")
            return False
            
        # Calculate active duration
        if metadata.activated_at:
            active_duration = datetime.now() - metadata.activated_at
            metadata.total_active_duration = active_duration
            
        # Update state
        previous_state = metadata.current_state
        metadata.current_state = EnhancementLifecycleState.DEACTIVATED
        metadata.deactivated_at = datetime.now()
        metadata.last_updated = datetime.now()
        
        # Record deactivation event
        deactivation_event = LifecycleEvent(
            event_id=str(uuid.uuid4()),
            enhancement_id=enhancement_id,
            event_type="deactivated",
            state_from=previous_state,
            state_to=EnhancementLifecycleState.DEACTIVATED,
            trigger_condition=deactivation_reason,
            triggering_metrics=triggering_metrics,
            timestamp=metadata.deactivated_at,
            metadata=deactivation_context
        )
        
        metadata.lifecycle_events.append(deactivation_event)
        self.event_sequence.append(deactivation_event)
        
        self._persist_metadata(metadata)
        
        self.logger.info(f"Enhancement {enhancement_id} deactivated: {deactivation_reason}")
        
        return True

    def record_performance_impact(
        self,
        enhancement_id: str,
        performance_metrics: Dict[str, float]
    ):
        """Record performance impact measurements"""
        if enhancement_id not in self.enhancement_metadata:
            raise ValueError(f"Enhancement {enhancement_id} not found")
            
        metadata = self.enhancement_metadata[enhancement_id]
        measurement_timestamp = datetime.now()
        
        # Calculate performance impacts
        for metric_name, current_value in performance_metrics.items():
            baseline_value = metadata.baseline_metrics.get(metric_name, 0.0)
            
            if baseline_value > 0:
                impact_percentage = ((current_value - baseline_value) / baseline_value) * 100
            else:
                impact_percentage = 0.0 if current_value == 0 else 100.0
                
            impact = PerformanceImpact(
                metric_name=metric_name,
                baseline_value=baseline_value,
                current_value=current_value,
                impact_percentage=impact_percentage,
                measurement_timestamp=measurement_timestamp
            )
            
            metadata.performance_impacts.append(impact)
            
        # Keep only recent performance impacts (last 100)
        if len(metadata.performance_impacts) > 100:
            metadata.performance_impacts = metadata.performance_impacts[-80:]
            
        metadata.current_metrics = performance_metrics
        metadata.last_updated = datetime.now()
        
        self._persist_metadata(metadata)

    def generate_enhancement_report(self, enhancement_id: str) -> Dict[str, Any]:
        """Generate comprehensive enhancement report"""
        if enhancement_id not in self.enhancement_metadata:
            raise ValueError(f"Enhancement {enhancement_id} not found")
            
        metadata = self.enhancement_metadata[enhancement_id]
        
        # Calculate summary statistics
        total_events = len(metadata.lifecycle_events)
        activations = len([e for e in metadata.lifecycle_events if e.event_type == "activated"])
        deactivations = len([e for e in metadata.lifecycle_events if e.event_type == "deactivated"])
        
        # Performance impact summary
        if metadata.performance_impacts:
            recent_impacts = metadata.performance_impacts[-10:]  # Last 10 measurements
            impact_summary = {}
            
            for metric_name in set(impact.metric_name for impact in recent_impacts):
                metric_impacts = [impact for impact in recent_impacts if impact.metric_name == metric_name]
                if metric_impacts:
                    latest_impact = metric_impacts[-1]
                    impact_summary[metric_name] = {
                        "baseline": latest_impact.baseline_value,
                        "current": latest_impact.current_value,
                        "impact_percentage": latest_impact.impact_percentage,
                        "trend": self._calculate_trend([i.current_value for i in metric_impacts])
                    }
        else:
            impact_summary = {}
            
        # Criteria satisfaction analysis
        criteria_analysis = {}
        if metadata.activation_criteria:
            for criteria in metadata.activation_criteria:
                criteria_analysis[criteria.criteria_id] = {
                    "type": criteria.criteria_type.value,
                    "description": criteria.description,
                    "threshold": criteria.threshold_value,
                    "current_value": criteria.current_value,
                    "satisfied": criteria.satisfied,
                    "satisfaction_gap": criteria.current_value - criteria.threshold_value
                }
                
        report = {
            "enhancement_id": enhancement_id,
            "enhancement_type": metadata.enhancement_type,
            "description": metadata.description,
            "current_state": metadata.current_state.value,
            
            # Lifecycle summary
            "lifecycle_summary": {
                "created_at": metadata.created_at.isoformat(),
                "activated_at": metadata.activated_at.isoformat() if metadata.activated_at else None,
                "deactivated_at": metadata.deactivated_at.isoformat() if metadata.deactivated_at else None,
                "total_active_duration_minutes": metadata.total_active_duration.total_seconds() / 60 if metadata.total_active_duration else None,
                "total_lifecycle_events": total_events,
                "activation_count": activations,
                "deactivation_count": deactivations
            },
            
            # Activation criteria
            "activation_analysis": {
                "criteria_satisfaction_score": metadata.criteria_satisfaction_score,
                "last_activation_decision": metadata.activation_decision,
                "criteria_details": criteria_analysis
            },
            
            # Performance analysis
            "performance_analysis": {
                "baseline_metrics": metadata.baseline_metrics,
                "current_metrics": metadata.current_metrics,
                "impact_summary": impact_summary,
                "total_measurements": len(metadata.performance_impacts)
            },
            
            # Configuration and dependencies
            "configuration": {
                "version": metadata.version,
                "dependencies": metadata.dependencies,
                "conflicts": metadata.conflicts,
                "configuration_checksum": metadata.checksum,
                "tags": metadata.tags
            },
            
            # Recent activity
            "recent_events": [
                {
                    "event_type": event.event_type,
                    "timestamp": event.timestamp.isoformat(),
                    "trigger_condition": event.trigger_condition,
                    "state_transition": f"{event.state_from.value if event.state_from else 'None'} -> {event.state_to.value}"
                }
                for event in metadata.lifecycle_events[-10:]  # Last 10 events
            ],
            
            "report_generated_at": datetime.now().isoformat(),
            "last_updated": metadata.last_updated.isoformat()
        }
        
        return report

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend from list of values"""
        if len(values) < 3:
            return "insufficient_data"
            
        # Simple trend analysis
        first_half = values[:len(values)//2]
        second_half = values[len(values)//2:]
        
        first_avg = sum(first_half) / len(first_half)
        second_avg = sum(second_half) / len(second_half)
        
        if second_avg > first_avg * 1.05:
            return "improving"
        elif second_avg < first_avg * 0.95:
            return "declining"
        else:
            return "stable"

    def _persist_metadata(self, metadata: EnhancementMetadata):
        """Persist metadata to disk"""
        try:
            # Update checksum
            config_str = json.dumps(metadata.configuration, sort_keys=True)
            metadata.checksum = hashlib.sha256(config_str.encode()).hexdigest()[:16]
            
            # Convert to serializable format
            serializable_data = self._make_serializable(metadata.dict())
            
            # Write to file
            metadata_file = self.metadata_dir / f"enhancement_{metadata.enhancement_id}.json"
            with open(metadata_file, 'w') as f:
                json.dump(serializable_data, f, indent=2, default=str)
                
        except Exception as e:
            self.logger.error(f"Failed to persist metadata for {metadata.enhancement_id}: {e}")

    def _make_serializable(self, data: Any) -> Any:
        """Convert data to JSON serializable format"""
        if isinstance(data, datetime):
            return data.isoformat()
        elif isinstance(data, timedelta):
            return data.total_seconds()
        elif isinstance(data, Enum):
            return data.value
        elif isinstance(data, dict):
            return {key: self._make_serializable(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._make_serializable(item) for item in data]
        else:
            return data

    def export_audit_trail(self, enhancement_id: Optional[str] = None, start_date: Optional[datetime] = None) -> Dict[str, Any]:
        """Export complete audit trail"""
        # Filter events
        if enhancement_id:
            events = [e for e in self.event_sequence if e.enhancement_id == enhancement_id]
        else:
            events = self.event_sequence.copy()
            
        if start_date:
            events = [e for e in events if e.timestamp >= start_date]
            
        # Sort by timestamp
        events.sort(key=lambda x: x.timestamp)
        
        audit_trail = {
            "export_timestamp": datetime.now().isoformat(),
            "filter_criteria": {
                "enhancement_id": enhancement_id,
                "start_date": start_date.isoformat() if start_date else None
            },
            "total_events": len(events),
            "enhancement_count": len(set(e.enhancement_id for e in events)),
            "date_range": {
                "earliest": events[0].timestamp.isoformat() if events else None,
                "latest": events[-1].timestamp.isoformat() if events else None
            },
            "events": [
                {
                    "event_id": event.event_id,
                    "enhancement_id": event.enhancement_id,
                    "event_type": event.event_type,
                    "state_transition": f"{event.state_from.value if event.state_from else 'None'} -> {event.state_to.value}",
                    "trigger_condition": event.trigger_condition,
                    "triggering_metrics": event.triggering_metrics,
                    "timestamp": event.timestamp.isoformat(),
                    "metadata": event.metadata
                }
                for event in events
            ]
        }
        
        return audit_trail

    def get_system_provenance_summary(self) -> Dict[str, Any]:
        """Get comprehensive system provenance summary"""
        active_enhancements = [m for m in self.enhancement_metadata.values() if m.current_state == EnhancementLifecycleState.ACTIVE]
        
        return {
            "summary_timestamp": datetime.now().isoformat(),
            "total_enhancements": len(self.enhancement_metadata),
            "active_enhancements": len(active_enhancements),
            "total_lifecycle_events": len(self.event_sequence),
            "enhancement_states": {
                state.value: len([m for m in self.enhancement_metadata.values() if m.current_state == state])
                for state in EnhancementLifecycleState
            },
            "recent_activity": len([e for e in self.event_sequence if e.timestamp > datetime.now() - timedelta(hours=24)]),
            "active_enhancement_details": [
                {
                    "enhancement_id": metadata.enhancement_id,
                    "type": metadata.enhancement_type,
                    "activated_at": metadata.activated_at.isoformat() if metadata.activated_at else None,
                    "active_duration_minutes": (datetime.now() - metadata.activated_at).total_seconds() / 60 if metadata.activated_at else None,
                    "criteria_satisfaction_score": metadata.criteria_satisfaction_score
                }
                for metadata in active_enhancements
            ]
        }
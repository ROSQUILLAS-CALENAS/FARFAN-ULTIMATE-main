"""
Refactored Analysis NLP Stage Orchestrator

This orchestrator uses dependency injection to work with validator_api interfaces
instead of concrete implementations, providing clean separation of concerns.
"""

import logging
import time
from typing import Any, Dict, List, Optional

from validator_api.interfaces import (
    IValidator,
    IEvidenceProcessor,
    IValidatorFactory,
    IEvidenceProcessorFactory
)
from validator_api.dtos import (
    ValidationRequest,
    ValidationResponse,
    EvidenceItem,
    ValidationCategory,
    DNPAlignmentCategory
)

logger = logging.getLogger(__name__)


class AnalysisNLPOrchestrator:
    """
    Orchestrator for Analysis NLP stage using dependency injection
    
    This orchestrator depends only on validator_api interfaces and uses
    dependency injection to obtain concrete implementations at runtime.
    """
    
    def __init__(
        self,
        validator_factory: IValidatorFactory,
        processor_factory: IEvidenceProcessorFactory,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize orchestrator with injected dependencies
        
        Args:
            validator_factory: Factory for creating validators
            processor_factory: Factory for creating evidence processors
            config: Optional configuration parameters
        """
        self.validator_factory = validator_factory
        self.processor_factory = processor_factory
        self.config = config or {}
        
        # Initialize components using dependency injection
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize components using the injected factories"""
        
        try:
            # Create validators based on configuration
            validator_config = self.config.get('validators', {})
            
            self.comprehensive_validator = self.validator_factory.create_validator(
                'comprehensive',
                validator_config.get('comprehensive', {})
            )
            
            self.dnp_validator = self.validator_factory.create_validator(
                'dnp_alignment',
                validator_config.get('dnp_alignment', {})
            )
            
            self.evidence_validator = self.validator_factory.create_validator(
                'evidence',
                validator_config.get('evidence', {})
            )
            
            # Create evidence processors based on configuration
            processor_config = self.config.get('processors', {})
            
            self.default_processor = self.processor_factory.create_processor(
                'default',
                processor_config.get('default', {})
            )
            
            self.dnp_processor = self.processor_factory.create_processor(
                'dnp',
                processor_config.get('dnp', {})
            )
            
            logger.info("Analysis NLP components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise
    
    def process_analysis_request(
        self,
        text_content: str,
        context: str = "",
        validation_categories: Optional[List[ValidationCategory]] = None,
        dnp_categories: Optional[List[DNPAlignmentCategory]] = None,
        evidence_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a complete analysis request
        
        Args:
            text_content: Text content to analyze
            context: Context for analysis
            validation_categories: Categories of validation to perform
            dnp_categories: DNP alignment categories to check
            evidence_data: Raw evidence data to process
            
        Returns:
            Complete analysis results
        """
        
        start_time = time.time()
        results = {
            'request_id': self._generate_request_id(text_content),
            'analysis_type': 'comprehensive',
            'status': 'processing'
        }
        
        try:
            # Step 1: Process evidence if provided
            processed_evidence = []
            if evidence_data:
                processed_evidence = self._process_evidence(evidence_data)
                results['evidence_processing'] = {
                    'items_processed': len(processed_evidence),
                    'processing_successful': True
                }
            
            # Step 2: Perform validation
            validation_results = self._perform_validation(
                text_content,
                context,
                validation_categories,
                dnp_categories
            )
            results['validation'] = validation_results
            
            # Step 3: Generate analysis summary
            analysis_summary = self._generate_analysis_summary(
                validation_results,
                processed_evidence,
                text_content
            )
            results['analysis_summary'] = analysis_summary
            
            # Step 4: Calculate overall scores
            overall_scores = self._calculate_overall_scores(validation_results)
            results['scores'] = overall_scores
            
            results['status'] = 'completed'
            results['processing_time_ms'] = int((time.time() - start_time) * 1000)
            
            logger.info(f"Analysis request processed successfully: {results['request_id']}")
            
        except Exception as e:
            logger.error(f"Analysis processing failed: {e}")
            results['status'] = 'failed'
            results['error'] = str(e)
            results['processing_time_ms'] = int((time.time() - start_time) * 1000)
        
        return results
    
    def _process_evidence(self, evidence_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process evidence data using appropriate processor"""
        
        # Determine which processor to use based on data characteristics
        processor = self._select_evidence_processor(evidence_data)
        
        try:
            processed_evidence = processor.process_evidence(evidence_data)
            
            # Validate processed evidence structure
            valid_evidence = []
            for item in processed_evidence:
                if processor.validate_evidence_structure(item):
                    # Enhance with metadata
                    metadata = processor.get_evidence_metadata(item)
                    item['metadata'] = metadata
                    valid_evidence.append(item)
                else:
                    logger.warning(f"Invalid evidence structure, skipping item: {item.get('id', 'unknown')}")
            
            logger.info(f"Processed {len(valid_evidence)} valid evidence items")
            return valid_evidence
            
        except Exception as e:
            logger.error(f"Evidence processing failed: {e}")
            return []
    
    def _select_evidence_processor(self, evidence_data: Dict[str, Any]) -> IEvidenceProcessor:
        """Select appropriate evidence processor based on data characteristics"""
        
        # Check if data has DNP-specific indicators
        if self._has_dnp_indicators(evidence_data):
            return self.dnp_processor
        else:
            return self.default_processor
    
    def _has_dnp_indicators(self, evidence_data: Dict[str, Any]) -> bool:
        """Check if evidence data has DNP-specific indicators"""
        
        # Convert data to string for analysis
        data_str = str(evidence_data).lower()
        
        dnp_keywords = [
            'departamento nacional de planeación', 'dnp', 'plan nacional',
            'política pública', 'desarrollo territorial', 'constitución',
            'decreto', 'ley', 'resolución', 'normativa'
        ]
        
        return any(keyword in data_str for keyword in dnp_keywords)
    
    def _perform_validation(
        self,
        text_content: str,
        context: str,
        validation_categories: Optional[List[ValidationCategory]],
        dnp_categories: Optional[List[DNPAlignmentCategory]]
    ) -> Dict[str, Any]:
        """Perform comprehensive validation using multiple validators"""
        
        validation_results = {
            'comprehensive': None,
            'dnp_alignment': None,
            'evidence': None,
            'overall': None
        }
        
        # Create validation request
        request = ValidationRequest(
            evidence_text=text_content,
            context=context,
            validation_type='comprehensive',
            validation_categories=validation_categories or [],
            dnp_alignment_categories=dnp_categories or []
        )
        
        # Comprehensive validation
        try:
            comprehensive_response = self.comprehensive_validator.validate(request)
            validation_results['comprehensive'] = comprehensive_response.to_dict()
        except Exception as e:
            logger.error(f"Comprehensive validation failed: {e}")
            validation_results['comprehensive'] = {'error': str(e)}
        
        # DNP alignment validation
        if dnp_categories:
            try:
                dnp_response = self.dnp_validator.validate(request)
                validation_results['dnp_alignment'] = dnp_response.to_dict()
            except Exception as e:
                logger.error(f"DNP validation failed: {e}")
                validation_results['dnp_alignment'] = {'error': str(e)}
        
        # Evidence structure validation
        try:
            evidence_response = self.evidence_validator.validate(request)
            validation_results['evidence'] = evidence_response.to_dict()
        except Exception as e:
            logger.error(f"Evidence validation failed: {e}")
            validation_results['evidence'] = {'error': str(e)}
        
        # Calculate overall validation result
        validation_results['overall'] = self._calculate_overall_validation(validation_results)
        
        return validation_results
    
    def _calculate_overall_validation(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall validation result from individual results"""
        
        overall_valid = True
        total_confidence = 0.0
        confidence_count = 0
        all_messages = []
        
        for key, result in validation_results.items():
            if key == 'overall' or result is None:
                continue
            
            if isinstance(result, dict) and 'error' not in result:
                # Extract validation status
                is_valid = result.get('is_valid', False)
                confidence = result.get('confidence_score', 0.0)
                
                overall_valid = overall_valid and is_valid
                total_confidence += confidence
                confidence_count += 1
                
                # Collect messages
                if 'validation_results' in result:
                    for vr in result['validation_results']:
                        if 'messages' in vr:
                            all_messages.extend(vr['messages'])
        
        avg_confidence = total_confidence / confidence_count if confidence_count > 0 else 0.0
        
        return {
            'is_valid': overall_valid,
            'confidence_score': avg_confidence,
            'validation_count': confidence_count,
            'messages': all_messages[:10],  # Limit to first 10 messages
            'summary': f"Overall validation {'passed' if overall_valid else 'failed'} with {avg_confidence:.2f} confidence"
        }
    
    def _generate_analysis_summary(
        self,
        validation_results: Dict[str, Any],
        processed_evidence: List[Dict[str, Any]],
        text_content: str
    ) -> Dict[str, Any]:
        """Generate analysis summary"""
        
        return {
            'content_length': len(text_content),
            'word_count': len(text_content.split()),
            'evidence_count': len(processed_evidence),
            'validation_performed': len([r for r in validation_results.values() if r is not None and 'error' not in (r if isinstance(r, dict) else {})]),
            'overall_status': validation_results.get('overall', {}).get('is_valid', False),
            'confidence_score': validation_results.get('overall', {}).get('confidence_score', 0.0),
            'primary_issues': self._extract_primary_issues(validation_results),
            'recommendations': self._generate_recommendations(validation_results, processed_evidence)
        }
    
    def _extract_primary_issues(self, validation_results: Dict[str, Any]) -> List[str]:
        """Extract primary issues from validation results"""
        
        issues = []
        
        for key, result in validation_results.items():
            if key == 'overall' or not isinstance(result, dict) or 'error' in result:
                continue
            
            if not result.get('is_valid', True):
                # Extract high-priority messages
                if 'validation_results' in result:
                    for vr in result['validation_results']:
                        if not vr.get('is_valid', True):
                            severity = vr.get('severity', 'unknown')
                            if severity in ['critical', 'high']:
                                issues.extend(vr.get('messages', []))
        
        return issues[:5]  # Return top 5 issues
    
    def _generate_recommendations(
        self,
        validation_results: Dict[str, Any],
        processed_evidence: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate recommendations based on analysis results"""
        
        recommendations = []
        
        # Check overall validation status
        overall_result = validation_results.get('overall', {})
        if not overall_result.get('is_valid', True):
            recommendations.append("Review and address validation issues before proceeding")
        
        # Check confidence scores
        confidence = overall_result.get('confidence_score', 1.0)
        if confidence < 0.5:
            recommendations.append("Consider providing additional context or evidence to improve confidence")
        elif confidence < 0.7:
            recommendations.append("Review evidence quality and source reliability")
        
        # Check evidence count
        if len(processed_evidence) == 0:
            recommendations.append("Consider providing supporting evidence")
        elif len(processed_evidence) < 3:
            recommendations.append("Additional evidence items may strengthen the analysis")
        
        # DNP-specific recommendations
        dnp_result = validation_results.get('dnp_alignment', {})
        if dnp_result and not dnp_result.get('is_valid', True):
            recommendations.append("Ensure compliance with DNP regulatory requirements")
        
        return recommendations[:5]  # Return top 5 recommendations
    
    def _calculate_overall_scores(self, validation_results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate overall scores from validation results"""
        
        scores = {
            'overall_confidence': 0.0,
            'factual_accuracy': 0.0,
            'logical_consistency': 0.0,
            'source_reliability': 0.0,
            'dnp_compliance': 0.0,
            'evidence_quality': 0.0
        }
        
        # Extract scores from validation results
        overall_result = validation_results.get('overall', {})
        scores['overall_confidence'] = overall_result.get('confidence_score', 0.0)
        
        # Extract category-specific scores
        for key, result in validation_results.items():
            if not isinstance(result, dict) or 'error' in result:
                continue
            
            if 'validation_results' in result:
                for vr in result['validation_results']:
                    category = vr.get('category')
                    if category == 'factual_accuracy' and vr.get('is_valid', False):
                        scores['factual_accuracy'] = max(scores['factual_accuracy'], 0.8)
                    elif category == 'logical_consistency' and vr.get('is_valid', False):
                        scores['logical_consistency'] = max(scores['logical_consistency'], 0.8)
                    elif category == 'source_reliability' and vr.get('is_valid', False):
                        scores['source_reliability'] = max(scores['source_reliability'], 0.8)
            
            if key == 'dnp_alignment':
                scores['dnp_compliance'] = result.get('confidence_score', 0.0)
            elif key == 'evidence':
                scores['evidence_quality'] = result.get('confidence_score', 0.0)
        
        return scores
    
    def _generate_request_id(self, content: str) -> str:
        """Generate a unique request ID"""
        import hashlib
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        return f"analysis_{content_hash[:12]}"
    
    def get_supported_validation_types(self) -> Dict[str, List[str]]:
        """Get supported validation types from all validators"""
        
        return {
            'comprehensive': self.comprehensive_validator.get_supported_validation_types(),
            'dnp_alignment': self.dnp_validator.get_supported_validation_types(),
            'evidence': self.evidence_validator.get_supported_validation_types()
        }
    
    def get_supported_processor_types(self) -> List[str]:
        """Get supported evidence processor types"""
        
        return self.processor_factory.get_available_processor_types()
    
    def reconfigure(self, new_config: Dict[str, Any]) -> None:
        """Reconfigure the orchestrator with new settings"""
        
        self.config.update(new_config)
        
        # Reconfigure validators
        if 'validators' in new_config:
            for validator_type, config in new_config['validators'].items():
                if hasattr(self, f'{validator_type}_validator'):
                    validator = getattr(self, f'{validator_type}_validator')
                    validator.configure(config)
        
        # Note: Evidence processors are stateless and don't need reconfiguration
        
        logger.info("Orchestrator reconfigured successfully")


def create_analysis_orchestrator(
    validator_factory: IValidatorFactory,
    processor_factory: IEvidenceProcessorFactory,
    config: Optional[Dict[str, Any]] = None
) -> AnalysisNLPOrchestrator:
    """
    Factory function to create Analysis NLP Orchestrator with dependency injection
    
    Args:
        validator_factory: Factory for creating validators
        processor_factory: Factory for creating evidence processors
        config: Optional configuration parameters
        
    Returns:
        Configured AnalysisNLPOrchestrator instance
    """
    
    return AnalysisNLPOrchestrator(validator_factory, processor_factory, config)
"""
Factory implementations for creating validators and evidence processors

These factories implement dependency injection patterns and provide
centralized creation and configuration of validator components.
"""

import logging
from typing import Any, Dict, List, Optional, Type

from validator_api.interfaces import (
    IValidator,
    IEvidenceProcessor,
    IValidatorFactory,
    IEvidenceProcessorFactory
)

from .validators import ComprehensiveValidator, DNPAlignmentValidator, EvidenceValidator
from .evidence_processors import DefaultEvidenceProcessor, DNPEvidenceProcessor

logger = logging.getLogger(__name__)


class ValidatorFactory(IValidatorFactory):
    """Factory for creating validator instances"""
    
    def __init__(self):
        self._validator_registry: Dict[str, Type[IValidator]] = {
            "comprehensive": ComprehensiveValidator,
            "dnp_alignment": DNPAlignmentValidator,
            "evidence": EvidenceValidator,
            "basic": EvidenceValidator,
            
            # Aliases for backward compatibility
            "factual_accuracy": ComprehensiveValidator,
            "logical_consistency": ComprehensiveValidator,
            "source_reliability": ComprehensiveValidator,
            "completeness": ComprehensiveValidator,
            "relevance": ComprehensiveValidator,
            
            "constitutional": DNPAlignmentValidator,
            "regulatory": DNPAlignmentValidator,
            "procedural": DNPAlignmentValidator,
            "ethical": DNPAlignmentValidator,
            "technical": DNPAlignmentValidator
        }
    
    def create_validator(self, validator_type: str, config: Optional[Dict[str, Any]] = None) -> IValidator:
        """
        Create a validator of the specified type
        
        Args:
            validator_type: Type of validator to create
            config: Optional configuration parameters
            
        Returns:
            Configured validator instance
            
        Raises:
            ValueError: If validator type is not supported
        """
        
        if validator_type not in self._validator_registry:
            raise ValueError(
                f"Unsupported validator type: {validator_type}. "
                f"Supported types: {list(self._validator_registry.keys())}"
            )
        
        validator_class = self._validator_registry[validator_type]
        
        try:
            validator = validator_class(config=config)
            logger.info(f"Created validator of type: {validator_type}")
            return validator
        except Exception as e:
            logger.error(f"Failed to create validator of type {validator_type}: {e}")
            raise
    
    def get_available_validator_types(self) -> List[str]:
        """
        Get list of available validator types
        
        Returns:
            List of available validator type strings
        """
        return list(self._validator_registry.keys())
    
    def register_validator(self, validator_type: str, validator_class: Type[IValidator]) -> None:
        """
        Register a new validator type
        
        Args:
            validator_type: String identifier for the validator type
            validator_class: Validator class to register
        """
        if not issubclass(validator_class, IValidator):
            raise ValueError("Validator class must implement IValidator interface")
        
        self._validator_registry[validator_type] = validator_class
        logger.info(f"Registered validator type: {validator_type}")
    
    def unregister_validator(self, validator_type: str) -> None:
        """
        Unregister a validator type
        
        Args:
            validator_type: String identifier for the validator type to remove
        """
        if validator_type in self._validator_registry:
            del self._validator_registry[validator_type]
            logger.info(f"Unregistered validator type: {validator_type}")
    
    def create_validator_chain(self, validator_types: List[str], config: Optional[Dict[str, Any]] = None) -> List[IValidator]:
        """
        Create a chain of validators
        
        Args:
            validator_types: List of validator types to create
            config: Optional shared configuration
            
        Returns:
            List of configured validator instances
        """
        validators = []
        
        for validator_type in validator_types:
            try:
                validator = self.create_validator(validator_type, config)
                validators.append(validator)
            except Exception as e:
                logger.warning(f"Failed to create validator {validator_type}: {e}")
                continue
        
        logger.info(f"Created validator chain with {len(validators)} validators")
        return validators


class EvidenceProcessorFactory(IEvidenceProcessorFactory):
    """Factory for creating evidence processor instances"""
    
    def __init__(self):
        self._processor_registry: Dict[str, Type[IEvidenceProcessor]] = {
            "default": DefaultEvidenceProcessor,
            "dnp": DNPEvidenceProcessor,
            "basic": DefaultEvidenceProcessor,
            
            # Aliases for specific use cases
            "general": DefaultEvidenceProcessor,
            "standard": DefaultEvidenceProcessor,
            "dnp_enhanced": DNPEvidenceProcessor,
            "regulatory": DNPEvidenceProcessor,
            "legal": DNPEvidenceProcessor
        }
    
    def create_processor(self, processor_type: str, config: Optional[Dict[str, Any]] = None) -> IEvidenceProcessor:
        """
        Create an evidence processor of the specified type
        
        Args:
            processor_type: Type of processor to create
            config: Optional configuration parameters
            
        Returns:
            Configured processor instance
            
        Raises:
            ValueError: If processor type is not supported
        """
        
        if processor_type not in self._processor_registry:
            raise ValueError(
                f"Unsupported processor type: {processor_type}. "
                f"Supported types: {list(self._processor_registry.keys())}"
            )
        
        processor_class = self._processor_registry[processor_type]
        
        try:
            processor = processor_class(config=config)
            logger.info(f"Created evidence processor of type: {processor_type}")
            return processor
        except Exception as e:
            logger.error(f"Failed to create processor of type {processor_type}: {e}")
            raise
    
    def get_available_processor_types(self) -> List[str]:
        """
        Get list of available processor types
        
        Returns:
            List of available processor type strings
        """
        return list(self._processor_registry.keys())
    
    def register_processor(self, processor_type: str, processor_class: Type[IEvidenceProcessor]) -> None:
        """
        Register a new processor type
        
        Args:
            processor_type: String identifier for the processor type
            processor_class: Processor class to register
        """
        if not issubclass(processor_class, IEvidenceProcessor):
            raise ValueError("Processor class must implement IEvidenceProcessor interface")
        
        self._processor_registry[processor_type] = processor_class
        logger.info(f"Registered processor type: {processor_type}")
    
    def unregister_processor(self, processor_type: str) -> None:
        """
        Unregister a processor type
        
        Args:
            processor_type: String identifier for the processor type to remove
        """
        if processor_type in self._processor_registry:
            del self._processor_registry[processor_type]
            logger.info(f"Unregistered processor type: {processor_type}")
    
    def create_processor_pipeline(self, processor_types: List[str], config: Optional[Dict[str, Any]] = None) -> List[IEvidenceProcessor]:
        """
        Create a pipeline of evidence processors
        
        Args:
            processor_types: List of processor types to create
            config: Optional shared configuration
            
        Returns:
            List of configured processor instances
        """
        processors = []
        
        for processor_type in processor_types:
            try:
                processor = self.create_processor(processor_type, config)
                processors.append(processor)
            except Exception as e:
                logger.warning(f"Failed to create processor {processor_type}: {e}")
                continue
        
        logger.info(f"Created processor pipeline with {len(processors)} processors")
        return processors


class DependencyInjectionContainer:
    """Simple dependency injection container for validator components"""
    
    def __init__(self):
        self._validator_factory = ValidatorFactory()
        self._processor_factory = EvidenceProcessorFactory()
        self._singletons: Dict[str, Any] = {}
        self._configurations: Dict[str, Dict[str, Any]] = {}
    
    def register_configuration(self, component_name: str, config: Dict[str, Any]) -> None:
        """
        Register configuration for a component
        
        Args:
            component_name: Name of the component
            config: Configuration dictionary
        """
        self._configurations[component_name] = config
        logger.info(f"Registered configuration for: {component_name}")
    
    def get_validator_factory(self) -> IValidatorFactory:
        """Get the validator factory instance"""
        return self._validator_factory
    
    def get_processor_factory(self) -> IEvidenceProcessorFactory:
        """Get the evidence processor factory instance"""
        return self._processor_factory
    
    def get_validator(self, validator_type: str, singleton: bool = False) -> IValidator:
        """
        Get a validator instance
        
        Args:
            validator_type: Type of validator to get
            singleton: Whether to use singleton pattern
            
        Returns:
            Validator instance
        """
        
        if singleton:
            key = f"validator_{validator_type}"
            if key not in self._singletons:
                config = self._configurations.get(validator_type, {})
                self._singletons[key] = self._validator_factory.create_validator(validator_type, config)
            return self._singletons[key]
        else:
            config = self._configurations.get(validator_type, {})
            return self._validator_factory.create_validator(validator_type, config)
    
    def get_processor(self, processor_type: str, singleton: bool = False) -> IEvidenceProcessor:
        """
        Get an evidence processor instance
        
        Args:
            processor_type: Type of processor to get
            singleton: Whether to use singleton pattern
            
        Returns:
            Evidence processor instance
        """
        
        if singleton:
            key = f"processor_{processor_type}"
            if key not in self._singletons:
                config = self._configurations.get(processor_type, {})
                self._singletons[key] = self._processor_factory.create_processor(processor_type, config)
            return self._singletons[key]
        else:
            config = self._configurations.get(processor_type, {})
            return self._processor_factory.create_processor(processor_type, config)
    
    def configure_defaults(self) -> None:
        """Configure default settings for common components"""
        
        # Default validator configurations
        self.register_configuration("comprehensive", {
            "enable_all_categories": True,
            "strict_mode": False,
            "confidence_threshold": 0.7
        })
        
        self.register_configuration("dnp_alignment", {
            "include_constitutional": True,
            "include_regulatory": True,
            "include_procedural": True,
            "strict_compliance": False
        })
        
        self.register_configuration("evidence", {
            "validate_structure": True,
            "validate_content": True,
            "min_content_length": 10
        })
        
        # Default processor configurations
        self.register_configuration("default", {
            "extract_features": True,
            "calculate_confidence": True,
            "enhance_metadata": True
        })
        
        self.register_configuration("dnp", {
            "enable_legal_detection": True,
            "calculate_compliance_score": True,
            "extract_policy_indicators": True,
            "formality_analysis": True
        })
        
        logger.info("Configured default settings for validator components")
    
    def clear_singletons(self) -> None:
        """Clear all singleton instances"""
        self._singletons.clear()
        logger.info("Cleared all singleton instances")


# Global container instance for convenience
_global_container = DependencyInjectionContainer()


def get_global_container() -> DependencyInjectionContainer:
    """Get the global dependency injection container"""
    return _global_container


def configure_global_defaults() -> None:
    """Configure global default settings"""
    _global_container.configure_defaults()
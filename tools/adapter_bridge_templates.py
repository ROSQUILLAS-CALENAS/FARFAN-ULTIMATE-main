"""
Adapter and Bridge Pattern Templates for Breaking Circular Dependencies

This module provides code templates for common dependency resolution patterns
identified by the Dependency Doctor tool.
"""

from typing import Dict, Any, Optional

ADAPTER_TEMPLATE = '''
class {adapter_name}:
    """
    Adapter pattern to break circular dependency between {source_phase} and {target_phase}.
    
    This adapter allows {source_phase} modules to interact with {target_phase} functionality
    without creating direct backward dependencies that violate phase ordering.
    """
    
    def __init__(self):
        self._target_instance = None
    
    def set_target(self, target_instance):
        """Inject the target instance at runtime to avoid import-time dependency."""
        self._target_instance = target_instance
    
    def process(self, data: Any, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Adapter process method that delegates to the target.
        
        Args:
            data: Input data to process
            context: Optional context dictionary
            
        Returns:
            Dict[str, Any]: Processing results from target
        """
        if self._target_instance is None:
            raise RuntimeError("Target not set. Call set_target() before processing.")
        
        return self._target_instance.process(data, context)
    
    def {method_name}(self, *args, **kwargs):
        """Adapter method for specific {target_phase} functionality."""
        if self._target_instance is None:
            raise RuntimeError("Target not set. Call set_target() before calling {method_name}.")
        
        return getattr(self._target_instance, '{method_name}')(*args, **kwargs)
'''

BRIDGE_TEMPLATE = '''
class {bridge_name}:
    """
    Bridge pattern to decouple {source_phase} from {target_phase} implementation details.
    
    This bridge provides a stable interface for {source_phase} modules while allowing
    {target_phase} implementation to vary without affecting the phase ordering.
    """
    
    def __init__(self):
        self._implementation = None
    
    def set_implementation(self, implementation):
        """Set the concrete implementation at runtime."""
        self._implementation = implementation
    
    def process(self, data: Any, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Bridge process method that delegates to the implementation.
        
        Args:
            data: Input data to process
            context: Optional context dictionary
            
        Returns:
            Dict[str, Any]: Processing results from implementation
        """
        if self._implementation is None:
            raise RuntimeError("Implementation not set. Call set_implementation() before processing.")
        
        return self._implementation.process(data, context)
'''

PORT_INTERFACE_TEMPLATE = '''
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class {interface_name}(ABC):
    """
    Port interface for quarantining hot nodes and breaking circular dependencies.
    
    This interface defines the contract for {phase_name} operations without
    coupling to specific implementations, enabling proper phase ordering.
    """
    
    @abstractmethod
    def process(self, data: Any, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Standard process method interface.
        
        Args:
            data: Input data to process
            context: Optional context dictionary
            
        Returns:
            Dict[str, Any]: Processing results
        """
        pass
    
    @abstractmethod
    def validate_input(self, data: Any) -> bool:
        """
        Validate input data before processing.
        
        Args:
            data: Input data to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        pass
    
    @abstractmethod
    def get_dependencies(self) -> Dict[str, str]:
        """
        Get required dependencies for this component.
        
        Returns:
            Dict[str, str]: Mapping of dependency name to phase
        """
        pass
'''

DEPENDENCY_INJECTION_TEMPLATE = '''
class {container_name}:
    """
    Dependency injection container for managing phase dependencies.
    
    This container ensures proper initialization order and prevents
    circular dependencies by managing component lifecycle.
    """
    
    def __init__(self):
        self._services = {{}}
        self._factories = {{}}
        self._singletons = {{}}
    
    def register_service(self, name: str, service_class, singleton: bool = True):
        """Register a service class."""
        if singleton:
            self._singletons[name] = None
        self._services[name] = service_class
    
    def register_factory(self, name: str, factory_func):
        """Register a factory function for lazy initialization."""
        self._factories[name] = factory_func
    
    def get_service(self, name: str):
        """Get a service instance."""
        if name in self._singletons:
            if self._singletons[name] is None:
                self._singletons[name] = self._create_service(name)
            return self._singletons[name]
        
        return self._create_service(name)
    
    def _create_service(self, name: str):
        """Create service instance with dependency injection."""
        if name in self._factories:
            return self._factories[name]()
        
        if name in self._services:
            service_class = self._services[name]
            # Inject dependencies based on constructor parameters
            return service_class()
        
        raise ValueError(f"Service {{name}} not registered")
    
    def process_with_service(self, service_name: str, data: Any, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process data using a specific service."""
        service = self.get_service(service_name)
        return service.process(data, context)
'''

def generate_adapter(source_phase: str, target_phase: str, adapter_name: str = None, method_name: str = "execute") -> str:
    """Generate adapter code for breaking circular dependencies."""
    if adapter_name is None:
        adapter_name = f"{source_phase}To{target_phase}Adapter"
    
    return ADAPTER_TEMPLATE.format(
        adapter_name=adapter_name,
        source_phase=source_phase,
        target_phase=target_phase,
        method_name=method_name
    )

def generate_bridge(source_phase: str, target_phase: str, bridge_name: str = None) -> str:
    """Generate bridge code for decoupling implementations."""
    if bridge_name is None:
        bridge_name = f"{source_phase}{target_phase}Bridge"
    
    return BRIDGE_TEMPLATE.format(
        bridge_name=bridge_name,
        source_phase=source_phase,
        target_phase=target_phase
    )

def generate_port_interface(phase_name: str, interface_name: str = None) -> str:
    """Generate port interface for quarantining hot nodes."""
    if interface_name is None:
        interface_name = f"{phase_name}ProcessorInterface"
    
    return PORT_INTERFACE_TEMPLATE.format(
        interface_name=interface_name,
        phase_name=phase_name
    )

def generate_dependency_container(container_name: str = "DependencyContainer") -> str:
    """Generate dependency injection container."""
    return DEPENDENCY_INJECTION_TEMPLATE.format(
        container_name=container_name
    )
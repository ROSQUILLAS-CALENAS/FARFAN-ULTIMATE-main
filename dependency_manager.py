"""
Gestor de dependencias que evita aislamiento de procesos
y maneja relaciones complejas entre servicios y workflows
"""

import asyncio
# # # from datetime import datetime, timedelta  # Module not found  # Module not found  # Module not found
# # # from enum import Enum  # Module not found  # Module not found  # Module not found
# # # from typing import Dict, List, Set, Optional, Any, Callable, Tuple  # Module not found  # Module not found  # Module not found
# # # from dataclasses import dataclass  # Module not found  # Module not found  # Module not found
import logging
import json
# # # from collections import defaultdict, deque  # Module not found  # Module not found  # Module not found
# Optional dependencies (guarded)

# Mandatory Pipeline Contract Annotations
__phase__ = "O"
__code__ = "63O"
__stage_order__ = 7

try:
    import networkx as nx  # type: ignore
    HAS_NETWORKX = True
except Exception:
    HAS_NETWORKX = False
    class _NX:
        class DiGraph:
            def __init__(self): self._adj = {}
            def add_node(self, n): self._adj.setdefault(n, set())
            def add_edge(self, u, v):
                self.add_node(u); self.add_node(v); self._adj[u].add(v)
            def successors(self, n): return list(self._adj.get(n, ()))
            def predecessors(self, n):
                return [u for u, vs in self._adj.items() if n in vs]
            def nodes(self): return list(self._adj.keys())
    nx = _NX()  # type: ignore
try:
# # #     from pydantic import BaseModel, Field  # type: ignore  # Module not found  # Module not found  # Module not found
    HAS_PYDANTIC = True
except Exception:
    HAS_PYDANTIC = False
    class BaseModel:  # type: ignore
        def dict(self, *a, **k): return self.__dict__
    def Field(default=None, **kwargs):  # type: ignore
        return default

logger = logging.getLogger(__name__)


class DependencyType(str, Enum):
    """Tipos de dependencias entre servicios"""
    REQUIRED = "required"        # Dependencia obligatoria
    OPTIONAL = "optional"        # Dependencia opcional
    CONDITIONAL = "conditional"  # Dependencia condicional
    TEMPORAL = "temporal"        # Dependencia temporal (orden de ejecución)
    RESOURCE = "resource"        # Dependencia de recurso compartido
    DATA = "data"               # Dependencia de datos


class ServiceState(str, Enum):
    """Estados de un servicio"""
    UNKNOWN = "unknown"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNAVAILABLE = "unavailable"
    STARTING = "starting"
    STOPPING = "stopping"


class DependencyHealth(str, Enum):
    """Estado de salud de una dependencia"""
    SATISFIED = "satisfied"
    PARTIALLY_SATISFIED = "partially_satisfied"
    UNSATISFIED = "unsatisfied"
    UNKNOWN = "unknown"


class ServiceDependency(BaseModel):
    """Definición de una dependencia entre servicios"""
    source_service: str
    target_service: str
    dependency_type: DependencyType
    required_state: ServiceState = ServiceState.HEALTHY
    timeout_seconds: int = 30
    retry_attempts: int = 3
    conditional_expression: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ServiceInfo(BaseModel):
    """Información de un servicio"""
    name: str
    current_state: ServiceState = ServiceState.UNKNOWN
    health_check_endpoint: Optional[str] = None
    health_check_function: Optional[str] = None  # Nombre de función registrada
    startup_time_estimate: int = 60  # segundos
    dependencies: List[ServiceDependency] = Field(default_factory=list)
    dependents: List[str] = Field(default_factory=list)  # Servicios que dependen de este
    last_health_check: Optional[datetime] = None
    health_check_interval: int = 30  # segundos
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DependencyViolation(BaseModel):
    """Violación de dependencia detectada"""
    source_service: str
    target_service: str
    dependency_type: DependencyType
    violation_type: str
    description: str
    detected_at: datetime = Field(default_factory=datetime.now)
    resolved_at: Optional[datetime] = None
    impact_assessment: str = ""


class DependencyGraph(BaseModel):
    """Grafo de dependencias del sistema"""
    services: Dict[str, ServiceInfo] = Field(default_factory=dict)
    dependency_matrix: Dict[str, Dict[str, DependencyHealth]] = Field(default_factory=dict)
    violations: List[DependencyViolation] = Field(default_factory=list)
    last_analysis: Optional[datetime] = None


class DependencyManager:
    """
    Gestor principal de dependencias que mantiene el grafo de servicios
    y previene problemas de aislamiento y procesos huérfanos
    """
    
    def __init__(self, telemetry_collector=None):
        self.services: Dict[str, ServiceInfo] = {}
        self.dependency_graph = nx.DiGraph()
        self.health_check_functions: Dict[str, Callable] = {}
        self.telemetry = telemetry_collector
        
        # Estado del análisis
        self.violations: List[DependencyViolation] = []
        self.dependency_matrix: Dict[str, Dict[str, DependencyHealth]] = defaultdict(dict)
        
        # Control de monitoreo
        self._monitoring_tasks: Dict[str, asyncio.Task] = {}
        self._analysis_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Configuración
        self.analysis_interval = 60  # segundos
        self.violation_retention_hours = 24
        self.dependency_timeout_default = 30
        
        # Estadísticas
        self.stats = {
            "services_registered": 0,
            "dependencies_defined": 0,
            "health_checks_performed": 0,
            "violations_detected": 0,
            "violations_resolved": 0
        }
    
    def register_service(self, service_info: ServiceInfo):
        """Registra un nuevo servicio en el grafo de dependencias"""
        
        self.services[service_info.name] = service_info
        self.dependency_graph.add_node(service_info.name, **service_info.metadata)
        
        # Agregar dependencias al grafo
        for dep in service_info.dependencies:
            self.add_dependency(
                dep.source_service,
                dep.target_service,
                dep.dependency_type,
                dep
            )
        
        self.stats["services_registered"] += 1
        logger.info(f"Registered service: {service_info.name}")
    
    def add_dependency(
        self,
        source_service: str,
        target_service: str,
        dependency_type: DependencyType,
        dependency_config: ServiceDependency = None
    ):
        """Añade una dependencia entre dos servicios"""
        
        # Asegurar que ambos servicios existen en el grafo
        if source_service not in self.services:
            self.services[source_service] = ServiceInfo(name=source_service)
            self.dependency_graph.add_node(source_service)
        
        if target_service not in self.services:
            self.services[target_service] = ServiceInfo(name=target_service)
            self.dependency_graph.add_node(target_service)
        
        # Crear dependencia si no se proporciona
        if dependency_config is None:
            dependency_config = ServiceDependency(
                source_service=source_service,
                target_service=target_service,
                dependency_type=dependency_type
            )
        
        # Añadir al grafo
        edge_data = {
            'dependency_type': dependency_type.value,
            'config': dependency_config.model_dump()
        }
        self.dependency_graph.add_edge(source_service, target_service, **edge_data)
        
        # Actualizar listas de dependencias
        source_info = self.services[source_service]
        if dependency_config not in source_info.dependencies:
            source_info.dependencies.append(dependency_config)
        
        target_info = self.services[target_service]
        if source_service not in target_info.dependents:
            target_info.dependents.append(source_service)
        
        self.stats["dependencies_defined"] += 1
        logger.info(f"Added dependency: {source_service} -> {target_service} ({dependency_type.value})")
    
    def register_health_check_function(self, service_name: str, function: Callable):
        """Registra función de health check para un servicio"""
        self.health_check_functions[service_name] = function
        
        if service_name in self.services:
            self.services[service_name].health_check_function = service_name
        
        logger.info(f"Registered health check function for {service_name}")
    
    async def start_monitoring(self):
        """Inicia el monitoreo continuo de dependencias"""
        if self._running:
            return
        
        self._running = True
        
        # Iniciar monitoreo individual de servicios
        for service_name in self.services:
            await self._start_service_monitoring(service_name)
        
        # Iniciar análisis global
        self._analysis_task = asyncio.create_task(self._dependency_analysis_loop())
        
        logger.info("Started dependency monitoring")
    
    async def stop_monitoring(self):
        """Detiene el monitoreo de dependencias"""
        self._running = False
        
        # Cancelar tareas de monitoreo
        for task in self._monitoring_tasks.values():
            task.cancel()
        
        if self._monitoring_tasks:
            await asyncio.gather(*self._monitoring_tasks.values(), return_exceptions=True)
        
        self._monitoring_tasks.clear()
        
        # Cancelar análisis global
        if self._analysis_task:
            self._analysis_task.cancel()
            try:
                await self._analysis_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Stopped dependency monitoring")
    
    async def _start_service_monitoring(self, service_name: str):
        """Inicia monitoreo de un servicio específico"""
        if service_name in self._monitoring_tasks:
            return
        
        task = asyncio.create_task(self._monitor_service_loop(service_name))
        self._monitoring_tasks[service_name] = task
    
    async def _monitor_service_loop(self, service_name: str):
        """Loop de monitoreo para un servicio específico"""
        service_info = self.services[service_name]
        
        while self._running:
            try:
                # Realizar health check
                await self._perform_health_check(service_name)
                
                # Esperar intervalo configurado
                await asyncio.sleep(service_info.health_check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error monitoring service {service_name}: {e}")
                await asyncio.sleep(service_info.health_check_interval)
    
    async def _perform_health_check(self, service_name: str) -> ServiceState:
        """Realiza health check de un servicio"""
        
        service_info = self.services[service_name]
        old_state = service_info.current_state
        
        try:
            # Usar función personalizada si está disponible
            if service_name in self.health_check_functions:
                health_function = self.health_check_functions[service_name]
                
                if asyncio.iscoroutinefunction(health_function):
                    is_healthy = await health_function()
                else:
                    is_healthy = health_function()
                
                new_state = ServiceState.HEALTHY if is_healthy else ServiceState.UNHEALTHY
            
            # Usar endpoint HTTP si está configurado
            elif service_info.health_check_endpoint:
                # TODO: Implementar check HTTP
                new_state = ServiceState.UNKNOWN
            
            else:
                # Sin health check configurado
                new_state = ServiceState.UNKNOWN
            
            # Actualizar estado
            service_info.current_state = new_state
            service_info.last_health_check = datetime.now()
            
            # Detectar cambios de estado
            if old_state != new_state:
                logger.info(f"Service {service_name} state changed: {old_state.value} -> {new_state.value}")
                
                if self.telemetry:
                    await self.telemetry.record_metric(
                        "service.state_change",
                        1.0,
                        {
                            "service_name": service_name,
                            "old_state": old_state.value,
                            "new_state": new_state.value
                        }
                    )
            
            self.stats["health_checks_performed"] += 1
            
            return new_state
            
        except Exception as e:
            logger.error(f"Health check failed for {service_name}: {e}")
            service_info.current_state = ServiceState.UNAVAILABLE
            service_info.last_health_check = datetime.now()
            return ServiceState.UNAVAILABLE
    
    async def _dependency_analysis_loop(self):
        """Loop principal de análisis de dependencias"""
        
        while self._running:
            try:
                await self._analyze_dependencies()
                await asyncio.sleep(self.analysis_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in dependency analysis: {e}")
                await asyncio.sleep(self.analysis_interval)
    
    async def _analyze_dependencies(self):
        """Analiza el estado de todas las dependencias"""
        
        # Limpiar violaciones antiguas
        cutoff_time = datetime.now() - timedelta(hours=self.violation_retention_hours)
        self.violations = [v for v in self.violations if v.detected_at >= cutoff_time]
        
        # Analizar cada dependencia
        for service_name, service_info in self.services.items():
            
            for dependency in service_info.dependencies:
                await self._analyze_single_dependency(dependency)
        
        # Detectar ciclos en dependencias
        await self._detect_dependency_cycles()
        
        # Detectar servicios huérfanos
        await self._detect_orphaned_services()
    
    async def _analyze_single_dependency(self, dependency: ServiceDependency):
        """Analiza una dependencia específica"""
        
        source_service = dependency.source_service
        target_service = dependency.target_service
        
        # Verificar que ambos servicios existan
        if target_service not in self.services:
            await self._record_violation(
                dependency,
                "missing_target",
                f"Target service {target_service} not registered"
            )
            return
        
        target_info = self.services[target_service]
        dependency_health = DependencyHealth.UNKNOWN
        
        # Evaluar salud de la dependencia según tipo
        if dependency.dependency_type == DependencyType.REQUIRED:
            if target_info.current_state == dependency.required_state:
                dependency_health = DependencyHealth.SATISFIED
            elif target_info.current_state in [ServiceState.DEGRADED, ServiceState.UNKNOWN]:
                dependency_health = DependencyHealth.PARTIALLY_SATISFIED
            else:
                dependency_health = DependencyHealth.UNSATISFIED
                await self._record_violation(
                    dependency,
                    "required_dependency_unhealthy",
                    f"Required service {target_service} is {target_info.current_state.value}"
                )
        
        elif dependency.dependency_type == DependencyType.OPTIONAL:
            if target_info.current_state in [ServiceState.HEALTHY, ServiceState.DEGRADED]:
                dependency_health = DependencyHealth.SATISFIED
            else:
                dependency_health = DependencyHealth.PARTIALLY_SATISFIED
        
        elif dependency.dependency_type == DependencyType.CONDITIONAL:
            # Evaluar expresión condicional
            should_be_available = await self._evaluate_conditional_dependency(dependency)
            
            if should_be_available:
                if target_info.current_state == dependency.required_state:
                    dependency_health = DependencyHealth.SATISFIED
                else:
                    dependency_health = DependencyHealth.UNSATISFIED
                    await self._record_violation(
                        dependency,
                        "conditional_dependency_not_met",
                        f"Conditional dependency not satisfied: {dependency.conditional_expression}"
                    )
            else:
                dependency_health = DependencyHealth.SATISFIED  # No se requiere
        
        # Actualizar matriz de dependencias
        self.dependency_matrix[source_service][target_service] = dependency_health
    
    async def _evaluate_conditional_dependency(self, dependency: ServiceDependency) -> bool:
        """Evalúa expresión condicional de dependencia"""

        global ConditionEvaluationError
        if not dependency.conditional_expression:
            return True
        
        try:
            # Contexto para evaluación con información del servicio
            context = {
                'source_service': dependency.source_service,
                'target_service': dependency.target_service,
                'services': {name: info.current_state.value for name, info in self.services.items()}
            }
            
            # Evaluación usando AST seguro
# # #             from compensation_engine import SafeExpressionEvaluator, ConditionEvaluationError  # Module not found  # Module not found  # Module not found
            
            result = SafeExpressionEvaluator.evaluate(dependency.conditional_expression, context)
            return bool(result)
            
        except ConditionEvaluationError as e:
            logger.warning(f"Failed to evaluate conditional expression: {e}")
            return True  # Por defecto, asumir que se requiere
        except Exception as e:
            logger.error(f"Unexpected error evaluating dependency condition: {e}")
            return True
    
    async def _detect_dependency_cycles(self):
        """Detecta ciclos en el grafo de dependencias"""
        
        try:
            cycles = list(nx.simple_cycles(self.dependency_graph))
            
            for cycle in cycles:
                cycle_services = " -> ".join(cycle + [cycle[0]])
                
                # Crear violación para el ciclo
                violation = DependencyViolation(
                    source_service=cycle[0],
                    target_service=cycle[-1],
                    dependency_type=DependencyType.REQUIRED,
                    violation_type="dependency_cycle",
                    description=f"Dependency cycle detected: {cycle_services}",
                    impact_assessment="May cause deadlocks and startup issues"
                )
                
                # Verificar si ya se registró esta violación
                if not self._violation_exists(violation):
                    self.violations.append(violation)
                    self.stats["violations_detected"] += 1
                    
                    logger.warning(f"Dependency cycle detected: {cycle_services}")
                    
                    if self.telemetry:
                        await self.telemetry.record_metric(
                            "dependency.cycle_detected",
                            1.0,
                            {"cycle_length": len(cycle)}
                        )
        
        except Exception as e:
            logger.error(f"Error detecting dependency cycles: {e}")
    
    async def _detect_orphaned_services(self):
        """Detecta servicios huérfanos (sin dependencias ni dependientes)"""
        
        for service_name, service_info in self.services.items():
            
            # Verificar si el servicio no tiene dependencias ni dependientes
            has_dependencies = len(service_info.dependencies) > 0
            has_dependents = len(service_info.dependents) > 0
            
            if not has_dependencies and not has_dependents:
                violation = DependencyViolation(
                    source_service=service_name,
                    target_service=service_name,
                    dependency_type=DependencyType.REQUIRED,
                    violation_type="orphaned_service",
                    description=f"Service {service_name} has no dependencies or dependents",
                    impact_assessment="Isolated service may indicate configuration issue"
                )
                
                if not self._violation_exists(violation):
                    self.violations.append(violation)
                    logger.info(f"Orphaned service detected: {service_name}")
    
    async def _record_violation(self, dependency: ServiceDependency, violation_type: str, description: str):
        """Registra una violación de dependencia"""
        
        violation = DependencyViolation(
            source_service=dependency.source_service,
            target_service=dependency.target_service,
            dependency_type=dependency.dependency_type,
            violation_type=violation_type,
            description=description
        )
        
        if not self._violation_exists(violation):
            self.violations.append(violation)
            self.stats["violations_detected"] += 1
            
            logger.warning(f"Dependency violation: {description}")
            
            if self.telemetry:
                await self.telemetry.record_metric(
                    "dependency.violation_detected",
                    1.0,
                    {
                        "source_service": dependency.source_service,
                        "target_service": dependency.target_service,
                        "violation_type": violation_type
                    }
                )
    
    def _violation_exists(self, new_violation: DependencyViolation) -> bool:
        """Verifica si ya existe una violación similar"""
        
        for existing_violation in self.violations:
            if (existing_violation.source_service == new_violation.source_service and
                existing_violation.target_service == new_violation.target_service and
                existing_violation.violation_type == new_violation.violation_type and
                existing_violation.resolved_at is None):
                return True
        
        return False
    
    def get_service_dependencies(self, service_name: str) -> List[str]:
        """Obtiene lista de dependencias directas de un servicio"""
        
        if service_name not in self.services:
            return []
        
        return [dep.target_service for dep in self.services[service_name].dependencies]
    
    def get_service_dependents(self, service_name: str) -> List[str]:
        """Obtiene lista de servicios que dependen de este servicio"""
        
        if service_name not in self.services:
            return []
        
        return self.services[service_name].dependents.copy()
    
    def get_dependency_path(self, source: str, target: str) -> List[str]:
        """Encuentra camino de dependencias entre dos servicios"""
        
        try:
            if nx.has_path(self.dependency_graph, source, target):
                path = nx.shortest_path(self.dependency_graph, source, target)
                return path
        except nx.NetworkXNoPath:
            pass
        
        return []
    
    def get_startup_order(self) -> List[List[str]]:
        """
        Determina orden óptimo de arranque de servicios respetando dependencias
        
        Returns:
            Lista de listas, donde cada sublista contiene servicios que pueden
            iniciarse en paralelo en ese nivel
        """
        
        try:
            # Usar algoritmo topológico para determinar orden
            topological_order = list(nx.topological_sort(self.dependency_graph))
            
            # Agrupar por niveles (servicios que pueden iniciarse en paralelo)
            levels = []
            remaining_services = set(topological_order)
            
            while remaining_services:
                current_level = []
                
                for service in list(remaining_services):
                    # Verificar si todas las dependencias están satisfechas
                    dependencies = self.get_service_dependencies(service)
                    if all(dep not in remaining_services for dep in dependencies):
                        current_level.append(service)
                
                # Remover servicios del nivel actual
                for service in current_level:
                    remaining_services.remove(service)
                
                if current_level:
                    levels.append(current_level)
                else:
                    # Evitar bucle infinito en caso de problemas
                    break
            
            return levels
            
        except nx.NetworkXError:
            logger.error("Cannot determine startup order due to dependency cycles")
            return [[service] for service in self.services.keys()]
    
    def get_dependency_summary(self) -> Dict[str, Any]:
        """Obtiene resumen del estado de dependencias"""
        
        total_dependencies = sum(len(info.dependencies) for info in self.services.values())
        
        # Contar violaciones por tipo
        violation_types = {}
        active_violations = [v for v in self.violations if v.resolved_at is None]
        
        for violation in active_violations:
            violation_types[violation.violation_type] = violation_types.get(violation.violation_type, 0) + 1
        
        # Contar servicios por estado
        service_states = {}
        for service_info in self.services.values():
            state = service_info.current_state.value
            service_states[state] = service_states.get(state, 0) + 1
        
        # Calcular métricas de salud
        healthy_services = service_states.get("healthy", 0)
        total_services = len(self.services)
        health_percentage = (healthy_services / total_services * 100) if total_services > 0 else 0
        
        return {
            "total_services": total_services,
            "total_dependencies": total_dependencies,
            "service_states": service_states,
            "active_violations": len(active_violations),
            "violation_types": violation_types,
            "health_percentage": health_percentage,
            "monitoring_active": self._running,
            "statistics": self.stats
        }
    
    def get_service_impact_analysis(self, service_name: str) -> Dict[str, Any]:
        """Analiza el impacto de la falla de un servicio específico"""
        
        if service_name not in self.services:
            return {"error": f"Service {service_name} not found"}
        
        # Encontrar servicios afectados directa e indirectamente
        directly_affected = self.get_service_dependents(service_name)
        
        # Usar BFS para encontrar todos los servicios afectados
        all_affected = set()
        queue = deque([service_name])
        visited = set()
        
        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            
            visited.add(current)
            dependents = self.get_service_dependents(current)
            
            for dependent in dependents:
                if dependent not in visited:
                    all_affected.add(dependent)
                    queue.append(dependent)
        
        # Clasificar impacto por tipo de dependencia
        critical_impact = []
        optional_impact = []
        
        for affected_service in all_affected:
            service_info = self.services[affected_service]
            
            # Buscar tipo de dependencia con el servicio original
            path = self.get_dependency_path(affected_service, service_name)
            if len(path) > 1:
                # Encontrar dependencia en el camino
                for dep in service_info.dependencies:
                    if dep.target_service in path:
                        if dep.dependency_type == DependencyType.REQUIRED:
                            critical_impact.append(affected_service)
                        else:
                            optional_impact.append(affected_service)
                        break
        
        return {
            "service_name": service_name,
            "directly_affected": directly_affected,
            "all_affected": list(all_affected),
            "critical_impact": critical_impact,
            "optional_impact": optional_impact,
            "impact_score": len(critical_impact) * 2 + len(optional_impact),
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    async def simulate_service_failure(self, service_name: str) -> Dict[str, Any]:
        """Simula la falla de un servicio para análisis de impacto"""
        
        if service_name not in self.services:
            return {"error": f"Service {service_name} not found"}
        
        # Guardar estado original
        original_state = self.services[service_name].current_state
        
        # Simular falla
        self.services[service_name].current_state = ServiceState.UNAVAILABLE
        
        # Ejecutar análisis
        await self._analyze_dependencies()
        
        # Obtener violaciones generadas por la simulación
        simulation_violations = [
            v for v in self.violations
            if (v.target_service == service_name and 
                v.detected_at >= datetime.now() - timedelta(seconds=10))
        ]
        
        # Restaurar estado original
        self.services[service_name].current_state = original_state
        
        # Generar reporte
        impact_analysis = self.get_service_impact_analysis(service_name)
        
        return {
            **impact_analysis,
            "simulation_violations": len(simulation_violations),
            "affected_dependencies": [v.source_service for v in simulation_violations],
            "recovery_recommendations": self._generate_recovery_recommendations(service_name)
        }
    
    def _generate_recovery_recommendations(self, failed_service: str) -> List[str]:
        """Genera recomendaciones de recuperación para un servicio fallido"""
        
        recommendations = []
        
        # Recomendaciones basadas en dependientes
        dependents = self.get_service_dependents(failed_service)
        
        if dependents:
            recommendations.append(f"Monitor and potentially restart dependent services: {', '.join(dependents)}")
        
        # Recomendaciones basadas en tipo de servicio
        service_info = self.services.get(failed_service)
        if service_info:
            if service_info.startup_time_estimate > 120:
                recommendations.append("Consider implementing health check with longer timeout")
            
            if len(service_info.dependencies) > 5:
                recommendations.append("Review service dependencies - high coupling detected")
        
        # Recomendación general
        recommendations.append("Implement graceful degradation in dependent services")
        recommendations.append("Consider circuit breaker pattern for this service")
        
        return recommendations
    
    async def resolve_violation(self, violation_id: str):
        """Marca una violación como resuelta"""
        
        for violation in self.violations:
            if violation.violation_type == violation_id and violation.resolved_at is None:
                violation.resolved_at = datetime.now()
                self.stats["violations_resolved"] += 1
                
                logger.info(f"Resolved violation: {violation.description}")
                
                if self.telemetry:
                    await self.telemetry.record_metric(
                        "dependency.violation_resolved",
                        1.0,
                        {"violation_type": violation.violation_type}
                    )
                break
if __name__ == "__main__":
    # Minimal demo to ensure execution even without networkx/pydantic
    dm = DependencyManager()
    dm.register_service(ServiceInfo(name="api", version="1.0", state=ServiceState.RUNNING))
    dm.register_service(ServiceInfo(name="db", version="1.0", state=ServiceState.RUNNING))
    dm.add_dependency("api", "db", DependencyType.SYNCHRONOUS)
    summary = dm.get_dependency_summary()
    print(json.dumps({
        "services": list(summary.get("services", {}).keys()) if isinstance(summary, dict) else [],
        "edges": summary.get("edges", []) if isinstance(summary, dict) else []
    }))

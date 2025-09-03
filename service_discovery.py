"""
Service discovery and coordination system using etcd/Consul
"""

import asyncio
import json
import logging
# # # from dataclasses import asdict, dataclass  # Module not found  # Module not found  # Module not found
# # # from datetime import datetime, timedelta  # Module not found  # Module not found  # Module not found
# # # from typing import Any, Callable, Dict, List, Optional, Set  # Module not found  # Module not found  # Module not found

import consul
import etcd3
# # # from kubernetes import client, config, watch  # Module not found  # Module not found  # Module not found

logger = logging.getLogger(__name__)


@dataclass
class ServiceInstance:
    """Represents a service instance"""

    id: str
    name: str
    address: str
    port: int
    health_status: str = "unknown"
    metadata: Dict[str, Any] = None
    tags: List[str] = None
    last_heartbeat: datetime = None
    version: str = "1.0.0"

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.tags is None:
            self.tags = []
        if self.last_heartbeat is None:
            self.last_heartbeat = datetime.utcnow()


class ServiceRegistry:
    """Base service registry interface"""

    async def register_service(self, service: ServiceInstance) -> bool:
        raise NotImplementedError

    async def deregister_service(self, service_id: str) -> bool:
        raise NotImplementedError

    async def get_service(self, service_name: str) -> List[ServiceInstance]:
        raise NotImplementedError

    async def get_all_services(self) -> Dict[str, List[ServiceInstance]]:
        raise NotImplementedError

    async def watch_service_changes(self, callback: Callable):
        raise NotImplementedError


class EtcdServiceRegistry(ServiceRegistry):
    """Service registry implementation using etcd"""

    def __init__(self, host: str = "localhost", port: int = 2379):
        try:
            self.client = etcd3.client(host=host, port=port)
            self.watch_callbacks: List[Callable] = []
            self.running = False
            logger.info(f"Connected to etcd at {host}:{port}")
        except Exception as e:
            logger.error(f"Failed to connect to etcd: {e}")
            self.client = None

    async def register_service(self, service: ServiceInstance) -> bool:
        """Register a service instance"""
        if not self.client:
            return False

        try:
            key = f"/services/{service.name}/{service.id}"
            value = json.dumps(asdict(service), default=str)

            # Set TTL for automatic cleanup
            lease = self.client.lease(ttl=30)
            self.client.put(key, value, lease=lease)

            logger.info(f"Registered service {service.name}:{service.id}")
            return True

        except Exception as e:
            logger.error(f"Failed to register service {service.name}: {e}")
            return False

    async def deregister_service(self, service_id: str) -> bool:
        """Deregister a service instance"""
        if not self.client:
            return False

        try:
            # Find and delete the service
            for value, metadata in self.client.get_prefix("/services/"):
                data = json.loads(value)
                if data["id"] == service_id:
                    self.client.delete(metadata.key)
                    logger.info(f"Deregistered service {service_id}")
                    return True

            return False

        except Exception as e:
            logger.error(f"Failed to deregister service {service_id}: {e}")
            return False

    async def get_service(self, service_name: str) -> List[ServiceInstance]:
        """Get all instances of a service"""
        if not self.client:
            return []

        instances = []
        try:
            prefix = f"/services/{service_name}/"
            for value, _ in self.client.get_prefix(prefix):
                data = json.loads(value)
                # Convert datetime strings back to datetime objects
                if "last_heartbeat" in data:
                    data["last_heartbeat"] = datetime.fromisoformat(
                        data["last_heartbeat"]
                    )
                instances.append(ServiceInstance(**data))

        except Exception as e:
            logger.error(f"Failed to get service {service_name}: {e}")

        return instances

    async def get_all_services(self) -> Dict[str, List[ServiceInstance]]:
        """Get all registered services"""
        if not self.client:
            return {}

        services = {}
        try:
            for value, metadata in self.client.get_prefix("/services/"):
                data = json.loads(value)
                service_name = data["name"]

                if service_name not in services:
                    services[service_name] = []

                # Convert datetime strings back
                if "last_heartbeat" in data:
                    data["last_heartbeat"] = datetime.fromisoformat(
                        data["last_heartbeat"]
                    )

                services[service_name].append(ServiceInstance(**data))

        except Exception as e:
            logger.error(f"Failed to get all services: {e}")

        return services

    async def watch_service_changes(self, callback: Callable):
        """Watch for service registry changes"""
        if not self.client:
            return

        self.watch_callbacks.append(callback)

        if not self.running:
            self.running = True
            asyncio.create_task(self._watch_loop())

    async def _watch_loop(self):
        """Background task to watch for etcd changes"""
        try:
            events_iterator, cancel = self.client.watch_prefix("/services/")

            for event in events_iterator:
                try:
                    if event.value:
                        data = json.loads(event.value)
                        service = ServiceInstance(**data)

                        for callback in self.watch_callbacks:
                            try:
                                await callback(event.key.decode(), service, event.type)
                            except Exception as e:
                                logger.error(f"Error in watch callback: {e}")

                except Exception as e:
                    logger.error(f"Error processing watch event: {e}")

        except Exception as e:
            logger.error(f"Error in etcd watch loop: {e}")
        finally:
            self.running = False


class ConsulServiceRegistry(ServiceRegistry):
    """Service registry implementation using Consul"""

    def __init__(self, host: str = "localhost", port: int = 8500):
        try:
            self.client = consul.Consul(host=host, port=port)
            self.watch_callbacks: List[Callable] = []
            logger.info(f"Connected to Consul at {host}:{port}")
        except Exception as e:
            logger.error(f"Failed to connect to Consul: {e}")
            self.client = None

    async def register_service(self, service: ServiceInstance) -> bool:
        """Register a service instance with Consul"""
        if not self.client:
            return False

        try:
            self.client.agent.service.register(
                name=service.name,
                service_id=service.id,
                address=service.address,
                port=service.port,
                tags=service.tags,
                meta=service.metadata,
                check=consul.Check.http(
                    f"http://{service.address}:{service.port}/health", interval="10s"
                ),
            )

            logger.info(f"Registered service {service.name}:{service.id} with Consul")
            return True

        except Exception as e:
            logger.error(f"Failed to register service with Consul: {e}")
            return False

    async def deregister_service(self, service_id: str) -> bool:
# # #         """Deregister a service from Consul"""  # Module not found  # Module not found  # Module not found
        if not self.client:
            return False

        try:
            self.client.agent.service.deregister(service_id)
# # #             logger.info(f"Deregistered service {service_id} from Consul")  # Module not found  # Module not found  # Module not found
            return True
        except Exception as e:
# # #             logger.error(f"Failed to deregister service from Consul: {e}")  # Module not found  # Module not found  # Module not found
            return False

    async def get_service(self, service_name: str) -> List[ServiceInstance]:
# # #         """Get service instances from Consul"""  # Module not found  # Module not found  # Module not found
        if not self.client:
            return []

        instances = []
        try:
            _, services = self.client.health.service(service_name, passing=True)

            for service in services:
                service_data = service["Service"]
                instances.append(
                    ServiceInstance(
                        id=service_data["ID"],
                        name=service_data["Service"],
                        address=service_data["Address"],
                        port=service_data["Port"],
                        tags=service_data.get("Tags", []),
                        metadata=service_data.get("Meta", {}),
                        health_status="healthy" if service["Checks"] else "unknown",
                    )
                )

        except Exception as e:
# # #             logger.error(f"Failed to get service from Consul: {e}")  # Module not found  # Module not found  # Module not found

        return instances

    async def get_all_services(self) -> Dict[str, List[ServiceInstance]]:
# # #         """Get all services from Consul"""  # Module not found  # Module not found  # Module not found
        if not self.client:
            return {}

        services = {}
        try:
            _, service_list = self.client.agent.services()

            for service_id, service_data in service_list.items():
                service_name = service_data["Service"]

                if service_name not in services:
                    services[service_name] = []

                services[service_name].append(
                    ServiceInstance(
                        id=service_data["ID"],
                        name=service_data["Service"],
                        address=service_data["Address"],
                        port=service_data["Port"],
                        tags=service_data.get("Tags", []),
                        metadata=service_data.get("Meta", {}),
                    )
                )

        except Exception as e:
# # #             logger.error(f"Failed to get all services from Consul: {e}")  # Module not found  # Module not found  # Module not found

        return services

    async def watch_service_changes(self, callback: Callable):
        """Watch for service changes in Consul"""
        self.watch_callbacks.append(callback)
        # Consul watching would be implemented here
        pass


class KubernetesServiceDiscovery:
    """Service discovery for Kubernetes environments"""

    def __init__(self, namespace: str = "default"):
        self.namespace = namespace

        try:
            # Try in-cluster config first, then local config
            try:
                config.load_incluster_config()
            except:
                config.load_kube_config()

            self.v1 = client.CoreV1Api()
            logger.info("Connected to Kubernetes cluster")
        except Exception as e:
            logger.error(f"Failed to connect to Kubernetes: {e}")
            self.v1 = None

    async def get_kubernetes_services(self) -> Dict[str, List[ServiceInstance]]:
# # #         """Get services from Kubernetes"""  # Module not found  # Module not found  # Module not found
        if not self.v1:
            return {}

        services = {}
        try:
            service_list = self.v1.list_namespaced_service(namespace=self.namespace)

            for service in service_list.items:
                service_name = service.metadata.name

                if service_name not in services:
                    services[service_name] = []

                # Get service endpoints
                endpoints = self.v1.list_namespaced_endpoints(namespace=self.namespace)

                for endpoint in endpoints.items:
                    if endpoint.metadata.name == service_name and endpoint.subsets:
                        for subset in endpoint.subsets:
                            if subset.addresses:
                                for address in subset.addresses:
                                    if subset.ports:
                                        for port in subset.ports:
                                            services[service_name].append(
                                                ServiceInstance(
                                                    id=f"{service_name}-{address.ip}-{port.port}",
                                                    name=service_name,
                                                    address=address.ip,
                                                    port=port.port,
                                                    metadata={
                                                        "namespace": self.namespace,
                                                        "labels": service.metadata.labels
                                                        or {},
                                                    },
                                                )
                                            )

        except Exception as e:
            logger.error(f"Failed to get Kubernetes services: {e}")

        return services

    async def watch_kubernetes_changes(self, callback: Callable):
        """Watch for Kubernetes service changes"""
        if not self.v1:
            return

        try:
            w = watch.Watch()
            for event in w.stream(
                self.v1.list_namespaced_service, namespace=self.namespace
            ):
                try:
                    await callback(event["type"], event["object"])
                except Exception as e:
                    logger.error(f"Error in Kubernetes watch callback: {e}")

        except Exception as e:
            logger.error(f"Error watching Kubernetes changes: {e}")


class ServiceDiscoveryManager:
    """
    Unified service discovery manager that integrates multiple discovery mechanisms
    """

    def __init__(
        self,
        discovery_backend: str = "etcd",
        etcd_config: Dict[str, Any] = None,
        consul_config: Dict[str, Any] = None,
        kubernetes_config: Dict[str, Any] = None,
    ):
        self.discovery_backend = discovery_backend
        self.registries: Dict[str, ServiceRegistry] = {}
        self.service_cache: Dict[str, List[ServiceInstance]] = {}
        self.load_balancers: Dict[str, Callable] = {
            "round_robin": self._round_robin_lb,
            "random": self._random_lb,
            "least_connections": self._least_connections_lb,
        }
        self.service_counters: Dict[str, int] = {}

        # Initialize registries
        if discovery_backend == "etcd":
            etcd_config = etcd_config or {}
            self.registries["etcd"] = EtcdServiceRegistry(
                host=etcd_config.get("host", "localhost"),
                port=etcd_config.get("port", 2379),
            )

        if discovery_backend == "consul":
            consul_config = consul_config or {}
            self.registries["consul"] = ConsulServiceRegistry(
                host=consul_config.get("host", "localhost"),
                port=consul_config.get("port", 8500),
            )

        # Always add Kubernetes discovery if available
        kubernetes_config = kubernetes_config or {}
        try:
            self.registries["kubernetes"] = KubernetesServiceDiscovery(
                namespace=kubernetes_config.get("namespace", "default")
            )
        except Exception as e:
            logger.warning(f"Kubernetes discovery not available: {e}")

        # Start background tasks
        asyncio.create_task(self._cache_refresh_loop())

    async def register_service(self, service: ServiceInstance) -> bool:
        """Register a service with the primary registry"""
        primary_registry = self.registries.get(self.discovery_backend)

        if primary_registry:
            success = await primary_registry.register_service(service)
            if success:
                # Invalidate cache
                self.service_cache.pop(service.name, None)
            return success

        return False

    async def deregister_service(self, service_id: str) -> bool:
# # #         """Deregister a service from the primary registry"""  # Module not found  # Module not found  # Module not found
        primary_registry = self.registries.get(self.discovery_backend)

        if primary_registry:
            success = await primary_registry.deregister_service(service_id)
            if success:
                # Clear entire cache since we don't know which service was removed
                self.service_cache.clear()
            return success

        return False

    async def get_service_instances(
        self, service_name: str, prefer_healthy: bool = True
    ) -> List[ServiceInstance]:
        """Get all instances of a service"""
        # Check cache first
        if service_name in self.service_cache:
            instances = self.service_cache[service_name]
        else:
            instances = []

# # #             # Aggregate from all registries  # Module not found  # Module not found  # Module not found
            for registry_name, registry in self.registries.items():
                try:
                    registry_instances = await registry.get_service(service_name)
                    instances.extend(registry_instances)
                except Exception as e:
# # #                     logger.warning(f"Failed to get service from {registry_name}: {e}")  # Module not found  # Module not found  # Module not found

            # Cache results
            self.service_cache[service_name] = instances

        # Filter by health status if requested
        if prefer_healthy:
            instances = [
                i for i in instances if i.health_status in ["healthy", "unknown"]
            ]

        return instances

    async def get_service_instance(
        self, service_name: str, load_balancer: str = "round_robin"
    ) -> Optional[ServiceInstance]:
        """Get a single service instance using load balancing"""
        instances = await self.get_service_instances(service_name)

        if not instances:
            return None

        lb_func = self.load_balancers.get(load_balancer, self._round_robin_lb)
        return lb_func(service_name, instances)

    async def get_available_services(self) -> Dict[str, List[ServiceInstance]]:
# # #         """Get all available services from all registries"""  # Module not found  # Module not found  # Module not found
        all_services = {}

        for registry_name, registry in self.registries.items():
            try:
                services = await registry.get_all_services()

                # Merge services
                for service_name, instances in services.items():
                    if service_name not in all_services:
                        all_services[service_name] = []
                    all_services[service_name].extend(instances)

            except Exception as e:
# # #                 logger.warning(f"Failed to get services from {registry_name}: {e}")  # Module not found  # Module not found  # Module not found

        return all_services

    async def health_check_service(self, service: ServiceInstance) -> bool:
        """Perform health check on a service instance"""
        try:
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"http://{service.address}:{service.port}/health", timeout=5.0
                )
                return response.status_code == 200

        except Exception as e:
            logger.debug(f"Health check failed for {service.name}: {e}")
            return False

    def _round_robin_lb(
        self, service_name: str, instances: List[ServiceInstance]
    ) -> ServiceInstance:
        """Round robin load balancing"""
        if service_name not in self.service_counters:
            self.service_counters[service_name] = 0

        index = self.service_counters[service_name] % len(instances)
        self.service_counters[service_name] += 1

        return instances[index]

    def _random_lb(
        self, service_name: str, instances: List[ServiceInstance]
    ) -> ServiceInstance:
        """Random load balancing"""
        import random

        return random.choice(instances)

    def _least_connections_lb(
        self, service_name: str, instances: List[ServiceInstance]
    ) -> ServiceInstance:
        """Least connections load balancing (simplified)"""
        # In a real implementation, this would track active connections
        # For now, just return the first instance
        return instances[0]

    async def _cache_refresh_loop(self):
        """Background task to refresh service cache"""
        while True:
            try:
                await asyncio.sleep(30)  # Refresh every 30 seconds

                # Clear cache to force refresh
                self.service_cache.clear()

                logger.debug("Service cache refreshed")

            except Exception as e:
                logger.error(f"Error in cache refresh loop: {e}")
                await asyncio.sleep(30)

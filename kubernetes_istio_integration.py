"""
Kubernetes CRDs and Istio Service Mesh Integration for Advanced Orchestration
File: kubernetes_integration.py
Status: NEW FILE
Impact: Adds Kubernetes CRDs for workflow management, Istio service mesh integration, and advanced traffic management
"""

import asyncio
import base64
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import istio_client
import yaml
from kubernetes import client, config, watch
from kubernetes.client.rest import ApiException
from prometheus_client import Counter, Gauge, Histogram

logger = logging.getLogger(__name__)


# Kubernetes Custom Resource Definitions for Workflows
WORKFLOW_CRD = {
    "apiVersion": "apiextensions.k8s.io/v1",
    "kind": "CustomResourceDefinition",
    "metadata": {"name": "workflows.orchestrator.io"},
    "spec": {
        "group": "orchestrator.io",
        "versions": [
            {
                "name": "v1",
                "served": True,
                "storage": True,
                "schema": {
                    "openAPIV3Schema": {
                        "type": "object",
                        "properties": {
                            "spec": {
                                "type": "object",
                                "properties": {
                                    "workflowId": {"type": "string"},
                                    "version": {"type": "string"},
                                    "steps": {"type": "array"},
                                    "resources": {"type": "object"},
                                    "scheduling": {"type": "object"},
                                    "monitoring": {"type": "object"},
                                },
                            },
                            "status": {
                                "type": "object",
                                "properties": {
                                    "phase": {"type": "string"},
                                    "conditions": {"type": "array"},
                                    "startTime": {"type": "string"},
                                    "completionTime": {"type": "string"},
                                },
                            },
                        },
                    }
                },
            }
        ],
        "scope": "Namespaced",
        "names": {
            "plural": "workflows",
            "singular": "workflow",
            "kind": "Workflow",
            "shortNames": ["wf"],
        },
    },
}


class KubernetesWorkflowController:
    """
    Kubernetes controller for managing workflows as custom resources
    """

    def __init__(self, namespace: str = "default"):
        self.namespace = namespace
        self._init_kubernetes_client()
        self._init_istio_client()
        self.workflows: Dict[str, Any] = {}
        self.running = False

        # Metrics
        self.workflow_counter = Counter(
            "k8s_workflows_total", "Total Kubernetes workflows", ["namespace", "status"]
        )
        self.pod_gauge = Gauge(
            "k8s_workflow_pods", "Current workflow pods", ["namespace", "workflow"]
        )

    def _init_kubernetes_client(self):
        """Initialize Kubernetes client"""
        try:
            # Try in-cluster config first
            try:
                config.load_incluster_config()
            except:
                config.load_kube_config()

            self.core_v1 = client.CoreV1Api()
            self.apps_v1 = client.AppsV1Api()
            self.batch_v1 = client.BatchV1Api()
            self.custom_api = client.CustomObjectsApi()
            self.extensions_v1 = client.ApiextensionsV1Api()

            logger.info("Kubernetes client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Kubernetes client: {e}")
            raise

    def _init_istio_client(self):
        """Initialize Istio client for service mesh management"""
        try:
            self.istio_client = istio_client.CustomObjectsApi()
            self.istio_networking = istio_client.NetworkingV1beta1Api()
            logger.info("Istio client initialized")
        except Exception as e:
            logger.warning(f"Could not initialize Istio client: {e}")
            self.istio_client = None

    async def install_crds(self):
        """Install workflow CRDs in Kubernetes"""
        try:
            # Check if CRD already exists
            existing_crds = self.extensions_v1.list_custom_resource_definition()
            crd_names = [crd.metadata.name for crd in existing_crds.items]

            if WORKFLOW_CRD["metadata"]["name"] not in crd_names:
                # Create the CRD
                self.extensions_v1.create_custom_resource_definition(body=WORKFLOW_CRD)
                logger.info("Workflow CRD installed successfully")
            else:
                logger.info("Workflow CRD already exists")

            return True

        except ApiException as e:
            logger.error(f"Failed to install CRDs: {e}")
            return False

    async def create_workflow_resource(self, workflow_def: Dict[str, Any]) -> str:
        """
        Create a workflow as a Kubernetes custom resource

        Args:
            workflow_def: Workflow definition

        Returns:
            Resource name
        """
        resource_name = (
            f"workflow-{workflow_def['id']}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        )

        workflow_resource = {
            "apiVersion": "orchestrator.io/v1",
            "kind": "Workflow",
            "metadata": {
                "name": resource_name,
                "namespace": self.namespace,
                "labels": {
                    "app": "hyper-orchestrator",
                    "workflow-id": workflow_def["id"],
                    "version": workflow_def.get("version", "1.0.0"),
                },
                "annotations": {
                    "orchestrator.io/description": workflow_def.get("description", ""),
                    "orchestrator.io/owner": workflow_def.get("owner", "system"),
                },
            },
            "spec": {
                "workflowId": workflow_def["id"],
                "version": workflow_def.get("version", "1.0.0"),
                "steps": workflow_def.get("steps", []),
                "resources": {
                    "requests": {
                        "memory": workflow_def.get("memory_request", "256Mi"),
                        "cpu": workflow_def.get("cpu_request", "100m"),
                    },
                    "limits": {
                        "memory": workflow_def.get("memory_limit", "1Gi"),
                        "cpu": workflow_def.get("cpu_limit", "500m"),
                    },
                },
                "scheduling": {
                    "nodeSelector": workflow_def.get("node_selector", {}),
                    "affinity": workflow_def.get("affinity", {}),
                    "tolerations": workflow_def.get("tolerations", []),
                },
                "monitoring": {
                    "enabled": True,
                    "metrics_port": 9090,
                    "tracing_enabled": True,
                },
            },
            "status": {
                "phase": "Pending",
                "conditions": [],
                "startTime": datetime.utcnow().isoformat(),
            },
        }

        try:
            # Create the custom resource
            response = self.custom_api.create_namespaced_custom_object(
                group="orchestrator.io",
                version="v1",
                namespace=self.namespace,
                plural="workflows",
                body=workflow_resource,
            )

            self.workflows[resource_name] = workflow_resource
            self.workflow_counter.labels(
                namespace=self.namespace, status="created"
            ).inc()

            logger.info(f"Created workflow resource: {resource_name}")
            return resource_name

        except ApiException as e:
            logger.error(f"Failed to create workflow resource: {e}")
            raise

    async def deploy_workflow_pods(
        self, workflow_name: str, steps: List[Dict[str, Any]]
    ):
        """Deploy pods for workflow execution"""
        deployed_pods = []

        for step in steps:
            pod_name = f"{workflow_name}-{step['id']}"

            # Create pod specification
            pod_spec = client.V1PodSpec(
                containers=[
                    client.V1Container(
                        name=step["id"],
                        image=step.get("image", "python:3.9-slim"),
                        command=step.get("command", []),
                        args=step.get("args", []),
                        env=[
                            client.V1EnvVar(name=k, value=v)
                            for k, v in step.get("env", {}).items()
                        ],
                        resources=client.V1ResourceRequirements(
                            requests={
                                "memory": step.get("memory_request", "128Mi"),
                                "cpu": step.get("cpu_request", "50m"),
                            },
                            limits={
                                "memory": step.get("memory_limit", "512Mi"),
                                "cpu": step.get("cpu_limit", "200m"),
                            },
                        ),
                        volume_mounts=self._create_volume_mounts(step),
                    )
                ],
                volumes=self._create_volumes(step),
                restart_policy="Never",
                service_account_name=step.get("service_account", "default"),
                init_containers=self._create_init_containers(step),
            )

            # Add Istio sidecar injection annotation
            pod = client.V1Pod(
                metadata=client.V1ObjectMeta(
                    name=pod_name,
                    namespace=self.namespace,
                    labels={
                        "app": "hyper-orchestrator",
                        "workflow": workflow_name,
                        "step": step["id"],
                    },
                    annotations={
                        "sidecar.istio.io/inject": "true",
                        "prometheus.io/scrape": "true",
                        "prometheus.io/port": "9090",
                    },
                ),
                spec=pod_spec,
            )

            try:
                # Create the pod
                response = self.core_v1.create_namespaced_pod(
                    namespace=self.namespace, body=pod
                )
                deployed_pods.append(pod_name)
                self.pod_gauge.labels(
                    namespace=self.namespace, workflow=workflow_name
                ).inc()

                logger.info(f"Deployed pod: {pod_name}")

            except ApiException as e:
                logger.error(f"Failed to deploy pod {pod_name}: {e}")
                # Rollback deployed pods on failure
                await self._rollback_pods(deployed_pods)
                raise

        return deployed_pods

    def _create_init_containers(self, step: Dict[str, Any]) -> List[client.V1Container]:
        """Create init containers for setup tasks"""
        init_containers = []

        # Add data preparation init container if needed
        if step.get("data_prep"):
            init_containers.append(
                client.V1Container(
                    name="data-prep",
                    image="busybox:latest",
                    command=["sh", "-c"],
                    args=[step["data_prep"]],
                    volume_mounts=self._create_volume_mounts(step),
                )
            )

        # Add service mesh configuration init container
        if step.get("mesh_config"):
            init_containers.append(
                client.V1Container(
                    name="mesh-config",
                    image="istio/pilot:latest",
                    command=["sh", "-c"],
                    args=["echo 'Configuring service mesh...'"],
                )
            )

        return init_containers

    def _create_volumes(self, step: Dict[str, Any]) -> List[client.V1Volume]:
        """Create volumes for pod"""
        volumes = []

        # Add config map volume
        if step.get("config_map"):
            volumes.append(
                client.V1Volume(
                    name="config",
                    config_map=client.V1ConfigMapVolumeSource(name=step["config_map"]),
                )
            )

        # Add secret volume
        if step.get("secret"):
            volumes.append(
                client.V1Volume(
                    name="secret",
                    secret=client.V1SecretVolumeSource(secret_name=step["secret"]),
                )
            )

        # Add persistent volume claim
        if step.get("pvc"):
            volumes.append(
                client.V1Volume(
                    name="data",
                    persistent_volume_claim=client.V1PersistentVolumeClaimVolumeSource(
                        claim_name=step["pvc"]
                    ),
                )
            )

        return volumes

    def _create_volume_mounts(self, step: Dict[str, Any]) -> List[client.V1VolumeMount]:
        """Create volume mounts for container"""
        mounts = []

        if step.get("config_map"):
            mounts.append(client.V1VolumeMount(name="config", mount_path="/etc/config"))

        if step.get("secret"):
            mounts.append(
                client.V1VolumeMount(
                    name="secret", mount_path="/etc/secret", read_only=True
                )
            )

        if step.get("pvc"):
            mounts.append(client.V1VolumeMount(name="data", mount_path="/data"))

        return mounts

    async def _rollback_pods(self, pod_names: List[str]):
        """Rollback deployed pods on failure"""
        for pod_name in pod_names:
            try:
                self.core_v1.delete_namespaced_pod(
                    name=pod_name, namespace=self.namespace
                )
                logger.info(f"Rolled back pod: {pod_name}")
            except ApiException as e:
                logger.error(f"Failed to rollback pod {pod_name}: {e}")

    async def create_istio_traffic_policy(
        self, service_name: str, policy_config: Dict[str, Any]
    ):
        """
        Create Istio traffic management policy

        Args:
            service_name: Target service name
            policy_config: Traffic policy configuration
        """
        if not self.istio_client:
            logger.warning("Istio client not available")
            return

        # Create VirtualService for traffic routing
        virtual_service = {
            "apiVersion": "networking.istio.io/v1beta1",
            "kind": "VirtualService",
            "metadata": {"name": f"{service_name}-vs", "namespace": self.namespace},
            "spec": {
                "hosts": [service_name],
                "http": [
                    {
                        "match": [{"headers": {"x-version": {"exact": "v2"}}}],
                        "route": [
                            {
                                "destination": {"host": service_name, "subset": "v2"},
                                "weight": policy_config.get("canary_weight", 10),
                            }
                        ],
                        "timeout": f"{policy_config.get('timeout', 30)}s",
                        "retries": {
                            "attempts": policy_config.get("retry_attempts", 3),
                            "perTryTimeout": f"{policy_config.get('retry_timeout', 10)}s",
                        },
                    },
                    {
                        "route": [
                            {
                                "destination": {"host": service_name, "subset": "v1"},
                                "weight": 100 - policy_config.get("canary_weight", 10),
                            }
                        ]
                    },
                ],
            },
        }

        # Create DestinationRule for load balancing
        destination_rule = {
            "apiVersion": "networking.istio.io/v1beta1",
            "kind": "DestinationRule",
            "metadata": {"name": f"{service_name}-dr", "namespace": self.namespace},
            "spec": {
                "host": service_name,
                "trafficPolicy": {
                    "connectionPool": {
                        "tcp": {
                            "maxConnections": policy_config.get("max_connections", 100)
                        },
                        "http": {
                            "http1MaxPendingRequests": policy_config.get(
                                "max_pending_requests", 1000
                            ),
                            "http2MaxRequests": policy_config.get("max_requests", 1000),
                        },
                    },
                    "loadBalancer": {
                        "simple": policy_config.get("load_balancer", "ROUND_ROBIN")
                    },
                    "outlierDetection": {
                        "consecutiveErrors": policy_config.get("consecutive_errors", 5),
                        "interval": f"{policy_config.get('interval', 30)}s",
                        "baseEjectionTime": f"{policy_config.get('ejection_time', 30)}s",
                        "maxEjectionPercent": policy_config.get(
                            "max_ejection_percent", 50
                        ),
                    },
                },
                "subsets": [
                    {"name": "v1", "labels": {"version": "v1"}},
                    {"name": "v2", "labels": {"version": "v2"}},
                ],
            },
        }

        try:
            # Apply VirtualService
            self.istio_client.create_namespaced_custom_object(
                group="networking.istio.io",
                version="v1beta1",
                namespace=self.namespace,
                plural="virtualservices",
                body=virtual_service,
            )

            # Apply DestinationRule
            self.istio_client.create_namespaced_custom_object(
                group="networking.istio.io",
                version="v1beta1",
                namespace=self.namespace,
                plural="destinationrules",
                body=destination_rule,
            )

            logger.info(f"Created Istio traffic policy for {service_name}")

        except ApiException as e:
            logger.error(f"Failed to create Istio traffic policy: {e}")

    async def deploy_canary_release(
        self, service_name: str, new_image: str, canary_percentage: int = 10
    ):
        """
        Deploy canary release with gradual rollout

        Args:
            service_name: Service to update
            new_image: New container image
            canary_percentage: Initial traffic percentage for canary
        """
        try:
            # Get current deployment
            deployment = self.apps_v1.read_namespaced_deployment(
                name=service_name, namespace=self.namespace
            )

            # Create canary deployment
            canary_deployment = deployment
            canary_deployment.metadata.name = f"{service_name}-canary"
            canary_deployment.spec.replicas = max(1, deployment.spec.replicas // 10)
            canary_deployment.spec.template.spec.containers[0].image = new_image
            canary_deployment.spec.template.metadata.labels["version"] = "v2"

            # Deploy canary
            self.apps_v1.create_namespaced_deployment(
                namespace=self.namespace, body=canary_deployment
            )

            # Create traffic policy for canary
            await self.create_istio_traffic_policy(
                service_name, {"canary_weight": canary_percentage}
            )

            logger.info(f"Deployed canary release for {service_name}")

            # Monitor canary health
            asyncio.create_task(
                self._monitor_canary_health(service_name, canary_percentage)
            )

        except ApiException as e:
            logger.error(f"Failed to deploy canary release: {e}")

    async def _monitor_canary_health(self, service_name: str, initial_percentage: int):
        """Monitor canary deployment health and adjust traffic"""
        current_percentage = initial_percentage
        max_percentage = 100
        increment = 10

        while current_percentage < max_percentage:
            await asyncio.sleep(60)  # Check every minute

            # Get canary metrics
            health_score = await self._get_canary_health_score(service_name)

            if health_score > 0.95:  # Healthy threshold
                # Increase traffic to canary
                current_percentage = min(current_percentage + increment, max_percentage)

                await self.create_istio_traffic_policy(
                    service_name, {"canary_weight": current_percentage}
                )

                logger.info(f"Increased canary traffic to {current_percentage}%")

            elif health_score < 0.8:  # Unhealthy threshold
                # Rollback canary
                logger.warning(
                    f"Canary unhealthy (score: {health_score}), rolling back"
                )
                await self._rollback_canary(service_name)
                break

        if current_percentage >= max_percentage:
            # Promote canary to production
            await self._promote_canary(service_name)

    async def _get_canary_health_score(self, service_name: str) -> float:
        """Calculate health score for canary deployment"""
        # This would integrate with your monitoring stack
        # For now, return a simulated score
        import random

        return random.uniform(0.85, 1.0)

    async def _rollback_canary(self, service_name: str):
        """Rollback canary deployment"""
        try:
            # Delete canary deployment
            self.apps_v1.delete_namespaced_deployment(
                name=f"{service_name}-canary", namespace=self.namespace
            )

            # Reset traffic policy
            await self.create_istio_traffic_policy(service_name, {"canary_weight": 0})

            logger.info(f"Rolled back canary for {service_name}")

        except ApiException as e:
            logger.error(f"Failed to rollback canary: {e}")

    async def _promote_canary(self, service_name: str):
        """Promote canary to production"""
        try:
            # Get canary deployment
            canary = self.apps_v1.read_namespaced_deployment(
                name=f"{service_name}-canary", namespace=self.namespace
            )

            # Update production deployment
            production = self.apps_v1.read_namespaced_deployment(
                name=service_name, namespace=self.namespace
            )

            production.spec.template.spec.containers[
                0
            ].image = canary.spec.template.spec.containers[0].image

            self.apps_v1.patch_namespaced_deployment(
                name=service_name, namespace=self.namespace, body=production
            )

            # Delete canary deployment
            self.apps_v1.delete_namespaced_deployment(
                name=f"{service_name}-canary", namespace=self.namespace
            )

            # Reset traffic policy
            await self.create_istio_traffic_policy(service_name, {"canary_weight": 0})

            logger.info(f"Promoted canary to production for {service_name}")

        except ApiException as e:
            logger.error(f"Failed to promote canary: {e}")

    async def watch_workflow_resources(self):
        """Watch for changes in workflow custom resources"""
        w = watch.Watch()

        try:
            for event in w.stream(
                self.custom_api.list_namespaced_custom_object,
                group="orchestrator.io",
                version="v1",
                namespace=self.namespace,
                plural="workflows",
            ):
                event_type = event["type"]
                workflow = event["object"]

                if event_type == "ADDED":
                    await self._handle_workflow_added(workflow)
                elif event_type == "MODIFIED":
                    await self._handle_workflow_modified(workflow)
                elif event_type == "DELETED":
                    await self._handle_workflow_deleted(workflow)

        except Exception as e:
            logger.error(f"Error watching workflow resources: {e}")

    async def _handle_workflow_added(self, workflow: Dict[str, Any]):
        """Handle new workflow resource"""
        workflow_name = workflow["metadata"]["name"]
        logger.info(f"New workflow detected: {workflow_name}")

        # Deploy workflow pods
        await self.deploy_workflow_pods(workflow_name, workflow["spec"]["steps"])

    async def _handle_workflow_modified(self, workflow: Dict[str, Any]):
        """Handle modified workflow resource"""
        workflow_name = workflow["metadata"]["name"]
        logger.info(f"Workflow modified: {workflow_name}")

        # Update workflow status
        self._update_workflow_status(workflow_name, workflow["status"])

    async def _handle_workflow_deleted(self, workflow: Dict[str, Any]):
        """Handle deleted workflow resource"""
        workflow_name = workflow["metadata"]["name"]
        logger.info(f"Workflow deleted: {workflow_name}")

        # Cleanup workflow resources
        await self._cleanup_workflow_resources(workflow_name)

    def _update_workflow_status(self, workflow_name: str, status: Dict[str, Any]):
        """Update workflow status in tracking"""
        if workflow_name in self.workflows:
            self.workflows[workflow_name]["status"] = status

    async def _cleanup_workflow_resources(self, workflow_name: str):
        """Cleanup all resources associated with a workflow"""
        try:
            # Delete all pods with workflow label
            pods = self.core_v1.list_namespaced_pod(
                namespace=self.namespace, label_selector=f"workflow={workflow_name}"
            )

            for pod in pods.items:
                self.core_v1.delete_namespaced_pod(
                    name=pod.metadata.name, namespace=self.namespace
                )

            # Remove from tracking
            self.workflows.pop(workflow_name, None)

            logger.info(f"Cleaned up resources for workflow {workflow_name}")

        except ApiException as e:
            logger.error(f"Failed to cleanup workflow resources: {e}")

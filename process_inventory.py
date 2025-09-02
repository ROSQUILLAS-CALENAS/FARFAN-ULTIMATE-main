"""
Process inventory management system for workflow tracking and dependency analysis
"""

import json
import logging
import hashlib
from typing import Dict, List, Optional, Set, Tuple, Any
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict, deque

# Optional external service clients (guarded imports)
try:
    import etcd3  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    etcd3 = None
try:
    import consul  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    consul = None
# Optional GitPython
try:
    from git import Repo  # type: ignore
except Exception:
    Repo = None
import networkx as nx

# Optional model classes (guarded)
try:
    from models import ProcessDefinition, WorkflowDefinition, DependencyNode  # type: ignore
except Exception:  # pragma: no cover - optional typing fallback
    ProcessDefinition = dict  # type: ignore
    WorkflowDefinition = dict  # type: ignore
    DependencyNode = dict  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class ProcessMetadata:
    """Metadata for a registered process"""
    id: str
    version: str
    checksum: str
    registered_at: datetime
    git_commit: Optional[str] = None
    dependencies: List[str] = None
    tags: List[str] = None
    status: str = "active"


class DependencyAnalyzer:
    """Analyzes process dependencies and detects circular dependencies and bottlenecks"""
    
    def __init__(self):
        self.dependency_graph = nx.DiGraph()
        self.process_metadata: Dict[str, ProcessMetadata] = {}
    
    def add_process(self, process_def: ProcessDefinition, metadata: ProcessMetadata):
        """Add a process to the dependency graph"""
        process_key = f"{process_def.id}:{process_def.version}"
        
        # Add node to graph
        self.dependency_graph.add_node(
            process_key,
            metadata=metadata,
            definition=process_def
        )
        
        self.process_metadata[process_key] = metadata
        
        # Add dependencies as edges
        for dep in process_def.dependencies:
            dep_key = f"{dep['id']}:{dep.get('version', 'latest')}"
            self.dependency_graph.add_edge(dep_key, process_key, 
                                         constraint=dep.get('constraint', 'required'))
    
    def detect_circular_dependencies(self) -> List[List[str]]:
        """Detect circular dependencies in the process graph"""
        try:
            cycles = list(nx.simple_cycles(self.dependency_graph))
            if cycles:
                logger.warning(f"Detected {len(cycles)} circular dependencies")
            return cycles
        except nx.NetworkXError as e:
            logger.error(f"Error detecting cycles: {e}")
            return []
    
    def find_bottlenecks(self, threshold: int = 5) -> List[str]:
        """Find processes that are dependencies for many other processes"""
        bottlenecks = []
        
        for node in self.dependency_graph.nodes():
            in_degree = self.dependency_graph.in_degree(node)
            if in_degree >= threshold:
                bottlenecks.append({
                    'process': node,
                    'dependents': in_degree,
                    'is_critical': in_degree >= threshold * 2
                })
        
        return sorted(bottlenecks, key=lambda x: x['dependents'], reverse=True)
    
    def get_execution_order(self) -> List[List[str]]:
        """Get topological order for process execution"""
        try:
            # Get topological sort
            topo_order = list(nx.topological_sort(self.dependency_graph))
            
            # Group by execution levels (processes that can run in parallel)
            levels = []
            remaining = set(topo_order)
            
            while remaining:
                # Find nodes with no dependencies in remaining set
                current_level = []
                for node in remaining:
                    dependencies = set(self.dependency_graph.predecessors(node))
                    if not dependencies.intersection(remaining):
                        current_level.append(node)
                
                if not current_level:
                    # Handle circular dependencies by breaking cycles
                    logger.warning("Breaking dependency cycle")
                    current_level = [remaining.pop()]
                
                levels.append(current_level)
                remaining -= set(current_level)
            
            return levels
            
        except nx.NetworkXError as e:
            logger.error(f"Error computing execution order: {e}")
            return [[p] for p in self.process_metadata.keys()]
    
    def analyze_impact(self, process_id: str, version: str = None) -> Dict[str, Any]:
        """Analyze the impact of modifying a specific process"""
        process_key = f"{process_id}:{version or 'latest'}"
        
        if process_key not in self.dependency_graph:
            return {"error": "Process not found"}
        
        # Find all downstream dependencies
        downstream = list(nx.descendants(self.dependency_graph, process_key))
        
        # Find all upstream dependencies
        upstream = list(nx.ancestors(self.dependency_graph, process_key))
        
        return {
            "process": process_key,
            "direct_dependents": list(self.dependency_graph.successors(process_key)),
            "all_downstream": downstream,
            "direct_dependencies": list(self.dependency_graph.predecessors(process_key)),
            "all_upstream": upstream,
            "impact_scope": len(downstream),
            "dependency_depth": len(upstream)
        }


class GitIntegration:
    """Handles Git integration for process versioning and dependency resolution"""
    
    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
        self.repo = None
        
        try:
            self.repo = Repo(repo_path)
        except Exception as e:
            logger.warning(f"Git repository not found at {repo_path}: {e}")
    
    def get_current_commit(self) -> Optional[str]:
        """Get current Git commit hash"""
        if not self.repo:
            return None
        
        try:
            return self.repo.head.commit.hexsha
        except Exception as e:
            logger.error(f"Error getting Git commit: {e}")
            return None
    
    def get_file_hash(self, file_path: str) -> str:
        """Get hash of a specific file"""
        full_path = self.repo_path / file_path
        
        if not full_path.exists():
            return ""
        
        with open(full_path, 'rb') as f:
            content = f.read()
            return hashlib.sha256(content).hexdigest()
    
    def track_process_changes(self, process_def: ProcessDefinition) -> Dict[str, str]:
        """Track changes to process definition files"""
        if not self.repo:
            return {}
        
        changes = {}
        
        # Check if process definition files have changed
        for file_path in process_def.source_files:
            try:
                # Get current hash
                current_hash = self.get_file_hash(file_path)
                
                # Get previous hash from git
                try:
                    previous_content = self.repo.git.show(f"HEAD~1:{file_path}")
                    previous_hash = hashlib.sha256(previous_content.encode()).hexdigest()
                except:
                    previous_hash = ""
                
                if current_hash != previous_hash:
                    changes[file_path] = {
                        'previous_hash': previous_hash,
                        'current_hash': current_hash,
                        'changed': True
                    }
            
            except Exception as e:
                logger.warning(f"Error tracking changes for {file_path}: {e}")
        
        return changes


class ProcessInventoryManager:
    """
    Centralized process inventory management with service discovery integration
    """
    
    def __init__(self, 
                 storage_backend: str = "etcd",
                 etcd_host: str = "localhost",
                 etcd_port: int = 2379,
                 consul_host: str = "localhost",
                 consul_port: int = 8500,
                 git_repo_path: str = "."):
        
        self.storage_backend = storage_backend
        self.dependency_analyzer = DependencyAnalyzer()
        self.git_integration = GitIntegration(git_repo_path)
        
        # Initialize storage backend
        if storage_backend == "etcd":
            try:
                self.etcd_client = etcd3.client(host=etcd_host, port=etcd_port)
                logger.info("Connected to etcd for process inventory")
            except Exception as e:
                logger.error(f"Failed to connect to etcd: {e}")
                self.etcd_client = None
        
        elif storage_backend == "consul":
            try:
                self.consul_client = consul.Consul(host=consul_host, port=consul_port)
                logger.info("Connected to Consul for process inventory")
            except Exception as e:
                logger.error(f"Failed to connect to Consul: {e}")
                self.consul_client = None
        
        # Cache for frequently accessed data
        self.process_cache: Dict[str, ProcessDefinition] = {}
        self.cache_invalidation_keys: Set[str] = set()
    
    def register_process(self, process_def: ProcessDefinition) -> str:
        """Register a new process definition"""
        process_key = f"{process_def.id}:{process_def.version}"
        
        # Generate checksum for the process definition
        process_json = json.dumps(asdict(process_def), sort_keys=True)
        checksum = hashlib.sha256(process_json.encode()).hexdigest()
        
        # Create metadata
        metadata = ProcessMetadata(
            id=process_def.id,
            version=process_def.version,
            checksum=checksum,
            registered_at=datetime.utcnow(),
            git_commit=self.git_integration.get_current_commit(),
            dependencies=[dep['id'] for dep in process_def.dependencies],
            tags=process_def.tags
        )
        
        # Store in backend
        try:
            self._store_process(process_key, process_def, metadata)
            
            # Add to dependency analyzer
            self.dependency_analyzer.add_process(process_def, metadata)
            
            # Invalidate cache
            self._invalidate_cache(process_key)
            
            logger.info(f"Registered process {process_key} with checksum {checksum[:8]}")
            return process_key
            
        except Exception as e:
            logger.error(f"Failed to register process {process_key}: {e}")
            raise
    
    def get_process(self, process_id: str, version: str = None) -> Optional[ProcessDefinition]:
        """Get a process definition by ID and version"""
        if version is None:
            version = self.get_latest_version(process_id)
        
        process_key = f"{process_id}:{version}"
        
        # Check cache first
        if process_key in self.process_cache and process_key not in self.cache_invalidation_keys:
            return self.process_cache[process_key]
        
        # Retrieve from storage
        process_def = self._retrieve_process(process_key)
        
        if process_def:
            # Update cache
            self.process_cache[process_key] = process_def
            self.cache_invalidation_keys.discard(process_key)
        
        return process_def
    
    def get_all_processes(self) -> List[ProcessDefinition]:
        """Get all registered processes"""
        return self._retrieve_all_processes()
    
    def get_latest_version(self, process_id: str) -> str:
        """Get the latest version of a process"""
        versions = self._get_process_versions(process_id)
        if not versions:
            return "1.0.0"
        
        # Sort versions semantically (simplified)
        versions.sort(key=lambda v: tuple(map(int, v.split('.'))))
        return versions[-1]
    
    def analyze_dependencies(self) -> Dict[str, Any]:
        """Perform comprehensive dependency analysis"""
        circular_deps = self.dependency_analyzer.detect_circular_dependencies()
        bottlenecks = self.dependency_analyzer.find_bottlenecks()
        execution_order = self.dependency_analyzer.get_execution_order()
        
        return {
            "circular_dependencies": circular_deps,
            "bottlenecks": bottlenecks,
            "execution_levels": execution_order,
            "total_processes": len(self.dependency_analyzer.process_metadata),
            "analysis_timestamp": datetime.utcnow().isoformat()
        }
    
    def get_process_impact(self, process_id: str, version: str = None) -> Dict[str, Any]:
        """Get impact analysis for a specific process"""
        return self.dependency_analyzer.analyze_impact(process_id, version)
    
    def track_process_changes(self, process_id: str) -> Dict[str, Any]:
        """Track changes to a process using Git integration"""
        process_def = self.get_process(process_id)
        if not process_def:
            return {"error": "Process not found"}
        
        return self.git_integration.track_process_changes(process_def)
    
    def invalidate_cache_for_git_changes(self) -> List[str]:
        """Invalidate cache for processes that have changed in Git"""
        invalidated = []
        
        for process_key, process_def in self.process_cache.items():
            changes = self.git_integration.track_process_changes(process_def)
            if any(change['changed'] for change in changes.values()):
                self._invalidate_cache(process_key)
                invalidated.append(process_key)
        
        return invalidated
    
    def _store_process(self, process_key: str, process_def: ProcessDefinition, metadata: ProcessMetadata):
        """Store process in the selected backend"""
        data = {
            'definition': asdict(process_def),
            'metadata': asdict(metadata)
        }
        
        if self.storage_backend == "etcd" and self.etcd_client:
            self.etcd_client.put(f"/processes/{process_key}", json.dumps(data))
        
        elif self.storage_backend == "consul" and self.consul_client:
            self.consul_client.kv.put(f"processes/{process_key}", json.dumps(data))
        
        else:
            # Fallback to local storage
            storage_dir = Path("./process_inventory")
            storage_dir.mkdir(exist_ok=True)
            
            with open(storage_dir / f"{process_key.replace(':', '_')}.json", 'w') as f:
                json.dump(data, f, indent=2, default=str)
    
    def _retrieve_process(self, process_key: str) -> Optional[ProcessDefinition]:
        """Retrieve process from the selected backend"""
        try:
            data = None
            
            if self.storage_backend == "etcd" and self.etcd_client:
                value, _ = self.etcd_client.get(f"/processes/{process_key}")
                if value:
                    data = json.loads(value)
            
            elif self.storage_backend == "consul" and self.consul_client:
                _, result = self.consul_client.kv.get(f"processes/{process_key}")
                if result and result['Value']:
                    data = json.loads(result['Value'])
            
            else:
                # Fallback to local storage
                storage_file = Path(f"./process_inventory/{process_key.replace(':', '_')}.json")
                if storage_file.exists():
                    with open(storage_file) as f:
                        data = json.load(f)
            
            if data:
                return ProcessDefinition(**data['definition'])
            
        except Exception as e:
            logger.error(f"Error retrieving process {process_key}: {e}")
        
        return None
    
    def _retrieve_all_processes(self) -> List[ProcessDefinition]:
        """Retrieve all processes from storage backend"""
        processes = []
        
        try:
            if self.storage_backend == "etcd" and self.etcd_client:
                for value, _ in self.etcd_client.get_prefix("/processes/"):
                    data = json.loads(value)
                    processes.append(ProcessDefinition(**data['definition']))
            
            elif self.storage_backend == "consul" and self.consul_client:
                _, results = self.consul_client.kv.get("processes/", recurse=True)
                if results:
                    for result in results:
                        data = json.loads(result['Value'])
                        processes.append(ProcessDefinition(**data['definition']))
            
            else:
                # Fallback to local storage
                storage_dir = Path("./process_inventory")
                if storage_dir.exists():
                    for json_file in storage_dir.glob("*.json"):
                        with open(json_file) as f:
                            data = json.load(f)
                            processes.append(ProcessDefinition(**data['definition']))
        
        except Exception as e:
            logger.error(f"Error retrieving all processes: {e}")
        
        return processes
    
    def _get_process_versions(self, process_id: str) -> List[str]:
        """Get all versions of a specific process"""
        versions = []
        
        try:
            if self.storage_backend == "etcd" and self.etcd_client:
                for value, metadata in self.etcd_client.get_prefix(f"/processes/{process_id}:"):
                    key = metadata.key.decode()
                    version = key.split(':')[-1]
                    versions.append(version)
            
            elif self.storage_backend == "consul" and self.consul_client:
                _, results = self.consul_client.kv.get(f"processes/{process_id}:", recurse=True)
                if results:
                    for result in results:
                        key = result['Key']
                        version = key.split(':')[-1]
                        versions.append(version)
            
            else:
                # Fallback to local storage
                storage_dir = Path("./process_inventory")
                if storage_dir.exists():
                    pattern = f"{process_id}_*.json"
                    for json_file in storage_dir.glob(pattern):
                        # Extract version from filename
                        filename = json_file.stem
                        version = filename.split('_', 1)[1].replace('_', ':')
                        if ':' in version:
                            versions.append(version.split(':', 1)[1])
        
        except Exception as e:
            logger.error(f"Error getting versions for {process_id}: {e}")
        
        return versions
    
    def _invalidate_cache(self, process_key: str):
        """Mark cache entry for invalidation"""
        self.cache_invalidation_keys.add(process_key)
        
        # Also invalidate related processes (dependents)
        if hasattr(self, 'dependency_analyzer'):
            try:
                graph = self.dependency_analyzer.dependency_graph
                if process_key in graph:
                    dependents = list(graph.successors(process_key))
                    for dependent in dependents:
                        self.cache_invalidation_keys.add(dependent)
            except:
                pass
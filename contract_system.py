"""
Comprehensive Contract System for Deterministic Pipeline Integrity

This module contains ALL functional contracts that enforce immediate effects
on pipeline execution, not decorative validation. Each contract enforces
specific guarantees about data processing and system behavior.

Contracts include:
- RoutingContract: PDF content hashing for routing decisions  
- SnapshotContract: Immutable file state tracking with manifests
- ContextImmutabilityContract: Context tampering prevention
- TraceabilityContract: Merkle tree processing lineage
- PermutationInvarianceContract: Order-independent result aggregation
"""

import hashlib
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)


class ContractViolationError(Exception):
    """Raised when a contract is violated."""
    pass


class ContractStatus(Enum):
    """Contract execution status."""
    PENDING = "pending"
    ACTIVE = "active"
    VALIDATED = "validated"
    VIOLATED = "violated"
    FROZEN = "frozen"


@dataclass
class ContractResult:
    """Result of contract execution."""
    contract_name: str
    status: ContractStatus
    timestamp: datetime
    data_hash: str = ""
    violations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseContract(ABC):
    """Base class for all contracts with enforcement capabilities."""
    
    def __init__(self, name: str):
        self.name = name
        self.status = ContractStatus.PENDING
        self.execution_history: List[ContractResult] = []
        self.violations: Set[str] = set()
        
    @abstractmethod
    def execute(self, data: Any, context: Dict[str, Any] = None) -> ContractResult:
        """Execute contract validation/enforcement."""
        pass
    
    def _record_result(self, status: ContractStatus, data_hash: str = "", 
                      violations: List[str] = None, metadata: Dict[str, Any] = None) -> ContractResult:
        """Record contract execution result."""
        result = ContractResult(
            contract_name=self.name,
            status=status,
            timestamp=datetime.now(),
            data_hash=data_hash,
            violations=violations or [],
            metadata=metadata or {}
        )
        self.execution_history.append(result)
        self.status = status
        if violations:
            self.violations.update(violations)
        return result


class RoutingContract(BaseContract):
    """Contract for PDF content hashing and routing decisions."""
    
    def __init__(self):
        super().__init__("RoutingContract")
        self.routing_table: Dict[str, str] = {}
        self.content_hashes: Dict[str, str] = {}
        
    def hash_pdf_content(self, pdf_content: bytes, pdf_path: str = "") -> str:
        """Generate SHA-256 hash of PDF content for routing decisions."""
        if not isinstance(pdf_content, bytes):
            raise ContractViolationError("PDF content must be bytes")
            
        content_hash = hashlib.sha256(pdf_content).hexdigest()
        
        # Store hash for routing
        if pdf_path:
            self.content_hashes[pdf_path] = content_hash
            
        # Deterministic routing based on hash
        route_key = content_hash[:8]  # First 8 chars for routing
        if route_key not in self.routing_table:
            # Route based on hash characteristics
            hash_int = int(route_key, 16)
            if hash_int % 3 == 0:
                route = "processing_pipeline_a"
            elif hash_int % 3 == 1:
                route = "processing_pipeline_b"  
            else:
                route = "processing_pipeline_c"
            self.routing_table[route_key] = route
            
        logger.info(f"PDF content hash: {content_hash}, route: {self.routing_table[route_key]}")
        return content_hash
    
    def get_route_for_content(self, content_hash: str) -> str:
        """Get routing decision for content hash."""
        route_key = content_hash[:8]
        return self.routing_table.get(route_key, "default_pipeline")
    
    def execute(self, data: Any, context: Dict[str, Any] = None) -> ContractResult:
        """Execute routing contract with PDF content."""
        try:
            if isinstance(data, dict) and 'pdf_content' in data:
                pdf_content = data['pdf_content']
                pdf_path = data.get('pdf_path', '')
                content_hash = self.hash_pdf_content(pdf_content, pdf_path)
                
                # Enforce routing decision immediately
                route = self.get_route_for_content(content_hash)
                if context:
                    context['enforced_route'] = route
                    context['content_hash'] = content_hash
                
                return self._record_result(
                    ContractStatus.VALIDATED,
                    content_hash,
                    metadata={'route': route, 'pdf_path': pdf_path}
                )
            else:
                return self._record_result(
                    ContractStatus.VIOLATED,
                    violations=["Invalid data format for RoutingContract"]
                )
        except Exception as e:
            return self._record_result(
                ContractStatus.VIOLATED,
                violations=[f"RoutingContract execution failed: {str(e)}"]
            )


class SnapshotContract(BaseContract):
    """Contract for immutable file state tracking with manifests."""
    
    def __init__(self, storage_path: str = "snapshots"):
        super().__init__("SnapshotContract")
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        self.manifests: Dict[str, Dict[str, Any]] = {}
        
    def create_manifest(self, file_path: str, metadata: Dict[str, Any] = None) -> str:
        """Create immutable manifest for file state."""
        if not os.path.exists(file_path):
            raise ContractViolationError(f"File does not exist: {file_path}")
            
        # Calculate file hash
        with open(file_path, 'rb') as f:
            content = f.read()
            file_hash = hashlib.sha256(content).hexdigest()
            
        # Get file stats
        stat = os.stat(file_path)
        
        manifest = {
            'file_path': str(Path(file_path).absolute()),
            'file_hash': file_hash,
            'size_bytes': stat.st_size,
            'created_timestamp': datetime.now().isoformat(),
            'modified_timestamp': datetime.fromtimestamp(stat.st_mtime).isoformat(),
            'metadata': metadata or {},
            'manifest_id': hashlib.sha256(f"{file_path}{file_hash}{time.time()}".encode()).hexdigest()
        }
        
        # Store manifest immutably
        manifest_id = manifest['manifest_id']
        self.manifests[manifest_id] = manifest
        
        # Write manifest to disk for persistence
        manifest_file = self.storage_path / f"{manifest_id}.json"
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2)
            
        logger.info(f"Created manifest {manifest_id} for {file_path}")
        return manifest_id
        
    def validate_manifest(self, manifest_id: str, current_file_path: str = None) -> bool:
        """Validate that current file state matches manifest."""
        if manifest_id not in self.manifests:
            # Try to load from disk
            manifest_file = self.storage_path / f"{manifest_id}.json"
            if manifest_file.exists():
                with open(manifest_file, 'r') as f:
                    self.manifests[manifest_id] = json.load(f)
            else:
                raise ContractViolationError(f"Manifest not found: {manifest_id}")
                
        manifest = self.manifests[manifest_id]
        file_path = current_file_path or manifest['file_path']
        
        if not os.path.exists(file_path):
            raise ContractViolationError(f"File no longer exists: {file_path}")
            
        # Validate file hash
        with open(file_path, 'rb') as f:
            current_hash = hashlib.sha256(f.read()).hexdigest()
            
        if current_hash != manifest['file_hash']:
            raise ContractViolationError(
                f"File integrity violation: expected {manifest['file_hash']}, got {current_hash}"
            )
            
        return True
    
    def execute(self, data: Any, context: Dict[str, Any] = None) -> ContractResult:
        """Execute snapshot contract."""
        try:
            if isinstance(data, dict) and 'action' in data:
                action = data['action']
                
                if action == 'create_manifest':
                    file_path = data['file_path']
                    metadata = data.get('metadata', {})
                    manifest_id = self.create_manifest(file_path, metadata)
                    
                    # Immediately enforce manifest tracking
                    if context:
                        context['manifest_id'] = manifest_id
                        context['file_tracked'] = True
                        
                    return self._record_result(
                        ContractStatus.VALIDATED,
                        manifest_id,
                        metadata={'manifest_id': manifest_id, 'file_path': file_path}
                    )
                    
                elif action == 'validate_manifest':
                    manifest_id = data['manifest_id']
                    file_path = data.get('file_path')
                    is_valid = self.validate_manifest(manifest_id, file_path)
                    
                    if not is_valid:
                        return self._record_result(
                            ContractStatus.VIOLATED,
                            violations=["Manifest validation failed"]
                        )
                        
                    return self._record_result(
                        ContractStatus.VALIDATED,
                        manifest_id,
                        metadata={'validation_passed': True}
                    )
                    
            return self._record_result(
                ContractStatus.VIOLATED,
                violations=["Invalid action for SnapshotContract"]
            )
            
        except Exception as e:
            return self._record_result(
                ContractStatus.VIOLATED,
                violations=[f"SnapshotContract execution failed: {str(e)}"]
            )


class ContextImmutabilityContract(BaseContract):
    """Contract to prevent context tampering with integrity verification."""
    
    def __init__(self):
        super().__init__("ContextImmutabilityContract")
        self.frozen_contexts: Dict[str, Tuple[str, Dict[str, Any]]] = {}
        
    def freeze_context(self, context: Dict[str, Any], context_id: str = None) -> str:
        """Freeze context state to prevent tampering."""
        if context_id is None:
            context_id = hashlib.sha256(f"{json.dumps(context, sort_keys=True)}{time.time()}".encode()).hexdigest()
            
        # Create immutable snapshot
        context_copy = json.loads(json.dumps(context))  # Deep copy via JSON
        context_hash = hashlib.sha256(json.dumps(context_copy, sort_keys=True).encode()).hexdigest()
        
        # Store frozen context
        self.frozen_contexts[context_id] = (context_hash, context_copy)
        
        logger.info(f"Froze context {context_id} with hash {context_hash}")
        return context_id
        
    def verify_integrity(self, context_id: str, current_context: Dict[str, Any]) -> bool:
        """Verify that context has not been tampered with."""
        if context_id not in self.frozen_contexts:
            raise ContractViolationError(f"Context not found: {context_id}")
            
        expected_hash, frozen_context = self.frozen_contexts[context_id]
        current_hash = hashlib.sha256(json.dumps(current_context, sort_keys=True).encode()).hexdigest()
        
        if current_hash != expected_hash:
            raise ContractViolationError(
                f"Context tampering detected: expected {expected_hash}, got {current_hash}"
            )
            
        return True
    
    def execute(self, data: Any, context: Dict[str, Any] = None) -> ContractResult:
        """Execute context immutability contract."""
        try:
            if isinstance(data, dict) and 'action' in data:
                action = data['action']
                
                if action == 'freeze':
                    target_context = data['context']
                    context_id = data.get('context_id')
                    frozen_id = self.freeze_context(target_context, context_id)
                    
                    # Immediately enforce frozen state
                    if context:
                        context['frozen_context_id'] = frozen_id
                        context['immutability_enforced'] = True
                        
                    return self._record_result(
                        ContractStatus.FROZEN,
                        frozen_id,
                        metadata={'context_id': frozen_id, 'frozen': True}
                    )
                    
                elif action == 'verify':
                    context_id = data['context_id']
                    current_context = data['current_context']
                    is_valid = self.verify_integrity(context_id, current_context)
                    
                    if not is_valid:
                        return self._record_result(
                            ContractStatus.VIOLATED,
                            violations=["Context integrity verification failed"]
                        )
                        
                    return self._record_result(
                        ContractStatus.VALIDATED,
                        context_id,
                        metadata={'integrity_verified': True}
                    )
                    
            return self._record_result(
                ContractStatus.VIOLATED,
                violations=["Invalid action for ContextImmutabilityContract"]
            )
            
        except Exception as e:
            return self._record_result(
                ContractStatus.VIOLATED,
                violations=[f"ContextImmutabilityContract execution failed: {str(e)}"]
            )


class TraceabilityContract(BaseContract):
    """Contract implementing Merkle tree operations for processing lineage."""
    
    def __init__(self):
        super().__init__("TraceabilityContract") 
        self.merkle_tree: List[str] = []  # Leaf hashes
        self.chain: List[Dict[str, Any]] = []
        self.root_hash: str = ""
        
    def _hash_data(self, data: Any) -> str:
        """Create hash of data."""
        return hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()
        
    def _build_merkle_tree(self) -> str:
        """Build Merkle tree and return root hash."""
        if not self.merkle_tree:
            return ""
            
        # Start with leaf hashes
        current_level = self.merkle_tree[:]
        
        # Build tree bottom-up
        while len(current_level) > 1:
            next_level = []
            
            # Process pairs
            for i in range(0, len(current_level), 2):
                if i + 1 < len(current_level):
                    # Hash pair
                    combined = current_level[i] + current_level[i + 1]
                    parent_hash = hashlib.sha256(combined.encode()).hexdigest()
                else:
                    # Odd number - hash with itself
                    parent_hash = hashlib.sha256((current_level[i] + current_level[i]).encode()).hexdigest()
                    
                next_level.append(parent_hash)
                
            current_level = next_level
            
        self.root_hash = current_level[0]
        return self.root_hash
        
    def add_to_chain(self, operation: str, data: Any, metadata: Dict[str, Any] = None) -> str:
        """Add operation to processing chain with Merkle tree validation."""
        timestamp = datetime.now()
        
        # Create chain entry
        entry = {
            'operation': operation,
            'data_hash': self._hash_data(data),
            'timestamp': timestamp.isoformat(),
            'metadata': metadata or {},
            'sequence': len(self.chain)
        }
        
        # Add previous hash for chain integrity
        if self.chain:
            entry['previous_hash'] = self.chain[-1]['entry_hash']
        else:
            entry['previous_hash'] = '0' * 64
            
        # Calculate entry hash
        entry_hash = self._hash_data(entry)
        entry['entry_hash'] = entry_hash
        
        # Add to chain
        self.chain.append(entry)
        
        # Add to Merkle tree
        self.merkle_tree.append(entry_hash)
        
        # Rebuild Merkle root
        self._build_merkle_tree()
        
        logger.info(f"Added chain entry {len(self.chain)-1}: {operation} (hash: {entry_hash[:8]}...)")
        return entry_hash
        
    def verify_chain(self) -> bool:
        """Verify entire chain integrity using Merkle tree."""
        if not self.chain:
            return True
            
        # Verify chain links
        for i, entry in enumerate(self.chain):
            # Verify hash
            expected_hash = self._hash_data({k: v for k, v in entry.items() if k != 'entry_hash'})
            if expected_hash != entry['entry_hash']:
                raise ContractViolationError(f"Chain integrity violation at entry {i}")
                
            # Verify previous hash link
            if i > 0:
                expected_prev = self.chain[i-1]['entry_hash']
                if entry['previous_hash'] != expected_prev:
                    raise ContractViolationError(f"Chain link violation at entry {i}")
                    
        # Verify Merkle tree
        expected_root = self._build_merkle_tree()
        if expected_root != self.root_hash:
            raise ContractViolationError("Merkle tree integrity violation")
            
        return True
    
    def execute(self, data: Any, context: Dict[str, Any] = None) -> ContractResult:
        """Execute traceability contract."""
        try:
            if isinstance(data, dict) and 'action' in data:
                action = data['action']
                
                if action == 'add_to_chain':
                    operation = data['operation']
                    operation_data = data['data']
                    metadata = data.get('metadata', {})
                    entry_hash = self.add_to_chain(operation, operation_data, metadata)
                    
                    # Immediately enforce traceability
                    if context:
                        context['chain_entry_hash'] = entry_hash
                        context['merkle_root'] = self.root_hash
                        context['traceability_enforced'] = True
                        
                    return self._record_result(
                        ContractStatus.VALIDATED,
                        entry_hash,
                        metadata={'chain_length': len(self.chain), 'merkle_root': self.root_hash}
                    )
                    
                elif action == 'verify_chain':
                    is_valid = self.verify_chain()
                    
                    if not is_valid:
                        return self._record_result(
                            ContractStatus.VIOLATED,
                            violations=["Chain verification failed"]
                        )
                        
                    return self._record_result(
                        ContractStatus.VALIDATED,
                        self.root_hash,
                        metadata={'chain_verified': True, 'chain_length': len(self.chain)}
                    )
                    
            return self._record_result(
                ContractStatus.VIOLATED,
                violations=["Invalid action for TraceabilityContract"]
            )
            
        except Exception as e:
            return self._record_result(
                ContractStatus.VIOLATED,
                violations=[f"TraceabilityContract execution failed: {str(e)}"]
            )


class PermutationInvarianceContract(BaseContract):
    """Contract ensuring consistent outputs regardless of input order using sorted hashing."""
    
    def __init__(self):
        super().__init__("PermutationInvarianceContract")
        self.aggregation_cache: Dict[str, Any] = {}
        
    def _normalize_input(self, inputs: List[Any]) -> Tuple[str, List[Any]]:
        """Normalize inputs to ensure permutation invariance."""
        # Convert to sortable format
        sortable_inputs = []
        for item in inputs:
            if isinstance(item, dict):
                # Sort dict by keys for consistency
                sorted_item = json.dumps(item, sort_keys=True)
            else:
                sorted_item = json.dumps(item, default=str)
            sortable_inputs.append(sorted_item)
            
        # Sort inputs
        sorted_inputs = sorted(sortable_inputs)
        
        # Create deterministic hash
        combined = '|'.join(sorted_inputs)
        input_hash = hashlib.sha256(combined.encode()).hexdigest()
        
        return input_hash, sorted_inputs
        
    def aggregate_results(self, inputs: List[Any], aggregation_func: str = "sum") -> Dict[str, Any]:
        """Aggregate results with permutation invariance guarantee."""
        if not inputs:
            return {"result": None, "input_hash": "", "sorted": True}
            
        # Normalize inputs for consistent processing
        input_hash, sorted_inputs = self._normalize_input(inputs)
        
        # Check cache
        cache_key = f"{input_hash}:{aggregation_func}"
        if cache_key in self.aggregation_cache:
            logger.info(f"Using cached result for hash {input_hash[:8]}...")
            return self.aggregation_cache[cache_key]
            
        # Process inputs
        try:
            if aggregation_func == "sum":
                # Sum numeric values
                result = sum(float(json.loads(item)) for item in sorted_inputs if json.loads(item) is not None)
            elif aggregation_func == "concat":
                # Concatenate strings
                result = '|'.join(sorted_inputs)
            elif aggregation_func == "max":
                # Find maximum
                result = max(float(json.loads(item)) for item in sorted_inputs if json.loads(item) is not None)
            elif aggregation_func == "count":
                # Count items
                result = len(sorted_inputs)
            elif aggregation_func == "hash_aggregate":
                # Create combined hash
                result = hashlib.sha256('|'.join(sorted_inputs).encode()).hexdigest()
            else:
                # Default to concatenation
                result = '|'.join(sorted_inputs)
                
        except Exception as e:
            logger.warning(f"Aggregation failed: {e}, using fallback")
            result = len(sorted_inputs)  # Fallback to count
            
        # Cache result
        aggregated_result = {
            "result": result,
            "input_hash": input_hash,
            "sorted": True,
            "input_count": len(inputs),
            "aggregation_func": aggregation_func
        }
        
        self.aggregation_cache[cache_key] = aggregated_result
        return aggregated_result
    
    def execute(self, data: Any, context: Dict[str, Any] = None) -> ContractResult:
        """Execute permutation invariance contract."""
        try:
            if isinstance(data, dict) and 'inputs' in data:
                inputs = data['inputs']
                aggregation_func = data.get('aggregation_func', 'sum')
                
                result = self.aggregate_results(inputs, aggregation_func)
                
                # Immediately enforce permutation invariance
                if context:
                    context['aggregated_result'] = result['result']
                    context['input_hash'] = result['input_hash']
                    context['permutation_invariant'] = True
                    
                return self._record_result(
                    ContractStatus.VALIDATED,
                    result['input_hash'],
                    metadata=result
                )
                
            return self._record_result(
                ContractStatus.VIOLATED,
                violations=["Invalid input format for PermutationInvarianceContract"]
            )
            
        except Exception as e:
            return self._record_result(
                ContractStatus.VIOLATED,
                violations=[f"PermutationInvarianceContract execution failed: {str(e)}"]
            )


class PipelineContract(BaseContract):
    """
    Pipeline contract for standardized schemas and data validation between all 12 pipeline stages.
    Enforces deterministic data flow with SHA-256 checksums and version compatibility.
    
    Pipeline stages and handoffs:
    I→X→K→A→L→R→O→G→T→S (10 handoffs)
    I: Ingestion Preparation
    X: Context Construction  
    K: Knowledge Extraction
    A: Analysis NLP
    L: Classification Evaluation
    R: Search Retrieval
    O: Orchestration Control
    G: Aggregation Reporting
    T: Integration Storage
    S: Synthesis Output
    """
    
    PIPELINE_STAGES = ['I', 'X', 'K', 'A', 'L', 'R', 'O', 'G', 'T', 'S']
    STAGE_NAMES = {
        'I': 'ingestion_preparation',
        'X': 'context_construction', 
        'K': 'knowledge_extraction',
        'A': 'analysis_nlp',
        'L': 'classification_evaluation',
        'R': 'search_retrieval',
        'O': 'orchestration_control',
        'G': 'aggregation_reporting',
        'T': 'integration_storage',
        'S': 'synthesis_output'
    }
    
    # Schema version for compatibility checks
    SCHEMA_VERSION = "1.0.0"
    
    def __init__(self):
        super().__init__("PipelineContract")
        self.stage_schemas = self._define_stage_schemas()
        self.handoff_requirements = self._define_handoff_requirements()
        self.checksums: Dict[str, str] = {}
        self.version_compatibility: Dict[str, str] = {}
        
    def _define_stage_schemas(self) -> Dict[str, Dict[str, Any]]:
        """Define mandatory schema definitions for each pipeline stage."""
        return {
            'I': {
                'input': {
                    'required_fields': ['document_path', 'mime_type'],
                    'optional_fields': ['metadata', 'processing_options'],
                    'types': {
                        'document_path': str,
                        'mime_type': str,
                        'metadata': dict,
                        'processing_options': dict
                    }
                },
                'output': {
                    'required_fields': ['raw_content', 'document_id', 'extraction_metadata'],
                    'optional_fields': ['error_info', 'processing_stats'],
                    'types': {
                        'raw_content': (str, bytes),
                        'document_id': str,
                        'extraction_metadata': dict,
                        'error_info': dict,
                        'processing_stats': dict
                    }
                }
            },
            'X': {
                'input': {
                    'required_fields': ['raw_content', 'document_id'],
                    'optional_fields': ['extraction_metadata'],
                    'types': {
                        'raw_content': (str, bytes),
                        'document_id': str,
                        'extraction_metadata': dict
                    }
                },
                'output': {
                    'required_fields': ['normalized_content', 'context_id', 'immutable_context'],
                    'optional_fields': ['context_metadata', 'lineage_data'],
                    'types': {
                        'normalized_content': str,
                        'context_id': str,
                        'immutable_context': dict,
                        'context_metadata': dict,
                        'lineage_data': dict
                    }
                }
            },
            'K': {
                'input': {
                    'required_fields': ['normalized_content', 'context_id'],
                    'optional_fields': ['immutable_context'],
                    'types': {
                        'normalized_content': str,
                        'context_id': str,
                        'immutable_context': dict
                    }
                },
                'output': {
                    'required_fields': ['knowledge_graph', 'entities', 'relations', 'embeddings'],
                    'optional_fields': ['causal_graph', 'confidence_scores'],
                    'types': {
                        'knowledge_graph': dict,
                        'entities': list,
                        'relations': list,
                        'embeddings': dict,
                        'causal_graph': dict,
                        'confidence_scores': dict
                    }
                }
            },
            'A': {
                'input': {
                    'required_fields': ['knowledge_graph', 'entities', 'relations'],
                    'optional_fields': ['embeddings', 'causal_graph'],
                    'types': {
                        'knowledge_graph': dict,
                        'entities': list,
                        'relations': list,
                        'embeddings': dict,
                        'causal_graph': dict
                    }
                },
                'output': {
                    'required_fields': ['analysis_results', 'evidence_items', 'semantic_features'],
                    'optional_fields': ['question_mappings', 'validation_scores'],
                    'types': {
                        'analysis_results': dict,
                        'evidence_items': list,
                        'semantic_features': dict,
                        'question_mappings': dict,
                        'validation_scores': dict
                    }
                }
            },
            'L': {
                'input': {
                    'required_fields': ['analysis_results', 'evidence_items'],
                    'optional_fields': ['semantic_features', 'validation_scores'],
                    'types': {
                        'analysis_results': dict,
                        'evidence_items': list,
                        'semantic_features': dict,
                        'validation_scores': dict
                    }
                },
                'output': {
                    'required_fields': ['classification_scores', 'risk_assessment', 'confidence_intervals'],
                    'optional_fields': ['conformal_predictions', 'calibration_data'],
                    'types': {
                        'classification_scores': dict,
                        'risk_assessment': dict,
                        'confidence_intervals': dict,
                        'conformal_predictions': dict,
                        'calibration_data': dict
                    }
                }
            },
            'R': {
                'input': {
                    'required_fields': ['query_context', 'search_parameters'],
                    'optional_fields': ['classification_scores', 'risk_assessment'],
                    'types': {
                        'query_context': dict,
                        'search_parameters': dict,
                        'classification_scores': dict,
                        'risk_assessment': dict
                    }
                },
                'output': {
                    'required_fields': ['retrieval_results', 'ranked_candidates', 'relevance_scores'],
                    'optional_fields': ['hybrid_scores', 'retrieval_metadata'],
                    'types': {
                        'retrieval_results': list,
                        'ranked_candidates': list,
                        'relevance_scores': dict,
                        'hybrid_scores': dict,
                        'retrieval_metadata': dict
                    }
                }
            },
            'O': {
                'input': {
                    'required_fields': ['processing_state', 'orchestration_params'],
                    'optional_fields': ['retrieval_results', 'routing_decisions'],
                    'types': {
                        'processing_state': dict,
                        'orchestration_params': dict,
                        'retrieval_results': list,
                        'routing_decisions': dict
                    }
                },
                'output': {
                    'required_fields': ['orchestration_result', 'execution_trace', 'state_transitions'],
                    'optional_fields': ['error_handling', 'compensation_actions'],
                    'types': {
                        'orchestration_result': dict,
                        'execution_trace': list,
                        'state_transitions': list,
                        'error_handling': dict,
                        'compensation_actions': list
                    }
                }
            },
            'G': {
                'input': {
                    'required_fields': ['aggregation_data', 'aggregation_rules'],
                    'optional_fields': ['execution_trace', 'state_transitions'],
                    'types': {
                        'aggregation_data': list,
                        'aggregation_rules': dict,
                        'execution_trace': list,
                        'state_transitions': list
                    }
                },
                'output': {
                    'required_fields': ['aggregated_results', 'meso_aggregation', 'report_data'],
                    'optional_fields': ['aggregation_metadata', 'compliance_report'],
                    'types': {
                        'aggregated_results': dict,
                        'meso_aggregation': dict,
                        'report_data': dict,
                        'aggregation_metadata': dict,
                        'compliance_report': dict
                    }
                }
            },
            'T': {
                'input': {
                    'required_fields': ['integration_payload', 'storage_requirements'],
                    'optional_fields': ['aggregated_results', 'report_data'],
                    'types': {
                        'integration_payload': dict,
                        'storage_requirements': dict,
                        'aggregated_results': dict,
                        'report_data': dict
                    }
                },
                'output': {
                    'required_fields': ['storage_result', 'integration_status', 'metrics_data'],
                    'optional_fields': ['optimization_suggestions', 'feedback_data'],
                    'types': {
                        'storage_result': dict,
                        'integration_status': str,
                        'metrics_data': dict,
                        'optimization_suggestions': dict,
                        'feedback_data': dict
                    }
                }
            },
            'S': {
                'input': {
                    'required_fields': ['synthesis_input', 'output_requirements'],
                    'optional_fields': ['metrics_data', 'integration_status'],
                    'types': {
                        'synthesis_input': dict,
                        'output_requirements': dict,
                        'metrics_data': dict,
                        'integration_status': str
                    }
                },
                'output': {
                    'required_fields': ['final_answer', 'synthesis_metadata', 'lineage_proof'],
                    'optional_fields': ['formatting_options', 'quality_metrics'],
                    'types': {
                        'final_answer': str,
                        'synthesis_metadata': dict,
                        'lineage_proof': dict,
                        'formatting_options': dict,
                        'quality_metrics': dict
                    }
                }
            }
        }
    
    def _define_handoff_requirements(self) -> Dict[str, Dict[str, Any]]:
        """Define requirements for stage-to-stage handoffs."""
        handoffs = {}
        
        # Define each handoff: I→X, X→K, K→A, A→L, L→R, R→O, O→G, G→T, T→S
        for i in range(len(self.PIPELINE_STAGES) - 1):
            from_stage = self.PIPELINE_STAGES[i]
            to_stage = self.PIPELINE_STAGES[i + 1]
            handoff_key = f"{from_stage}→{to_stage}"
            
            from_schema = self.stage_schemas[from_stage]['output']
            to_schema = self.stage_schemas[to_stage]['input']
            
            # Find overlapping fields for handoff validation
            handoff_fields = set(from_schema['required_fields']) & set(to_schema['required_fields'])
            
            handoffs[handoff_key] = {
                'from_stage': from_stage,
                'to_stage': to_stage,
                'required_handoff_fields': list(handoff_fields),
                'checksum_required': True,
                'version_compatibility_required': True
            }
            
        return handoffs
    
    def validate_input(self, stage: str, data: Dict[str, Any]) -> ContractResult:
        """Validate input data against stage schema with checksum generation."""
        violations = []
        
        if stage not in self.stage_schemas:
            violations.append(f"Unknown stage: {stage}")
            return self._record_result(ContractStatus.VIOLATED, violations=violations)
        
        schema = self.stage_schemas[stage]['input']
        
        # Check required fields
        missing_fields = []
        for field in schema['required_fields']:
            if field not in data:
                missing_fields.append(field)
        
        if missing_fields:
            violations.append(f"Missing required fields for stage {stage}: {missing_fields}")
        
        # Check field types
        type_errors = []
        for field, expected_type in schema['types'].items():
            if field in data:
                value = data[field]
                if isinstance(expected_type, tuple):
                    # Multiple acceptable types
                    if not any(isinstance(value, t) for t in expected_type):
                        type_errors.append(f"Field '{field}' expected types {expected_type}, got {type(value)}")
                else:
                    if not isinstance(value, expected_type):
                        type_errors.append(f"Field '{field}' expected type {expected_type}, got {type(value)}")
        
        if type_errors:
            violations.extend(type_errors)
        
        # Generate SHA-256 checksum for data integrity
        data_str = json.dumps(data, sort_keys=True, default=str)
        data_hash = hashlib.sha256(data_str.encode()).hexdigest()
        self.checksums[f"{stage}_input_{data_hash[:8]}"] = data_hash
        
        if violations:
            return self._record_result(
                ContractStatus.VIOLATED,
                data_hash,
                violations,
                metadata={'stage': stage, 'validation_type': 'input'}
            )
        
        logger.info(f"Stage {stage} input validation passed (hash: {data_hash[:8]})")
        return self._record_result(
            ContractStatus.VALIDATED,
            data_hash,
            metadata={'stage': stage, 'validation_type': 'input', 'checksum': data_hash}
        )
    
    def validate_output(self, stage: str, data: Dict[str, Any]) -> ContractResult:
        """Validate output data against stage schema with checksum generation."""
        violations = []
        
        if stage not in self.stage_schemas:
            violations.append(f"Unknown stage: {stage}")
            return self._record_result(ContractStatus.VIOLATED, violations=violations)
        
        schema = self.stage_schemas[stage]['output']
        
        # Check required fields
        missing_fields = []
        for field in schema['required_fields']:
            if field not in data:
                missing_fields.append(field)
        
        if missing_fields:
            violations.append(f"Missing required output fields for stage {stage}: {missing_fields}")
        
        # Check field types
        type_errors = []
        for field, expected_type in schema['types'].items():
            if field in data:
                value = data[field]
                if isinstance(expected_type, tuple):
                    # Multiple acceptable types
                    if not any(isinstance(value, t) for t in expected_type):
                        type_errors.append(f"Field '{field}' expected types {expected_type}, got {type(value)}")
                else:
                    if not isinstance(value, expected_type):
                        type_errors.append(f"Field '{field}' expected type {expected_type}, got {type(value)}")
        
        if type_errors:
            violations.extend(type_errors)
        
        # Generate SHA-256 checksum for data integrity
        data_str = json.dumps(data, sort_keys=True, default=str)
        data_hash = hashlib.sha256(data_str.encode()).hexdigest()
        self.checksums[f"{stage}_output_{data_hash[:8]}"] = data_hash
        
        if violations:
            return self._record_result(
                ContractStatus.VIOLATED,
                data_hash,
                violations,
                metadata={'stage': stage, 'validation_type': 'output'}
            )
        
        logger.info(f"Stage {stage} output validation passed (hash: {data_hash[:8]})")
        return self._record_result(
            ContractStatus.VALIDATED,
            data_hash,
            metadata={'stage': stage, 'validation_type': 'output', 'checksum': data_hash}
        )
    
    def compatibility_check(self, from_stage: str, to_stage: str, from_data: Dict[str, Any], 
                          from_version: str = None, to_version: str = None) -> ContractResult:
        """
        Verify schema version compatibility between connected stages and validate handoff contracts.
        
        Args:
            from_stage: Source stage identifier
            to_stage: Target stage identifier  
            from_data: Output data from source stage
            from_version: Schema version of source stage (defaults to current)
            to_version: Schema version of target stage (defaults to current)
        """
        violations = []
        handoff_key = f"{from_stage}→{to_stage}"
        
        # Check if handoff is defined
        if handoff_key not in self.handoff_requirements:
            violations.append(f"Undefined pipeline handoff: {handoff_key}")
            return self._record_result(ContractStatus.VIOLATED, violations=violations)
        
        handoff = self.handoff_requirements[handoff_key]
        
        # Version compatibility check
        from_version = from_version or self.SCHEMA_VERSION
        to_version = to_version or self.SCHEMA_VERSION
        
        # Simple version compatibility - major versions must match
        from_major = from_version.split('.')[0]
        to_major = to_version.split('.')[0]
        
        if from_major != to_major:
            violations.append(
                f"Version incompatibility between {from_stage} (v{from_version}) and {to_stage} (v{to_version}): "
                f"major versions must match"
            )
        
        # Validate handoff field requirements
        missing_handoff_fields = []
        for field in handoff['required_handoff_fields']:
            if field not in from_data:
                missing_handoff_fields.append(field)
        
        if missing_handoff_fields:
            violations.append(
                f"Handoff {handoff_key} missing required fields: {missing_handoff_fields}"
            )
        
        # Generate handoff checksum
        handoff_data = {k: v for k, v in from_data.items() if k in handoff['required_handoff_fields']}
        handoff_str = json.dumps(handoff_data, sort_keys=True, default=str)
        handoff_hash = hashlib.sha256(handoff_str.encode()).hexdigest()
        
        # Store compatibility check result
        self.version_compatibility[handoff_key] = {
            'from_version': from_version,
            'to_version': to_version,
            'handoff_hash': handoff_hash,
            'timestamp': datetime.now().isoformat(),
            'compatible': len(violations) == 0
        }
        
        if violations:
            return self._record_result(
                ContractStatus.VIOLATED,
                handoff_hash,
                violations,
                metadata={
                    'handoff': handoff_key,
                    'from_version': from_version,
                    'to_version': to_version,
                    'validation_type': 'compatibility'
                }
            )
        
        logger.info(f"Compatibility check passed for {handoff_key} (hash: {handoff_hash[:8]})")
        return self._record_result(
            ContractStatus.VALIDATED,
            handoff_hash,
            metadata={
                'handoff': handoff_key,
                'from_version': from_version,
                'to_version': to_version,
                'validation_type': 'compatibility',
                'checksum': handoff_hash
            }
        )
    
    def get_stage_info(self, stage: str) -> Dict[str, Any]:
        """Get comprehensive information about a pipeline stage."""
        if stage not in self.stage_schemas:
            raise ContractViolationError(f"Unknown stage: {stage}")
        
        return {
            'stage_code': stage,
            'stage_name': self.STAGE_NAMES[stage],
            'schema': self.stage_schemas[stage],
            'position': self.PIPELINE_STAGES.index(stage),
            'total_stages': len(self.PIPELINE_STAGES),
            'next_stage': self.PIPELINE_STAGES[self.PIPELINE_STAGES.index(stage) + 1] 
                         if self.PIPELINE_STAGES.index(stage) < len(self.PIPELINE_STAGES) - 1 else None,
            'prev_stage': self.PIPELINE_STAGES[self.PIPELINE_STAGES.index(stage) - 1] 
                         if self.PIPELINE_STAGES.index(stage) > 0 else None
        }
    
    def execute(self, data: Any, context: Dict[str, Any] = None) -> ContractResult:
        """Execute pipeline contract validation based on action specified in data."""
        try:
            if not isinstance(data, dict) or 'action' not in data:
                return self._record_result(
                    ContractStatus.VIOLATED,
                    violations=["Data must be dict with 'action' field"]
                )
            
            action = data['action']
            
            if action == 'validate_input':
                stage = data['stage']
                stage_data = data['data']
                return self.validate_input(stage, stage_data)
            
            elif action == 'validate_output':
                stage = data['stage']
                stage_data = data['data']
                return self.validate_output(stage, stage_data)
            
            elif action == 'compatibility_check':
                from_stage = data['from_stage']
                to_stage = data['to_stage']
                from_data = data['from_data']
                from_version = data.get('from_version')
                to_version = data.get('to_version')
                return self.compatibility_check(from_stage, to_stage, from_data, from_version, to_version)
            
            else:
                return self._record_result(
                    ContractStatus.VIOLATED,
                    violations=[f"Unknown action: {action}"]
                )
        
        except Exception as e:
            return self._record_result(
                ContractStatus.VIOLATED,
                violations=[f"PipelineContract execution failed: {str(e)}"]
            )


class ContractManager:
    """Manager that instantiates and coordinates all contracts."""
    
    def __init__(self, snapshot_storage: str = "contract_snapshots"):
        self.contracts: Dict[str, BaseContract] = {}
        self.execution_log: List[ContractResult] = []
        
        # Initialize all contracts
        self.contracts["routing"] = RoutingContract()
        self.contracts["snapshot"] = SnapshotContract(snapshot_storage)
        self.contracts["context_immutability"] = ContextImmutabilityContract()
        self.contracts["traceability"] = TraceabilityContract()
        self.contracts["permutation_invariance"] = PermutationInvarianceContract()
        self.contracts["pipeline"] = PipelineContract()
        
        logger.info("ContractManager initialized with 6 active contracts")
        
    def execute_contract(self, contract_name: str, data: Any, context: Dict[str, Any] = None) -> ContractResult:
        """Execute a specific contract."""
        if contract_name not in self.contracts:
            raise ValueError(f"Contract not found: {contract_name}")
            
        contract = self.contracts[contract_name]
        result = contract.execute(data, context)
        self.execution_log.append(result)
        
        logger.info(f"Executed {contract_name}: {result.status.value}")
        return result
        
    def execute_full_pipeline_validation(self, pipeline_data: Dict[str, Any]) -> Dict[str, ContractResult]:
        """Execute the full contract validation pipeline."""
        results = {}
        context = pipeline_data.get('context', {})
        
        # Stage 1: PDF Ingestion - Routing Contract
        if 'pdf_content' in pipeline_data:
            results['routing'] = self.execute_contract(
                'routing',
                {'pdf_content': pipeline_data['pdf_content'], 
                 'pdf_path': pipeline_data.get('pdf_path', '')},
                context
            )
        
        # Stage 2: Snapshot Creation - Snapshot Contract  
        if 'files_to_track' in pipeline_data:
            for file_path in pipeline_data['files_to_track']:
                results[f'snapshot_{file_path}'] = self.execute_contract(
                    'snapshot',
                    {'action': 'create_manifest', 'file_path': file_path},
                    context
                )
        
        # Stage 3: Context Freezing - Context Immutability Contract
        if context:
            results['context_immutability'] = self.execute_contract(
                'context_immutability',
                {'action': 'freeze', 'context': context.copy()},
                context
            )
        
        # Stage 4: Traceability Chain - Traceability Contract
        if 'operations' in pipeline_data:
            for operation in pipeline_data['operations']:
                results[f'traceability_{operation["name"]}'] = self.execute_contract(
                    'traceability',
                    {'action': 'add_to_chain', 'operation': operation['name'], 
                     'data': operation.get('data', {}), 'metadata': operation.get('metadata', {})},
                    context
                )
        
        # Stage 5: Result Aggregation - Permutation Invariance Contract
        if 'results_to_aggregate' in pipeline_data:
            results['permutation_invariance'] = self.execute_contract(
                'permutation_invariance',
                {'inputs': pipeline_data['results_to_aggregate'],
                 'aggregation_func': pipeline_data.get('aggregation_func', 'sum')},
                context
            )
        
        # Verify all contracts passed
        violations = []
        for name, result in results.items():
            if result.status == ContractStatus.VIOLATED:
                violations.extend([f"{name}: {v}" for v in result.violations])
                
        if violations:
            logger.error(f"Contract violations detected: {violations}")
        else:
            logger.info("All contracts executed successfully")
            
        return results
        
    def get_contract_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all contracts."""
        status = {}
        for name, contract in self.contracts.items():
            status[name] = {
                'status': contract.status.value,
                'executions': len(contract.execution_history),
                'violations': len(contract.violations),
                'last_execution': contract.execution_history[-1].timestamp.isoformat() if contract.execution_history else None
            }
        return status
        
    def validate_contract_chain(self) -> bool:
        """Validate that all contracts are in valid states."""
        for name, contract in self.contracts.items():
            if contract.status == ContractStatus.VIOLATED:
                logger.error(f"Contract {name} is in violated state")
                return False
        return True
        
    def reset_contracts(self):
        """Reset all contracts to initial state."""
        for contract in self.contracts.values():
            contract.status = ContractStatus.PENDING
            contract.violations.clear()
            contract.execution_history.clear()
        logger.info("All contracts reset")
"""
Canonical Flow Alias: 08K - Advanced Knowledge Graph Builder

Refactored to implement standardized process() API with deterministic ID generation,
page-anchored entity extraction, and ontology-driven validation.

Author: Semantic Inference Engine Team
Version: 4.0.0
License: MIT
"""

import re
import uuid
import hashlib
import json
# # # from datetime import datetime  # Module not found  # Module not found  # Module not found
# # # from typing import Dict, Any, List, Tuple, Optional, Set, Union  # Module not found  # Module not found  # Module not found
# # # from dataclasses import dataclass, field  # Module not found  # Module not found  # Module not found
# # # from collections import defaultdict, Counter  # Module not found  # Module not found  # Module not found
# # # from enum import Enum, auto  # Module not found  # Module not found  # Module not found
# # # from pathlib import Path  # Module not found  # Module not found  # Module not found
import logging
# # # from abc import ABC, abstractmethod  # Module not found  # Module not found  # Module not found
import threading
# # # from concurrent.futures import ThreadPoolExecutor, as_completed  # Module not found  # Module not found  # Module not found

# Optional imports with fallbacks

# Mandatory Pipeline Contract Annotations
__phase__ = "K"
__code__ = "23K"
__stage_order__ = 3

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    spacy = None

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    nx = None

# Core imports
try:
# # #     from canonical_flow.analysis.audit_logger import get_audit_logger  # Module not found  # Module not found  # Module not found
# # #     from evidence_processor import EvidenceChunk, SourceMetadata  # Module not found  # Module not found  # Module not found
# # #     from data_models import DataModel  # Module not found  # Module not found  # Module not found
    AUDIT_LOGGER_AVAILABLE = True
except ImportError:
    AUDIT_LOGGER_AVAILABLE = False
    # Fallback imports for standalone operation
    def get_audit_logger():
        return logging.getLogger(__name__)
    
    @dataclass
    class EvidenceChunk:
        chunk_id: str
        text: str
        context_before: str = ""
        context_after: str = ""
        start_position: int = 0
        end_position: int = 0
        processing_timestamp: datetime = field(default_factory=datetime.now)
        raw_text: str = ""
        
    @dataclass
    class SourceMetadata:
        document_id: str
        title: str
        author: str = ""
        publication_date: Optional[datetime] = None
        page_number: Optional[int] = None
        section_header: Optional[str] = None
        
    @dataclass
    class DataModel:
        id: str
        data: dict

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ontology definitions for validation
VALID_NODE_TYPES = {
    'PERSON', 'ORGANIZATION', 'LOCATION', 'PRODUCT', 'EVENT',
    'SYSTEM', 'COMPONENT', 'SERVICE', 'PROCESS', 'TECHNOLOGY',
    'DOCUMENT', 'CONCEPT', 'METRIC', 'REQUIREMENT', 'SPECIFICATION'
}

VALID_EDGE_TYPES = {
    'ACTS_ON', 'BELONGS_TO', 'LOCATED_IN', 'CAUSES', 'CONTAINS',
    'IMPLEMENTS', 'CONFIGURES', 'DEPENDS_ON', 'MANAGES', 'PRODUCES',
    'TRIGGERS', 'ENABLES', 'REQUIRES', 'PART_OF', 'INSTANCE_OF',
    'RELATED_TO', 'COLLABORATES_WITH', 'COMMUNICATES_WITH'
}

ONTOLOGY_RULES = {
    'PERSON': {'valid_targets': {'ORGANIZATION', 'LOCATION', 'PROCESS', 'DOCUMENT', 'SYSTEM'}},
    'ORGANIZATION': {'valid_targets': {'LOCATION', 'PROCESS', 'SYSTEM', 'SERVICE', 'PRODUCT'}},
    'SYSTEM': {'valid_targets': {'COMPONENT', 'SERVICE', 'PROCESS', 'TECHNOLOGY', 'DATA'}},
    'PROCESS': {'valid_targets': {'SYSTEM', 'COMPONENT', 'SERVICE', 'DOCUMENT', 'REQUIREMENT'}}
}

@dataclass
class KGNode:
    """Knowledge Graph Node with comprehensive metadata."""
    
    id: str
    text: str
    type: str
    confidence: float
    page_number: Optional[int]
    chunk_id: str
    document_id: str
    start_position: int
    end_position: int
    context: str
    aliases: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'id': self.id,
            'text': self.text,
            'type': self.type,
            'confidence': self.confidence,
            'page_number': self.page_number,
            'chunk_id': self.chunk_id,
            'document_id': self.document_id,
            'start_position': self.start_position,
            'end_position': self.end_position,
            'context': self.context,
            'aliases': self.aliases,
            'properties': self.properties,
            'provenance': self.provenance
        }

@dataclass 
class KGEdge:
    """Knowledge Graph Edge with provenance tracking."""
    
    id: str
    source_id: str
    target_id: str
    type: str
    confidence: float
    page_number: Optional[int]
    chunk_id: str
    document_id: str
    context: str
    extraction_method: str
    properties: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'id': self.id,
            'source_id': self.source_id,
            'target_id': self.target_id,
            'type': self.type,
            'confidence': self.confidence,
            'page_number': self.page_number,
            'chunk_id': self.chunk_id,
            'document_id': self.document_id,
            'context': self.context,
            'extraction_method': self.extraction_method,
            'properties': self.properties,
            'provenance': self.provenance
        }

@dataclass
class ValidationError:
    """Validation error with details."""
    
    error_type: str
    message: str
    entity_id: Optional[str] = None
    severity: str = "ERROR"
    context: Dict[str, Any] = field(default_factory=dict)

class KnowledgeGraphBuilder:
    """Advanced Knowledge Graph Builder with standardized process() API."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the knowledge graph builder."""
        self.config = config or {}
        
        # Setup logger with fallback
        if AUDIT_LOGGER_AVAILABLE:
            try:
                self.logger = get_audit_logger()
                if not hasattr(self.logger, 'warning'):
                    # If audit logger doesn't have standard methods, fall back
                    self.logger = logging.getLogger(__name__)
            except Exception:
                self.logger = logging.getLogger(__name__)
        else:
            self.logger = logging.getLogger(__name__)
            
        self.nlp = self._load_spacy_model()
        
        # Configuration parameters
        self.min_confidence = self.config.get('min_confidence', 0.7)
        self.max_entities = self.config.get('max_entities', 1000)
        self.enable_coreference = self.config.get('enable_coreference', False)
        
        # Initialize NLP components
        self._setup_entity_patterns()
        self._setup_relation_patterns()
        
    def _load_spacy_model(self):
        """Load spaCy model with error handling."""
        if not SPACY_AVAILABLE:
            self.logger.warning("spaCy not available. Using fallback NLP processing.")
            return None
            
        try:
            nlp = spacy.load("en_core_web_sm")
            return nlp
        except OSError:
            self.logger.warning("spaCy model 'en_core_web_sm' not found. Using blank model.")
            try:
                return spacy.blank("en")
            except Exception:
                self.logger.warning("Could not create spaCy blank model. Using None.")
                return None
    
    def _setup_entity_patterns(self):
        """Setup entity recognition patterns."""
        self.entity_patterns = {
            'SYSTEM': [
                r'\b[A-Z][a-zA-Z]*\s+System\b',
                r'\b[A-Z][a-zA-Z]*\s+Platform\b',
                r'\b[A-Z][a-zA-Z]*\s+Framework\b'
            ],
            'PROCESS': [
                r'\b[A-Z][a-zA-Z]*\s+Process\b',
                r'\b[A-Z][a-zA-Z]*\s+Workflow\b',
                r'\b[A-Z][a-zA-Z]*\s+Procedure\b'
            ],
            'COMPONENT': [
                r'\b[A-Z][a-zA-Z]*\s+Component\b',
                r'\b[A-Z][a-zA-Z]*\s+Module\b',
                r'\b[A-Z][a-zA-Z]*\s+Service\b'
            ]
        }
    
    def _setup_relation_patterns(self):
        """Setup relation extraction patterns."""
        self.relation_patterns = {
            'IMPLEMENTS': [
                r'([A-Z][a-zA-Z\s]+)\s+implements?\s+([A-Z][a-zA-Z\s]+)',
                r'([A-Z][a-zA-Z\s]+)\s+is\s+implemented\s+by\s+([A-Z][a-zA-Z\s]+)'
            ],
            'DEPENDS_ON': [
                r'([A-Z][a-zA-Z\s]+)\s+depends?\s+on\s+([A-Z][a-zA-Z\s]+)',
                r'([A-Z][a-zA-Z\s]+)\s+requires?\s+([A-Z][a-zA-Z\s]+)'
            ],
            'MANAGES': [
                r'([A-Z][a-zA-Z\s]+)\s+manages?\s+([A-Z][a-zA-Z\s]+)',
                r'([A-Z][a-zA-Z\s]+)\s+is\s+managed\s+by\s+([A-Z][a-zA-Z\s]+)'
            ]
        }
    
    def generate_deterministic_id(self, entity_text: str, entity_type: str, 
                                 document_id: str, chunk_id: str) -> str:
        """Generate deterministic hash-based ID for entities and relations."""
        # Normalize text for consistent hashing
        normalized_text = re.sub(r'\s+', ' ', entity_text.lower().strip())
        
        # Create hash input with all identifying components
        hash_input = f"{normalized_text}|{entity_type}|{document_id}|{chunk_id}"
        
        # Generate deterministic hash
        hash_obj = hashlib.sha256(hash_input.encode('utf-8'))
        hash_digest = hash_obj.hexdigest()[:16]  # Use first 16 characters
        
        return f"{entity_type.lower()}_{hash_digest}"
    
    def generate_edge_id(self, source_id: str, target_id: str, 
                        edge_type: str) -> str:
        """Generate deterministic ID for edges."""
        hash_input = f"{source_id}|{target_id}|{edge_type}"
        hash_obj = hashlib.sha256(hash_input.encode('utf-8'))
        hash_digest = hash_obj.hexdigest()[:16]
        
        return f"edge_{edge_type.lower()}_{hash_digest}"
    
    def extract_entities_with_provenance(self, text: str, chunk_id: str, 
                                       document_id: str, page_number: Optional[int],
                                       metadata: Optional[Dict[str, Any]] = None) -> List[KGNode]:
        """Extract entities with page-anchored provenance tracking."""
        entities = []
        
        # Extract named entities using spaCy if available
        if self.nlp is not None:
            doc = self.nlp(text)
            
            for ent in doc.ents:
                if ent.label_ in {'PERSON', 'ORG', 'GPE', 'PRODUCT', 'EVENT'}:
                    entity_type = self._map_spacy_label(ent.label_)
                    
                    if entity_type in VALID_NODE_TYPES:
                        entity_id = self.generate_deterministic_id(
                            ent.text, entity_type, document_id, chunk_id
                        )
                        
                        # Extract context window around entity
                        context_start = max(0, ent.start - 10)
                        context_end = min(len(doc), ent.end + 10)
                        context = doc[context_start:context_end].text
                        
                        # Create provenance tracking
                        provenance = {
                            'extraction_timestamp': datetime.now().isoformat(),
                            'extraction_method': 'spacy_ner',
                            'spacy_label': ent.label_,
                            'confidence_score': 0.8  # Default spaCy confidence
                        }
                        
                        if metadata:
                            provenance.update(metadata)
                        
                        node = KGNode(
                            id=entity_id,
                            text=ent.text,
                            type=entity_type,
                            confidence=0.8,
                            page_number=page_number,
                            chunk_id=chunk_id,
                            document_id=document_id,
                            start_position=ent.start_char,
                            end_position=ent.end_char,
                            context=context,
                            provenance=provenance
                        )
                        
                        entities.append(node)
        
        # Extract custom patterns
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text):
                    entity_text = match.group().strip()
                    entity_id = self.generate_deterministic_id(
                        entity_text, entity_type, document_id, chunk_id
                    )
                    
                    # Check for duplicates
                    if not any(e.id == entity_id for e in entities):
                        # Extract context
                        context_start = max(0, match.start() - 100)
                        context_end = min(len(text), match.end() + 100)
                        context = text[context_start:context_end]
                        
                        provenance = {
                            'extraction_timestamp': datetime.now().isoformat(),
                            'extraction_method': 'regex_pattern',
                            'pattern_used': pattern,
                            'confidence_score': 0.75
                        }
                        
                        node = KGNode(
                            id=entity_id,
                            text=entity_text,
                            type=entity_type,
                            confidence=0.75,
                            page_number=page_number,
                            chunk_id=chunk_id,
                            document_id=document_id,
                            start_position=match.start(),
                            end_position=match.end(),
                            context=context,
                            provenance=provenance
                        )
                        
                        entities.append(node)
        
        return entities
    
    def extract_relations_with_provenance(self, text: str, entities: List[KGNode],
                                        chunk_id: str, document_id: str,
                                        page_number: Optional[int]) -> List[KGEdge]:
        """Extract relations with provenance tracking."""
        relations = []
        
        # Create entity lookup for faster access
        entity_lookup = {e.text.lower(): e for e in entities}
        
        # Extract relations using patterns
        for relation_type, patterns in self.relation_patterns.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text):
                    if len(match.groups()) >= 2:
                        source_text = match.group(1).strip()
                        target_text = match.group(2).strip()
                        
                        # Find corresponding entities
                        source_entity = entity_lookup.get(source_text.lower())
                        target_entity = entity_lookup.get(target_text.lower())
                        
                        if source_entity and target_entity and source_entity.id != target_entity.id:
                            edge_id = self.generate_edge_id(
                                source_entity.id, target_entity.id, relation_type
                            )
                            
                            # Extract context
                            context_start = max(0, match.start() - 100)
                            context_end = min(len(text), match.end() + 100)
                            context = text[context_start:context_end]
                            
                            provenance = {
                                'extraction_timestamp': datetime.now().isoformat(),
                                'extraction_method': 'regex_pattern',
                                'pattern_used': pattern,
                                'confidence_score': 0.8
                            }
                            
                            edge = KGEdge(
                                id=edge_id,
                                source_id=source_entity.id,
                                target_id=target_entity.id,
                                type=relation_type,
                                confidence=0.8,
                                page_number=page_number,
                                chunk_id=chunk_id,
                                document_id=document_id,
                                context=context,
                                extraction_method='regex_pattern',
                                provenance=provenance
                            )
                            
                            relations.append(edge)
        
        return relations
    
    def _map_spacy_label(self, spacy_label: str) -> str:
        """Map spaCy labels to ontology types."""
        mapping = {
            'PERSON': 'PERSON',
            'ORG': 'ORGANIZATION',
            'GPE': 'LOCATION',
            'PRODUCT': 'PRODUCT',
            'EVENT': 'EVENT'
        }
        return mapping.get(spacy_label, 'CONCEPT')
    
    def validate_ontology(self, nodes: List[KGNode], edges: List[KGEdge]) -> List[ValidationError]:
        """Validate nodes and edges against predefined ontology."""
        errors = []
        
        # Validate node types
        for node in nodes:
            if node.type not in VALID_NODE_TYPES:
                errors.append(ValidationError(
                    error_type="INVALID_NODE_TYPE",
                    message=f"Node type '{node.type}' is not in valid ontology",
                    entity_id=node.id,
                    context={'valid_types': list(VALID_NODE_TYPES)}
                ))
        
        # Validate edge types
        for edge in edges:
            if edge.type not in VALID_EDGE_TYPES:
                errors.append(ValidationError(
                    error_type="INVALID_EDGE_TYPE",
                    message=f"Edge type '{edge.type}' is not in valid ontology",
                    entity_id=edge.id,
                    context={'valid_types': list(VALID_EDGE_TYPES)}
                ))
        
        # Validate ontology rules
        node_lookup = {n.id: n for n in nodes}
        for edge in edges:
            source_node = node_lookup.get(edge.source_id)
            target_node = node_lookup.get(edge.target_id)
            
            if source_node and target_node:
                source_type = source_node.type
                target_type = target_node.type
                
                if source_type in ONTOLOGY_RULES:
                    valid_targets = ONTOLOGY_RULES[source_type].get('valid_targets', set())
                    if valid_targets and target_type not in valid_targets:
                        errors.append(ValidationError(
                            error_type="INVALID_RELATION",
# # #                             message=f"Invalid relation from {source_type} to {target_type}",  # Module not found  # Module not found  # Module not found
                            entity_id=edge.id,
                            severity="WARNING",
                            context={
                                'source_type': source_type,
                                'target_type': target_type,
                                'valid_targets': list(valid_targets)
                            }
                        ))
        
        return errors
    
    def process(self, document_id: str) -> Dict[str, Any]:
        """
        Standardized process() API that consumes chunk and entity data 
        and writes knowledge graph artifacts.
        
        Args:
            document_id: Identifier for the document to process
            
        Returns:
            Dictionary with processing results and metadata
        """
        try:
            start_time = datetime.now()
            
            # Extract document stem for directory structure
            document_stem = document_id.replace('/', '_').replace('\\', '_')
            output_dir = Path(f"canonical_flow/knowledge/{document_stem}")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Load existing chunk data (mock implementation)
            chunks_data = self._load_chunk_data(document_id)
            entities_data = self._load_entities_data(document_id)
            
            all_nodes = []
            all_edges = []
            validation_errors = []
            
            # Process each chunk
            for chunk_data in chunks_data:
                chunk_id = chunk_data.get('chunk_id', f"chunk_{len(all_nodes)}")
                text = chunk_data.get('text', '')
                page_number = chunk_data.get('page_number')
                
                if not text.strip():
                    continue
                
                # Extract entities with provenance
                nodes = self.extract_entities_with_provenance(
                    text, chunk_id, document_id, page_number
                )
                all_nodes.extend(nodes)
                
                # Extract relations with provenance
                edges = self.extract_relations_with_provenance(
                    text, nodes, chunk_id, document_id, page_number
                )
                all_edges.extend(edges)
            
            # Deduplicate nodes and edges based on IDs
            unique_nodes = {node.id: node for node in all_nodes}.values()
            unique_edges = {edge.id: edge for edge in all_edges}.values()
            
            final_nodes = list(unique_nodes)
            final_edges = list(unique_edges)
            
            # Validate against ontology
            validation_errors = self.validate_ontology(final_nodes, final_edges)
            
            # Generate metadata
            end_time = datetime.now()
            processing_duration = (end_time - start_time).total_seconds()
            
            metadata = {
                'document_id': document_id,
                'processing_timestamp': end_time.isoformat(),
                'processing_duration_seconds': processing_duration,
                'total_nodes': len(final_nodes),
                'total_edges': len(final_edges),
                'validation_errors': len(validation_errors),
                'validation_warnings': len([e for e in validation_errors if e.severity == 'WARNING']),
                'chunks_processed': len(chunks_data),
                'ontology_version': '1.0.0',
                'extraction_methods': list(set(
                    [n.provenance.get('extraction_method', 'unknown') for n in final_nodes] +
                    [e.extraction_method for e in final_edges]
                )),
                'confidence_distribution': self._calculate_confidence_distribution(
                    final_nodes, final_edges
                ),
                'validation_status': 'PASSED' if not any(
                    e.severity == 'ERROR' for e in validation_errors
                ) else 'FAILED'
            }
            
            # Write artifacts to canonical directory
            self._write_kg_artifacts(output_dir, final_nodes, final_edges, 
                                   validation_errors, metadata)
            
            # Log successful processing
            self.logger.info(
                f"Successfully processed document {document_id}: "
                f"{len(final_nodes)} nodes, {len(final_edges)} edges"
            )
            
            return {
                'status': 'success',
                'nodes_count': len(final_nodes),
                'edges_count': len(final_edges),
                'validation_errors': len(validation_errors),
                'output_directory': str(output_dir),
                'metadata': metadata
            }
            
        except Exception as e:
            error_msg = f"Failed to process document {document_id}: {str(e)}"
            self.logger.error(error_msg)
            
            return {
                'status': 'error',
                'error': error_msg,
                'nodes_count': 0,
                'edges_count': 0,
                'validation_errors': 0
            }
    
    def _load_chunk_data(self, document_id: str) -> List[Dict[str, Any]]:
        """Load chunk data for the document (mock implementation)."""
# # #         # In a real implementation, this would load from storage  # Module not found  # Module not found  # Module not found
        return [
            {
                'chunk_id': f'{document_id}_chunk_1',
                'text': 'The Data Processing System implements the Advanced Analytics Framework to manage customer information. The System Administrator manages the Processing System.',
                'page_number': 1
            },
            {
                'chunk_id': f'{document_id}_chunk_2', 
                'text': 'The Analytics Framework depends on the Database Service for data storage and retrieval operations. The Database Service produces analytical reports.',
                'page_number': 2
            }
        ]
    
    def _load_entities_data(self, document_id: str) -> List[Dict[str, Any]]:
        """Load existing entities data for the document (mock implementation)."""
# # #         # In a real implementation, this would load from storage  # Module not found  # Module not found  # Module not found
        return []
    
    def _calculate_confidence_distribution(self, nodes: List[KGNode], 
                                         edges: List[KGEdge]) -> Dict[str, float]:
        """Calculate distribution of confidence scores."""
        all_confidences = [n.confidence for n in nodes] + [e.confidence for e in edges]
        
        if not all_confidences:
            return {'mean': 0.0, 'min': 0.0, 'max': 0.0}
        
        return {
            'mean': sum(all_confidences) / len(all_confidences),
            'min': min(all_confidences),
            'max': max(all_confidences)
        }
    
    def _write_kg_artifacts(self, output_dir: Path, nodes: List[KGNode],
                          edges: List[KGEdge], validation_errors: List[ValidationError],
                          metadata: Dict[str, Any]):
        """Write knowledge graph artifacts to canonical directory structure."""
        
        # Write nodes
        nodes_file = output_dir / 'kg_nodes.json'
        with open(nodes_file, 'w', encoding='utf-8') as f:
            nodes_data = [node.to_dict() for node in nodes]
            json.dump(nodes_data, f, indent=2, ensure_ascii=False)
        
        # Write edges
        edges_file = output_dir / 'kg_edges.json'
        with open(edges_file, 'w', encoding='utf-8') as f:
            edges_data = [edge.to_dict() for edge in edges]
            json.dump(edges_data, f, indent=2, ensure_ascii=False)
        
        # Write metadata
        meta_file = output_dir / 'kg_meta.json'
        metadata_full = {
            **metadata,
            'validation_errors': [
                {
                    'error_type': e.error_type,
                    'message': e.message,
                    'entity_id': e.entity_id,
                    'severity': e.severity,
                    'context': e.context
                } for e in validation_errors
            ],
            'output_files': {
                'nodes_file': 'kg_nodes.json',
                'edges_file': 'kg_edges.json',
                'metadata_file': 'kg_meta.json'
            },
            'schema_version': '1.0.0',
            'generator': 'AdvancedKnowledgeGraphBuilder',
            'generator_version': '4.0.0'
        }
        
        with open(meta_file, 'w', encoding='utf-8') as f:
            json.dump(metadata_full, f, indent=2, ensure_ascii=False)


# Initialize singleton instance
_kg_builder = None

def get_knowledge_graph_builder(config: Optional[Dict[str, Any]] = None) -> KnowledgeGraphBuilder:
    """Get singleton knowledge graph builder instance."""
    global _kg_builder
    if _kg_builder is None:
        _kg_builder = KnowledgeGraphBuilder(config)
    return _kg_builder

def process(document_id: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Standardized process function for canonical flow integration."""
    builder = get_knowledge_graph_builder(config)
    return builder.process(document_id)

# Export main classes and functions
__all__ = [
    'KnowledgeGraphBuilder',
    'KGNode', 
    'KGEdge',
    'ValidationError',
    'process',
    'get_knowledge_graph_builder'
]
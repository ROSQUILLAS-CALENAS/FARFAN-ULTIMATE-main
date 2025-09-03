"""
Canonical Flow Alias: 13K
Chunking Processor with Deterministic Chunk Generation

# # # Implements standardized process() API to consume page-anchored text from ingestion artifacts  # Module not found  # Module not found  # Module not found
and generate deterministic chunks with stable IDs based on content hashing and position.

Stage: K_knowledge_extraction
Code: 13K
"""

import hashlib
import json
import logging
import re
import sys
# # # from dataclasses import dataclass, field  # Module not found  # Module not found  # Module not found
# # # from enum import Enum  # Module not found  # Module not found  # Module not found
# # # from pathlib import Path  # Module not found  # Module not found  # Module not found
# # # from typing import Any, Dict, List, Optional, Union, Tuple  # Module not found  # Module not found  # Module not found
# # # from collections import OrderedDict  # Module not found  # Module not found  # Module not found

# Import total ordering base

# Mandatory Pipeline Contract Annotations
__phase__ = "K"
__code__ = "24K"
__stage_order__ = 3

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
# # # from total_ordering_base import TotalOrderingBase, DeterministicCollectionMixin  # Module not found  # Module not found  # Module not found

logger = logging.getLogger(__name__)


class PartitioningPolicy(Enum):
    """Chunking strategies for text partitioning."""
    SENTENCE_BOUNDARIES = "sentence_boundaries"
    PARAGRAPH_BREAKS = "paragraph_breaks" 
    FIXED_TOKEN_COUNTS = "fixed_token_counts"
    HYBRID = "hybrid"


@dataclass
class ChunkConfig:
    """Configuration for chunking behavior."""
    policy: PartitioningPolicy = PartitioningPolicy.SENTENCE_BOUNDARIES
    max_tokens: int = 512
    min_tokens: int = 50
    overlap_tokens: int = 0
    overlap_ratio: float = 0.1
    preserve_boundaries: bool = True
    context_window: int = 2
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if self.min_tokens <= 0:
            raise ValueError("min_tokens must be positive") 
        if self.min_tokens >= self.max_tokens:
            raise ValueError("min_tokens must be less than max_tokens")
        if not 0 <= self.overlap_ratio <= 1:
            raise ValueError("overlap_ratio must be between 0 and 1")


@dataclass
class PageAnchor:
    """Page anchor information for provenance tracking."""
    page_number: int
    start_char: int
    end_char: int
    page_text_hash: Optional[str] = None
    
    def __post_init__(self):
        if self.start_char < 0:
            raise ValueError("start_char must be non-negative")
        if self.end_char < self.start_char:
            raise ValueError("end_char must be >= start_char")


@dataclass
class ChunkProvenance:
    """Provenance metadata for chunks."""
    source_document: str
    document_stem: str
    page_anchors: List[PageAnchor] = field(default_factory=list)
    character_offsets: Tuple[int, int] = (0, 0)
    section_path: List[str] = field(default_factory=list)
    processing_timestamp: str = ""
    source_hash: str = ""
    
    def __post_init__(self):
        # Ensure deterministic ordering
        self.page_anchors = sorted(self.page_anchors, key=lambda x: (x.page_number, x.start_char))
        self.section_path = list(self.section_path)  # Ensure mutable list


@dataclass
class ProcessedChunk:
    """A processed text chunk with full metadata."""
    chunk_id: str
    content: str
    page_anchors: List[PageAnchor]
    metadata: Dict[str, Any]
    provenance: ChunkProvenance
    token_count: int = 0
    sentence_count: int = 0
    paragraph_count: int = 0
    
    def __post_init__(self):
        # Ensure deterministic ordering of page anchors
        self.page_anchors = sorted(self.page_anchors, key=lambda x: (x.page_number, x.start_char))
        
        # Calculate basic statistics if not provided
        if self.token_count == 0:
            self.token_count = self._count_tokens(self.content)
        if self.sentence_count == 0:
            self.sentence_count = self._count_sentences(self.content)
        if self.paragraph_count == 0:
            self.paragraph_count = self._count_paragraphs(self.content)
    
    def _count_tokens(self, text: str) -> int:
        """Simple token counting by whitespace splitting."""
        return len(text.split())
    
    def _count_sentences(self, text: str) -> int:
        """Count sentences using basic punctuation."""
        sentences = re.split(r'[.!?]+', text.strip())
        return len([s for s in sentences if s.strip()])
    
    def _count_paragraphs(self, text: str) -> int:
        """Count paragraphs by line breaks."""
        paragraphs = text.split('\n\n')
        return len([p for p in paragraphs if p.strip()])


class ChunkingProcessor(TotalOrderingBase, DeterministicCollectionMixin):
    """
    Deterministic text chunking processor with standardized process() API.
    
# # #     Consumes page-anchored text from ingestion artifacts and generates   # Module not found  # Module not found  # Module not found
    deterministic chunks with stable IDs based on content hashing and position.
    """
    
    def __init__(self, config: Optional[ChunkConfig] = None):
        """
        Initialize chunking processor with configuration.
        
        Args:
            config: Chunking configuration (uses defaults if None)
        """
        super().__init__("ChunkingProcessor")
        
        self.config = config or ChunkConfig()
        self.processing_stats = {
            "documents_processed": 0,
            "chunks_generated": 0,
            "total_tokens": 0,
            "average_chunk_size": 0.0,
            "errors_encountered": 0,
        }
        
        # Regex patterns for text processing (compiled for performance)
        self._sentence_pattern = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
        self._paragraph_pattern = re.compile(r'\n\s*\n')
        self._token_pattern = re.compile(r'\b\w+\b')
        
        # Update state hash
        self.update_state_hash(self._get_initial_state())
    
    def _get_initial_state(self) -> Dict[str, Any]:
        """Get initial state for hash calculation."""
        return {
            "config": {
                "policy": self.config.policy.value,
                "max_tokens": self.config.max_tokens,
                "min_tokens": self.config.min_tokens,
                "overlap_tokens": self.config.overlap_tokens,
                "overlap_ratio": self.config.overlap_ratio,
                "preserve_boundaries": self.config.preserve_boundaries,
                "context_window": self.config.context_window,
            }
        }
    
    def process(self, data: Any = None, context: Any = None) -> Dict[str, Any]:
        """
        Main processing function implementing standardized process() API.
        
        Args:
            data: Input data containing page-anchored text or ingestion artifacts
            context: Processing context
            
        Returns:
            Deterministic chunking results with chunk_id, content, page_anchors, metadata
        """
        operation_id = self.generate_operation_id("process", {"data": data, "context": context})
        
        try:
            # Canonicalize inputs for deterministic processing
            canonical_data = self.canonicalize_data(data) if data else {}
            canonical_context = self.canonicalize_data(context) if context else {}
            
            # Extract document information
            document_info = self._extract_document_info(canonical_data)
            if not document_info:
                return self._generate_error_result("No valid document data found", operation_id)
            
            # Generate chunks deterministically
            chunks = self._generate_chunks_deterministic(document_info)
            
            # Generate output artifacts
            output_artifacts = self._generate_output_artifacts(chunks, document_info)
            
            # Create final result
            result = {
                "component": self.component_name,
                "operation_id": operation_id,
                "status": "success",
                "document_stem": document_info.get("document_stem", "unknown"),
                "chunks_generated": len(chunks),
                "artifacts": output_artifacts,
                "processing_stats": self._update_processing_stats(chunks),
                "timestamp": self._get_deterministic_timestamp(),
            }
            
            # Update state hash
            self.update_state_hash(result)
            
            return self.sort_dict_by_keys(result)
            
        except Exception as e:
            logger.error(f"Error in chunking processor: {str(e)}")
            self.processing_stats["errors_encountered"] += 1
            return self._generate_error_result(str(e), operation_id)
    
    def _extract_document_info(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
# # #         Extract document information from input data.  # Module not found  # Module not found  # Module not found
        
        Args:
            data: Input data dictionary
            
        Returns:
            Document information or None if invalid
        """
        # Handle direct ingestion bundle format
        if "document_content" in data and "document_stem" in data:
            return {
                "document_stem": data["document_stem"],
                "content": self._extract_text_content(data["document_content"]),
                "metadata": data.get("content_metadata", {}),
                "source_hash": data.get("document_hash", ""),
                "bundle_id": data.get("bundle_id", ""),
            }
        
        # Handle direct content format
        if "content" in data:
            document_stem = data.get("document_stem", "unknown")
            if not document_stem or document_stem == "unknown":
# # #                 # Generate document stem from content hash  # Module not found  # Module not found  # Module not found
                content_hash = hashlib.sha256(str(data["content"]).encode()).hexdigest()[:16]
                document_stem = f"doc_{content_hash}"
            
            return {
                "document_stem": document_stem,
                "content": data["content"],
                "metadata": data.get("metadata", {}),
                "source_hash": hashlib.sha256(str(data["content"]).encode()).hexdigest(),
                "bundle_id": data.get("bundle_id", ""),
            }
        
        # Handle text-only input
        if "text" in data:
            text_hash = hashlib.sha256(data["text"].encode()).hexdigest()[:16]
            return {
                "document_stem": f"text_{text_hash}",
                "content": data["text"],
                "metadata": data.get("metadata", {}),
                "source_hash": hashlib.sha256(data["text"].encode()).hexdigest(),
                "bundle_id": "",
            }
        
        return None
    
    def _extract_text_content(self, document_content: Any) -> str:
        """
# # #         Extract text content from document content structure.  # Module not found  # Module not found  # Module not found
        
        Args:
            document_content: Document content structure
            
        Returns:
            Extracted text content
        """
        if isinstance(document_content, str):
            return document_content
        
        if isinstance(document_content, dict):
            text_parts = []
            
            # Add title if present
            if "title" in document_content:
                text_parts.append(str(document_content["title"]))
            
            # Add main content if present
            if "content" in document_content:
                text_parts.append(str(document_content["content"]))
            
            # Add sections if present
            if "sections" in document_content and isinstance(document_content["sections"], list):
                for section in document_content["sections"]:
                    if isinstance(section, dict):
                        if "heading" in section:
                            text_parts.append(str(section["heading"]))
                        if "text" in section:
                            text_parts.append(str(section["text"]))
                    elif isinstance(section, str):
                        text_parts.append(section)
            
            return "\n\n".join(text_parts)
        
        return str(document_content)
    
    def _generate_chunks_deterministic(self, document_info: Dict[str, Any]) -> List[ProcessedChunk]:
        """
        Generate chunks using deterministic splitting logic.
        
        Args:
            document_info: Document information dictionary
            
        Returns:
            List of processed chunks with stable ordering
        """
        content = document_info["content"]
        document_stem = document_info["document_stem"]
        
        # Split text based on configured policy
        raw_chunks = self._split_text_by_policy(content)
        
        # Process each raw chunk
        processed_chunks = []
        for i, chunk_text in enumerate(raw_chunks):
            if not chunk_text.strip():
                continue
                
            # Generate stable chunk ID based on content and position
            chunk_data = {
                "content": chunk_text,
                "position": i,
                "document_stem": document_stem,
                "policy": self.config.policy.value,
            }
            chunk_id = self.generate_stable_id(chunk_data, prefix="chunk")
            
            # Create page anchors (simplified for this implementation)
            page_anchors = self._generate_page_anchors(chunk_text, content, i)
            
            # Create provenance metadata
            provenance = ChunkProvenance(
                source_document=document_stem,
                document_stem=document_stem,
                page_anchors=page_anchors,
                character_offsets=self._calculate_character_offsets(chunk_text, content),
                processing_timestamp=self._get_deterministic_timestamp(),
                source_hash=document_info["source_hash"],
            )
            
            # Create chunk metadata
            metadata = {
                "chunk_index": i,
                "total_chunks": len(raw_chunks),
                "policy": self.config.policy.value,
                "creation_timestamp": self._get_deterministic_timestamp(),
                "source_bundle_id": document_info["bundle_id"],
            }
            
            # Create processed chunk
            chunk = ProcessedChunk(
                chunk_id=chunk_id,
                content=chunk_text,
                page_anchors=page_anchors,
                metadata=metadata,
                provenance=provenance,
            )
            
            processed_chunks.append(chunk)
        
        # Sort chunks by position for stable ordering
        processed_chunks.sort(key=lambda x: x.metadata["chunk_index"])
        
        return processed_chunks
    
    def _split_text_by_policy(self, content: str) -> List[str]:
        """
        Split text according to the configured partitioning policy.
        
        Args:
            content: Input text content
            
        Returns:
            List of text chunks
        """
        if self.config.policy == PartitioningPolicy.SENTENCE_BOUNDARIES:
            return self._split_by_sentences(content)
        elif self.config.policy == PartitioningPolicy.PARAGRAPH_BREAKS:
            return self._split_by_paragraphs(content)
        elif self.config.policy == PartitioningPolicy.FIXED_TOKEN_COUNTS:
            return self._split_by_tokens(content)
        elif self.config.policy == PartitioningPolicy.HYBRID:
            return self._split_hybrid(content)
        else:
            # Fallback to sentence boundaries
            return self._split_by_sentences(content)
    
    def _split_by_sentences(self, content: str) -> List[str]:
        """Split text by sentence boundaries."""
        sentences = self._sentence_pattern.split(content.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Group sentences into chunks respecting max_tokens
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = len(self._token_pattern.findall(sentence))
            
            if (current_tokens + sentence_tokens > self.config.max_tokens and current_chunk):
                # Complete current chunk
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_tokens = sentence_tokens
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
        
        # Add final chunk if any
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return self._apply_overlap(chunks)
    
    def _split_by_paragraphs(self, content: str) -> List[str]:
        """Split text by paragraph breaks."""
        paragraphs = self._paragraph_pattern.split(content.strip())
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        # Group paragraphs into chunks respecting max_tokens
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for paragraph in paragraphs:
            paragraph_tokens = len(self._token_pattern.findall(paragraph))
            
            if (current_tokens + paragraph_tokens > self.config.max_tokens and current_chunk):
                # Complete current chunk
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = [paragraph]
                current_tokens = paragraph_tokens
            else:
                current_chunk.append(paragraph)
                current_tokens += paragraph_tokens
        
        # Add final chunk if any
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        return self._apply_overlap(chunks)
    
    def _split_by_tokens(self, content: str) -> List[str]:
        """Split text by fixed token counts."""
        tokens = self._token_pattern.findall(content)
        
        if not tokens:
            return [content] if content.strip() else []
        
        chunks = []
        
        # Create chunks of fixed token size
        for i in range(0, len(tokens), self.config.max_tokens):
            chunk_tokens = tokens[i:i + self.config.max_tokens]
            
            # Reconstruct text preserving original spacing (simplified)
            chunk_text = ' '.join(chunk_tokens)
            
            if len(chunk_tokens) >= self.config.min_tokens or not chunks:
                chunks.append(chunk_text)
        
        return self._apply_overlap(chunks)
    
    def _split_hybrid(self, content: str) -> List[str]:
        """Split using hybrid approach combining multiple policies."""
        # First split by paragraphs
        paragraphs = self._split_by_paragraphs(content)
        
        # Then refine by sentences if paragraphs are too large
        refined_chunks = []
        for paragraph in paragraphs:
            tokens = len(self._token_pattern.findall(paragraph))
            
            if tokens > self.config.max_tokens:
                # Further split large paragraphs by sentences
                sentence_chunks = self._split_by_sentences(paragraph)
                refined_chunks.extend(sentence_chunks)
            else:
                refined_chunks.append(paragraph)
        
        return refined_chunks
    
    def _apply_overlap(self, chunks: List[str]) -> List[str]:
        """Apply overlap between chunks if configured."""
        if self.config.overlap_tokens == 0 and self.config.overlap_ratio == 0:
            return chunks
        
        if len(chunks) <= 1:
            return chunks
        
        overlapped_chunks = [chunks[0]]  # First chunk unchanged
        
        for i in range(1, len(chunks)):
            prev_chunk = chunks[i - 1]
            current_chunk = chunks[i]
            
            # Calculate overlap
            prev_tokens = self._token_pattern.findall(prev_chunk)
            overlap_size = max(
                self.config.overlap_tokens,
                int(len(prev_tokens) * self.config.overlap_ratio)
            )
            
            if overlap_size > 0 and overlap_size < len(prev_tokens):
                overlap_text = ' '.join(prev_tokens[-overlap_size:])
                overlapped_chunk = f"{overlap_text} {current_chunk}"
                overlapped_chunks.append(overlapped_chunk)
            else:
                overlapped_chunks.append(current_chunk)
        
        return overlapped_chunks
    
    def _generate_page_anchors(self, chunk_text: str, full_content: str, position: int) -> List[PageAnchor]:
        """
        Generate page anchors for chunk (simplified implementation).
        
        Args:
            chunk_text: Text of the chunk
            full_content: Full document content
            position: Chunk position
            
        Returns:
            List of page anchors
        """
        # Find chunk position in full content
        start_pos = full_content.find(chunk_text)
        if start_pos == -1:
            # If exact match not found, use estimated position
            chars_per_chunk = len(full_content) // max(1, position + 1)
            start_pos = position * chars_per_chunk
        
        end_pos = start_pos + len(chunk_text)
        
        # Estimate page number (assuming ~2000 chars per page)
        chars_per_page = 2000
        page_number = (start_pos // chars_per_page) + 1
        
        page_start_char = start_pos % chars_per_page
        page_end_char = page_start_char + len(chunk_text)
        
        # Create page anchor
        anchor = PageAnchor(
            page_number=page_number,
            start_char=page_start_char,
            end_char=page_end_char,
            page_text_hash=self.generate_stable_id(chunk_text, prefix="page")
        )
        
        return [anchor]
    
    def _calculate_character_offsets(self, chunk_text: str, full_content: str) -> Tuple[int, int]:
        """Calculate character offsets for the chunk in the full content."""
        start_pos = full_content.find(chunk_text)
        if start_pos == -1:
            return (0, len(chunk_text))
        
        return (start_pos, start_pos + len(chunk_text))
    
    def _generate_output_artifacts(self, chunks: List[ProcessedChunk], document_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate canonical output artifacts.
        
        Args:
            chunks: Processed chunks
            document_info: Document information
            
        Returns:
            Output artifacts dictionary
        """
        document_stem = document_info["document_stem"]
        
        # Generate chunks JSON artifact
        chunks_data = []
        for chunk in chunks:
            chunk_data = {
                "chunk_id": chunk.chunk_id,
                "content": chunk.content,
                "page_anchors": [
                    {
                        "page_number": anchor.page_number,
                        "start_char": anchor.start_char,
                        "end_char": anchor.end_char,
                        "page_text_hash": anchor.page_text_hash,
                    }
                    for anchor in chunk.page_anchors
                ],
                "metadata": {
                    **chunk.metadata,
                    "token_count": chunk.token_count,
                    "sentence_count": chunk.sentence_count,
                    "paragraph_count": chunk.paragraph_count,
                    "provenance": {
                        "source_document": chunk.provenance.source_document,
                        "document_stem": chunk.provenance.document_stem,
                        "character_offsets": chunk.provenance.character_offsets,
                        "section_path": chunk.provenance.section_path,
                        "processing_timestamp": chunk.provenance.processing_timestamp,
                        "source_hash": chunk.provenance.source_hash,
                    }
                }
            }
            chunks_data.append(chunk_data)
        
        # Create final artifact structure
        artifact = {
            "document_stem": document_stem,
            "chunks": chunks_data,
            "summary": {
                "total_chunks": len(chunks),
                "total_tokens": sum(chunk.token_count for chunk in chunks),
                "chunking_policy": self.config.policy.value,
                "processing_timestamp": self._get_deterministic_timestamp(),
            },
            "component_metadata": self.get_deterministic_metadata(),
        }
        
        # Write artifact to canonical location
        output_path = f"canonical_flow/knowledge/{document_stem}_chunks.json"
        
        try:
            # Ensure directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Write with canonical JSON formatting
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(self.canonical_json_dumps(artifact, indent=2))
            
            return {
                "chunks_artifact": {
                    "path": output_path,
                    "format": "json",
                    "chunks_count": len(chunks),
                    "file_hash": self.generate_stable_id(artifact, prefix="file"),
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to write chunks artifact: {str(e)}")
            return {
                "chunks_artifact": {
                    "error": str(e),
                    "attempted_path": output_path,
                    "chunks_count": len(chunks),
                }
            }
    
    def _update_processing_stats(self, chunks: List[ProcessedChunk]) -> Dict[str, Any]:
        """Update and return processing statistics."""
        self.processing_stats["documents_processed"] += 1
        self.processing_stats["chunks_generated"] += len(chunks)
        
        total_tokens = sum(chunk.token_count for chunk in chunks)
        self.processing_stats["total_tokens"] += total_tokens
        
        if self.processing_stats["chunks_generated"] > 0:
            self.processing_stats["average_chunk_size"] = (
                self.processing_stats["total_tokens"] / 
                self.processing_stats["chunks_generated"]
            )
        
        return dict(self.processing_stats)
    
    def _generate_error_result(self, error_message: str, operation_id: str) -> Dict[str, Any]:
        """Generate standardized error result."""
        self.processing_stats["errors_encountered"] += 1
        
        result = {
            "component": self.component_name,
            "operation_id": operation_id,
            "status": "error",
            "error": error_message,
            "timestamp": self._get_deterministic_timestamp(),
            "processing_stats": dict(self.processing_stats),
        }
        
        return self.sort_dict_by_keys(result)


def main():
    """Example usage of the chunking processor."""
    # Create processor with default configuration
    processor = ChunkingProcessor()
    
    # Example input data (ingestion bundle format)
    test_data = {
        "document_stem": "example_doc",
        "document_content": {
            "title": "Example Document",
            "content": "This is the first paragraph with some sample content. It contains multiple sentences to demonstrate chunking.\n\nThis is the second paragraph. It also has multiple sentences for testing purposes.\n\nThis is the third paragraph with additional content for comprehensive testing.",
        },
        "content_metadata": {
            "source": "test"
        },
        "document_hash": "example_hash",
        "bundle_id": "example_bundle_123"
    }
    
    # Process the data
    result = processor.process(test_data)
    
    # Print results
    print("Chunking Processor Results:")
    print(f"Status: {result['status']}")
    print(f"Chunks Generated: {result.get('chunks_generated', 0)}")
    
    if result['status'] == 'success':
        artifacts = result.get('artifacts', {})
        if 'chunks_artifact' in artifacts:
            print(f"Output Path: {artifacts['chunks_artifact'].get('path', 'N/A')}")


if __name__ == "__main__":
    main()
"""
Retrieval-Analysis Adapter

Anti-corruption layer that sits between retrieval and analysis phases.
Translates data transfer objects and prevents circular dependencies.
"""

import logging
from typing import Any, Dict, List, Optional
from .data_transfer_objects import RetrievalOutputDTO, AnalysisInputDTO
from .schema_mismatch_logger import SchemaMismatchLogger
from .lineage_tracker import LineageTracker


class RetrievalAnalysisAdapter:
    """Adapter between retrieval and analysis components"""
    
    # Expected schema mappings
    RETRIEVAL_OUTPUT_SCHEMA = {
        'query_id': 'str',
        'retrieved_chunks': 'List[Dict[str, Any]]',
        'similarity_scores': 'List[float]',
        'retrieval_metadata': 'Dict[str, Any]'
    }
    
    ANALYSIS_INPUT_SCHEMA = {
        'document_chunks': 'List[Dict[str, Any]]',
        'context': 'Dict[str, Any]',
        'processing_metadata': 'Dict[str, Any]'
    }
    
    def __init__(self, adapter_id: str = "retrieval_analysis_adapter"):
        self.adapter_id = adapter_id
        self.logger = logging.getLogger(__name__)
        
        # Initialize tracking components
        self.lineage_tracker = LineageTracker()
        self.schema_logger = SchemaMismatchLogger(self.lineage_tracker)
        
        # Adapter statistics
        self.translation_count = 0
        self.successful_translations = 0
        self.failed_translations = 0
        
        self.logger.info(f"Initialized {self.adapter_id}")
    
    def translate_retrieval_to_analysis(
        self, 
        retrieval_output: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Translate retrieval output to analysis input format
        
        Args:
            retrieval_output: Raw output from retrieval components
            context: Optional processing context
            
        Returns:
            Translated data in analysis input format
        """
        
        self.translation_count += 1
        
        try:
            # Track the operation in lineage
            self.lineage_tracker.track_component_operation(
                component_id=self.adapter_id,
                operation_type="retrieval_to_analysis_translation",
                input_schema="retrieval_output",
                output_schema="analysis_input",
                dependencies=["retrieval_components"]
            )
            
            # Validate input schema
            mismatches = self.schema_logger.validate_retrieval_to_analysis(
                retrieval_output, 
                self.RETRIEVAL_OUTPUT_SCHEMA
            )
            
            if mismatches:
                self.schema_logger.log_mismatch(
                    source_schema="retrieval_output",
                    target_schema="analysis_input",
                    source_data=retrieval_output,
                    mismatch_details=mismatches,
                    adapter_id=self.adapter_id
                )
            
            # Perform translation
            translated_data = self._perform_translation(retrieval_output, context)
            
            # Validate output schema
            output_mismatches = self.schema_logger.validate_retrieval_to_analysis(
                translated_data,
                self.ANALYSIS_INPUT_SCHEMA
            )
            
            if output_mismatches:
                self.logger.warning(f"Output schema issues: {output_mismatches}")
            
            self.successful_translations += 1
            return translated_data
            
        except Exception as e:
            self.failed_translations += 1
            self.logger.error(f"Translation failed: {e}")
            
            # Return a safe fallback
            return self._create_fallback_analysis_input(retrieval_output, context)
    
    def _perform_translation(
        self, 
        retrieval_output: Dict[str, Any], 
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Perform the actual data translation"""
        
        # Extract and transform retrieved chunks
        document_chunks = []
        
        if 'retrieved_chunks' in retrieval_output:
            chunks = retrieval_output['retrieved_chunks']
            if isinstance(chunks, list):
                document_chunks = self._normalize_chunks(chunks)
        
        # Build processing context
        processing_context = {
            'query_id': retrieval_output.get('query_id', 'unknown'),
            'retrieval_timestamp': retrieval_output.get('timestamp'),
            'similarity_scores': retrieval_output.get('similarity_scores', []),
            'adapter_id': self.adapter_id
        }
        
        if context:
            processing_context.update(context)
        
        # Build processing metadata
        processing_metadata = {
            'source': 'retrieval_engine',
            'adapter': self.adapter_id,
            'translation_version': '1.0.0',
            'chunk_count': len(document_chunks),
            'has_scores': bool(retrieval_output.get('similarity_scores')),
            'retrieval_metadata': retrieval_output.get('retrieval_metadata', {})
        }
        
        # Create analysis input DTO
        analysis_input = {
            'document_chunks': document_chunks,
            'context': processing_context,
            'processing_metadata': processing_metadata,
            'adapter_metadata': {
                'adapter_id': self.adapter_id,
                'translation_count': self.translation_count,
                'source_schema': 'retrieval_output',
                'target_schema': 'analysis_input'
            }
        }
        
        return analysis_input
    
    def _normalize_chunks(self, chunks: List[Any]) -> List[Dict[str, Any]]:
        """Normalize document chunks to expected format"""
        
        normalized = []
        
        for i, chunk in enumerate(chunks):
            if isinstance(chunk, dict):
                # Already in dict format, ensure required fields
                normalized_chunk = {
                    'chunk_id': chunk.get('chunk_id', f'chunk_{i}'),
                    'content': chunk.get('content', chunk.get('text', str(chunk))),
                    'metadata': chunk.get('metadata', {}),
                    'source': chunk.get('source', 'unknown'),
                    'chunk_index': i
                }
            elif isinstance(chunk, str):
                # Text chunk, create structured representation
                normalized_chunk = {
                    'chunk_id': f'text_chunk_{i}',
                    'content': chunk,
                    'metadata': {},
                    'source': 'text_input',
                    'chunk_index': i
                }
            else:
                # Other types, convert to string
                normalized_chunk = {
                    'chunk_id': f'raw_chunk_{i}',
                    'content': str(chunk),
                    'metadata': {'original_type': type(chunk).__name__},
                    'source': 'raw_input',
                    'chunk_index': i
                }
            
            normalized.append(normalized_chunk)
        
        return normalized
    
    def _create_fallback_analysis_input(
        self, 
        retrieval_output: Dict[str, Any], 
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Create fallback analysis input when translation fails"""
        
        return {
            'document_chunks': [
                {
                    'chunk_id': 'fallback_chunk_0',
                    'content': str(retrieval_output),
                    'metadata': {'fallback': True, 'source_type': 'failed_translation'},
                    'source': 'adapter_fallback',
                    'chunk_index': 0
                }
            ],
            'context': context or {},
            'processing_metadata': {
                'source': 'adapter_fallback',
                'adapter': self.adapter_id,
                'translation_version': 'fallback',
                'chunk_count': 1,
                'has_scores': False,
                'retrieval_metadata': {}
            },
            'adapter_metadata': {
                'adapter_id': self.adapter_id,
                'translation_count': self.translation_count,
                'source_schema': 'unknown',
                'target_schema': 'analysis_input_fallback',
                'fallback_used': True
            }
        }
    
    def get_adapter_statistics(self) -> Dict[str, Any]:
        """Get adapter performance statistics"""
        
        success_rate = 0.0
        if self.translation_count > 0:
            success_rate = self.successful_translations / self.translation_count
        
        return {
            'adapter_id': self.adapter_id,
            'total_translations': self.translation_count,
            'successful_translations': self.successful_translations,
            'failed_translations': self.failed_translations,
            'success_rate': success_rate,
            'schema_mismatches': len(self.schema_logger.mismatch_events),
            'lineage_events': len(self.lineage_tracker.lineage_events),
            'dependency_violations': len(self.lineage_tracker.violation_history)
        }
    
    def validate_translation(
        self, 
        input_data: Dict[str, Any], 
        output_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate a completed translation"""
        
        # Check input schema compliance
        input_mismatches = self.schema_logger.validate_retrieval_to_analysis(
            input_data, 
            self.RETRIEVAL_OUTPUT_SCHEMA
        )
        
        # Check output schema compliance
        output_mismatches = self.schema_logger.validate_retrieval_to_analysis(
            output_data,
            self.ANALYSIS_INPUT_SCHEMA
        )
        
        # Check data preservation
        input_chunks = input_data.get('retrieved_chunks', [])
        output_chunks = output_data.get('document_chunks', [])
        
        data_preserved = len(input_chunks) == len(output_chunks)
        
        return {
            'input_schema_valid': len(input_mismatches) == 0,
            'output_schema_valid': len(output_mismatches) == 0,
            'data_preserved': data_preserved,
            'input_mismatches': input_mismatches,
            'output_mismatches': output_mismatches,
            'chunk_count_match': len(input_chunks) == len(output_chunks),
            'validation_passed': (
                len(input_mismatches) == 0 and 
                len(output_mismatches) == 0 and 
                data_preserved
            )
        }
    
    def get_lineage_info(self) -> Dict[str, Any]:
        """Get lineage information for this adapter"""
        return self.lineage_tracker.get_component_lineage(self.adapter_id)
    
    def get_schema_mismatch_summary(self) -> Dict[str, Any]:
        """Get schema mismatch summary for this adapter"""
        return self.schema_logger.get_mismatch_summary(self.adapter_id)
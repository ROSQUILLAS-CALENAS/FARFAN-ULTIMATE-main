"""
Concrete evidence processor implementations

These implementations depend only on validator_api interfaces and have
no direct imports from pipeline stages.
"""

import hashlib
import json
import logging
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from validator_api.interfaces import IEvidenceProcessor
from validator_api.dtos import EvidenceItem

logger = logging.getLogger(__name__)


class DefaultEvidenceProcessor(IEvidenceProcessor):
    """Default implementation of evidence processor"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.supported_evidence_types = [
            "direct_quote",
            "paraphrase", 
            "statistical",
            "expert_opinion",
            "case_study",
            "general"
        ]
    
    def process_evidence(self, evidence_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process raw evidence data into structured format"""
        
        if not evidence_data:
            return []
        
        processed_items = []
        
        # Handle different input formats
        if isinstance(evidence_data, dict):
            if 'items' in evidence_data:
                # Bulk processing format
                for item in evidence_data['items']:
                    processed_item = self._process_single_item(item)
                    if processed_item:
                        processed_items.append(processed_item)
            else:
                # Single item format
                processed_item = self._process_single_item(evidence_data)
                if processed_item:
                    processed_items.append(processed_item)
        
        return processed_items
    
    def _process_single_item(self, item_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single evidence item"""
        
        try:
            # Extract required fields
            content = item_data.get('content', '').strip()
            if not content:
                logger.warning("Skipping evidence item with empty content")
                return None
            
            # Generate ID if not provided
            item_id = item_data.get('id')
            if not item_id:
                item_id = self._generate_evidence_id(content)
            
            # Determine evidence type
            evidence_type = item_data.get('type', 'general')
            if evidence_type not in self.supported_evidence_types:
                evidence_type = self._infer_evidence_type(content)
            
            # Extract source information
            source = item_data.get('source', 'unknown')
            
            # Calculate confidence level
            confidence_level = item_data.get('confidence_level')
            if confidence_level is None:
                confidence_level = self._calculate_confidence(content, item_data)
            
            # Extract and enhance metadata
            metadata = item_data.get('metadata', {}).copy()
            metadata.update({
                'processed_at': datetime.utcnow().isoformat(),
                'processor_type': 'default',
                'word_count': len(content.split()),
                'char_count': len(content),
                'has_citations': self._has_citations(content)
            })
            
            # Create evidence item
            evidence_item = EvidenceItem(
                id=item_id,
                content=content,
                source=source,
                evidence_type=evidence_type,
                confidence_level=confidence_level,
                metadata=metadata
            )
            
            return evidence_item.to_dict()
            
        except Exception as e:
            logger.error(f"Error processing evidence item: {e}")
            return None
    
    def _generate_evidence_id(self, content: str) -> str:
        """Generate a deterministic ID for evidence content"""
        content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
        return f"ev_{content_hash[:12]}"
    
    def _infer_evidence_type(self, content: str) -> str:
        """Infer evidence type from content"""
        content_lower = content.lower()
        
        # Check for direct quotes
        if '"' in content or "'" in content or 'said' in content_lower:
            return "direct_quote"
        
        # Check for statistical evidence
        if any(word in content_lower for word in ['percent', '%', 'statistic', 'data', 'survey', 'study']):
            return "statistical"
        
        # Check for expert opinion
        if any(word in content_lower for word in ['expert', 'professor', 'dr.', 'researcher', 'according to']):
            return "expert_opinion"
        
        # Check for case studies
        if any(word in content_lower for word in ['case study', 'example', 'instance', 'scenario']):
            return "case_study"
        
        # Default to paraphrase for other content
        return "paraphrase"
    
    def _calculate_confidence(self, content: str, item_data: Dict[str, Any]) -> float:
        """Calculate confidence level for evidence"""
        base_confidence = 0.5
        
        # Boost for source reliability indicators
        if item_data.get('source', '').lower() in ['academic', 'government', 'official', 'peer-reviewed']:
            base_confidence += 0.2
        
        # Boost for citations
        if self._has_citations(content):
            base_confidence += 0.1
        
        # Boost for specific data
        if any(word in content.lower() for word in ['study', 'research', 'data', 'analysis']):
            base_confidence += 0.1
        
        # Penalty for uncertainty indicators
        if any(word in content.lower() for word in ['maybe', 'perhaps', 'possibly', 'might']):
            base_confidence -= 0.1
        
        return max(0.0, min(1.0, base_confidence))
    
    def _has_citations(self, content: str) -> bool:
        """Check if content has citation indicators"""
        citation_patterns = [
            r'\(\d{4}\)',  # (2023)
            r'\[\d+\]',    # [1]
            r'et al\.?',   # et al.
            r'doi:',       # doi:
            r'https?://',  # URLs
        ]
        
        for pattern in citation_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return True
        
        return False
    
    def extract_features(self, evidence_item: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features from a single evidence item"""
        
        if not evidence_item or 'content' not in evidence_item:
            return {}
        
        content = evidence_item['content']
        features = {}
        
        # Basic text features
        features['word_count'] = len(content.split())
        features['char_count'] = len(content)
        features['sentence_count'] = len(re.split(r'[.!?]+', content))
        features['avg_word_length'] = sum(len(word) for word in content.split()) / len(content.split()) if content.split() else 0
        
        # Content type features
        features['has_numbers'] = bool(re.search(r'\d', content))
        features['has_citations'] = self._has_citations(content)
        features['has_quotes'] = '"' in content or "'" in content
        features['has_questions'] = '?' in content
        features['has_exclamations'] = '!' in content
        
        # Linguistic features
        features['uppercase_ratio'] = sum(1 for c in content if c.isupper()) / len(content) if content else 0
        features['punctuation_density'] = sum(1 for c in content if not c.isalnum() and not c.isspace()) / len(content) if content else 0
        
        # Evidence quality indicators
        content_lower = content.lower()
        features['certainty_indicators'] = sum(1 for word in ['definitely', 'certainly', 'clearly', 'obviously'] if word in content_lower)
        features['uncertainty_indicators'] = sum(1 for word in ['maybe', 'perhaps', 'possibly', 'might'] if word in content_lower)
        features['authority_indicators'] = sum(1 for word in ['expert', 'study', 'research', 'official'] if word in content_lower)
        
        return features
    
    def validate_evidence_structure(self, evidence_item: Dict[str, Any]) -> bool:
        """Validate the structure of an evidence item"""
        
        required_fields = ['id', 'content', 'source', 'evidence_type']
        
        # Check required fields
        for field in required_fields:
            if field not in evidence_item:
                logger.warning(f"Missing required field: {field}")
                return False
            
            if not evidence_item[field]:
                logger.warning(f"Empty required field: {field}")
                return False
        
        # Validate field types
        if not isinstance(evidence_item['id'], str):
            logger.warning("ID must be a string")
            return False
        
        if not isinstance(evidence_item['content'], str):
            logger.warning("Content must be a string")
            return False
        
        if not isinstance(evidence_item['source'], str):
            logger.warning("Source must be a string")
            return False
        
        if not isinstance(evidence_item['evidence_type'], str):
            logger.warning("Evidence type must be a string")
            return False
        
        # Validate confidence level if present
        if 'confidence_level' in evidence_item:
            confidence = evidence_item['confidence_level']
            if not isinstance(confidence, (int, float)) or not (0 <= confidence <= 1):
                logger.warning("Confidence level must be a number between 0 and 1")
                return False
        
        return True
    
    def get_evidence_metadata(self, evidence_item: Dict[str, Any]) -> Dict[str, Any]:
        """Get metadata for an evidence item"""
        
        base_metadata = evidence_item.get('metadata', {}).copy()
        
        # Add computed metadata
        if 'content' in evidence_item:
            features = self.extract_features(evidence_item)
            base_metadata.update({
                'features': features,
                'quality_score': self._calculate_quality_score(features),
                'processing_timestamp': datetime.utcnow().isoformat()
            })
        
        return base_metadata
    
    def _calculate_quality_score(self, features: Dict[str, Any]) -> float:
        """Calculate a quality score based on features"""
        score = 0.5  # Base score
        
        # Positive indicators
        if features.get('has_citations', False):
            score += 0.15
        
        if features.get('authority_indicators', 0) > 0:
            score += 0.10
        
        if features.get('word_count', 0) >= 20:  # Sufficient detail
            score += 0.10
        
        # Negative indicators
        if features.get('uncertainty_indicators', 0) > features.get('certainty_indicators', 0):
            score -= 0.10
        
        if features.get('word_count', 0) < 5:  # Too short
            score -= 0.20
        
        return max(0.0, min(1.0, score))


class DNPEvidenceProcessor(IEvidenceProcessor):
    """DNP-specific evidence processor with enhanced validation"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.supported_evidence_types = [
            "legal_reference",
            "regulatory_text",
            "policy_statement",
            "statistical_data",
            "case_precedent",
            "procedural_evidence",
            "constitutional_reference"
        ]
        
        # DNP-specific keywords and patterns
        self.dnp_keywords = [
            'departamento nacional de planeación', 'dnp', 'plan nacional de desarrollo',
            'política pública', 'desarrollo territorial', 'gestión pública'
        ]
        
        self.legal_indicators = [
            'ley', 'decreto', 'resolución', 'constitución', 'código',
            'jurisprudencia', 'sentencia', 'norma'
        ]
    
    def process_evidence(self, evidence_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process evidence data with DNP-specific enhancements"""
        
        if not evidence_data:
            return []
        
        processed_items = []
        
        # Handle different input formats
        if isinstance(evidence_data, dict):
            if 'items' in evidence_data:
                for item in evidence_data['items']:
                    processed_item = self._process_dnp_item(item)
                    if processed_item:
                        processed_items.append(processed_item)
            else:
                processed_item = self._process_dnp_item(evidence_data)
                if processed_item:
                    processed_items.append(processed_item)
        
        return processed_items
    
    def _process_dnp_item(self, item_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single evidence item with DNP enhancements"""
        
        try:
            # Extract required fields
            content = item_data.get('content', '').strip()
            if not content:
                return None
            
            # Generate ID
            item_id = item_data.get('id') or self._generate_dnp_evidence_id(content)
            
            # Determine DNP-specific evidence type
            evidence_type = self._classify_dnp_evidence_type(content, item_data.get('type'))
            
            # Extract source with DNP context
            source = self._enhance_source_info(item_data.get('source', 'unknown'), content)
            
            # Calculate DNP-specific confidence
            confidence_level = self._calculate_dnp_confidence(content, item_data, evidence_type)
            
            # Enhanced metadata with DNP context
            metadata = self._create_dnp_metadata(content, item_data.get('metadata', {}), evidence_type)
            
            # Create evidence item
            evidence_item = EvidenceItem(
                id=item_id,
                content=content,
                source=source,
                evidence_type=evidence_type,
                confidence_level=confidence_level,
                metadata=metadata
            )
            
            return evidence_item.to_dict()
            
        except Exception as e:
            logger.error(f"Error processing DNP evidence item: {e}")
            return None
    
    def _generate_dnp_evidence_id(self, content: str) -> str:
        """Generate DNP-specific evidence ID"""
        content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
        return f"dnp_ev_{content_hash[:12]}"
    
    def _classify_dnp_evidence_type(self, content: str, provided_type: Optional[str] = None) -> str:
        """Classify evidence type with DNP context"""
        
        if provided_type and provided_type in self.supported_evidence_types:
            return provided_type
        
        content_lower = content.lower()
        
        # Check for legal references
        if any(indicator in content_lower for indicator in self.legal_indicators):
            return "legal_reference"
        
        # Check for constitutional references
        if 'constitución' in content_lower or 'constitutional' in content_lower:
            return "constitutional_reference"
        
        # Check for regulatory text
        if any(word in content_lower for word in ['decreto', 'resolución', 'reglamento', 'normativa']):
            return "regulatory_text"
        
        # Check for statistical data
        if any(word in content_lower for word in ['estadística', 'datos', 'cifras', 'porcentaje', '%']):
            return "statistical_data"
        
        # Check for policy statements
        if any(word in content_lower for word in ['política', 'estrategia', 'plan', 'programa']):
            return "policy_statement"
        
        # Check for procedural evidence
        if any(word in content_lower for word in ['procedimiento', 'proceso', 'trámite', 'protocolo']):
            return "procedural_evidence"
        
        # Check for case precedents
        if any(word in content_lower for word in ['caso', 'precedente', 'ejemplo', 'antecedente']):
            return "case_precedent"
        
        return "policy_statement"  # Default for DNP context
    
    def _enhance_source_info(self, source: str, content: str) -> str:
        """Enhance source information with DNP context"""
        
        content_lower = content.lower()
        
        # Identify government sources
        if any(keyword in content_lower for keyword in self.dnp_keywords):
            if source == 'unknown':
                return 'DNP - Departamento Nacional de Planeación'
            else:
                return f"{source} (DNP Context)"
        
        # Identify legal sources
        if any(indicator in content_lower for indicator in self.legal_indicators):
            if source == 'unknown':
                return 'Fuente Legal/Normativa'
            else:
                return f"{source} (Legal)"
        
        return source
    
    def _calculate_dnp_confidence(self, content: str, item_data: Dict[str, Any], evidence_type: str) -> float:
        """Calculate DNP-specific confidence score"""
        
        base_confidence = 0.5
        content_lower = content.lower()
        
        # Boost for official DNP sources
        if any(keyword in content_lower for keyword in self.dnp_keywords):
            base_confidence += 0.25
        
        # Boost for legal/regulatory content
        if evidence_type in ['legal_reference', 'constitutional_reference', 'regulatory_text']:
            base_confidence += 0.20
        
        # Boost for specific legal citations
        if any(indicator in content_lower for indicator in self.legal_indicators):
            base_confidence += 0.15
        
        # Boost for statistical data with sources
        if evidence_type == 'statistical_data' and any(word in content_lower for word in ['dane', 'estadísticas', 'censo']):
            base_confidence += 0.15
        
        # Penalty for informal language
        informal_indicators = ['creo que', 'pienso que', 'me parece', 'tal vez']
        if any(indicator in content_lower for indicator in informal_indicators):
            base_confidence -= 0.15
        
        return max(0.0, min(1.0, base_confidence))
    
    def _create_dnp_metadata(self, content: str, base_metadata: Dict[str, Any], evidence_type: str) -> Dict[str, Any]:
        """Create enhanced metadata with DNP context"""
        
        metadata = base_metadata.copy()
        content_lower = content.lower()
        
        # Add DNP-specific metadata
        metadata.update({
            'processor_type': 'dnp_evidence',
            'processed_at': datetime.utcnow().isoformat(),
            'dnp_context': any(keyword in content_lower for keyword in self.dnp_keywords),
            'legal_context': any(indicator in content_lower for indicator in self.legal_indicators),
            'evidence_classification': evidence_type
        })
        
        # Add regulatory compliance indicators
        compliance_indicators = ['cumple', 'conforme', 'acorde', 'según normativa']
        metadata['compliance_indicators'] = sum(1 for indicator in compliance_indicators if indicator in content_lower)
        
        # Add policy relevance indicators
        policy_indicators = ['desarrollo', 'planeación', 'territorial', 'social', 'económico']
        metadata['policy_relevance'] = sum(1 for indicator in policy_indicators if indicator in content_lower)
        
        return metadata
    
    def extract_features(self, evidence_item: Dict[str, Any]) -> Dict[str, Any]:
        """Extract DNP-specific features"""
        
        base_features = self._extract_basic_features(evidence_item)
        
        if 'content' not in evidence_item:
            return base_features
        
        content = evidence_item['content']
        content_lower = content.lower()
        
        # Add DNP-specific features
        dnp_features = {
            'dnp_keyword_count': sum(1 for keyword in self.dnp_keywords if keyword in content_lower),
            'legal_indicator_count': sum(1 for indicator in self.legal_indicators if indicator in content_lower),
            'has_constitutional_reference': 'constitución' in content_lower,
            'has_regulatory_language': any(word in content_lower for word in ['decreto', 'ley', 'resolución']),
            'has_statistical_data': bool(re.search(r'\d+\%|\d+\.\d+', content)),
            'formality_score': self._calculate_formality_score(content_lower)
        }
        
        base_features.update(dnp_features)
        return base_features
    
    def _extract_basic_features(self, evidence_item: Dict[str, Any]) -> Dict[str, Any]:
        """Extract basic text features"""
        
        if 'content' not in evidence_item:
            return {}
        
        content = evidence_item['content']
        
        return {
            'word_count': len(content.split()),
            'char_count': len(content),
            'sentence_count': len(re.split(r'[.!?]+', content)),
            'has_numbers': bool(re.search(r'\d', content)),
            'has_citations': bool(re.search(r'\[\d+\]|\(\d{4}\)', content)),
            'uppercase_ratio': sum(1 for c in content if c.isupper()) / len(content) if content else 0
        }
    
    def _calculate_formality_score(self, content_lower: str) -> float:
        """Calculate formality score for content"""
        
        formal_indicators = [
            'por tanto', 'en consecuencia', 'considerando', 'teniendo en cuenta',
            'de conformidad con', 'según', 'conforme a', 'en virtud de'
        ]
        
        informal_indicators = [
            'creo', 'pienso', 'me parece', 'tal vez', 'quizás', 'puede ser'
        ]
        
        formal_count = sum(1 for indicator in formal_indicators if indicator in content_lower)
        informal_count = sum(1 for indicator in informal_indicators if indicator in content_lower)
        
        if formal_count + informal_count == 0:
            return 0.5  # Neutral
        
        return formal_count / (formal_count + informal_count)
    
    def validate_evidence_structure(self, evidence_item: Dict[str, Any]) -> bool:
        """Validate evidence structure with DNP requirements"""
        
        # Basic validation
        if not self._basic_structure_validation(evidence_item):
            return False
        
        # DNP-specific validation
        if 'evidence_type' in evidence_item:
            evidence_type = evidence_item['evidence_type']
            
            # Legal references should have proper citations
            if evidence_type in ['legal_reference', 'constitutional_reference']:
                content = evidence_item.get('content', '').lower()
                if not any(indicator in content for indicator in self.legal_indicators):
                    logger.warning("Legal evidence type lacks proper legal indicators")
                    return False
        
        return True
    
    def _basic_structure_validation(self, evidence_item: Dict[str, Any]) -> bool:
        """Basic structure validation"""
        
        required_fields = ['id', 'content', 'source', 'evidence_type']
        
        for field in required_fields:
            if field not in evidence_item or not evidence_item[field]:
                logger.warning(f"Missing or empty required field: {field}")
                return False
        
        return True
    
    def get_evidence_metadata(self, evidence_item: Dict[str, Any]) -> Dict[str, Any]:
        """Get comprehensive metadata for evidence item"""
        
        base_metadata = evidence_item.get('metadata', {}).copy()
        
        if 'content' in evidence_item:
            features = self.extract_features(evidence_item)
            quality_score = self._calculate_dnp_quality_score(features)
            
            base_metadata.update({
                'features': features,
                'quality_score': quality_score,
                'dnp_compliance_score': self._calculate_compliance_score(features),
                'processing_timestamp': datetime.utcnow().isoformat()
            })
        
        return base_metadata
    
    def _calculate_dnp_quality_score(self, features: Dict[str, Any]) -> float:
        """Calculate DNP-specific quality score"""
        
        score = 0.5  # Base score
        
        # Positive factors
        if features.get('dnp_keyword_count', 0) > 0:
            score += 0.2
        
        if features.get('legal_indicator_count', 0) > 0:
            score += 0.15
        
        if features.get('formality_score', 0) > 0.7:
            score += 0.1
        
        if features.get('has_statistical_data', False):
            score += 0.1
        
        # Negative factors
        if features.get('word_count', 0) < 10:
            score -= 0.2
        
        if features.get('formality_score', 0) < 0.3:
            score -= 0.1
        
        return max(0.0, min(1.0, score))
    
    def _calculate_compliance_score(self, features: Dict[str, Any]) -> float:
        """Calculate regulatory compliance score"""
        
        compliance_score = 0.5
        
        # Legal compliance indicators
        if features.get('has_constitutional_reference', False):
            compliance_score += 0.25
        
        if features.get('has_regulatory_language', False):
            compliance_score += 0.2
        
        if features.get('legal_indicator_count', 0) > 0:
            compliance_score += 0.15
        
        # Formality indicates proper procedure
        formality = features.get('formality_score', 0.5)
        compliance_score += (formality - 0.5) * 0.2
        
        return max(0.0, min(1.0, compliance_score))
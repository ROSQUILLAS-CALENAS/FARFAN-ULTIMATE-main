"""
Canonical Flow Alias: 15A
Extractor Evidencias Contextual with Total Ordering and Deterministic Processing

Source: extractor_evidencias_contextual.py
Stage: analysis_nlp
Code: 15A
"""

import json
import logging
import re
import sys
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import OrderedDict

# Import total ordering base
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from total_ordering_base import TotalOrderingBase, DeterministicCollectionMixin

logger = logging.getLogger(__name__)


class EvidenceRelevance(Enum):
    """Niveles de relevancia de evidencia"""
    ALTA = "alta"
    MEDIA = "media"
    BAJA = "baja"
    NULA = "nula"


class EvidenceType(Enum):
    """Tipos de evidencia"""
    DIRECT_QUOTE = "direct_quote"
    PARAPHRASE = "paraphrase"
    STATISTICAL = "statistical"
    EXPERT_OPINION = "expert_opinion"
    CASE_STUDY = "case_study"
    REGULATORY = "regulatory"
    FINANCIAL = "financial"


@dataclass
class EvidenceContext:
    """Contexto de evidencia extraída"""
    context_id: str
    text_before: str = ""
    text_after: str = ""
    section_header: str = ""
    page_number: Optional[int] = None
    document_section: str = ""
    
    def __post_init__(self):
        # Ensure deterministic ordering
        self.context_id = self.context_id or ""


@dataclass
class ExtractedEvidence:
    """Evidencia extraída con contexto completo"""
    evidence_id: str
    text: str
    evidence_type: EvidenceType
    relevance: EvidenceRelevance
    confidence: float
    context: EvidenceContext
    keywords: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    extraction_timestamp: str = ""
    
    def __post_init__(self):
        # Ensure deterministic ordering
        self.keywords = sorted(self.keywords)
        if self.metadata:
            self.metadata = OrderedDict(sorted(self.metadata.items()))


class ExtractorEvidenciasContextual(TotalOrderingBase, DeterministicCollectionMixin):
    """
    Extractor de Evidencias Contextual with deterministic processing and total ordering.
    
    Provides consistent evidence extraction results and stable ID generation across runs.
    """
    
    def __init__(self):
        super().__init__("ExtractorEvidenciasContextual")
        
        # Configuration with deterministic defaults
        self.context_window_size = 200
        self.relevance_threshold = 0.5
        self.max_evidences_per_document = 50
        
        # Keyword patterns for evidence detection
        self.evidence_patterns = self._initialize_evidence_patterns()
        
        # Statistics tracking
        self.extraction_stats = {
            "documents_processed": 0,
            "evidences_extracted": 0,
            "high_relevance_count": 0,
            "medium_relevance_count": 0,
            "low_relevance_count": 0,
        }
        
        # Update state hash
        self.update_state_hash(self._get_initial_state())
    
    def _get_comparison_key(self) -> Tuple[str, ...]:
        """Return comparison key for deterministic ordering"""
        return (
            self.component_id,
            str(self.context_window_size),
            str(self.relevance_threshold),
            str(self.max_evidences_per_document),
            str(sorted(self.evidence_patterns.keys())),
            str(self._state_hash or "")
        )
    
    def _get_initial_state(self) -> Dict[str, Any]:
        """Get initial state for hash calculation"""
        return {
            "context_window_size": self.context_window_size,
            "max_evidences_per_document": self.max_evidences_per_document,
            "pattern_count": len(self.evidence_patterns),
            "relevance_threshold": self.relevance_threshold,
        }
    
    def _initialize_evidence_patterns(self) -> Dict[str, List[str]]:
        """Initialize evidence detection patterns with deterministic ordering"""
        patterns = {
            "comparative": [
                "comparado con",
                "en relación a",
                "frente a",
                "mayor que",
                "menor que",
                "versus",
            ],
            "financial": [
                "inversión", 
                "presupuesto",
                "recursos",
                "costo",
                "financiamiento",
                "millones",
                "pesos",
            ],
            "quantitative": [
                "cantidad",
                "número",
                "porcentaje",
                "cifra", 
                "dato",
                "estadística",
                "métrica",
            ],
            "qualitative": [
                "calidad",
                "característica",
                "descripción",
                "explicación",
                "proceso",
                "método",
            ],
            "regulatory": [
                "decreto",
                "ley",
                "norma",
                "reglamento",
                "resolución",
                "artículo",
            ],
        }
        
        # Ensure deterministic ordering
        return {k: sorted(v) for k, v in sorted(patterns.items())}
    
    def process(self, data: Any = None, context: Any = None) -> Dict[str, Any]:
        """
        Main processing function with deterministic output.
        
        Args:
            data: Input data containing documents or text
            context: Processing context
            
        Returns:
            Deterministic evidence extraction results
        """
        operation_id = self.generate_operation_id("process", {"data": data, "context": context})
        
        try:
            # Canonicalize inputs
            canonical_data = self.canonicalize_data(data) if data else {}
            canonical_context = self.canonicalize_data(context) if context else {}
            
            # Extract documents
            documents = self._extract_documents_deterministic(canonical_data)
            
            # Process each document
            all_evidences = []
            for doc in documents:
                evidences = self._extract_evidences_from_document_deterministic(doc)
                all_evidences.extend(evidences)
            
            # Filter and rank evidences
            filtered_evidences = self._filter_and_rank_evidences_deterministic(all_evidences)
            
            # Generate deterministic output
            output = self._generate_deterministic_output(filtered_evidences, operation_id)
            
            # Update statistics
            self._update_extraction_stats(filtered_evidences)
            
            # Update state hash
            self.update_state_hash(output)
            
            return output
            
        except Exception as e:
            error_output = {
                "component": self.component_name,
                "error": str(e),
                "operation_id": operation_id,
                "status": "error", 
                "timestamp": self._get_deterministic_timestamp(),
            }
            return self.sort_dict_by_keys(error_output)
    
    def _extract_documents_deterministic(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract documents from input data with stable ordering"""
        documents = []
        
        # Handle single document
        if "document" in data:
            doc_data = data["document"]
            if isinstance(doc_data, str):
                documents.append({
                    "id": self.generate_stable_id(doc_data, prefix="doc"),
                    "text": doc_data,
                    "metadata": {}
                })
            elif isinstance(doc_data, dict):
                documents.append({
                    "id": doc_data.get("id", self.generate_stable_id(str(doc_data), prefix="doc")),
                    "text": doc_data.get("text", ""),
                    "metadata": doc_data.get("metadata", {})
                })
        
        # Handle multiple documents
        if "documents" in data and isinstance(data["documents"], list):
            for i, doc in enumerate(data["documents"]):
                if isinstance(doc, str):
                    documents.append({
                        "id": self.generate_stable_id(doc, prefix="doc"),
                        "text": doc,
                        "metadata": {}
                    })
                elif isinstance(doc, dict):
                    documents.append({
                        "id": doc.get("id", self.generate_stable_id(str(doc), prefix="doc")),
                        "text": doc.get("text", ""),
                        "metadata": doc.get("metadata", {})
                    })
        
        # Handle text directly
        if "text" in data and not documents:
            documents.append({
                "id": self.generate_stable_id(data["text"], prefix="doc"),
                "text": str(data["text"]),
                "metadata": {}
            })
        
        # If no documents found, create empty document
        if not documents:
            documents.append({
                "id": "default_empty",
                "text": "",
                "metadata": {}
            })
        
        return documents
    
    def _extract_evidences_from_document_deterministic(self, document: Dict[str, Any]) -> List[ExtractedEvidence]:
        """Extract evidences from a single document with deterministic processing"""
        doc_id = document["id"]
        doc_text = document["text"]
        doc_metadata = document.get("metadata", {})
        
        evidences = []
        
        # Split text into sentences for processing
        sentences = self._split_text_into_sentences(doc_text)
        
        # Process each sentence
        for i, sentence in enumerate(sentences):
            # Skip very short sentences
            if len(sentence.strip()) < 20:
                continue
            
            # Check for evidence patterns
            evidence_type, relevance, confidence = self._analyze_sentence_for_evidence(sentence)
            
            # Only keep relevant evidences
            if relevance != EvidenceRelevance.NULA and confidence >= self.relevance_threshold:
                # Extract context
                context = self._extract_sentence_context(sentences, i, doc_id)
                
                # Extract keywords
                keywords = self._extract_keywords_deterministic(sentence)
                
                # Create evidence object
                evidence = ExtractedEvidence(
                    evidence_id=self.generate_stable_id(
                        {"doc_id": doc_id, "sentence": sentence, "index": i}, 
                        prefix="ev"
                    ),
                    text=sentence.strip(),
                    evidence_type=evidence_type,
                    relevance=relevance,
                    confidence=confidence,
                    context=context,
                    keywords=keywords,
                    metadata={
                        "document_id": doc_id,
                        "sentence_index": i,
                        "document_metadata": doc_metadata,
                    },
                    extraction_timestamp=self._get_deterministic_timestamp(),
                )
                
                evidences.append(evidence)
            
            # Limit evidences per document
            if len(evidences) >= self.max_evidences_per_document:
                break
        
        return evidences
    
    def _split_text_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences deterministically"""
        # Simple sentence splitting using regex
        sentences = re.split(r'[.!?]+\s+', text)
        
        # Clean up sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def _analyze_sentence_for_evidence(self, sentence: str) -> Tuple[EvidenceType, EvidenceRelevance, float]:
        """Analyze sentence to determine evidence type, relevance, and confidence"""
        sentence_lower = sentence.lower()
        
        # Initialize scores
        pattern_scores = {}
        for pattern_type, patterns in self.evidence_patterns.items():
            score = 0
            for pattern in patterns:
                if pattern in sentence_lower:
                    score += 1
            pattern_scores[pattern_type] = score / len(patterns) if patterns else 0
        
        # Determine evidence type based on highest scoring pattern
        best_pattern_type = max(pattern_scores.items(), key=lambda x: x[1])
        
        evidence_type = EvidenceType.PARAPHRASE  # Default
        if best_pattern_type[0] == "quantitative":
            evidence_type = EvidenceType.STATISTICAL
        elif best_pattern_type[0] == "regulatory":
            evidence_type = EvidenceType.REGULATORY
        elif best_pattern_type[0] == "financial":
            evidence_type = EvidenceType.FINANCIAL
        elif '"' in sentence or "'" in sentence:
            evidence_type = EvidenceType.DIRECT_QUOTE
        
        # Calculate overall confidence
        max_score = best_pattern_type[1]
        confidence = max_score
        
        # Determine relevance
        if confidence >= 0.7:
            relevance = EvidenceRelevance.ALTA
        elif confidence >= 0.4:
            relevance = EvidenceRelevance.MEDIA
        elif confidence >= 0.2:
            relevance = EvidenceRelevance.BAJA
        else:
            relevance = EvidenceRelevance.NULA
        
        return evidence_type, relevance, confidence
    
    def _extract_sentence_context(self, sentences: List[str], sentence_index: int, doc_id: str) -> EvidenceContext:
        """Extract context around a sentence"""
        context_before = ""
        context_after = ""
        
        # Get context before
        start_idx = max(0, sentence_index - 1)
        if start_idx < sentence_index:
            context_before = " ".join(sentences[start_idx:sentence_index])
        
        # Get context after
        end_idx = min(len(sentences), sentence_index + 2)
        if end_idx > sentence_index + 1:
            context_after = " ".join(sentences[sentence_index + 1:end_idx])
        
        # Truncate context if too long
        if len(context_before) > self.context_window_size:
            context_before = context_before[-self.context_window_size:]
        if len(context_after) > self.context_window_size:
            context_after = context_after[:self.context_window_size]
        
        return EvidenceContext(
            context_id=self.generate_stable_id(
                {"doc_id": doc_id, "sentence_index": sentence_index}, 
                prefix="ctx"
            ),
            text_before=context_before,
            text_after=context_after,
            section_header="",
            page_number=None,
            document_section="",
        )
    
    def _extract_keywords_deterministic(self, text: str) -> List[str]:
        """Extract keywords from text deterministically"""
        # Simple keyword extraction using word frequency
        words = re.findall(r'\b[a-zA-ZáéíóúñÁÉÍÓÚÑ]+\b', text.lower())
        
        # Filter out common stop words (Spanish)
        stop_words = {
            "a", "al", "algo", "algunas", "algunos", "ante", "antes", "como", "con", "contra",
            "cual", "cuando", "de", "del", "desde", "donde", "durante", "e", "el", "ella",
            "ellas", "ellos", "en", "entre", "era", "erais", "eran", "eras", "eres", "es",
            "esa", "esas", "ese", "eso", "esos", "esta", "estaba", "estabais", "estaban",
            "estabas", "estad", "estada", "estadas", "estado", "estados", "estamos", "estando",
            "estar", "estaremos", "estará", "estarán", "estarás", "estaré", "estaréis", "estaría",
            "estaríais", "estaríamos", "estarían", "estarías", "estas", "este", "estemos", "esto",
            "estos", "estoy", "estuve", "estuviera", "estuvierais", "estuvieran", "estuvieras",
            "estuvieron", "estuviese", "estuvieseis", "estuviesen", "estuvieses", "estuvimos",
            "estuviste", "estuvisteis", "estuvo", "está", "estábamos", "estáis", "están", "estás",
            "esté", "estéis", "estén", "estés", "fue", "fuera", "fuerais", "fueran", "fueras",
            "fueron", "fuese", "fueseis", "fuesen", "fueses", "fui", "fuimos", "fuiste",
            "fuisteis", "ha", "habida", "habidas", "habido", "habidos", "habiendo", "habremos",
            "habrá", "habrán", "habrás", "habré", "habréis", "habría", "habríais", "habríamos",
            "habrían", "habrías", "habéis", "había", "habíais", "habíamos", "habían", "habías",
            "han", "has", "hasta", "hay", "haya", "hayamos", "hayan", "hayas", "hayáis", "he",
            "hemos", "hube", "hubiera", "hubierais", "hubieran", "hubieras", "hubieron", "hubiese",
            "hubieseis", "hubiesen", "hubieses", "hubimos", "hubiste", "hubisteis", "hubo", "la",
            "las", "le", "les", "lo", "los", "me", "mi", "mis", "mucho", "muchos", "muy", "más",
            "mí", "mía", "mías", "mío", "míos", "nada", "ni", "no", "nos", "nosotras", "nosotros",
            "nuestra", "nuestras", "nuestro", "nuestros", "o", "otra", "otras", "otro", "otros",
            "para", "pero", "poco", "por", "porque", "que", "quien", "quienes", "qué", "se",
            "sea", "seamos", "sean", "seas", "seáis", "sed", "seguir", "ser", "será", "seremos",
            "serán", "serás", "seré", "seréis", "sería", "seríais", "seríamos", "serían", "serías",
            "si", "sido", "siendo", "sin", "sobre", "sois", "somos", "son", "soy", "su", "sus",
            "suya", "suyas", "suyo", "suyos", "sí", "también", "tanto", "te", "tendremos", "tendrá",
            "tendrán", "tendrás", "tendré", "tendréis", "tendría", "tendríais", "tendríamos",
            "tendrían", "tendrías", "tened", "tenemos", "tenga", "tengamos", "tengan", "tengas",
            "tengáis", "tengo", "tenía", "teníais", "teníamos", "tenían", "tenías", "tenido",
            "teniendo", "tenéis", "tener", "tenga", "tengamos", "tengan", "tengas", "tengáis",
            "tengo", "tenía", "teníais", "teníamos", "tenían", "tenías", "tenido", "teniendo",
            "tenéis", "tener", "tenga", "tengamos", "tengan", "tengas", "tengáis", "tengo",
            "tenía", "teníais", "teníamos", "tenían", "tenías", "tenido", "teniendo", "tenéis",
            "tener", "tenga", "tengamos", "tengan", "tengas", "tengáis", "tengo", "tenía",
            "teníais", "teníamos", "tenían", "tenías", "tenido", "teniendo", "tenéis", "tener",
            "tenga", "tengamos", "tengan", "tengas", "tengáis", "tengo", "tenía", "teníais",
            "teníamos", "tenían", "tenías", "tenido", "teniendo", "tenéis", "tener", "tenga",
            "tengamos", "tengan", "tengas", "tengáis", "tengo", "tenía", "teníais", "teníamos",
            "tenían", "tenías", "tenido", "teniendo", "tenéis", "tener", "tenga", "tengamos",
            "tengan", "tengas", "tengáis", "tengo", "tenía", "teníais", "teníamos", "tenían",
            "tenías", "tenido", "teniendo", "tenéis", "tener", "todo", "todos", "tu", "tus",
            "tuve", "tuviera", "tuvierais", "tuvieran", "tuvieras", "tuvieron", "tuviese",
            "tuvieseis", "tuviesen", "tuvieses", "tuvimos", "tuviste", "tuvisteis", "tuvo",
            "tuya", "tuyas", "tuyo", "tuyos", "tú", "un", "una", "uno", "unos", "vosotras",
            "vosotros", "vuestra", "vuestras", "vuestro", "vuestros", "y", "ya", "yo", "él", "ésta",
            "éstas", "éste", "éstos", "última", "últimas", "último", "últimos"
        }
        
        # Filter keywords
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Count frequencies
        word_freq = {}
        for word in keywords:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get top keywords (sorted for deterministic ordering)
        top_keywords = sorted(word_freq.items(), key=lambda x: (-x[1], x[0]))[:10]
        
        return [word for word, freq in top_keywords]
    
    def _filter_and_rank_evidences_deterministic(self, evidences: List[ExtractedEvidence]) -> List[ExtractedEvidence]:
        """Filter and rank evidences with deterministic ordering"""
        
        # Sort evidences by multiple criteria for stable ordering
        sorted_evidences = sorted(
            evidences,
            key=lambda e: (
                -self._relevance_to_numeric(e.relevance),  # Higher relevance first
                -e.confidence,  # Higher confidence first
                e.evidence_id  # Stable secondary sort
            )
        )
        
        # Remove duplicates based on text similarity
        unique_evidences = []
        seen_texts = set()
        
        for evidence in sorted_evidences:
            # Simple deduplication by text hash
            text_hash = self.generate_stable_id(evidence.text.lower().strip())
            if text_hash not in seen_texts:
                seen_texts.add(text_hash)
                unique_evidences.append(evidence)
        
        return unique_evidences
    
    def _relevance_to_numeric(self, relevance: EvidenceRelevance) -> float:
        """Convert relevance to numeric value for sorting"""
        relevance_map = {
            EvidenceRelevance.ALTA: 3.0,
            EvidenceRelevance.MEDIA: 2.0,
            EvidenceRelevance.BAJA: 1.0,
            EvidenceRelevance.NULA: 0.0,
        }
        return relevance_map.get(relevance, 0.0)
    
    def _update_extraction_stats(self, evidences: List[ExtractedEvidence]):
        """Update extraction statistics"""
        self.extraction_stats["documents_processed"] += 1
        self.extraction_stats["evidences_extracted"] += len(evidences)
        
        for evidence in evidences:
            if evidence.relevance == EvidenceRelevance.ALTA:
                self.extraction_stats["high_relevance_count"] += 1
            elif evidence.relevance == EvidenceRelevance.MEDIA:
                self.extraction_stats["medium_relevance_count"] += 1
            elif evidence.relevance == EvidenceRelevance.BAJA:
                self.extraction_stats["low_relevance_count"] += 1
    
    def _generate_deterministic_output(self, evidences: List[ExtractedEvidence], operation_id: str) -> Dict[str, Any]:
        """Generate deterministic output structure"""
        
        # Convert evidences to dictionaries for JSON serialization
        evidence_dicts = []
        for evidence in evidences:
            evidence_dict = {
                "confidence": evidence.confidence,
                "context": {
                    "context_id": evidence.context.context_id,
                    "document_section": evidence.context.document_section,
                    "page_number": evidence.context.page_number,
                    "section_header": evidence.context.section_header,
                    "text_after": evidence.context.text_after,
                    "text_before": evidence.context.text_before,
                },
                "evidence_id": evidence.evidence_id,
                "evidence_type": evidence.evidence_type.value,
                "extraction_timestamp": evidence.extraction_timestamp,
                "keywords": evidence.keywords,
                "metadata": evidence.metadata,
                "relevance": evidence.relevance.value,
                "text": evidence.text,
            }
            evidence_dicts.append(evidence_dict)
        
        # Generate summary statistics
        summary = {
            "evidence_counts_by_relevance": {},
            "evidence_counts_by_type": {},
            "total_evidences": len(evidences),
        }
        
        # Count by relevance
        for evidence in evidences:
            relevance = evidence.relevance.value
            summary["evidence_counts_by_relevance"][relevance] = (
                summary["evidence_counts_by_relevance"].get(relevance, 0) + 1
            )
        
        # Count by type
        for evidence in evidences:
            evidence_type = evidence.evidence_type.value
            summary["evidence_counts_by_type"][evidence_type] = (
                summary["evidence_counts_by_type"].get(evidence_type, 0) + 1
            )
        
        # Sort counts for deterministic output
        summary["evidence_counts_by_relevance"] = self.sort_dict_by_keys(
            summary["evidence_counts_by_relevance"]
        )
        summary["evidence_counts_by_type"] = self.sort_dict_by_keys(
            summary["evidence_counts_by_type"]
        )
        
        output = {
            "component": self.component_name,
            "component_id": self.component_id,
            "evidences": evidence_dicts,
            "extraction_stats": self.sort_dict_by_keys(self.extraction_stats),
            "metadata": self.get_deterministic_metadata(),
            "operation_id": operation_id,
            "status": "success",
            "summary": summary,
            "timestamp": self._get_deterministic_timestamp(),
        }
        
        return self.sort_dict_by_keys(output)
    
    def save_artifact(self, data: Dict[str, Any], document_stem: str, output_dir: str = "canonical_flow/analysis") -> str:
        """
        Save extractor evidencias output to canonical_flow/analysis/ directory with standardized naming.
        
        Args:
            data: Extractor evidencias data to save
            document_stem: Base document identifier (without extension)
            output_dir: Output directory (defaults to canonical_flow/analysis)
            
        Returns:
            Path to saved artifact
        """
        try:
            # Create output directory if it doesn't exist
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            # Generate standardized filename
            filename = f"{document_stem}_extractor.json"
            artifact_path = Path(output_dir) / filename
            
            # Save with consistent JSON formatting
            with open(artifact_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"ExtractorEvidenciasContextual artifact saved: {artifact_path}")
            return str(artifact_path)
            
        except Exception as e:
            error_msg = f"Failed to save ExtractorEvidenciasContextual artifact to {output_dir}/{document_stem}_extractor.json: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)


# Backward compatibility functions
def extract_evidences(text: str, document_id: str = None) -> Dict[str, Any]:
    """Extract evidences from text"""
    extractor = ExtractorEvidenciasContextual()
    
    data = {"text": text}
    if document_id:
        data["document"] = {"id": document_id, "text": text}
    
    return extractor.process(data)


def process(data=None, context=None):
    """Backward compatible process function"""
    extractor = ExtractorEvidenciasContextual()
    result = extractor.process(data, context)
    
    # Save artifact if document_stem is provided
    if data and isinstance(data, dict) and 'document_stem' in data:
        extractor.save_artifact(result, data['document_stem'])
    
    return result
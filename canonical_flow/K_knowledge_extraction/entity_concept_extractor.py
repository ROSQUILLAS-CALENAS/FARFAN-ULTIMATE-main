"""
Entity and Concept Extractor Component for Knowledge Extraction Stage

This component extracts terms, entities, and concepts from document text with page metadata,
generating deterministic and reproducible inventories with stable tokenization and entity recognition.

Outputs:
- terms.json: Alphabetically sorted term frequencies with page anchors
- entities.json: Named entity recognition results with location metadata
- concepts.json: Extracted conceptual knowledge with page associations

All outputs use UTF-8 encoding and consistent JSON formatting for deterministic results.
"""

import json
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
import os

from total_ordering_base import TotalOrderingBase

# Optional NLP dependencies with fallbacks
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    spacy = None
    SPACY_AVAILABLE = False

try:
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import SnowballStemmer
    NLTK_AVAILABLE = True
    
    # Download required NLTK data if needed
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        
except ImportError:
    nltk = None
    NLTK_AVAILABLE = False

logger = logging.getLogger(__name__)


class EntityConceptExtractor(TotalOrderingBase):
    """
    Extracts terms, entities, and concepts from document text with deterministic results.
    
    Features:
    - Stable tokenization using consistent algorithms
    - Named entity recognition with configurable models
    - Page-anchored inventories with character offsets
    - Frequency counting with deterministic sorting
    - JSON output with UTF-8 encoding and alphabetical ordering
    """
    
    def __init__(self, language: str = "es", min_term_length: int = 2, 
                 max_term_length: int = 50, min_frequency: int = 1):
        """
        Initialize the entity concept extractor.
        
        Args:
            language: Language code for NLP models (default: 'es' for Spanish)
            min_term_length: Minimum character length for extracted terms
            max_term_length: Maximum character length for extracted terms
            min_frequency: Minimum frequency threshold for terms
        """
        super().__init__(component_name="EntityConceptExtractor")
        
        self.language = language
        self.min_term_length = min_term_length
        self.max_term_length = max_term_length
        self.min_frequency = min_frequency
        
        # Initialize NLP components
        self._init_spacy_model()
        self._init_nltk_components()
        
        # Concept patterns for Spanish documents (deterministic order)
        self._concept_patterns = sorted([
            r'\b(?:desarrollo|development)\s+(?:territorial|urban[oa]|sostenible)\b',
            r'\b(?:plan|programa|proyecto)\s+(?:de|para)\s+\w+\b',
            r'\b(?:objetivo|meta)\s+(?:estratégic[oa]|específic[oa])\b',
            r'\b(?:indicador|métrica)\s+(?:de|para)\s+\w+\b',
            r'\b(?:gestión|administración)\s+(?:públic[oa]|municipal)\b',
            r'\b(?:presupuesto|financiación|inversión)\s+(?:públic[oa]|municipal)\b',
            r'\b(?:participación|transparencia|accountability)\s+ciudadan[oa]\b',
            r'\b(?:sostenibilidad|medio\s+ambiente|cambio\s+climático)\b',
            r'\b(?:competitividad|productividad|innovación)\s+territorial\b',
            r'\b(?:riesgo|vulnerabilidad|resiliencia)\s+territorial\b'
        ])
        
        # Stop words for filtering (deterministic set)
        self._stop_words = self._get_stop_words()
        
        # Generate configuration hash
        config_data = {
            "language": self.language,
            "min_term_length": self.min_term_length,
            "max_term_length": self.max_term_length,
            "min_frequency": self.min_frequency,
            "spacy_available": SPACY_AVAILABLE,
            "nltk_available": NLTK_AVAILABLE,
            "spacy_model": getattr(self, '_spacy_model_name', None),
            "concept_patterns_count": len(self._concept_patterns),
            "stop_words_count": len(self._stop_words)
        }
        self._config_hash = self.generate_stable_id(config_data, prefix="ececonf")
        
        logger.info(f"Initialized EntityConceptExtractor with config hash: {self._config_hash}")
    
    def _init_spacy_model(self):
        """Initialize spaCy model for entity recognition."""
        self._spacy_model_name = None
        self.nlp = None
        
        if not SPACY_AVAILABLE:
            logger.warning("spaCy not available - entity recognition will be limited")
            return
            
        # Try to load appropriate language model
        model_options = {
            'es': ['es_core_news_sm', 'es_core_news_md', 'es_core_news_lg'],
            'en': ['en_core_web_sm', 'en_core_web_md', 'en_core_web_lg']
        }
        
        models_to_try = model_options.get(self.language, ['en_core_web_sm'])
        
        for model_name in models_to_try:
            try:
                self.nlp = spacy.load(model_name)
                self._spacy_model_name = model_name
                logger.info(f"Loaded spaCy model: {model_name}")
                break
            except OSError:
                continue
        
        if self.nlp is None:
            logger.warning(f"No spaCy model available for language '{self.language}'")
    
    def _init_nltk_components(self):
        """Initialize NLTK components for tokenization."""
        self.stemmer = None
        
        if not NLTK_AVAILABLE:
            logger.warning("NLTK not available - using basic tokenization")
            return
            
        try:
            # Initialize stemmer for the language
            if self.language == 'es':
                self.stemmer = SnowballStemmer('spanish')
            else:
                self.stemmer = SnowballStemmer('english')
            
            logger.info(f"Initialized NLTK stemmer for language: {self.language}")
        except Exception as e:
            logger.warning(f"Failed to initialize NLTK stemmer: {e}")
    
    def _get_stop_words(self) -> Set[str]:
        """Get stop words for the configured language."""
        stop_words = set()
        
        # Default Spanish stop words (sorted for determinism)
        default_spanish_stops = sorted([
            'el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'es', 'se',
            'no', 'te', 'lo', 'le', 'da', 'su', 'por', 'son', 'con',
            'para', 'como', 'las', 'del', 'los', 'una', 'al', 'más',
            'pero', 'sus', 'me', 'hasta', 'hay', 'donde', 'quien',
            'desde', 'todo', 'nos', 'durante', 'todos', 'uno', 'les',
            'ni', 'contra', 'otros', 'ese', 'eso', 'ante', 'ellos',
            'estas', 'fue', 'este', 'si', 'ya', 'entre', 'cuando'
        ])
        
        if NLTK_AVAILABLE:
            try:
                if self.language == 'es':
                    stop_words = set(stopwords.words('spanish'))
                else:
                    stop_words = set(stopwords.words('english'))
            except:
                # Fallback to default
                if self.language == 'es':
                    stop_words = set(default_spanish_stops)
        else:
            if self.language == 'es':
                stop_words = set(default_spanish_stops)
        
        return stop_words
    
    def process(self, document_text: str, page_metadata: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process document text and extract terms, entities, and concepts.
        
        Args:
            document_text: Full document text
            page_metadata: List of page metadata dictionaries with page numbers and text
            
        Returns:
            Dictionary with extraction results and metadata
            
        Raises:
            ValueError: If input is malformed or invalid
        """
        try:
            # Validate inputs
            self._validate_inputs(document_text, page_metadata)
            
            # Generate operation ID
            operation_inputs = {
                "text_length": len(document_text),
                "page_count": len(page_metadata),
                "text_hash": self.generate_stable_id(document_text, prefix="txt")[:16]
            }
            operation_id = self.generate_operation_id("extract_entities_concepts", operation_inputs)
            
            logger.info(f"Starting entity/concept extraction - Operation ID: {operation_id}")
            
            # Extract terms, entities, and concepts
            terms_result = self._extract_terms(document_text, page_metadata)
            entities_result = self._extract_entities(document_text, page_metadata)
            concepts_result = self._extract_concepts(document_text, page_metadata)
            
            # Generate output directory path
            stem = self._generate_document_stem(document_text, page_metadata)
            output_dir = Path("canonical_flow/knowledge") / stem
            
            # Write JSON artifacts
            self._write_json_artifacts(output_dir, terms_result, entities_result, concepts_result)
            
            # Compile final result
            result = {
                "operation_id": operation_id,
                "config_hash": self._config_hash,
                "document_stem": stem,
                "output_directory": str(output_dir),
                "extraction_summary": {
                    "terms_count": len(terms_result.get("terms", {})),
                    "entities_count": len(entities_result.get("entities", {})),
                    "concepts_count": len(concepts_result.get("concepts", {})),
                    "pages_processed": len(page_metadata)
                },
                "artifacts_generated": [
                    str(output_dir / "terms.json"),
                    str(output_dir / "entities.json"),
                    str(output_dir / "concepts.json")
                ]
            }
            
            # Update state hash
            self.update_state_hash(result)
            
            logger.info(f"Entity/concept extraction completed - {result['extraction_summary']}")
            return result
            
        except Exception as e:
            logger.error(f"Error in entity/concept extraction: {str(e)}")
            raise ValueError(f"Failed to process document: {str(e)}") from e
    
    def _validate_inputs(self, document_text: str, page_metadata: List[Dict[str, Any]]):
        """Validate input parameters."""
        if not isinstance(document_text, str) or not document_text.strip():
            raise ValueError("Document text must be a non-empty string")
        
        if not isinstance(page_metadata, list) or not page_metadata:
            raise ValueError("Page metadata must be a non-empty list")
        
        for i, page_data in enumerate(page_metadata):
            if not isinstance(page_data, dict):
                raise ValueError(f"Page metadata item {i} must be a dictionary")
            if "page_number" not in page_data:
                raise ValueError(f"Page metadata item {i} missing 'page_number'")
            if "text" not in page_data:
                raise ValueError(f"Page metadata item {i} missing 'text'")
    
    def _extract_terms(self, document_text: str, page_metadata: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract terms with frequency counts and page anchors."""
        terms_by_page = defaultdict(lambda: defaultdict(list))
        global_term_counts = Counter()
        
        for page_data in page_metadata:
            page_num = page_data["page_number"]
            page_text = page_data.get("text", "")
            
            if not page_text.strip():
                continue
            
            # Tokenize text deterministically
            tokens = self._tokenize_text(page_text)
            
            # Extract valid terms
            for token, positions in tokens.items():
                if self._is_valid_term(token):
                    terms_by_page[page_num][token].extend(positions)
                    global_term_counts[token] += len(positions)
        
        # Filter by frequency and build final structure
        filtered_terms = {}
        for term, count in global_term_counts.items():
            if count >= self.min_frequency:
                term_data = {
                    "term": term,
                    "frequency": count,
                    "pages": {}
                }
                
                for page_num, page_terms in terms_by_page.items():
                    if term in page_terms:
                        positions = page_terms[term]
                        term_data["pages"][str(page_num)] = {
                            "frequency": len(positions),
                            "positions": sorted(positions)  # Deterministic order
                        }
                
                # Sort pages by page number for determinism
                term_data["pages"] = dict(sorted(term_data["pages"].items(), 
                                               key=lambda x: int(x[0])))
                filtered_terms[term] = term_data
        
        return {
            "extraction_type": "terms",
            "total_terms": len(filtered_terms),
            "terms": dict(sorted(filtered_terms.items()))  # Alphabetical order
        }
    
    def _extract_entities(self, document_text: str, page_metadata: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract named entities with location metadata."""
        entities_by_page = defaultdict(lambda: defaultdict(list))
        global_entity_counts = Counter()
        
        if not self.nlp:
            # Fallback entity extraction using simple patterns
            return self._extract_entities_fallback(document_text, page_metadata)
        
        for page_data in page_metadata:
            page_num = page_data["page_number"]
            page_text = page_data.get("text", "")
            
            if not page_text.strip():
                continue
            
            # Process with spaCy
            doc = self.nlp(page_text)
            
            for ent in doc.ents:
                entity_text = ent.text.strip()
                entity_label = ent.label_
                start_char = ent.start_char
                end_char = ent.end_char
                
                if self._is_valid_entity(entity_text, entity_label):
                    entity_key = f"{entity_text}|{entity_label}"
                    position_data = {
                        "start": start_char,
                        "end": end_char,
                        "context": self._get_entity_context(page_text, start_char, end_char)
                    }
                    
                    entities_by_page[page_num][entity_key].append(position_data)
                    global_entity_counts[entity_key] += 1
        
        # Build final entity structure
        filtered_entities = {}
        for entity_key, count in global_entity_counts.items():
            if count >= self.min_frequency:
                entity_text, entity_label = entity_key.split('|', 1)
                entity_data = {
                    "text": entity_text,
                    "label": entity_label,
                    "frequency": count,
                    "pages": {}
                }
                
                for page_num, page_entities in entities_by_page.items():
                    if entity_key in page_entities:
                        positions = page_entities[entity_key]
                        entity_data["pages"][str(page_num)] = {
                            "frequency": len(positions),
                            "positions": sorted(positions, key=lambda x: x["start"])
                        }
                
                # Sort pages by page number for determinism
                entity_data["pages"] = dict(sorted(entity_data["pages"].items(),
                                                 key=lambda x: int(x[0])))
                filtered_entities[entity_key] = entity_data
        
        return {
            "extraction_type": "entities",
            "total_entities": len(filtered_entities),
            "entities": dict(sorted(filtered_entities.items()))  # Alphabetical order
        }
    
    def _extract_entities_fallback(self, document_text: str, page_metadata: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Fallback entity extraction using simple patterns."""
        entities_by_page = defaultdict(lambda: defaultdict(list))
        global_entity_counts = Counter()
        
        # Simple patterns for common entity types
        entity_patterns = [
            (r'\b[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)*\b', 'PERSON'),
            (r'\b(?:Municipio|Ciudad|Departamento)\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+\b', 'LOCATION'),
            (r'\b\d{4}\b', 'DATE'),
            (r'\$\s*[\d,]+(?:\.\d{2})?\b', 'MONEY')
        ]
        
        for page_data in page_metadata:
            page_num = page_data["page_number"]
            page_text = page_data.get("text", "")
            
            if not page_text.strip():
                continue
            
            for pattern, label in entity_patterns:
                for match in re.finditer(pattern, page_text):
                    entity_text = match.group().strip()
                    start_char = match.start()
                    end_char = match.end()
                    
                    if self._is_valid_entity(entity_text, label):
                        entity_key = f"{entity_text}|{label}"
                        position_data = {
                            "start": start_char,
                            "end": end_char,
                            "context": self._get_entity_context(page_text, start_char, end_char)
                        }
                        
                        entities_by_page[page_num][entity_key].append(position_data)
                        global_entity_counts[entity_key] += 1
        
        # Build final structure (same as spaCy version)
        filtered_entities = {}
        for entity_key, count in global_entity_counts.items():
            if count >= self.min_frequency:
                entity_text, entity_label = entity_key.split('|', 1)
                entity_data = {
                    "text": entity_text,
                    "label": entity_label,
                    "frequency": count,
                    "pages": {}
                }
                
                for page_num, page_entities in entities_by_page.items():
                    if entity_key in page_entities:
                        positions = page_entities[entity_key]
                        entity_data["pages"][str(page_num)] = {
                            "frequency": len(positions),
                            "positions": sorted(positions, key=lambda x: x["start"])
                        }
                
                entity_data["pages"] = dict(sorted(entity_data["pages"].items(),
                                                 key=lambda x: int(x[0])))
                filtered_entities[entity_key] = entity_data
        
        return {
            "extraction_type": "entities_fallback",
            "total_entities": len(filtered_entities),
            "entities": dict(sorted(filtered_entities.items()))
        }
    
    def _extract_concepts(self, document_text: str, page_metadata: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract conceptual knowledge using pattern matching."""
        concepts_by_page = defaultdict(lambda: defaultdict(list))
        global_concept_counts = Counter()
        
        for page_data in page_metadata:
            page_num = page_data["page_number"]
            page_text = page_data.get("text", "")
            
            if not page_text.strip():
                continue
            
            # Apply concept patterns
            for pattern in self._concept_patterns:
                for match in re.finditer(pattern, page_text, re.IGNORECASE):
                    concept_text = match.group().strip()
                    start_char = match.start()
                    end_char = match.end()
                    
                    if self._is_valid_concept(concept_text):
                        # Normalize concept for grouping
                        normalized_concept = self._normalize_concept(concept_text)
                        
                        position_data = {
                            "original_text": concept_text,
                            "start": start_char,
                            "end": end_char,
                            "context": self._get_entity_context(page_text, start_char, end_char),
                            "pattern_matched": pattern
                        }
                        
                        concepts_by_page[page_num][normalized_concept].append(position_data)
                        global_concept_counts[normalized_concept] += 1
        
        # Build final concept structure
        filtered_concepts = {}
        for concept, count in global_concept_counts.items():
            if count >= self.min_frequency:
                concept_data = {
                    "concept": concept,
                    "frequency": count,
                    "pages": {}
                }
                
                for page_num, page_concepts in concepts_by_page.items():
                    if concept in page_concepts:
                        occurrences = page_concepts[concept]
                        concept_data["pages"][str(page_num)] = {
                            "frequency": len(occurrences),
                            "occurrences": sorted(occurrences, key=lambda x: x["start"])
                        }
                
                # Sort pages by page number for determinism
                concept_data["pages"] = dict(sorted(concept_data["pages"].items(),
                                                  key=lambda x: int(x[0])))
                filtered_concepts[concept] = concept_data
        
        return {
            "extraction_type": "concepts",
            "total_concepts": len(filtered_concepts),
            "concepts": dict(sorted(filtered_concepts.items()))  # Alphabetical order
        }
    
    def _tokenize_text(self, text: str) -> Dict[str, List[int]]:
        """Tokenize text and return tokens with their positions."""
        tokens = defaultdict(list)
        
        if NLTK_AVAILABLE:
            # Use NLTK tokenization
            try:
                word_tokens = word_tokenize(text, language='spanish' if self.language == 'es' else 'english')
                
                # Find positions of tokens in original text
                search_start = 0
                for token in word_tokens:
                    token_lower = token.lower()
                    if token_lower.isalpha():  # Only alphabetic tokens
                        # Find token position in text
                        pos = text.lower().find(token_lower, search_start)
                        if pos != -1:
                            tokens[token_lower].append(pos)
                            search_start = pos + len(token)
                
                return dict(tokens)
            except:
                pass  # Fall back to regex
        
        # Fallback regex tokenization
        for match in re.finditer(r'\b[a-záéíóúñA-ZÁÉÍÓÚÑ]+\b', text):
            token = match.group().lower()
            if token.isalpha():
                tokens[token].append(match.start())
        
        return dict(tokens)
    
    def _is_valid_term(self, term: str) -> bool:
        """Check if a term is valid for extraction."""
        if len(term) < self.min_term_length or len(term) > self.max_term_length:
            return False
        if term in self._stop_words:
            return False
        if not re.match(r'^[a-záéíóúñA-ZÁÉÍÓÚÑ]+$', term):
            return False
        return True
    
    def _is_valid_entity(self, entity_text: str, entity_label: str) -> bool:
        """Check if an entity is valid for extraction."""
        if len(entity_text.strip()) < 2:
            return False
        if entity_text.strip() in self._stop_words:
            return False
        return True
    
    def _is_valid_concept(self, concept_text: str) -> bool:
        """Check if a concept is valid for extraction."""
        if len(concept_text.strip()) < 5:
            return False
        return True
    
    def _normalize_concept(self, concept_text: str) -> str:
        """Normalize concept text for grouping."""
        # Convert to lowercase and normalize whitespace
        normalized = re.sub(r'\s+', ' ', concept_text.lower().strip())
        
        # Stem if stemmer available
        if self.stemmer:
            words = normalized.split()
            stemmed_words = []
            for word in words:
                try:
                    stemmed = self.stemmer.stem(word)
                    stemmed_words.append(stemmed)
                except:
                    stemmed_words.append(word)
            normalized = ' '.join(stemmed_words)
        
        return normalized
    
    def _get_entity_context(self, text: str, start: int, end: int, context_window: int = 50) -> str:
        """Get context around an entity occurrence."""
        context_start = max(0, start - context_window)
        context_end = min(len(text), end + context_window)
        return text[context_start:context_end].strip()
    
    def _generate_document_stem(self, document_text: str, page_metadata: List[Dict[str, Any]]) -> str:
        """Generate a deterministic stem for the document."""
        # Use document characteristics to create stem
        stem_data = {
            "text_length": len(document_text),
            "page_count": len(page_metadata),
            "first_page_text": page_metadata[0].get("text", "")[:100] if page_metadata else "",
            "text_hash": self.generate_stable_id(document_text[:1000], prefix="")[:8]
        }
        
        return self.generate_stable_id(stem_data, prefix="doc")[:16]
    
    def _write_json_artifacts(self, output_dir: Path, terms_result: Dict, 
                            entities_result: Dict, concepts_result: Dict):
        """Write JSON artifacts to the output directory."""
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Write artifacts with UTF-8 encoding and consistent formatting
        artifacts = [
            ("terms.json", terms_result),
            ("entities.json", entities_result),
            ("concepts.json", concepts_result)
        ]
        
        for filename, data in artifacts:
            file_path = output_dir / filename
            
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(
                        data,
                        f,
                        ensure_ascii=False,
                        indent=2,
                        sort_keys=True,
                        separators=(',', ': ')
                    )
                
                logger.debug(f"Wrote artifact: {file_path}")
            except Exception as e:
                logger.error(f"Failed to write {filename}: {e}")
                raise
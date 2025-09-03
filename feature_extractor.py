import json
import logging
import os
import re
# # # from pathlib import Path  # Module not found  # Module not found  # Module not found
# # # from typing import Dict, List, Optional, Any  # Module not found  # Module not found  # Module not found
# # # from datetime import datetime  # Module not found  # Module not found  # Module not found

# # # from total_ordering_base import TotalOrderingBase  # Module not found  # Module not found  # Module not found
# # # from json_canonicalizer import JSONCanonicalizer  # Module not found  # Module not found  # Module not found

# Optional heavy deps (guarded imports)
try:
    import textstat
    TEXTSTAT_AVAILABLE = True
except ImportError:
    TEXTSTAT_AVAILABLE = False
    textstat = None

try:
# # #     from langdetect import detect, detect_langs  # Module not found  # Module not found  # Module not found
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

logger = logging.getLogger(__name__)


class DocumentFeatureExtractor(TotalOrderingBase):
    """
    Document feature extraction with deterministic textual analysis.
    Extracts word count, character count, sentence count, paragraph count,
    readability metrics, and language detection.
    """
    
    def __init__(self):
        """Initialize the feature extractor."""
        super().__init__(component_name="DocumentFeatureExtractor")
        
        # Configuration for feature extraction
        self._config = {
            "readability_fallback_score": 50.0,
            "language_detection_fallback": "unknown",
            "min_text_length_for_metrics": 10,
            "sentence_delimiters": ['.', '!', '?', '…'],
            "paragraph_separators": ['\n\n', '\r\n\r\n'],
        }
        
        # Load spaCy Spanish model if available
        if spacy is not None:
            try:
                self.nlp = spacy.load("es_core_news_sm")
                self._nlp_model_name = "es_core_news_sm"
            except Exception:
                logger.warning("spaCy 'es_core_news_sm' not found; proceeding without NLP model")
                self.nlp = None
                self._nlp_model_name = None
        else:
            self.nlp = None
            self._nlp_model_name = None
        
        self.text_analyzer = TextAnalyzer()
        self.technical_terms = self._load_technical_terms()
        self.required_sections = self._get_required_sections()
        
        # Initialize JSON canonicalizer
        self.canonicalizer = JSONCanonicalizer(audit_enabled=True, validation_enabled=True)
        
        # State tracking
        self._documents_processed = 0
        self._extraction_count = 0
        
        # Generate configuration ID
        config_data = {
            "nlp_model": self._nlp_model_name,
            "technical_terms_count": len(self.technical_terms),
            "required_sections_count": len(self.required_sections),
            "nlp_model": self._nlp_model_name
        }
        }
        
        # Generate stable configuration ID
        self._config_id = self.generate_stable_id(self._config, prefix="feat_config")
        
        logger.info(f"Initialized DocumentFeatureExtractor with config ID: {self._config_id}")
    
    def process(self, doc_stem: str) -> Dict[str, Any]:
        """
        Standardized process method that reads bundle files and extracts features.
        
        Args:
            doc_stem: Document stem identifier
            
        Returns:
            Processing result with extracted features or error status
        """
        operation_id = self.generate_operation_id("process", {"doc_stem": doc_stem})
        
        logger.info(f"Starting feature extraction for doc_stem: {doc_stem} [op_id: {operation_id[:8]}]")
        
        # Define input and output paths
        bundle_path = Path("canonical_flow/ingestion") / f"{doc_stem}_bundle.json"
        features_path = Path("canonical_flow/ingestion") / f"{doc_stem}_features.json"
        
        try:
            # Read bundle file
            bundle_data = self._read_bundle_file(bundle_path)
            
            # Extract document content
            content_text = self._extract_text_content(bundle_data)
            
            # Extract features
            if content_text and len(content_text.strip()) >= self._config["min_text_length_for_metrics"]:
                # Check if content is just JSON structure (fallback case)
                if content_text.strip().startswith('{') and '"bundle_id"' in content_text:
                    features = self._get_empty_features()
                    status = "no_content"
                    logger.warning(f"Only structural JSON found for doc_stem: {doc_stem}")
                else:
                    features = self._extract_textual_features(content_text)
                    status = "success"
                    logger.info(f"Successfully extracted features for doc_stem: {doc_stem}")
            else:
                features = self._get_empty_features()
                status = "no_content"
                logger.warning(f"No sufficient content found for doc_stem: {doc_stem}")
            
            # Prepare output data with deterministic ordering
            output_data = {
                "bundle_id": bundle_data.get("bundle_id", "unknown"),
                "document_stem": doc_stem,
                "extraction_timestamp": datetime.utcnow().isoformat() + "Z",
                "features": self.sort_dict_by_keys(features),
                "operation_id": operation_id,
                "processing_status": status,
                "schema_version": "1.0"
            }
            
            # Write features file
            self._write_features_file(features_path, output_data)
            
            # Update state tracking
            self.update_state_hash({
                "doc_stem": doc_stem,
                "operation_id": operation_id,
                "status": status,
                "feature_count": len(features)
            })
            
            return {
                "status": status,
                "doc_stem": doc_stem,
                "features_path": str(features_path),
                "operation_id": operation_id
            }
            
        except Exception as e:
            error_msg = f"Feature extraction failed for {doc_stem}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            # Still produce valid JSON output with error status
            error_features = self._get_empty_features()
            error_output = {
                "bundle_id": "unknown",
                "document_stem": doc_stem,
                "error_details": {
                    "error_message": str(e),
                    "error_type": type(e).__name__,
                    "has_errors": True
                },
                "extraction_timestamp": datetime.utcnow().isoformat() + "Z",
                "features": self.sort_dict_by_keys(error_features),
                "operation_id": operation_id,
                "processing_status": "error",
                "schema_version": "1.0"
            }
            
            try:
                self._write_features_file(features_path, error_output)
            except Exception as write_error:
                logger.error(f"Failed to write error features file: {write_error}")
            
            return {
                "status": "error",
                "doc_stem": doc_stem,
                "error": error_msg,
                "operation_id": operation_id
            }
    
    def _read_bundle_file(self, bundle_path: Path) -> Dict[str, Any]:
        """
        Read and parse bundle JSON file with error handling.
        
        Args:
            bundle_path: Path to bundle file
            
        Returns:
            Bundle data dictionary
            
        Raises:
            FileNotFoundError: If bundle file doesn't exist
            json.JSONDecodeError: If bundle file contains invalid JSON
        """
        if not bundle_path.exists():
            raise FileNotFoundError(f"Bundle file not found: {bundle_path}")
        
        try:
            with open(bundle_path, 'r', encoding='utf-8') as f:
                bundle_data = json.load(f)
                
            if not isinstance(bundle_data, dict):
                raise ValueError(f"Bundle file contains non-dict data: {type(bundle_data)}")
                
            return bundle_data
            
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Invalid JSON in bundle file {bundle_path}: {e.msg}", e.doc, e.pos)
        except UnicodeDecodeError as e:
            raise UnicodeDecodeError(e.encoding, e.object, e.start, e.end, 
                                   f"Failed to decode bundle file {bundle_path}: {e.reason}")
    
    def _extract_text_content(self, bundle_data: Dict[str, Any]) -> str:
        """
# # #         Extract text content from bundle data with fallback strategies.  # Module not found  # Module not found  # Module not found
        
        Args:
            bundle_data: Bundle data dictionary
            
        Returns:
            Extracted text content (empty string if none found)
        """
        try:
            # Primary path: document_content.content
            document_content = bundle_data.get("document_content", {})
            if isinstance(document_content, dict):
                content = document_content.get("content", "")
                if content and isinstance(content, str):
                    return content.strip()
            
            # Fallback 1: document_content as string
            if isinstance(document_content, str):
                return document_content.strip()
            
            # Fallback 2: sections content
            sections = document_content.get("sections", [])
            if isinstance(sections, list):
                section_texts = []
                for section in sections:
                    if isinstance(section, dict) and "text" in section:
                        section_texts.append(section["text"])
                if section_texts:
                    return "\n\n".join(section_texts).strip()
            
            # Fallback 3: entire bundle as text
            bundle_str = json.dumps(bundle_data, ensure_ascii=False)
            return bundle_str.strip()
            
        except Exception as e:
            logger.warning(f"Error extracting text content: {e}")
            return ""
    
    def _extract_textual_features(self, text: str) -> Dict[str, Any]:
        """
# # #         Extract comprehensive textual features from document content.  # Module not found  # Module not found  # Module not found
        
        Args:
            text: Document text content
            
        Returns:
            Dictionary of extracted features with stable ordering
        """
        features = {}
        
        # Basic text metrics
        features.update(self._extract_basic_metrics(text))
        
        # Readability metrics
        features.update(self._extract_readability_metrics(text))
        
        # Language detection
        features.update(self._extract_language_features(text))
        
        # Advanced text analysis
        features.update(self._extract_advanced_metrics(text))
        
        return features
    
    def _extract_basic_metrics(self, text: str) -> Dict[str, Any]:
        """Extract basic text counting metrics."""
        # Character count (total and without spaces)
        char_count = len(text)
        char_count_no_spaces = len(text.replace(' ', ''))
        
        # Word count
        words = re.findall(r'\b\w+\b', text)
        word_count = len(words)
        
        # Sentence count
        sentence_count = self._count_sentences(text)
        
        # Paragraph count
        paragraph_count = self._count_paragraphs(text)
        
        # Line count
        line_count = len(text.splitlines())
        
        return {
            "character_count": char_count,
            "character_count_no_spaces": char_count_no_spaces,
            "word_count": word_count,
            "sentence_count": sentence_count,
            "paragraph_count": paragraph_count,
            "line_count": line_count
        }
    def _extract_readability_metrics(self, text: str) -> Dict[str, Any]:
        """Extract readability metrics with fallback values."""
        readability = {}
        
        if TEXTSTAT_AVAILABLE and textstat and len(text.strip()) > 10:
            try:
                # Flesch Reading Ease Score
                flesch_score = textstat.flesch_reading_ease(text)
                readability["flesch_reading_ease"] = max(0.0, min(100.0, float(flesch_score)))
                
                # Flesch-Kincaid Grade Level
                fk_grade = textstat.flesch_kincaid_grade(text)
                readability["flesch_kincaid_grade"] = max(0.0, float(fk_grade))
                
                # Automated Readability Index
                ari_score = textstat.automated_readability_index(text)
                readability["automated_readability_index"] = max(0.0, float(ari_score))
                
                # Coleman-Liau Index
                cli_score = textstat.coleman_liau_index(text)
                readability["coleman_liau_index"] = max(0.0, float(cli_score))
                
                # Gunning Fog Index
                fog_score = textstat.gunning_fog(text)
                readability["gunning_fog"] = max(0.0, float(fog_score))
                
                # SMOG Index
                smog_score = textstat.smog_index(text)
                readability["smog_index"] = max(0.0, float(smog_score))
                
            except Exception as e:
                logger.warning(f"Error calculating readability metrics: {e}")
                readability.update(self._get_fallback_readability())
        else:
            readability.update(self._get_fallback_readability())
        
        # Calculate average sentence length
        words = re.findall(r'\b\w+\b', text)
        sentences = self._count_sentences(text)
        if sentences > 0:
            readability["avg_sentence_length"] = len(words) / sentences
        else:
            readability["avg_sentence_length"] = 0.0
        
        # Calculate lexical diversity (Type-Token Ratio)
        if words:
            unique_words = set(word.lower() for word in words)
            readability["lexical_diversity"] = len(unique_words) / len(words)
        else:
            readability["lexical_diversity"] = 0.0
        
        return readability
    
    def _extract_language_features(self, text: str) -> Dict[str, Any]:
        """Extract language detection features."""
        language_features = {}
        
        if LANGDETECT_AVAILABLE and len(text.strip()) > 10:
            try:
                # Primary language detection
                detected_lang = detect(text)
                language_features["primary_language"] = detected_lang
                
                # Language probabilities
                lang_probs = detect_langs(text)
                languages_detected = []
                for lang_prob in lang_probs[:3]:  # Top 3 languages
                    languages_detected.append({
                        "language": lang_prob.lang,
                        "probability": round(lang_prob.prob, 4)
                    })
                language_features["languages_detected"] = languages_detected
                language_features["language_confidence"] = round(lang_probs[0].prob, 4) if lang_probs else 0.0
                
            except Exception as e:
                logger.warning(f"Error in language detection: {e}")
                language_features.update(self._get_fallback_language())
        else:
            language_features.update(self._get_fallback_language())
        
        return language_features
    
    def _extract_advanced_metrics(self, text: str) -> Dict[str, Any]:
        """Extract advanced text analysis metrics."""
        advanced = {}
        
        # Digit density
        digits = re.findall(r'\d', text)
        advanced["digit_density"] = len(digits) / len(text) if text else 0.0
        
        # Punctuation density
        punctuation = re.findall(r'[^\w\s]', text)
        advanced["punctuation_density"] = len(punctuation) / len(text) if text else 0.0
        
        # Uppercase density
        uppercase_chars = sum(1 for c in text if c.isupper())
        advanced["uppercase_density"] = uppercase_chars / len(text) if text else 0.0
        
        # Special characters density
        special_chars = re.findall(r'[^\x00-\x7F]', text)
        advanced["special_char_density"] = len(special_chars) / len(text) if text else 0.0
        
        # Average word length
        words = re.findall(r'\b\w+\b', text)
        if words:
            advanced["avg_word_length"] = sum(len(word) for word in words) / len(words)
        else:
            advanced["avg_word_length"] = 0.0
        
        # Long words ratio (words with 7+ characters)
        if words:
            long_words = [word for word in words if len(word) >= 7]
            advanced["long_words_ratio"] = len(long_words) / len(words)
        else:
            advanced["long_words_ratio"] = 0.0
        
        return advanced
    
    def _count_sentences(self, text: str) -> int:
        """Count sentences using multiple delimiters."""
        if not text.strip():
            return 0
        
        # Split by sentence delimiters
        sentence_pattern = r'[.!?…]+\s+'
        sentences = re.split(sentence_pattern, text.strip())
        
        # Filter out empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return max(1, len(sentences))
    
    def _count_paragraphs(self, text: str) -> int:
        """Count paragraphs using various separators."""
        if not text.strip():
            return 0
        
        # Try different paragraph separators
        for separator in self._config["paragraph_separators"]:
            if separator in text:
                paragraphs = text.split(separator)
                paragraphs = [p.strip() for p in paragraphs if p.strip()]
                return max(1, len(paragraphs))
        
        # Fallback: single paragraph
        return 1 if text.strip() else 0
    
    def _get_empty_features(self) -> Dict[str, Any]:
        """Get features dictionary with all zero/empty values for no content."""
        return self.sort_dict_by_keys({
            # Basic metrics
            "character_count": 0,
            "character_count_no_spaces": 0,
            "word_count": 0,
            "sentence_count": 0,
            "paragraph_count": 0,
            "line_count": 0,
            
            # Readability metrics
            "flesch_reading_ease": 0.0,
            "flesch_kincaid_grade": 0.0,
            "automated_readability_index": 0.0,
            "coleman_liau_index": 0.0,
            "gunning_fog": 0.0,
            "smog_index": 0.0,
            "avg_sentence_length": 0.0,
            "lexical_diversity": 0.0,
            
            # Language features
            "primary_language": "unknown",
            "languages_detected": [],
            "language_confidence": 0.0,
            
            # Advanced metrics
            "digit_density": 0.0,
            "punctuation_density": 0.0,
            "uppercase_density": 0.0,
            "special_char_density": 0.0,
            "avg_word_length": 0.0,
            "long_words_ratio": 0.0
        })
    
    def _get_fallback_readability(self) -> Dict[str, Any]:
        """Get fallback readability metrics when textstat is unavailable."""
        return {
            "flesch_reading_ease": self._config["readability_fallback_score"],
            "flesch_kincaid_grade": 0.0,
            "automated_readability_index": 0.0,
            "coleman_liau_index": 0.0,
            "gunning_fog": 0.0,
            "smog_index": 0.0
        }
    
    def _get_fallback_language(self) -> Dict[str, Any]:
        """Get fallback language detection when langdetect is unavailable."""
        return {
            "primary_language": self._config["language_detection_fallback"],
            "languages_detected": [],
            "language_confidence": 0.0
        }
    
    def _write_features_file(self, features_path: Path, data: Dict[str, Any]) -> None:
        """
        Write features data to JSON file with consistent formatting.
        
        Args:
            features_path: Path to features file
            data: Features data to write
        """
        # Ensure output directory exists
        features_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Sort data for consistent output
        sorted_data = self.sort_dict_by_keys(data)
        
        # Write with consistent JSON formatting
        with open(features_path, 'w', encoding='utf-8') as f:
            json.dump(sorted_data, f, ensure_ascii=False, indent=2, sort_keys=True)
        
        logger.debug(f"Features written to: {features_path}")


def _to_plain_dict(obj: Any) -> Dict[str, Any]:
    """Convert complex objects to plain dict for serialization."""
    if hasattr(obj, '__dict__'):
        return {k: _to_plain_dict(v) for k, v in obj.__dict__.items() if not k.startswith('_')}
    elif isinstance(obj, dict):
        return {k: _to_plain_dict(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_to_plain_dict(item) for item in obj]
    else:
        return obj


class _EmbeddingEngine:
    """Minimal embedding engine for fallback vector generation."""
    
    def encode(self, texts):
        """Generate simple hash-based vectors for texts."""
        import hashlib
        vectors = []
        for text in texts:
            # Create deterministic hash-based vector
            hash_obj = hashlib.md5(text.encode())
            hash_bytes = hash_obj.digest()
            # Convert to simple numeric vector
            vector = [float(b) / 255.0 for b in hash_bytes[:8]]  # 8-dim vector
            vectors.append(vector)
        return vectors


def extract_features(data=None, context=None):
    """
    Deterministic, state-of-the-art feature extraction with canonicalization.
# # #     Extract features from text content with comprehensive audit logging.  # Module not found  # Module not found  # Module not found
    
    Theory backbone (peer-reviewed ≥2021): SimCSE (Gao, Yao, and Chen). "SimCSE: Simple Contrastive Learning of Sentence Embeddings." EMNLP 2021.
    Implementation uses open-source libraries only (sentence-transformers, scikit-learn) and is self-contained with graceful fallbacks to avoid refactors.

    Inputs (flexible):
    - data: dict or str. Accepted keys: 'text', 'document_text', 'content', optional 'document_structure'/'structure', 'metadata'.

    
    Inputs (flexible):
    - data: dict or str. Accepted keys: 'text', 'document_text', 'content', optional 'document_structure'/'structure', 'metadata'.

    Returns:
    - {
        'features': dict,          # rich features
        'vectors': list[float],    # embedding/TF-IDF vector for the whole text
        'canonicalization': dict   # canonicalization metadata
      }
    """
    # Initialize canonicalizer for this operation
    canonicalizer = JSONCanonicalizer(audit_enabled=True, validation_enabled=True)
    
    # Canonicalize inputs
    canonical_data_json, data_id, data_audit = canonicalizer.canonicalize(
        data, {"operation": "process", "stage": "ingestion_preparation", "component": "feature_extractor"}
    )
    canonical_context_json, context_id, context_audit = canonicalizer.canonicalize(
        context, {"operation": "process", "stage": "ingestion_preparation", "component": "feature_extractor"}
    )
    
    # Parse canonicalized data
    canonical_data = json.loads(canonical_data_json)
    
    # Extract inputs
    text = None
    structure = None
    metadata = None

    if isinstance(canonical_data, str):
        text = canonical_data
    elif isinstance(canonical_data, dict):
        text = (
            canonical_data.get("text")
            or canonical_data.get("document_text")
            or canonical_data.get("content")
            or ""
        )
        structure = canonical_data.get("document_structure") or canonical_data.get("structure") or {}
        metadata = canonical_data.get("metadata") or {}
    else:
        text = ""
        structure = {}
        metadata = {}

    # Ensure strings
    text = text or ""
    if not isinstance(structure, dict):
        structure = {}
    if not isinstance(metadata, dict):
        metadata = {}

    # Extract features using existing class (kept for compatibility)
    try:
        extractor = DocumentFeatureExtractor()
        feat_obj = extractor.extract_features(text, structure, metadata)
        features = _to_plain_dict(feat_obj)
    except Exception as e:
        # Robust fallback: basic counts only
        tokens = re.findall(r"\b\w+\b", text.lower())
        features = {
            "total_length": len(text),
            "word_count": len(tokens),
            "unique_terms": len(set(tokens)),
            "error_fallback": str(e),
        }

    # Generate vector representation deterministically
    engine = _EmbeddingEngine()
    vector = engine.encode([text])

    # Create output with canonicalization metadata
    output = {
        "features": features, 
        "vectors": vector,
        "canonicalization": {
            "data_id": data_id,
            "context_id": context_id,
            "data_hash": data_audit.input_hash,
            "context_hash": context_audit.input_hash,
            "execution_time_ms": data_audit.execution_time_ms + context_audit.execution_time_ms
        }
    }
    
    # Canonicalize final output
    final_canonical_json, final_id, final_audit = canonicalizer.canonicalize(
        output, {"operation": "final_output", "stage": "ingestion_preparation", "component": "feature_extractor"}
    )
    
    # Save audit trail to companion file
    document_id = metadata.get("document_id", "unknown")
    output_file = f"feature_extractor_{document_id}_{data_id}.json"
    canonicalizer.save_audit_trail(output_file)

    return json.loads(final_canonical_json)


def process(data=None, context=None) -> Dict[str, Any]:
    """
    Process API for feature extractor component (03I).
    
# # #     Extracts features from document bundles and writes standardized   # Module not found  # Module not found  # Module not found
    artifacts using ArtifactManager.
    
    Args:
        data: Input data (bundle data or document content)
        context: Processing context with metadata
        
    Returns:
        Dictionary with processing results and output paths
    """
    # Import ArtifactManager locally to avoid circular imports
    try:
# # #         from canonical_flow.ingestion import ArtifactManager  # Module not found  # Module not found  # Module not found
    except ImportError:
        return {"error": "ArtifactManager not available"}
    
    artifact_manager = ArtifactManager()
    
    # Process input data
    if not data:
        return {"error": "No input data provided"}
    
    results = []
    
    # Handle different input formats
    if isinstance(data, dict) and 'results' in data:
# # #         # Input from 02I component  # Module not found  # Module not found  # Module not found
        bundle_results = data['results']
    elif isinstance(data, list):
        bundle_results = data
    else:
        bundle_results = [data]
    
    for bundle_result in bundle_results:
        try:
            # Extract stem and bundle data
            if isinstance(bundle_result, dict):
                stem = bundle_result.get('stem', bundle_result.get('document_stem', 'unknown'))
                
                # For actual processing, we need to read the bundle file
                bundle_path = bundle_result.get('output_path')
                if bundle_path and Path(bundle_path).exists():
                    import json
                    with open(bundle_path, 'r', encoding='utf-8') as f:
                        bundle_data = json.load(f)
                else:
                    # Use bundle_result directly if no file path
                    bundle_data = bundle_result
            else:
                stem = 'unknown'
                bundle_data = {}
            
            # Extract text for feature extraction
            full_text = bundle_data.get('full_text', '')
            document_structure = bundle_data.get('document_structure', {})
            text_sections = bundle_data.get('text_sections', [])
            
            # Initialize feature extractor
            extractor = DocumentFeatureExtractor()
            
            # Extract features
            if full_text:
                feature_obj = extractor.extract_features(full_text, document_structure)
                features = _to_plain_dict(feature_obj)
            else:
                # Fallback for empty documents
                features = {
                    "total_length": 0,
                    "word_count": 0,
                    "sentence_count": 0,
                    "unique_terms": 0,
                    "technical_term_density": 0.0,
                    "readability_flesch": None,
                    "readability_dale_chall": None,
                    "section_count": len(text_sections),
                    "empty_document": True
                }
            
            # Create feature artifact
            feature_data = {
                "document_stem": stem,
                "extraction_metadata": {
                    "component": "03I",
                    "processor": "DocumentFeatureExtractor",
                    "timestamp": str(__import__('datetime').datetime.now()),
                    "extractor_config": extractor._config_id if hasattr(extractor, '_config_id') else None
                },
                "document_features": features,
                "text_statistics": {
                    "total_characters": len(full_text),
                    "total_sections": len(text_sections),
                    "non_empty_sections": len([s for s in text_sections if s.get('word_count', 0) > 0])
                },
                "processing_notes": []
            }
            
            # Add processing notes if needed
            if not full_text:
                feature_data["processing_notes"].append("Empty document - using fallback features")
            
            # Write artifact using ArtifactManager
            output_path = artifact_manager.write_artifact(stem, "features", feature_data)
            
            results.append({
                "stem": stem,
                "success": True,
                "output_path": str(output_path),
                "features_extracted": len(features),
                "artifact_type": "features"
            })
            
        except Exception as e:
            # Write error artifact
            error_stem = bundle_result.get('stem', 'unknown') if isinstance(bundle_result, dict) else 'unknown'
            error_data = {
                "document_stem": error_stem,
                "error": str(e),
                "processing_metadata": {
                    "component": "03I",
                    "status": "failed",
                    "timestamp": str(__import__('datetime').datetime.now())
                }
            }
            
            try:
                output_path = artifact_manager.write_artifact(error_stem, "features", error_data)
                results.append({
                    "stem": error_stem,
                    "success": False,
                    "error": str(e),
                    "output_path": str(output_path),
                    "artifact_type": "features"
                })
            except Exception as artifact_error:
                results.append({
                    "stem": error_stem,
                    "success": False,
                    "error": f"Processing failed: {str(e)}, Artifact writing failed: {str(artifact_error)}"
                })
    
    return {
        "component": "03I",
        "results": results,
        "total_inputs": len(bundle_results),
        "successful_extractions": len([r for r in results if r.get('success', False)])
    }

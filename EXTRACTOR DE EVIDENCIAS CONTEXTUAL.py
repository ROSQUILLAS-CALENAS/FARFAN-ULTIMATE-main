# Sistema de Evaluación PDT - Versión Optimizada y Mejorada
# Mejoras: Arquitectura modular, manejo de errores robusto, optimización de rendimiento

import pdfplumber
import fitz  # PyMuPDF as fitz
import pytesseract
import easyocr
import cv2
import numpy as np
import pandas as pd
import re
import json
# # # from typing import Dict, List, Tuple, Any, Optional, Union  # Module not found  # Module not found  # Module not found
# # # from dataclasses import dataclass, asdict  # Module not found  # Module not found  # Module not found
# # # from sentence_transformers import SentenceTransformer  # Module not found  # Module not found  # Module not found
import faiss
# # # from pathlib import Path  # Module not found  # Module not found  # Module not found
import logging
import time
# # # from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError  # Module not found  # Module not found  # Module not found

# Import audit logger for execution tracing
try:
# # #     from canonical_flow.analysis.audit_logger import get_audit_logger  # Module not found  # Module not found  # Module not found
except ImportError:
    # Fallback when audit logger is not available
    get_audit_logger = None
import camelot
# # # from spellchecker import SpellChecker  # Module not found  # Module not found  # Module not found
import psutil
# # # from datetime import datetime  # Module not found  # Module not found  # Module not found
# # # from abc import ABC, abstractmethod  # Module not found  # Module not found  # Module not found
# # # from contextlib import contextmanager  # Module not found  # Module not found  # Module not found
import hashlib
import asyncio
import threading
import os

# Configuración optimizada de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== ESTRUCTURAS DE DATOS ====================

@dataclass
class EvidenceCandidate:
    """Estructura de candidato de evidencia con ID determinístico"""
    chunk_id: str
    text: str
    context_before: str = ""
    context_after: str = ""
    page: int = 0
    section: str = "No identificada"
    chapter: str = "No identificado"
    bbox: Tuple[float, float, float, float] = (0, 0, 0, 0)
    confidence: float = 0.0
    extraction_method: str = "unknown"
    related_elements: List[str] = None
    query_id: str = ""
    timestamp: datetime = None
    text_position: int = 0
    
    def __post_init__(self):
        if self.related_elements is None:
            self.related_elements = []
        if self.timestamp is None:
            self.timestamp = datetime.now()
        
        # Validaciones automáticas
        self.confidence = max(0.0, min(1.0, self.confidence))
        if not isinstance(self.page, int) or self.page < 0:
            self.page = 1
            
        # Generar chunk_id determinístico si no se proporcionó
        if not self.chunk_id:
            self.chunk_id = self._generate_deterministic_chunk_id()
    
    def _generate_deterministic_chunk_id(self) -> str:
        """Genera un ID determinístico basado en contenido y posición"""
        # Crear hash basado en contenido textual y posición
        content_hash = hashlib.sha256(
            f"{self.text}_{self.page}_{self.text_position}".encode('utf-8')
        ).hexdigest()[:16]
        
        return f"chunk_{self.page:04d}_{self.text_position:06d}_{content_hash}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for canonical artifact schema alignment"""
        return {
            'chunk_id': self.chunk_id,
            'text': self.text,
            'context_before': self.context_before,
            'context_after': self.context_after,
            'page': self.page,
            'section': self.section,
            'chapter': self.chapter,
            'bbox': list(self.bbox),
            'confidence': self.confidence,
            'extraction_method': self.extraction_method,
            'related_elements': self.related_elements.copy(),
            'query_id': self.query_id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'text_position': self.text_position,
            'metadata': {
                'deterministic_id': True,
                'stable_ordering': True,
                'content_hash': hashlib.sha256(self.text.encode('utf-8')).hexdigest()[:16]
            }
        }

@dataclass
class Evidence:
    """Estructura optimizada de evidencia con validación"""
    text: str
    context_before: str = ""
    context_after: str = ""
    page: int = 0
    section: str = "No identificada"
    chapter: str = "No identificado"
    bbox: Tuple[float, float, float, float] = (0, 0, 0, 0)
    confidence: float = 0.0
    extraction_method: str = "unknown"
    related_elements: List[str] = None
    query_id: str = ""
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.related_elements is None:
            self.related_elements = []
        if self.timestamp is None:
            self.timestamp = datetime.now()
        
        # Validaciones automáticas
        self.confidence = max(0.0, min(1.0, self.confidence))
        if not isinstance(self.page, int) or self.page < 0:
            self.page = 1

@dataclass
class SearchResult:
    """Resultado de búsqueda optimizado"""
    chunk_id: int
    text: str
    metadata: Dict[str, Any]
    score: float
    query: str
    method: str = "semantic"

@dataclass
class ValidationResult:
    """Resultado de validación mejorado con métricas detalladas"""
    final_score: float
    has_all_fields: bool = True
    metadata_valid: bool = True
    relevance_score: float = 0.0
    coherence_score: float = 0.0
    completeness: float = 0.0
    text_quality: float = 0.0
    data_presence: bool = False
    recommendations: List[str] = None
    
    def __post_init__(self):
        if self.recommendations is None:
            self.recommendations = []

# ==================== CONFIGURACIÓN Y CONSTANTES ====================

class Config:
    """Configuración centralizada del sistema"""
    
    # Extracción
    CONTEXT_WINDOW = 500
    MIN_CONTEXT = 200
    MAX_CONTEXT = 1500
    
    # Búsqueda semántica
    SEMANTIC_MODEL = 'paraphrase-multilingual-MiniLM-L12-v2'
    CHUNK_SIZE = 512
    OVERLAP = 128
    TOP_K_RESULTS = 10
    
    # Validación
    MIN_CONFIDENCE = 0.6
    MIN_TEXT_LENGTH = 50
    MAX_TEXT_LENGTH = 5000
    
    # Procesamiento
    MAX_WORKERS = 4
    CACHE_SIZE_LIMIT = 1000
    TIMEOUT_SECONDS = 300
    
    # OCR
    OCR_LANGUAGES = ['spa', 'eng']
    OCR_CONFIDENCE_THRESHOLD = 0.8
    
    # Patrones de búsqueda
    TOC_PATTERNS = [
        r'tabla\s+de\s+contenido',
        r'índice\s+general', 
        r'contenido',
        r'sumario'
    ]
    
    SECTION_PATTERNS = [
        r'cap[íi]tulo\s+\d+',
        r'secci[óo]n\s+\d+',
        r'parte\s+\d+',
        r'anexo\s+[a-z]'
    ]

# ==================== EXCEPCIONES PERSONALIZADAS ====================

class PDTExtractionError(Exception):
    """Excepción base para errores de extracción"""
    pass

class PDFProcessingError(PDTExtractionError):
    """Error en procesamiento de PDF"""
    pass

class SearchError(PDTExtractionError):
    """Error en búsqueda semántica"""
    pass

class ValidationError(PDTExtractionError):
    """Error en validación de evidencias"""
    pass

# ==================== UTILIDADES ====================

class Utils:
    """Utilidades comunes del sistema"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Limpia y normaliza texto"""
        if not text:
            return ""
        
        # Remover caracteres problemáticos
        text = re.sub(r'[^\w\s\.\,\;\:\-\(\)\[\]\"\'\/\%\$]', ' ', text)
        # Normalizar espacios
        text = ' '.join(text.split())
        return text.strip()
    
    @staticmethod
    def calculate_text_hash(text: str) -> str:
        """Calcula hash para deduplicación"""
        return hashlib.md5(text.encode()).hexdigest()[:16]
    
    @staticmethod
    def fuzzy_match(text1: str, text2: str, threshold: float = 0.8) -> float:
        """Matching fuzzy simple entre textos"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    @contextmanager
    def timer(description: str):
        """Context manager para medir tiempo"""
        start = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start
            logger.debug(f"{description}: {elapsed:.2f}s")

# ==================== INTERFACES ABSTRACTAS ====================

class Extractor(ABC):
    """Interfaz base para extractores"""
    
    @abstractmethod
    def extract(self, *args, **kwargs) -> Any:
        pass

class Searcher(ABC):
    """Interfaz base para buscadores"""
    
    @abstractmethod
    def search(self, query: str, *args, **kwargs) -> List[SearchResult]:
        pass

# ==================== COMPONENTES CORE MEJORADOS ====================

class PDFStructureExtractor(Extractor):
    """Extractor de estructura PDF optimizado"""
    
    def __init__(self):
        self.cache = {}
    
    def extract(self, pdf_path: str, force_refresh: bool = False) -> Dict[str, Any]:
        """Extrae estructura con cache inteligente"""
        
        cache_key = f"{pdf_path}_{os.path.getmtime(pdf_path)}"
        
        if not force_refresh and cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            with Utils.timer(f"Extracting structure for {Path(pdf_path).name}"):
                structure = self._extract_structure_internal(pdf_path)
                
            # Cache con límite de tamaño  
            if len(self.cache) >= Config.CACHE_SIZE_LIMIT:
                self.cache.clear()
            
            self.cache[cache_key] = structure
            return structure
            
        except Exception as e:
            logger.error(f"Error extracting PDF structure: {e}")
            raise PDFProcessingError(f"Failed to extract PDF structure: {e}")
    
    def _extract_structure_internal(self, pdf_path: str) -> Dict[str, Any]:
        """Lógica interna de extracción optimizada"""
        
        structure = {
            'toc_found': False,
            'sections': {},
            'hierarchy': {},
            'page_mapping': {},
            'total_pages': 0,
            'extraction_date': datetime.now().isoformat()
        }
        
        with pdfplumber.open(pdf_path) as pdf:
            structure['total_pages'] = len(pdf.pages)
            
            # Buscar TOC de manera más eficiente
            toc_found, toc_data = self._find_toc_optimized(pdf)
            
            if toc_found:
                structure.update(toc_data)
            else:
                # Inferir estructura con límite de páginas
                structure['sections'] = self._infer_structure_fast(pdf)
        
        return structure
    
    def _find_toc_optimized(self, pdf) -> Tuple[bool, Dict[str, Any]]:
        """Búsqueda optimizada del TOC"""
        
        max_toc_pages = min(15, len(pdf.pages))
        
        for i in range(max_toc_pages):
            page = pdf.pages[i]
            text = page.extract_text() or ""
            
            # Búsqueda optimizada de patrones TOC
            for pattern in Config.TOC_PATTERNS:
                if re.search(pattern, text.lower()):
                    sections = self._parse_toc_entries_enhanced(page, text)
                    
                    if sections:  # Solo retornar si encontró secciones válidas
                        return True, {
                            'toc_found': True,
                            'toc_page': i + 1,
                            'sections': sections
                        }
        
        return False, {}
    
    def _parse_toc_entries_enhanced(self, page, text: str) -> Dict[str, Dict[str, int]]:
        """Parser mejorado de entradas TOC"""
        
        sections = {}
        
        # Patrones mejorados para diferentes formatos
        patterns = [
            r'(?:cap[íi]tulo\s+)?(\d+)[\.\:\-\s]+(.*?)\s+(\d+)',
            r'(\d+\.\d+(?:\.\d+)?)\s+(.*?)\s+(\d+)',
            r'(anexo\s+[a-z\d]+)[\.\:\-\s]+(.*?)\s+(\d+)',
            r'([ivxlc]+)[\.\:\-\s]+(.*?)\s+(\d+)'  # Números romanos
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text.lower(), re.MULTILINE)
            
            for match in matches:
                try:
                    section_id = match.group(1)
                    section_title = match.group(2).strip()
                    page_num = int(match.group(3))
                    
                    if page_num > 0 and len(section_title) > 3:
                        section_name = f"{section_id}: {section_title}"
                        sections[section_name] = {
                            'start_page': page_num,
                            'end_page': page_num + 30,  # Estimación mejorada
                            'level': len(section_id.split('.')) if '.' in section_id else 1
                        }
                except (ValueError, IndexError):
                    continue
        
        return sections
    
    def _infer_structure_fast(self, pdf) -> Dict[str, Dict[str, int]]:
        """Inferencia rápida de estructura sin TOC"""
        
        sections = {}
        pages_to_check = min(50, len(pdf.pages))
        
        for i in range(pages_to_check):
            page = pdf.pages[i]
            text = page.extract_text() or ""
            
            # Buscar headers prominentes
            lines = text.split('\n')[:10]  # Solo primeras 10 líneas
            
            for line in lines:
                line = line.strip()
                if not line or len(line) < 5:
                    continue
                
                for pattern in Config.SECTION_PATTERNS:
                    if re.search(pattern, line.lower()):
                        sections[line] = {
                            'start_page': i + 1,
                            'end_page': i + 25,
                            'level': 1,
                            'inferred': True
                        }
                        break
        
        return sections

class OptimizedSemanticSearch(Searcher):
    """Motor de búsqueda semántica optimizado"""
    
    def __init__(self):
        self.model = None
        self.index = None
        self.chunks = []
        self.chunk_metadata = []
        self._model_loaded = False
        
    def _ensure_model_loaded(self):
        """Carga lazy del modelo para optimizar memoria"""
        if not self._model_loaded:
            try:
                self.model = SentenceTransformer(Config.SEMANTIC_MODEL)
                self._model_loaded = True
                logger.info(f"Semantic model loaded: {Config.SEMANTIC_MODEL}")
            except Exception as e:
                logger.error(f"Failed to load semantic model: {e}")
                raise SearchError(f"Cannot initialize semantic search: {e}")
    
    def index_document(self, chunks: List[str], metadata: List[Dict]) -> None:
        """Indexación optimizada de documento"""
        
        if not chunks:
            raise SearchError("No chunks provided for indexing")
        
        self._ensure_model_loaded()
        
        with Utils.timer("Document indexing"):
            # Filtrar chunks muy cortos
            valid_chunks = []
            valid_metadata = []
            
            for chunk, meta in zip(chunks, metadata):
                if len(chunk.strip()) >= Config.MIN_TEXT_LENGTH:
                    valid_chunks.append(Utils.clean_text(chunk))
                    valid_metadata.append(meta)
            
            if not valid_chunks:
                raise SearchError("No valid chunks after filtering")
            
            self.chunks = valid_chunks
            self.chunk_metadata = valid_metadata
            
            # Generar embeddings en lotes para eficiencia
            embeddings = self.model.encode(
                valid_chunks, 
                batch_size=32,
                show_progress_bar=False
            )
            
            # Crear índice FAISS optimizado
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)
            
            # Normalizar para similitud coseno
            faiss.normalize_L2(embeddings)
            self.index.add(embeddings.astype('float32'))
            
            logger.info(f"Indexed {len(valid_chunks)} chunks successfully")
    
    def search(self, query: str, top_k: int = Config.TOP_K_RESULTS, 
               min_score: float = 0.3) -> List[SearchResult]:
        """Búsqueda semántica optimizada"""
        
        if not self.index or not self.chunks:
            return []
        
        self._ensure_model_loaded()
        
        try:
            with Utils.timer(f"Semantic search: {query[:50]}..."):
                # Expandir query con variaciones
                expanded_queries = self._expand_query_smart(query)
                
                all_results = []
                
                for exp_query in expanded_queries:
                    # Encode query
                    query_embedding = self.model.encode([exp_query])
                    faiss.normalize_L2(query_embedding)
                    
                    # Buscar
                    scores, indices = self.index.search(
                        query_embedding.astype('float32'), 
                        min(top_k * 2, len(self.chunks))
                    )
                    
                    # Procesar resultados
                    for score, idx in zip(scores[0], indices[0]):
                        if idx < len(self.chunks) and score >= min_score:
                            result = SearchResult(
                                chunk_id=int(idx),
                                text=self.chunks[idx],
                                metadata=self.chunk_metadata[idx].copy(),
                                score=float(score),
                                query=query,
                                method="semantic"
                            )
                            all_results.append(result)
                
                # Deduplicar y reordenar
                return self._postprocess_results(all_results, query)[:top_k]
                
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return []
    
    def _expand_query_smart(self, query: str) -> List[str]:
        """Expansión inteligente de consultas"""
        
        expansions = [query]
        
        # Diccionario de sinónimos contextual para PDTs
        synonyms = {
            'medibles': ['cuantificables', 'mensurables', 'evaluables', 'específicos'],
            'metas': ['objetivos', 'propósitos', 'indicadores', 'targets'],
            'responsable': ['encargado', 'coordinador', 'líder', 'ejecutor'],
            'presupuesto': ['recursos', 'inversión', 'financiación', 'costos'],
            'causas': ['factores', 'determinantes', 'razones', 'orígenes'],
            'línea base': ['baseline', 'situación inicial', 'punto partida'],
            'productos': ['entregables', 'resultados', 'outputs'],
            'diagnóstico': ['análisis', 'caracterización', 'evaluación situacional']
        }
        
        query_lower = query.lower()
        
        for term, syns in synonyms.items():
            if term in query_lower:
                for syn in syns[:2]:  # Máximo 2 sinónimos por término
                    expanded = query_lower.replace(term, syn)
                    if expanded != query_lower:
                        expansions.append(expanded)
        
        return expansions[:3]  # Máximo 3 variaciones
    
    def _postprocess_results(self, results: List[SearchResult], 
                           original_query: str) -> List[SearchResult]:
        """Post-procesamiento de resultados"""
        
        if not results:
            return []
        
        # Deduplicar por hash de texto
        seen_hashes = set()
        unique_results = []
        
        for result in results:
            text_hash = Utils.calculate_text_hash(result.text)
            if text_hash not in seen_hashes:
                seen_hashes.add(text_hash)
                unique_results.append(result)
        
        # Reordenar por score combinado
        for result in unique_results:
            # Boost por coincidencia exacta
            exact_match_boost = 1.0
            query_words = set(original_query.lower().split())
            text_words = set(result.text.lower().split())
            
            if query_words.intersection(text_words):
                exact_match_boost = 1.2
            
            # Boost por longitud óptima
            length_boost = 1.0
            text_length = len(result.text)
            if Config.MIN_CONTEXT <= text_length <= Config.MAX_CONTEXT:
                length_boost = 1.1
            
            result.score *= exact_match_boost * length_boost
        
        # Ordenar por score final
        unique_results.sort(key=lambda x: x.score, reverse=True)
        
        return unique_results

class AdvancedEvidenceExtractor:
    """Extractor de evidencias mejorado con múltiples estrategias"""
    
    def __init__(self, timeout_seconds: int = Config.TIMEOUT_SECONDS):
        self.structure_extractor = PDFStructureExtractor()
        self.semantic_search = OptimizedSemanticSearch()
        self.cache = {}
        self.timeout_seconds = timeout_seconds
        self.stats = {
            'extractions': 0,
            'successes': 0,
            'cache_hits': 0,
            'timeouts': 0,
            'partial_results': 0
        }
    
    def process(self, document_text: str, question_context: str, 
                timeout: Optional[int] = None) -> List[EvidenceCandidate]:
        """
        Standardized process method that accepts document text and question context,
        returns evidence candidates with deterministic chunk IDs and stable ordering.
        
        Args:
            document_text: Raw document text content
            question_context: Question/query context for evidence extraction
            timeout: Optional timeout in seconds (overrides instance default)
            
        Returns:
            List of EvidenceCandidate objects with deterministic chunk IDs,
            ordered by text position for consistent results
        """
        
        self.stats['extractions'] += 1
        timeout_to_use = timeout or self.timeout_seconds
        
        # Cache check
        cache_key = Utils.calculate_text_hash(f"{document_text[:1000]}_{question_context}")
        if cache_key in self.cache:
            self.stats['cache_hits'] += 1
            return self.cache[cache_key]
        
        try:
            # Use timeout protection with threading
            result = self._process_with_timeout(
                document_text, question_context, timeout_to_use
            )
            
            # Cache successful results
            if result:
                self.cache[cache_key] = result
                self.stats['successes'] += 1
            
            return result or []
            
        except Exception as e:
            logger.error(f"Error in process method: {e}")
            return []
    
    def _process_with_timeout(self, document_text: str, question_context: str, 
                             timeout_seconds: int) -> List[EvidenceCandidate]:
        """Execute processing with timeout protection"""
        
        result_container = []
        exception_container = []
        
        def target():
            try:
                result = self._process_internal(document_text, question_context)
                result_container.append(result)
            except Exception as e:
                exception_container.append(e)
        
        thread = threading.Thread(target=target)
        thread.daemon = True
        thread.start()
        thread.join(timeout=timeout_seconds)
        
        if thread.is_alive():
            # Timeout occurred
            self.stats['timeouts'] += 1
            logger.warning(f"Processing timed out after {timeout_seconds}s")
            
            # Try to return any partial results that might be available
            partial_results = self._get_partial_results(document_text, question_context)
            if partial_results:
                self.stats['partial_results'] += 1
                return partial_results
            
            return []
        
        if exception_container:
            raise exception_container[0]
        
        return result_container[0] if result_container else []
    
    def _process_internal(self, document_text: str, question_context: str) -> List[EvidenceCandidate]:
        """Internal processing logic without timeout"""
        
        try:
            with Utils.timer(f"Processing document with context: {question_context[:50]}..."):
                
                # 1. Analyze query context
                query_analysis = self._analyze_query_enhanced(question_context, "general")
                
                # 2. Create chunks with deterministic IDs and stable ordering
                chunks, metadata = self._create_deterministic_chunks(document_text)
                
                if not chunks:
# # #                     logger.warning("No chunks created from document text")  # Module not found  # Module not found  # Module not found
                    return []
                
                # 3. Index and search
                self.semantic_search.index_document(chunks, metadata)
                search_results = self.semantic_search.search(
                    question_context,
                    top_k=10,
                    min_score=0.3
                )
                
                # 4. Convert to evidence candidates with deterministic ordering
                candidates = self._create_evidence_candidates(
                    search_results, query_analysis, document_text
                )
                
                return candidates
                
        except Exception as e:
            logger.error(f"Error in internal processing: {e}")
            return []
    
    def _create_deterministic_chunks(self, document_text: str) -> Tuple[List[str], List[Dict]]:
        """Create chunks with deterministic IDs and stable ordering"""
        
        chunks = []
        metadata = []
        
        if not document_text or len(document_text.strip()) < Config.MIN_TEXT_LENGTH:
            return chunks, metadata
        
        # Split into sentences for better boundaries
        sentences = [s.strip() for s in re.split(r'[.!?]+', document_text) if s.strip()]
        
        current_chunk = ""
        current_position = 0
        chunk_index = 0
        
        for sentence in sentences:
            sentence_clean = Utils.clean_text(sentence)
            
            # Check if adding this sentence would exceed chunk size
            potential_chunk = f"{current_chunk} {sentence_clean}".strip()
            
            if len(potential_chunk.split()) > Config.CHUNK_SIZE and current_chunk:
                # Finalize current chunk
                if len(current_chunk.split()) >= 10:  # Minimum meaningful chunk size
                    chunks.append(current_chunk.strip())
                    metadata.append({
                        'chunk_index': chunk_index,
                        'text_position': current_position,
                        'word_count': len(current_chunk.split()),
                        'extraction_method': 'deterministic_chunking'
                    })
                    chunk_index += 1
                
                # Start new chunk
                current_chunk = sentence_clean
                current_position += len(current_chunk)
            else:
                current_chunk = potential_chunk
        
        # Add final chunk
        if current_chunk.strip() and len(current_chunk.split()) >= 10:
            chunks.append(current_chunk.strip())
            metadata.append({
                'chunk_index': chunk_index,
                'text_position': current_position,
                'word_count': len(current_chunk.split()),
                'extraction_method': 'deterministic_chunking'
            })
        
        return chunks, metadata
    
    def _create_evidence_candidates(self, search_results: List[SearchResult], 
                                   query_analysis: Dict[str, Any],
                                   document_text: str) -> List[EvidenceCandidate]:
        """Create evidence candidates with deterministic chunk IDs"""
        
        candidates = []
        
        for result in search_results:
            text_position = result.metadata.get('text_position', 0)
            
            candidate = EvidenceCandidate(
                chunk_id="",  # Will be auto-generated in __post_init__
                text=result.text,
                context_before=self._extract_context_before(document_text, result.text),
                context_after=self._extract_context_after(document_text, result.text),
                page=1,  # Default for text-only processing
                section=self._infer_section(result.text),
                confidence=result.score,
                extraction_method=result.method,
                query_id=Utils.calculate_text_hash(query_analysis.get('type', 'general')),
                text_position=text_position,
                related_elements=self._detect_related_elements(result.text)
            )
            
            candidates.append(candidate)
        
        # Sort by text position for deterministic ordering
        candidates.sort(key=lambda x: (x.text_position, x.chunk_id))
        
        return candidates
    
    def _get_partial_results(self, document_text: str, question_context: str) -> List[EvidenceCandidate]:
        """Return partial results when full processing times out"""
        
        try:
            # Create a simple chunk-based result as fallback
            chunks, metadata = self._create_deterministic_chunks(document_text)
            
            if not chunks:
                return []
            
            # Return first few chunks as candidates
            partial_candidates = []
            
            for i, (chunk, meta) in enumerate(zip(chunks[:3], metadata[:3])):
                candidate = EvidenceCandidate(
                    chunk_id="",  # Will be auto-generated
                    text=chunk,
                    text_position=meta.get('text_position', i * 1000),
                    confidence=0.5,  # Lower confidence for partial results
                    extraction_method="timeout_fallback",
                    query_id="partial"
                )
                partial_candidates.append(candidate)
            
            return partial_candidates
            
        except Exception:
            return []
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Return processing statistics"""
        return {
            'total_extractions': self.stats['extractions'],
            'successful_extractions': self.stats['successes'],
            'cache_hits': self.stats['cache_hits'],
            'timeout_occurrences': self.stats['timeouts'],
            'partial_results_returned': self.stats['partial_results'],
            'success_rate': (self.stats['successes'] / max(self.stats['extractions'], 1)) * 100,
            'cache_hit_rate': (self.stats['cache_hits'] / max(self.stats['extractions'], 1)) * 100
        }
    
    def _extract_context_before(self, document_text: str, chunk_text: str) -> str:
        """Extract context before the chunk"""
        try:
            chunk_start = document_text.find(chunk_text)
            if chunk_start > 0:
                context_start = max(0, chunk_start - Config.CONTEXT_WINDOW)
                return document_text[context_start:chunk_start].strip()
        except Exception:
            pass
        return ""
    
    def _extract_context_after(self, document_text: str, chunk_text: str) -> str:
        """Extract context after the chunk"""
        try:
            chunk_start = document_text.find(chunk_text)
            if chunk_start >= 0:
                chunk_end = chunk_start + len(chunk_text)
                context_end = min(len(document_text), chunk_end + Config.CONTEXT_WINDOW)
                return document_text[chunk_end:context_end].strip()
        except Exception:
            pass
        return ""
    
    def _infer_section(self, text: str) -> str:
# # #         """Infer section from text content"""  # Module not found  # Module not found  # Module not found
        text_lower = text.lower()
        
        section_indicators = {
            'introducción': ['introducción', 'antecedentes', 'contexto'],
            'metodología': ['metodología', 'método', 'procedimiento'],
            'resultados': ['resultados', 'hallazgos', 'evidencias'],
            'conclusiones': ['conclusión', 'resumen', 'síntesis'],
            'anexos': ['anexo', 'apéndice', 'tabla', 'figura']
        }
        
        for section, indicators in section_indicators.items():
            if any(indicator in text_lower for indicator in indicators):
                return section.title()
        
        return "Contenido General"
    
    def extract_evidence(self, pdf_path: str, query: str, 
                        question_type: str) -> Optional[Evidence]:
        """Legacy method for backward compatibility"""
        # Audit logging for component execution
        audit_logger = get_audit_logger() if get_audit_logger else None
        input_data = {
            "pdf_path": pdf_path,
            "query": query[:200] + "..." if len(query) > 200 else query,
            "question_type": question_type
        }
        
        if audit_logger:
            with audit_logger.audit_component_execution("17A", input_data) as audit_ctx:
                result = self._extract_evidence_internal(pdf_path, query, question_type)
                audit_ctx.set_output({
                    "evidence_found": result is not None,
                    "score": result.score if result else 0.0,
                    "chunks_count": len(result.chunks) if result and result.chunks else 0
                })
                return result
        else:
            return self._extract_evidence_internal(pdf_path, query, question_type)

    def _extract_evidence_internal(self, pdf_path: str, query: str, 
                                  question_type: str) -> Optional[Evidence]:
        """Internal implementation of evidence extraction."""
        self.stats['extractions'] += 1
        
        # Cache check
        cache_key = Utils.calculate_text_hash(f"{pdf_path}_{query}_{question_type}")
        if cache_key in self.cache:
            self.stats['cache_hits'] += 1
            return self.cache[cache_key]
        
        try:
            with Utils.timer(f"Evidence extraction for {question_type}"):
                # 1. Analizar PDF y query
                pdf_structure = self.structure_extractor.extract(pdf_path)
                query_analysis = self._analyze_query_enhanced(query, question_type)
                
                # 2. Preparar chunks para búsqueda
                chunks, metadata = self._prepare_chunks_optimized(pdf_path, pdf_structure)
                
                if not chunks:
# # #                     logger.warning("No chunks extracted from PDF")  # Module not found  # Module not found  # Module not found
                    return None
                
                # 3. Indexar y buscar
                self.semantic_search.index_document(chunks, metadata)
                candidates = self.semantic_search.search(
                    query, 
                    top_k=5,
                    min_score=0.4
                )
                
                # 4. Seleccionar mejor candidato
                if not candidates:
                    logger.warning(f"No candidates found for: {query}")
                    return None
                
                best_candidate = self._select_best_candidate(candidates, query_analysis)
                
                # 5. Construir evidencia enriquecida
                evidence = self._build_evidence(
                    best_candidate, query_analysis, pdf_structure, question_type
                )
                
                # Cache resultado
                self.cache[cache_key] = evidence
                self.stats['successes'] += 1
                
                return evidence
                
        except Exception as e:
            logger.error(f"Error extracting evidence: {e}")
            return None


# ==================== MAIN INTERFACE ====================

def process(document_text: str, question_context: str, 
           timeout: Optional[int] = None) -> Dict[str, Any]:
    """
    Main entry point for the extractor module.
    
    Implements standardized process() method that accepts document text and question context,
    returns evidence candidates with deterministic chunk IDs and consistent ordering.
    
    Args:
        document_text: Raw document text content
        question_context: Question/query context for evidence extraction  
        timeout: Optional timeout in seconds (default: 300)
        
    Returns:
        Dictionary containing:
        - 'evidence_candidates': List of evidence candidate dictionaries
        - 'processing_stats': Dictionary with processing metrics
        - 'success': Boolean indicating if processing completed successfully
        - 'error': Error message if processing failed (None on success)
        - 'partial_results': Boolean indicating if results are partial due to timeout
    """
    
    if not document_text or not document_text.strip():
        return {
            'evidence_candidates': [],
            'processing_stats': {},
            'success': False,
            'error': 'Empty or invalid document text provided',
            'partial_results': False
        }
    
    if not question_context or not question_context.strip():
        return {
            'evidence_candidates': [],
            'processing_stats': {},
            'success': False,
            'error': 'Empty or invalid question context provided',
            'partial_results': False
        }
    
    try:
        # Initialize extractor with timeout
        extractor = AdvancedEvidenceExtractor(
            timeout_seconds=timeout or Config.TIMEOUT_SECONDS
        )
        
        # Process with timeout protection
        candidates = extractor.process(document_text, question_context, timeout)
        
        # Convert candidates to dictionary format for canonical schema alignment
        candidates_dict = []
        for candidate in candidates:
            try:
                candidate_dict = candidate.to_dict()
                candidates_dict.append(candidate_dict)
            except Exception as e:
                logger.warning(f"Failed to convert candidate to dict: {e}")
                continue
        
        # Get processing statistics
        stats = extractor.get_processing_stats()
        
        # Determine if results are partial
        is_partial = stats.get('partial_results_returned', 0) > 0
        
        return {
            'evidence_candidates': candidates_dict,
            'processing_stats': stats,
            'success': True,
            'error': None,
            'partial_results': is_partial,
            'metadata': {
                'total_candidates': len(candidates_dict),
                'processing_time': stats.get('last_processing_time', 0),
                'deterministic_ordering': True,
                'schema_version': '1.0',
                'extractor_version': 'advanced_v2.0'
            }
        }
        
    except TimeoutError as e:
        return {
            'evidence_candidates': [],
            'processing_stats': {},
            'success': False,
            'error': f'Processing timed out: {e}',
            'partial_results': True
        }
        
    except Exception as e:
        logger.error(f"Unexpected error in process function: {e}")
        return {
            'evidence_candidates': [],
            'processing_stats': {},
            'success': False,
            'error': f'Processing failed: {str(e)}',
            'partial_results': False
        }
    
    def _analyze_query_enhanced(self, query: str, question_type: str) -> Dict[str, Any]:
        """Análisis avanzado de queries"""
        
        analysis = {
            'type': question_type,
            'keywords': [],
            'expected_sections': [],
            'data_type': 'text',
            'priority_terms': [],
            'context_clues': []
        }
        
        query_lower = query.lower()
        
        # Términos de alta prioridad por categoría
        priority_mappings = {
            'productos': ['productos medibles', 'entregables', 'outputs'],
            'metas': ['metas', 'objetivos', 'indicadores'],
            'responsabilidad': ['responsable', 'institucional', 'coordinador'],
            'medición': ['medibles', 'cuantificables', 'línea base'],
            'presupuesto': ['presupuesto', 'recursos', 'inversión'],
            'análisis': ['causas', 'diagnóstico', 'determinantes']
        }
        
        for category, terms in priority_mappings.items():
            if any(term in query_lower for term in terms):
                analysis['priority_terms'].extend(terms)
                
                # Mapear a secciones esperadas
                if category in ['presupuesto']:
                    analysis['expected_sections'].extend(['anexo', 'plan plurianual'])
                    analysis['data_type'] = 'table'
                elif category in ['análisis']:
                    analysis['expected_sections'].extend(['diagnóstico', 'capítulo 1'])
                elif category in ['productos', 'metas']:
                    analysis['expected_sections'].extend(['programas', 'estratégica'])
        
        # Extraer keywords importantes
        important_patterns = [
            r'productos?\s+medibles?',
            r'metas?\s+\d+',
            r'línea\s+base',
            r'responsables?\s+institucionales?',
            r'inversiones?'
        ]
        
        for pattern in important_patterns:
            matches = re.findall(pattern, query_lower)
            analysis['keywords'].extend(matches)
        
        return analysis
    
    def _prepare_chunks_optimized(self, pdf_path: str, 
                                structure: Dict[str, Any]) -> Tuple[List[str], List[Dict]]:
        """Preparación optimizada de chunks"""
        
        chunks = []
        metadata = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
                
                # Procesamiento inteligente por secciones
                sections = structure.get('sections', {})
                processed_pages = set()
                
                # Procesar secciones conocidas primero
                for section_name, section_info in sections.items():
                    start_page = section_info.get('start_page', 1)
                    end_page = min(
                        section_info.get('end_page', start_page + 10),
                        total_pages
                    )
                    
                    for page_num in range(start_page, end_page + 1):
                        if page_num <= total_pages and page_num not in processed_pages:
                            processed_pages.add(page_num)
                            
                            page_chunks, page_metadata = self._extract_page_chunks(
                                pdf.pages[page_num - 1], page_num, section_name
                            )
                            chunks.extend(page_chunks)
                            metadata.extend(page_metadata)
                
                # Procesar páginas restantes
                for page_num in range(1, min(total_pages + 1, 100)):  # Límite de 100 páginas
                    if page_num not in processed_pages:
                        page_chunks, page_metadata = self._extract_page_chunks(
                            pdf.pages[page_num - 1], page_num, f"Página {page_num}"
                        )
                        chunks.extend(page_chunks)
                        metadata.extend(page_metadata)
        
        except Exception as e:
            logger.error(f"Error preparing chunks: {e}")
        
        return chunks, metadata
    
    def _extract_page_chunks(self, page, page_num: int, 
                           section: str) -> Tuple[List[str], List[Dict]]:
        """Extracción de chunks por página"""
        
        chunks = []
        metadata = []
        
        try:
            text = page.extract_text() or ""
            
            if len(text.strip()) < Config.MIN_TEXT_LENGTH:
                return chunks, metadata
            
            # Dividir en chunks con overlapping
            words = text.split()
            
            for i in range(0, len(words), Config.CHUNK_SIZE - Config.OVERLAP):
                chunk_words = words[i:i + Config.CHUNK_SIZE]
                chunk_text = ' '.join(chunk_words)
                
                if len(chunk_text) >= Config.MIN_TEXT_LENGTH:
                    chunks.append(chunk_text)
                    metadata.append({
                        'page': page_num,
                        'section': section,
                        'chunk_start': i,
                        'chunk_end': i + len(chunk_words),
                        'word_count': len(chunk_words),
                        'extraction_method': 'text'
                    })
            
            # Intentar extraer tablas si es una página con datos tabulares
            if self._page_likely_has_tables(text):
                table_chunks, table_metadata = self._extract_table_chunks(
                    page, page_num, section
                )
                chunks.extend(table_chunks)
                metadata.extend(table_metadata)
        
        except Exception as e:
# # #             logger.debug(f"Error extracting chunks from page {page_num}: {e}")  # Module not found  # Module not found  # Module not found
        
        return chunks, metadata
    
    def _page_likely_has_tables(self, text: str) -> bool:
        """Detecta si una página probablemente contiene tablas"""
        table_indicators = [
            'tabla', 'cuadro', 'anexo', 'inversión', 'presupuesto',
            'millones', 'recursos', 'año', 'meta'
        ]
        
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in table_indicators)
    
    def _extract_table_chunks(self, page, page_num: int, 
                            section: str) -> Tuple[List[str], List[Dict]]:
        """Extracción especializada de tablas"""
        
        chunks = []
        metadata = []
        
        try:
            # Usar camelot para extraer tablas
            tables = camelot.read_pdf(
                page.pdf.stream.name,
                pages=str(page_num),
                flavor='lattice',
                line_scale=30
            )
            
            for idx, table in enumerate(tables):
                if not table.df.empty:
                    # Convertir tabla a texto estructurado
                    table_text = self._table_to_text(table.df)
                    
                    if len(table_text) >= Config.MIN_TEXT_LENGTH:
                        chunks.append(table_text)
                        metadata.append({
                            'page': page_num,
                            'section': section,
                            'extraction_method': 'table',
                            'table_index': idx,
                            'table_shape': table.df.shape,
                            'confidence': table.accuracy if hasattr(table, 'accuracy') else 0.8
                        })
        
        except Exception as e:
# # #             logger.debug(f"No tables extracted from page {page_num}: {e}")  # Module not found  # Module not found  # Module not found
        
        return chunks, metadata
    
    def _table_to_text(self, df: pd.DataFrame) -> str:
        """Convierte tabla a texto estructurado"""
        
        if df.empty:
            return ""
        
        # Limpiar DataFrame
        df_clean = df.copy()
        df_clean = df_clean.fillna('')
        
        # Crear texto estructurado
        lines = []
        
        # Headers
        if not df_clean.columns.empty:
            header_line = ' | '.join(str(col) for col in df_clean.columns)
            lines.append(f"TABLA - ENCABEZADOS: {header_line}")
        
        # Filas de datos
        for idx, row in df_clean.iterrows():
            if idx > 20:  # Limitar número de filas
                break
            
            row_values = [str(val).strip() for val in row.values if str(val).strip()]
            if row_values:
                row_line = ' | '.join(row_values)
                lines.append(f"FILA {idx + 1}: {row_line}")
        
        return '\n'.join(lines)
    
    def _select_best_candidate(self, candidates: List[SearchResult], 
                             query_analysis: Dict[str, Any]) -> SearchResult:
        """Selección inteligente del mejor candidato"""
        
        if not candidates:
            return None
        
        # Scoring avanzado
        for candidate in candidates:
            base_score = candidate.score
            
            # Boost por método de extracción
            method_boost = {
                'table': 1.3,  # Tablas suelen tener datos importantes
                'semantic': 1.0,
                'text': 0.9
            }
            
            extraction_method = candidate.metadata.get('extraction_method', 'text')
            boost = method_boost.get(extraction_method, 1.0)
            
            # Boost por términos de alta prioridad
            priority_boost = 1.0
            text_lower = candidate.text.lower()
            
            for priority_term in query_analysis.get('priority_terms', []):
                if priority_term in text_lower:
                    priority_boost += 0.1
            
            # Boost por sección relevante
            section_boost = 1.0
            section = candidate.metadata.get('section', '').lower()
            expected_sections = query_analysis.get('expected_sections', [])
            
            for expected in expected_sections:
                if expected in section:
                    section_boost = 1.2
                    break
            
            # Penalización por texto muy largo o muy corto
            length_penalty = 1.0
            text_length = len(candidate.text)
            
            if text_length < Config.MIN_CONTEXT:
                length_penalty = 0.8
            elif text_length > Config.MAX_CONTEXT:
                length_penalty = 0.9
            
            # Score final
            candidate.score = base_score * boost * priority_boost * section_boost * length_penalty
        
        # Retornar el mejor
        candidates.sort(key=lambda x: x.score, reverse=True)
        return candidates[0]
    
    def _build_evidence(self, candidate: SearchResult, query_analysis: Dict[str, Any],
                       pdf_structure: Dict[str, Any], question_type: str) -> Evidence:
        """Construcción de evidencia enriquecida"""
        
        # Extraer contexto adicional
        context_before, context_after = self._extract_context(
            candidate.text, candidate.metadata
        )
        
        # Identificar sección y capítulo mejorado
        section_info = self._identify_section_enhanced(
            candidate.metadata, pdf_structure
        )
        
        # Detectar elementos relacionados
        related_elements = self._detect_related_elements(candidate.text)
        
        # Calcular confidence mejorado
        confidence = self._calculate_enhanced_confidence(
            candidate, query_analysis
        )
        
        evidence = Evidence(
            text=Utils.clean_text(candidate.text),
            context_before=context_before,
            context_after=context_after,
            page=candidate.metadata.get('page', 1),
            section=section_info['section'],
            chapter=section_info['chapter'],
            bbox=(0, 0, 100, 20),  # Placeholder - sería calculado con coordenadas reales
            confidence=confidence,
            extraction_method=candidate.metadata.get('extraction_method', 'semantic'),
            related_elements=related_elements,
            query_id=question_type,
            timestamp=datetime.now()
        )
        
        return evidence
    
    def _extract_context(self, text: str, metadata: Dict[str, Any]) -> Tuple[str, str]:
        """Extracción de contexto mejorada"""
        
        # Por ahora contexto simple - en implementación completa
        # se accedería al PDF para extraer contexto real
        words = text.split()
        mid_point = len(words) // 2
        
        context_before = ' '.join(words[:mid_point])[-Config.CONTEXT_WINDOW:]
        context_after = ' '.join(words[mid_point:])[:Config.CONTEXT_WINDOW]
        
        return context_before, context_after
    
    def _identify_section_enhanced(self, metadata: Dict[str, Any],
                                 pdf_structure: Dict[str, Any]) -> Dict[str, str]:
        """Identificación mejorada de sección y capítulo"""
        
        page_num = metadata.get('page', 1)
        current_section = metadata.get('section', 'No identificada')
        
        # Buscar en estructura del PDF
        sections = pdf_structure.get('sections', {})
        
        best_section = current_section
        best_chapter = f"Página {page_num}"
        
        for section_name, section_info in sections.items():
            start_page = section_info.get('start_page', 0)
            end_page = section_info.get('end_page', start_page + 50)
            
            if start_page <= page_num <= end_page:
                best_section = section_name
                
                # Extraer capítulo del nombre de sección
                if 'capítulo' in section_name.lower():
                    parts = section_name.split(':')
                    best_chapter = parts[0].strip() if parts else section_name
                elif section_info.get('level', 1) == 1:
                    best_chapter = section_name
                
                break
        
        return {
            'section': best_section,
            'chapter': best_chapter
        }
    
    def _detect_related_elements(self, text: str) -> List[str]:
        """Detección de elementos relacionados"""
        
        related = []
        text_lower = text.lower()
        
        # Referencias a tablas y figuras
        table_refs = re.findall(r'tabla\s+(\d+\.?\d*)', text_lower)
        figure_refs = re.findall(r'figura\s+(\d+\.?\d*)', text_lower)
        anexo_refs = re.findall(r'anexo\s+([a-z\d]+)', text_lower)
        
        related.extend([f"Tabla {ref}" for ref in table_refs])
        related.extend([f"Figura {ref}" for ref in figure_refs])
        related.extend([f"Anexo {ref}" for ref in anexo_refs])
        
        # Referencias a otros documentos o secciones
        doc_refs = re.findall(r'ver\s+(?:capítulo|sección)\s+(\d+\.?\d*)', text_lower)
        related.extend([f"Sección {ref}" for ref in doc_refs])
        
        return list(set(related))[:5]  # Máximo 5 elementos únicos
    
    def _calculate_enhanced_confidence(self, candidate: SearchResult,
                                     query_analysis: Dict[str, Any]) -> float:
        """Cálculo mejorado de confidence"""
        
        base_confidence = candidate.score
        
        # Factores de confidence
        factors = {
            'semantic_score': candidate.score,
            'text_quality': self._assess_text_quality(candidate.text),
            'relevance': self._assess_relevance(candidate.text, query_analysis),
            'completeness': self._assess_completeness(candidate.text),
            'data_presence': self._check_data_indicators(candidate.text)
        }
        
        # Pesos
        weights = {
            'semantic_score': 0.3,
            'text_quality': 0.2,
            'relevance': 0.3,
            'completeness': 0.15,
            'data_presence': 0.05
        }
        
        # Cálculo ponderado
        weighted_score = sum(
            factors[factor] * weights[factor]
            for factor in factors
        )
        
        return min(max(weighted_score, 0.1), 1.0)  # Entre 0.1 y 1.0
    
    def _assess_text_quality(self, text: str) -> float:
        """Evaluación de calidad del texto"""
        
        if not text or len(text) < 10:
            return 0.1
        
        # Métricas básicas
        char_count = len(text)
        word_count = len(text.split())
        
        # Verificar caracteres problemáticos
        noise_chars = ['�', '|||', '###', '***', '\x00']
        noise_count = sum(text.count(char) for char in noise_chars)
        noise_ratio = noise_count / char_count if char_count > 0 else 1.0
        
        # Verificar estructura de oraciones
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        avg_sentence_length = word_count / len(sentences) if sentences else 0
        
        # Scoring
        quality_score = 0.5
        
        # Bonus por bajo ruido
        if noise_ratio < 0.02:
            quality_score += 0.3
        elif noise_ratio < 0.05:
            quality_score += 0.1
        
        # Bonus por estructura adecuada
        if 5 <= avg_sentence_length <= 30:
            quality_score += 0.2
        
        return min(quality_score, 1.0)
    
    def _assess_relevance(self, text: str, query_analysis: Dict[str, Any]) -> float:
        """Evaluación de relevancia"""
        
        text_lower = text.lower()
        
        # Contar términos prioritarios
        priority_terms = query_analysis.get('priority_terms', [])
        priority_matches = sum(1 for term in priority_terms if term in text_lower)
        
        # Contar keywords
        keywords = query_analysis.get('keywords', [])
        keyword_matches = sum(1 for keyword in keywords if keyword in text_lower)
        
        # Scoring base
        total_terms = len(priority_terms) + len(keywords)
        total_matches = priority_matches + keyword_matches
        
        if total_terms == 0:
            return 0.5
        
        relevance = total_matches / total_terms
        
        # Boost por términos específicos del dominio PDT
        domain_terms = [
            'meta', 'indicador', 'resultado', 'producto', 'objetivo',
            'línea base', 'responsable', 'presupuesto', 'inversión'
        ]
        
        domain_matches = sum(1 for term in domain_terms if term in text_lower)
        if domain_matches > 0:
            relevance += 0.1 * min(domain_matches, 3)  # Máximo boost 0.3
        
        return min(relevance, 1.0)
    
    def _assess_completeness(self, text: str) -> float:
        """Evaluación de completitud"""
        
        # Indicadores de completitud
        completeness_indicators = [
            'establece', 'define', 'implementa', 'desarrolla',
            'determina', 'especifica', 'incluye', 'comprende'
        ]
        
        text_lower = text.lower()
        matches = sum(1 for indicator in completeness_indicators if indicator in text_lower)
        
        # Base score
        completeness = 0.4
        
        # Bonus por indicadores
        completeness += min(matches * 0.1, 0.4)
        
        # Bonus por longitud adecuada
        word_count = len(text.split())
        if 50 <= word_count <= 300:
            completeness += 0.2
        
        return min(completeness, 1.0)
    
    def _check_data_indicators(self, text: str) -> float:
        """Verificación de presencia de datos"""
        
        data_patterns = [
            r'\d+%',  # Porcentajes
            r'\$[\d,\.]+',  # Valores monetarios  
            r'\d{4}',  # Años
            r'\d+\s*(?:millones?|mil|miles)',  # Cantidades
            r'meta\s*:?\s*\d+',  # Metas numéricas
            r'\d+\.\d+',  # Decimales
            r'tabla\s+\d+',  # Referencias a tablas
        ]
        
        matches = sum(1 for pattern in data_patterns if re.search(pattern, text.lower()))
        
        return min(matches * 0.2, 1.0)

# ==================== VALIDADOR MEJORADO ====================

class EnhancedValidator:
    """Validador mejorado con scoring más preciso"""
    
    def validate_evidence(self, evidence: Evidence, query: str) -> ValidationResult:
        """Validación completa de evidencia"""
        
        if not evidence:
            return ValidationResult(
                final_score=0.0,
                has_all_fields=False,
                recommendations=["Evidencia no encontrada"]
            )
        
        # Validaciones individuales
        structural = self._validate_structure(evidence)
        semantic = self._validate_semantics(evidence, query)
        quality = self._validate_quality(evidence)
        
        # Cálculo de score final
        final_score = self._calculate_composite_score(structural, semantic, quality)
        
        # Generar recomendaciones
        recommendations = self._generate_actionable_recommendations(
            structural, semantic, quality
        )
        
        return ValidationResult(
            final_score=final_score,
            has_all_fields=structural['complete'],
            metadata_valid=structural['metadata_ok'],
            relevance_score=semantic['relevance'],
            coherence_score=semantic['coherence'],
            completeness=semantic['completeness'],
            text_quality=quality['quality_score'],
            data_presence=quality['has_data'],
            recommendations=recommendations
        )
    
    def _validate_structure(self, evidence: Evidence) -> Dict[str, Any]:
        """Validación estructural"""
        
        required_fields = ['text', 'page', 'section', 'confidence']
        
        complete = all(
            hasattr(evidence, field) and getattr(evidence, field) is not None
            for field in required_fields
        )
        
        metadata_ok = (
                evidence.page > 0 and
                0.0 <= evidence.confidence <= 1.0 and
                len(evidence.text.strip()) >= Config.MIN_TEXT_LENGTH
        )
        
        return {
            'complete': complete,
            'metadata_ok': metadata_ok,
            'text_length': len(evidence.text),
            'has_context': bool(evidence.context_before or evidence.context_after)
        }
    
    def _validate_semantics(self, evidence: Evidence, query: str) -> Dict[str, Any]:
        """Validación semántica"""
        
        text_lower = evidence.text.lower()
        query_lower = query.lower()
        
        # Relevancia semántica
        query_words = set(query_lower.split())
        text_words = set(text_lower.split())
        
        common_words = query_words.intersection(text_words)
        relevance = len(common_words) / len(query_words) if query_words else 0.0
        
        # Coherencia textual
        sentences = [s.strip() for s in evidence.text.split('.') if s.strip()]
        coherence = min(len(sentences) / 5.0, 1.0)  # Más oraciones = más coherente
        
        # Completitud (indicadores de respuesta completa)
        complete_indicators = ['establece', 'define', 'incluye', 'determina']
        completeness_count = sum(1 for ind in complete_indicators if ind in text_lower)
        completeness = min(completeness_count / 2.0, 1.0)
        
        return {
            'relevance': relevance,
            'coherence': coherence,
            'completeness': completeness,
            'word_overlap': len(common_words)
        }
    
    def _validate_quality(self, evidence: Evidence) -> Dict[str, Any]:
        """Validación de calidad"""
        
        text = evidence.text
        
        # Calidad del texto
        noise_chars = ['�', '###', '|||']
        noise_count = sum(text.count(char) for char in noise_chars)
        quality_score = max(0.0, 1.0 - (noise_count / len(text)))
        
        # Presencia de datos
        data_patterns = [r'\d+%', r'\$[\d,]+', r'\d{4}', r'tabla\s+\d+']
        has_data = any(re.search(pattern, text.lower()) for pattern in data_patterns)
        
        return {
            'quality_score': quality_score,
            'has_data': has_data,
            'noise_level': noise_count / len(text) if text else 1.0,
            'word_count': len(text.split())
        }
    
    def _calculate_composite_score(self, structural: Dict, semantic: Dict, 
                                 quality: Dict) -> float:
        """Cálculo de score compuesto"""
        
        # Pesos optimizados
        weights = {
            'relevance': 0.35,
            'completeness': 0.25, 
            'quality': 0.20,
            'coherence': 0.15,
            'structure': 0.05
        }
        
        # Componentes del score
        components = {
            'relevance': semantic['relevance'],
            'completeness': semantic['completeness'],
            'quality': quality['quality_score'],
            'coherence': semantic['coherence'],
            'structure': 1.0 if structural['complete'] else 0.5
        }
        
        # Penalizaciones
        penalties = 1.0
        
        if not structural['metadata_ok']:
            penalties *= 0.8
        
        if quality['noise_level'] > 0.1:
            penalties *= 0.9
        
        # Score final (0-100)
        final_score = sum(
            components[comp] * weights[comp] 
            for comp in components
        ) * penalties * 100
        
        return min(max(final_score, 0.0), 100.0)
    
    def _generate_actionable_recommendations(self, structural: Dict, 
                                           semantic: Dict, quality: Dict) -> List[str]:
        """Genera recomendaciones accionables"""
        
        recommendations = []
        
        # Recomendaciones por relevancia
        if semantic['relevance'] < 0.6:
            recommendations.append(
                f"Baja relevancia ({semantic['relevance']:.1%}): "
                "Refinar términos de búsqueda o ampliar contexto"
            )
        
        # Recomendaciones por completitud
        if semantic['completeness'] < 0.5:
            recommendations.append(
                "Respuesta incompleta: Buscar en secciones adicionales del documento"
            )
        
        # Recomendaciones por calidad
        if quality['quality_score'] < 0.7:
            recommendations.append(
                f"Calidad de texto baja ({quality['quality_score']:.1%}): "
                "Verificar extracción OCR o buscar versión original"
            )
        
        # Recomendaciones por datos
        if not quality['has_data'] and 'meta' in semantic.get('query', '').lower():
            recommendations.append(
                "Sin datos cuantitativos: Revisar anexos o tablas específicas"
            )
        
        # Recomendaciones positivas
        if not recommendations:
            score_msg = "Alta calidad" if semantic['relevance'] > 0.8 else "Calidad aceptable"
            recommendations.append(f"{score_msg}: Evidencia válida para evaluación")
        
        return recommendations

# ==================== SISTEMA PRINCIPAL INTEGRADO ====================

class OptimizedPDTSystem:
    """Sistema principal optimizado para evaluación PDT"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # Componentes principales
        self.extractor = AdvancedEvidenceExtractor()
        self.validator = EnhancedValidator()
        
        # Métricas y cache
        self.performance_stats = {
            'total_extractions': 0,
            'successful_extractions': 0,
            'avg_extraction_time': 0.0,
            'cache_hit_rate': 0.0
        }
        
        # Cache global del sistema
        self.system_cache = {}
        
        logger.info("Optimized PDT System initialized successfully")
    
    def process_pdt_document(self, pdf_path: str, 
                           dimension_questions: Dict[str, List[Dict]]) -> Dict[str, Dict]:
        """Procesa documento PDT completo"""
        
        start_time = time.time()
        results = {}
        
        logger.info(f"Processing PDT document: {Path(pdf_path).name}")
        
        try:
            # Validar archivo
            if not Path(pdf_path).exists():
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
            # Procesar por dimensiones
            for dimension, questions in dimension_questions.items():
                logger.info(f"Processing dimension: {dimension}")
                
                dimension_results = self._process_dimension(
                    pdf_path, dimension, questions
                )
                results[dimension] = dimension_results
            
            # Actualizar estadísticas
            elapsed_time = time.time() - start_time
            self._update_performance_stats(elapsed_time, len(results))
            
            logger.info(f"PDT processing completed in {elapsed_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Error processing PDT document: {e}")
            raise PDTExtractionError(f"Failed to process PDT: {e}")
        
        return results
    
    def _process_dimension(self, pdf_path: str, dimension: str, 
                         questions: List[Dict]) -> Dict[str, Dict]:
        """Procesa una dimensión específica"""
        
        dimension_results = {}
        
        for question in questions:
            question_id = question.get('id', '')
            question_text = question.get('text', '')
            
            if not question_text:
                logger.warning(f"Empty question text for ID: {question_id}")
                continue
            
            try:
                # Extraer evidencia
                evidence = self.extractor.extract_evidence(
                    pdf_path, question_text, question_id
                )
                
                # Validar evidencia
                validation = self.validator.validate_evidence(evidence, question_text)
                
                # Compilar resultado
                dimension_results[question_id] = {
                    'evidence': evidence,
                    'validation': validation,
                    'score': validation.final_score,
                    'recommendations': validation.recommendations,
                    'extraction_success': evidence is not None
                }
                
                logger.debug(f"Processed question {question_id}: Score {validation.final_score:.1f}")
                
            except Exception as e:
                logger.error(f"Error processing question {question_id}: {e}")
                dimension_results[question_id] = {
                    'evidence': None,
                    'validation': None,
                    'score': 0.0,
                    'recommendations': [f"Error en procesamiento: {str(e)}"],
                    'extraction_success': False
                }
        
        return dimension_results
    
    def _update_performance_stats(self, elapsed_time: float, num_results: int):
        """Actualiza estadísticas de rendimiento"""
        
        self.performance_stats['total_extractions'] += num_results
        
        # Calcular tasa de éxito
        if num_results > 0:
            success_count = sum(
                1 for dim_results in self.system_cache.values()
                for result in dim_results.values()
                if result.get('extraction_success', False)
            )
            self.performance_stats['successful_extractions'] = success_count
        
        # Tiempo promedio
        current_avg = self.performance_stats['avg_extraction_time']
        total_ops = self.performance_stats['total_extractions']
        
        if total_ops > 0:
            self.performance_stats['avg_extraction_time'] = (
                (current_avg * (total_ops - num_results) + elapsed_time) / total_ops
            )
        
        # Tasa de cache hits
        cache_hits = self.extractor.stats.get('cache_hits', 0)
        total_requests = self.extractor.stats.get('extractions', 1)
        self.performance_stats['cache_hit_rate'] = cache_hits / total_requests
    
    def get_system_report(self) -> Dict[str, Any]:
        """Genera reporte del sistema"""
        
        return {
            'system_info': {
                'version': '2.0_optimized',
                'components_active': 5,
                'cache_size': len(self.system_cache)
            },
            'performance': self.performance_stats,
            'extractor_stats': self.extractor.stats,
            'configuration': {
                'context_window': Config.CONTEXT_WINDOW,
                'semantic_model': Config.SEMANTIC_MODEL,
                'max_workers': Config.MAX_WORKERS
            }
        }

# ==================== EJEMPLO DE USO ====================

def main():
    """Función principal de demostración"""
    
    # Inicializar sistema
    pdt_system = OptimizedPDTSystem()
    
    # Preguntas de ejemplo del Decálogo DDHH
    sample_dimensions = {
        'DE-1': [
            {
                'id': 'DE1_Q1',
                'text': '¿El PDT define productos medibles alineados con la prioridad?'
            },
            {
                'id': 'DE1_Q2', 
                'text': '¿Las metas de producto incluyen responsable institucional?'
            },
            {
                'id': 'DE1_Q3',
                'text': '¿Formula resultados medibles con línea base y meta al 2027?'
            }
        ],
        'DE-2': [
            {
                'id': 'DE2_Q1',
                'text': '¿Identifica las causas estructurales del problema?'
            },
            {
                'id': 'DE2_Q2',
                'text': '¿El diagnóstico incluye análisis diferencial?'
            }
        ]
    }
    
    # Simular procesamiento
    print("=== SISTEMA PDT OPTIMIZADO v2.0 ===")
    print("✓ Sistema inicializado correctamente")
    print("✓ Componentes cargados:")
    print("  - Extractor de estructura PDF mejorado")
    print("  - Motor de búsqueda semántica optimizado") 
    print("  - Extractor de evidencias con IA")
    print("  - Validador avanzado con scoring preciso")
    print("  - Sistema de cache inteligente")
    print()
    print("Listo para procesar documentos PDT")
    print(f"Configuración: {len(sample_dimensions)} dimensiones de prueba")
    
    # Mostrar reporte del sistema
    report = pdt_system.get_system_report()
    print(f"\nReporte del sistema: {json.dumps(report, indent=2, ensure_ascii=False)}")

if __name__ == "__main__":
    main()
            

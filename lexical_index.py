"""
Índice léxico usando Whoosh para búsqueda full-text con BM25.
"""

import os
import shutil
from whoosh import index
from whoosh.fields import Schema, TEXT, ID, STORED, NUMERIC
from whoosh.analysis import StandardAnalyzer, LanguageAnalyzer
from whoosh.qparser import MultifieldParser, QueryParser
from whoosh.query import Term, And, Or
from whoosh.scoring import BM25F
import logging
from typing import List, Dict, Any, Optional, Union
from models import SearchResult

logger = logging.getLogger(__name__)


class LexicalIndex:
    """Índice léxico Whoosh para búsqueda full-text."""
    
    def __init__(self, index_dir: str, language: str = "es"):
        """
        Inicializa el índice léxico.
        
        Args:
            index_dir: Directorio para almacenar el índice
            language: Idioma para análisis de texto ("es" | "en")
        """
        self.index_dir = index_dir
        self.language = language
        
        # Configurar analizador según idioma
        if language == "es":
            self.analyzer = LanguageAnalyzer("es")
        elif language == "en":
            self.analyzer = LanguageAnalyzer("en") 
        else:
            self.analyzer = StandardAnalyzer()
        
        # Definir esquema
        self.schema = Schema(
            chunk_id=ID(stored=True, unique=True),
            content=TEXT(analyzer=self.analyzer, stored=True),
            title=TEXT(analyzer=self.analyzer, stored=True),
            pdt_id=ID(stored=True),
            seccion=TEXT(stored=True),
            pagina_inicio=NUMERIC(stored=True),
            pagina_fin=NUMERIC(stored=True),
            posicion_documento=NUMERIC(stored=True),
            hash_texto=ID(stored=True)
        )
        
        # Crear o abrir índice
        self.index = self._initialize_index()
        self.searcher = None
        
        logger.info(f"LexicalIndex inicializado en {index_dir}")
    
    def _initialize_index(self):
        """Inicializa o crea el índice Whoosh."""
        if not os.path.exists(self.index_dir):
            os.makedirs(self.index_dir)
        
        if index.exists_in(self.index_dir):
            return index.open_dir(self.index_dir)
        else:
            return index.create_in(self.index_dir, self.schema)
    
    def index_documents(self, chunks: List[Dict[str, Any]], batch_size: int = 1000) -> int:
        """
        Indexa documentos en lotes.
        
        Args:
            chunks: Lista de chunks con contenido y metadatos
            batch_size: Tamaño del lote para indexación
            
        Returns:
            Número de documentos indexados
        """
        writer = self.index.writer()
        indexed_count = 0
        
        try:
            for i, chunk in enumerate(chunks):
                # Validar campos requeridos
                required_fields = ["chunk_id", "content", "pdt_id", "seccion"]
                if not all(field in chunk for field in required_fields):
                    logger.warning(f"Chunk {i} no tiene todos los campos requeridos")
                    continue
                
                # Preparar documento para Whoosh
                doc = {
                    "chunk_id": chunk["chunk_id"],
                    "content": chunk["content"],
                    "title": chunk.get("title", ""),
                    "pdt_id": chunk["pdt_id"],
                    "seccion": chunk["seccion"],
                    "pagina_inicio": chunk.get("pagina_inicio", 0),
                    "pagina_fin": chunk.get("pagina_fin", 0),
                    "posicion_documento": chunk.get("posicion_documento", 0),
                    "hash_texto": chunk.get("hash_texto", "")
                }
                
                writer.add_document(**doc)
                indexed_count += 1
                
                # Commit en lotes
                if (i + 1) % batch_size == 0:
                    writer.commit()
                    writer = self.index.writer()
                    logger.info(f"Indexados {i + 1} chunks")
            
            # Commit final
            writer.commit()
            
        except Exception as e:
            writer.cancel()
            logger.error(f"Error indexando documentos: {e}")
            raise
        
        logger.info(f"Indexación completada: {indexed_count} documentos")
        return indexed_count
    
    def search_documents(self, query: str, top_k: int = 20, 
                        filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """
        Busca documentos usando BM25.
        
        Args:
            query: Consulta de búsqueda
            top_k: Número de resultados a retornar  
            filters: Filtros opcionales por campo
            
        Returns:
            Lista de SearchResult ordenados por relevancia
        """
        if not self.searcher:
            self.searcher = self.index.searcher(weighting=BM25F())
        
        try:
            # Parser multifield para buscar en content y title
            parser = MultifieldParser(["content", "title"], self.schema)
            parsed_query = parser.parse(query)
            
            # Aplicar filtros si existen
            if filters:
                filter_queries = []
                for field, value in filters.items():
                    if field in self.schema:
                        if isinstance(value, list):
                            # Filtro OR para múltiples valores
                            or_terms = [Term(field, str(v)) for v in value]
                            filter_queries.append(Or(or_terms))
                        else:
                            filter_queries.append(Term(field, str(value)))
                
                if filter_queries:
                    parsed_query = And([parsed_query] + filter_queries)
            
            # Ejecutar búsqueda
            results = self.searcher.search(parsed_query, limit=top_k)
            
            search_results = []
            for result in results:
                # Crear citation
                citation = {
                    "pdt_id": result["pdt_id"],
                    "seccion": result["seccion"],
                    "pagina": f"{result['pagina_inicio']}-{result['pagina_fin']}"
                }
                
                # Crear SearchResult
                search_result = SearchResult(
                    chunk_id=result["chunk_id"],
                    score=result.score,
                    source="lexical",
                    citation=citation,
                    text_snippet=self._extract_snippet(result, query)
                )
                search_results.append(search_result)
            
            return search_results
            
        except Exception as e:
            logger.error(f"Error en búsqueda léxica: {e}")
            return []
    
    def _extract_snippet(self, result, query: str, max_chars: int = 300) -> str:
        """
        Extrae snippet relevante del resultado.
        
        Args:
            result: Resultado de Whoosh
            query: Query original
            max_chars: Máximo caracteres del snippet
            
        Returns:
            Snippet de texto
        """
        content = result.get("content", "")
        if len(content) <= max_chars:
            return content
            
        # Buscar términos de la query en el contenido
        query_terms = query.lower().split()
        content_lower = content.lower()
        
        # Encontrar primera coincidencia
        best_pos = 0
        for term in query_terms:
            pos = content_lower.find(term)
            if pos != -1:
                best_pos = max(0, pos - max_chars // 3)
                break
        
        # Extraer snippet alrededor de la posición
        snippet = content[best_pos:best_pos + max_chars]
        
        # Agregar ellipsis si es necesario
        if best_pos > 0:
            snippet = "..." + snippet
        if best_pos + max_chars < len(content):
            snippet = snippet + "..."
            
        return snippet
    
    def update_document(self, chunk_id: str, chunk_data: Dict[str, Any]) -> bool:
        """
        Actualiza un documento existente.
        
        Args:
            chunk_id: ID del chunk a actualizar
            chunk_data: Nuevos datos del chunk
            
        Returns:
            True si se actualizó exitosamente
        """
        global writer
        try:
            writer = self.index.writer()
            
            # Preparar documento actualizado
            doc = {
                "chunk_id": chunk_id,
                "content": chunk_data["content"],
                "title": chunk_data.get("title", ""),
                "pdt_id": chunk_data["pdt_id"],
                "seccion": chunk_data["seccion"],
                "pagina_inicio": chunk_data.get("pagina_inicio", 0),
                "pagina_fin": chunk_data.get("pagina_fin", 0),
                "posicion_documento": chunk_data.get("posicion_documento", 0),
                "hash_texto": chunk_data.get("hash_texto", "")
            }
            
            # Actualizar documento (elimina el anterior con mismo chunk_id)
            writer.update_document(**doc)
            writer.commit()
            
            return True
            
        except Exception as e:
            logger.error(f"Error actualizando documento {chunk_id}: {e}")
            writer.cancel()
            return False
    
    def remove_documents(self, chunk_ids: List[str]) -> int:
        """
        Elimina documentos del índice.
        
        Args:
            chunk_ids: Lista de IDs de chunks a eliminar
            
        Returns:
            Número de documentos eliminados
        """
        global writer
        try:
            writer = self.index.writer()
            removed_count = 0
            
            for chunk_id in chunk_ids:
                writer.delete_by_term("chunk_id", chunk_id)
                removed_count += 1
            
            writer.commit()
            logger.info(f"Eliminados {removed_count} documentos")
            return removed_count
            
        except Exception as e:
            logger.error(f"Error eliminando documentos: {e}")
            writer.cancel()
            return 0
    
    def optimize_index(self) -> None:
        """Optimiza el índice para mejor performance."""
        try:
            writer = self.index.writer()
            writer.commit(optimize=True)
            logger.info("Índice optimizado")
        except Exception as e:
            logger.error(f"Error optimizando índice: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Retorna estadísticas del índice.
        
        Returns:
            Diccionario con estadísticas
        """
        with self.index.searcher() as searcher:
            doc_count = searcher.doc_count()
            
        return {
            "total_documents": doc_count,
            "index_directory": self.index_dir,
            "language": self.language,
            "schema_fields": list(self.schema.names()),
            "index_size_mb": self._get_directory_size() / (1024*1024)
        }
    
    def _get_directory_size(self) -> int:
        """Calcula tamaño del directorio de índice en bytes."""
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(self.index_dir):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                total_size += os.path.getsize(filepath)
        return total_size
    
    def close(self) -> None:
        """Cierra el searcher y libera recursos."""
        if self.searcher:
            self.searcher.close()
            self.searcher = None
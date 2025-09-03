"""
Normalizador de texto con preservación de citas
"""

import re
import ftfy
# # # from unidecode import unidecode  # Module not found  # Module not found  # Module not found
# # # from typing import Dict, List, Tuple  # Module not found  # Module not found  # Module not found
# # # from models import SectionBlock, Citation  # Module not found  # Module not found  # Module not found


class TextNormalizer:
    """Normalizador de texto para PDTs"""
    
    def __init__(self):
        # Patterns para normalización
        self.whitespace_patterns = [
            (re.compile(r'\s+'), ' '),  # Espacios múltiples
            (re.compile(r'\n\s*\n\s*\n+'), '\n\n'),  # Saltos múltiples
            (re.compile(r'[ \t]+\n'), '\n'),  # Espacios antes de salto
            (re.compile(r'\n[ \t]+'), '\n'),  # Espacios después de salto
        ]
        
        # Patterns para bullets y numerales
        self.bullet_patterns = [
            (re.compile(r'^\s*[•·▪▫○●]\s*'), '• '),  # Bullets diversos
            (re.compile(r'^\s*[-‐‑‒–—]\s*'), '- '),  # Guiones diversos
            (re.compile(r'^\s*(\d+)[\.)\]]\s*'), r'\1. '),  # Numerales
        ]
        
        # Patterns para hyphenation
        self.hyphen_patterns = [
            (re.compile(r'(\w)-\s*\n\s*(\w)'), r'\1\2'),  # Palabras cortadas
            (re.compile(r'(\w)\s*-\s*(\w)'), r'\1-\2'),  # Guiones en palabras
        ]
    
    def normalize_text(self, block: SectionBlock) -> SectionBlock:
        """
        Normaliza el texto de un bloque manteniendo offsets para citas
        
        Args:
            block: Bloque de sección a normalizar
        
        Returns:
            Bloque normalizado con citations actualizadas
        """
        original_text = block.text
        
        # Crear mapa de offsets antes de normalizar
        offset_map = self._create_offset_map(original_text)
        
        # Aplicar normalizaciones
        normalized_text = self._apply_normalizations(original_text)
        
        # Actualizar citations si existen
        updated_citations = []
        for citation in block.citations:
            new_start, new_end = self._map_citation_offsets(
                citation.char_start, citation.char_end, 
                offset_map, normalized_text
            )
            
            updated_citation = Citation(
                page=citation.page,
                char_start=new_start,
                char_end=new_end
            )
            updated_citations.append(updated_citation)
        
        # Crear nuevo bloque normalizado
        normalized_block = SectionBlock(
            section_id=block.section_id,
            section_type=block.section_type,
            page_start=block.page_start,
            page_end=block.page_end,
            text=normalized_text,
            citations=updated_citations,
            confidence=block.confidence
        )
        
        return normalized_block
    
    def preserve_citations(self, block: SectionBlock, page_to_offset: Dict[int, List[Tuple[int, int]]]) -> SectionBlock:
        """
        Preserva citas creando mapas página→offset
        
        Args:
            block: Bloque de sección
            page_to_offset: Mapeo de página a lista de (char_start, char_end)
        
        Returns:
            Bloque con citations preservadas
        """
        citations = []
        
        # Para cada página del bloque
        for page in range(block.page_start, block.page_end + 1):
            if page in page_to_offset:
                for char_start, char_end in page_to_offset[page]:
                    citation = Citation(
                        page=page,
                        char_start=char_start,
                        char_end=char_end
                    )
                    citations.append(citation)
        
        block.citations = citations
        return block
    
    def _create_offset_map(self, text: str) -> List[Tuple[int, int]]:
        """Crea mapa de offsets original→normalizado"""
        offset_map = []
        current_pos = 0
        
        for i, char in enumerate(text):
            offset_map.append((i, current_pos))
            if not char.isspace() or char in [' ', '\n']:
                current_pos += 1
        
        return offset_map
    
    def _apply_normalizations(self, text: str) -> str:
        """Aplica todas las normalizaciones al texto"""
        # 1. Fix encoding issues
        text = ftfy.fix_text(text)
        
        # 2. Normalizar espacios
        for pattern, replacement in self.whitespace_patterns:
            text = pattern.sub(replacement, text)
        
        # 3. Normalizar bullets y numerales
        lines = text.split('\n')
        normalized_lines = []
        
        for line in lines:
            for pattern, replacement in self.bullet_patterns:
                line = pattern.sub(replacement, line)
            normalized_lines.append(line)
        
        text = '\n'.join(normalized_lines)
        
        # 4. Fix hyphenation
        for pattern, replacement in self.hyphen_patterns:
            text = pattern.sub(replacement, text)
        
        # 5. Normalizar quotes y comillas
        text = self._normalize_quotes(text)
        
        # 6. Clean up final
        text = text.strip()
        
        return text
    
    def _normalize_quotes(self, text: str) -> str:
        """Normaliza comillas y quotes diversos"""
        quote_replacements = [
            ('"', '"'),  # Smart quotes
            ('"', '"'),
            (''', "'"),
            (''', "'"),
            ('«', '"'),
            ('»', '"'),
            ('„', '"'),
            ('"', '"'),
        ]
        
        for old_quote, new_quote in quote_replacements:
            text = text.replace(old_quote, new_quote)
        
        return text
    
    def _map_citation_offsets(self, 
                            original_start: int, 
                            original_end: int,
                            offset_map: List[Tuple[int, int]], 
                            normalized_text: str) -> Tuple[int, int]:
        """Mapea offsets de texto original a normalizado"""
        if not offset_map:
            return original_start, original_end
        
        # Buscar posiciones en el mapa
        new_start = 0
        new_end = 0
        
        for orig_pos, norm_pos in offset_map:
            if orig_pos <= original_start:
                new_start = norm_pos
            if orig_pos <= original_end:
                new_end = norm_pos
            if orig_pos > original_end:
                break
        
        # Asegurar que no exceda el texto normalizado
        new_start = min(new_start, len(normalized_text))
        new_end = min(new_end, len(normalized_text))
        
        return new_start, new_end
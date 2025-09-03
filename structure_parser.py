"""
Parser de estructura semántica para detectar secciones de PDT
"""

import re
# # # from typing import Dict, List, Tuple, Optional  # Module not found  # Module not found  # Module not found
# # # from models import SectionBlock, SectionType  # Module not found  # Module not found  # Module not found
import spacy


class StructureParser:
    """Parser para detectar estructura semántica de PDTs"""
    
    def __init__(self):
        # Patterns para secciones principales (basados en estándares DNP)
        self.section_patterns = {
            SectionType.DIAGNOSTICO: [
                r'(?i)\b(?:diagnóstico|diagnósticos?)\b',
                r'(?i)\b(?:situación\s+actual|estado\s+actual)\b',
                r'(?i)\b(?:problemática|problemas?\s+identificados?)\b',
                r'(?i)\b(?:análisis\s+situacional)\b'
            ],
            SectionType.PROGRAMAS: [
                r'(?i)\b(?:programas?|proyectos?)\b',
                r'(?i)\b(?:plan\s+de\s+acción|acciones?\s+estratégicas?)\b',
                r'(?i)\b(?:intervenciones?|estrategias?)\b',
                r'(?i)\b(?:componente\s+programático)\b'
            ],
            SectionType.PRESUPUESTO: [
                r'(?i)\b(?:presupuesto|presupuestos?|financiero|financiamiento)\b',
                r'(?i)\b(?:inversión|inversiones|recursos?\s+financieros?)\b',
                r'(?i)\b(?:costos?|gastos?|financiación)\b',
                r'(?i)\b(?:plan\s+financiero|marco\s+fiscal)\b'
            ],
            SectionType.METAS: [
                r'(?i)\b(?:metas?|objetivos?|indicadores?)\b',
                r'(?i)\b(?:resultados?\s+esperados?)\b',
                r'(?i)\b(?:logros?\s+proyectados?)\b',
                r'(?i)\b(?:marco\s+de\s+resultados)\b'
            ],
            SectionType.SEGUIMIENTO: [
                r'(?i)\b(?:seguimiento|monitoreo|evaluación)\b',
                r'(?i)\b(?:sistema\s+de\s+seguimiento)\b',
                r'(?i)\b(?:control|supervisión)\b',
                r'(?i)\b(?:indicadores?\s+de\s+gestión)\b'
            ]
        }
        
        # Patterns para títulos/encabezados
        self.title_patterns = [
            r'^[A-ZÁÉÍÓÚÜÑ\s\d\.\-]{5,100}$',  # MAYÚSCULAS
            r'^\d+\.?\s+[A-Za-záéíóúüñ\s\-]{5,100}$',  # Numerados
            r'^[IVX]+\.?\s+[A-Za-záéíóúüñ\s\-]{5,100}$',  # Romanos
            r'^(?:CAPÍTULO|SECCIÓN|PARTE)\s+[IVX\d]+',  # Capítulos
        ]
        
        # Cargar modelo de spaCy para español
        try:
            self.nlp = spacy.load("es_core_news_sm")
        except OSError:
            print("Advertencia: Modelo spaCy no encontrado, usando regex únicamente")
            self.nlp = None
    
    def detect_sections(self, text_by_page: Dict[int, str]) -> List[SectionBlock]:
        """
        Detecta secciones semánticas en el texto del documento
        
        Args:
            text_by_page: Diccionario con texto por página
        
        Returns:
            Lista de bloques de sección detectados
        """
        sections = []
        
        # Combinar todo el texto con marcadores de página
        full_text = ""
        page_offsets = {}
        current_offset = 0
        
        for page_num in sorted(text_by_page.keys()):
            page_offsets[page_num] = current_offset
            full_text += f"\n\n--- PÁGINA {page_num} ---\n\n"
            full_text += text_by_page[page_num]
            current_offset = len(full_text)
        
        # Detectar títulos y estructuras
        titles = self._detect_titles(full_text)
        
        # Asignar tipos de sección a títulos
        typed_titles = self._classify_titles(titles)
        
        # Crear bloques de sección
        sections = self._create_section_blocks(typed_titles, full_text, page_offsets)
        
        return sections
    
    def _detect_titles(self, text: str) -> List[Tuple[int, int, str, int]]:
        """
        Detecta títulos en el texto
        
        Returns:
            Lista de (start_pos, end_pos, title_text, level)
        """
        titles = []
        lines = text.split('\n')
        current_pos = 0
        
        for line in lines:
            line = line.strip()
            
            if self._is_title(line):
                level = self._get_title_level(line)
                start_pos = current_pos
                end_pos = current_pos + len(line)
                
                titles.append((start_pos, end_pos, line, level))
            
            current_pos += len(line) + 1  # +1 por el \n
        
        return titles
    
    def _is_title(self, line: str) -> bool:
        """Determina si una línea es un título"""
        if not line or len(line) < 5:
            return False
            
        for pattern in self.title_patterns:
            if re.match(pattern, line):
                return True
        
        # Verificar con spaCy si está disponible
        if self.nlp:
            doc = self.nlp(line)
            # Heurística: títulos suelen tener muchos nombres propios o sustantivos
            pos_tags = [token.pos_ for token in doc]
            if pos_tags.count('PROPN') + pos_tags.count('NOUN') > len(pos_tags) * 0.6:
                return True
        
        return False
    
    def _get_title_level(self, title: str) -> int:
        """Determina el nivel jerárquico del título"""
        if re.match(r'^[IVX]+\.', title):
            return 1  # Nivel principal
        elif re.match(r'^\d+\.', title):
            return 2  # Nivel secundario
        elif title.isupper():
            return 1  # MAYÚSCULAS = principal
        else:
            return 3  # Nivel terciario
    
    def _classify_titles(self, titles: List[Tuple[int, int, str, int]]) -> List[Tuple[int, int, str, int, Optional[SectionType]]]:
        """
        Clasifica títulos por tipo de sección
        
        Returns:
            Lista de (start_pos, end_pos, title_text, level, section_type)
        """
        classified = []
        
        for start_pos, end_pos, title_text, level in titles:
            section_type = None
            
            # Buscar coincidencias con patterns de sección
            for stype, patterns in self.section_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, title_text):
                        section_type = stype
                        break
                if section_type:
                    break
            
            classified.append((start_pos, end_pos, title_text, level, section_type))
        
        return classified
    
    def _create_section_blocks(self, 
                             typed_titles: List[Tuple[int, int, str, int, Optional[SectionType]]], 
                             full_text: str,
                             page_offsets: Dict[int, int]) -> List[SectionBlock]:
        """Crea bloques de sección a partir de títulos clasificados"""
        sections = []
        
        # Filtrar solo títulos con tipo de sección asignado
        section_titles = [t for t in typed_titles if t[4] is not None]
        
        for i, (start_pos, end_pos, title_text, level, section_type) in enumerate(section_titles):
            # Determinar fin de sección (inicio de la siguiente o fin del documento)
            if i + 1 < len(section_titles):
                section_end = section_titles[i + 1][0]
            else:
                section_end = len(full_text)
            
            # Extraer texto de la sección
            section_text = full_text[end_pos:section_end].strip()
            
            # Determinar páginas de inicio y fin
            page_start = self._find_page_for_position(start_pos, page_offsets)
            page_end = self._find_page_for_position(section_end, page_offsets)
            
            # Crear bloque de sección
            section_block = SectionBlock(
                section_id=f"{section_type.value}_{i}",
                section_type=section_type,
                page_start=page_start,
                page_end=page_end,
                text=section_text,
                confidence=self._calculate_section_confidence(section_text, section_type)
            )
            
            sections.append(section_block)
        
        return sections
    
    def _find_page_for_position(self, position: int, page_offsets: Dict[int, int]) -> int:
        """Encuentra la página correspondiente a una posición de texto"""
        current_page = 1
        
        for page_num in sorted(page_offsets.keys()):
            if page_offsets[page_num] <= position:
                current_page = page_num
            else:
                break
        
        return current_page
    
    def _calculate_section_confidence(self, text: str, section_type: SectionType) -> float:
        """Calcula confianza de clasificación de sección"""
        if not text:
            return 0.0
        
        # Contar coincidencias con keywords de la sección
        matches = 0
        patterns = self.section_patterns.get(section_type, [])
        
        for pattern in patterns:
            matches += len(re.findall(pattern, text))
        
        # Normalizar por longitud del texto
        confidence = min(1.0, matches / max(1, len(text.split()) // 100))
        
        return confidence
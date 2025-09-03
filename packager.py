"""
Empaquetador de documentos PDT procesados
"""

import orjson
import pandas as pd
# # # from typing import Dict, Any, List, Optional  # Module not found  # Module not found  # Module not found
# # # from datetime import datetime  # Module not found  # Module not found  # Module not found
# # # from models import DocumentEnvelope, SectionBlock, TableArtifact, DocumentPackage, QualityIndicators  # Module not found  # Module not found  # Module not found


class DocumentPackager:
    """Empaquetador para crear paquetes normalizados de PDTs"""
    
    def __init__(self):
        self.schema_version = "1.0.0"
        self.generator = "pdt-ingestion-engine"
    
    def build_document_package(self, 
                             envelope: DocumentEnvelope,
                             blocks: List[SectionBlock],
                             tables: List[TableArtifact],
                             quality_indicators: QualityIndicators) -> DocumentPackage:
        """
        Construye paquete normalizado del documento
        
        Args:
            envelope: Información del documento
            blocks: Bloques de sección procesados
            tables: Tablas extraídas
            quality_indicators: Indicadores de calidad
        
        Returns:
            Paquete de documento completo
        """
        # Crear header
        header = {
            'pdt_id': envelope.pdt_id,
            'hash': envelope.sha256_hash,
            'municipality': envelope.clean_metadata.get('municipality'),
            'department': envelope.clean_metadata.get('department'),
            'year_start': envelope.clean_metadata.get('year_start'),
            'year_end': envelope.clean_metadata.get('year_end'),
            'pages': envelope.clean_metadata.get('document_pages'),
            'generator': self.generator,
            'schema_version': self.schema_version,
            'processing_timestamp': envelope.processing_timestamp or datetime.utcnow().isoformat(),
            'gcs_source_uri': envelope.gcs_uri
        }
        
        # Crear paquete
        package = DocumentPackage(
            header=header,
            sections=blocks,
            tables=tables,
            quality_indicators=quality_indicators
        )
        
        return package
    
    def serialize_to_jsonl(self, package: DocumentPackage) -> str:
        """
        Serializa paquete a formato JSONL
        
        Args:
            package: Paquete de documento
        
        Returns:
            String en formato JSONL
        """
        lines = []
        
        # Línea 1: Header
        header_line = orjson.dumps(package.header).decode('utf-8')
        lines.append(header_line)
        
        # Líneas de secciones
        for section in package.sections:
            section_dict = {
                'type': 'section',
                'section_id': section.section_id,
                'section_type': section.section_type.value,
                'page_start': section.page_start,
                'page_end': section.page_end,
                'text': section.text,
                'confidence': section.confidence,
                'citations': [
                    {
                        'page': cite.page,
                        'char_start': cite.char_start,
                        'char_end': cite.char_end
                    }
                    for cite in section.citations
                ]
            }
            section_line = orjson.dumps(section_dict).decode('utf-8')
            lines.append(section_line)
        
        # Líneas de tablas (metadatos solamente, datos van en parquet)
        for table in package.tables:
            table_dict = {
                'type': 'table',
                'table_id': table.table_id,
                'source': table.source,
                'page': table.page,
                'csv_uri': table.csv_uri,
                'parquet_uri': table.parquet_uri,
                'quality_score': table.quality_score,
                'header_map': table.header_map,
                'shape': [table.dataframe.shape[0], table.dataframe.shape[1]] if not table.dataframe.empty else [0, 0]
            }
            table_line = orjson.dumps(table_dict).decode('utf-8')
            lines.append(table_line)
        
        # Línea de indicadores de calidad
        quality_dict = {
            'type': 'quality_indicators',
            'completeness_index': package.quality_indicators.completeness_index,
            'logical_coherence_hint': package.quality_indicators.logical_coherence_hint,
            'tables_found': package.quality_indicators.tables_found,
            'ocr_ratio': package.quality_indicators.ocr_ratio,
            'mandatory_sections_present': package.quality_indicators.mandatory_sections_present,
            'missing_sections': package.quality_indicators.missing_sections
        }
        quality_line = orjson.dumps(quality_dict).decode('utf-8')
        lines.append(quality_line)
        
        return '\n'.join(lines)
    
    def save_tables_to_parquet(self, 
                             tables: List[TableArtifact], 
                             base_uri: str) -> List[TableArtifact]:
        """
        Guarda tablas en formato Parquet
        
        Args:
            tables: Lista de artefactos de tabla
            base_uri: URI base para almacenamiento
        
        Returns:
            Lista actualizada con URIs de parquet
        """
        updated_tables = []
        
        for table in tables:
            if table.dataframe.empty:
                updated_tables.append(table)
                continue
            
            # Generar URI para parquet
            parquet_uri = f"{base_uri}/tables/{table.table_id}.parquet"
            
            # En producción, aquí se guardaría en GCS
            # Por ahora solo actualizamos la URI
            table.parquet_uri = parquet_uri
            
            # También crear CSV si es pequeña
            if table.dataframe.shape[0] < 1000:  # Menos de 1000 filas
                csv_uri = f"{base_uri}/tables/{table.table_id}.csv"
                table.csv_uri = csv_uri
            
            updated_tables.append(table)
        
        return updated_tables
    
    def create_package_manifest(self, package: DocumentPackage) -> Dict[str, Any]:
        """
        Crea manifiesto del paquete para indexación
        
        Args:
            package: Paquete de documento
        
        Returns:
            Diccionario con manifiesto
        """
        manifest = {
            'pdt_id': package.header['pdt_id'],
            'processing_info': {
                'timestamp': package.header['processing_timestamp'],
                'generator': package.header['generator'],
                'schema_version': package.header['schema_version']
            },
            'document_info': {
                'municipality': package.header['municipality'],
                'department': package.header['department'],
                'year_range': f"{package.header.get('year_start', 'N/A')}-{package.header.get('year_end', 'N/A')}",
                'pages': package.header['pages'],
                'source_uri': package.header['gcs_source_uri']
            },
            'content_summary': {
                'sections_count': len(package.sections),
                'sections_by_type': self._count_sections_by_type(package.sections),
                'tables_count': len(package.tables),
                'total_text_chars': sum(len(section.text) for section in package.sections)
            },
            'quality_summary': {
                'completeness_index': package.quality_indicators.completeness_index,
                'coherence_hint': package.quality_indicators.logical_coherence_hint,
                'ocr_ratio': package.quality_indicators.ocr_ratio,
                'missing_sections': package.quality_indicators.missing_sections
            },
            'ready_for_indexing': self._check_indexing_readiness(package)
        }
        
        return manifest
    
    def _count_sections_by_type(self, sections: List[SectionBlock]) -> Dict[str, int]:
        """Cuenta secciones por tipo"""
        counts = {}
        for section in sections:
            section_type = section.section_type.value
            counts[section_type] = counts.get(section_type, 0) + 1
        return counts
    
    def _check_indexing_readiness(self, package: DocumentPackage) -> bool:
        """
        Verifica si el paquete está listo para indexación semántica
        
        Args:
            package: Paquete de documento
        
        Returns:
            True si está listo para indexación
        """
        # Criterios mínimos para indexación
        criteria = [
            len(package.sections) > 0,  # Al menos una sección
            package.quality_indicators.completeness_index >= 0.5,  # 50% completitud mínima
            any(len(section.text) > 500 for section in package.sections),  # Al menos una sección sustancial
            package.header.get('municipality') is not None  # Municipio identificado
        ]
        
        return all(criteria)
    
    def generate_citations_index(self, sections: List[SectionBlock]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Genera índice de citas para navegación rápida
        
        Args:
            sections: Lista de secciones
        
        Returns:
            Índice de citas por tipo de sección
        """
        citations_index = {}
        
        for section in sections:
            section_type = section.section_type.value
            if section_type not in citations_index:
                citations_index[section_type] = []
            
            for citation in section.citations:
                citation_entry = {
                    'section_id': section.section_id,
                    'page': citation.page,
                    'char_start': citation.char_start,
                    'char_end': citation.char_end,
                    'text_preview': section.text[citation.char_start:citation.char_end][:100] + "..." 
                                   if citation.char_end > citation.char_start else ""
                }
                citations_index[section_type].append(citation_entry)
        
        return citations_index
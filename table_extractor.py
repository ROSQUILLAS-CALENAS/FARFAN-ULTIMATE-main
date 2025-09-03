"""
Extractor de tablas con sistema de fallback inteligente
"""

import camelot
import tabula
import pandas as pd
# # # from typing import List, Optional, Union, Dict, Any, Tuple  # Module not found  # Module not found  # Module not found
# # # from PIL import Image  # Module not found  # Module not found  # Module not found
import tempfile
import os
import logging
import time
# # # from datetime import datetime  # Module not found  # Module not found  # Module not found
# # # from models import TableArtifact  # Module not found  # Module not found  # Module not found
# # # from ocr import run_ocr, get_ocr_confidence, preprocess_image  # Module not found  # Module not found  # Module not found
import cv2
import numpy as np



# Mandatory Pipeline Contract Annotations
__phase__ = "O"
__code__ = "110O"
__stage_order__ = 7

class TableExtractor:
    """Extractor de tablas con sistema de fallback inteligente"""
    
    def __init__(self):
        self.camelot_config = {
            'flavor': 'lattice',
            'pages': 'all',
            'table_areas': None,
            'columns': None,
            'split_text': False,
            'flag_size': False,
            'strip_text': '\n'
        }
        
        self.tabula_config = {
            'pages': 'all',
            'multiple_tables': True,
            'pandas_options': {'header': 'infer'},
            'lattice': True
        }
        
        # Configuración de fallback
        self.fallback_config = {
            'min_confidence_threshold': 0.3,  # Umbral mínimo de confianza
            'min_table_size': 4,  # Mínimo número de celdas
            'min_rows': 2,
            'min_cols': 2,
            'max_empty_ratio': 0.8  # Máximo ratio de celdas vacías
        }
        
        # Sistema de logging para fallback
        self.logger = logging.getLogger(__name__)
        self.extraction_stats = {
            'primary_success': 0,
            'fallback_activated': 0,
            'ocr_fallback_success': 0,
            'total_extractions': 0
        }
    
    def extract_tables(self, 
                      pdf_path: Optional[str] = None, 
                      images: Optional[List[Image.Image]] = None,
                      page_nums: Optional[List[int]] = None) -> List[TableArtifact]:
        """
        Extrae tablas usando sistema de fallback inteligente
        
        Args:
            pdf_path: Ruta al archivo PDF
            images: Lista de imágenes (alternativa a PDF)
            page_nums: Números de página específicos
        
        Returns:
            Lista de artefactos de tabla
        """
        start_time = time.time()
        self.extraction_stats['total_extractions'] += 1
        
        tables = []
        extraction_log = {
            'timestamp': datetime.now().isoformat(),
            'pdf_path': pdf_path,
            'page_nums': page_nums,
            'methods_used': [],
            'success_rates': {},
            'fallback_triggered': False
        }
        
        if pdf_path:
            # Método primario: Camelot + Tabula tradicional
            primary_tables = self._extract_primary_methods(pdf_path, page_nums, extraction_log)
            
            # Evaluar calidad de extracción primaria
            primary_quality = self._evaluate_extraction_quality(primary_tables)
            extraction_log['primary_quality'] = primary_quality
            
            # Determinar si activar fallback
            if self._should_activate_fallback(primary_tables, primary_quality):
                self.logger.info(f"Activando fallback para {pdf_path} - Calidad primaria: {primary_quality}")
                self.extraction_stats['fallback_activated'] += 1
                extraction_log['fallback_triggered'] = True
                
                # Métodos de fallback
                fallback_tables = self._extract_fallback_methods(pdf_path, page_nums, extraction_log)
                
                # Combinar resultados, priorizando fallback si es mejor
                tables = self._merge_extraction_results(primary_tables, fallback_tables, extraction_log)
            else:
                self.extraction_stats['primary_success'] += 1
                tables = primary_tables
        
        elif images:
            # Para imágenes, usar pipeline especial con OCR
            tables = self._extract_from_images_with_fallback(images, page_nums, extraction_log)
        
        # Logging de estadísticas
        extraction_time = time.time() - start_time
        extraction_log['extraction_time_seconds'] = extraction_time
        extraction_log['tables_found'] = len(tables)
        
        self.logger.info(f"Extracción completada: {len(tables)} tablas en {extraction_time:.2f}s")
        self._log_extraction_stats(extraction_log)
        
        return self._deduplicate_and_rank(tables)
    
    def _extract_with_camelot(self, pdf_path: str, page_nums: Optional[List[int]] = None) -> List[TableArtifact]:
        """Extrae tablas usando Camelot"""
        tables = []
        
        try:
            pages = 'all' if not page_nums else ','.join(map(str, page_nums))
            
            # Intentar con lattice primero
            camelot_tables = camelot.read_pdf(
                pdf_path, 
                flavor='lattice',
                pages=pages,
                **{k: v for k, v in self.camelot_config.items() 
                   if k not in ['flavor', 'pages']}
            )
            
            # Si lattice no funciona bien, intentar con stream
            if len(camelot_tables) == 0 or all(t.accuracy < 50 for t in camelot_tables):
                camelot_tables = camelot.read_pdf(
                    pdf_path,
                    flavor='stream', 
                    pages=pages,
                    **{k: v for k, v in self.camelot_config.items() 
                       if k not in ['flavor', 'pages']}
                )
            
            for i, table in enumerate(camelot_tables):
                artifact = TableArtifact(
                    table_id=f"camelot_{table.page}_{i}",
                    source="camelot",
                    page=table.page,
                    dataframe=table.df,
                    quality_score=table.accuracy / 100.0
                )
                tables.append(artifact)
                
        except Exception as e:
            print(f"Error con Camelot: {e}")
            
        return tables
    
    def _extract_with_tabula(self, pdf_path: str, page_nums: Optional[List[int]] = None) -> List[TableArtifact]:
        """Extrae tablas usando Tabula como fallback"""
        tables = []
        
        try:
            pages = 'all' if not page_nums else page_nums
            
            dfs = tabula.read_pdf(
                pdf_path,
                pages=pages,
                **self.tabula_config
            )
            
            if not isinstance(dfs, list):
                dfs = [dfs]
            
            for i, df in enumerate(dfs):
                if df.empty:
                    continue
                    
                # Estimar página (Tabula no siempre la reporta bien)
                page = page_nums[i] if page_nums and i < len(page_nums) else i + 1
                
                # Calcular score de calidad simple
                quality_score = self._calculate_tabula_quality(df)
                
                artifact = TableArtifact(
                    table_id=f"tabula_{page}_{i}",
                    source="tabula", 
                    page=page,
                    dataframe=df,
                    quality_score=quality_score
                )
                tables.append(artifact)
                
        except Exception as e:
            print(f"Error con Tabula: {e}")
            
        return tables
    
    def _extract_from_images(self, images: List[Image.Image], page_nums: Optional[List[int]] = None) -> List[TableArtifact]:
        """Extrae tablas de imágenes convirtiendo a PDF temporal"""
        tables = []
        
        try:
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_pdf:
                # Convertir imágenes a PDF
                if images:
                    images[0].save(tmp_pdf.name, "PDF", 
                                 resolution=100.0, 
                                 save_all=True, 
                                 append_images=images[1:])
                
                # Extraer tablas del PDF temporal
                tables = self.extract_tables(pdf_path=tmp_pdf.name, page_nums=page_nums)
                
                # Limpiar archivo temporal
                os.unlink(tmp_pdf.name)
                
        except Exception as e:
            print(f"Error extrayendo tablas de imágenes: {e}")
            
        return tables
    
    def _calculate_tabula_quality(self, df: pd.DataFrame) -> float:
        """Calcula score de calidad simple para tablas de Tabula"""
        if df.empty:
            return 0.0
        
        # Métricas simples
        non_null_ratio = df.count().sum() / (df.shape[0] * df.shape[1])
        
        # Penalizar tablas con muchas celdas vacías o "Unnamed"
        unnamed_cols = sum(1 for col in df.columns if str(col).startswith('Unnamed'))
        unnamed_penalty = unnamed_cols / len(df.columns) if len(df.columns) > 0 else 0
        
        quality = non_null_ratio * (1 - unnamed_penalty)
        return max(0.0, min(1.0, quality))
    
    def _deduplicate_and_rank(self, tables: List[TableArtifact]) -> List[TableArtifact]:
        """Deduplica tablas y las ordena por calidad"""
        if not tables:
            return tables
        
        # Agrupar por página
        by_page = {}
        for table in tables:
            if table.page not in by_page:
                by_page[table.page] = []
            by_page[table.page].append(table)
        
        # Para cada página, quedarse con las mejores tablas
        result = []
        for page, page_tables in by_page.items():
            # Ordenar por calidad
            page_tables.sort(key=lambda t: t.quality_score, reverse=True)
            
            # Tomar las mejores (máximo 3 por página)
            result.extend(page_tables[:3])
        
        return result
    
    def _extract_primary_methods(self, pdf_path: str, page_nums: Optional[List[int]], 
                                extraction_log: Dict[str, Any]) -> List[TableArtifact]:
        """Extracción usando métodos primarios (Camelot + Tabula)"""
        tables = []
        
        # Camelot como método principal
        camelot_tables = self._extract_with_camelot(pdf_path, page_nums)
        tables.extend(camelot_tables)
        extraction_log['methods_used'].append('camelot')
        extraction_log['success_rates']['camelot'] = len(camelot_tables)
        
        # Tabula como complemento si Camelot encuentra pocas tablas
        if len(camelot_tables) < 2:
            tabula_tables = self._extract_with_tabula(pdf_path, page_nums)
            tables.extend(tabula_tables)
            extraction_log['methods_used'].append('tabula')
            extraction_log['success_rates']['tabula'] = len(tabula_tables)
        
        return tables
    
    def _evaluate_extraction_quality(self, tables: List[TableArtifact]) -> Dict[str, float]:
        """Evalúa la calidad de extracción de tablas"""
        if not tables:
            return {
                'overall_score': 0.0,
                'confidence_score': 0.0,
                'completeness_score': 0.0,
                'structure_score': 0.0
            }
        
        confidence_scores = []
        structure_scores = []
        completeness_scores = []
        
        for table in tables:
            # Score de confianza
            confidence_scores.append(table.quality_score)
            
            # Score de estructura (filas x columnas)
            if not table.dataframe.empty:
                rows, cols = table.dataframe.shape
                structure_score = min(1.0, (rows * cols) / 20)  # Normalizar a 20 celdas
                structure_scores.append(structure_score)
                
                # Score de completitud (ratio de celdas no vacías)
                non_empty = table.dataframe.count().sum()
                total_cells = rows * cols
                completeness_score = non_empty / total_cells if total_cells > 0 else 0
                completeness_scores.append(completeness_score)
            else:
                structure_scores.append(0.0)
                completeness_scores.append(0.0)
        
        return {
            'overall_score': sum(confidence_scores) / len(confidence_scores),
            'confidence_score': sum(confidence_scores) / len(confidence_scores),
            'completeness_score': sum(completeness_scores) / len(completeness_scores) if completeness_scores else 0,
            'structure_score': sum(structure_scores) / len(structure_scores) if structure_scores else 0,
            'table_count': len(tables)
        }
    
    def _should_activate_fallback(self, tables: List[TableArtifact], 
                                 quality_metrics: Dict[str, float]) -> bool:
        """Determina si debe activarse el sistema de fallback"""
        config = self.fallback_config
        
        # Criterios para activar fallback:
        # 1. No se encontraron tablas
        if not tables:
            self.logger.debug("Fallback: No se encontraron tablas")
            return True
        
        # 2. Confianza promedio muy baja
        if quality_metrics['confidence_score'] < config['min_confidence_threshold']:
            self.logger.debug(f"Fallback: Confianza baja ({quality_metrics['confidence_score']:.3f})")
            return True
        
        # 3. Completitud muy baja (muchas celdas vacías)
        if quality_metrics['completeness_score'] < (1 - config['max_empty_ratio']):
            self.logger.debug(f"Fallback: Completitud baja ({quality_metrics['completeness_score']:.3f})")
            return True
        
        # 4. Tablas demasiado pequeñas
        small_tables = sum(1 for t in tables if not t.dataframe.empty and 
                          (t.dataframe.shape[0] < config['min_rows'] or 
                           t.dataframe.shape[1] < config['min_cols']))
        
        if small_tables == len(tables) and small_tables > 0:
            self.logger.debug(f"Fallback: Todas las tablas son muy pequeñas ({small_tables}/{len(tables)})")
            return True
        
        return False
    
    def _extract_fallback_methods(self, pdf_path: str, page_nums: Optional[List[int]],
                                 extraction_log: Dict[str, Any]) -> List[TableArtifact]:
        """Extracción usando métodos de fallback alternativos"""
        tables = []
        
        # Método 1: OCR-based table extraction
        ocr_tables = self._extract_with_ocr(pdf_path, page_nums)
        if ocr_tables:
            tables.extend(ocr_tables)
            extraction_log['methods_used'].append('ocr')
            extraction_log['success_rates']['ocr'] = len(ocr_tables)
            self.extraction_stats['ocr_fallback_success'] += 1
        
        # Método 2: Camelot con parámetros alternativos
        alt_camelot_tables = self._extract_with_alternative_camelot(pdf_path, page_nums)
        if alt_camelot_tables:
            tables.extend(alt_camelot_tables)
            extraction_log['methods_used'].append('alt_camelot')
            extraction_log['success_rates']['alt_camelot'] = len(alt_camelot_tables)
        
        # Método 3: Tabula con configuraciones alternativas
        alt_tabula_tables = self._extract_with_alternative_tabula(pdf_path, page_nums)
        if alt_tabula_tables:
            tables.extend(alt_tabula_tables)
            extraction_log['methods_used'].append('alt_tabula')
            extraction_log['success_rates']['alt_tabula'] = len(alt_tabula_tables)
        
        return tables
    
    def _extract_with_ocr(self, pdf_path: str, page_nums: Optional[List[int]]) -> List[TableArtifact]:
        """Extracción de tablas usando OCR como método de fallback"""
        import fitz  # PyMuPDF
        
        tables = []
        
        try:
            doc = fitz.open(pdf_path)
            pages_to_process = page_nums if page_nums else range(len(doc))
            
            for page_num in pages_to_process:
                if page_num >= len(doc):
                    continue
                    
                page = doc.load_page(page_num)
                
                # Renderizar página como imagen
                mat = fitz.Matrix(2.0, 2.0)  # Escala 2x para mejor OCR
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("ppm")
                
                # Convertir a PIL Image
# # #                 from io import BytesIO  # Module not found  # Module not found  # Module not found
                image = Image.open(BytesIO(img_data))
                
                # Detectar áreas de tabla usando OCR
                table_data = self._detect_table_with_ocr(image, page_num)
                if table_data:
                    tables.extend(table_data)
            
            doc.close()
            
        except Exception as e:
            self.logger.error(f"Error en extracción OCR: {e}")
        
        return tables
    
    def _detect_table_with_ocr(self, image: Image.Image, page_num: int) -> List[TableArtifact]:
        """Detecta y extrae tablas de una imagen usando OCR"""
        tables = []
        
        try:
            # Preprocesar imagen para detectar líneas/estructuras tabulares
            img_array = np.array(image)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Detectar líneas horizontales y verticales
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
            
            horizontal_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel)
            vertical_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, vertical_kernel)
            
            # Combinar líneas para detectar estructura tabular
            table_mask = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0.0)
            
            # Encontrar contornos de posibles tablas
            contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for i, contour in enumerate(contours):
                # Filtrar contornos muy pequeños
                area = cv2.contourArea(contour)
                if area < 1000:  # Área mínima para considerar una tabla
                    continue
                
                # Extraer región de la tabla
                x, y, w, h = cv2.boundingRect(contour)
                table_region = image.crop((x, y, x + w, y + h))
                
                # Aplicar OCR a la región
                table_text = run_ocr(table_region)
                confidence = get_ocr_confidence(table_region)
                
                if table_text and confidence > 0.1:
                    # Convertir texto OCR a DataFrame
                    df = self._parse_ocr_text_to_dataframe(table_text)
                    
                    if not df.empty:
                        artifact = TableArtifact(
                            table_id=f"ocr_{page_num}_{i}",
                            source="ocr",
                            page=page_num,
                            dataframe=df,
                            quality_score=confidence
                        )
                        tables.append(artifact)
        
        except Exception as e:
            self.logger.error(f"Error detectando tabla con OCR: {e}")
        
        return tables
    
    def _parse_ocr_text_to_dataframe(self, text: str) -> pd.DataFrame:
        """Convierte texto OCR estructurado en DataFrame"""
        try:
            lines = text.strip().split('\n')
            if len(lines) < 2:
                return pd.DataFrame()
            
            # Detectar separadores comunes
            separators = ['\t', '|', '  ', ',']
            best_separator = None
            max_columns = 0
            
            for sep in separators:
                cols = len(lines[0].split(sep))
                if cols > max_columns:
                    max_columns = cols
                    best_separator = sep
            
            if not best_separator or max_columns < 2:
                return pd.DataFrame()
            
            # Parsear líneas
            data = []
            for line in lines:
                row = [col.strip() for col in line.split(best_separator) if col.strip()]
                if len(row) >= 2:  # Al menos 2 columnas
                    data.append(row)
            
            if len(data) < 2:  # Al menos header + 1 fila
                return pd.DataFrame()
            
            # Crear DataFrame
            max_cols = max(len(row) for row in data)
            normalized_data = []
            
            for row in data:
                # Normalizar longitud de filas
                normalized_row = row + [''] * (max_cols - len(row))
                normalized_data.append(normalized_row[:max_cols])
            
            df = pd.DataFrame(normalized_data[1:], columns=normalized_data[0])
            return df
            
        except Exception as e:
            self.logger.error(f"Error parseando texto OCR: {e}")
            return pd.DataFrame()
    
    def _extract_with_alternative_camelot(self, pdf_path: str, 
                                        page_nums: Optional[List[int]]) -> List[TableArtifact]:
        """Camelot con configuraciones alternativas para casos difíciles"""
        tables = []
        
        try:
            pages = 'all' if not page_nums else ','.join(map(str, page_nums))
            
            # Configuraciones alternativas para Camelot
            alt_configs = [
                {'flavor': 'stream', 'edge_tol': 500},
                {'flavor': 'lattice', 'line_scale': 15},
                {'flavor': 'stream', 'row_tol': 10, 'column_tol': 0}
            ]
            
            for config in alt_configs:
                try:
                    camelot_tables = camelot.read_pdf(pdf_path, pages=pages, **config)
                    
                    for i, table in enumerate(camelot_tables):
                        if table.accuracy > 20:  # Umbral más bajo para alternativo
                            artifact = TableArtifact(
                                table_id=f"alt_camelot_{table.page}_{i}",
                                source="alt_camelot",
                                page=table.page,
                                dataframe=table.df,
                                quality_score=table.accuracy / 100.0
                            )
                            tables.append(artifact)
                            
                except Exception as e:
                    self.logger.debug(f"Config alternativa Camelot falló: {e}")
                    continue
                    
        except Exception as e:
            self.logger.error(f"Error en Camelot alternativo: {e}")
        
        return tables
    
    def _extract_with_alternative_tabula(self, pdf_path: str,
                                       page_nums: Optional[List[int]]) -> List[TableArtifact]:
        """Tabula con configuraciones alternativas"""
        tables = []
        
        try:
            pages = 'all' if not page_nums else page_nums
            
            # Configuraciones alternativas
            alt_configs = [
                {'lattice': False, 'stream': True},
                {'area': None, 'columns': None, 'guess': False},
                {'multiple_tables': False, 'pages': pages}
            ]
            
            for config in alt_configs:
                try:
                    dfs = tabula.read_pdf(pdf_path, pages=pages, **config)
                    if not isinstance(dfs, list):
                        dfs = [dfs]
                    
                    for i, df in enumerate(dfs):
                        if not df.empty:
                            quality_score = self._calculate_tabula_quality(df)
                            if quality_score > 0.1:  # Umbral más bajo
                                page = page_nums[0] if page_nums else 1
                                artifact = TableArtifact(
                                    table_id=f"alt_tabula_{page}_{i}",
                                    source="alt_tabula",
                                    page=page,
                                    dataframe=df,
                                    quality_score=quality_score
                                )
                                tables.append(artifact)
                                
                except Exception as e:
                    self.logger.debug(f"Config alternativa Tabula falló: {e}")
                    continue
                    
        except Exception as e:
            self.logger.error(f"Error en Tabula alternativo: {e}")
        
        return tables
    
    def _extract_from_images_with_fallback(self, images: List[Image.Image],
                                         page_nums: Optional[List[int]],
                                         extraction_log: Dict[str, Any]) -> List[TableArtifact]:
        """Extracción de imágenes con sistema de fallback"""
        # Primero intentar método tradicional
        traditional_tables = self._extract_from_images(images, page_nums)
        extraction_log['methods_used'].append('image_traditional')
        
        # Evaluar calidad
        quality = self._evaluate_extraction_quality(traditional_tables)
        
        if self._should_activate_fallback(traditional_tables, quality):
            extraction_log['fallback_triggered'] = True
            
            # OCR directo en imágenes
            ocr_tables = []
            for i, image in enumerate(images):
                page_num = page_nums[i] if page_nums and i < len(page_nums) else i + 1
                page_tables = self._detect_table_with_ocr(image, page_num)
                ocr_tables.extend(page_tables)
            
            extraction_log['methods_used'].append('direct_ocr')
            extraction_log['success_rates']['direct_ocr'] = len(ocr_tables)
            
            return self._merge_extraction_results(traditional_tables, ocr_tables, extraction_log)
        
        return traditional_tables
    
    def _merge_extraction_results(self, primary_tables: List[TableArtifact],
                                fallback_tables: List[TableArtifact],
                                extraction_log: Dict[str, Any]) -> List[TableArtifact]:
        """Combina resultados de métodos primarios y fallback"""
        all_tables = primary_tables + fallback_tables
        
        if not all_tables:
            return []
        
        # Evaluar calidad de cada conjunto
        primary_quality = self._evaluate_extraction_quality(primary_tables)
        fallback_quality = self._evaluate_extraction_quality(fallback_tables)
        
        extraction_log['primary_quality'] = primary_quality
        extraction_log['fallback_quality'] = fallback_quality
        
        # Estrategia de combinación
        if fallback_quality['overall_score'] > primary_quality['overall_score'] * 1.2:
            # Fallback significativamente mejor
            self.logger.info("Usando resultados de fallback (mejor calidad)")
            return fallback_tables
        elif primary_quality['overall_score'] > 0.5:
            # Método primario aceptable, combinar
            self.logger.info("Combinando resultados primarios y fallback")
            return all_tables
        else:
            # Método primario pobre, usar fallback
            self.logger.info("Método primario pobre, usando fallback")
            return fallback_tables if fallback_tables else primary_tables
    
    def _log_extraction_stats(self, extraction_log: Dict[str, Any]) -> None:
        """Registra estadísticas de extracción para monitoreo"""
        stats_msg = (
            f"Estadísticas extracción - "
            f"Métodos: {extraction_log.get('methods_used', [])} | "
            f"Fallback: {extraction_log.get('fallback_triggered', False)} | "
            f"Tablas: {extraction_log.get('tables_found', 0)} | "
            f"Tiempo: {extraction_log.get('extraction_time_seconds', 0):.2f}s"
        )
        self.logger.info(stats_msg)
        
        # Log de tasas de éxito del fallback
        total = self.extraction_stats['total_extractions']
        fallback_rate = self.extraction_stats['fallback_activated'] / total if total > 0 else 0
        ocr_success_rate = self.extraction_stats['ocr_fallback_success'] / total if total > 0 else 0
        
        if total % 10 == 0:  # Log estadísticas cada 10 extracciones
            stats_summary = (
                f"Resumen estadísticas (últimas {total} extracciones): "
                f"Tasa fallback: {fallback_rate:.2%} | "
                f"OCR éxito: {ocr_success_rate:.2%} | "
                f"Primario éxito: {(total - self.extraction_stats['fallback_activated']) / total:.2%}"
            )
            self.logger.info(stats_summary)
    
    def get_extraction_statistics(self) -> Dict[str, Any]:
        """Retorna estadísticas de extracción para monitoreo"""
        total = self.extraction_stats['total_extractions']
        
        if total == 0:
            return {'message': 'No extractions performed yet'}
        
        return {
            'total_extractions': total,
            'primary_success_rate': (total - self.extraction_stats['fallback_activated']) / total,
            'fallback_activation_rate': self.extraction_stats['fallback_activated'] / total,
            'ocr_fallback_success_rate': self.extraction_stats['ocr_fallback_success'] / total,
            'primary_successes': self.extraction_stats['primary_success'],
            'fallback_activations': self.extraction_stats['fallback_activated'],
            'ocr_successes': self.extraction_stats['ocr_fallback_success']
        }
"""
Lector de PDF con extracción de texto y layout
"""

import gc
import io
import logging
import tempfile
import weakref
# # # from contextlib import contextmanager  # Module not found  # Module not found  # Module not found
# # # from dataclasses import dataclass  # Module not found  # Module not found  # Module not found
# # # from pathlib import Path  # Module not found  # Module not found  # Module not found
# # # from typing import Any, Dict, Iterator, List, Optional, Tuple  # Module not found  # Module not found  # Module not found

import time
import fitz  # PyMuPDF
import pdfplumber
# # # from PIL import Image  # Module not found  # Module not found  # Module not found

# # # from ocr import IntelligentOCRProcessor, needs_ocr  # Module not found  # Module not found  # Module not found
# # # from pdf_processing_error_handler import PDFErrorHandler, process_pdf_batch_with_error_handling  # Module not found  # Module not found  # Module not found
# # # from total_ordering_base import TotalOrderingBase  # Module not found  # Module not found  # Module not found


# Mandatory Pipeline Contract Annotations
__phase__ = "I"
__code__ = "05I"
__stage_order__ = 1

logger = logging.getLogger(__name__)


class PDFTemporaryFileManager:
    """Manages temporary files created during PDF processing with automatic cleanup."""
    
    def __init__(self):
        self.temp_files: List[Path] = []
        self._cleanup_finalizer = weakref.finalize(self, self._cleanup_temp_files, self.temp_files.copy())
    
    def create_temp_file(self, suffix: str = ".tmp") -> Path:
        """Create a temporary file and track it for cleanup."""
        temp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        temp_path = Path(temp_file.name)
        temp_file.close()
        self.temp_files.append(temp_path)
        return temp_path
    
    @contextmanager
    def temp_file_context(self, suffix: str = ".tmp"):
        """Context manager for temporary file with guaranteed cleanup."""
        temp_path = self.create_temp_file(suffix)
        try:
            yield temp_path
        finally:
            self._cleanup_single_file(temp_path)
    
    def cleanup(self):
        """Explicitly cleanup all temporary files."""
        for temp_file in self.temp_files:
            self._cleanup_single_file(temp_file)
        self.temp_files.clear()
    
    @staticmethod
    def _cleanup_single_file(temp_path: Path):
        """Cleanup a single temporary file."""
        try:
            if temp_path.exists():
                temp_path.unlink()
                logger.debug(f"Cleaned up temporary file: {temp_path}")
        except Exception as e:
            logger.warning(f"Failed to cleanup temporary file {temp_path}: {e}")
    
    @staticmethod
    def _cleanup_temp_files(temp_files: List[Path]):
        """Cleanup method for weakref finalizer."""
        for temp_file in temp_files:
            PDFTemporaryFileManager._cleanup_single_file(temp_file)


@dataclass
class TextSpan:
    """Span de texto con información de formato"""

    text: str
    font: str
    size: float
    bbox: Tuple[float, float, float, float]  # x0, y0, x1, y1
    page: int


@dataclass
class PageContent:
    """Contenido de una página del PDF"""

    page_num: int
    text: str
    spans: List[TextSpan]
    bbox: Tuple[float, float, float, float]
    image: Optional[Image.Image] = None


class PDFPageIterator(TotalOrderingBase):
    """Iterador para procesar páginas de PDF con OCR inteligente y streaming"""

    def __init__(self, file_path: str, enable_intelligent_ocr: bool = True, chunk_size: int = 10):
        super().__init__(component_name="PDFPageIterator")
        
        self.file_path = Path(file_path)
        self.doc = None
        self.plumber_pdf = None
        self.enable_intelligent_ocr = enable_intelligent_ocr
        self.chunk_size = chunk_size  # Number of pages to process in chunks
        self.ocr_processor = (
            IntelligentOCRProcessor() if enable_intelligent_ocr else None
        )
        self.temp_manager = PDFTemporaryFileManager()
        
        # State tracking attributes
        self._current_page = 0
        self._total_pages = 0
        self._processing_status = "initialized"
        self._pages_processed = 0
        
        # Generate deterministic ID based on file stem and settings
        self._document_stem = self.file_path.stem
        id_data = {
            "file_path": str(self.file_path),
            "file_stem": self._document_stem,
            "ocr_enabled": enable_intelligent_ocr,
            "chunk_size": chunk_size
        }
        self._deterministic_id = self.generate_stable_id(id_data, prefix="pdf")

    def __enter__(self):
        try:
            self.doc = fitz.open(str(self.file_path))
            self.plumber_pdf = pdfplumber.open(str(self.file_path))
            self._total_pages = len(self.doc)
            self._processing_status = "ready"
            
            # Update state hash for change tracking
            state_data = {
                "status": self._processing_status,
                "total_pages": self._total_pages,
                "file_path": str(self.file_path)
            }
            self.update_state_hash(state_data)
            
            return self
        except Exception as e:
            self._cleanup_resources()
            raise

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._processing_status = "completed"
        state_data = {
            "status": self._processing_status,
            "pages_processed": self._pages_processed,
            "total_pages": self._total_pages
        }
        self.update_state_hash(state_data)
        self._cleanup_resources()

    def _cleanup_resources(self):
        """Cleanup all resources including temporary files."""
        try:
            if self.doc:
                self.doc.close()
                self.doc = None
            if self.plumber_pdf:
                self.plumber_pdf.close()
                self.plumber_pdf = None
        finally:
            # Always cleanup temp files
            self.temp_manager.cleanup()
            # Force garbage collection
            gc.collect()

    def __iter__(self) -> Iterator[PageContent]:
        """Itera sobre todas las páginas del PDF con streaming por chunks"""
        total_pages = len(self.doc)
        self._processing_status = "processing"
        
        # Process pages in chunks to manage memory
        for chunk_start in range(0, total_pages, self.chunk_size):
            chunk_end = min(chunk_start + self.chunk_size, total_pages)
            
            # Generate operation ID for this chunk
            chunk_operation_id = self.generate_operation_id(
                "process_chunk", 
                {"chunk_start": chunk_start, "chunk_end": chunk_end}
            )
            
            logger.debug(f"Processing chunk {chunk_start}-{chunk_end} of {total_pages} pages [op_id: {chunk_operation_id[:8]}]")
            
            # Process chunk
            for page_num in range(chunk_start, chunk_end):
                self._current_page = page_num
                yield self._process_page(page_num)
                self._pages_processed += 1
                
                # Update state periodically
                if self._pages_processed % 5 == 0:
                    state_data = {
                        "status": self._processing_status,
                        "current_page": self._current_page,
                        "pages_processed": self._pages_processed
                    }
                    self.update_state_hash(state_data)
            
            # Force garbage collection after each chunk
            gc.collect()
            logger.debug(f"Completed chunk {chunk_start}-{chunk_end}, memory cleaned")

    def get_page(self, page_num: int) -> PageContent:
        """Obtiene una página específica"""
        operation_id = self.generate_operation_id("get_page", {"page_num": page_num})
        return self._process_page(page_num)
    
    def __lt__(self, other):
        """Comparison based on document stem for stable sorting"""
        if not isinstance(other, PDFPageIterator):
            return NotImplemented
        return self._document_stem < other._document_stem
    
    def to_json_dict(self) -> Dict[str, Any]:
        """Generate canonical JSON representation"""
        return self.sort_dict_by_keys({
            "component_id": self.component_id,
            "component_name": self.component_name,
            "document_stem": self._document_stem,
            "file_path": str(self.file_path),
            "deterministic_id": self._deterministic_id,
            "processing_status": self._processing_status,
            "total_pages": self._total_pages,
            "pages_processed": self._pages_processed,
            "enable_intelligent_ocr": self.enable_intelligent_ocr,
            "chunk_size": self.chunk_size,
            "metadata": self.get_deterministic_metadata()
        })

    def _process_page(self, page_num: int) -> PageContent:
        """Procesa una página individual con OCR inteligente si es necesario"""
        pix = None
        image = None
        
        try:
            fitz_page = self.doc[page_num]
            plumber_page = self.plumber_pdf.pages[page_num]

            # Extraer texto básico
            text = fitz_page.get_text()

            # Extraer spans con formato
            spans = self._extract_text_layout(fitz_page, page_num)

            # Obtener bbox de la página
            bbox = fitz_page.rect

            # Generar imagen usando context manager for temp files
            with self.temp_manager.temp_file_context('.png') as temp_image_path:
                pix = fitz_page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x scale
                img_data = pix.tobytes("png")
                
                # Save to temp file and load as PIL Image to manage memory
                with open(temp_image_path, 'wb') as f:
                    f.write(img_data)
                
                image = Image.open(temp_image_path)
                # Convert to RGB to ensure compatibility
                if image.mode != 'RGB':
                    image = image.convert('RGB')

                # Determinar si necesita OCR y aplicar OCR inteligente si es necesario
                if self.enable_intelligent_ocr and needs_ocr(text):
                    try:
                        logger.debug(
                            f"Page {page_num + 1} needs OCR, applying intelligent processing"
                        )
                        # For individual page, we create a single-page list
                        ocr_result = self.ocr_processor.process_document(
                            self.file_path,
                            [image],
                            additional_context={
                                "page_number": page_num + 1,
                                "single_page": True,
                            },
                        )

                        if ocr_result["success"] and ocr_result["ocr_results"]:
                            ocr_text = ocr_result["ocr_results"].get("1", "")
                            if ocr_text.strip():
                                text = ocr_text
                                logger.info(f"Enhanced page {page_num + 1} text with OCR")
                    except Exception as e:
                        logger.warning(f"OCR processing failed for page {page_num + 1}: {e}")

                page_content = PageContent(
                    page_num=page_num + 1,  # 1-indexed
                    text=text,
                    spans=spans,
                    bbox=(bbox.x0, bbox.y0, bbox.x1, bbox.y1),
                    image=image.copy(),  # Copy image to ensure it persists after temp file cleanup
                )

        finally:
            # Cleanup pixmap memory immediately
            if pix:
                pix = None
            if image:
                image.close()
            # Force garbage collection for this page
            gc.collect()

        return page_content

    def process_full_document_with_ocr(self) -> Dict[str, Any]:
        """Procesa todo el documento con OCR inteligente si es necesario usando streaming"""
        if not self.enable_intelligent_ocr:
            return {"success": False, "message": "Intelligent OCR not enabled"}

        page_images = []
        try:
            total_pages = len(self.doc)
            
            # Process in chunks to manage memory
            for chunk_start in range(0, total_pages, self.chunk_size):
                chunk_end = min(chunk_start + self.chunk_size, total_pages)
                
                # Extract images for current chunk
                chunk_images = []
                for page_num in range(chunk_start, chunk_end):
                    with self.temp_manager.temp_file_context('.png') as temp_path:
                        fitz_page = self.doc[page_num]
                        pix = fitz_page.get_pixmap(matrix=fitz.Matrix(2, 2))
                        img_data = pix.tobytes("png")
                        
                        # Save to temp file and load
                        with open(temp_path, 'wb') as f:
                            f.write(img_data)
                        
                        image = Image.open(temp_path)
                        if image.mode != 'RGB':
                            image = image.convert('RGB')
                        
                        chunk_images.append(image.copy())
                        
                        # Cleanup immediately
                        image.close()
                        pix = None
                
                page_images.extend(chunk_images)
                
                # Force garbage collection after each chunk
                gc.collect()
                logger.debug(f"Processed OCR image chunk {chunk_start}-{chunk_end}")

            # Process with intelligent OCR
            result = self.ocr_processor.process_document(
                self.file_path, page_images, additional_context={"full_document": True}
            )
            
            return result

        except Exception as e:
            logger.error(f"Error in full document OCR processing: {e}")
            return {"success": False, "error": str(e)}
        finally:
            # Cleanup all page images
            for img in page_images:
                if img and not img.closed:
                    img.close()
            page_images.clear()
            gc.collect()


def extract_text_layout(page: fitz.Page, page_num: int) -> List[TextSpan]:
    """Extrae layout de texto con información de formato"""
    spans = []

    blocks = page.get_text("dict")

    for block in blocks.get("blocks", []):
        if "lines" not in block:
            continue

        for line in block["lines"]:
            for span in line.get("spans", []):
                text_span = TextSpan(
                    text=span["text"],
                    font=span["font"],
                    size=span["size"],
                    bbox=(
                        span["bbox"][0],
                        span["bbox"][1],
                        span["bbox"][2],
                        span["bbox"][3],
                    ),
                    page=page_num + 1,
                )
                spans.append(text_span)

    return spans


def _extract_text_layout(self, page: fitz.Page, page_num: int) -> List[TextSpan]:
    """Método helper para extraer layout"""
    return extract_text_layout(page, page_num)


def stream_pdf_documents(pdf_paths: List[str], 
                        chunk_size: int = 10,
                        enable_ocr: bool = True,
                        callback_func: callable = None) -> Iterator[Tuple[str, PageContent]]:
    """
    Stream process multiple PDF documents with automatic memory management.
    
    Args:
        pdf_paths: List of PDF file paths
        chunk_size: Number of pages to process per chunk
        enable_ocr: Enable intelligent OCR processing
        callback_func: Optional callback function called after each document
    
    Yields:
        Tuple of (pdf_path, page_content) for each page
    """
    for pdf_path in pdf_paths:
        logger.info(f"Starting streaming processing of: {pdf_path}")
        
        try:
            with PDFPageIterator(pdf_path, enable_intelligent_ocr=enable_ocr, 
                               chunk_size=chunk_size) as pdf_iter:
                
                page_count = 0
                for page_content in pdf_iter:
                    yield (pdf_path, page_content)
                    page_count += 1
                
# # #                 logger.info(f"Completed processing {page_count} pages from {pdf_path}")  # Module not found  # Module not found  # Module not found
                
                # Call optional callback after each document
                if callback_func:
                    try:
                        callback_func(pdf_path, page_count)
                    except Exception as e:
                        logger.warning(f"Callback function failed for {pdf_path}: {e}")
                
        except Exception as e:
            logger.error(f"Failed to process PDF {pdf_path}: {e}")
            raise
        finally:
            # Force garbage collection after each document
            gc.collect()
            logger.debug(f"Memory cleanup completed for {pdf_path}")


def process_pdf_batch_with_cleanup(pdf_paths: List[str],
                                 processing_func: callable,
                                 chunk_size: int = 10,
                                 enable_ocr: bool = True) -> List[Dict[str, Any]]:
    """
    Process a batch of PDFs with automatic cleanup and memory management.
    
    Args:
        pdf_paths: List of PDF file paths to process
        processing_func: Function that takes (pdf_path, page_content) and returns processed data
        chunk_size: Number of pages to process per chunk
        enable_ocr: Enable intelligent OCR processing
        
    Returns:
        List of processing results for each PDF
    """
    results = []
    temp_manager = PDFTemporaryFileManager()
    
    try:
        for pdf_path, page_content in stream_pdf_documents(
            pdf_paths, chunk_size=chunk_size, enable_ocr=enable_ocr
        ):
            try:
                # Process the page content
                result = processing_func(pdf_path, page_content)
                results.append({
                    'pdf_path': pdf_path,
                    'page_num': page_content.page_num,
                    'result': result,
                    'success': True
                })
                
            except Exception as e:
                logger.error(f"Processing failed for {pdf_path} page {page_content.page_num}: {e}")
                results.append({
                    'pdf_path': pdf_path,
                    'page_num': page_content.page_num,
                    'error': str(e),
                    'success': False
                })
            
            # Force garbage collection after each page
            gc.collect()
    
    finally:
        # Cleanup any temporary files
        temp_manager.cleanup()
        # Final garbage collection
        gc.collect()
    
    return results


def _extract_text_layout(self, page: fitz.Page, page_num: int) -> List[TextSpan]:
    """Método helper para extraer layout"""
    return extract_text_layout(page, page_num)


class PDFBatchProcessor(TotalOrderingBase):
    """Enhanced PDF batch processing with comprehensive error handling"""
    
    def __init__(self, 
                 enable_intelligent_ocr: bool = True,
                 checkpoint_frequency: int = 10,
                 max_file_size_mb: int = 100,
                 memory_threshold_mb: int = 2048,
                 max_retry_attempts: int = 3):
        """
        Initialize PDF batch processor with error handling
        
        Args:
            enable_intelligent_ocr: Enable OCR processing
            checkpoint_frequency: Save checkpoint every N documents
            max_file_size_mb: Maximum PDF file size in MB
            memory_threshold_mb: Memory usage threshold in MB
            max_retry_attempts: Maximum retry attempts for failed operations
        """
        super().__init__(component_name="PDFBatchProcessor")
        
        self.enable_intelligent_ocr = enable_intelligent_ocr
        self.logger = logging.getLogger(__name__)
        
        # State tracking attributes
        self._batch_count = 0
        self._processed_files = 0
        self._failed_files = 0
        self._processing_order = 0
        
        # Initialize error handler
        self.error_handler = PDFErrorHandler(
            checkpoint_frequency=checkpoint_frequency,
            max_file_size_mb=max_file_size_mb,
            memory_threshold_mb=memory_threshold_mb,
            max_retry_attempts=max_retry_attempts
        )
        
        # Generate deterministic ID based on configuration
        config_data = {
            "enable_ocr": enable_intelligent_ocr,
            "checkpoint_frequency": checkpoint_frequency,
            "max_file_size_mb": max_file_size_mb,
            "memory_threshold_mb": memory_threshold_mb,
            "max_retry_attempts": max_retry_attempts
        }
        self._config_id = self.generate_stable_id(config_data, prefix="batch")
    
    def process_pdf_batch(self, 
                         file_paths: List[str],
                         batch_id: Optional[str] = None,
                         resume_from_checkpoint: bool = True) -> Dict[str, Any]:
        """
        Process a batch of PDF files with comprehensive error handling
        
        Args:
            file_paths: List of PDF file paths
            batch_id: Optional batch identifier
# # #             resume_from_checkpoint: Whether to resume from existing checkpoint  # Module not found  # Module not found  # Module not found
            
        Returns:
            Processing results summary
        """
        self._batch_count += 1
        
        # Generate batch operation ID
        batch_operation_id = self.generate_operation_id(
            "process_batch", 
            {
                "batch_id": batch_id or f"batch_{self._batch_count}",
                "file_count": len(file_paths),
                "resume_from_checkpoint": resume_from_checkpoint
            }
        )
        
        # Sort file paths deterministically
        sorted_paths = self.sort_collection(file_paths)
        
        result = self.error_handler.process_pdf_batch(
            file_paths=sorted_paths,
            processing_function=self._process_single_pdf,
            batch_id=batch_id,
            resume_from_checkpoint=resume_from_checkpoint
        )
        
        # Update processing statistics
        self._processed_files += result.get("successful_count", 0)
        self._failed_files += result.get("failed_count", 0)
        
        # Update state hash
        state_data = {
            "batch_count": self._batch_count,
            "processed_files": self._processed_files,
            "failed_files": self._failed_files,
            "config_id": self._config_id
        }
        self.update_state_hash(state_data)
        
        return result
    
    def _process_single_pdf(self, file_path: Path) -> Dict[str, Any]:
        """
        Process a single PDF file
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Dictionary with extracted content
        """
        self._processing_order += 1
        
        # Generate artifact ID for this processing result
        artifact_id = self.generate_artifact_id("pdf_result", {"file_path": str(file_path)})
        
        try:
            result = {
                "file_path": str(file_path),
                "pages": [],
                "total_pages": 0,
                "processing_time_ms": 0,
                "ocr_used": False,
                "artifact_id": artifact_id,
                "processing_order": self._processing_order
            }
            
            start_time = time.time()
            
            with PDFPageIterator(str(file_path), self.enable_intelligent_ocr) as pdf_iterator:
                # Get total page count
                result["total_pages"] = len(pdf_iterator.doc)
                
                # Process each page with deterministic ordering
                pages_data = []
                for page_content in pdf_iterator:
                    page_data = {
                        "page_num": page_content.page_num,
                        "text": page_content.text,
                        "bbox": page_content.bbox,
                        "spans_count": len(page_content.spans),
                        "has_image": page_content.image is not None
                    }
                    pages_data.append(page_data)
                
                # Sort pages by page number for deterministic output
                result["pages"] = sorted(pages_data, key=lambda x: x["page_num"])
                
                # Check if full document OCR was used
                if self.enable_intelligent_ocr:
                    ocr_result = pdf_iterator.process_full_document_with_ocr()
                    if ocr_result.get("success", False):
                        result["ocr_used"] = True
                        result["ocr_confidence"] = ocr_result.get("confidence", 0.0)
            
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000
            result["processing_time_ms"] = round(processing_time, 2)
            
            self.logger.info(f"Successfully processed {file_path}: {result['total_pages']} pages, "
                           f"{processing_time:.1f}ms [artifact_id: {artifact_id[:8]}]")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing PDF {file_path}: {e}")
            raise
    
    def __lt__(self, other):
        """Comparison based on processing order for stable sorting"""
        if not isinstance(other, PDFBatchProcessor):
            return NotImplemented
        return self._processing_order < other._processing_order
    
    def to_json_dict(self) -> Dict[str, Any]:
        """Generate canonical JSON representation"""
        return self.serialize_output({
            "component_id": self.component_id,
            "component_name": self.component_name,
            "config_id": self._config_id,
            "enable_intelligent_ocr": self.enable_intelligent_ocr,
            "batch_count": self._batch_count,
            "processed_files": self._processed_files,
            "failed_files": self._failed_files,
            "processing_order": self._processing_order,
            "metadata": self.get_deterministic_metadata()
        })
    
    def resume_from_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
# # #         """Resume processing from a checkpoint"""  # Module not found  # Module not found  # Module not found
        return self.error_handler.resume_from_checkpoint(
            checkpoint_path=checkpoint_path,
            processing_function=self._process_single_pdf
        )
    
    def cleanup_resources(self):
        """Clean up resources"""
        if hasattr(self.error_handler, 'resource_monitor'):
            self.error_handler.resource_monitor.stop_monitoring()


# Convenience functions for backward compatibility
def process(data=None, context=None) -> Dict[str, Any]:
    """
    Process API for PDF reader component (01I).
    
# # #     Extracts text from PDF files and writes standardized artifacts using ArtifactManager.  # Module not found  # Module not found  # Module not found
    
    Args:
        data: Input data (file paths or file content)
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
    
# # #     # Extract file paths from input  # Module not found  # Module not found  # Module not found
    file_paths = []
    if isinstance(data, str):
        file_paths = [data]
    elif isinstance(data, list):
        file_paths = data
    elif context and 'file_path' in context:
        file_paths = [context['file_path']]
    elif context and 'file_paths' in context:
        file_paths = context['file_paths']
    else:
        return {"error": "No file paths provided in data or context"}
    
    results = []
    
    for file_path in file_paths:
        try:
# # #             # Determine stem from file path  # Module not found  # Module not found  # Module not found
            stem = Path(file_path).stem
            
            # Process PDF
            with PDFPageIterator(file_path, enable_intelligent_ocr=True) as pdf_iter:
                extracted_pages = []
                for page_content in pdf_iter:
                    extracted_pages.append({
                        "page_num": page_content.page_num,
                        "text": page_content.text,
                        "bbox": page_content.bbox
                    })
                
                # Prepare artifact data
                artifact_data = {
                    "document_stem": stem,
                    "file_path": str(file_path),
                    "total_pages": len(extracted_pages),
                    "pages": extracted_pages,
                    "processing_metadata": {
                        "component": "01I",
                        "processor": "PDFPageIterator",
                        "ocr_enabled": True,
                        "timestamp": str(__import__('datetime').datetime.now())
                    }
                }
                
                # Write artifact using ArtifactManager
                output_path = artifact_manager.write_artifact(stem, "text", artifact_data)
                
                results.append({
                    "file_path": file_path,
                    "stem": stem,
                    "success": True,
                    "output_path": str(output_path),
                    "pages_processed": len(extracted_pages),
                    "artifact_type": "text"
                })
                
        except Exception as e:
            # Write error artifact
            error_data = {
                "document_stem": Path(file_path).stem,
                "file_path": str(file_path),
                "error": str(e),
                "processing_metadata": {
                    "component": "01I", 
                    "status": "failed",
                    "timestamp": str(__import__('datetime').datetime.now())
                }
            }
            
            try:
                error_stem = Path(file_path).stem
                output_path = artifact_manager.write_artifact(error_stem, "text", error_data)
                
                results.append({
                    "file_path": file_path,
                    "stem": error_stem,
                    "success": False,
                    "error": str(e),
                    "output_path": str(output_path),
                    "artifact_type": "text"
                })
            except Exception as artifact_error:
                results.append({
                    "file_path": file_path,
                    "success": False,
                    "error": f"Processing failed: {str(e)}, Artifact writing failed: {str(artifact_error)}"
                })
    
    return {
        "component": "01I",
        "results": results,
        "total_files": len(file_paths),
        "successful_files": len([r for r in results if r.get('success', False)])
    }


def process_pdf_files_with_error_handling(file_paths: List[str], **kwargs) -> Dict[str, Any]:
    """
    Convenience function for processing PDF files with error handling
    
    Args:
        file_paths: List of PDF file paths
        **kwargs: Additional arguments for PDFBatchProcessor
        
    Returns:
        Processing results
    """
    processor = PDFBatchProcessor(**kwargs)
    try:
        return processor.process_pdf_batch(file_paths)
    finally:
        processor.cleanup_resources()


def extract_content_from_pdf_with_retry(file_path: str, 
                                      max_attempts: int = 3,
                                      enable_ocr: bool = True) -> Dict[str, Any]:
    """
# # #     Extract content from a single PDF with retry logic  # Module not found  # Module not found  # Module not found
    
    Args:
        file_path: PDF file path
        max_attempts: Maximum retry attempts
        enable_ocr: Enable OCR processing
        
    Returns:
        Extracted content dictionary
    """
    processor = PDFBatchProcessor(
        enable_intelligent_ocr=enable_ocr,
        max_retry_attempts=max_attempts
    )
    
    try:
        results = processor.process_pdf_batch([file_path])
        if results["successful_files"] > 0:
            return results["results"][0]["result"]
        else:
            raise Exception(f"Failed to process PDF: {results['failed_documents']}")
    finally:
        processor.cleanup_resources()

"""
Intelligent OCR Decision System with Cloud Storage Caching
Evaluates document characteristics to determine OCR cost-effectiveness
"""

import hashlib
import json
import logging
import re
# # # from dataclasses import dataclass  # Module not found  # Module not found  # Module not found
# # # from datetime import datetime, timedelta  # Module not found  # Module not found  # Module not found
# # # from pathlib import Path  # Module not found  # Module not found  # Module not found
# # # from typing import Any, Dict, Optional, Tuple  # Module not found  # Module not found  # Module not found

import cv2
import numpy as np
import pytesseract
# # # from config_consolidated.settings import get_settings  # Module not found  # Module not found  # Module not found
# # # from google.cloud import storage  # Module not found  # Module not found  # Module not found
# # # from PIL import Image  # Module not found  # Module not found  # Module not found


# Mandatory Pipeline Contract Annotations
__phase__ = "O"
__code__ = "90O"
__stage_order__ = 7

logger = logging.getLogger(__name__)


@dataclass
class DocumentCharacteristics:
    """Document characteristics for OCR decision making."""

    file_size_mb: float
    page_count: int
    avg_text_density: float
    image_complexity_score: float
    has_tables: bool
    has_images: bool
    estimated_processing_time: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_size_mb": self.file_size_mb,
            "page_count": self.page_count,
            "avg_text_density": self.avg_text_density,
            "image_complexity_score": self.image_complexity_score,
            "has_tables": self.has_tables,
            "has_images": self.has_images,
            "estimated_processing_time": self.estimated_processing_time,
        }


@dataclass
class OCRDecision:
    """OCR processing decision with rationale."""

    should_process: bool
    estimated_cost: float
    decision_reasons: list[str]
    confidence: float
    alternative_suggestions: list[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "should_process": self.should_process,
            "estimated_cost": self.estimated_cost,
            "decision_reasons": self.decision_reasons,
            "confidence": self.confidence,
            "alternative_suggestions": self.alternative_suggestions,
        }


@dataclass
class OCRCacheEntry:
    """OCR cache entry with metadata."""

    document_hash: str
    ocr_results: Dict[str, str]  # page_num -> text
    processing_metadata: Dict[str, Any]
    timestamp: datetime
    expiration_date: datetime

    def is_expired(self) -> bool:
        return datetime.utcnow() > self.expiration_date

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_hash": self.document_hash,
            "ocr_results": self.ocr_results,
            "processing_metadata": self.processing_metadata,
            "timestamp": self.timestamp.isoformat(),
            "expiration_date": self.expiration_date.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OCRCacheEntry":
        return cls(
            document_hash=data["document_hash"],
            ocr_results=data["ocr_results"],
            processing_metadata=data["processing_metadata"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            expiration_date=datetime.fromisoformat(data["expiration_date"]),
        )


class IntelligentOCRDecisionSystem:
    """Intelligent decision system for OCR processing."""

    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self.ocr_settings = self.settings.ocr

    def evaluate_document(
        self, file_path: str, page_images: list[Image.Image]
    ) -> DocumentCharacteristics:
        """Evaluate document characteristics for OCR decision."""
        file_size_mb = Path(file_path).stat().st_size / (1024 * 1024)
        page_count = len(page_images)

        # Calculate average text density and image complexity
        text_densities = []
        complexity_scores = []

        for image in page_images[: min(5, len(page_images))]:  # Sample first 5 pages
            text_density = self._calculate_text_density(image)
            complexity_score = self._calculate_image_complexity(image)

            text_densities.append(text_density)
            complexity_scores.append(complexity_score)

        avg_text_density = (
            sum(text_densities) / len(text_densities) if text_densities else 0.0
        )
        image_complexity_score = (
            sum(complexity_scores) / len(complexity_scores)
            if complexity_scores
            else 0.0
        )

        # Detect tables and images (simplified heuristics)
        has_tables = self._detect_tables(page_images[:3])
        has_images = self._detect_images(page_images[:3])

        # Estimate processing time
        estimated_processing_time = self._estimate_processing_time(
            page_count, image_complexity_score
        )

        return DocumentCharacteristics(
            file_size_mb=file_size_mb,
            page_count=page_count,
            avg_text_density=avg_text_density,
            image_complexity_score=image_complexity_score,
            has_tables=has_tables,
            has_images=has_images,
            estimated_processing_time=estimated_processing_time,
        )

    def make_ocr_decision(
        self, characteristics: DocumentCharacteristics
    ) -> OCRDecision:
        """Make intelligent decision about OCR processing."""
        decision_reasons = []
        alternative_suggestions = []
        should_process = True
        confidence = 1.0

        # Calculate estimated cost
        estimated_cost = characteristics.page_count * self.ocr_settings.cost_per_page

        # Check file size constraints
        if characteristics.file_size_mb > self.ocr_settings.max_file_size_mb:
            should_process = False
            decision_reasons.append(
                f"File size ({characteristics.file_size_mb:.1f}MB) exceeds limit ({self.ocr_settings.max_file_size_mb}MB)"
            )
            alternative_suggestions.append(
                "Consider splitting the document into smaller parts"
            )

        # Check page count constraints
        if characteristics.page_count > self.ocr_settings.max_page_count:
            should_process = False
            decision_reasons.append(
                f"Page count ({characteristics.page_count}) exceeds limit ({self.ocr_settings.max_page_count})"
            )
            alternative_suggestions.append("Process document in batches")

        # Check cost threshold
        if estimated_cost > self.ocr_settings.max_cost_threshold:
            should_process = False
            decision_reasons.append(
                f"Estimated cost (${estimated_cost:.2f}) exceeds threshold (${self.ocr_settings.max_cost_threshold:.2f})"
            )
            alternative_suggestions.append("Use local OCR for cost savings")

        # Check text density - skip OCR if already high text density
        if characteristics.avg_text_density > 0.8:
            should_process = False
            decision_reasons.append(
                f"High text density ({characteristics.avg_text_density:.2f}) suggests text extraction may be sufficient"
            )
            alternative_suggestions.append("Try direct PDF text extraction first")
            confidence = 0.9

        # Check image complexity
        if (
            self.ocr_settings.enable_image_complexity_check
            and characteristics.image_complexity_score
            > self.ocr_settings.max_image_complexity_score
        ):
            confidence *= 0.8
            decision_reasons.append(
                f"High image complexity ({characteristics.image_complexity_score:.2f}) may reduce OCR accuracy"
            )
            alternative_suggestions.append("Consider manual review of complex pages")

        # Positive indicators for OCR
        if (
            characteristics.avg_text_density
            < self.ocr_settings.min_text_density_threshold
        ):
            if should_process:
                decision_reasons.append(
                    f"Low text density ({characteristics.avg_text_density:.2f}) indicates OCR may be beneficial"
                )

        if characteristics.has_tables and should_process:
            decision_reasons.append(
# # #                 "Document contains tables that may benefit from OCR"  # Module not found  # Module not found  # Module not found
            )

        if characteristics.has_images and should_process:
            decision_reasons.append(
                "Document contains images with potential text content"
            )

        if not decision_reasons:
            decision_reasons.append("Document characteristics support OCR processing")

        return OCRDecision(
            should_process=should_process,
            estimated_cost=estimated_cost,
            decision_reasons=decision_reasons,
            confidence=confidence,
            alternative_suggestions=alternative_suggestions,
        )

    def _calculate_text_density(self, image: Image.Image) -> float:
        """Calculate text density in image."""
        try:
            # Convert to grayscale
            gray = image.convert("L")
            img_array = np.array(gray)

            # Apply threshold to find potential text regions
            _, binary = cv2.threshold(
                img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

            # Find contours
            contours, _ = cv2.findContours(
                binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            # Calculate text area ratio
            total_area = image.width * image.height
            text_area = sum(cv2.contourArea(contour) for contour in contours)

            return min(text_area / total_area, 1.0)
        except Exception as e:
            logger.warning(f"Error calculating text density: {e}")
            return 0.5  # Default moderate density

    def _calculate_image_complexity(self, image: Image.Image) -> float:
        """Calculate image complexity score."""
        try:
            # Convert to OpenCV format
            img_array = np.array(image.convert("RGB"))
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

            # Calculate edge density using Canny
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size

            # Calculate color variance
            color_variance = np.var(img_array) / (255.0 * 255.0)

            # Combine metrics
            complexity_score = (edge_density * 0.7) + (color_variance * 0.3)

            return min(complexity_score, 1.0)
        except Exception as e:
            logger.warning(f"Error calculating image complexity: {e}")
            return 0.5  # Default moderate complexity

    def _detect_tables(self, sample_images: list[Image.Image]) -> bool:
        """Detect potential tables in sample images."""
        try:
            for image in sample_images:
                img_array = np.array(image.convert("L"))

                # Detect horizontal and vertical lines
                horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
                vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))

                horizontal_lines = cv2.morphologyEx(
                    img_array, cv2.MORPH_OPEN, horizontal_kernel
                )
                vertical_lines = cv2.morphologyEx(
                    img_array, cv2.MORPH_OPEN, vertical_kernel
                )

                # Count significant lines
                h_lines = np.sum(horizontal_lines > 0)
                v_lines = np.sum(vertical_lines > 0)

                if h_lines > 1000 and v_lines > 1000:  # Threshold for table detection
                    return True

            return False
        except Exception as e:
            logger.warning(f"Error detecting tables: {e}")
            return False

    def _detect_images(self, sample_images: list[Image.Image]) -> bool:
        """Detect embedded images in sample images."""
        try:
            for image in sample_images:
                img_array = np.array(image.convert("RGB"))

                # Look for large uniform color regions (potential images)
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

                # Find contours of large regions
                _, binary = cv2.threshold(
                    gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )
                contours, _ = cv2.findContours(
                    binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

                large_regions = [c for c in contours if cv2.contourArea(c) > 10000]

                if len(large_regions) > 2:  # Multiple large regions suggest images
                    return True

            return False
        except Exception as e:
            logger.warning(f"Error detecting images: {e}")
            return False

    def _estimate_processing_time(
        self, page_count: int, complexity_score: float
    ) -> float:
        """Estimate OCR processing time in minutes."""
        base_time_per_page = 0.5  # 30 seconds per page baseline
        complexity_multiplier = 1.0 + complexity_score

        return page_count * base_time_per_page * complexity_multiplier


class CloudOCRCache:
    """Cloud storage-based caching system for OCR results."""

    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self.ocr_settings = self.settings.ocr

        if self.ocr_settings.enable_cache:
            self.storage_client = storage.Client(project=self.settings.gcp.project_id)
            self.bucket = self.storage_client.bucket(self.ocr_settings.cache_bucket)

    def generate_document_hash(
        self, file_path: str, additional_context: Dict[str, Any] = None
    ) -> str:
        """Generate document hash for cache key."""
        # Use file content hash as primary identifier
        with open(file_path, "rb") as f:
            content_hash = hashlib.sha256(f.read()).hexdigest()

        # Include additional context if provided
        if additional_context:
            context_str = json.dumps(additional_context, sort_keys=True)
            content_hash = hashlib.sha256(
                (content_hash + context_str).encode()
            ).hexdigest()

        return content_hash

    def get_cache_key(self, document_hash: str) -> str:
# # #         """Generate cache key from document hash."""  # Module not found  # Module not found  # Module not found
        return f"{self.ocr_settings.cache_key_prefix}{document_hash}"

    def get_cached_results(self, document_hash: str) -> Optional[OCRCacheEntry]:
        """Retrieve cached OCR results."""
        if not self.ocr_settings.enable_cache:
            return None

        try:
            cache_key = self.get_cache_key(document_hash)
            blob = self.bucket.blob(cache_key)

            if not blob.exists():
                logger.debug(f"No cache entry found for document hash: {document_hash}")
                return None

            # Download and parse cache entry
            cache_data = json.loads(blob.download_as_text())
            cache_entry = OCRCacheEntry.from_dict(cache_data)

            # Check if cache entry is expired
            if cache_entry.is_expired():
                logger.info(f"Cache entry expired for document hash: {document_hash}")
                self._delete_cache_entry(cache_key)
                return None

            logger.info(f"Found valid cache entry for document hash: {document_hash}")
            return cache_entry

        except Exception as e:
            logger.error(f"Error retrieving cache entry: {e}")
            return None

    def store_cache_entry(
        self,
        document_hash: str,
        ocr_results: Dict[str, str],
        processing_metadata: Dict[str, Any],
    ) -> bool:
        """Store OCR results in cache."""
        if not self.ocr_settings.enable_cache:
            return False

        try:
            cache_key = self.get_cache_key(document_hash)
            expiration_date = datetime.utcnow() + timedelta(
                days=self.ocr_settings.cache_expiration_days
            )

            cache_entry = OCRCacheEntry(
                document_hash=document_hash,
                ocr_results=ocr_results,
                processing_metadata=processing_metadata,
                timestamp=datetime.utcnow(),
                expiration_date=expiration_date,
            )

            # Upload to cloud storage
            blob = self.bucket.blob(cache_key)
            blob.upload_from_string(
                json.dumps(cache_entry.to_dict(), default=str),
                content_type="application/json",
            )

            # Set metadata
            blob.metadata = {
                "document_hash": document_hash,
                "created_at": cache_entry.timestamp.isoformat(),
                "expires_at": expiration_date.isoformat(),
            }
            blob.patch()

            logger.info(f"Stored cache entry for document hash: {document_hash}")
            return True

        except Exception as e:
            logger.error(f"Error storing cache entry: {e}")
            return False

    def _delete_cache_entry(self, cache_key: str) -> bool:
        """Delete expired cache entry."""
        try:
            blob = self.bucket.blob(cache_key)
            blob.delete()
            logger.debug(f"Deleted expired cache entry: {cache_key}")
            return True
        except Exception as e:
            logger.error(f"Error deleting cache entry: {e}")
            return False

    def cleanup_expired_entries(self) -> int:
        """Cleanup expired cache entries."""
        if not self.ocr_settings.enable_cache:
            return 0

        cleaned_count = 0
        try:
            # List all cache entries
            prefix = self.ocr_settings.cache_key_prefix
            blobs = self.bucket.list_blobs(prefix=prefix)

            for blob in blobs:
                try:
# # #                     # Check expiration from metadata  # Module not found  # Module not found  # Module not found
                    if blob.metadata and "expires_at" in blob.metadata:
                        expires_at = datetime.fromisoformat(blob.metadata["expires_at"])
                        if datetime.utcnow() > expires_at:
                            blob.delete()
                            cleaned_count += 1
                            logger.debug(f"Cleaned expired cache entry: {blob.name}")
                except Exception as e:
                    logger.warning(f"Error checking cache entry {blob.name}: {e}")
                    continue

            if cleaned_count > 0:
                logger.info(f"Cleaned {cleaned_count} expired cache entries")

        except Exception as e:
            logger.error(f"Error during cache cleanup: {e}")

        return cleaned_count


def needs_ocr(page_text: str, min_text_threshold: int = 50) -> bool:
    """
    Determina si una página necesita OCR basado en heurística de cobertura de texto

    Args:
        page_text: Texto extraído directamente del PDF
        min_text_threshold: Mínimo de caracteres para considerar que no necesita OCR

    Returns:
        True si necesita OCR, False si el texto directo es suficiente
    """
    if not page_text or len(page_text.strip()) < min_text_threshold:
        return True

    # Verificar si el texto tiene caracteres extraños (indicador de mal parsing)
    weird_chars = len(re.findall(r'[^\w\s\-.,;:()\[\]{}"]', page_text))
    total_chars = len(page_text)

    if total_chars > 0 and weird_chars / total_chars > 0.3:
        return True

    return False


def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Preprocesa imagen para mejorar OCR

    Args:
        image: Imagen PIL

    Returns:
        Imagen numpy array preprocesada
    """
    # Convertir a numpy array
    img_array = np.array(image)

    # Convertir a escala de grises si es necesario
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array

    # Deskew (corrección de inclinación)
    coords = np.column_stack(np.where(gray > 0))
    if len(coords) > 0:
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        if abs(angle) > 0.5:  # Solo corregir si hay inclinación significativa
            (h, w) = gray.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            gray = cv2.warpAffine(
                gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
            )

    # Binarización adaptativa
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # Denoising
    denoised = cv2.medianBlur(binary, 3)

    return denoised


def run_ocr(image: Image.Image, lang: str = "spa") -> str:
    """
    Ejecuta OCR en una imagen preprocesada

    Args:
        image: Imagen PIL
        lang: Idioma para Tesseract

    Returns:
        Texto extraído por OCR
    """
    try:
        # Preprocesar imagen
        preprocessed = preprocess_image(image)

        # Configurar Tesseract
        config = f"--oem 3 --psm 1 -l {lang}"

        # Ejecutar OCR
        text = pytesseract.image_to_string(preprocessed, config=config)

        # Limpieza básica
        text = text.strip()
        text = re.sub(r"\n\s*\n\s*\n", "\n\n", text)  # Reducir saltos múltiples

        return text

    except Exception as e:
        print(f"Error en OCR: {e}")
        return ""


def run_cloud_ocr(
    image: Image.Image, provider: str = "google_vision"
) -> Dict[str, Any]:
    """
    Ejecuta OCR usando servicios en la nube

    Args:
        image: Imagen PIL
        provider: Proveedor de OCR (google_vision, azure_ocr, aws_textract)

    Returns:
        Diccionario con texto y metadatos de confianza
    """
    if provider == "google_vision":
        return _run_google_vision_ocr(image)
    elif provider == "azure_ocr":
        return _run_azure_ocr(image)
    elif provider == "aws_textract":
        return _run_aws_textract(image)
    else:
        # Fallback to local OCR
        text = run_ocr(image)
        return {
            "text": text,
            "confidence": get_ocr_confidence(image),
            "provider": "local_tesseract",
        }


def _run_google_vision_ocr(image: Image.Image) -> Dict[str, Any]:
    """Ejecuta Google Vision OCR."""
    try:
# # #         from google.cloud import vision  # Module not found  # Module not found  # Module not found

        client = vision.ImageAnnotatorClient()

        # Convert PIL image to bytes
        import io

        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format="PNG")
        img_byte_arr = img_byte_arr.getvalue()

        vision_image = vision.Image(content=img_byte_arr)

        # Perform OCR
        response = client.text_detection(image=vision_image)
        texts = response.text_annotations

        if texts:
            full_text = texts[0].description
# # #             # Calculate average confidence from bounding polygon data  # Module not found  # Module not found  # Module not found
            confidence = 0.9  # Google Vision typically has high confidence
        else:
            full_text = ""
            confidence = 0.0

        return {
            "text": full_text,
            "confidence": confidence,
            "provider": "google_vision",
            "raw_response": response,
        }

    except Exception as e:
        logger.error(f"Google Vision OCR error: {e}")
        # Fallback to local OCR
        text = run_ocr(image)
        return {
            "text": text,
            "confidence": get_ocr_confidence(image),
            "provider": "local_tesseract_fallback",
        }


def _run_azure_ocr(image: Image.Image) -> Dict[str, Any]:
    """Ejecuta Azure Computer Vision OCR."""
    # Placeholder for Azure OCR implementation
    # In a real implementation, you would use Azure Cognitive Services
    logger.warning("Azure OCR not implemented, falling back to local OCR")
    text = run_ocr(image)
    return {
        "text": text,
        "confidence": get_ocr_confidence(image),
        "provider": "local_tesseract_fallback",
    }


def _run_aws_textract(image: Image.Image) -> Dict[str, Any]:
    """Ejecuta AWS Textract OCR."""
    # Placeholder for AWS Textract implementation
    # In a real implementation, you would use AWS Textract
    logger.warning("AWS Textract not implemented, falling back to local OCR")
    text = run_ocr(image)
    return {
        "text": text,
        "confidence": get_ocr_confidence(image),
        "provider": "local_tesseract_fallback",
    }


class IntelligentOCRProcessor:
    """Main OCR processor with intelligent decision making and caching."""

    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self.decision_system = IntelligentOCRDecisionSystem(settings)
        self.cache = CloudOCRCache(settings)

    def process_document(
        self,
        file_path: str,
        page_images: list[Image.Image],
        additional_context: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Process document with intelligent OCR decision making and caching.

        Args:
            file_path: Path to the document file
            page_images: List of page images
            additional_context: Additional context for caching

        Returns:
            Dictionary with OCR results and metadata
        """
        try:
            # Generate document hash for caching
            document_hash = self.cache.generate_document_hash(
                file_path, additional_context
            )

            # Check cache first
            cached_entry = self.cache.get_cached_results(document_hash)
            if cached_entry:
                logger.info(f"Using cached OCR results for document: {file_path}")
                return {
                    "success": True,
                    "cached": True,
                    "ocr_results": cached_entry.ocr_results,
                    "processing_metadata": cached_entry.processing_metadata,
                    "document_hash": document_hash,
                }

            # Evaluate document characteristics
            characteristics = self.decision_system.evaluate_document(
                file_path, page_images
            )
            logger.info(f"Document characteristics: {characteristics.to_dict()}")

            # Make OCR decision
            decision = self.decision_system.make_ocr_decision(characteristics)
            logger.info(f"OCR decision: {decision.to_dict()}")

            if not decision.should_process:
                return {
                    "success": False,
                    "cached": False,
                    "decision": decision.to_dict(),
                    "characteristics": characteristics.to_dict(),
                    "message": f"OCR processing skipped: {'; '.join(decision.decision_reasons)}",
                }

            # Process with OCR
            ocr_results = {}
            processing_metadata = {
                "characteristics": characteristics.to_dict(),
                "decision": decision.to_dict(),
                "provider": self.settings.ocr.cloud_ocr_provider,
                "processing_time": 0.0,
                "total_pages_processed": 0,
            }

            start_time = datetime.utcnow()

            for i, image in enumerate(page_images):
                page_num = str(i + 1)

                try:
                    if self.settings.ocr.cloud_ocr_provider != "local_tesseract":
                        # Use cloud OCR
                        ocr_result = run_cloud_ocr(
                            image, self.settings.ocr.cloud_ocr_provider
                        )
                        ocr_results[page_num] = ocr_result["text"]

                        # Store additional metadata for first page
                        if i == 0:
                            processing_metadata["ocr_confidence"] = ocr_result.get(
                                "confidence", 0.0
                            )
                            processing_metadata["actual_provider"] = ocr_result.get(
                                "provider", "unknown"
                            )
                    else:
                        # Use local OCR
                        text = run_ocr(image)
                        ocr_results[page_num] = text

                        if i == 0:
                            processing_metadata["ocr_confidence"] = get_ocr_confidence(
                                image
                            )
                            processing_metadata["actual_provider"] = "local_tesseract"

                    processing_metadata["total_pages_processed"] += 1

                except Exception as e:
                    logger.error(f"Error processing page {page_num}: {e}")
                    ocr_results[page_num] = ""

            processing_time = (datetime.utcnow() - start_time).total_seconds() / 60.0
            processing_metadata["processing_time"] = processing_time

            # Store results in cache
            cache_stored = self.cache.store_cache_entry(
                document_hash, ocr_results, processing_metadata
            )

            return {
                "success": True,
                "cached": False,
                "cache_stored": cache_stored,
                "ocr_results": ocr_results,
                "processing_metadata": processing_metadata,
                "document_hash": document_hash,
            }

        except Exception as e:
            logger.error(f"Error in intelligent OCR processing: {e}")
            return {
                "success": False,
                "cached": False,
                "error": str(e),
                "message": f"OCR processing failed: {e}",
            }


def get_ocr_confidence(image: Image.Image, lang: str = "spa") -> float:
    """
    Obtiene score de confianza del OCR

    Args:
        image: Imagen PIL
        lang: Idioma para Tesseract

    Returns:
        Score de confianza (0.0 a 1.0)
    """
    try:
        preprocessed = preprocess_image(image)
        config = f"--oem 3 --psm 1 -l {lang}"

        # Obtener datos detallados de OCR
        data = pytesseract.image_to_data(
            preprocessed, config=config, output_type=pytesseract.Output.DICT
        )

        confidences = [int(conf) for conf in data["conf"] if int(conf) > 0]

        if not confidences:
            return 0.0

        return sum(confidences) / (len(confidences) * 100.0)

    except Exception as e:
        print(f"Error obteniendo confianza OCR: {e}")
        return 0.0

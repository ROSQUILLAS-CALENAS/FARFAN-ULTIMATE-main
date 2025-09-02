"""
Extractor de metadatos para PDTs
"""

import hashlib
import re
from datetime import datetime
from typing import Any, Dict, Optional, Tuple


class MetadataExtractor:
    """Extractor de metadatos de documentos PDT"""

    def __init__(self):
        # Patterns para municipios colombianos (muestra)
        self.municipality_patterns = [
            r"(?i)municipio\s+de\s+([A-ZÁÉÍÓÚÜÑ][a-záéíóúüñ\s]+)",
            r"(?i)alcaldía\s+de\s+([A-ZÁÉÍÓÚÜÑ][a-záéíóúüñ\s]+)",
            r"(?i)([A-ZÁÉÍÓÚÜÑ][a-záéíóúüñ\s]+)\s*[-–—]\s*[A-Za-z\s]*(?:plan|desarrollo)",
        ]

        # Patterns para departamentos
        self.department_patterns = [
            r"(?i)departamento\s+(?:de\s+|del\s+)?([A-ZÁÉÍÓÚÜÑ][a-záéíóúüñ\s]+)",
            r"(?i)gobernación\s+(?:de\s+|del\s+)?([A-ZÁÉÍÓÚÜÑ][a-záéíóúüñ\s]+)",
        ]

        # Patterns para años
        self.year_patterns = [
            r"(?:20[12]\d)\s*[-–—]\s*(20[12]\d)",  # Rango de años
            r"(?:período|periodo)\s*:?\s*(20[12]\d)\s*[-–—]\s*(20[12]\d)",
            r"plan\s+(?:de\s+)?desarrollo\s+(?:territorial\s+)?(20[12]\d)\s*[-–—]\s*(20[12]\d)",
            r"vigencia\s*:?\s*(20[12]\d)\s*[-–—]\s*(20[12]\d)",
        ]

        # Mapeo de departamentos conocidos (ampliable)
        self.known_departments = {
            "antioquia",
            "cundinamarca",
            "valle del cauca",
            "atlántico",
            "bolívar",
            "santander",
            "norte de santander",
            "córdoba",
            "tolima",
            "huila",
            "nariño",
            "cauca",
            "magdalena",
            "cesar",
            "boyacá",
            "risaralda",
            "caldas",
            "quindío",
            "caquetá",
            "casanare",
            "meta",
            "arauca",
            "putumayo",
            "amazonas",
            "guainía",
            "guaviare",
            "vaupés",
            "vichada",
            "la guajira",
            "sucre",
            "chocó",
            "san andrés y providencia",
        }

        # Esquema BigQuery para metadatos
        self.pdt_metadata_schema = {
            "pdt_id": "STRING",
            "municipality": "STRING",
            "department": "STRING",
            "evaluation_date": "DATE",
            "document_pages": "INTEGER",
            "document_size_mb": "FLOAT",
            "year_start": "INTEGER",
            "year_end": "INTEGER",
            "extraction_timestamp": "TIMESTAMP",
            "sha256_hash": "STRING",
            "processing_version": "STRING",
        }

    def infer_metadata(self, text: str, file_path: str = None) -> Dict[str, Any]:
        """
        Infiere metadatos del texto del PDT

        Args:
            text: Texto completo del documento
            file_path: Ruta del archivo (opcional)

        Returns:
            Diccionario con metadatos extraídos
        """
        metadata = {
            "municipality": None,
            "department": None,
            "year_start": None,
            "year_end": None,
            "extraction_timestamp": datetime.utcnow().isoformat(),
            "processing_version": "1.0.0",
        }

        # Extraer municipio
        municipality = self._extract_municipality(text)
        if municipality:
            metadata["municipality"] = municipality

        # Extraer departamento
        department = self._extract_department(text)
        if department:
            metadata["department"] = department

        # Extraer años
        year_start, year_end = self._extract_years(text)
        if year_start:
            metadata["year_start"] = year_start
        if year_end:
            metadata["year_end"] = year_end

        # Calcular hash si hay path
        if file_path:
            metadata["sha256_hash"] = self._calculate_file_hash(file_path)

        return metadata

    def create_bigquery_row(
        self,
        metadata: Dict[str, Any],
        pdt_id: str,
        document_pages: int,
        document_size_mb: float,
    ) -> Dict[str, Any]:
        """
        Crea fila para inserción en BigQuery

        Args:
            metadata: Metadatos extraídos
            pdt_id: ID del documento PDT
            document_pages: Número de páginas
            document_size_mb: Tamaño en MB

        Returns:
            Fila formateada para BigQuery
        """
        row = {
            "pdt_id": pdt_id,
            "municipality": metadata.get("municipality"),
            "department": metadata.get("department"),
            "evaluation_date": datetime.now().date().isoformat(),
            "document_pages": document_pages,
            "document_size_mb": document_size_mb,
            "year_start": metadata.get("year_start"),
            "year_end": metadata.get("year_end"),
            "extraction_timestamp": metadata.get("extraction_timestamp"),
            "sha256_hash": metadata.get("sha256_hash"),
            "processing_version": metadata.get("processing_version", "1.0.0"),
        }

        return row

    def _extract_municipality(self, text: str) -> Optional[str]:
        """Extrae nombre del municipio del texto"""
        # Buscar en las primeras páginas donde suele estar
        text_sample = text[:5000]  # Primeros 5000 caracteres

        for pattern in self.municipality_patterns:
            matches = re.findall(pattern, text_sample)
            if matches:
                municipality = matches[0].strip().title()
                # Limpiar texto común que no es parte del nombre
                municipality = re.sub(r"\s*[-–—]\s*.*$", "", municipality)
                return municipality

        return None

    def _extract_department(self, text: str) -> Optional[str]:
        """Extrae nombre del departamento del texto"""
        text_sample = text[:5000]

        for pattern in self.department_patterns:
            matches = re.findall(pattern, text_sample)
            if matches:
                department = matches[0].strip().lower()
                # Verificar si es un departamento conocido
                if department in self.known_departments:
                    return department.title()
                # Buscar coincidencia parcial
                for known_dept in self.known_departments:
                    if known_dept in department or department in known_dept:
                        return known_dept.title()

        # Buscar departamentos mencionados en el texto
        for dept in self.known_departments:
            dept_pattern = r"(?i)\b" + re.escape(dept) + r"\b"
            if re.search(dept_pattern, text_sample):
                return dept.title()

        return None

    def _extract_years(self, text: str) -> Tuple[Optional[int], Optional[int]]:
        """Extrae años de vigencia del plan"""
        text_sample = text[:3000]

        for pattern in self.year_patterns:
            matches = re.findall(pattern, text_sample)
            if matches:
                if isinstance(matches[0], tuple):
                    year_start, year_end = matches[0]
                else:
                    year_start = year_end = matches[0]

                try:
                    return int(year_start), int(year_end)
                except (ValueError, TypeError):
                    continue

        # Fallback: buscar cualquier año 20XX en título o primeras líneas
        year_matches = re.findall(r"20[12]\d", text_sample[:500])
        if year_matches:
            years = [int(y) for y in year_matches]
            if len(years) >= 2:
                return min(years), max(years)
            elif len(years) == 1:
                # Asumir periodo de 4 años
                year = years[0]
                return year, year + 3

        return None, None

    def _calculate_file_hash(self, file_path: str) -> str:
        """Calcula hash SHA-256 del archivo"""
        try:
            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            return sha256_hash.hexdigest()
        except Exception as e:
            print(f"Error calculando hash: {e}")
            return ""

    def validate_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Valida y completa metadatos extraídos

        Args:
            metadata: Metadatos a validar

        Returns:
            Metadatos validados con flags de calidad
        """
        validated = metadata.copy()
        quality_flags = []

        # Validar municipio
        if not validated.get("municipality"):
            quality_flags.append("missing_municipality")

        # Validar departamento
        if not validated.get("department"):
            quality_flags.append("missing_department")

        # Validar años
        if not validated.get("year_start") or not validated.get("year_end"):
            quality_flags.append("missing_years")
        else:
            # Verificar que los años sean razonables
            year_start = validated["year_start"]
            year_end = validated["year_end"]
            current_year = datetime.now().year

            if year_start < 2000 or year_start > current_year + 10:
                quality_flags.append("invalid_year_start")

            if year_end < year_start or year_end > current_year + 20:
                quality_flags.append("invalid_year_end")

        validated["quality_flags"] = quality_flags
        validated["metadata_completeness"] = 1.0 - (
            len(quality_flags) / 4.0
        )  # 4 validaciones principales

        return validated

"""
Operaciones de I/O con Google Cloud Platform
"""

import io
import datetime
# # # from datetime import datetime  # Module not found  # Module not found  # Module not found
# # # from typing import Any, Dict, List, Optional  # Module not found  # Module not found  # Module not found

import orjson
# # # from google.cloud import bigquery, pubsub_v1, storage  # Module not found  # Module not found  # Module not found



# Mandatory Pipeline Contract Annotations
__phase__ = "O"
__code__ = "105O"
__stage_order__ = 7

class GCPIOManager:
    """Manejador de operaciones I/O con GCP"""

    def __init__(
        self,
        project_id: str,
        storage_bucket: str = "pdt-documents",
        pubsub_topic: str = "pdt-events",
    ):
        self.project_id = project_id
        self.storage_bucket = storage_bucket
        self.pubsub_topic = pubsub_topic

        # Inicializar clientes
        self.storage_client = storage.Client(project=project_id)
        self.publisher = pubsub_v1.PublisherClient()
        self.bigquery_client = bigquery.Client(project=project_id)

        # URIs
        self.topic_path = self.publisher.topic_path(project_id, pubsub_topic)

    def read_from_gcs(self, gcs_uri: str) -> bytes:
        """
        Lee archivo desde Google Cloud Storage

        Args:
            gcs_uri: URI del archivo (gs://bucket/path)

        Returns:
            Contenido del archivo como bytes
        """
        try:
            # Parsear URI
            if not gcs_uri.startswith("gs://"):
                raise ValueError(f"URI inválida: {gcs_uri}")

            parts = gcs_uri[5:].split("/", 1)  # Remover 'gs://'
            bucket_name = parts[0]
            blob_name = parts[1] if len(parts) > 1 else ""

            # Obtener bucket y blob
            bucket = self.storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_name)

            # Leer contenido
            return blob.download_as_bytes()

        except Exception as e:
            raise Exception(f"Error leyendo desde GCS {gcs_uri}: {e}")

    def write_to_gcs(
        self, content: Any, gcs_uri: str, content_type: str = None
    ) -> None:
        """
        Escribe contenido a Google Cloud Storage

        Args:
            content: Contenido a escribir (str, bytes, o dict)
            gcs_uri: URI de destino (gs://bucket/path)
            content_type: Tipo de contenido MIME
        """
        try:
            # Parsear URI
            parts = gcs_uri[5:].split("/", 1)
            bucket_name = parts[0]
            blob_name = parts[1]

            # Obtener bucket y blob
            bucket = self.storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_name)

            # Preparar contenido
            if isinstance(content, dict):
                content_bytes = orjson.dumps(content)
                content_type = content_type or "application/json"
            elif isinstance(content, str):
                content_bytes = content.encode("utf-8")
                content_type = content_type or "text/plain"
            else:
                content_bytes = content

            # Escribir
            blob.upload_from_string(content_bytes, content_type=content_type)

        except Exception as e:
            raise Exception(f"Error escribiendo a GCS {gcs_uri}: {e}")

    def write_jsonl_to_gcs(self, jsonl_content: str, gcs_uri: str) -> None:
        """
        Escribe contenido JSONL a GCS

        Args:
            jsonl_content: Contenido en formato JSONL
            gcs_uri: URI de destino
        """
        self.write_to_gcs(jsonl_content, gcs_uri, "application/x-jsonlines")

    def publish_event(self, topic: str, payload: Dict[str, Any]) -> str:
        """
        Publica evento a Pub/Sub

        Args:
            topic: Nombre del topic (sin ruta completa)
            payload: Payload del evento

        Returns:
            ID del mensaje publicado
        """
        try:
            # Crear ruta completa del topic si es necesario
            if not topic.startswith("projects/"):
                topic_path = self.publisher.topic_path(self.project_id, topic)
            else:
                topic_path = topic

            # Serializar payload
            message_data = orjson.dumps(payload)

            # Publicar
            future = self.publisher.publish(topic_path, message_data)
            message_id = future.result()

            return message_id

        except Exception as e:
            raise Exception(f"Error publicando evento: {e}")

    def insert_bigquery_row(
        self, dataset_id: str, table_id: str, row: Dict[str, Any]
    ) -> None:
        """
        Inserta fila en BigQuery

        Args:
            dataset_id: ID del dataset
            table_id: ID de la tabla
            row: Fila a insertar
        """
        try:
            table_ref = self.bigquery_client.dataset(dataset_id).table(table_id)
            table = self.bigquery_client.get_table(table_ref)

            # Insertar fila
            errors = self.bigquery_client.insert_rows_json(table, [row])

            if errors:
                raise Exception(f"Errores insertando en BigQuery: {errors}")

        except Exception as e:
            raise Exception(f"Error insertando en BigQuery: {e}")

    def batch_insert_bigquery(
        self, dataset_id: str, table_id: str, rows: List[Dict[str, Any]]
    ) -> None:
        """
        Inserta múltiples filas en BigQuery

        Args:
            dataset_id: ID del dataset
            table_id: ID de la tabla
            rows: Lista de filas a insertar
        """
        try:
            table_ref = self.bigquery_client.dataset(dataset_id).table(table_id)
            table = self.bigquery_client.get_table(table_ref)

            # Insertar en lotes de 1000
            batch_size = 1000
            for i in range(0, len(rows), batch_size):
                batch = rows[i : i + batch_size]
                errors = self.bigquery_client.insert_rows_json(table, batch)

                if errors:
                    raise Exception(f"Errores en lote {i//batch_size + 1}: {errors}")

        except Exception as e:
            raise Exception(f"Error insertando lote en BigQuery: {e}")

    def check_file_exists(self, gcs_uri: str) -> bool:
        """
        Verifica si un archivo existe en GCS

        Args:
            gcs_uri: URI del archivo

        Returns:
            True si existe, False si no
        """
        try:
            parts = gcs_uri[5:].split("/", 1)
            bucket_name = parts[0]
            blob_name = parts[1]

            bucket = self.storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_name)

            return blob.exists()

        except Exception:
            return False

    def get_file_metadata(self, gcs_uri: str) -> Dict[str, Any]:
        """
        Obtiene metadatos de un archivo en GCS

        Args:
            gcs_uri: URI del archivo

        Returns:
            Diccionario con metadatos
        """
        try:
            parts = gcs_uri[5:].split("/", 1)
            bucket_name = parts[0]
            blob_name = parts[1]

            bucket = self.storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_name)

            # Recargar para obtener metadatos actualizados
            blob.reload()

            return {
                "name": blob.name,
                "size": blob.size,
                "created": blob.time_created,
                "updated": blob.updated,
                "content_type": blob.content_type,
                "md5_hash": blob.md5_hash,
                "etag": blob.etag,
            }

        except Exception as e:
            raise Exception(f"Error obteniendo metadatos: {e}")

    def create_signed_url(self, gcs_uri: str, expiration_hours: int = 1) -> str:
        """
        Crea URL firmada temporal para acceso a archivo

        Args:
            gcs_uri: URI del archivo
            expiration_hours: Horas de expiración

        Returns:
            URL firmada
        """
        try:
# # #             from datetime import timedelta  # Module not found  # Module not found  # Module not found

            parts = gcs_uri[5:].split("/", 1)
            bucket_name = parts[0]
            blob_name = parts[1]

            bucket = self.storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_name)

            url = blob.generate_signed_url(
                expiration=datetime.now(datetime.UTC)
                + timedelta(hours=expiration_hours),
                method="GET",
            )

            return url

        except Exception as e:
            raise Exception(f"Error creando URL firmada: {e}")


class EventPayloads:
    """Payloads estándar para eventos Pub/Sub"""

    @staticmethod
    def pdt_uploaded(
        gcs_uri: str, file_size: int, upload_timestamp: str = None
    ) -> Dict[str, Any]:
        """Evento de carga de PDT"""
        return {
            "event_type": "pdt.uploaded",
            "gcs_uri": gcs_uri,
            "file_size": file_size,
            "upload_timestamp": upload_timestamp
            or datetime.now(datetime.UTC).isoformat(),
            "processing_requested": True,
        }

    @staticmethod
    def pdt_ingested(
        pdt_id: str,
        package_uri: str,
        metadata_row_id: str,
        processing_time_seconds: float = None,
    ) -> Dict[str, Any]:
        """Evento de ingesta completada"""
        return {
            "event_type": "pdt.ingested",
            "pdt_id": pdt_id,
            "package_uri": package_uri,
            "metadata_row_id": metadata_row_id,
            "ingestion_timestamp": datetime.now(datetime.UTC).isoformat(),
            "processing_time_seconds": processing_time_seconds,
            "ready_for_semantic_indexing": True,
        }

    @staticmethod
    def processing_error(
        gcs_uri: str, error_message: str, error_type: str = "processing_error"
    ) -> Dict[str, Any]:
        """Evento de error de procesamiento"""
        return {
            "event_type": "pdt.processing_error",
            "gcs_uri": gcs_uri,
            "error_type": error_type,
            "error_message": error_message,
            "error_timestamp": datetime.now(datetime.UTC).isoformat(),
            "retry_suggested": error_type in ["temporary_error", "timeout"],
        }

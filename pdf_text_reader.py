#!/usr/bin/env python3
"""
Canonical Flow Component: CF-INGEST-PDF-TEXT
Minimal PDF text reader - text extraction only (no OCR)

Entrypoints
- Python API: process(data: dict | str, context: dict | None = None) -> dict
- Optional CLI wrapper via: python -c "from pdf_text_reader import process; import json; print(json.dumps(process({'pdf_path': 'planes_input/plan.pdf'}), ensure_ascii=False, indent=2))"

Input Contract
- data: dict with key 'pdf_path' or a string path to a PDF
- File must exist and be readable; .pdf extension is recommended but not enforced

Output Artifact (always written)
- canonical_flow/<pdf_stem>.json (UTF-8, ensure_ascii=False, indent=2)
- Schema: { file, text, pages, status: 'success'|'failed', error? }

Failure Semantics
- If PyPDF2 is unavailable or file cannot be read, returns status='failed' with error and writes failure artifact.
- Page extraction errors are tolerated; pages are counted and empty text is contributed for failed pages.
- No unhandled exceptions should escape; always returns a dict with 'status'.

Notes
- This component is idempotent; reruns overwrite the same artifact for the same input path.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import PyPDF2  # type: ignore
except Exception as _imp_err:  # pragma: no cover - environment dependent
    PyPDF2 = None  # type: ignore
    _PDF_IMPORT_ERROR = _imp_err
else:
    _PDF_IMPORT_ERROR = None


def _resolve_pdf_path(data: Any) -> Optional[Path]:
    """Resolve PDF path from dict or string; return absolute Path or None."""
    if isinstance(data, dict):
        candidate = data.get("pdf_path") or data.get("path") or data.get("file")
    else:
        candidate = data
    if not candidate:
        return None
    try:
        return Path(str(candidate)).resolve()
    except Exception:
        return None


def _write_artifact(out_path: Path, payload: Dict[str, Any]) -> Optional[str]:
    """Write JSON payload to out_path in UTF-8 with indentation. Returns error message on failure."""
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return None
    except Exception as e:  # pragma: no cover - filesystem dependent
        return str(e)


def _extract_text_with_pypdf2(pdf_path: Path) -> Dict[str, Any]:
    """Extract text from PDF using PyPDF2; tolerate per-page errors."""
    if PyPDF2 is None:
        return {
            "file": pdf_path.name,
            "text": "",
            "pages": 0,
            "status": "failed",
            "error": f"PyPDF2 not available: {_PDF_IMPORT_ERROR}",
        }

    pages_count = 0
    texts: list[str] = []
    try:
        with pdf_path.open("rb") as f:
            reader = PyPDF2.PdfReader(f)
            # Some PDFs may have len(reader.pages) property
            for page in reader.pages:
                try:
                    raw = page.extract_text()
                except Exception:
                    raw = None
                texts.append(raw or "")
                pages_count += 1
    except Exception as e:
        return {
            "file": pdf_path.name,
            "text": "",
            "pages": pages_count,
            "status": "failed",
            "error": str(e),
        }

    combined = "\n\n".join(t for t in texts if t)
    return {
        "file": pdf_path.name,
        "text": combined,
        "pages": pages_count,
        "status": "success",
    }


def process(data: Any, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Extract plain text from a single PDF and persist canonical artifact.

    Args:
        data: dict with 'pdf_path' or a string path
        context: optional, ignored by this minimal reader

    Returns:
        dict with keys: file, text, pages, status, and optional error
    """
    # Resolve path
    pdf_path = _resolve_pdf_path(data)
    if pdf_path is None:
        payload = {
            "file": "",
            "text": "",
            "pages": 0,
            "status": "failed",
            "error": "No PDF path provided",
        }
        # Write generic failure artifact to canonical_flow/unknown.json for observability
        _write_artifact(Path("canonical_flow") / "unknown.json", payload)
        return payload

    if not pdf_path.exists() or not pdf_path.is_file():
        payload = {
            "file": pdf_path.name,
            "text": "",
            "pages": 0,
            "status": "failed",
            "error": f"File not found: {pdf_path}",
        }
        _write_artifact(Path("canonical_flow") / f"{pdf_path.stem}.json", payload)
        return payload

    # Extract
    result = _extract_text_with_pypdf2(pdf_path)

    # Always persist artifact (success or failure)
    out_path = Path("canonical_flow") / f"{pdf_path.stem}.json"
    write_err = _write_artifact(out_path, result)
    if write_err and result.get("status") == "success":
        # Convert to failed due to write error but return best effort info
        result = {
            **result,
            "status": "failed",
            "error": f"Failed to write output: {write_err}",
        }
    return result


# Common aliases expected by orchestrators
run = execute = main = process

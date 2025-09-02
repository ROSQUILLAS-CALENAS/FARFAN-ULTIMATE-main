# Canonical PDF Flow Dependencies (Authoritative)

This document lists the exact dependencies required to run the canonical PDF analysis flow. It distinguishes the minimal text-only path from the extended PDF reader + OCR path.

If you follow this document precisely, the canonical flow will run without hidden omissions.

---

## 1) Minimal: Canonical Text Reader (CF-INGEST-PDF-TEXT)
- Purpose: Extract plain text from PDFs and write canonical artifacts to `canonical_flow/<name>.json`.
- Module: `pdf_text_reader.py`
- Python package required:
  - PyPDF2 >= 3.0.1

Install options:
- Full project install (recommended):
  ```bash
  pip install -r requirements.txt
  ```
  This includes PyPDF2 (already added to the file).

- Minimal install for this component only:
  ```bash
  pip install PyPDF2>=3.0.1
  ```

Sanity checks:
```bash
python -c "import PyPDF2; print('PyPDF2 OK', PyPDF2.__version__)"
python -c "from pdf_text_reader import process; import json; print(json.dumps(process({'pdf_path': 'planes_input/nonexistent.pdf'}), ensure_ascii=False, indent=2))"
# Expect status="failed" and an error with 'File not found'
```

Run canonical text extraction over a real PDF:
```bash
python -c "from pdf_text_reader import process; import json; print(json.dumps(process({'pdf_path': 'planes_input/your.pdf'}), ensure_ascii=False, indent=2))"
# Expect status="success", pages>=1 (non-empty PDF), and canonical_flow/your.json created
```

---

## 2) Extended: PDF Reader with OCR and Layout (pdf_reader.py path)
Use this if you invoke the orchestrated path that leverages `pdf_reader.PDFPageIterator` and OCR.

Python packages (already declared in requirements.txt):
- PyMuPDF (fitz) >= 1.23.0
- pdfplumber >= 0.9.0
- pillow >= 10.0.0
- pytesseract >= 0.3.10
- easyocr >= 1.7.0
- opencv-python >= 4.8.0
- camelot-py >= 0.11.0

Install:
```bash
pip install -r requirements.txt
```

System packages for OCR (mandatory if OCR is used):
- tesseract-ocr binary + language data (at least `eng`, and `spa` if Spanish PDFs)

Examples per OS:
- Ubuntu/Debian:
  ```bash
  sudo apt-get update
  sudo apt-get install -y tesseract-ocr tesseract-ocr-eng tesseract-ocr-spa
  ```
- macOS (Homebrew):
  ```bash
  brew install tesseract
  # Additional languages can be added via traineddata files if needed
  ```
- Windows (Chocolatey):
  ```powershell
  choco install tesseract
  ```

Optional helpers for table extraction and PDF utilities (already in requirements):
- tabula-py (requires Java), ghostscript/poppler-utils may be needed on some systems depending on features.

Sanity checks for OCR path:
```bash
python -c "import fitz, pdfplumber, PIL, pytesseract, cv2; print('PyMuPDF OK', fitz.__doc__ is not None)"
python -c "import easyocr; print('easyocr OK')"
# Ensure 'tesseract' is available on PATH
which tesseract || where tesseract
```

---

## 3) What the canonical main.py uses
- `--canonical` mode in `main.py` does two things:
  1. Runs the minimal text reader (`pdf_text_reader.process`) to emit canonical artifacts in `canonical_flow/`.
  2. Runs the `ComprehensivePipelineOrchestrator` pipeline (which may use `pdf_reader.py` depending on graph).

Therefore:
- Minimal success path requires only: PyPDF2.
- Full orchestrated path with OCR requires: the extended list above + the tesseract binary.

---

## 4) Quick, reliable install recipes
- Clean venv:
  ```bash
  python -m venv .venv && source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
  python -m pip install --upgrade pip
  pip install -r requirements.txt
  ```
- If you only need the minimal canonical text reader in a fresh environment:
  ```bash
  python -m venv .venv && source .venv/bin/activate
  pip install PyPDF2>=3.0.1
  ```

---

## 5) Frequently asked causes of failure
- PyPDF2 missing → minimal reader fails with status="failed" and writes a failure artifact.
- `tesseract` not installed on OS → OCR path fails; install system package.
- Missing language data for OCR (e.g., Spanish) → degraded OCR results; install `tesseract-ocr-spa`.
- Headless containers without system packages → ensure Dockerfile installs `requirements.txt` (it does) and system packages as needed.

---

This file is maintained as the authoritative reference for canonical PDF flow dependencies. If you find a missing dependency for these paths, update `requirements.txt` and this document in the same commit.
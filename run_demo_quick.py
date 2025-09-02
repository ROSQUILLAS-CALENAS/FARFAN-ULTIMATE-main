#!/usr/bin/env python3
"""
Quick demo runner for PDF Error Handling System
Runs a subset of demos without long-running operations
"""

import logging
import tempfile
from pathlib import Path
from typing import Any, Dict
from datetime import datetime

from pdf_processing_error_handler import (
    PDFValidator,
    ExponentialBackoffRetry,
    CheckpointManager,
    ProcessingState
)


def setup_logging():
    """Setup simple logging"""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def create_minimal_demo_pdfs():
    """Create minimal demo PDF files for testing"""
    demo_dir = Path("demo_pdfs")
    demo_dir.mkdir(exist_ok=True)
    
    # Create valid PDF files
    valid_pdfs = []
    for i in range(2):  # Just 2 files for quick demo
        pdf_path = demo_dir / f"demo_{i:02d}.pdf"
        
        # Minimal PDF content
        pdf_content = f"""%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj  
3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]>>endobj
trailer<</Size 4/Root 1 0 R>>startxref
0
%%EOF""".encode()
        
        with open(pdf_path, 'wb') as f:
            f.write(pdf_content)
        valid_pdfs.append(pdf_path)
    
    # Create one invalid file
    invalid_file = demo_dir / "invalid.pdf"
    with open(invalid_file, 'w') as f:
        f.write("This is not a PDF")
    
    return valid_pdfs, [invalid_file]


def demo_validation_quick():
    """Quick PDF validation demo"""
    print("\n=== Quick PDF Validation Demo ===")
    
    valid_pdfs, invalid_files = create_minimal_demo_pdfs()
    validator = PDFValidator(max_file_size_mb=10)
    
    # Test validation
    for pdf_path in valid_pdfs:
        is_valid, message = validator.validate_pdf(pdf_path)
        status = "✓" if is_valid else "✗"
        print(f"  {status} {pdf_path.name}: {message}")
    
    for invalid_path in invalid_files:
        is_valid, message = validator.validate_pdf(invalid_path)
        status = "✓" if is_valid else "✗"
        print(f"  {status} {invalid_path.name}: {message}")


def demo_retry_quick():
    """Quick retry mechanism demo"""
    print("\n=== Quick Retry Demo ===")
    
    retry_decorator = ExponentialBackoffRetry(
        max_attempts=3,
        base_delay=0.1,  # Fast for demo
        max_delay=1.0
    )
    
    # Test function that succeeds on third attempt
    attempt_count = 0
    
    @retry_decorator
    def test_function():
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count < 3:
            raise ValueError(f"Attempt {attempt_count} failed")
        return f"Success on attempt {attempt_count}"
    
    try:
        result = test_function()
        print(f"  ✓ {result}")
    except Exception as e:
        print(f"  ✗ Final failure: {e}")


def demo_checkpoint_quick():
    """Quick checkpointing demo"""
    print("\n=== Quick Checkpoint Demo ===")
    
    checkpoint_dir = Path("demo_checkpoints")
    checkpoint_manager = CheckpointManager(checkpoint_dir)
    
    # Create test state
    state = ProcessingState(
        batch_id="quick_demo",
        total_documents=5,
        processed_documents=["doc1.pdf", "doc2.pdf"],
        failed_documents=[],
        current_index=2,
        checkpoint_frequency=2,
        start_time=datetime.now()
    )
    
    # Save and load checkpoint
    checkpoint_path = checkpoint_manager.save_checkpoint(state)
    print(f"  ✓ Saved checkpoint: {Path(checkpoint_path).name}")
    
    loaded_state = checkpoint_manager.load_checkpoint(checkpoint_path)
    print(f"  ✓ Loaded checkpoint: {len(loaded_state.processed_documents)} processed docs")


def cleanup_demo():
    """Clean up demo files"""
    import shutil
    
    demo_dirs = ["demo_pdfs", "demo_checkpoints"]
    for dir_name in demo_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            shutil.rmtree(dir_path)
            print(f"  Cleaned: {dir_name}/")


def main():
    """Run quick demo"""
    print("⚡ PDF Error Handling System - Quick Demo")
    print("=" * 45)
    
    setup_logging()
    
    try:
        demo_validation_quick()
        demo_retry_quick()
        demo_checkpoint_quick()
        
        print("\n✅ Quick demo completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        print("\n🧹 Cleaning up...")
        cleanup_demo()
        print("Done!")
    
    return 0


if __name__ == "__main__":
    exit(main())
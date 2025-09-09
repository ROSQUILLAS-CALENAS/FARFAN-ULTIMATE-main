#!/usr/bin/env python3
import json
import os
import sys
import traceback
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import tarfile

# Add project root to path for canonical imports
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Defer ProjectAnalyzer import to runtime to allow packaging/help even if analyzer has issues
ProjectAnalyzer = None


def compile_python_files(project_root: Path, python_files: List[str]) -> Dict[str, Any]:
    import py_compile

    errors = []
    compiled = 0

    # Exclusion patterns
    EXCLUDE_DIRS = {".git", "__pycache__", ".venv", "venv", "env", ".env", "build", "dist", "node_modules"}

    for rel_path in python_files:
        # Normalize and apply exclusions
        rel_p = Path(rel_path)
        if any(part in EXCLUDE_DIRS for part in rel_p.parts):
            continue
        abs_path = project_root / rel_p
        if not abs_path.exists():
            continue
        try:
            py_compile.compile(str(abs_path), doraise=True)
            compiled += 1
        except Exception as ex:
            # Capture syntax and compile-time errors
            errors.append({
                "file": str(rel_p),
                "error": f"{type(ex).__name__}: {ex}"
            })

    return {"success": len(errors) == 0, "compiled_count": compiled, "errors": errors}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _should_exclude(path: Path) -> bool:
    EXCLUDE_DIRS = {".git", "__pycache__", ".venv", "venv", "env", ".env", "build", "dist", "node_modules"}
    return any(part in EXCLUDE_DIRS for part in path.parts)


def _discover_py_files(project_root: Path) -> List[str]:
    files: List[str] = []
    for p in project_root.rglob("*.py"):
        if _should_exclude(p.relative_to(project_root)):
            continue
        files.append(str(p.relative_to(project_root)))
    return files


def _gather_required_files(project_root: Path, analyzer: Any = None) -> List[str]:
    # Base: Python source files
    py_files: List[str] = []
    if analyzer is not None and hasattr(analyzer, "file_types"):
        py_files = analyzer.file_types.get(".py", []) or []
    if not py_files:
        py_files = _discover_py_files(project_root)

    # Add common config/docs
    extras = [
        "requirements.txt", "requirements_minimal.txt", "requirements-minimal.txt",
        "setup.py", "pyproject.toml", "README.md", "README_INNOVATION.md",
    ]
    files = set(py_files)
    for name in extras:
        p = project_root / name
        if p.exists():
            files.add(str(p.relative_to(project_root)))
    return sorted(files)


def _create_tar(project_root: Path, files: List[str], out_path: Path) -> Dict[str, Any]:
    ensure_dir(out_path.parent)
    added: List[str] = []
    with tarfile.open(out_path, "w:gz") as tar:
        for rel in files:
            src = project_root / rel
            if not src.exists():
                continue
            tar.add(str(src), arcname=rel)
            added.append(rel)
        # write manifest as a temp file content via TarInfo
        # build manifest with timezone-aware UTC timestamp
        dt = __import__("datetime")
        created_ts = dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")
        manifest = {
            "created_at": created_ts,
            "root": str(project_root),
            "count": len(added),
            "files": added,
        }
        import io, json as _json
        data = _json.dumps(manifest, indent=2).encode("utf-8")
        ti = tarfile.TarInfo(name="PACKING_MANIFEST.json")
        ti.size = len(data)
        tar.addfile(ti, io.BytesIO(data))
    return {"out": str(out_path), "count": len(added)}


def process_canonical_pdf(pdf_path: Path, verbose: bool = False) -> dict:
    """
    Process a single PDF through the canonical pipeline.
    
    Args:
        pdf_path: Path to the PDF file
        verbose: Enable verbose output
        
    Returns:
        dict: Pipeline execution results
    """
    try:
        # Import the orchestrator
# # #         from comprehensive_pipeline_orchestrator import ComprehensivePipelineOrchestrator  # Module not found  # Module not found  # Module not found
        
        if verbose:
            print(f"[CANONICAL] Initializing ComprehensivePipelineOrchestrator...")
        
        # Initialize orchestrator
        orch = ComprehensivePipelineOrchestrator()
        
        # Prepare input data structure
        # The pipeline expects a dict with the PDF path and metadata
        input_data = {
            "pdf_path": str(pdf_path.absolute()),
            "filename": pdf_path.name,
            "initial_data": str(pdf_path.name),  # For compatibility
            "metadata": {
                "source": "canonical_mode",
                "timestamp": datetime.now().isoformat(),
                "file_size": pdf_path.stat().st_size if pdf_path.exists() else 0
            }
        }
        
        # Optional: Add question context if you want specific analysis
# # #         question_text = f"Analyze the municipal development plan from {pdf_path.name}"  # Module not found  # Module not found  # Module not found
        
        if verbose:
            print(f"[CANONICAL] Processing PDF: {pdf_path}")
            print(f"[CANONICAL] Input data structure: {list(input_data.keys())}")
        
        # Execute the pipeline
        orch_result = orch.execute_pipeline(input_data, question_text=question_text)
        
        if verbose:
            # Extract summary information
            if "monitoring_summary" in orch_result:
                summary = orch_result["monitoring_summary"]
                print(f"[CANONICAL] Pipeline execution completed:")
                print(f"  - Total nodes: {summary.get('total_nodes', 0)}")
                print(f"  - Completed nodes: {summary.get('completed_nodes', 0)}")
                print(f"  - Failed nodes: {summary.get('failed_nodes', 0)}")
                try:
                    print(f"  - Total duration: {summary.get('total_duration_seconds', 0):.2f}s")
                except Exception:
                    print(f"  - Total duration: {summary.get('total_duration_seconds', 0)}s")
            
            if "value_chain" in orch_result:
                try:
                    total_value = sum(v.get("value_added", 0) for v in orch_result["value_chain"].values())
                except Exception:
                    total_value = 0
                print(f"  - Total value added: {total_value:.2f}")
        
        return orch_result
        
    except ImportError as e:
        print(f"[ERROR] Failed to import orchestrator: {e}")
        print("[HINT] Ensure comprehensive_pipeline_orchestrator.py is in the same directory")
        raise
    except Exception as e:
        print(f"[ERROR] Pipeline execution failed: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        raise


def run_canonical_mode(args):
    """
    Execute canonical pipeline mode for PDF processing.
    """
    print("\n" + "="*80)
    print("CANONICAL PIPELINE MODE - PDF Document Analysis")
    print("="*80 + "\n")
    
    # Define input directory
    input_dir = Path("planes_input")
    
    if not input_dir.exists():
        print(f"[ERROR] Input directory not found: {input_dir}")
        print("[HINT] Create 'planes_input/' directory and add PDF files")
        return
    
    # Get all PDF files
    pdf_files = list(input_dir.glob("*.pdf"))
    
    if not pdf_files:
        print(f"[WARNING] No PDF files found in {input_dir}")
        print("[HINT] Add PDF files to process")
        return
    
    print(f"Found {len(pdf_files)} PDF files to process")
    
    # Process limit for testing
    max_files = args.max_files if hasattr(args, 'max_files') else 3
    if len(pdf_files) > max_files and not args.all:
        print(f"[INFO] Processing first {max_files} files (use --all to process all)")
        pdf_files = pdf_files[:max_files]
    
    # Initialize results storage
    results = []
    success_count = 0
    failure_count = 0
    
    # Process each PDF
    for idx, pdf_path in enumerate(pdf_files, 1):
        print(f"\n[{idx}/{len(pdf_files)}] Processing: {pdf_path.name}")
        print("-" * 60)

        # Minimal text-only extraction to canonical_flow via pdf_text_reader (CF-INGEST-PDF-TEXT)
        try:
# # #             from pdf_text_reader import process as pdf_text_process  # Module not found  # Module not found  # Module not found
            txt_res = pdf_text_process({"pdf_path": str(pdf_path)})
            if args.verbose:
                print(f"[CANONICAL][TEXT] {pdf_path.name}: {txt_res.get('status')}")
        except Exception as e:
            print(f"[WARNING] pdf_text_reader failed for {pdf_path.name}: {e}")
        
        try:
            result = process_canonical_pdf(pdf_path, verbose=args.verbose)
            
            # Store result
            results.append({
                "file": pdf_path.name,
                "status": "success",
                "summary": {
                    "total_value_added": result.get("total_value_added", 0),
                    "execution_time": result.get("monitoring_summary", {}).get("total_duration_seconds", 0),
                    "completed_nodes": result.get("monitoring_summary", {}).get("completed_nodes", 0),
                    "failed_nodes": result.get("monitoring_summary", {}).get("failed_nodes", 0)
                }
            })
            success_count += 1
            
            # Save individual result
            output_dir = Path("canonical_flow")
            output_dir.mkdir(exist_ok=True)
            
            output_file = output_dir / f"{pdf_path.stem}_result.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False, default=str)
            
            print(f"[SUCCESS] Result saved to: {output_file}")
            
        except Exception as e:
            print(f"[FAILED] Error processing {pdf_path.name}: {e}")
            results.append({
                "file": pdf_path.name,
                "status": "failed",
                "error": str(e)
            })
            failure_count += 1
    
    # Print summary
    print("\n" + "="*80)
    print("CANONICAL PIPELINE EXECUTION SUMMARY")
    print("="*80)
    print(f"Total files processed: {len(results)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {failure_count}")
    
    if success_count > 0:
        avg_time = sum(r["summary"]["execution_time"] for r in results if r["status"] == "success") / success_count
        print(f"Average execution time: {avg_time:.2f}s")
    
    # Save summary report
    output_dir = Path("canonical_flow")
    output_dir.mkdir(exist_ok=True)
    summary_file = output_dir / "execution_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "total_processed": len(results),
            "success_count": success_count,
            "failure_count": failure_count,
            "results": results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\nSummary report saved to: {summary_file}")
    
    # Generate excellence analysis report if requested
    if getattr(args, 'excellence', False):
        print("\n[EXCELLENCE] Generating pipeline excellence analysis...")
        try:
# # #             from comprehensive_pipeline_orchestrator import generate_excellence_analysis_report  # Module not found  # Module not found  # Module not found
            excellence_report = generate_excellence_analysis_report()
            
            excellence_file = output_dir / "excellence_analysis.json"
            with open(excellence_file, 'w', encoding='utf-8') as f:
                json.dump(excellence_report, f, indent=2, ensure_ascii=False)
            
            print(f"[EXCELLENCE] Report saved to: {excellence_file}")
        except Exception as e:
            print(f"[EXCELLENCE] Failed to generate report: {e}")


def add_canonical_arguments(parser):
    """Add canonical mode specific arguments."""
    parser.add_argument(
        "--canonical",
        action="store_true",
        help="Run canonical pipeline mode for PDF processing"
    )
    parser.add_argument(
        "--recover",
        action="store_true",
        help="Run document recovery for failed processing attempts"
    )
    parser.add_argument(
        "--redis-url",
        default="redis://localhost:6379",
        help="Redis URL for recovery system"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all PDFs (default: first 3 for testing)"
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=3,
        help="Maximum number of files to process (default: 3)"
    )
    parser.add_argument(
        "--excellence",
        action="store_true",
        help="Generate excellence analysis report"
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Project compilation and analysis orchestrator")
    parser.add_argument("project_path", nargs="?", default=".", help="Path to project root (default: .)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--no-compile", action="store_true", help="Skip Python byte-compilation step")
    parser.add_argument("--pack", metavar="OUT_TAR", help="Create a tar.gz with required files at the given path")
    parser.add_argument("--pack-only", action="store_true", help="Only create the package; skip analysis/audit/compile")
    add_canonical_arguments(parser)

    args = parser.parse_args()

    project_root = Path(args.project_path).resolve()
    if not project_root.exists() or not project_root.is_dir():
        print(f"Invalid project path: {project_root}")
        sys.exit(1)

    # Prepare canonical output directory
    canonical_dir = project_root / "canonical_flow"
    ensure_dir(canonical_dir)

    # Recovery mode dispatcher
    if getattr(args, "recover", False):
        try:
            import asyncio
# # #             from recovery_system import run_document_recovery  # Module not found  # Module not found  # Module not found
            
            if args.verbose:
                print("[RECOVERY] Starting document recovery...")
            
            config = {
                'max_retry_count': 3,
                'min_retry_interval_hours': 0.5,
                'recovery_batch_size': 10,
                'enable_periodic_recovery': False
            }
            
            result = asyncio.run(run_document_recovery(redis_url=args.redis_url, config=config))
            
            print(f"✓ Recovery completed:")
            print(f"  Attempted documents: {result.get('attempted_documents', 0)}")
            print(f"  Successful recoveries: {result.get('successful_recoveries', 0)}")
            print(f"  Failed recoveries: {result.get('failed_recoveries', 0)}")
            print(f"  Success rate: {result.get('success_rate', 0.0):.1%}")
            print(f"  Recovery time: {result.get('recovery_time', 0.0):.2f}s")
            
            sys.exit(0)
            
        except Exception as e:
            print(f"✗ Recovery failed: {e}")
            if args.verbose:
                traceback.print_exc()
            sys.exit(1)

    # Canonical processing mode dispatcher
    if getattr(args, "canonical", False):
        # First try the enhanced processing with error handling
        try:
# # #             from pdf_reader import process_pdf_files_with_error_handling  # Module not found  # Module not found  # Module not found
# # #             from comprehensive_pipeline_orchestrator import ComprehensivePipelineOrchestrator  # Module not found  # Module not found  # Module not found
            
            input_dir = project_root / "planes_input"
            if not input_dir.exists():
                ensure_dir(input_dir)
                if args.verbose:
                    print(f"Directorio de entrada creado (vacío): {input_dir}")

            pdf_files = sorted(input_dir.glob("*.pdf"))
            if not pdf_files:
                print(f"No se encontraron archivos .pdf en {input_dir}")
                # Fall back to canonical mode
                run_canonical_mode(args)
                sys.exit(0)

            if args.verbose:
                print(f"Processing {len(pdf_files)} PDFs with enhanced error handling...")
            
            # Process batch with error handling
            batch_results = process_pdf_files_with_error_handling(
                file_paths=[str(p) for p in pdf_files],
                checkpoint_frequency=5,
                max_retry_attempts=3,
                enable_intelligent_ocr=True
            )
            
            results_index = []
            for result_item in batch_results.get("results", []):
                pdf_path = Path(result_item["file"])
                
                if result_item["status"] == "success":
                    try:
                        orch = ComprehensivePipelineOrchestrator()
                        result = orch.execute_pipeline(str(pdf_path))
                        
                        out_file = canonical_dir / f"{pdf_path.stem}.json"
                        with open(out_file, "w", encoding="utf-8") as f:
                            json.dump(result, f, indent=2)
                        
                        results_index.append({
                            "pdf": str(pdf_path.relative_to(project_root)),
                            "output": str(out_file.relative_to(project_root)),
                            "status": "ok",
                            "processing_info": result_item["result"]
                        })
                        
                        if args.verbose:
                            print(f"✓ Processed: {pdf_path.name}")
                    except Exception as e:
                        results_index.append({
                            "pdf": str(pdf_path.relative_to(project_root)),
                            "status": "orchestrator_error",
                            "error": str(e)
                        })
                        if args.verbose:
                            print(f"✗ Orchestrator failed for {pdf_path.name}: {e}")
                else:
                    results_index.append({
                        "pdf": str(pdf_path.relative_to(project_root)),
                        "status": "pdf_processing_error",
                        "error": result_item.get("error", "Unknown error")
                    })
                    if args.verbose:
                        print(f"✗ PDF processing failed for {pdf_path.name}")
            
            summary = {
                "batch_id": batch_results.get("batch_id"),
                "total_files": batch_results.get("total_files", 0),
                "successful_pdf_processing": batch_results.get("successful_files", 0),
                "failed_pdf_processing": batch_results.get("failed_files", 0),
                "processing_time_seconds": batch_results.get("processing_time_seconds", 0),
                "failed_documents": batch_results.get("failed_documents", [])
            }
            
            # Save results index
            try:
                index_path = canonical_dir / "canonical_index.json"
                index_data = {
                    "processed": results_index,
                    "batch_summary": summary,
                    "processing_timestamp": datetime.now().isoformat()
                }
                with open(index_path, "w", encoding="utf-8") as f:
                    json.dump(index_data, f, indent=2)
                if args.verbose:
                    print(f"Índice de resultados guardado en {index_path}")
            except Exception as e:
                print(f"No se pudo escribir el índice de resultados: {e}")
                
        except Exception as e:
            # Fall back to original canonical mode if enhanced processing fails
            print(f"Enhanced processing failed: {e}")
            if args.verbose:
                print("Falling back to original canonical mode...")
            run_canonical_mode(args)
        
        sys.exit(0)

    # Lazy import of analyzer
    global ProjectAnalyzer
    if ProjectAnalyzer is None:
        try:
# # #             from egw_query_expansion.core.project_analyzer import ProjectAnalyzer as _PA  # Module not found  # Module not found  # Module not found
            ProjectAnalyzer = _PA
        except Exception as e:
            if args.verbose:
                print(f"Warning: could not import ProjectAnalyzer: {e}")

    # If requested, allow packaging without analysis
    if args.pack and args.pack_only:
        files_for_pack = _gather_required_files(project_root)
        pkg_res = _create_tar(project_root, files_for_pack, Path(args.pack).resolve())
        print(f"Package created: {pkg_res['out']} ({pkg_res['count']} files)")
        sys.exit(0)

    # Run analyzer if available
    analyzer = None
    if ProjectAnalyzer is not None:
        try:
            analyzer = ProjectAnalyzer(str(project_root))
            analyzer.analyze()
        except Exception as e:
            print(f"Analyzer failed: {e}")
            if args.verbose:
                traceback.print_exc()
            # Continue with best-effort
    else:
        if args.verbose:
            print("ProjectAnalyzer unavailable; proceeding without detailed analysis.")

    # Readiness report
    try:
        readiness = analyzer.check_compilation_readiness()
    except Exception as e:
        readiness = {"error": f"failed_to_compute_readiness: {e}"}

    readiness_path = canonical_dir / "readiness.json"
    try:
        with open(readiness_path, "w") as f:
            json.dump(readiness, f, indent=2)
        if args.verbose:
            print(f"Saved readiness report to {readiness_path}")
    except Exception as e:
        print(f"Failed to write readiness report: {e}")

    # Export a copy of analyzer report into canonical_folder if exists in root
    try:
        root_report = project_root / "project_analysis_report.json"
        if root_report.exists():
            # Load and re-save into canonical_flow
            with open(root_report, "r") as rf:
                data = json.load(rf)
            with open(canonical_dir / "project_analysis_report.json", "w") as wf:
                json.dump(data, wf, indent=2)
        else:
# # #             # Generate a minimal analysis dump from analyzer state  # Module not found  # Module not found  # Module not found
            snapshot = {
                "project_root": str(analyzer.project_root),
                "summary": {
                    "total_files": sum(len(files) for files in analyzer.file_types.values()),
                    "total_modules": len(analyzer.modules),
                },
                "readiness": readiness,
                "errors": getattr(analyzer, "errors", []),
            }
            with open(canonical_dir / "project_analysis_report.json", "w") as wf:
                json.dump(snapshot, wf, indent=2)
    except Exception as e:
        print(f"Failed to export analysis report to canonical_flow: {e}")

    # Dependency audit (non-fatal)
    try:
        try:
# # #             from tools.dependency_audit import run as run_dep_audit  # Module not found  # Module not found  # Module not found
            dep_path = run_dep_audit(str(project_root))
            if args.verbose:
                print(f"Saved dependency audit to {dep_path}")
        except Exception:
            # Fallback: execute as a script if import fails
            import subprocess
            subprocess.run([sys.executable, str(project_root / 'tools' / 'dependency_audit.py'), str(project_root)], check=False)
    except Exception as e:
        print(f"Dependency audit failed: {e}")

    # Compile Python files
    compilation_result = {"skipped": True}
    exit_code = 0
    if not args.no_compile:
        try:
            if analyzer is not None and hasattr(analyzer, "file_types"):
                py_files = analyzer.file_types.get(".py", []) or []
            else:
                py_files = _discover_py_files(project_root)
            result = compile_python_files(project_root, py_files)
            compilation_result = result
            with open(canonical_dir / "compilation_report.json", "w") as f:
                json.dump(result, f, indent=2)
            if not result.get("success", False):
                print("Compilation errors detected. See canonical_flow/compilation_report.json")
                exit_code = 2
            else:
                print("Python byte-compilation successful.")
        except Exception as e:
            print(f"Compilation step failed: {e}")
            if args.verbose:
                traceback.print_exc()
            exit_code = max(exit_code, 2)

    # Final status considering readiness
    try:
        if isinstance(readiness, dict) and not readiness.get("ready", True):
            # Missing entry point or deps is a readiness concern, but not hard failure for syntax compile
            print("Project not fully ready per analyzer readiness check.")
            # Use a non-zero but distinct code to indicate readiness issues
            exit_code = max(exit_code, 3)
    except Exception:
        pass

    # Optional packaging after analysis
    try:
        if args.pack and not args.pack_only:
            files_for_pack = _gather_required_files(project_root, analyzer)
            # Include canonical artifacts if present
            cf_artifacts = [
                "canonical_flow/readiness.json",
                "canonical_flow/compilation_report.json",
                "canonical_flow/project_analysis_report.json",
                "canonical_flow/dependency_audit.json",
            ]
            for rel in cf_artifacts:
                if (project_root / rel).exists():
                    files_for_pack.append(rel)
            # Deduplicate and sort
            files_for_pack = sorted(set(files_for_pack))
            pkg_res = _create_tar(project_root, files_for_pack, Path(args.pack).resolve())
            print(f"Package created: {pkg_res['out']} ({pkg_res['count']} files)")
    except Exception as e:
        print(f"Packaging step failed: {e}")
        if args.verbose:
            traceback.print_exc()
        # keep exit_code as is

    # Ensure non-zero exit on hard failures
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

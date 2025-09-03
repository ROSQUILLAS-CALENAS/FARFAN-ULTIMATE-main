#!/usr/bin/env python3
"""
Comprehensive Repository Scanner for Canonical Pipeline Components
Creates INVENTORY.jsonl catalog of all discovered pipeline components
"""

import os
import re
import json
import ast
import hashlib
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ComponentRecord:
    """Structure for pipeline component inventory record"""
    file_path: str
    phase_assignment: str
    confidence_score: float
    evidence_patterns: List[str]
    status_classification: str
    discovery_metadata: Dict[str, Any]
    
    def to_jsonl(self) -> str:
        """Convert to JSONL format"""
        return json.dumps(asdict(self), ensure_ascii=False)

class CanonicalPipelineScanner:
    """Comprehensive scanner for canonical pipeline components"""
    
    def __init__(self, root_path: str = "."):
        self.root_path = Path(root_path)
        self.components: List[ComponentRecord] = []
        
        # Phase mapping patterns
        self.phase_patterns = {
            "A_analysis_nlp": "Analysis & NLP Processing",
            "G_aggregation_reporting": "Aggregation & Reporting", 
            "I_ingestion_preparation": "Data Ingestion & Preparation",
            "K_knowledge_extraction": "Knowledge Extraction",
            "L_classification_evaluation": "Classification & Evaluation",
            "O_orchestration_control": "Orchestration & Control",
            "R_search_retrieval": "Search & Retrieval",
            "S_synthesis_output": "Synthesis & Output",
            "T_integration_storage": "Integration & Storage",
            "X_context_construction": "Context Construction"
        }
        
        # Pipeline component patterns
        self.component_patterns = {
            "canonical_process_function": re.compile(r"def\s+process\s*\(.*?\):", re.MULTILINE | re.DOTALL),
            "pipeline_class": re.compile(r"class\s+\w*Pipeline\w*", re.MULTILINE),
            "orchestrator_class": re.compile(r"class\s+\w*Orchestrator\w*", re.MULTILINE),
            "processor_class": re.compile(r"class\s+\w*Processor\w*", re.MULTILINE),
            "engine_class": re.compile(r"class\s+\w*Engine\w*", re.MULTILINE),
            "handler_class": re.compile(r"class\s+\w*Handler\w*", re.MULTILINE),
            "adapter_class": re.compile(r"class\s+\w*Adapter\w*", re.MULTILINE),
            "validator_class": re.compile(r"class\s+\w*Validator\w*", re.MULTILINE),
            "manager_class": re.compile(r"class\s+\w*Manager\w*", re.MULTILINE),
            "controller_class": re.compile(r"class\s+\w*Controller\w*", re.MULTILINE),
            "analyzer_class": re.compile(r"class\s+\w*Analyzer\w*", re.MULTILINE),
            "builder_class": re.compile(r"class\s+\w*Builder\w*", re.MULTILINE),
            "extractor_class": re.compile(r"class\s+\w*Extractor\w*", re.MULTILINE),
            "generator_class": re.compile(r"class\s+\w*Generator\w*", re.MULTILINE),
            "router_class": re.compile(r"class\s+\w*Router\w*", re.MULTILINE),
            "aggregator_class": re.compile(r"class\s+\w*Aggregator\w*", re.MULTILINE),
            "collector_class": re.compile(r"class\s+\w*Collector\w*", re.MULTILINE),
            "synthesizer_class": re.compile(r"class\s+\w*Synthesizer\w*", re.MULTILINE),
            "formatter_class": re.compile(r"class\s+\w*Formatter\w*", re.MULTILINE),
            "enhancer_class": re.compile(r"class\s+\w*Enhancer\w*", re.MULTILINE),
            "mathematical_class": re.compile(r"class\s+\w*(Mathematical|Math)\w*", re.MULTILINE),
            "pipeline_import": re.compile(r"from\s+.*pipeline.*import|import\s+.*pipeline", re.MULTILINE),
            "canonical_import": re.compile(r"from\s+canonical_flow|import\s+canonical_flow", re.MULTILINE),
            "process_data_pattern": re.compile(r"process_data|process_document|process_batch", re.MULTILINE),
            "integration_marker": re.compile(r"@pipeline_component|@canonical_component|pipeline.*decorator", re.MULTILINE)
        }
        
        # File pattern exclusions
        self.exclude_patterns = {
            ".git", "__pycache__", ".pytest_cache", "node_modules", 
            ".vscode", ".idea", "venv", ".env", "logs", ".DS_Store",
            "*.pyc", "*.pyo", "*.pyd", "*.so", "*.egg-info"
        }

    def should_exclude_path(self, path: Path) -> bool:
        """Check if path should be excluded from scanning"""
        path_str = str(path)
        for pattern in self.exclude_patterns:
            if pattern in path_str or path.name.startswith('.'):
                return True
        return False

    def extract_file_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract metadata from file"""
        stat = file_path.stat()
        return {
            "size_bytes": stat.st_size,
            "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "file_extension": file_path.suffix,
            "relative_path": str(file_path.relative_to(self.root_path)),
            "directory_depth": len(file_path.relative_to(self.root_path).parts) - 1,
            "parent_directory": file_path.parent.name
        }

    def analyze_python_content(self, content: str) -> Dict[str, Any]:
        """Analyze Python file content for pipeline patterns"""
        analysis = {
            "has_ast_parse_error": False,
            "function_count": 0,
            "class_count": 0,
            "import_count": 0,
            "docstring_present": False,
            "functions": [],
            "classes": [],
            "imports": []
        }
        
        try:
            tree = ast.parse(content)
            
            # Extract functions and classes
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    analysis["functions"].append(node.name)
                    analysis["function_count"] += 1
                elif isinstance(node, ast.ClassDef):
                    analysis["classes"].append(node.name)
                    analysis["class_count"] += 1
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            analysis["imports"].append(alias.name)
                    else:
                        module = node.module or ""
                        analysis["imports"].append(module)
                    analysis["import_count"] += 1
            
            # Check for docstrings
            if (tree.body and isinstance(tree.body[0], ast.Expr) 
                and isinstance(tree.body[0].value, ast.Str)):
                analysis["docstring_present"] = True
                
        except SyntaxError:
            analysis["has_ast_parse_error"] = True
            
        return analysis

    def detect_evidence_patterns(self, content: str, file_path: Path) -> List[str]:
        """Detect evidence patterns in file content"""
        evidence = []
        
        # Check each pattern
        for pattern_name, pattern_regex in self.component_patterns.items():
            if pattern_regex.search(content):
                evidence.append(pattern_name)
        
        # Additional content-based evidence
        content_lower = content.lower()
        
        if "pipeline" in content_lower:
            evidence.append("pipeline_reference")
        if "orchestrat" in content_lower:
            evidence.append("orchestration_reference") 
        if "process" in content_lower and ("data" in content_lower or "document" in content_lower):
            evidence.append("data_processing_reference")
        if "canonical" in content_lower:
            evidence.append("canonical_reference")
        if "stage" in content_lower and ("next" in content_lower or "previous" in content_lower):
            evidence.append("stage_flow_reference")
        if re.search(r"def\s+__call__", content):
            evidence.append("callable_interface")
        if re.search(r"async\s+def", content):
            evidence.append("async_processing")
            
        return list(set(evidence))  # Remove duplicates

    def determine_phase_assignment(self, file_path: Path) -> Tuple[str, float]:
        """Determine phase assignment and confidence score"""
        path_str = str(file_path.relative_to(self.root_path))
        
        # Check canonical_flow directory structure
        for phase_prefix, phase_name in self.phase_patterns.items():
            if phase_prefix in path_str:
                return phase_name, 0.95
        
        # Check by file location patterns
        path_parts = file_path.parts
        
        if "analysis_nlp" in path_parts:
            return "Analysis & NLP Processing", 0.8
        if "aggregation" in path_parts or "reporting" in path_parts:
            return "Aggregation & Reporting", 0.8
        if "ingestion" in path_parts or "preparation" in path_parts:
            return "Data Ingestion & Preparation", 0.8
        if "knowledge" in path_parts or "extraction" in path_parts:
            return "Knowledge Extraction", 0.8
        if "classification" in path_parts or "evaluation" in path_parts:
            return "Classification & Evaluation", 0.8
        if "orchestration" in path_parts or "control" in path_parts:
            return "Orchestration & Control", 0.8
        if "search" in path_parts or "retrieval" in path_parts:
            return "Search & Retrieval", 0.8
        if "synthesis" in path_parts or "output" in path_parts:
            return "Synthesis & Output", 0.8
        if "integration" in path_parts or "storage" in path_parts:
            return "Integration & Storage", 0.8
        if "context" in path_parts:
            return "Context Construction", 0.8
        if "mathematical" in path_parts or "enhancer" in path_parts:
            return "Mathematical Enhancement", 0.75
        
        # Check by filename patterns
        filename_lower = file_path.name.lower()
        
        if any(term in filename_lower for term in ["orchestrat", "control", "manage"]):
            return "Orchestration & Control", 0.7
        if any(term in filename_lower for term in ["retriev", "search", "index"]):
            return "Search & Retrieval", 0.7
        if any(term in filename_lower for term in ["process", "extract", "analyz"]):
            return "Analysis & NLP Processing", 0.65
        if any(term in filename_lower for term in ["aggregat", "report", "compil"]):
            return "Aggregation & Reporting", 0.65
        if any(term in filename_lower for term in ["ingest", "load", "prepar"]):
            return "Data Ingestion & Preparation", 0.65
        if any(term in filename_lower for term in ["classif", "evaluat", "scor"]):
            return "Classification & Evaluation", 0.65
        if any(term in filename_lower for term in ["synthes", "format", "output"]):
            return "Synthesis & Output", 0.65
        if any(term in filename_lower for term in ["integrat", "storag", "persist"]):
            return "Integration & Storage", 0.65
        if any(term in filename_lower for term in ["context", "construct"]):
            return "Context Construction", 0.65
        
        return "Unclassified", 0.3

    def classify_status(self, evidence_patterns: List[str], confidence_score: float) -> str:
        """Classify component status based on evidence and confidence"""
        if confidence_score >= 0.9:
            return "canonical_confirmed"
        elif confidence_score >= 0.75:
            if "canonical_process_function" in evidence_patterns:
                return "canonical_compliant"
            else:
                return "canonical_candidate"
        elif confidence_score >= 0.6:
            if any(pattern in evidence_patterns for pattern in ["pipeline_class", "orchestrator_class", "processor_class"]):
                return "pipeline_component"
            else:
                return "potential_component"
        elif confidence_score >= 0.4:
            return "utility_component"
        else:
            return "unclassified"

    def scan_file(self, file_path: Path) -> Optional[ComponentRecord]:
        """Scan individual file for pipeline component patterns"""
        if self.should_exclude_path(file_path):
            return None
            
        if not file_path.is_file():
            return None
            
        # Only scan Python files for now
        if file_path.suffix != '.py':
            return None
            
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            logger.warning(f"Could not read {file_path}: {e}")
            return None
            
        # Extract evidence patterns
        evidence_patterns = self.detect_evidence_patterns(content, file_path)
        
        # Only include files that have some pipeline-related evidence
        if not evidence_patterns:
            return None
            
        # Determine phase assignment and confidence
        phase_assignment, confidence_score = self.determine_phase_assignment(file_path)
        
        # Classify status
        status_classification = self.classify_status(evidence_patterns, confidence_score)
        
        # Extract metadata
        file_metadata = self.extract_file_metadata(file_path)
        python_analysis = self.analyze_python_content(content) if file_path.suffix == '.py' else {}
        
        discovery_metadata = {
            **file_metadata,
            **python_analysis,
            "scan_timestamp": datetime.utcnow().isoformat(),
            "content_hash": hashlib.sha256(content.encode()).hexdigest()[:16],
            "evidence_count": len(evidence_patterns)
        }
        
        return ComponentRecord(
            file_path=str(file_path.relative_to(self.root_path)),
            phase_assignment=phase_assignment,
            confidence_score=confidence_score,
            evidence_patterns=evidence_patterns,
            status_classification=status_classification,
            discovery_metadata=discovery_metadata
        )

    def scan_repository(self) -> List[ComponentRecord]:
        """Scan entire repository for pipeline components"""
        logger.info(f"Scanning repository at {self.root_path}")
        
        # Get all Python files, sorted deterministically
        python_files = sorted(
            self.root_path.rglob("*.py"),
            key=lambda p: str(p.relative_to(self.root_path))
        )
        
        logger.info(f"Found {len(python_files)} Python files to analyze")
        
        components = []
        for file_path in python_files:
            if component_record := self.scan_file(file_path):
                components.append(component_record)
                
        logger.info(f"Identified {len(components)} pipeline components")
        
        # Sort components by confidence score (descending) then by path
        components.sort(key=lambda c: (-c.confidence_score, c.file_path))
        
        self.components = components
        return components

    def generate_inventory_jsonl(self, output_path: str = "INVENTORY.jsonl") -> None:
        """Generate INVENTORY.jsonl file"""
        if not self.components:
            self.scan_repository()
            
        output_file = Path(output_path)
        logger.info(f"Writing inventory to {output_file}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for component in self.components:
                f.write(component.to_jsonl() + '\n')
                
        logger.info(f"Inventory complete: {len(self.components)} components catalogued")

    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate summary statistics"""
        if not self.components:
            return {}
            
        # Phase distribution
        phase_counts = {}
        status_counts = {}
        confidence_distribution = {"high": 0, "medium": 0, "low": 0}
        evidence_patterns = {}
        
        for component in self.components:
            phase_counts[component.phase_assignment] = phase_counts.get(component.phase_assignment, 0) + 1
            status_counts[component.status_classification] = status_counts.get(component.status_classification, 0) + 1
            
            if component.confidence_score >= 0.8:
                confidence_distribution["high"] += 1
            elif component.confidence_score >= 0.6:
                confidence_distribution["medium"] += 1
            else:
                confidence_distribution["low"] += 1
                
            # Track evidence patterns
            for pattern in component.evidence_patterns:
                evidence_patterns[pattern] = evidence_patterns.get(pattern, 0) + 1
        
        # Top evidence patterns
        top_evidence_patterns = dict(sorted(evidence_patterns.items(), key=lambda x: x[1], reverse=True)[:10])
                
        return {
            "total_components": len(self.components),
            "phase_distribution": dict(sorted(phase_counts.items())),
            "status_distribution": dict(sorted(status_counts.items())),
            "confidence_distribution": confidence_distribution,
            "canonical_flow_components": len([c for c in self.components if "canonical_flow" in c.file_path]),
            "external_components": len([c for c in self.components if "canonical_flow" not in c.file_path]),
            "high_confidence_canonical": len([c for c in self.components if c.confidence_score >= 0.9 and c.status_classification == "canonical_confirmed"]),
            "top_evidence_patterns": top_evidence_patterns,
            "canonical_process_function_count": len([c for c in self.components if "canonical_process_function" in c.evidence_patterns]),
            "pipeline_classes_count": len([c for c in self.components if any(p in c.evidence_patterns for p in ["pipeline_class", "orchestrator_class", "processor_class"])]),
            "mathematical_enhancers_count": len([c for c in self.components if "mathematical_enhancers" in c.file_path])
        }

def main():
    """Main execution function"""
    scanner = CanonicalPipelineScanner()
    
    # Scan repository
    components = scanner.scan_repository()
    
    # Generate inventory
    scanner.generate_inventory_jsonl()
    
    # Generate summary
    summary = scanner.generate_summary_report()
    
    # Print summary
    print("\n=== CANONICAL PIPELINE COMPONENT INVENTORY SUMMARY ===")
    print(f"Total Components Discovered: {summary['total_components']}")
    print(f"Canonical Flow Components: {summary['canonical_flow_components']}")
    print(f"External Components: {summary['external_components']}")
    print(f"High Confidence Canonical: {summary['high_confidence_canonical']}")
    print(f"Canonical Process Functions: {summary['canonical_process_function_count']}")
    print(f"Pipeline Classes: {summary['pipeline_classes_count']}")
    print(f"Mathematical Enhancers: {summary['mathematical_enhancers_count']}")
    
    print("\nPhase Distribution:")
    for phase, count in summary['phase_distribution'].items():
        print(f"  {phase}: {count}")
    
    print("\nStatus Distribution:")
    for status, count in summary['status_distribution'].items():
        print(f"  {status}: {count}")
        
    print("\nConfidence Distribution:")
    for level, count in summary['confidence_distribution'].items():
        print(f"  {level.capitalize()}: {count}")
        
    print("\nTop Evidence Patterns:")
    for pattern, count in summary['top_evidence_patterns'].items():
        print(f"  {pattern}: {count}")

if __name__ == "__main__":
    main()
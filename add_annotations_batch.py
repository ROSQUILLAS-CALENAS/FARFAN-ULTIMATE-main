#!/usr/bin/env python3
"""
Industrial Pipeline Contract Annotation Tool

A sophisticated tool for batch adding pipeline contract annotations to all components
in a codebase. Provides comprehensive error handling, logging, configuration management,
and validation capabilities.

Author: Pipeline Engineering Team
Version: 2.0.0
"""

import os
import re
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional, NamedTuple
from dataclasses import dataclass
from contextlib import contextmanager
from datetime import datetime


class ComponentInfo(NamedTuple):
    """Container for component information"""
    file_path: str
    phase: str
    relative_path: str


@dataclass
class AnnotationConfig:
    """Configuration for annotation process"""
    dry_run: bool = False
    force_overwrite: bool = False
    backup_files: bool = True
    validate_only: bool = False
    excluded_paths: List[str] = None
    included_extensions: List[str] = None
    log_level: str = "INFO"

    def __post_init__(self):
        if self.excluded_paths is None:
            self.excluded_paths = [
                '.git', '__pycache__', '.venv', 'venv', 'node_modules',
                '.pytest_cache', 'build', 'dist', '.tox'
            ]
        if self.included_extensions is None:
            self.included_extensions = ['.py']


class PipelinePhaseExtractor:
    """Handles phase extraction logic with sophisticated pattern matching"""

    # Phase definitions with detailed mappings
    PHASE_DEFINITIONS = {
        'I': {
            'name': 'Ingestion',
            'description': 'Data ingestion and preparation',
            'order': 1,
            'patterns': [
                r'pdf_reader', r'loader', r'feature_extractor', r'normative',
                r'ingestion', r'raw_data', r'extractor', r'reader'
            ],
            'paths': ['/I_ingestion_preparation/']
        },
        'X': {
            'name': 'Context',
            'description': 'Context construction and lineage',
            'order': 2,
            'patterns': [r'context', r'lineage', r'immutable', r'provenance'],
            'paths': ['/X_context_construction/']
        },
        'K': {
            'name': 'Knowledge',
            'description': 'Knowledge extraction and graph construction',
            'order': 3,
            'patterns': [
                r'knowledge', r'graph', r'embedding', r'causal', r'dnp',
                r'entity', r'chunking', r'ontology'
            ],
            'paths': ['/K_knowledge_extraction/']
        },
        'A': {
            'name': 'Analysis',
            'description': 'NLP analysis and processing',
            'order': 4,
            'patterns': [
                r'analyzer', r'analysis', r'question', r'evidence',
                r'mapeo', r'nlp', r'linguistic'
            ],
            'paths': ['/A_analysis_nlp/']
        },
        'L': {
            'name': 'Classification',
            'description': 'Classification and evaluation',
            'order': 5,
            'patterns': [
                r'scoring', r'score', r'classification', r'evaluation',
                r'conformal', r'adaptive_scoring', r'classifier'
            ],
            'paths': ['/L_classification_evaluation/']
        },
        'R': {
            'name': 'Retrieval',
            'description': 'Search and retrieval operations',
            'order': 6,
            'patterns': [
                r'retrieval', r'search', r'index', r'lexical', r'vector',
                r'hybrid', r'reranker', r'recommendation', r'query'
            ],
            'paths': ['/R_search_retrieval/']
        },
        'O': {
            'name': 'Orchestration',
            'description': 'Orchestration and control',
            'order': 7,
            'patterns': [
                r'orchestrator', r'router', r'engine', r'controller', r'manager',
                r'validator', r'monitor', r'telemetry', r'circuit', r'alert',
                r'coordinator', r'dispatcher'
            ],
            'paths': ['/O_orchestration_control/']
        },
        'G': {
            'name': 'Aggregation',
            'description': 'Aggregation and reporting',
            'order': 8,
            'patterns': [
                r'aggregat', r'report', r'compiler', r'meso', r'audit_logger',
                r'consolidat'
            ],
            'paths': ['/G_aggregation_reporting/']
        },
        'T': {
            'name': 'Integration',
            'description': 'Integration and storage',
            'order': 9,
            'patterns': [
                r'metrics', r'analytics', r'feedback', r'compensation',
                r'optimization', r'integration', r'storage', r'persist'
            ],
            'paths': ['/T_integration_storage/']
        },
        'S': {
            'name': 'Synthesis',
            'description': 'Synthesis and output generation',
            'order': 10,
            'patterns': [
                r'synthesis', r'answer', r'formatter', r'output',
                r'generator', r'response'
            ],
            'paths': ['/S_synthesis_output/']
        }
    }

    @classmethod
    def extract_phase(cls, file_path: str, content: Optional[str] = None) -> str:
        """Extract phase from file path and optionally content patterns"""

        # Primary: Check canonical_flow directory structure
        normalized_path = file_path.replace('\\', '/')

        for phase, definition in cls.PHASE_DEFINITIONS.items():
            for path_pattern in definition['paths']:
                if path_pattern in normalized_path:
                    return phase

        # Secondary: Filename pattern analysis
        filename = Path(file_path).stem.lower()

        # Score each phase based on pattern matches
        phase_scores = {}
        for phase, definition in cls.PHASE_DEFINITIONS.items():
            score = 0
            for pattern in definition['patterns']:
                if re.search(pattern, filename, re.IGNORECASE):
                    score += 1
            if score > 0:
                phase_scores[phase] = score

        # Return phase with highest score, or default to Orchestration
        if phase_scores:
            return max(phase_scores.items(), key=lambda x: x[1])[0]

        # Tertiary: Content analysis if provided
        if content:
            for phase, definition in cls.PHASE_DEFINITIONS.items():
                for pattern in definition['patterns']:
                    if re.search(pattern, content, re.IGNORECASE):
                        return phase

        # Default fallback
        return 'O'

    @classmethod
    def get_stage_order(cls, phase: str) -> int:
        """Get stage order for a phase"""
        return cls.PHASE_DEFINITIONS.get(phase, {}).get('order', 7)


class ComponentDetector:
    """Detects pipeline components in source files"""

    COMPONENT_PATTERNS = [
        r'def\s+process\s*\(',
        r'class\s+\w*Processor\b',
        r'class\s+\w*Engine\b',
        r'class\s+\w*Analyzer\b',
        r'class\s+\w*Router\b',
        r'class\s+\w*Orchestrator\b',
        r'class\s+\w*Generator\b',
        r'class\s+\w*Extractor\b',
        r'class\s+\w*Validator\b',
        r'class\s+\w*Builder\b',
        r'class\s+\w*Manager\b',
        r'class\s+\w*Handler\b',
        r'class\s+\w*Controller\b',
        r'class\s+\w*Service\b',
    ]

    ANNOTATION_MARKERS = ['__phase__', '__code__', '__stage_order__']

    @classmethod
    def is_pipeline_component(cls, content: str) -> bool:
        """Check if file contains pipeline component patterns"""
        return any(
            re.search(pattern, content, re.IGNORECASE | re.MULTILINE)
            for pattern in cls.COMPONENT_PATTERNS
        )

    @classmethod
    def has_annotations(cls, content: str) -> bool:
        """Check if file already has required annotations"""
        return all(marker in content for marker in cls.ANNOTATION_MARKERS)


class AnnotationInserter:
    """Handles annotation insertion with sophisticated placement logic"""

    @staticmethod
    def find_insertion_point(lines: List[str]) -> int:
        """Find the optimal place to insert annotations"""
        insert_index = 0

        # Skip shebang
        if lines and lines[0].startswith('#!'):
            insert_index = 1

        # Skip module docstring
        if insert_index < len(lines):
            line = lines[insert_index].strip()
            if line.startswith('"""') or line.startswith("'''"):
                quote_type = '"""' if '"""' in line else "'''"

                # Single-line docstring
                if line.count(quote_type) >= 2:
                    insert_index += 1
                else:
                    # Multi-line docstring
                    for i in range(insert_index + 1, len(lines)):
                        if quote_type in lines[i]:
                            insert_index = i + 1
                            break

        # Skip imports, __future__, and top-level comments
        while insert_index < len(lines):
            stripped = lines[insert_index].strip()
            if (stripped == '' or
                    stripped.startswith('#') or
                    stripped.startswith('import ') or
                    stripped.startswith('from ') or
                    stripped.startswith('__future__')):
                insert_index += 1
            else:
                break

        return insert_index

    @staticmethod
    def generate_annotations(phase: str, component_code: str) -> str:
        """Generate annotation block"""
        stage_order = PipelinePhaseExtractor.get_stage_order(phase)
        phase_info = PipelinePhaseExtractor.PHASE_DEFINITIONS[phase]

        return f"""
# Pipeline Contract Annotations
# Phase: {phase_info['name']} - {phase_info['description']}
# Generated: {datetime.now().isoformat()}
__phase__ = "{phase}"
__code__ = "{component_code}"
__stage_order__ = {stage_order}
"""


class CodeManager:
    """Manages component code generation and tracking"""

    def __init__(self, index_file: Optional[Path] = None):
        self.index_file = index_file or Path("canonical_flow/index.json")
        self.existing_codes: Set[str] = set()
        self.sequence_counters: Dict[str, int] = {}
        self._load_existing_codes()

    def _load_existing_codes(self) -> None:
        """Load existing component codes to avoid conflicts"""
        # Initialize counters
        self.sequence_counters = {
            phase: 1 for phase in PipelinePhaseExtractor.PHASE_DEFINITIONS.keys()
        }

        if not self.index_file.exists():
            return

        try:
            with open(self.index_file, 'r', encoding='utf-8') as f:
                components = json.load(f)

            for comp in components:
                code = comp.get('code', '')
                if code and len(code) >= 3:
                    self.existing_codes.add(code)
                    try:
                        seq_num = int(code[:2])
                        phase_char = code[2:]
                        if phase_char in self.sequence_counters:
                            self.sequence_counters[phase_char] = max(
                                self.sequence_counters[phase_char], seq_num + 1
                            )
                    except (ValueError, IndexError):
                        logging.warning(f"Invalid component code format: {code}")

        except Exception as e:
            logging.warning(f"Could not load existing codes from {self.index_file}: {e}")

    def generate_component_code(self, phase: str) -> str:
        """Generate next available component code for phase"""
        while True:
            code = f"{self.sequence_counters[phase]:02d}{phase}"
            if code not in self.existing_codes:
                self.existing_codes.add(code)
                self.sequence_counters[phase] += 1
                return code
            self.sequence_counters[phase] += 1


class PipelineAnnotator:
    """Main annotator class with comprehensive functionality"""

    def __init__(self, config: AnnotationConfig):
        self.config = config
        self.logger = self._setup_logging()
        self.code_manager = CodeManager()
        self.stats = {
            'total_files': 0,
            'components_found': 0,
            'annotations_added': 0,
            'skipped_existing': 0,
            'errors': 0
        }

    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging"""
        logger = logging.getLogger('pipeline_annotator')
        logger.setLevel(getattr(logging, self.config.log_level.upper()))

        # Clear existing handlers
        logger.handlers.clear()

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # File handler
        log_file = Path(f"annotation_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        return logger

    @contextmanager
    def _file_backup(self, file_path: Path):
        """Context manager for file backup and restore"""
        if not self.config.backup_files:
            yield
            return

        backup_path = file_path.with_suffix(f"{file_path.suffix}.backup")
        try:
            # Create backup
            backup_path.write_bytes(file_path.read_bytes())
            self.logger.debug(f"Created backup: {backup_path}")
            yield

            # Remove backup on success
            if backup_path.exists():
                backup_path.unlink()

        except Exception as e:
            # Restore from backup on error
            if backup_path.exists():
                file_path.write_bytes(backup_path.read_bytes())
                backup_path.unlink()
                self.logger.error(f"Restored from backup due to error: {e}")
            raise

    def _should_process_file(self, file_path: Path) -> bool:
        """Determine if file should be processed"""
        # Check extension
        if file_path.suffix not in self.config.included_extensions:
            return False

        # Skip special files
        if file_path.name in ['__init__.py'] or file_path.name.startswith('test_'):
            return False

        # Check excluded paths
        path_str = str(file_path)
        return not any(excluded in path_str for excluded in self.config.excluded_paths)

    def _read_file_content(self, file_path: Path) -> Optional[str]:
        """Safely read file content"""
        try:
            return file_path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            try:
                return file_path.read_text(encoding='latin-1')
            except Exception as e:
                self.logger.error(f"Could not read {file_path}: {e}")
                return None
        except Exception as e:
            self.logger.error(f"Error reading {file_path}: {e}")
            return None

    def _process_file(self, component: ComponentInfo) -> bool:
        """Process a single component file"""
        file_path = Path(component.file_path)

        try:
            content = self._read_file_content(file_path)
            if content is None:
                return False

            # Check if already annotated
            if ComponentDetector.has_annotations(content):
                if not self.config.force_overwrite:
                    self.logger.info(f"✓ {component.relative_path} already has annotations")
                    self.stats['skipped_existing'] += 1
                    return True

            # Generate component code
            component_code = self.code_manager.generate_component_code(component.phase)

            if self.config.dry_run or self.config.validate_only:
                self.logger.info(
                    f"[DRY RUN] Would add annotations to {component.relative_path} "
                    f"[{component_code}:{component.phase}]"
                )
                return True

            # Process file with backup
            with self._file_backup(file_path):
                lines = content.split('\n')
                insert_index = AnnotationInserter.find_insertion_point(lines)
                annotations = AnnotationInserter.generate_annotations(
                    component.phase, component_code
                )

                lines.insert(insert_index, annotations)

                file_path.write_text('\n'.join(lines), encoding='utf-8')

            self.logger.info(
                f"✅ Added annotations to {component.relative_path} "
                f"[{component_code}:{component.phase}]"
            )
            self.stats['annotations_added'] += 1
            return True

        except Exception as e:
            self.logger.error(f"❌ Error processing {component.relative_path}: {e}")
            self.stats['errors'] += 1
            return False

    def discover_components(self, root_path: Path = None) -> List[ComponentInfo]:
        """Discover all pipeline components in the codebase"""
        if root_path is None:
            root_path = Path.cwd()

        components = []

        self.logger.info(f"🔍 Discovering pipeline components in {root_path}")

        for file_path in root_path.rglob("*.py"):
            self.stats['total_files'] += 1

            if not self._should_process_file(file_path):
                continue

            content = self._read_file_content(file_path)
            if content is None:
                continue

            if ComponentDetector.is_pipeline_component(content):
                if not ComponentDetector.has_annotations(content) or self.config.force_overwrite:
                    phase = PipelinePhaseExtractor.extract_phase(str(file_path), content)
                    relative_path = str(file_path.relative_to(root_path))

                    components.append(ComponentInfo(
                        file_path=str(file_path),
                        phase=phase,
                        relative_path=relative_path
                    ))
                    self.stats['components_found'] += 1

        # Sort components by phase order for consistent processing
        components.sort(
            key=lambda c: PipelinePhaseExtractor.get_stage_order(c.phase)
        )

        return components

    def annotate_components(self, components: List[ComponentInfo]) -> None:
        """Annotate all discovered components"""
        if not components:
            self.logger.warning("No components found to annotate")
            return

        self.logger.info(f"📝 Processing {len(components)} components")

        for component in components:
            self._process_file(component)

    def print_summary(self) -> None:
        """Print comprehensive summary"""
        print("\n" + "=" * 80)
        print("PIPELINE ANNOTATION SUMMARY")
        print("=" * 80)
        print(f"Total files scanned:      {self.stats['total_files']:>6}")
        print(f"Components discovered:    {self.stats['components_found']:>6}")
        print(f"Annotations added:        {self.stats['annotations_added']:>6}")
        print(f"Already annotated:        {self.stats['skipped_existing']:>6}")
        print(f"Errors encountered:       {self.stats['errors']:>6}")
        print("=" * 80)

        if self.config.dry_run:
            print("👆 This was a dry run - no files were modified")
        elif self.config.validate_only:
            print("👆 This was validation only - no files were modified")

        print()


def create_config_from_args(args: argparse.Namespace) -> AnnotationConfig:
    """Create configuration from command line arguments"""
    return AnnotationConfig(
        dry_run=args.dry_run,
        force_overwrite=args.force,
        backup_files=not args.no_backup,
        validate_only=args.validate,
        log_level=args.log_level,
        excluded_paths=args.exclude.split(',') if args.exclude else None
    )


def main():
    """Main entry point with comprehensive argument parsing"""
    parser = argparse.ArgumentParser(
        description="Industrial Pipeline Contract Annotation Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Annotate all components
  %(prog)s --dry-run               # Show what would be annotated
  %(prog)s --validate              # Validate existing annotations
  %(prog)s --force                 # Overwrite existing annotations
  %(prog)s --exclude=test,demo     # Exclude specific directories
  %(prog)s --log-level=DEBUG       # Verbose logging
        """
    )

    parser.add_argument(
        '--dry-run', '-n',
        action='store_true',
        help='Show what would be done without making changes'
    )

    parser.add_argument(
        '--force', '-f',
        action='store_true',
        help='Overwrite existing annotations'
    )

    parser.add_argument(
        '--validate', '-v',
        action='store_true',
        help='Validate existing annotations without modifying files'
    )

    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Skip creating backup files'
    )

    parser.add_argument(
        '--exclude', '-e',
        type=str,
        help='Comma-separated list of paths to exclude'
    )

    parser.add_argument(
        '--log-level', '-l',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Set logging level'
    )

    args = parser.parse_args()

    try:
        config = create_config_from_args(args)
        annotator = PipelineAnnotator(config)

        # Discover and process components
        components = annotator.discover_components()
        annotator.annotate_components(components)
        annotator.print_summary()

        # Exit with appropriate code
        sys.exit(0 if annotator.stats['errors'] == 0 else 1)

    except KeyboardInterrupt:
        print("\n❌ Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
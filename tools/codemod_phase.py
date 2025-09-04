#!/usr/bin/env python3
"""
LibCST-based codemod tool for canonical pipeline phase standardization.

This tool performs AST-based refactoring targeting:
1. 05I_* file naming corrections  
2. Scattered module consolidation into proper phase directories
3. Safe import rewriting when modules are moved/renamed
4. Integrated mypy and ruff validation for type safety and code quality
5. Command-line interface with dry-run, phase targeting, and rollback capabilities

Usage:
    python tools/codemod_phase.py --dry-run
    python tools/codemod_phase.py --phase I_ingestion_preparation
    python tools/codemod_phase.py --rollback backup_20231201_143022
"""

import argparse
import ast
import datetime
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

try:
    import libcst as cst
    from libcst import matchers as m, metadata
    from libcst.codemod import CodemodContext, VisitorBasedCodemodCommand
except ImportError:
    print("Error: libcst is required. Install with: pip install libcst")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PhaseMapping:
    """Phase directory and file naming standards."""
    
    PHASE_DIRECTORIES = {
        'A': 'A_analysis_nlp',
        'G': 'G_aggregation_reporting', 
        'I': 'I_ingestion_preparation',
        'K': 'K_knowledge_extraction',
        'L': 'L_classification_evaluation',
        'O': 'O_orchestration_control',
        'R': 'R_search_retrieval',
        'S': 'S_synthesis_output',
        'T': 'T_integration_storage',
        'X': 'X_context_construction'
    }
    
    # Pattern for detecting non-standard file names that should be corrected
    LEGACY_PATTERNS = [
        re.compile(r'^(\d{2})([AGIKLRSTX])_(.+)\.py$'),  # e.g., 05I_raw_data_generator.py
        re.compile(r'^([AGIKLRSTX])(\d{2})_(.+)\.py$'),  # e.g., I05_raw_data_generator.py  
        re.compile(r'^phase_([AGIKLRSTX])_(\d{2})_(.+)\.py$'),  # e.g., phase_I_05_raw_data.py
    ]
    
    @classmethod
    def normalize_filename(cls, filename: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract phase and standardized name from legacy filename.
        
        Returns:
            (phase_letter, standardized_name) or (None, None) if no match
        """
        for pattern in cls.LEGACY_PATTERNS:
            match = pattern.match(filename)
            if match:
                groups = match.groups()
                # First pattern: (\d{2})([AGIKLRSTX])_(.+)\.py$ - e.g., 05I_raw_data_generator.py
                if len(groups) == 3 and groups[0].isdigit() and groups[1] in 'AGIKLRSTX':
                    order, phase, name = groups
                    return phase, f"{name}.py"
                # Second pattern: ([AGIKLRSTX])(\d{2})_(.+)\.py$ - e.g., I05_raw_data_generator.py  
                elif len(groups) == 3 and groups[0] in 'AGIKLRSTX' and groups[1].isdigit():
                    phase, order, name = groups
                    return phase, f"{name}.py"
                # Third pattern: phase_([AGIKLRSTX])_(\d{2})_(.+)\.py$ - e.g., phase_I_05_raw_data.py
                elif len(groups) == 3 and 'phase_' in pattern.pattern:
                    phase, order, name = groups 
                    return phase, f"{name}.py"
        return None, None
    
    @classmethod
    def get_canonical_path(cls, phase_letter: str, filename: str) -> Optional[Path]:
        """Get canonical path for a phase file."""
        phase_dir = cls.PHASE_DIRECTORIES.get(phase_letter)
        if not phase_dir:
            return None
        return Path("canonical_flow") / phase_dir / filename


class ImportRewriter(VisitorBasedCodemodCommand):
    """LibCST codemod for rewriting import statements when modules are moved."""
    
    DESCRIPTION: str = "Rewrites import statements for moved/renamed modules"
    
    def __init__(self, context: CodemodContext, import_mapping: Dict[str, str]):
        super().__init__(context)
        self.import_mapping = import_mapping
        self.changes_made = 0
    
    def leave_ImportFrom(
        self, original_node: cst.ImportFrom, updated_node: cst.ImportFrom
    ) -> cst.ImportFrom:
        """Rewrite 'from X import Y' statements."""
        if updated_node.module is None:
            return updated_node
            
        module_name = self._get_module_name(updated_node.module)
        if module_name in self.import_mapping:
            new_module = self.import_mapping[module_name]
            logger.debug(f"Rewriting import: {module_name} -> {new_module}")
            
            new_module_node = self._create_module_attribute(new_module)
            self.changes_made += 1
            return updated_node.with_changes(module=new_module_node)
            
        return updated_node
    
    def leave_Import(
        self, original_node: cst.Import, updated_node: cst.Import
    ) -> cst.Import:
        """Rewrite 'import X' statements."""
        new_names = []
        changed = False
        
        for name in updated_node.names:
            if isinstance(name, cst.ImportStar):
                new_names.append(name)
                continue
                
            if isinstance(name, cst.ImportAlias):
                module_name = self._get_module_name(name.name)
                if module_name in self.import_mapping:
                    new_module = self.import_mapping[module_name]
                    new_name_node = self._create_module_attribute(new_module)
                    new_alias = name.with_changes(name=new_name_node)
                    new_names.append(new_alias)
                    changed = True
                    logger.debug(f"Rewriting import: {module_name} -> {new_module}")
                else:
                    new_names.append(name)
            else:
                new_names.append(name)
        
        if changed:
            self.changes_made += 1
            return updated_node.with_changes(names=new_names)
        return updated_node
    
    def _get_module_name(self, node: cst.CSTNode) -> str:
        """Extract module name from AST node."""
        if isinstance(node, cst.Name):
            return node.value
        elif isinstance(node, cst.Attribute):
            # Handle dotted imports like a.b.c
            parts = []
            current = node
            while isinstance(current, cst.Attribute):
                parts.append(current.attr.value)
                current = current.value
            if isinstance(current, cst.Name):
                parts.append(current.value)
            return '.'.join(reversed(parts))
        return ""
    
    def _create_module_attribute(self, module_name: str) -> Union[cst.Name, cst.Attribute]:
        """Create AST node for dotted module name."""
        parts = module_name.split('.')
        if len(parts) == 1:
            return cst.Name(parts[0])
        
        result = cst.Name(parts[0])
        for part in parts[1:]:
            result = cst.Attribute(value=result, attr=cst.Name(part))
        return result


class PhaseRefactoringEngine:
    """Main engine for phase standardization refactoring."""
    
    def __init__(self, dry_run: bool = True):
        self.dry_run = dry_run
        self.backup_dir: Optional[Path] = None
        self.moves_made: List[Tuple[Path, Path]] = []
        self.import_changes: Dict[str, int] = {}
        self.validation_errors: List[str] = []
        
    def create_backup(self) -> Path:
        """Create backup of current state before refactoring."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = Path(f"backups/codemod_phase_{timestamp}")
        
        if not self.dry_run:
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Backup canonical_flow directory
            canonical_flow = Path("canonical_flow")
            if canonical_flow.exists():
                shutil.copytree(canonical_flow, backup_dir / "canonical_flow")
                
            # Backup any root-level files that might be moved
            for pattern in PhaseMapping.LEGACY_PATTERNS:
                for file_path in Path(".").glob("*.py"):
                    if pattern.match(file_path.name):
                        shutil.copy2(file_path, backup_dir / file_path.name)
                        
            logger.info(f"Created backup at {backup_dir}")
        else:
            logger.info(f"[DRY RUN] Would create backup at {backup_dir}")
            
        self.backup_dir = backup_dir
        return backup_dir
    
    def find_files_to_refactor(self, target_phase: Optional[str] = None) -> List[Tuple[Path, str, str]]:
        """
        Find files that need phase standardization.
        
        Returns:
            List of (current_path, phase_letter, new_filename) tuples
        """
        files_to_move = []
        
        # Search in current directory and canonical_flow
        search_paths = [Path("."), Path("canonical_flow")]
        
        for search_path in search_paths:
            if not search_path.exists():
                continue
                
            # For root directory, only check immediate .py files, not recursive
            if search_path == Path("."):
                file_paths = search_path.glob("*.py")
            else:
                file_paths = search_path.rglob("*.py")
                
            for file_path in file_paths:
                # Skip __pycache__ and .git directories
                if any(part.startswith('.') or part == '__pycache__' 
                      for part in file_path.parts):
                    continue
                    
                phase_letter, new_name = PhaseMapping.normalize_filename(file_path.name)
                if file_path.name.startswith("05I_"):
                    logger.debug(f"FOUND 05I FILE: Checking {file_path.name}: phase={phase_letter}, new_name={new_name}")
                else:
                    logger.debug(f"Checking {file_path.name}: phase={phase_letter}, new_name={new_name}")
                
                if phase_letter and new_name:
                    # Filter by target phase if specified
                    if target_phase and phase_letter != target_phase:
                        continue
                        
                    canonical_path = PhaseMapping.get_canonical_path(phase_letter, new_name)
                    if canonical_path and canonical_path != file_path:
                        files_to_move.append((file_path, phase_letter, new_name))
                        logger.debug(f"Found file to move: {file_path} -> {canonical_path}")
                        
        return files_to_move
    
    def move_files(self, files_to_move: List[Tuple[Path, str, str]]) -> Dict[str, str]:
        """
        Move files to their canonical locations.
        
        Returns:
            Dictionary mapping old module paths to new module paths
        """
        import_mapping = {}
        
        for old_path, phase_letter, new_name in files_to_move:
            canonical_path = PhaseMapping.get_canonical_path(phase_letter, new_name)
            
            if not canonical_path:
                logger.warning(f"Could not determine canonical path for {old_path}")
                continue
                
            # Create mapping for import rewriting
            old_module = self._path_to_module(old_path)
            new_module = self._path_to_module(canonical_path)
            import_mapping[old_module] = new_module
            
            if self.dry_run:
                logger.info(f"[DRY RUN] Would move {old_path} -> {canonical_path}")
            else:
                # Ensure target directory exists
                canonical_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Move the file
                shutil.move(str(old_path), str(canonical_path))
                logger.info(f"Moved {old_path} -> {canonical_path}")
                
            self.moves_made.append((old_path, canonical_path))
            
        return import_mapping
    
    def rewrite_imports(self, import_mapping: Dict[str, str]) -> None:
        """Rewrite import statements across the codebase."""
        if not import_mapping:
            return
            
        # Find all Python files to update
        python_files = []
        for root_dir in [Path("."), Path("canonical_flow"), Path("tests"), Path("src")]:
            if root_dir.exists():
                python_files.extend(root_dir.rglob("*.py"))
        
        context = CodemodContext()
        rewriter = ImportRewriter(context, import_mapping)
        
        for file_path in python_files:
            try:
                if self.dry_run:
                    logger.debug(f"[DRY RUN] Would analyze imports in {file_path}")
                    continue
                    
                with open(file_path, 'r', encoding='utf-8') as f:
                    source_code = f.read()
                    
                # Parse with LibCST
                source_tree = cst.parse_expression(source_code) if source_code.strip().startswith('(') else cst.parse_module(source_code)
                
                # Apply the transform
                updated_tree = source_tree.visit(rewriter)
                
                # Write back if changes were made
                if rewriter.changes_made > 0:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(updated_tree.code)
                    logger.info(f"Updated imports in {file_path} ({rewriter.changes_made} changes)")
                    self.import_changes[str(file_path)] = rewriter.changes_made
                    
            except Exception as e:
                error_msg = f"Error processing {file_path}: {e}"
                logger.error(error_msg)
                self.validation_errors.append(error_msg)
    
    def validate_with_mypy(self) -> bool:
        """Run mypy validation on refactored code."""
        try:
            if self.dry_run:
                logger.info("[DRY RUN] Would run mypy validation")
                return True
                
            result = subprocess.run(
                ["mypy", "canonical_flow/", "--ignore-missing-imports"],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                logger.info("✓ MyPy validation passed")
                return True
            else:
                error_msg = f"MyPy validation failed:\n{result.stdout}\n{result.stderr}"
                logger.error(error_msg)
                self.validation_errors.append(error_msg)
                return False
                
        except subprocess.TimeoutExpired:
            error_msg = "MyPy validation timed out"
            logger.error(error_msg)
            self.validation_errors.append(error_msg)
            return False
        except FileNotFoundError:
            logger.warning("MyPy not installed, skipping type validation")
            return True
    
    def validate_with_ruff(self) -> bool:
        """Run ruff validation on refactored code."""
        try:
            if self.dry_run:
                logger.info("[DRY RUN] Would run ruff validation")
                return True
                
            result = subprocess.run(
                ["ruff", "check", "canonical_flow/"],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                logger.info("✓ Ruff validation passed")
                return True
            else:
                error_msg = f"Ruff validation failed:\n{result.stdout}\n{result.stderr}"
                logger.error(error_msg)
                self.validation_errors.append(error_msg)
                return False
                
        except subprocess.TimeoutExpired:
            error_msg = "Ruff validation timed out"
            logger.error(error_msg)
            self.validation_errors.append(error_msg)
            return False
        except FileNotFoundError:
            logger.warning("Ruff not installed, skipping linting validation")
            return True
    
    def rollback(self, backup_path: str) -> bool:
        """Rollback changes from a backup."""
        backup_dir = Path(backup_path)
        
        if not backup_dir.exists():
            logger.error(f"Backup directory {backup_dir} does not exist")
            return False
            
        try:
            # Remove current canonical_flow if it exists
            canonical_flow = Path("canonical_flow")
            if canonical_flow.exists():
                shutil.rmtree(canonical_flow)
                
            # Restore from backup
            backup_canonical = backup_dir / "canonical_flow"
            if backup_canonical.exists():
                shutil.copytree(backup_canonical, canonical_flow)
                
            # Restore root-level files
            for backup_file in backup_dir.glob("*.py"):
                target_file = Path(backup_file.name)
                shutil.copy2(backup_file, target_file)
                
            logger.info(f"Successfully rolled back from {backup_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False
    
    def _path_to_module(self, file_path: Path) -> str:
        """Convert file path to Python module path."""
        # Remove .py extension and convert path separators to dots
        module_path = str(file_path.with_suffix(''))
        module_path = module_path.replace('/', '.').replace('\\', '.')
        
        # Remove leading dot if present
        if module_path.startswith('.'):
            module_path = module_path[1:]
            
        return module_path
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive refactoring report."""
        return {
            'timestamp': datetime.datetime.now().isoformat(),
            'dry_run': self.dry_run,
            'backup_directory': str(self.backup_dir) if self.backup_dir else None,
            'files_moved': [
                {'from': str(old), 'to': str(new)} 
                for old, new in self.moves_made
            ],
            'import_changes': self.import_changes,
            'validation_errors': self.validation_errors,
            'summary': {
                'files_moved': len(self.moves_made),
                'files_with_import_changes': len(self.import_changes),
                'total_import_changes': sum(self.import_changes.values()),
                'validation_errors': len(self.validation_errors)
            }
        }


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="LibCST-based codemod for canonical pipeline phase standardization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run (default) - show what would be changed
  python tools/codemod_phase.py --dry-run
  
  # Target specific phase 
  python tools/codemod_phase.py --phase I --dry-run
  
  # Execute refactoring (remove --dry-run)
  python tools/codemod_phase.py --phase I
  
  # Rollback from backup
  python tools/codemod_phase.py --rollback backups/codemod_phase_20231201_143022
        """
    )
    
    parser.add_argument(
        '--dry-run', 
        action='store_true', 
        default=True,
        help='Show what would be changed without making actual changes'
    )
    
    parser.add_argument(
        '--execute',
        action='store_true',
        help='Execute the refactoring (overrides --dry-run)'
    )
    
    parser.add_argument(
        '--phase',
        choices=['A', 'G', 'I', 'K', 'L', 'O', 'R', 'S', 'T', 'X'],
        help='Target specific phase for refactoring'
    )
    
    parser.add_argument(
        '--rollback',
        type=str,
        help='Rollback changes from specified backup directory'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Handle rollback case
    if args.rollback:
        engine = PhaseRefactoringEngine(dry_run=False)
        success = engine.rollback(args.rollback)
        sys.exit(0 if success else 1)
    
    # Determine dry-run mode
    dry_run = args.dry_run and not args.execute
    
    logger.info(f"Starting phase refactoring {'(DRY RUN)' if dry_run else '(EXECUTING)'}")
    
    # Initialize refactoring engine
    engine = PhaseRefactoringEngine(dry_run=dry_run)
    
    try:
        # Step 1: Create backup
        backup_dir = engine.create_backup()
        
        # Step 2: Find files to refactor
        logger.info("Scanning for files to refactor...")
        files_to_move = engine.find_files_to_refactor(target_phase=args.phase)
        
        if not files_to_move:
            logger.info("No files found that require phase standardization")
            return 0
            
        logger.info(f"Found {len(files_to_move)} files to refactor")
        for old_path, phase, new_name in files_to_move:
            canonical_path = PhaseMapping.get_canonical_path(phase, new_name)
            logger.info(f"  {old_path} -> {canonical_path}")
        
        # Step 3: Move files and generate import mapping
        logger.info("Moving files to canonical locations...")
        import_mapping = engine.move_files(files_to_move)
        
        # Step 4: Rewrite imports
        if import_mapping:
            logger.info("Rewriting import statements...")
            engine.rewrite_imports(import_mapping)
        
        # Step 5: Validate with mypy and ruff
        if not dry_run:
            logger.info("Running validation...")
            mypy_ok = engine.validate_with_mypy()
            ruff_ok = engine.validate_with_ruff()
            
            if not (mypy_ok and ruff_ok):
                logger.error("Validation failed - consider rolling back")
                logger.info(f"To rollback: python tools/codemod_phase.py --rollback {backup_dir}")
        
        # Step 6: Generate report
        report = engine.generate_report()
        report_path = Path("codemod_phase_report.json")
        
        if not dry_run:
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Report saved to {report_path}")
        else:
            logger.info("Report:")
            print(json.dumps(report, indent=2))
        
        # Summary
        summary = report['summary']
        logger.info(f"Refactoring complete:")
        logger.info(f"  Files moved: {summary['files_moved']}")
        logger.info(f"  Import changes: {summary['total_import_changes']} across {summary['files_with_import_changes']} files")
        
        if summary['validation_errors'] > 0:
            logger.warning(f"  Validation errors: {summary['validation_errors']}")
            return 1
            
        return 0
        
    except Exception as e:
        logger.error(f"Refactoring failed: {e}")
        if engine.backup_dir and not dry_run:
            logger.info(f"To rollback: python tools/codemod_phase.py --rollback {engine.backup_dir}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
"""
Question Analyzer with Standardized Process API

This module implements standardized question analysis with:
- Document path processing 
- Canonical flow artifact generation
- Intent classification with frozen vocabulary
- Dimension mapping to 4-dimension taxonomy (DE-1, DE-2, DE-3, DE-4)
- Graceful handling for empty/malformed questions
- Deterministic ordering and stable vocabulary mapping
"""

import json
import re
import os
# # # from pathlib import Path  # Module not found  # Module not found  # Module not found
# # # from dataclasses import dataclass, field  # Module not found  # Module not found  # Module not found
# # # from enum import Enum  # Module not found  # Module not found  # Module not found
# # # from typing import Any, Dict, List, Optional, Set, Tuple, Union  # Module not found  # Module not found  # Module not found
# # # from collections import OrderedDict  # Module not found  # Module not found  # Module not found

# Optional imports with graceful fallbacks
try:
    import networkx as nx
except ImportError:
    nx = None

try:
    import numpy as np
except ImportError:
    np = None

try:
# # #     from sentence_transformers import SentenceTransformer  # Module not found  # Module not found  # Module not found
except ImportError:
    SentenceTransformer = None

try:
# # #     from transformers import AutoModel, AutoTokenizer  # Module not found  # Module not found  # Module not found
except ImportError:
    AutoModel = None
    AutoTokenizer = None

# Import audit logger for execution tracing
try:
# # #     from canonical_flow.analysis.audit_logger import get_audit_logger  # Module not found  # Module not found  # Module not found
except ImportError:
    get_audit_logger = None


# Frozen vocabulary for stable intent classification
FROZEN_INTENT_VOCABULARY = OrderedDict([
    ("causal_analysis", ["cause", "effect", "impact", "influence", "due to", "because of", "leads to"]),
    ("comparative_analysis", ["compare", "versus", "vs", "difference", "contrast", "better", "worse"]),
    ("definitional_analysis", ["what is", "define", "definition", "meaning", "concept", "explanation"]),
    ("descriptive_analysis", ["describe", "characteristics", "features", "properties", "attributes"]),
    ("evaluative_analysis", ["evaluate", "assess", "judge", "rate", "quality", "effectiveness"]),
    ("predictive_analysis", ["predict", "forecast", "future", "will", "expected", "likely"]),
    ("procedural_analysis", ["how to", "process", "steps", "procedure", "method", "approach"]),
    ("quantitative_analysis", ["how many", "how much", "count", "measure", "statistics", "numbers"]),
])

# Fixed intent-to-dimension mapping with 4-dimension taxonomy
INTENT_DIMENSION_MAPPING = OrderedDict([
    ("causal_analysis", "DE-3"),        # Effects - what impacts result
    ("comparative_analysis", "DE-2"),    # Efficiency - comparing alternatives
    ("definitional_analysis", "DE-1"),   # Products - defining scope and deliverables
    ("descriptive_analysis", "DE-1"),    # Products - describing what will be produced
    ("evaluative_analysis", "DE-3"),     # Effects - assessing outcomes
    ("predictive_analysis", "DE-4"),     # Impact - long-term projections
    ("procedural_analysis", "DE-2"),     # Efficiency - optimizing processes
    ("quantitative_analysis", "DE-1"),   # Products - measuring deliverables
    ("unknown_intent", "unassigned_dimension"),  # Default fallback
])


class QuestionIntent(Enum):
    """Standardized question intent classification"""
    CAUSAL_ANALYSIS = "causal_analysis"
    COMPARATIVE_ANALYSIS = "comparative_analysis"
    DEFINITIONAL_ANALYSIS = "definitional_analysis"
    DESCRIPTIVE_ANALYSIS = "descriptive_analysis"
    EVALUATIVE_ANALYSIS = "evaluative_analysis"
    PREDICTIVE_ANALYSIS = "predictive_analysis"
    PROCEDURAL_ANALYSIS = "procedural_analysis"
    QUANTITATIVE_ANALYSIS = "quantitative_analysis"
    UNKNOWN_INTENT = "unknown_intent"


class DimensionCode(Enum):
    """4-dimension taxonomy for question classification"""
    DE_1 = "DE-1"  # Products - deliverables and outputs
    DE_2 = "DE-2"  # Efficiency - resource optimization
    DE_3 = "DE-3"  # Effects - outcomes and results
    DE_4 = "DE-4"  # Impact - long-term transformations
    UNASSIGNED = "unassigned_dimension"


@dataclass
class QuestionAnalysis:
    """Analysis result for a single question"""
    question_id: str
    question_text: str
    intent: str = "unknown_intent"
    dimension: str = "unassigned_dimension"
    confidence_score: float = 0.0
    keywords: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        # Ensure deterministic ordering
        self.keywords = sorted(self.keywords) if self.keywords else []
        if self.metadata:
            self.metadata = dict(sorted(self.metadata.items()))


class QuestionAnalyzer:
    """
    Standardized Question Analyzer with Process API.
    
    Implements:
    - Document path processing
    - Canonical flow artifact generation
    - Intent classification using frozen vocabulary
    - Dimension mapping to 4-dimension taxonomy
    - Graceful handling for empty/malformed questions
    - Deterministic ordering through stable vocabulary
    """

    def __init__(self, encoder_model: str = None):
        """
        Initialize with optional encoder model for advanced analysis.
        
        Args:
            encoder_model: Optional sentence encoder model name
        """
        # Optional model initialization
        self.encoder = None
        if SentenceTransformer and encoder_model:
            try:
                self.encoder = SentenceTransformer(encoder_model)
            except Exception:
                pass  # Graceful fallback
        
        # Stable vocabulary with deterministic ordering
        self.intent_vocabulary = FROZEN_INTENT_VOCABULARY
        self.dimension_mapping = INTENT_DIMENSION_MAPPING
        
        # Create sorted keyword sets for fast lookup
        self._keyword_to_intent = {}
        for intent, keywords in self.intent_vocabulary.items():
            for keyword in keywords:
                if keyword not in self._keyword_to_intent:
                    self._keyword_to_intent[keyword] = []
                self._keyword_to_intent[keyword].append(intent)
        
        # Ensure deterministic keyword ordering
        self._sorted_keywords = sorted(self._keyword_to_intent.keys(), key=len, reverse=True)

    def process(self, document_path: str) -> Dict[str, Any]:
        """
        Standardized process API that accepts document path and generates canonical flow artifacts.
        
        Args:
            document_path: Path to document containing questions to analyze
            
        Returns:
            Dict containing analysis results and artifact metadata
        """
        try:
            # Parse document path and extract stem
            doc_path = Path(document_path)
            if not doc_path.exists():
                return self._create_error_result(f"Document not found: {document_path}")
            
            stem = doc_path.stem
            
            # Read and parse document
            questions = self._extract_questions_from_document(doc_path)
            
            if not questions:
                return self._create_empty_result(stem)
            
            # Analyze all questions
            analyses = []
            for i, question_text in enumerate(questions):
                question_id = f"{stem}_q{i+1:03d}"
                analysis = self._analyze_single_question(question_id, question_text)
                analyses.append(analysis)
            
            # Create canonical flow artifact
            artifact_data = {
                "document_stem": stem,
                "document_path": str(document_path),
                "total_questions": len(questions),
                "analyses": [self._analysis_to_dict(analysis) for analysis in analyses],
                "processing_metadata": {
                    "vocabulary_version": "1.0",
                    "dimension_taxonomy": "4D-DE",
                    "processing_timestamp": self._get_timestamp(),
                    "deterministic_ordering": True
                },
                "dimension_summary": self._create_dimension_summary(analyses),
                "intent_summary": self._create_intent_summary(analyses)
            }
            
            # Save artifact to canonical_flow/analysis/
            artifact_path = self._save_canonical_artifact(stem, artifact_data)
            
            return {
                "status": "success",
                "artifact_path": artifact_path,
                "total_questions": len(questions),
                "dimension_distribution": artifact_data["dimension_summary"],
                "intent_distribution": artifact_data["intent_summary"]
            }
            
        except Exception as e:
            return self._create_error_result(f"Processing failed: {str(e)}")
    
    def _extract_questions_from_document(self, doc_path: Path) -> List[str]:
# # #         """Extract questions from document with multiple format support"""  # Module not found  # Module not found  # Module not found
        questions = []
        
        try:
            content = doc_path.read_text(encoding='utf-8')
        except Exception:
            try:
                content = doc_path.read_text(encoding='latin-1')
            except Exception:
                return []
        
        # Try JSON format first
        if doc_path.suffix.lower() == '.json':
            try:
                data = json.loads(content)
                questions = self._extract_from_json(data)
            except Exception:
                pass
        
        # Fallback to text extraction
        if not questions:
            questions = self._extract_from_text(content)
        
        # Remove empty or whitespace-only questions
        questions = [q.strip() for q in questions if q.strip()]
        
        return questions
    
    def _extract_from_json(self, data: Any) -> List[str]:
# # #         """Extract questions from JSON data structure"""  # Module not found  # Module not found  # Module not found
        questions = []
        
        if isinstance(data, dict):
            # Look for common question fields
            for key in ['questions', 'question_list', 'items', 'data']:
                if key in data and isinstance(data[key], list):
                    for item in data[key]:
                        if isinstance(item, str):
                            questions.append(item)
                        elif isinstance(item, dict):
                            # Look for question text in dict
                            for subkey in ['question', 'text', 'content', 'question_text']:
                                if subkey in item and isinstance(item[subkey], str):
                                    questions.append(item[subkey])
                                    break
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, str):
                    questions.append(item)
                elif isinstance(item, dict):
                    # Look for question text in dict
                    for key in ['question', 'text', 'content', 'question_text']:
                        if key in item and isinstance(item[key], str):
                            questions.append(item[key])
                            break
        
        return questions
    
    def _extract_from_text(self, content: str) -> List[str]:
# # #         """Extract questions from plain text using patterns"""  # Module not found  # Module not found  # Module not found
        questions = []
        
        # Split by lines and look for question patterns
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Direct question detection
            if line.endswith('?'):
                questions.append(line)
            # Question markers
            elif any(line.startswith(marker) for marker in ['Q:', 'Question:', '¿']):
                questions.append(line)
            # Numbered questions
            elif re.match(r'^\d+[.)]\s*', line):
                question_text = re.sub(r'^\d+[.)]\s*', '', line)
                if question_text:
                    questions.append(question_text)
        
        return questions
    
    def _analyze_single_question(self, question_id: str, question_text: str) -> QuestionAnalysis:
        """Analyze a single question with graceful error handling"""
        if not question_text or not question_text.strip():
            return QuestionAnalysis(
                question_id=question_id,
                question_text=question_text or "",
                intent="unknown_intent",
                dimension="unassigned_dimension",
                confidence_score=0.0
            )
        
        # Classify intent using frozen vocabulary
        intent = self._classify_intent(question_text)
        
        # Map to dimension using stable mapping
        dimension = self.dimension_mapping.get(intent, "unassigned_dimension")
        
        # Calculate confidence score
        confidence = self._calculate_confidence(question_text, intent)
        
        # Extract keywords
        keywords = self._extract_keywords(question_text, intent)
        
        return QuestionAnalysis(
            question_id=question_id,
            question_text=question_text.strip(),
            intent=intent,
            dimension=dimension,
            confidence_score=confidence,
            keywords=keywords,
            metadata={
                "text_length": len(question_text.strip()),
                "has_question_mark": question_text.strip().endswith('?'),
                "processing_method": "frozen_vocabulary"
            }
        )
    
    def _classify_intent(self, question_text: str) -> str:
        """Classify question intent using frozen vocabulary with stable ordering"""
        text_lower = question_text.lower()
        
        # Score each intent based on keyword matches
        intent_scores = {intent: 0 for intent in self.intent_vocabulary.keys()}
        
        # Check keywords in deterministic order (longest first)
        for keyword in self._sorted_keywords:
            if keyword in text_lower:
                for intent in self._keyword_to_intent[keyword]:
                    # Weight by keyword length for specificity
                    intent_scores[intent] += len(keyword.split())
        
        # Find best intent with deterministic tiebreaking
        if any(score > 0 for score in intent_scores.values()):
            # Sort by score (desc) then by intent name (asc) for stable tiebreaking
            best_intent = max(intent_scores.items(), key=lambda x: (x[1], -ord(x[0][0])))
            return best_intent[0]
        
        return "unknown_intent"
    
    def _calculate_confidence(self, question_text: str, intent: str) -> float:
        """Calculate confidence score for intent classification"""
        if intent == "unknown_intent":
            return 0.0
        
        text_lower = question_text.lower()
        
        # Count matching keywords for this intent
        matching_keywords = 0
        total_keywords = len(self.intent_vocabulary.get(intent, []))
        
        if total_keywords > 0:
            for keyword in self.intent_vocabulary[intent]:
                if keyword in text_lower:
                    matching_keywords += 1
        
            # Basic confidence based on keyword coverage
            keyword_confidence = matching_keywords / total_keywords
        else:
            keyword_confidence = 0.0
        
        # Adjust for question structure
        structure_bonus = 0.1 if question_text.strip().endswith('?') else 0.0
        length_factor = min(1.0, len(question_text.split()) / 10.0)
        
        confidence = min(1.0, keyword_confidence + structure_bonus + (length_factor * 0.1))
        return round(confidence, 3)
    
    def _extract_keywords(self, question_text: str, intent: str) -> List[str]:
# # #         """Extract relevant keywords from question text"""  # Module not found  # Module not found  # Module not found
        if intent == "unknown_intent":
            return []
        
        text_lower = question_text.lower()
        found_keywords = []
        
# # #         # Find matching keywords from vocabulary  # Module not found  # Module not found  # Module not found
        intent_keywords = self.intent_vocabulary.get(intent, [])
        for keyword in intent_keywords:
            if keyword in text_lower:
                found_keywords.append(keyword)
        
        return sorted(found_keywords)  # Deterministic ordering
    
    def _analysis_to_dict(self, analysis: QuestionAnalysis) -> Dict[str, Any]:
        """Convert analysis object to dictionary for JSON serialization"""
        return {
            "question_id": analysis.question_id,
            "question_text": analysis.question_text,
            "intent": analysis.intent,
            "dimension": analysis.dimension,
            "confidence_score": analysis.confidence_score,
            "keywords": analysis.keywords,
            "metadata": analysis.metadata
        }
    
    def _create_dimension_summary(self, analyses: List[QuestionAnalysis]) -> Dict[str, int]:
        """Create summary of dimension distribution"""
        dimension_counts = {}
        for analysis in analyses:
            dim = analysis.dimension
            dimension_counts[dim] = dimension_counts.get(dim, 0) + 1
        
        # Ensure all dimensions are represented with deterministic ordering
        all_dimensions = ["DE-1", "DE-2", "DE-3", "DE-4", "unassigned_dimension"]
        return OrderedDict((dim, dimension_counts.get(dim, 0)) for dim in all_dimensions)
    
    def _create_intent_summary(self, analyses: List[QuestionAnalysis]) -> Dict[str, int]:
        """Create summary of intent distribution"""
        intent_counts = {}
        for analysis in analyses:
            intent = analysis.intent
            intent_counts[intent] = intent_counts.get(intent, 0) + 1
        
        # Ensure all intents are represented with deterministic ordering
        all_intents = sorted(self.intent_vocabulary.keys()) + ["unknown_intent"]
        return OrderedDict((intent, intent_counts.get(intent, 0)) for intent in all_intents)
    
    def _save_canonical_artifact(self, stem: str, artifact_data: Dict[str, Any]) -> str:
        """Save artifact to canonical_flow/analysis/<stem>_question.json"""
        # Ensure directory exists
        analysis_dir = Path("canonical_flow/analysis")
        analysis_dir.mkdir(parents=True, exist_ok=True)
        
        # Create artifact filename
        artifact_filename = f"{stem}_question.json"
        artifact_path = analysis_dir / artifact_filename
        
        # Save with pretty formatting
        with open(artifact_path, 'w', encoding='utf-8') as f:
            json.dump(artifact_data, f, indent=2, ensure_ascii=False)
        
        return str(artifact_path)
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create standardized error result"""
        return {
            "status": "error",
            "error": error_message,
            "artifact_path": None,
            "total_questions": 0,
            "dimension_distribution": self._get_empty_dimension_summary(),
            "intent_distribution": self._get_empty_intent_summary()
        }
    
    def _create_empty_result(self, stem: str) -> Dict[str, Any]:
        """Create result for empty document"""
        artifact_data = {
            "document_stem": stem,
            "document_path": "",
            "total_questions": 0,
            "analyses": [],
            "processing_metadata": {
                "vocabulary_version": "1.0",
                "dimension_taxonomy": "4D-DE",
                "processing_timestamp": self._get_timestamp(),
                "deterministic_ordering": True
            },
            "dimension_summary": self._get_empty_dimension_summary(),
            "intent_summary": self._get_empty_intent_summary()
        }
        
        artifact_path = self._save_canonical_artifact(stem, artifact_data)
        
        return {
            "status": "success",
            "artifact_path": artifact_path,
            "total_questions": 0,
            "dimension_distribution": artifact_data["dimension_summary"],
            "intent_distribution": artifact_data["intent_summary"]
        }
    
    def _get_empty_dimension_summary(self) -> Dict[str, int]:
        """Get empty dimension summary with all dimensions"""
        all_dimensions = ["DE-1", "DE-2", "DE-3", "DE-4", "unassigned_dimension"]
        return OrderedDict((dim, 0) for dim in all_dimensions)
    
    def _get_empty_intent_summary(self) -> Dict[str, int]:
        """Get empty intent summary with all intents"""
        all_intents = sorted(self.intent_vocabulary.keys()) + ["unknown_intent"]
        return OrderedDict((intent, 0) for intent in all_intents)
    
    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format"""
# # #         from datetime import datetime  # Module not found  # Module not found  # Module not found
        return datetime.now().isoformat()


    # Legacy methods for backward compatibility - retained but deprecated
    def analyze_question(self, question: str, question_id: str) -> 'QuestionAnalysis':
        """Legacy method - use process() instead"""
        return self._analyze_single_question(question_id, question)
    
    def extract_search_patterns(self, question: str) -> List[str]:
        """Legacy method - basic pattern extraction"""
        keywords = []
        words = question.lower().split()
        
        # Extract meaningful words (>2 chars, not stopwords)
        stopwords = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
        for word in words:
            if len(word) > 2 and word not in stopwords:
                keywords.append(word)
        
        return sorted(keywords)
    
    def identify_evidence_types(self, question: str) -> List[str]:
        """Legacy method - basic evidence type identification"""
        evidence_types = []
        text_lower = question.lower()
        
        if any(word in text_lower for word in ['cause', 'effect', 'impact']):
            evidence_types.append('causal_evidence')
        if any(word in text_lower for word in ['compare', 'versus', 'difference']):
            evidence_types.append('comparative_evidence')
        if any(word in text_lower for word in ['count', 'number', 'amount']):
            evidence_types.append('quantitative_evidence')
        
        return sorted(evidence_types) if evidence_types else ['general_evidence']
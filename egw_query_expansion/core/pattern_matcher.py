"""
Pattern Matcher Module
Enhanced placeholder implementation for EGW Query Expansion system
"""

# # # from __future__ import annotations  # Module not found  # Module not found  # Module not found

import logging
import re
import threading
import unicodedata
# # # from dataclasses import dataclass, field  # Module not found  # Module not found  # Module not found
# # # from functools import lru_cache  # Module not found  # Module not found  # Module not found
# # # from typing import Any, Dict, List, Optional, Sequence, Tuple, Set  # Module not found  # Module not found  # Module not found

# # # from .import_safety import safe_import  # Module not found  # Module not found  # Module not found

# Safe imports with fallbacks
numpy_result = safe_import('numpy', required=False)
if numpy_result.success:
    np = numpy_result.module

sklearn_tfidf_result = safe_import('sklearn.feature_extraction.text', required=False)
sklearn_similarity_result = safe_import('sklearn.metrics.pairwise', required=False)

_SKLEARN_AVAILABLE = False
if sklearn_tfidf_result.success and sklearn_similarity_result.success:
# # #     from sklearn.feature_extraction.text import TfidfVectorizer  # Module not found  # Module not found  # Module not found
# # #     from sklearn.metrics.pairwise import cosine_similarity  # Module not found  # Module not found  # Module not found
    _SKLEARN_AVAILABLE = True


@dataclass
class PatternMatch:
    """Container for pattern match results"""
    pattern: str
    match_text: str
    start_pos: int
    end_pos: int
    confidence: float
    semantic_score: Optional[float] = None
    # Extensible metadata for advanced uses
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CompiledPattern:
    """Internal storage for a compiled pattern"""
    original_pattern: str
    regex_pattern: re.Pattern
    tokens: List[str]
    pattern_type: str  # 'literal' | 'wildcard' | 'regex'
    embedding: Optional[np.ndarray] = None
    synonyms: Set[str] = field(default_factory=set)


class PatternMatcher:
    """Enhanced placeholder for pattern matching functionality"""

    # ----------------------------
    # Construction
    # ----------------------------

    def __init__(self, semantic_model=None, similarity_threshold=0.7, **kwargs):
        """
        Initialize pattern matcher.

        Args:
            semantic_model: External model for semantic similarity (must provide encode(list[str]) -> embeddings)
            similarity_threshold: Threshold for pattern matching (semantic)
            **kwargs: Additional arguments (ignored for compatibility)
                     Supported (optional, backward compatible):
                       - patterns: Optional[List[str]] initial patterns to compile
                       - case_sensitive: bool (default False)
                       - max_patterns: int (default 1000)
                       - synonyms: Dict[str, Iterable[str]] per-pattern synonyms
                       - ngram_sizes: Tuple[int, ...] N-gram sizes for semantic phrases (default (1,2,3))
                       - max_matches: int (default 10)
        """
        self.semantic_model = semantic_model
        self.similarity_threshold = float(similarity_threshold)

        # Backward-compatible optional config
        self.case_sensitive: bool = bool(kwargs.get("case_sensitive", False))
        self.max_patterns: int = int(kwargs.get("max_patterns", 1000))
        self.default_max_matches: int = int(kwargs.get("max_matches", 10))
        self.synonyms_config: Dict[str, Set[str]] = {
            k: set(v) for k, v in (kwargs.get("synonyms", {}) or {}).items()
        }
        self.ngram_sizes: Tuple[int, ...] = tuple(kwargs.get("ngram_sizes", (1, 2, 3)))

        # Internal state
        self._compiled: Dict[str, CompiledPattern] = {}
        self._pattern_keys: List[str] = []
        self._pattern_embeddings: Optional[np.ndarray] = None
        self._match_stats: Dict[str, int] = {}
        self._logger = logging.getLogger(__name__)
        self._lock = threading.RLock()

        # Optional TF-IDF fallback
        self._tfidf: Optional[TfidfVectorizer] = None
        self._tfidf_matrix: Optional[Any] = None  # scipy sparse or ndarray

        # Compile initial patterns if given
        initial_patterns = kwargs.get("patterns")
        if isinstance(initial_patterns, (list, tuple)) and initial_patterns:
            self.compile_patterns(list(initial_patterns))

    # ----------------------------
    # Public minimal API (backward compatible)
    # ----------------------------

    def match_patterns(self, query: str) -> List[PatternMatch]:
        """Find patterns in query with combined exact and semantic strategies."""
        if not query or not isinstance(query, str):
            return []

        text = self._normalize_text(query)
        if not self._compiled:
            # Keep behavior compatible with placeholder: no patterns compiled -> no matches
            return []

        exact = self._find_exact_matches(text)
        semantic = self._find_semantic_matches(text) if self._can_semantic() else []

        all_matches = self._deduplicate_and_sort(exact + semantic)
        # Update statistics
        for m in all_matches[: self.default_max_matches]:
            self._match_stats[m.pattern] = self._match_stats.get(m.pattern, 0) + 1
        return all_matches[: self.default_max_matches]

    # ----------------------------
    # Advanced public API (optional)
    # ----------------------------

    def compile_patterns(self, patterns: Sequence[str]) -> Dict[str, CompiledPattern]:
        """Compile and index patterns for matching."""
        if not patterns:
            return {}
        compiled: Dict[str, CompiledPattern] = {}
        with self._lock:
            for p in patterns[: max(0, self.max_patterns - len(self._compiled))]:
                if not isinstance(p, str) or not p.strip():
                    continue
                if p in self._compiled:
                    continue
                cp = self._compile_single_pattern(p)
                compiled[p] = cp
                self._compiled[p] = cp
            self._refresh_embeddings_locked()
        return compiled

    def add_patterns(self, patterns: Sequence[str]) -> int:
        """Add new patterns incrementally."""
        return len(self.compile_patterns(patterns))

    def remove_pattern(self, pattern: str) -> bool:
        """Remove a pattern if present."""
        with self._lock:
            if pattern in self._compiled:
                del self._compiled[pattern]
                self._refresh_embeddings_locked()
                return True
        return False

    def export_patterns(self) -> List[str]:
        """Return a list of compiled pattern strings."""
        with self._lock:
            return list(self._compiled.keys())

    def clear_patterns(self) -> None:
        """Clear all patterns and indices."""
        with self._lock:
            self._compiled.clear()
            self._pattern_keys = []
            self._pattern_embeddings = None
            self._tfidf = None
            self._tfidf_matrix = None
            self._match_stats.clear()

    def get_pattern_statistics(self) -> Dict[str, int]:
        """Usage counts per pattern."""
        return dict(self._match_stats)

    # ----------------------------
    # Compilation
    # ----------------------------

    def _compile_single_pattern(self, pattern: str) -> CompiledPattern:
        ptype = self._determine_pattern_type(pattern)
        flags = 0 if self.case_sensitive else re.IGNORECASE
        if ptype == "literal":
            expr = re.compile(re.escape(pattern), flags)
        elif ptype == "wildcard":
            # Convert shell-like wildcards to regex
            rx = "^" + re.escape(pattern).replace(r"\*", ".*").replace(r"\?", ".") + "$"
            # For in-text search, drop anchors but keep behavior equivalent by scanning
            rx = rx[1:-1]
            expr = re.compile(rx, flags)
        elif ptype == "regex":
            expr = re.compile(pattern, flags)
        else:
            expr = re.compile(re.escape(pattern), flags)
        tokens = self._tokenize(pattern)
        syns = self.synonyms_config.get(pattern, set())
        return CompiledPattern(
            original_pattern=pattern,
            regex_pattern=expr,
            tokens=tokens,
            pattern_type=ptype,
            synonyms=syns,
        )

    def _determine_pattern_type(self, pattern: str) -> str:
        # Heuristic: if contains regex metacharacters (excluding wildcards), consider 'regex'
        regex_chars = set("[](){}^$+|\\")
        if any(c in pattern for c in regex_chars):
            return "regex"
        if "*" in pattern or "?" in pattern:
            return "wildcard"
        return "literal"

    # ----------------------------
    # Matching
    # ----------------------------

    def _find_exact_matches(self, text: str) -> List[PatternMatch]:
        out: List[PatternMatch] = []
        # Run exact search for each compiled pattern
        for p, cp in self._compiled.items():
            for m in cp.regex_pattern.finditer(text):
                # Token-aware confidence boost if aligned with word boundaries
                w_bound = self._is_token_boundary(text, m.start(), m.end())
                conf = 1.0 if w_bound else 0.95
                out.append(
                    PatternMatch(
                        pattern=p,
                        match_text=m.group(0),
                        start_pos=m.start(),
                        end_pos=m.end(),
                        confidence=conf,
                        semantic_score=None,
                        metadata={"match_type": "exact", "word_boundary": w_bound},
                    )
                )
            # Try synonyms (if any)
            for syn in cp.synonyms:
                syn_rx = self._compile_synonym_regex(syn)
                for m in syn_rx.finditer(text):
                    w_bound = self._is_token_boundary(text, m.start(), m.end())
                    conf = 0.92 if w_bound else 0.88
                    out.append(
                        PatternMatch(
                            pattern=p,
                            match_text=m.group(0),
                            start_pos=m.start(),
                            end_pos=m.end(),
                            confidence=conf,
                            semantic_score=None,
                            metadata={"match_type": "synonym", "synonym": syn, "word_boundary": w_bound},
                        )
                    )
        return out

    def _find_semantic_matches(self, text: str) -> List[PatternMatch]:
        phrases = self._extract_phrases(text, ngram_sizes=self.ngram_sizes)
        if not phrases:
            return []
        out: List[PatternMatch] = []

        if self.semantic_model is not None and self._pattern_embeddings is not None and numpy_result.success:
            # External embeddings path
            phrase_texts = [p[0] for p in phrases]
            ph_emb = self._encode_with_model(phrase_texts)  # shape [P, D]
            if ph_emb is None or ph_emb.size == 0:
                return []
            # cosine similarity
            denom = (np.linalg.norm(self._pattern_embeddings, axis=1, keepdims=True) * np.linalg.norm(ph_emb, axis=1)).T
            sims = (self._pattern_embeddings @ ph_emb.T) / np.clip(denom, 1e-9, None)  # [N_patterns, P]
            for pi, pkey in enumerate(self._pattern_keys):
                for pj, (phrase, s, e) in enumerate(phrases):
                    score = float(sims[pi, pj])
                    if score >= self.similarity_threshold:
                        out.append(
                            PatternMatch(
                                pattern=pkey,
                                match_text=phrase,
                                start_pos=s,
                                end_pos=e,
                                confidence=score,
                                semantic_score=score,
                                metadata={"match_type": "semantic", "source": "external_model"},
                            )
                        )
            return out

        # TF-IDF fallback path (self-contained)
        if _SKLEARN_AVAILABLE and self._tfidf is not None and self._tfidf_matrix is not None and len(self._pattern_keys) > 0:
            phrase_texts = [p[0] for p in phrases]
            # Transform phrases with existing vectorizer
            Xp = self._tfidf.transform(phrase_texts)  # [P, V]
            sims = cosine_similarity(self._tfidf_matrix, Xp)  # [N_patterns, P]
            for pi, pkey in enumerate(self._pattern_keys):
                for pj, (phrase, s, e) in enumerate(phrases):
                    score = float(sims[pi, pj])
                    if score >= self.similarity_threshold:
                        out.append(
                            PatternMatch(
                                pattern=pkey,
                                match_text=phrase,
                                start_pos=s,
                                end_pos=e,
                                confidence=score,
                                semantic_score=score,
                                metadata={"match_type": "semantic", "source": "tfidf"},
                            )
                        )
        return out

    # ----------------------------
    # Embeddings / indexing
    # ----------------------------

    def _refresh_embeddings_locked(self) -> None:
        """Refresh internal embedding/index structures. Assumes self._lock is held."""
        self._pattern_keys = list(self._compiled.keys())
        if not self._pattern_keys:
            self._pattern_embeddings = None
            self._tfidf = None
            self._tfidf_matrix = None
            return

        # External semantic model embeddings (if provided)
        if self.semantic_model is not None and numpy_result.success:
            embs = self._encode_with_model(self._pattern_keys)
            self._pattern_embeddings = embs if embs is not None else None
        else:
            self._pattern_embeddings = None

        # TF-IDF fallback index if no external model or encoding failed
        if self._pattern_embeddings is None and _SKLEARN_AVAILABLE:
            try:
                self._tfidf = TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 3),
                    lowercase=not self.case_sensitive,
                    norm="l2",
                    max_features=None,
                )
                self._tfidf_matrix = self._tfidf.fit_transform(self._pattern_keys)
            except Exception as e:  # pragma: no cover
                self._logger.warning(f"TF-IDF initialization failed: {e}")
                self._tfidf = None
                self._tfidf_matrix = None

    def _encode_with_model(self, texts: List[str]) -> Optional[Any]:
        """Encode texts using the external semantic_model if available."""
        try:
            embs = self.semantic_model.encode(texts, convert_to_numpy=True)
            if numpy_result.success and isinstance(embs, list):
                embs = np.asarray(embs)
            return embs
        except Exception as e:  # pragma: no cover
            self._logger.warning(f"semantic_model.encode failed, disabling external embeddings: {e}")
            return None

    def _can_semantic(self) -> bool:
        if self.semantic_model is not None:
            return self._pattern_embeddings is not None
        if _SKLEARN_AVAILABLE:
            return self._tfidf is not None and self._tfidf_matrix is not None
        return False

    # ----------------------------
    # Utilities
    # ----------------------------

    @staticmethod
    def _tokenize(pattern: str) -> List[str]:
        return [t for t in re.split(r"\s+", pattern.strip()) if t]

    @staticmethod
    def _is_token_boundary(text: str, start: int, end: int) -> bool:
        left_ok = start == 0 or not text[start - 1].isalnum()
        right_ok = end == len(text) or not text[end].isalnum()
        return left_ok and right_ok

    def _compile_synonym_regex(self, synonym: str) -> re.Pattern:
        flags = 0 if self.case_sensitive else re.IGNORECASE
        return re.compile(re.escape(self._normalize_text(synonym)), flags)

    def _normalize_text(self, s: str) -> str:
        if self.case_sensitive:
            # Normalize Unicode but preserve case
            return unicodedata.normalize("NFKC", s)
        return unicodedata.normalize("NFKC", s).lower()

    def _extract_phrases(self, text: str, ngram_sizes: Tuple[int, ...]) -> List[Tuple[str, int, int]]:
        """
        Extract phrases and n-grams with character offsets.
        - Splits by sentence-like boundaries.
        - Generates n-grams per sentence.
        """
        text_norm = text  # offsets must refer to original input text; we do not normalize positions
        # Sentence segmentation (lightweight)
        sentence_spans = []
        last = 0
        for m in re.finditer(r"[.!?]+|\n+", text_norm):
            end = m.start()
            if end > last:
                sentence_spans.append((last, end))
            last = m.end()
        if last < len(text_norm):
            sentence_spans.append((last, len(text_norm)))

        phrases: List[Tuple[str, int, int]] = []
        for s, e in sentence_spans:
            sent = text_norm[s:e].strip()
            if not sent:
                continue
            # Whole sentence
            phrases.append((sent, s + text_norm[s:e].find(sent), s + text_norm[s:e].find(sent) + len(sent)))
            # N-grams
            words = re.findall(r"\w+|\S", sent)
            # Build positions for words to map n-grams back to char offsets
            offsets: List[Tuple[int, int]] = []
            cursor = 0
            for w in words:
                idx = sent.find(w, cursor)
                if idx == -1:
                    continue
                offsets.append((idx, idx + len(w)))
                cursor = idx + len(w)
            for n in ngram_sizes:
                if n <= 1:
                    continue
                for i in range(0, max(0, len(words) - n + 1)):
                    w_slice = words[i : i + n]
                    if not w_slice:
                        continue
                    start_local = offsets[i][0]
                    end_local = offsets[i + n - 1][1]
                    phrase = sent[start_local:end_local]
                    start_global = s + start_local
                    end_global = s + end_local
                    phrases.append((phrase, start_global, end_global))
        # Normalize each phrase text for semantic matching but keep offsets intact
        return [(self._normalize_text(p), a, b) for (p, a, b) in phrases if len(p.strip()) > 0]

    @staticmethod
    @lru_cache(maxsize=1024)
    def _is_regex(pattern: str) -> bool:
        try:
            re.compile(pattern)
            return True
        except Exception:
            return False

    @staticmethod
    def _merge_duplicate_matches(matches: List[PatternMatch]) -> List[PatternMatch]:
        """
        Merge duplicates keeping the highest confidence and merging metadata shallowly.
        Dedup key: (pattern, start_pos, end_pos, match_text)
        """
        best: Dict[Tuple[str, int, int, str], PatternMatch] = {}
        for m in matches:
            k = (m.pattern, m.start_pos, m.end_pos, m.match_text)
            ex = best.get(k)
            if ex is None or m.confidence > ex.confidence:
                best[k] = m
            elif ex is not None and m.metadata:
                ex.metadata.update(m.metadata)
        return list(best.values())

    def _deduplicate_and_sort(self, matches: List[PatternMatch]) -> List[PatternMatch]:
        dedup = self._merge_duplicate_matches(matches)
        # Prefer higher confidence, then shorter span (more precise), then earlier occurrence
        dedup.sort(key=lambda m: (-m.confidence, (m.end_pos - m.start_pos), m.start_pos))
        return dedup
